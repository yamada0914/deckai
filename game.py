"""
ポケカ風ルールのゲームロジック。

- 初手 7 枚、ターンでドロー・エネルギー付与・アイテム・攻撃。
- ベンチは最大 5 体。きぜつしたらベンチから 1 体をバトル場に出す。
- 相手のポケモンを 3 回きぜつさせたら勝ち。プレイヤーは先行・後攻の 2 人。
"""
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Literal

from card import PokemonCard, get_card_by_id, is_basic_pokemon, is_energy, is_goods, is_pokemon, is_stage2_pokemon, is_support
from deck import STARTING_HAND_SIZE, create_deck, create_deck_from_deck_code, format_deck_recipe, get_deck_name

BENCH_SIZE = 5
WIN_KO_COUNT = 3
PRIZE_COUNT = 6
MAX_TURNS_SAFETY = 200

SpecialState = Literal["sleep", "paralysis", "confusion"]


@dataclass
class BattlePokemon:
    """バトル場／ベンチに出すポケモン（HP・エネルギー・状態異常・どうぐを管理）"""
    card: PokemonCard
    attached_energy: int = 0
    attached_energy_types: list = field(default_factory=list)
    attached_tool: "GoodsCard | None" = None
    special_state: SpecialState | None = None
    poison_damage: int = 0
    burn: bool = False
    evolved_this_turn: bool = False
    put_on_bench_this_turn: bool = False
    protected_next_opponent_turn: bool = False
    damage_reduction_next_turn: int = 0
    disabled_attack_name: str | None = None

    @property
    def hp(self) -> int:
        return self.card.hp

    @hp.setter
    def hp(self, value: int) -> None:
        self.card.hp = min(value, self.card.max_hp)

    @property
    def max_hp(self) -> int:
        return self.card.max_hp


@dataclass
class PlayerState:
    deck: list = field(default_factory=list)
    hand: list = field(default_factory=list)
    discard: list = field(default_factory=list)
    prize_pile: list = field(default_factory=list)
    active: BattlePokemon | None = None
    bench: list[BattlePokemon] = field(default_factory=list)
    knockouts_suffered: int = 0

    def draw(self, n: int = 1) -> list:
        drawn = []
        for _ in range(n):
            if not self.deck:
                return drawn
            drawn.append(self.deck.pop())
        return drawn

    def has_active_pokemon(self) -> bool:
        return self.active is not None and self.active.hp > 0


def _card_label(card) -> str:
    return getattr(card, "name", card.__class__.__name__)


def _basic_energy_id(energy_type: str) -> str:
    """エネルギータイプ名から基本エネルギーカード ID を返す。"""
    if energy_type == "lightning":
        return "basic-energy-lightning"
    if energy_type == "fighting":
        return "basic-energy-fighting"
    return "basic-energy"


def _put_energy_cards_in_discard(player: "PlayerState", energy_types: list, state: "GameState") -> None:
    """ポケモンから捨てるエネルギー分の基本エネルギーカードをトラッシュに加える。"""
    for i, ty in enumerate(energy_types):
        try:
            c = get_card_by_id(_basic_energy_id(ty), f"trash-{id(state)}-{len(player.discard)}-{i}")
            player.discard.append(c)
        except ValueError:
            pass


def _flip_coin() -> bool:
    """コインを1回投げる。True = 表、False = 裏。"""
    return random.random() < 0.5


def _clear_status(bp: BattlePokemon) -> None:
    """状態異常を全て解除（ベンチに下がったときなど）。"""
    bp.special_state = None
    bp.poison_damage = 0
    bp.burn = False
    bp.protected_next_opponent_turn = False
    bp.damage_reduction_next_turn = 0


def _apply_status(
    state: "GameState",
    target: BattlePokemon,
    status_effect: str,
    poison_damage: int = 10,
    log_prefix: str = "",
) -> None:
    """
    1 つの状態異常を付与。ねむり・マヒ・こんらんは互いに上書き。どく・やけどは重複可。
    """
    status_names = {"sleep": "ねむり", "paralysis": "マヒ", "confusion": "こんらん", "poison": "どく", "burn": "やけど"}
    name_ja = status_names.get(status_effect, status_effect)
    if status_effect in ("sleep", "paralysis", "confusion"):
        target.special_state = status_effect
        state.log(f"{log_prefix}{target.card.name} は {name_ja} になった")
    elif status_effect == "poison":
        target.poison_damage = poison_damage
        state.log(f"{log_prefix}{target.card.name} は {name_ja}（{poison_damage}）になった")
    elif status_effect == "burn":
        target.burn = True
        state.log(f"{log_prefix}{target.card.name} は {name_ja} になった")


def _can_pay_energy_cost(
    attached_count: int,
    attached_types: list,
    cost_total: int,
    cost_typed: list | None,
) -> bool:
    """
    付与エネルギーで技コストを支払えるか。
    cost_typed が None のときは cost_total 以上の個数があれば可。
    cost_typed があるときは、各タイプが必要数あり、無色は任意で埋める。
    """
    if cost_typed is None:
        return attached_count >= cost_total
    if attached_count < len(cost_typed):
        return False
    need = Counter(t for t in cost_typed if t != "colorless")
    have = Counter(attached_types)
    for typ, n in need.items():
        if have.get(typ, 0) < n:
            return False
    return True


def _player_name(state: "GameState", i: int) -> str:
    """使用デッキに応じたプレイヤー名を返す。同じデッキ同士のときは「デッキ名（1）（2）」、それ以外はデッキ名。"""
    d0, d1 = state.deck_indices[0], state.deck_indices[1]
    idx = state.deck_indices[i] if i < len(state.deck_indices) else i
    name = get_deck_name(idx)
    if d0 == d1:
        return f"{name}（{i + 1}）"
    return name


def _turn_label(state: "GameState") -> str:
    """ログ用に「先行Nターン」「後行Nターン」のラベルを返す。"""
    is_first_player_turn = state.current_player == state.first_player
    n = (state.turn_count // 2) + 1
    return f"先行{n}ターン" if is_first_player_turn else f"後行{n}ターン"


@dataclass
class GameState:
    current_player: Literal[0, 1] = 0
    players: list[PlayerState] = field(default_factory=lambda: [PlayerState(), PlayerState()])
    turn_count: int = 0
    first_player: Literal[0, 1] = 0
    winner: Literal[0, 1, -1] | None = None
    log_fn: Callable[[str], None] | None = None
    deck_indices: tuple = (0, 1)
    shippegaeshi_120_used: bool = False
    support_used_this_turn: bool = False
    this_turn_attack_name: str | None = None
    this_turn_attack_actor_id: str | None = None
    last_turn_attack_name: list = field(default_factory=lambda: [None, None])
    last_turn_attack_actor_id: list = field(default_factory=lambda: [None, None])
    our_ko_by_damage_last_turn: list = field(default_factory=lambda: [False, False])
    drawn_this_turn: list = field(default_factory=list)
    record_frame_fn: Callable[["GameState"], None] | None = field(default=None, repr=False)

    def log(self, msg: str) -> None:
        if self.log_fn:
            self.log_fn(msg)

    def _record_frame(self) -> None:
        if self.record_frame_fn:
            self.record_frame_fn(self)

    def opponent(self) -> int:
        return 1 - self.current_player

    def active_player_state(self) -> PlayerState:
        return self.players[self.current_player]

    def defending_player_state(self) -> PlayerState:
        return self.players[self.opponent()]

    def player_name(self, i: int) -> str:
        return _player_name(self, i)


def setup_game(
    seed: int | None = None,
    log_fn: Callable[[str], None] | None = None,
    record_frame_fn: Callable[["GameState"], None] | None = None,
    deck0: int = 0,
    deck1: int = 1,
    deck_code0: str | None = None,
    deck_code1: str | None = None,
) -> GameState:
    """
    デッキを組んで初手 7 枚をセット。
    deck0 / deck1 で固定デッキ番号（0=A, 1=B, 2=C, 3=D, 4=E）。
    deck_code0 / deck_code1 を指定すると read_cards_result.json のそのデッキコードの一覧でデッキを組む（未指定なら deck0 / deck1 を使用）。
    """
    if seed is not None:
        random.seed(seed)
    state = GameState(log_fn=log_fn, record_frame_fn=record_frame_fn, deck_indices=(deck0, deck1))
    state.log("========== ゲーム開始 ==========")
    deck_specs: list[tuple[int, str | None]] = [(deck0, deck_code0), (deck1, deck_code1)]
    for i in range(2):
        idx, code = deck_specs[i]
    for i in range(2):
        idx, code = deck_specs[i]
        if code:
            deck_list = create_deck_from_deck_code(code)
            if deck_list is None:
                raise ValueError(f"デッキコード {code} のレシピを read_cards_result.json から読み込めませんでした。")
            state.players[i].deck = deck_list
        else:
            state.players[i].deck = create_deck(idx)
        random.shuffle(state.players[i].deck)
        state.log(f"{state.player_name(i)}: デッキ {len(state.players[i].deck)} 枚をシャッフル")

    penalty_draws = [0, 0]
    for i in range(2):
        opp = 1 - i
        retry_count = 0
        while True:
            state.players[i].hand = state.players[i].draw(STARTING_HAND_SIZE)
            hand_names = [_card_label(c) for c in state.players[i].hand]
            has_basic_pokemon = any(
                is_pokemon(c) and not getattr(c, "evolves_from", None) for c in state.players[i].hand
            )
            if has_basic_pokemon:
                break
            retry_count += 1
            state.players[i].deck.extend(state.players[i].hand)
            state.players[i].hand = []
            random.shuffle(state.players[i].deck)
            state.log(f"{state.player_name(i)}: 初手にたねポケモンなし → 引き直し（{retry_count} 回目）")
            penalty_draws[opp] += 1

        hand_names = [_card_label(c) for c in state.players[i].hand]
        msg = f"初手 {len(state.players[i].hand)} 枚ドロー → 手札 [{', '.join(hand_names)}]" if retry_count == 0 else f"引き直し後、初手 {len(state.players[i].hand)} 枚 → 手札 [{', '.join(hand_names)}]"
        state.log(f"{state.player_name(i)}: {msg}")

    for i in range(2):
        _put_one_pokemon_active(state.players[i])
        if state.players[i].active:
            state.log(f"{state.player_name(i)}: バトル場に {state.players[i].active.card.name} を出す（HP {state.players[i].active.max_hp}）")
        _fill_bench_from_hand(state.players[i], state, i, log_each=False)
        if state.players[i].bench:
            bench_names = [b.card.name for b in state.players[i].bench]
            state.log(f"{state.player_name(i)}: ベンチに {len(state.players[i].bench)} 体 [{', '.join(bench_names)}]")

    for i in range(2):
        for _ in range(PRIZE_COUNT):
            if state.players[i].deck:
                state.players[i].prize_pile.append(state.players[i].deck.pop())
        side_names = [_card_label(c) for c in state.players[i].prize_pile]
        state.log(f"{state.player_name(i)}: サイドを {PRIZE_COUNT} 枚置く → [{', '.join(side_names)}]（残りデッキ {len(state.players[i].deck)} 枚）")

    for i in range(2):
        if penalty_draws[i] > 0:
            extra = state.players[i].draw(penalty_draws[i])
            state.players[i].hand.extend(extra)
            extra_names = [_card_label(c) for c in extra]
            state.log(f"{state.player_name(i)}: 相手の引き直しペナルティで {penalty_draws[i]} 枚ドロー → [{', '.join(extra_names)}]（手札 {len(state.players[i].hand)} 枚）")
    state.first_player = random.randint(0, 1)
    state.current_player = state.first_player
    state.log(f"先行: {state.player_name(state.first_player)} / 後攻: {state.player_name(1 - state.first_player)}")
    state.log("")
    state._record_frame()
    return state


def _put_one_pokemon_active(player: PlayerState) -> bool:
    """手札からたねポケモン（進化ポケモンでない）1 体をバトル場に出す。出せなければ何もしない。"""
    for i, c in enumerate(player.hand):
        if is_pokemon(c) and not c.evolves_from:
            p = c.copy()
            player.active = BattlePokemon(card=p)
            player.hand.pop(i)
            return True
    return False


def _put_one_pokemon_on_bench(
    player: PlayerState, state: GameState, player_index: int, *, log: bool = True
) -> bool:
    """手札からたねポケモン（進化ポケモンでない）1 体をベンチに出す（ベンチが 5 体未満のとき）。"""
    if len(player.bench) >= BENCH_SIZE:
        return False
    existing_names = {getattr(bp.card, "name", "") for bp in player.bench}
    for i, c in enumerate(player.hand):
        if is_pokemon(c) and not c.evolves_from:
            # すでに同名のポケモンがベンチにいる場合は出さない
            if getattr(c, "name", "") in existing_names:
                continue
            p = c.copy()
            bp = BattlePokemon(card=p)
            bp.put_on_bench_this_turn = True  # その番には進化できない（ルール A-05）
            player.bench.append(bp)
            player.hand.pop(i)
            if log:
                state.log(f"{state.player_name(player_index)}: ベンチに {p.name} を出す（ベンチ {len(player.bench)} 体）")
            return True
    return False


def _fill_bench_from_hand(
    player: PlayerState, state: GameState, player_index: int, *, log_each: bool = True
) -> None:
    """手札のポケモンをベンチに出す（最大 BENCH_SIZE まで）。"""
    while len(player.bench) < BENCH_SIZE and _put_one_pokemon_on_bench(
        player, state, player_index, log=log_each
    ):
        pass


def _promote_from_bench(player: PlayerState, state: GameState, player_index: int) -> bool:
    """バトル場が空のとき、ベンチから 1 体をバトル場に出す。"""
    if player.active is not None or not player.bench:
        return False
    player.active = player.bench.pop(0)
    state.log(f"{state.player_name(player_index)}: ベンチから {player.active.card.name} をバトル場に出す（HP {player.active.hp}/{player.active.max_hp}）")
    return True


def _prizes_for_ko(koed_bp: BattlePokemon) -> int:
    """きぜつしたポケモンのフラグで判定：is_mega なら 3 枚、is_ex なら 2 枚、それ以外は 1 枚。"""
    card = koed_bp.card
    if getattr(card, "is_mega", False):
        return 3
    if getattr(card, "is_ex", False):
        return 2
    return 1


def _take_prize(state: GameState, taker_index: int) -> bool:
    """
    プレイヤー taker_index がサイドを 1 枚とって手札に加える。
    サイドが 0 枚になったらそのプレイヤーの勝ちで state.winner をセットする。
    戻り値: 勝敗がついたら True。
    """
    p = state.players[taker_index]
    if not p.prize_pile:
        return False
    card = p.prize_pile.pop()
    p.hand.append(card)
    remaining = [_card_label(c) for c in p.prize_pile]
    state.log(
        f"{state.player_name(taker_index)}: サイドを 1 枚とる → {_card_label(card)}（手札 {len(p.hand)} 枚、サイド残り {len(p.prize_pile)} 枚）"
    )
    if len(p.prize_pile) == 0:
        state.winner = taker_index
        state.log(f"{state.player_name(taker_index)}: サイドをすべてとり終えた → 勝ち！")
        return True
    return False


def _handle_opponent_ko(opp: PlayerState, state: GameState, koed_bp: BattlePokemon) -> bool:
    """相手のきぜつを 1 回加算し、攻撃側がサイドをとる（ex なら 2 枚、それ以外は 1 枚）。サイド 0 なら勝ち、否則ベンチから繰り出す。"""
    opp.discard.append(koed_bp.card)
    tool = getattr(koed_bp, "attached_tool", None)
    if tool:
        opp.discard.append(tool)
    opp.knockouts_suffered += 1
    prize_count = _prizes_for_ko(koed_bp)
    if prize_count > 1:
        state.log(f"{state.player_name(state.current_player)}: {koed_bp.card.name} は {'メガ' if prize_count == 3 else 'ex'} のため、サイドを {prize_count} 枚とる")
    for _ in range(prize_count):
        if _take_prize(state, state.current_player):
            return True
    _promote_from_bench(opp, state, state.opponent())
    return False


def _check_game_end(state: GameState) -> bool:
    """勝敗が決まっていれば state.winner をセットして True を返す。サイド 0 またはバトル場・ベンチともにいないと負け。"""
    for i in range(2):
        p = state.players[i]
        if len(p.prize_pile) == 0:
            state.winner = i
            return True
        if not p.has_active_pokemon() and len(p.bench) == 0:
            state.winner = 1 - i
            state.log(f"{state.player_name(i)} のバトル場・ベンチにポケモンがいない → {state.player_name(state.winner)} の勝ち")
            return True
    return False


def start_turn(state: GameState) -> None:
    """ターン開始：ドロー 1 枚、サポート未使用にリセット。ねむりならコインで解除判定。"""
    state.support_used_this_turn = False
    state.our_ko_by_damage_last_turn[state.current_player] = False
    state.drawn_this_turn = []
    p = state.active_player_state()
    for bp in ([p.active] if p.active else []) + list(p.bench):
        if bp:
            bp.disabled_attack_name = None
    turn_label = _turn_label(state)
    state.log(f"---------- {turn_label} ---------- {state.player_name(state.current_player)} のターン")
    drawn = p.draw(1)
    if drawn:
        p.hand.extend(drawn)
        state.drawn_this_turn.extend(drawn)
        state.log(f"{state.player_name(state.current_player)}: 1 枚ドロー → {_card_label(drawn[0])}（手札 {len(p.hand)} 枚、デッキ {len(p.deck)} 枚）")
    else:
        state.winner = state.opponent()
        state.log(f"{state.player_name(state.current_player)}: デッキが空のためドローできず → {state.player_name(state.winner)} の勝ち")
        state._record_frame()
        return
    if p.active and getattr(p.active, "special_state", None) == "sleep":
        if _flip_coin():
            p.active.special_state = None
            state.log(f"{state.player_name(state.current_player)}: {p.active.card.name} のねむりが解けた（コイン：表）")
        else:
            state.log(f"{state.player_name(state.current_player)}: {p.active.card.name} はねむったまま（コイン：裏）")
    state._record_frame()


def _attack_damage_for_eval(atk) -> int:
    """技のダメージを評価用に返す。コイン技の場合は期待値（表0.5×回数×damage_per_coin）。"""
    cf = getattr(atk, "coin_flips", 0)
    dpc = getattr(atk, "damage_per_coin", 0)
    if cf > 0 and dpc > 0:
        return int(cf * 0.5 * dpc)
    return atk.damage


def _effective_damage_to_defender(
    attacker_card: PokemonCard, defender: BattlePokemon, base_damage: int
) -> int:
    """弱点・抵抗力・どうぐを考慮した、守備側が受けるダメージを返す。"""
    defender_card = defender.card
    damage = base_damage
    if (
        getattr(defender_card, "weakness", None)
        and getattr(attacker_card, "pokemon_type", None)
        and defender_card.weakness == attacker_card.pokemon_type
    ):
        damage *= 2
    if (
        getattr(defender_card, "resistance", None)
        and getattr(attacker_card, "pokemon_type", None)
        and defender_card.resistance == attacker_card.pokemon_type
    ):
        damage = max(0, damage - 30)
    tool = getattr(defender, "attached_tool", None)
    if tool and getattr(tool, "is_tool", False) and getattr(tool, "tool_damage_reduce", 0) > 0:
        cond = getattr(tool, "tool_condition_type", None)
        if cond is None or getattr(defender_card, "pokemon_type", None) == cond:
            damage = max(0, damage - getattr(tool, "tool_damage_reduce", 0))
    return damage


def _opponent_max_damage(state: GameState) -> int:
    """相手のバトル場ポケモンが今出せる最大ダメージを返す（エネルギーに応じた最強ワザ）。"""
    opp = state.defending_player_state()
    if not opp.active or not opp.active.card.attacks:
        return 0
    max_dmg = 0
    types = getattr(opp.active, "attached_energy_types", [])
    for atk in opp.active.card.attacks:
        if _can_pay_energy_cost(
            opp.active.attached_energy, types,
            atk.energy_cost, getattr(atk, "energy_cost_typed", None),
        ):
            dmg = _attack_damage_for_eval(atk)
            if atk.name == "しっぺがえし" and len(state.defending_player_state().prize_pile) == 1:
                dmg += 90
            max_dmg = max(max_dmg, dmg)
    return max_dmg


def _opponent_max_effective_damage(state: GameState) -> int:
    """相手のバトル場ポケモンが自分のバトル場に与えうる最大の有效ダメージ（弱点・抵抗・どうぐ込み）。"""
    p = state.active_player_state()
    opp = state.defending_player_state()
    if not opp.active or not opp.active.card.attacks or not p.active:
        return 0
    max_dmg = 0
    types = getattr(opp.active, "attached_energy_types", [])
    for atk in opp.active.card.attacks:
        if _can_pay_energy_cost(
            opp.active.attached_energy, types,
            atk.energy_cost, getattr(atk, "energy_cost_typed", None),
        ):
            base = _attack_damage_for_eval(atk)
            if atk.name == "しっぺがえし" and len(opp.prize_pile) == 1:
                base += 90
            eff = _effective_damage_to_defender(opp.active.card, p.active, base)
            max_dmg = max(max_dmg, eff)
    return max_dmg


def _our_max_damage(state: GameState) -> int:
    """自分のバトル場ポケモンがこのターン出せる最大ダメージを返す。"""
    p = state.active_player_state()
    opp = state.defending_player_state()
    if not p.active or not p.active.card.attacks:
        return 0
    max_dmg = 0
    types = getattr(p.active, "attached_energy_types", [])
    for atk in p.active.card.attacks:
        if _can_pay_energy_cost(
            p.active.attached_energy, types,
            atk.energy_cost, getattr(atk, "energy_cost_typed", None),
        ):
            dmg = _attack_damage_for_eval(atk)
            if atk.name == "しっぺがえし" and len(opp.prize_pile) == 1:
                dmg += 90
            if atk.name == "アベンジナックル" and state.our_ko_by_damage_last_turn[state.current_player]:
                dmg += 120
            max_dmg = max(max_dmg, dmg)
    return max_dmg


def _our_max_effective_damage(state: GameState) -> int:
    """自分のバトル場ポケモンが相手のバトル場に与えうる最大の有效ダメージ（弱点・抵抗・どうぐ込み）。"""
    p = state.active_player_state()
    opp = state.defending_player_state()
    if not p.active or not p.active.card.attacks or not opp.active:
        return 0
    max_dmg = 0
    types = getattr(p.active, "attached_energy_types", [])
    for atk in p.active.card.attacks:
        if _can_pay_energy_cost(
            p.active.attached_energy, types,
            atk.energy_cost, getattr(atk, "energy_cost_typed", None),
        ):
            base = _attack_damage_for_eval(atk)
            if atk.name == "しっぺがえし" and len(opp.prize_pile) == 1:
                base += 90
            if atk.name == "アベンジナックル" and state.our_ko_by_damage_last_turn[state.current_player]:
                base += 120
            eff = _effective_damage_to_defender(p.active.card, opp.active, base)
            max_dmg = max(max_dmg, eff)
    return max_dmg


def _max_effective_damage_for_attacker(
    state: GameState,
    attacker_bp: BattlePokemon,
    defender_bp: BattlePokemon | None,
    player_index: int,
) -> int:
    """任意のバトルポケモンが相手のバトル場に与えうる最大の有效ダメージを返す。"""
    if not attacker_bp.card.attacks or not defender_bp:
        return 0
    max_dmg = 0
    types = getattr(attacker_bp, "attached_energy_types", [])
    opp = state.players[1 - player_index]
    for atk in attacker_bp.card.attacks:
        if not _can_pay_energy_cost(
            attacker_bp.attached_energy, types,
            atk.energy_cost, getattr(atk, "energy_cost_typed", None),
        ):
            continue
        base = _attack_damage_for_eval(atk)
        if atk.name == "しっぺがえし" and len(opp.prize_pile) == 1:
            base += 90
        if atk.name == "アベンジナックル" and state.our_ko_by_damage_last_turn[player_index]:
            base += 120
        eff = _effective_damage_to_defender(attacker_bp.card, defender_bp, base)
        max_dmg = max(max_dmg, eff)
    return max_dmg


def retreat(state: GameState, bench_index: int) -> bool:
    """バトル場のポケモンとベンチの bench_index 番目を入れ替える（逃げる）。にげるエネルギー分を捨てる。ねむり・マヒ中はにげられない。"""
    p = state.active_player_state()
    if not p.active or bench_index < 0 or bench_index >= len(p.bench):
        return False
    if getattr(p.active, "special_state", None) in ("sleep", "paralysis"):
        state.log(f"{state.player_name(state.current_player)}: {p.active.card.name} は状態異常のためにげられない")
        return False
    old_active = p.active
    cost = getattr(old_active.card, "retreat_cost", 1)
    if old_active.attached_energy < cost:
        return False
    types = getattr(old_active, "attached_energy_types", [])
    discarded_types = types[-cost:] if len(types) >= cost else list(types)
    _put_energy_cards_in_discard(p, discarded_types, state)
    old_active.attached_energy -= cost
    if len(types) >= cost:
        old_active.attached_energy_types = types[:-cost]
    else:
        old_active.attached_energy_types = []
    if cost > 0:
        state.log(
            f"{state.player_name(state.current_player)}: 逃げるために {old_active.card.name} のエネルギーを {cost} 個捨てる"
        )
    p.active = p.bench[bench_index]
    p.bench[bench_index] = old_active
    _clear_status(old_active)
    state.log(
        f"{state.player_name(state.current_player)}: {old_active.card.name} をベンチに退き、"
        f"{p.active.card.name} をバトル場に出す（HP {p.active.hp}/{p.active.max_hp}）"
    )
    return True


def attach_energy(state: GameState, hand_index: int, bench_index: int | None = None) -> bool:
    """手札の energy_index 番目のエネルギーをバトル場またはベンチのポケモンに付与。bench_index 指定でベンチに付与。"""
    p = state.active_player_state()
    if hand_index < 0 or hand_index >= len(p.hand):
        return False
    if not is_energy(p.hand[hand_index]):
        return False
    energy_card = p.hand[hand_index]
    energy_type = getattr(energy_card, "energy_type", None)
    slot_type = energy_type if energy_type else "colorless"

    if bench_index is not None:
        if bench_index < 0 or bench_index >= len(p.bench):
            return False
        target = p.bench[bench_index]
        target.attached_energy += 1
        target.attached_energy_types.append(slot_type)
        p.hand.pop(hand_index)
        state.log(
            f"{state.player_name(state.current_player)}: エネルギーを 1 つ付与（ベンチの {target.card.name}、エネルギー {target.attached_energy} 個）"
        )
        return True
    if not p.active:
        return False
    p.active.attached_energy += 1
    p.active.attached_energy_types.append(slot_type)
    p.hand.pop(hand_index)
    state.log(f"{state.player_name(state.current_player)}: エネルギーを 1 つ付与（バトル場のエネルギー {p.active.attached_energy} 個）")
    return True


def use_potion(state: GameState, hand_index: int) -> bool:
    """きずぐすり（id=potion）を使用して自分のバトル場のポケモンを回復。手札の hand_index 番目がきずぐすりのときだけ実行。"""
    p = state.active_player_state()
    if not p.active or hand_index < 0 or hand_index >= len(p.hand):
        return False
    card = p.hand[hand_index]
    if not is_goods(card) or getattr(card, "effect", None) != "heal":
        return False
    if getattr(card, "id", None) != "potion":
        return False
    amount = getattr(card, "heal_amount", 20)
    before = p.active.hp
    p.active.hp = min(p.active.hp + amount, p.active.max_hp)
    p.discard.append(p.hand.pop(hand_index))
    when_drawn = "（今ターンのドローで引いた）" if card in state.drawn_this_turn else ""
    state.log(f"{state.player_name(state.current_player)}: きずぐすりを使用 → バトル場のポケモンを {before} → {p.active.hp} に回復{when_drawn}")
    return True


def use_trainer_goods(state: "GameState", hand_index: int) -> bool:
    """
    トレーナー（グッズ）の効果を実行。きずぐすり・いれかえ・どうぐ以外のアイテムを id / 名前で判定して処理する。
    対応: おとどけドローン, エネルギー回収, エレキジェネレーター, スーパーボール, ハイパーボール, ポケモンキャッチャー。
    """
    p = state.active_player_state()
    if hand_index < 0 or hand_index >= len(p.hand):
        return False
    card = p.hand[hand_index]
    if not is_goods(card):
        return False
    cid = getattr(card, "id", "")
    if (getattr(card, "effect", None) in ("heal", "swap_active") or getattr(card, "is_tool", False)) and cid not in ("supaboru", "haipaboru"):
        return False
    name_ja = getattr(card, "name", "")

    # ふしぎなアメ：手札の 2 進化 1 枚を、場のたねポケモンにのせて 1 進化をとばして進化
    # 先行・後行とも、そのプレイヤーの 1 ターン目は進化できない。
    if cid == "fushiginaame":
        if state.turn_count < 2:
            return False
        stage1_id = None
        stage2_card = None
        stage2_idx = None
        for i, c in enumerate(p.hand):
            if i == hand_index or not is_pokemon(c) or not getattr(c, "evolves_from", None):
                continue
            try:
                stage1_ref = get_card_by_id((c.evolves_from or "").strip())
            except ValueError:
                base = (c.evolves_from or "").strip()
                stage1_ref = next(
                    (h for h in p.hand if is_pokemon(h) and (
                        (getattr(h, "id", "") or "").strip() == base
                        or (getattr(h, "id", "") or "").startswith(base + "-")
                        or (getattr(h, "name", "") or "").strip() == base
                    )),
                    None,
                )
            if not stage1_ref or not is_pokemon(stage1_ref):
                continue
            # 2 進化の判定: evolution_stage が "stage2" か、または進化元が 1 進化（evolves_from を持つ）か
            is_stage2 = is_stage2_pokemon(c) or bool(getattr(stage1_ref, "evolves_from", None))
            if not is_stage2:
                continue
            stage1_id = (stage1_ref.evolves_from or "").strip()
            stage2_card = c
            stage2_idx = i
            break
        if stage2_card is None or stage1_id is None:
            return False
        try:
            stage1_ref = get_card_by_id(stage2_card.evolves_from.strip())
        except ValueError:
            base = (stage2_card.evolves_from or "").strip()
            stage1_ref = next(
                (h for h in p.hand if is_pokemon(h) and (
                    (getattr(h, "id", "") or "").strip() == base
                    or (getattr(h, "id", "") or "").startswith(base + "-")
                    or (getattr(h, "name", "") or "").strip() == base
                )),
                None,
            )
        if not stage1_ref:
            return False
        target_bp = None
        # たねポケモンのみ対象（evolution_stage が "basic" または evolves_from が無い）
        if p.active and is_basic_pokemon(p.active.card) and not getattr(p.active, "put_on_bench_this_turn", False):
            if _can_evolve_onto(p.active.card, stage1_ref):
                target_bp = p.active
        if target_bp is None:
            for bp in p.bench:
                if not is_basic_pokemon(bp.card) or getattr(bp, "put_on_bench_this_turn", False):
                    continue
                if _can_evolve_onto(bp.card, stage1_ref):
                    target_bp = bp
                    break
        if target_bp is None:
            return False
        _apply_evolution(
            target_bp, stage2_card, state,
            f"{state.player_name(state.current_player)}: ふしぎなアメで ",
        )
        if stage2_idx < hand_index:
            p.hand.pop(hand_index)
            p.hand.pop(stage2_idx)
        else:
            p.hand.pop(stage2_idx)
            p.hand.pop(hand_index)
        p.discard.append(card)
        state.log(f"{state.player_name(state.current_player)}: ふしぎなアメを使用（手札から {stage2_card.name} を場のたねにのせて 1 進化をとばして進化）")
        return True

    if cid == "otodokedoron":
        if _flip_coin() and _flip_coin():
            if p.deck:
                idx = random.randint(0, len(p.deck) - 1)
                chosen = p.deck.pop(idx)
                p.hand.append(chosen)
                state.drawn_this_turn.append(chosen)
                random.shuffle(p.deck)
                p.discard.append(p.hand.pop(hand_index))
                state.log(f"{state.player_name(state.current_player)}: おとどけドローンを使用（コイン2回オモテ）→ 山札から {_card_label(chosen)} を手札に加えた")
                return True
        p.discard.append(p.hand.pop(hand_index))
        state.log(f"{state.player_name(state.current_player)}: おとどけドローンを使用（コイン裏）→ 効果なし")
        return True

    if (cid == "unknown" and name_ja == "エネルギー回収") or cid == "enerugikaishixyuu":
        lightning_count = sum(1 for c in p.discard if is_energy(c) and getattr(c, "energy_type", None) == "lightning")
        fighting_count = sum(1 for c in p.discard if is_energy(c) and getattr(c, "energy_type", None) == "fighting")
        other_count = sum(1 for c in p.discard if is_energy(c) and getattr(c, "energy_type", None) not in ("lightning", "fighting"))
        taken = []
        for _ in range(2):
            for i, c in enumerate(p.discard):
                if is_energy(c):
                    p.discard.pop(i)
                    taken.append(c)
                    break
        if taken:
            p.hand.extend(taken)
            p.discard.append(p.hand.pop(hand_index))
            state.log(f"{state.player_name(state.current_player)}: エネルギー回収を使用 → トラッシュから基本エネルギー {len(taken)} 枚を手札に加えた")
            return True
        return False

    if cid == "erekijienereta" and p.bench:
        look = min(5, len(p.deck))
        top = [p.deck.pop(0) for _ in range(look)]
        energies = [c for c in top if is_energy(c)][:2]
        for c in energies:
            bi = min(range(len(p.bench)), key=lambda i: p.bench[i].attached_energy)
            p.bench[bi].attached_energy += 1
            et = getattr(c, "energy_type", None) or "colorless"
            p.bench[bi].attached_energy_types.append(et)
        rest = [c for c in top if c not in energies]
        p.deck.extend(rest)
        random.shuffle(p.deck)
        p.discard.append(p.hand.pop(hand_index))
        if energies:
            state.log(f"{state.player_name(state.current_player)}: エレキジェネレーターを使用 → 山札上から 5 枚のうち基本エネルギー {len(energies)} 枚をベンチにつけた")
            return True
        state.log(f"{state.player_name(state.current_player)}: エレキジェネレーターを使用（エネルギーなし）")
        return True

    if cid == "supaboru" and p.deck:
        look = min(7, len(p.deck))
        top = [p.deck.pop(0) for _ in range(look)]
        pokemon = next((c for c in top if is_pokemon(c)), None)
        if pokemon:
            p.hand.append(pokemon)
            state.drawn_this_turn.append(pokemon)
            top.remove(pokemon)
        random.shuffle(p.deck)
        p.deck.extend(top)
        random.shuffle(p.deck)
        p.discard.append(p.hand.pop(hand_index))
        pokemon_label = _card_label(pokemon) if pokemon else ""
        state.log(
            f"{state.player_name(state.current_player)}: スーパーボールを使用 → 山札上から {look} 枚を見てポケモン 1 枚を手札に加えた → {pokemon_label}"
            if pokemon_label
            else f"{state.player_name(state.current_player)}: スーパーボールを使用 → 山札上から {look} 枚を見たがポケモンなし"
        )
        return True

    if cid == "haipaboru" and len(p.hand) >= 3 and p.deck:
        to_discard_idx = [i for i in range(len(p.hand)) if i != hand_index][:2]
        for i in sorted(to_discard_idx, reverse=True):
            p.discard.append(p.hand.pop(i))
        new_hi = p.hand.index(card)
        pokemon_found = None
        for i, c in enumerate(p.deck):
            if is_pokemon(c):
                pokemon_found = (i, c)
                break
        if pokemon_found:
            i, c = pokemon_found
            p.deck.pop(i)
            p.hand.append(c)
            state.drawn_this_turn.append(c)
            random.shuffle(p.deck)
        p.discard.append(p.hand.pop(new_hi))
        add_label = f" → {_card_label(pokemon_found[1])}" if pokemon_found else ""
        state.log(
            f"{state.player_name(state.current_player)}: ハイパーボールを使用（手札 2 枚トラッシュ）→ 山札からポケモン 1 枚を手札に加えた{add_label}"
            if pokemon_found
            else f"{state.player_name(state.current_player)}: ハイパーボールを使用（手札 2 枚トラッシュ、山札にポケモンなし）"
        )
        return True

    if cid == "pokemonkixyatchixya":
        opp = state.defending_player_state()
        if opp.bench and opp.active and _flip_coin():
            idx = random.randint(0, len(opp.bench) - 1)
            opp.active, opp.bench[idx] = opp.bench[idx], opp.active
            p.discard.append(p.hand.pop(hand_index))
            state.log(f"{state.player_name(state.current_player)}: ポケモンキャッチャーを使用（コイン表）→ 相手のベンチとバトルポケモンを入れ替えた")
            return True
        p.discard.append(p.hand.pop(hand_index))
        state.log(f"{state.player_name(state.current_player)}: ポケモンキャッチャーを使用（コイン裏）→ 効果なし")
        return True

    return False


def attach_tool(state: GameState, hand_index: int, bench_index: int | None = None) -> bool:
    """ポケモンのどうぐを手札から自分のバトル場またはベンチのポケモンにつける（1 匹に 1 枚まで）。"""
    p = state.active_player_state()
    if hand_index < 0 or hand_index >= len(p.hand):
        return False
    card = p.hand[hand_index]
    if not is_goods(card) or not getattr(card, "is_tool", False):
        return False
    if bench_index is None:
        target = p.active
    else:
        if bench_index < 0 or bench_index >= len(p.bench):
            return False
        target = p.bench[bench_index]
    if not target or getattr(target, "attached_tool", None) is not None:
        return False
    cond = getattr(card, "tool_condition_type", None)
    if cond is not None and getattr(target.card, "pokemon_type", None) != cond:
        return False
    target.attached_tool = card
    p.hand.pop(hand_index)
    where = "バトル場" if bench_index is None else "ベンチ"
    state.log(
        f"{state.player_name(state.current_player)}: {card.name} を {where}の {target.card.name} につけた"
    )
    return True


def use_pokemon_swap(state: GameState, hand_index: int, bench_index: int) -> bool:
    """ポケモンいれかえを使用して自分のバトル場のポケモンとベンチの指定 1 体を入れ替える。"""
    p = state.active_player_state()
    if not p.active or bench_index < 0 or bench_index >= len(p.bench):
        return False
    if hand_index < 0 or hand_index >= len(p.hand):
        return False
    card = p.hand[hand_index]
    if not is_goods(card):
        return False
    if getattr(card, "effect", None) != "swap_active" and getattr(card, "id", "") not in ("pokemon_irekae", "pokemonirekae"):
        return False
    old_active = p.active
    p.active = p.bench[bench_index]
    p.bench[bench_index] = old_active
    _clear_status(old_active)
    p.discard.append(p.hand.pop(hand_index))
    state.log(
        f"{state.player_name(state.current_player)}: ポケモンいれかえを使用 → バトル場に {p.active.card.name}、ベンチに {old_active.card.name} を入れ替え"
    )
    return True


def _is_first_player_first_turn(state: GameState) -> bool:
    """先行の 1 ターン目か（サポートは原則使用不可）。"""
    return state.turn_count == 0 and state.current_player == state.first_player


def use_support(state: GameState, hand_index: int) -> bool:
    """
    サポートカードを使用。1 ターンに 1 枚まで。先行の 1 ターン目は使用不可。
    ネモ: デッキから 3 枚引く。
    """
    p = state.active_player_state()
    if hand_index < 0 or hand_index >= len(p.hand):
        return False
    card = p.hand[hand_index]
    if not is_support(card):
        return False
    if state.support_used_this_turn:
        return False
    if _is_first_player_first_turn(state):
        return False
    effect = getattr(card, "effect", "")
    cid = getattr(card, "id", "")
    if cid == "tanpankozou":
        n_hand = len(p.hand)
        used_card = p.hand[hand_index]
        rest = [p.hand[j] for j in range(len(p.hand)) if j != hand_index]
        p.deck.extend(rest)
        p.hand = []
        random.shuffle(p.deck)
        drawn = p.draw(5)
        p.hand.extend(drawn)
        state.drawn_this_turn.extend(drawn)
        p.discard.append(used_card)
        state.support_used_this_turn = True
        drawn_names = ", ".join(_card_label(c) for c in drawn)
        state.log(f"{state.player_name(state.current_player)}: たんぱんこぞうを使用 → 手札 {n_hand} 枚を山札にもどして切り、山札から 5 枚ドロー → [{drawn_names}]（手札 {len(p.hand)} 枚）")
        return True
    if cid in ("hakasenokenkyuu", "hakasenokenkyuufutouhakase"):
        p.discard.extend(p.hand)
        p.hand.clear()
        drawn = p.draw(7)
        p.hand.extend(drawn)
        state.drawn_this_turn.extend(drawn)
        state.support_used_this_turn = True
        drawn_names = ", ".join(_card_label(c) for c in drawn)
        state.log(f"{state.player_name(state.current_player)}: 博士の研究を使用 → 手札をすべてトラッシュし、山札から 7 枚ドロー → [{drawn_names}]（手札 {len(p.hand)} 枚）")
        return True
    if cid == "jixyajjiman":
        opp = state.defending_player_state()
        opp.deck.extend(opp.hand)
        opp.hand = []
        random.shuffle(opp.deck)
        opp_drawn = opp.draw(4)
        opp.hand = opp_drawn
        opp_drawn_names = ", ".join(_card_label(c) for c in opp_drawn)
        state.log(f"{state.player_name(state.opponent())}: ジャッジマンの効果で手札を山札にもどして切り、山札から 4 枚ドロー → [{opp_drawn_names}]（手札 {len(opp.hand)} 枚）")
        p.hand.pop(hand_index)
        p.deck.extend(p.hand)
        p.hand = []
        random.shuffle(p.deck)
        p_drawn = p.draw(4)
        p.hand = p_drawn
        state.drawn_this_turn.extend(p_drawn)
        p.discard.append(card)
        p_drawn_names = ", ".join(_card_label(c) for c in p_drawn)
        state.log(f"{state.player_name(state.current_player)}: ジャッジマンの効果で手札を山札にもどして切り、山札から 4 枚ドロー → [{p_drawn_names}]（手札 {len(p.hand)} 枚）")
        state.support_used_this_turn = True
        return True
    if cid == "kihada":
        if len(p.hand) <= 1:
            return False
        kihada_card = p.hand.pop(hand_index)
        card_to_bottom = p.hand.pop(0)
        p.deck.append(card_to_bottom)
        need = 5 - len(p.hand)
        drawn = p.draw(need)
        p.hand.extend(drawn)
        state.drawn_this_turn.extend(drawn)
        p.discard.append(kihada_card)
        state.support_used_this_turn = True
        drawn_names = ", ".join(_card_label(c) for c in drawn)
        state.log(f"{state.player_name(state.current_player)}: キハダを使用 → 手札の 1 枚を山札の下にもどし、{need} 枚ドロー → [{drawn_names}]（手札 {len(p.hand)} 枚）")
        return True
    if effect == "draw_3" or cid in ("nemo", "nemokako", "nemomirai"):
        n = getattr(card, "draw_count", 3)
        drawn = p.draw(n)
        p.hand.extend(drawn)
        state.drawn_this_turn.extend(drawn)
        p.discard.append(p.hand.pop(hand_index))
        state.support_used_this_turn = True
        drawn_names = ", ".join(_card_label(c) for c in drawn)
        state.log(
            f"{state.player_name(state.current_player)}: {card.name} を使用 → 山札から {len(drawn)} 枚ドロー → [{drawn_names}]（手札 {len(p.hand)} 枚、山札 {len(p.deck)} 枚）"
        )
        return True
    return False


def _apply_evolution(
    target: BattlePokemon,
    evolution_card: PokemonCard,
    state: GameState,
    log_prefix: str,
) -> None:
    """進化を 1 体に適用（HP・エネルギーを引き継ぎ、カード差し替え）。"""
    old_name = target.card.name
    old_hp = target.hp
    old_max_hp = target.card.max_hp
    old_energy = target.attached_energy
    old_energy_types = getattr(target, "attached_energy_types", [])
    damage_taken = old_max_hp - old_hp
    evolved = evolution_card.copy()
    target.card = evolved
    target.card.max_hp = evolved.max_hp
    target.card.hp = max(0, min(evolved.max_hp, evolved.max_hp - damage_taken))
    target.attached_energy = old_energy
    target.attached_energy_types = list(old_energy_types)
    target.evolved_this_turn = True  # 同一ターンでの 2 進化を禁止するため
    _clear_status(target)  # 進化したとき特殊状態・ワザの効果はなくなる（ルール A-05）
    state.log(f"{log_prefix}{old_name} を {evolved.name} に進化（HP {target.hp}/{target.max_hp}）")


def _can_evolve_onto(field_card, evolution_card) -> bool:
    """場のポケモン（field_card）が進化カード（evolution_card）の進化元か。id または name で一致させる。"""
    base = (evolution_card.evolves_from or "").strip()
    if not base:
        return False
    fid = (getattr(field_card, "id", None) or "").strip()
    fname = (getattr(field_card, "name", "") or getattr(field_card, "name_ja", "") or "").strip()
    if fid == base or fname == base:
        return True
    # 進化元が短い id（例: meguroko）で、場のカードがセット付き id（例: meguroko-svd-062）のときも一致
    if base and (fid.startswith(base + "-") or fid.startswith(base + "_")):
        return True
    return False


def evolve_pokemon(state: GameState, hand_index: int, bench_index: int | None = None) -> bool:
    """
    手札の進化ポケモンで、バトル場またはベンチのポケモンを進化させる。
    bench_index=None ならバトル場、数値ならベンチのそのインデックス。
    evolves_from は進化元の id または name_ja（日本語名）でよい。set 付き id のポケモンも名前で一致すれば進化できる。
    """
    p = state.active_player_state()
    if hand_index < 0 or hand_index >= len(p.hand):
        return False
    evolution_card = p.hand[hand_index]
    if not is_pokemon(evolution_card) or not evolution_card.evolves_from:
        return False

    if bench_index is None:
        if not p.active or not _can_evolve_onto(p.active.card, evolution_card):
            return False
        # 2 進化：そのターンに 1 進化していたら同じターンには 2 進化できない（ルール A-05）
        if getattr(p.active.card, "evolves_from", None) is not None and getattr(p.active, "evolved_this_turn", False):
            return False
        # 場に出したばかりのポケモンはその番には進化できない（ルール A-05）
        if getattr(p.active, "put_on_bench_this_turn", False):
            return False
        _apply_evolution(
            p.active, evolution_card, state,
            f"{state.player_name(state.current_player)}: ",
        )
        p.hand.pop(hand_index)
        return True
    if bench_index < 0 or bench_index >= len(p.bench):
        return False
    bench_pokemon = p.bench[bench_index]
    if not _can_evolve_onto(bench_pokemon.card, evolution_card):
        return False
    # 2 進化：そのターンに 1 進化していたら同じターンには 2 進化できない（ルール A-05）
    if getattr(bench_pokemon.card, "evolves_from", None) is not None and getattr(bench_pokemon, "evolved_this_turn", False):
        return False
    # 場に出したばかりのポケモンはその番には進化できない（ルール A-05）
    if getattr(bench_pokemon, "put_on_bench_this_turn", False):
        return False
    _apply_evolution(
        bench_pokemon, evolution_card, state,
        f"{state.player_name(state.current_player)}: ベンチの ",
    )
    p.hand.pop(hand_index)
    return True


def attack(
    state: GameState,
    attack_index: int,
) -> bool:
    """
    攻撃を実行。ねむり・マヒ中は攻撃不可。こんらん時はコインで失敗時は自分に30ダメージ。
    相手にダメージ・状態異常、自分に self_damage を適用。
    """
    p = state.active_player_state()
    opp = state.defending_player_state()
    if not p.active or not opp.active:
        return False
    if getattr(p.active, "special_state", None) in ("sleep", "paralysis"):
        state.log(f"{state.player_name(state.current_player)}: {p.active.card.name} は状態異常のためワザが使えない")
        return False
    pokemon_card = p.active.card
    if attack_index < 0 or attack_index >= len(pokemon_card.attacks):
        return False
    atk = pokemon_card.attacks[attack_index]
    if getattr(p.active, "disabled_attack_name", None) == atk.name:
        state.log(f"{state.player_name(state.current_player)}: {p.active.card.name} はこのターン「{atk.name}」が使えない")
        return False
    if atk.name == "ついげきバリバリ":
        last_name = state.last_turn_attack_name[state.current_player]
        last_id = state.last_turn_attack_actor_id[state.current_player]
        actor_id = getattr(p.active.card, "id", getattr(p.active.card, "name", ""))
        if last_name != "しびれはり" or last_id != actor_id:
            state.log(f"{state.player_name(state.current_player)}: 「ついげきバリバリ」は前の自分の番にこのポケモンが「しびれはり」を使っていないと使えない")
            return False
    types = getattr(p.active, "attached_energy_types", [])
    if not _can_pay_energy_cost(
        p.active.attached_energy, types,
        atk.energy_cost, getattr(atk, "energy_cost_typed", None),
    ):
        return False
    if getattr(p.active, "special_state", None) == "confusion":
        if not _flip_coin():
            self_before = p.active.hp
            p.active.hp -= 30
            state.log(f"{state.player_name(state.current_player)}: こんらんでコインが裏 → 自分に 30 ダメージ（HP {self_before} → {max(0, p.active.hp)}）、攻撃失敗")
            if p.active and p.active.hp <= 0:
                state.log(f"{state.player_name(state.current_player)} のバトル場のポケモンがこんらんの自傷できぜつ！")
                koed_conf = p.active
                p.discard.append(koed_conf.card)
                tool = getattr(koed_conf, "attached_tool", None)
                if tool:
                    p.discard.append(tool)
                p.active = None
                _promote_from_bench(p, state, state.current_player)
            return True
        state.log(f"{state.player_name(state.current_player)}: こんらんコイン：表 → 通常通り攻撃")
    opp_before = opp.active.hp
    coin_flips = getattr(atk, "coin_flips", 0)
    damage_per_coin = getattr(atk, "damage_per_coin", 0)
    if coin_flips > 0 and damage_per_coin > 0:
        heads = sum(1 for _ in range(coin_flips) if _flip_coin())
        damage = heads * damage_per_coin
        state.log(f"{state.player_name(state.current_player)}: 「{atk.name}」コイン {coin_flips} 回 → 表 {heads} 回、ダメージ {damage}")
    else:
        damage = atk.damage
    defender_card = opp.active.card
    attacker_card = pokemon_card
    if (
        getattr(defender_card, "weakness", None)
        and getattr(attacker_card, "pokemon_type", None)
        and defender_card.weakness == attacker_card.pokemon_type
    ):
        damage *= 2
        state.log(f"{state.player_name(state.current_player)}: 弱点一致！ダメージが 2 倍（{damage // 2} → {damage}）")
    if (
        getattr(defender_card, "resistance", None)
        and getattr(attacker_card, "pokemon_type", None)
        and defender_card.resistance == attacker_card.pokemon_type
    ):
        before_r = damage
        damage = max(0, damage - 30)
        state.log(f"{state.player_name(state.current_player)}: 抵抗力一致！ダメージが -30（{before_r} → {damage}）")
    tool = getattr(opp.active, "attached_tool", None)
    if tool and getattr(tool, "is_tool", False) and getattr(tool, "tool_damage_reduce", 0) > 0:
        cond = getattr(tool, "tool_condition_type", None)
        if cond is None or getattr(defender_card, "pokemon_type", None) == cond:
            before_t = damage
            damage = max(0, damage - getattr(tool, "tool_damage_reduce", 0))
            state.log(f"{state.player_name(state.current_player)}: {opp.active.card.name} の {tool.name} でダメージ -{before_t - damage}（{before_t} → {damage}）")
    if atk.name == "しっぺがえし" and len(opp.prize_pile) == 1:
        damage += 90
        state.log(f"{state.player_name(state.current_player)}: 「{atk.name}」で相手に {damage} ダメージ（相手サイド残り 1 枚のため 90 ダメージ追加、相手 HP {opp_before} → {max(0, opp_before - damage)}）")
    elif atk.name == "アベンジナックル" and state.our_ko_by_damage_last_turn[state.current_player]:
        damage += 120
        state.log(f"{state.player_name(state.current_player)}: 「{atk.name}」で相手に {damage} ダメージ（前の相手の番に自分の闘ポケモンがきぜつしたため 120 ダメージ追加、相手 HP {opp_before} → {max(0, opp_before - damage)}）")
    else:
        state.log(f"{state.player_name(state.current_player)}: 「{atk.name}」で相手に {damage} ダメージ（相手 HP {opp_before} → {max(0, opp_before - damage)}）")
    if getattr(p.active, "damage_reduction_next_turn", 0) > 0:
        red = p.active.damage_reduction_next_turn
        damage = max(0, damage - red)
        p.active.damage_reduction_next_turn = 0
        if red > 0:
            state.log(f"{state.player_name(state.current_player)}: 前の相手の番に受けた「なきごえ」のためダメージ -{red}（→ {damage}）")
    skip_opponent_status = False
    if getattr(opp.active, "protected_next_opponent_turn", False):
        opp.active.protected_next_opponent_turn = False
        damage = 0
        skip_opponent_status = True
        state.log(f"{state.player_name(state.current_player)}: 相手の {opp.active.card.name} は「なぐってかくれる」のためワザのダメージ・効果を受けない")
    opp.active.hp -= damage
    bench_dmg = getattr(atk, "bench_damage", 0)
    bench_count = getattr(atk, "bench_damage_count", 1)
    bench_target_self = getattr(atk, "bench_damage_target", "opponent") == "self"
    bench_list = p.bench if bench_target_self else opp.bench
    if bench_dmg > 0 and bench_list:
        n = len(bench_list) if bench_count == 0 else min(bench_count, len(bench_list))
        for i in range(n):
            bench_target = bench_list[i]
            bench_before = bench_target.hp
            bench_target.hp -= bench_dmg
            who = "自分のベンチの" if bench_target_self else f"相手のベンチの"
            state.log(f"{state.player_name(state.current_player)}: 「{atk.name}」で{who} {bench_target.card.name} に {bench_dmg} ダメージ（HP {bench_before} → {max(0, bench_target.hp)}）")
        for i in range(n - 1, -1, -1):
            if i < len(bench_list) and bench_list[i].hp <= 0:
                koed_bench_bp = bench_list[i]
                bp_name = koed_bench_bp.card.name
                bench_list.pop(i)
                owner = p if bench_target_self else opp
                owner.discard.append(koed_bench_bp.card)
                tool = getattr(koed_bench_bp, "attached_tool", None)
                if tool:
                    owner.discard.append(tool)
                if bench_target_self:
                    state.log(f"{state.player_name(state.current_player)} のベンチの {bp_name} がきぜつ！")
                    prize_count = _prizes_for_ko(koed_bench_bp)
                    if prize_count > 1:
                        state.log(f"{state.player_name(state.opponent())}: {koed_bench_bp.card.name} は {'メガ' if prize_count == 3 else 'ex'} のため、サイドを {prize_count} 枚とる")
                    for _ in range(prize_count):
                        if _take_prize(state, state.opponent()):
                            return True
                else:
                    state.log(f"{state.player_name(state.opponent())} のベンチの {bp_name} がきぜつ！（{opp.knockouts_suffered + 1} 回目）")
                    if _handle_opponent_ko(opp, state, koed_bench_bp):
                        return True

    self_dmg = getattr(atk, "self_damage", 0)
    if self_dmg > 0 and p.active:
        self_before = p.active.hp
        p.active.hp -= self_dmg
        state.log(f"{state.player_name(state.current_player)}: 反動で自分に {self_dmg} ダメージ（自分 HP {self_before} → {max(0, p.active.hp)}）")

    status_effect = getattr(atk, "status_effect", None)
    if status_effect:
        target_bp = p.active if getattr(atk, "status_effect_target", "opponent") == "self" else (opp.active if opp.active else None)
        if target_bp and not (skip_opponent_status and target_bp == opp.active):
            on_coin_heads = getattr(atk, "status_effect_on_coin_heads", False)
            if on_coin_heads:
                if _flip_coin():
                    _apply_status(
                        state, target_bp, status_effect,
                        poison_damage=getattr(atk, "poison_damage_if_poison", 10),
                        log_prefix=f"{state.player_name(state.current_player)}: 「{atk.name}」コイン表 → ",
                    )
                else:
                    state.log(f"{state.player_name(state.current_player)}: 「{atk.name}」コイン裏 → 状態異常は付与されなかった")
            else:
                _apply_status(
                    state, target_bp, status_effect,
                    poison_damage=getattr(atk, "poison_damage_if_poison", 10),
                    log_prefix=f"{state.player_name(state.current_player)}: 「{atk.name}」→ ",
                )

    if atk.name == "なぐってかくれる" and _flip_coin():
        p.active.protected_next_opponent_turn = True
        state.log(f"{state.player_name(state.current_player)}: 「{atk.name}」コイン表 → 次の相手の番、このポケモンはワザのダメージや効果を受けない")
    elif atk.name == "なきごえ" and opp.active:
        opp.active.damage_reduction_next_turn = 20
        state.log(f"{state.player_name(state.current_player)}: 「{atk.name}」→ 次の相手の番、相手のワザのダメージが -20 される")
    if atk.name == "かそくづき":
        p.active.disabled_attack_name = "かそくづき"
        state.log(f"{state.player_name(state.current_player)}: 次の自分の番、このポケモンは「かそくづき」が使えなくなる")

    state.this_turn_attack_name = atk.name
    state.this_turn_attack_actor_id = getattr(p.active.card, "id", getattr(p.active.card, "name", ""))

    if atk.name == "ふきあらす":
        n_hand = len(opp.hand)
        opp.deck.extend(opp.hand)
        opp.hand = []
        random.shuffle(opp.deck)
        opp_drawn = opp.draw(4)
        opp.hand = opp_drawn
        opp_drawn_names = ", ".join(_card_label(c) for c in opp_drawn)
        state.log(f"{state.player_name(state.current_player)}: 「{atk.name}」→ 相手は手札 {n_hand} 枚を山札にもどして切り、山札から 4 枚ドロー → [{opp_drawn_names}]（手札 {len(opp.hand)} 枚）")
    elif atk.name == "マグネリジェクト" and opp.active and opp.bench:
        idx = random.randint(0, len(opp.bench) - 1)
        opp.active, opp.bench[idx] = opp.bench[idx], opp.active
        state.log(f"{state.player_name(state.current_player)}: 「{atk.name}」→ 相手のバトルポケモンとベンチを入れ替えた（{opp.active.card.name} がバトル場に）")
    elif atk.name == "ともだちをさがす":
        for i, c in enumerate(p.deck):
            if is_pokemon(c):
                p.deck.pop(i)
                p.hand.append(c)
                state.drawn_this_turn.append(c)
                random.shuffle(p.deck)
                state.log(f"{state.player_name(state.current_player)}: 「{atk.name}」→ 山札からポケモン 1 枚（{_card_label(c)}）を手札に加え、山札を切った")
                break
    elif atk.name == "クイックドロー":
        drawn = p.draw(2)
        p.hand.extend(drawn)
        state.drawn_this_turn.extend(drawn)
        state.log(f"{state.player_name(state.current_player)}: 「{atk.name}」→ 山札から 2 枚ドロー → [{', '.join(_card_label(c) for c in drawn)}]（手札 {len(p.hand)} 枚）")
    elif atk.name == "エレキチャージ" and p.active:
        attached = 0
        for i in range(len(p.deck) - 1, -1, -1):
            if attached >= 2:
                break
            if is_energy(p.deck[i]):
                p.active.attached_energy += 1
                et = getattr(p.deck[i], "energy_type", None) or "colorless"
                p.active.attached_energy_types.append(et)
                p.deck.pop(i)
                attached += 1
        if attached > 0:
            random.shuffle(p.deck)
            state.log(f"{state.player_name(state.current_player)}: 「{atk.name}」→ 山札から基本エネルギー {attached} 枚をこのポケモンにつけた")
    elif atk.name == "10まんボルト" and p.active:
        num = p.active.attached_energy
        types_to_discard = list(getattr(p.active, "attached_energy_types", []))
        _put_energy_cards_in_discard(p, types_to_discard, state)
        p.active.attached_energy = 0
        p.active.attached_energy_types = []
        state.log(f"{state.player_name(state.current_player)}: 「{atk.name}」→ このポケモンについているエネルギー {num} 個をすべてトラッシュした")
    elif atk.name == "ガブガブバイト" and opp.active:
        heads = 0
        while _flip_coin():
            heads += 1
        discard_n = min(heads, opp.active.attached_energy)
        if discard_n > 0:
            types = getattr(opp.active, "attached_energy_types", [])
            discarded_types = types[-discard_n:] if len(types) >= discard_n else list(types)
            _put_energy_cards_in_discard(opp, discarded_types, state)
            opp.active.attached_energy -= discard_n
            if len(types) >= discard_n:
                opp.active.attached_energy_types = types[:-discard_n]
            else:
                opp.active.attached_energy_types = []
            state.log(f"{state.player_name(state.current_player)}: 「{atk.name}」→ コイン表 {heads} 回、相手のバトルポケモンのエネルギー {discard_n} 個をトラッシュした")
    elif atk.name == "テクノターボ" and p.discard and p.bench:
        lightning_idx = None
        for i, c in enumerate(p.discard):
            if is_energy(c) and getattr(c, "energy_type", None) == "lightning":
                lightning_idx = i
                break
        if lightning_idx is not None:
            p.discard.pop(lightning_idx)
            bench_idx = min(range(len(p.bench)), key=lambda i: p.bench[i].attached_energy)
            target = p.bench[bench_idx]
            target.attached_energy += 1
            target.attached_energy_types.append("lightning")
            state.log(f"{state.player_name(state.current_player)}: 「{atk.name}」→ トラッシュから基本雷エネルギー 1 枚をベンチの {target.card.name} につけた")

    if opp.active and opp.active.hp <= 0:
        koed_active = opp.active
        state.our_ko_by_damage_last_turn[state.opponent()] = True
        state.log(f"{state.player_name(state.opponent())} のバトル場の {koed_active.card.name} がきぜつ！（{opp.knockouts_suffered + 1} 回目）")
        opp.active = None
        if _handle_opponent_ko(opp, state, koed_active):
            return True

    if p.active and p.active.hp <= 0:
        state.log(f"{state.player_name(state.current_player)} のバトル場のポケモンが反動できぜつ！")
        koed_recoil = p.active
        p.discard.append(koed_recoil.card)
        tool = getattr(koed_recoil, "attached_tool", None)
        if tool:
            p.discard.append(tool)
        p.active = None
        _promote_from_bench(p, state, state.current_player)
    return True


# 攻撃選択で KO を最優先するためのボーナス（有效ダメージは通常 300 以下なので十分大きい値）
_KO_BONUS_FOR_ATTACK = 10000


def _choose_best_attack_index(state: GameState, p: PlayerState, opp: PlayerState) -> int | None:
    """出せる技のうち、相手をきぜつさせられる技を最優先し、否則有效ダメージが最大のインデックスを返す。"""
    if not p.active or not p.active.card.attacks:
        return None
    opp_hp = opp.active.hp if opp.active else 0
    best_idx = None
    best_score = -1
    types = getattr(p.active, "attached_energy_types", [])
    for idx, atk in enumerate(p.active.card.attacks):
        if not _can_pay_energy_cost(
            p.active.attached_energy, types,
            atk.energy_cost, getattr(atk, "energy_cost_typed", None),
        ):
            continue
        if getattr(p.active, "disabled_attack_name", None) == atk.name:
            continue
        if atk.name == "ついげきバリバリ":
            last_name = state.last_turn_attack_name[state.current_player]
            last_id = state.last_turn_attack_actor_id[state.current_player]
            actor_id = getattr(p.active.card, "id", getattr(p.active.card, "name", ""))
            if last_name != "しびれはり" or last_id != actor_id:
                continue
        base_dmg = _attack_damage_for_eval(atk)
        if atk.name == "しっぺがえし" and len(opp.prize_pile) == 1:
            base_dmg += 90
        if atk.name == "アベンジナックル" and state.our_ko_by_damage_last_turn[state.current_player]:
            base_dmg += 120
        effective_dmg = _effective_damage_to_defender(p.active.card, opp.active, base_dmg) if opp.active else base_dmg
        # きぜつさせられる技を最優先、否則有效ダメージで比較
        ko_bonus = _KO_BONUS_FOR_ATTACK if (opp.active and effective_dmg >= opp_hp) else 0
        score = effective_dmg + ko_bonus
        if score > best_score:
            best_score = score
            best_idx = idx
    return best_idx


def _should_attach_for_evolution(p: PlayerState) -> bool:
    """バトル場のポケモンが手札の進化で必要エネルギーに足りていなければ True。"""
    if not p.active:
        return False
    types = getattr(p.active, "attached_energy_types", [])
    for c in p.hand:
        if not (is_pokemon(c) and _can_evolve_onto(p.active.card, c)):
            continue
        for a in c.attacks:
            if _can_pay_energy_cost(
                p.active.attached_energy, types,
                a.energy_cost, getattr(a, "energy_cost_typed", None),
            ):
                return False
        return True
    return False


def _energy_needed_for_active(p: PlayerState) -> int:
    """バトル場（＋手札の進化）に必要なエネルギーコスト合計を返す。"""
    if not p.active:
        return 0
    need = max((a.energy_cost for a in p.active.card.attacks), default=0)
    for c in p.hand:
        if not (is_pokemon(c) and _can_evolve_onto(p.active.card, c)):
            continue
        need = max(need, max((a.energy_cost for a in c.attacks), default=0))
        break
    return need


def _can_active_use_any_attack(p: PlayerState) -> bool:
    """バトル場のポケモンがどれか 1 つでも技を出せるか（タイプ指定込みで判定）。"""
    if not p.active or not p.active.card.attacks:
        return False
    types = getattr(p.active, "attached_energy_types", [])
    for atk in p.active.card.attacks:
        if _can_pay_energy_cost(
            p.active.attached_energy, types,
            atk.energy_cost, getattr(atk, "energy_cost_typed", None),
        ):
            return True
    return False


def _max_energy_for_pokemon(attacks: list) -> int:
    """そのポケモンの技で必要な最大エネルギー数（個数）を返す。"""
    return max((a.energy_cost for a in attacks), default=0)


def _max_effective_damage_if_attach(
    state: GameState,
    attacker_card: PokemonCard,
    attached_count: int,
    attached_types: list,
    extra_type: str,
    defender: "BattlePokemon | None",
    current_player: int,
) -> int:
    """
    このポケモンに extra_type を 1 つ付けたとき、
    相手のバトル場（defender）に与えられる最大の有效ダメージを返す。
    """
    if not attacker_card.attacks:
        return 0
    sim_count = attached_count + 1
    sim_types = list(attached_types) + [extra_type]
    max_eff = 0
    opp = state.defending_player_state()
    for atk in attacker_card.attacks:
        if not _can_pay_energy_cost(
            sim_count, sim_types,
            atk.energy_cost, getattr(atk, "energy_cost_typed", None),
        ):
            continue
        base = _attack_damage_for_eval(atk)
        if atk.name == "しっぺがえし" and len(opp.prize_pile) == 1:
            base += 90
        if atk.name == "アベンジナックル" and state.our_ko_by_damage_last_turn[current_player]:
            base += 120
        eff = _effective_damage_to_defender(attacker_card, defender, base) if defender else base
        max_eff = max(max_eff, eff)
    return max_eff


def _try_attach_energy_auto(state: GameState) -> bool:
    """
    自動ターン用：エネルギーを 1 枚付与するなら True を返す。
    付与先は「このエネルギーを付けたときに相手に与えられる有效ダメージが最大のポケモン」で決める。
    技で必要な最大までしか付与しない。
    """
    p = state.active_player_state()
    can_evolve_this_turn = not _is_first_player_first_turn(state)
    energy_hand_idx = next((i for i, c in enumerate(p.hand) if is_energy(c)), None)
    if energy_hand_idx is None or not p.active:
        return False

    energy_card = p.hand[energy_hand_idx]
    new_type = getattr(energy_card, "energy_type", None) or "colorless"
    opp = state.defending_player_state()
    max_active = _max_energy_for_pokemon(p.active.card.attacks)
    energy_needed_active = _energy_needed_for_active(p)

    if p.active.attached_energy == 0:
        attach_energy(state, energy_hand_idx)
        return True
    if can_evolve_this_turn and _should_attach_for_evolution(p):
        attach_energy(state, energy_hand_idx)
        return True

    candidates = []
    if p.active.attached_energy < max_active:
        dmg = _max_effective_damage_if_attach(
            state,
            p.active.card,
            p.active.attached_energy,
            getattr(p.active, "attached_energy_types", []),
            new_type,
            opp.active,
            state.current_player,
        )
        candidates.append((None, dmg))
    for bi, b in enumerate(p.bench):
        max_b = _max_energy_for_pokemon(b.card.attacks)
        if b.attached_energy >= max_b:
            continue
        dmg = _max_effective_damage_if_attach(
            state,
            b.card,
            b.attached_energy,
            getattr(b, "attached_energy_types", []),
            new_type,
            opp.active,
            state.current_player,
        )
        candidates.append((bi, dmg))

    if not candidates:
        return False
    best = max(candidates, key=lambda x: (x[1], x[0] is None))
    if best[0] is None:
        attach_energy(state, energy_hand_idx)
        return True
    attach_energy(state, energy_hand_idx, bench_index=best[0])
    return True


def _apply_poison_burn_paralysis_for_active(
    state: GameState, player_index: int
) -> None:
    """
    ポケモンチェック（ルール F）：そのプレイヤーのバトル場ポケモンに
    どく・やけどダメージ、やけど回復コイン、マヒ解除を適用する。
    きぜつ時はベンチから繰り出し、勝敗判定は呼び出し元で行う。
    """
    p = state.players[player_index]
    if not p.active:
        return
    poison_dmg = getattr(p.active, "poison_damage", 0)
    if poison_dmg > 0:
        before = p.active.hp
        p.active.hp -= poison_dmg
        state.log(f"{state.player_name(player_index)}: どくで {p.active.card.name} に {poison_dmg} ダメージ（HP {before} → {max(0, p.active.hp)}）")
    if p.active and p.active.hp <= 0:
        koed_poison = p.active
        state.log(f"{state.player_name(player_index)} のバトル場の {koed_poison.card.name} がきぜつ！（どくのダメージ）")
        p.discard.append(koed_poison.card)
        tool = getattr(koed_poison, "attached_tool", None)
        if tool:
            p.discard.append(tool)
        p.knockouts_suffered += 1
        p.active = None
        prize_count = _prizes_for_ko(koed_poison)
        if prize_count > 1:
            state.log(f"{state.player_name(1 - player_index)}: {koed_poison.card.name} は {'メガ' if prize_count == 3 else 'ex'} のため、サイドを {prize_count} 枚とる")
        for _ in range(prize_count):
            if _take_prize(state, 1 - player_index):
                return
        if state.winner is None:
            _promote_from_bench(p, state, player_index)
    if not p.active:
        return
    if getattr(p.active, "burn", False):
        before = p.active.hp
        p.active.hp -= 20
        state.log(f"{state.player_name(player_index)}: やけどで {p.active.card.name} に 20 ダメージ（HP {before} → {max(0, p.active.hp)}）")
        if _flip_coin():
            p.active.burn = False
            state.log(f"{state.player_name(player_index)}: {p.active.card.name} のやけどが治った（コイン：表）")
        else:
            state.log(f"{state.player_name(player_index)}: {p.active.card.name} のやけどは継続（コイン：裏）")
    if player_index == state.current_player and getattr(p.active, "special_state", None) == "paralysis":
        p.active.special_state = None
        state.log(f"{state.player_name(player_index)}: {p.active.card.name} のマヒが解けた")
    if p.active and p.active.hp <= 0:
        koed_status = p.active
        state.log(f"{state.player_name(player_index)} のバトル場の {koed_status.card.name} がきぜつ！（状態異常のダメージ）")
        p.discard.append(koed_status.card)
        tool = getattr(koed_status, "attached_tool", None)
        if tool:
            p.discard.append(tool)
        p.knockouts_suffered += 1
        p.active = None
        prize_count = _prizes_for_ko(koed_status)
        if prize_count > 1:
            state.log(f"{state.player_name(1 - player_index)}: {koed_status.card.name} は {'メガ' if prize_count == 3 else 'ex'} のため、サイドを {prize_count} 枚とる")
        for _ in range(prize_count):
            if _take_prize(state, 1 - player_index):
                return
        if state.winner is None:
            _promote_from_bench(p, state, player_index)


def end_turn(state: GameState) -> None:
    """
    ターン終了：ポケモンチェック（ルール F）。
    おたがいのポケモンのどく・やけどを確認し、自分の番のマヒ解除。
    その後相手に交代、ターン数増加。
    """
    _apply_poison_burn_paralysis_for_active(state, state.current_player)
    if not _check_game_end(state):
        _apply_poison_burn_paralysis_for_active(state, state.opponent())
        _check_game_end(state)
    for pl in state.players:
        if pl.active:
            pl.active.evolved_this_turn = False
            pl.active.put_on_bench_this_turn = False
        for bp in pl.bench:
            bp.evolved_this_turn = False
            bp.put_on_bench_this_turn = False
    cp = state.current_player
    state.last_turn_attack_name[cp] = state.this_turn_attack_name
    state.last_turn_attack_actor_id[cp] = state.this_turn_attack_actor_id
    state.this_turn_attack_name = None
    state.this_turn_attack_actor_id = None
    state.current_player = state.opponent()
    state.turn_count += 1
    state.log("")


def run_turn_auto(state: GameState) -> bool:
    """
    現在のプレイヤーが「可能な行動を順番に実行」する。
    順序：0. ベンチにポケモン出す、1. 進化（最優先）、2. エネルギー付与、3. サポート（ネモ等でドロー）、4. ポケモンいれかえ、5. にげる（条件満たすとき）、6. どうぐ・グッズ、7. 攻撃。
    何もできなければ False を返す。True = ターン内で何かした。
    """
    p = state.active_player_state()
    opp = state.defending_player_state()
    if not p.active:
        return False
    
    acted = False

    def _try_put_bench_until_full():
        nonlocal p, acted
        while _put_one_pokemon_on_bench(p, state, state.current_player):
            acted = True
            p = state.active_player_state()
            state._record_frame()

    _try_put_bench_until_full()

    # 進化を最優先で行う（先行・後行の 1 ターン目以外）
    can_evolve = state.turn_count >= 2
    while can_evolve:
        p = state.active_player_state()
        evolved_this_round = False
        for hand_idx, c in enumerate(p.hand):
            if not is_pokemon(c) or not c.evolves_from:
                continue
            if p.active and _can_evolve_onto(p.active.card, c):
                evolve_pokemon(state, hand_idx, bench_index=None)
                acted = True
                evolved_this_round = True
                state._record_frame()
                break
            for bench_idx, bench_poke in enumerate(p.bench):
                if _can_evolve_onto(bench_poke.card, c):
                    evolve_pokemon(state, hand_idx, bench_index=bench_idx)
                    acted = True
                    evolved_this_round = True
                    state._record_frame()
                    break
            if evolved_this_round:
                break
        if not evolved_this_round:
            break

    if _try_attach_energy_auto(state):
        acted = True
        p = state.active_player_state()
        state._record_frame()

    opp_max_effective = _opponent_max_effective_damage(state)
    our_max_effective = _our_max_effective_damage(state)
    would_be_koed = p.active and opp_max_effective > 0 and p.active.hp <= opp_max_effective
    can_ko_opponent = opp.active and our_max_effective >= opp.active.hp

    _BALL_GOODS_IDS = ("supaboru", "haipaboru", "otodokedoron", "pokemonkixyatchixya")
    if not _is_first_player_first_turn(state):
        while True:
            used_ball = False
            for i, c in enumerate(p.hand):
                if not is_goods(c) or getattr(c, "is_tool", False):
                    continue
                if getattr(c, "id", "") not in _BALL_GOODS_IDS:
                    continue
                if use_trainer_goods(state, i):
                    acted = True
                    used_ball = True
                    p = state.active_player_state()
                    state._record_frame()
                    break
            if not used_ball:
                break
        _try_put_bench_until_full()

    if not _is_first_player_first_turn(state) and not state.support_used_this_turn:
        for i, c in enumerate(p.hand):
            if is_support(c) and use_support(state, i):
                acted = True
                p = state.active_player_state()
                state._record_frame()
                break
        _try_put_bench_until_full()

    # ポケモンいれかえ: 次の相手の番でやられそうなときだけ使う（そうでないときは攻撃を優先）
    p = state.active_player_state()
    opp_max_effective = _opponent_max_effective_damage(state)
    our_max_effective = _our_max_effective_damage(state)
    would_be_koed = p.active and opp_max_effective > 0 and p.active.hp <= opp_max_effective
    can_ko_opponent = opp.active and our_max_effective >= opp.active.hp
    if would_be_koed and p.active and p.bench:
        for i, c in enumerate(p.hand):
            if is_goods(c) and (getattr(c, "effect", None) == "swap_active" or getattr(c, "id", "") in ("pokemon_irekae", "pokemonirekae")):
                survives = [(bi, p.bench[bi].hp) for bi in range(len(p.bench)) if p.bench[bi].hp > opp_max_effective]
                if survives:
                    best_bench = max(survives, key=lambda x: x[1])[0]
                else:
                    best_bench = max(range(len(p.bench)), key=lambda b: p.bench[b].hp, default=None)
                if best_bench is not None and use_pokemon_swap(state, i, best_bench):
                    acted = True
                    p = state.active_player_state()
                    state._record_frame()
                break

    # にげる: サポート・ポケモンいれかえの後に判定（ネモで引いたポケモンいれかえを先に使えるように）
    p = state.active_player_state()
    opp_max_effective = _opponent_max_effective_damage(state)
    our_max_effective = _our_max_effective_damage(state)
    would_be_koed = p.active and opp_max_effective > 0 and p.active.hp <= opp_max_effective
    can_ko_opponent = opp.active and our_max_effective >= opp.active.hp
    retreat_cost = getattr(p.active.card, "retreat_cost", 1) if p.active else 0
    can_retreat = p.active and getattr(p.active, "special_state", None) not in ("sleep", "paralysis")
    if can_retreat and p.active and p.bench and would_be_koed and not can_ko_opponent and p.active.attached_energy >= retreat_cost:
        # ベンチ候補ごとの最大ダメージを見て、相手をきぜつさせられるベンチを最優先
        best_idx = None
        best_score = (-1, -1, -1)  # (can_ko_flag, damage, hp/energyの目安)
        for i, bp in enumerate(p.bench):
            dmg = _max_effective_damage_for_attacker(state, bp, opp.active, state.current_player) if opp.active else 0
            can_ko = int(opp.active is not None and dmg >= opp.active.hp)
            score = (can_ko, dmg, bp.hp)
            if score > best_score:
                best_score = score
                best_idx = i
        # それでも候補がなければ、従来どおり生存しやすいベンチを選ぶ
        if best_idx is None:
            survivors = [(i, p.bench[i].hp, p.bench[i].attached_energy) for i in range(len(p.bench)) if p.bench[i].hp > opp_max_effective]
            if survivors:
                best_idx = max(survivors, key=lambda x: (x[1], x[2]))[0]
            else:
                best_idx = max(
                    range(len(p.bench)),
                    key=lambda i: (p.bench[i].hp, p.bench[i].attached_energy),
                    default=None,
                )
        if best_idx is not None and retreat(state, best_idx):
            acted = True
            p = state.active_player_state()
            state._record_frame()

    # どうぐ（tool）を先に試す（ダメージ軽減など、つけておきたいカードを優先）
    for i, c in enumerate(p.hand):
        if not is_goods(c) or not getattr(c, "is_tool", False):
            continue
        cond = getattr(c, "tool_condition_type", None)
        if p.active and getattr(p.active, "attached_tool", None) is None:
            if cond is None or getattr(p.active.card, "pokemon_type", None) == cond:
                if attach_tool(state, i, bench_index=None):
                    acted = True
                    p = state.active_player_state()
                    state._record_frame()
                    break
        for bi, bp in enumerate(p.bench):
            if getattr(bp, "attached_tool", None) is not None:
                continue
            if cond is not None and getattr(bp.card, "pokemon_type", None) != cond:
                continue
            if attach_tool(state, i, bench_index=bi):
                acted = True
                p = state.active_player_state()
                state._record_frame()
                break
        else:
            continue
        break

    for i, c in enumerate(p.hand):
        if not is_goods(c) or getattr(c, "effect", None) == "swap_active" or getattr(c, "is_tool", False):
            continue
        if getattr(c, "id", None) == "potion" and getattr(c, "effect", None) == "heal":
            used = use_potion(state, i)
        elif getattr(c, "effect", None) == "heal":
            used = False
        else:
            used = use_trainer_goods(state, i)
        if used:
            acted = True
            p = state.active_player_state()
            state._record_frame()
            break
    p = state.active_player_state()
    _try_put_bench_until_full()

    # 進化を最優先で行う（サポートやグッズで手札が変わったあとも再度チェック）
    can_evolve = state.turn_count >= 2
    while can_evolve:
        p = state.active_player_state()
        evolved_this_round = False
        for hand_idx, c in enumerate(p.hand):
            if not is_pokemon(c) or not c.evolves_from:
                continue
            if p.active and _can_evolve_onto(p.active.card, c):
                evolve_pokemon(state, hand_idx, bench_index=None)
                acted = True
                evolved_this_round = True
                state._record_frame()
                break
            for bench_idx, bench_poke in enumerate(p.bench):
                if _can_evolve_onto(bench_poke.card, c):
                    evolve_pokemon(state, hand_idx, bench_index=bench_idx)
                    acted = True
                    evolved_this_round = True
                    state._record_frame()
                    break
            if evolved_this_round:
                break
        if not evolved_this_round:
            break

    is_game_first_turn = state.turn_count == 0
    can_attack = not is_game_first_turn and p.active and getattr(p.active, "special_state", None) not in ("sleep", "paralysis")
    if can_attack:
        best_idx = _choose_best_attack_index(state, p, opp)
        if best_idx is not None:
            attack(state, best_idx)
            acted = True
            state._record_frame()

    if not acted:
        state.log(f"{state.player_name(state.current_player)}: 実行できるアクションなし（パス）")
        state._record_frame()

    return acted


def run_game_auto(state: GameState) -> int:
    """
    自動でターンを進め、勝者が決まるまで実行。
    デッキ切れ・サイド取り切り・ポケモン全滅で必ず決着。戻り値: 0 or 1 = 勝者。
    """
    if _check_game_end(state):
        if state.log_fn:
            state.log("========== ゲーム終了 ==========\n")
        return state.winner
    while True:
        start_turn(state)
        if state.winner is not None:
            if state.log_fn:
                state.log("========== ゲーム終了 ==========\n")
            return state.winner
        acted = run_turn_auto(state)
        if _check_game_end(state):
            if state.log_fn:
                state.log("========== ゲーム終了 ==========\n")
            return state.winner
        if not acted:
            end_turn(state)
        else:
            end_turn(state)
        if state.turn_count >= MAX_TURNS_SAFETY:
            p0, p1 = state.players[0], state.players[1]
            taken0 = PRIZE_COUNT - len(p0.prize_pile)
            taken1 = PRIZE_COUNT - len(p1.prize_pile)
            state.winner = 0 if taken0 >= taken1 else 1
            if state.log_fn:
                state.log(f"{MAX_TURNS_SAFETY} ターンで打ち切り（サイド取得で判定）\n")
                state.log("========== ゲーム終了 ==========\n")
            return state.winner
