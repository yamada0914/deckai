"""
ポケカ風ルールのゲームロジック。

- 初手 4 枚、ターンでドロー・エネルギー付与・アイテム・攻撃。
- ベンチは最大 5 体。きぜつしたらベンチから 1 体をバトル場に出す。
- 相手のポケモンを 3 回きぜつさせたら勝ち。プレイヤーは先行・後攻の 2 人。
"""
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Literal

from card import PokemonCard, is_energy, is_goods, is_pokemon, is_support
from deck import STARTING_HAND_SIZE, create_deck, create_deck_from_deck_code, format_deck_recipe

BENCH_SIZE = 5
WIN_KO_COUNT = 3  # しっぺがえし等の参照用。勝敗はサイド取り切りで判定
PRIZE_COUNT = 3  # サイド（賞品札）の枚数。すべてとると勝ち
# デッキ切れ負けを無効にしている間の打ち切り用（通常時はデッキ切れで決着するため未到達）
MAX_TURNS_SAFETY = 200

# 状態異常：ねむり・マヒ・こんらんはどれか1つのみ。どく・やけどは重複可。ベンチに下がると全て解除。
SpecialState = Literal["sleep", "paralysis", "confusion"]


@dataclass
class BattlePokemon:
    """バトル場／ベンチに出すポケモン（HP・エネルギー・状態異常・どうぐを管理）"""
    card: PokemonCard
    attached_energy: int = 0
    # 付いているエネルギーをタイプごとに（1 つずつ）。長さは attached_energy と一致。無色＝任意として消費可能。
    attached_energy_types: list = field(default_factory=list)
    # ポケモンのどうぐ（1 匹に 1 枚まで。きぜつするまでついたまま）
    attached_tool: "GoodsCard | None" = None
    # 状態異常：ねむり・マヒ・こんらんはどれか1つのみ
    special_state: SpecialState | None = None
    poison_damage: int = 0  # 0 = どくにかかっていない。10/20/30 で毎ターン末尾にダメージ
    burn: bool = False
    # このターンに進化したか（そのターンに 1 進化したポケモンは同じターンに 2 進化できない）
    evolved_this_turn: bool = False
    # このターンにベンチに出したか（場に出したばかりのポケモンはその番には進化できない）
    put_on_bench_this_turn: bool = False

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
    prize_pile: list = field(default_factory=list)  # サイド。相手のポケモンをきぜつさせるとここから 1 枚とる
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


def _flip_coin() -> bool:
    """コインを1回投げる。True = 表、False = 裏。"""
    return random.random() < 0.5


def _clear_status(bp: BattlePokemon) -> None:
    """状態異常を全て解除（ベンチに下がったときなど）。"""
    bp.special_state = None
    bp.poison_damage = 0
    bp.burn = False


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
    # タイプ別必要数（無色以外）
    need = Counter(t for t in cost_typed if t != "colorless")
    have = Counter(attached_types)
    for typ, n in need.items():
        if have.get(typ, 0) < n:
            return False
    return True


def _player_name(state: "GameState", i: int) -> str:
    """使用デッキに応じたプレイヤー名を返す。同じデッキ同士のときはデッキ1/デッキ2、それ以外はオタチ/ワニ/カエルデッキ。"""
    d0, d1 = state.deck_indices[0], state.deck_indices[1]
    if d0 == d1:
        return f"デッキ{i + 1}"
    names = ["オタチデッキ", "ワニデッキ", "カエルデッキ", "ワルビアルデッキ", "ジバコイルデッキ"]
    idx = state.deck_indices[i] if i < len(state.deck_indices) else i
    return names[idx] if idx < len(names) else f"プレイヤー{i}"


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

    def log(self, msg: str) -> None:
        if self.log_fn:
            self.log_fn(msg)

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
    deck0: int = 0,
    deck1: int = 1,
    deck_code0: str | None = None,
    deck_code1: str | None = None,
) -> GameState:
    """
    デッキを組んで初手 4 枚をセット。
    deck0 / deck1 で固定デッキ番号（0=A, 1=B, 2=C, 3=D, 4=E）。
    deck_code0 / deck_code1 を指定すると read_cards_result.json のそのデッキコードの一覧でデッキを組む（未指定なら deck0 / deck1 を使用）。
    """
    if seed is not None:
        random.seed(seed)
    state = GameState(log_fn=log_fn, deck_indices=(deck0, deck1))
    state.log("========== ゲーム開始 ==========")
    deck_specs: list[tuple[int, str | None]] = [(deck0, deck_code0), (deck1, deck_code1)]
    for i in range(2):
        idx, code = deck_specs[i]
        if code:
            state.log(f"{state.player_name(i)} デッキ: [デッキコード {code}]")
        else:
            state.log(f"{state.player_name(i)} デッキ: [{format_deck_recipe(idx)}]")
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
        # サイドを PRIZE_COUNT 枚、デッキの上から取る（ルール G）
        for _ in range(PRIZE_COUNT):
            if state.players[i].deck:
                state.players[i].prize_pile.append(state.players[i].deck.pop())
        side_names = [_card_label(c) for c in state.players[i].prize_pile]
        state.log(f"{state.player_name(i)}: サイドを {PRIZE_COUNT} 枚置く → [{', '.join(side_names)}]（残りデッキ {len(state.players[i].deck)} 枚）")
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
            state.log(f"{state.player_name(i)}: 初手にたねポケモンなし → 再シャッフルして引き直し")
        msg = f"再シャッフル後、初手 {STARTING_HAND_SIZE} 枚ドロー → 手札 [{', '.join(hand_names)}]" if retry_count > 0 else f"初手 {STARTING_HAND_SIZE} 枚ドロー → 手札 [{', '.join(hand_names)}]"
        state.log(f"{state.player_name(i)}: {msg}")
        _put_one_pokemon_active(state.players[i])
        if state.players[i].active:
            state.log(f"{state.player_name(i)}: バトル場に {state.players[i].active.card.name} を出す（HP {state.players[i].active.max_hp}）")
        _fill_bench_from_hand(state.players[i], state, i, log_each=False)
        if state.players[i].bench:
            bench_names = [b.card.name for b in state.players[i].bench]
            state.log(f"{state.player_name(i)}: ベンチに {len(state.players[i].bench)} 体 [{', '.join(bench_names)}]")
    state.first_player = random.randint(0, 1)
    state.current_player = state.first_player
    state.log(f"先行: {state.player_name(state.first_player)} / 後攻: {state.player_name(1 - state.first_player)}")
    state.log("")
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
    for i, c in enumerate(player.hand):
        if is_pokemon(c) and not c.evolves_from:
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
        f"{state.player_name(taker_index)}: サイドを 1 枚とる → {_card_label(card)}（手札 {len(p.hand)} 枚、サイド残り {len(p.prize_pile)} 枚 → [{', '.join(remaining)}]）"
    )
    if len(p.prize_pile) == 0:
        state.winner = taker_index
        state.log(f"{state.player_name(taker_index)}: サイドをすべてとり終えた → 勝ち！")
        return True
    return False


def _handle_opponent_ko(opp: PlayerState, state: GameState, koed_bp: BattlePokemon) -> bool:
    """相手のきぜつを 1 回加算し、攻撃側がサイドをとる（ex なら 2 枚、それ以外は 1 枚）。サイド 0 なら勝ち、否則ベンチから繰り出す。"""
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
            # ログは _take_prize で「サイドをすべてとり終えた → 勝ち！」と出しているのでここでは出さない
            return True
        if not p.has_active_pokemon() and len(p.bench) == 0:
            state.winner = 1 - i
            state.log(f"{state.player_name(i)} のバトル場・ベンチにポケモンがいない → {state.player_name(state.winner)} の勝ち")
            return True
    return False


def start_turn(state: GameState) -> None:
    """ターン開始：ドロー 1 枚、サポート未使用にリセット。ねむりならコインで解除判定。"""
    state.support_used_this_turn = False
    p = state.active_player_state()
    turn_label = _turn_label(state)
    state.log(f"---------- {turn_label} ---------- {state.player_name(state.current_player)} のターン")
    drawn = p.draw(1)
    if drawn:
        p.hand.extend(drawn)
        state.log(f"{state.player_name(state.current_player)}: 1 枚ドロー → {_card_label(drawn[0])}（手札 {len(p.hand)} 枚、デッキ {len(p.deck)} 枚）")
    else:
        # 一時的にデッキ切れでも負けにしない（あとで戻す）
        state.log(f"{state.player_name(state.current_player)}: デッキが空のためドローなし（手札 {len(p.hand)} 枚）")
    # ねむり：自分の番のはじめにコイン。表 → 回復、裏 → そのまま
    if p.active and getattr(p.active, "special_state", None) == "sleep":
        if _flip_coin():
            p.active.special_state = None
            state.log(f"{state.player_name(state.current_player)}: {p.active.card.name} のねむりが解けた（コイン：表）")
        else:
            state.log(f"{state.player_name(state.current_player)}: {p.active.card.name} はねむったまま（コイン：裏）")


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
    # ポケモンのどうぐ（例: 岩のむねあて＝闘ポケモンが受けるダメージ -30）
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
            if atk.name == "しっぺがえし" and state.active_player_state().knockouts_suffered == 2:
                dmg += 90
            max_dmg = max(max_dmg, dmg)
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
            if atk.name == "しっぺがえし" and opp.knockouts_suffered == 2:
                dmg += 90
            max_dmg = max(max_dmg, dmg)
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
    old_active.attached_energy -= cost
    # 付与タイプリストからも末尾を削除（にげるコストは個数のみなのでタイプは問わない）
    types = getattr(old_active, "attached_energy_types", [])
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
    _clear_status(old_active)  # ベンチに下がると状態異常は全て回復
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
    """きずぐすりを使用して自分のバトル場のポケモンを 20 回復。"""
    p = state.active_player_state()
    if not p.active or hand_index < 0 or hand_index >= len(p.hand):
        return False
    card = p.hand[hand_index]
    if not is_goods(card) or getattr(card, "effect", None) != "heal":
        return False
    amount = getattr(card, "heal_amount", 20)
    before = p.active.hp
    p.active.hp = min(p.active.hp + amount, p.active.max_hp)
    p.hand.pop(hand_index)
    state.log(f"{state.player_name(state.current_player)}: きずぐすりを使用 → バトル場のポケモンを {before} → {p.active.hp} に回復")
    return True


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
    p.hand.pop(hand_index)
    target.attached_tool = card
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
    if not is_goods(card) or getattr(card, "effect", None) != "swap_active":
        return False
    old_active = p.active
    p.active = p.bench[bench_index]
    p.bench[bench_index] = old_active
    _clear_status(old_active)
    p.hand.pop(hand_index)
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
    if effect == "draw_3":
        n = getattr(card, "draw_count", 3)
        drawn = p.draw(n)
        p.hand.extend(drawn)
        p.hand.pop(hand_index)
        state.support_used_this_turn = True
        drawn_names = ", ".join(_card_label(c) for c in drawn)
        state.log(
            f"{state.player_name(state.current_player)}: ネモを使用 → デッキから {len(drawn)} 枚ドロー → [{drawn_names}]（手札 {len(p.hand)} 枚、デッキ {len(p.deck)} 枚）"
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
    """場のポケモン（field_card）が進化カード（evolution_card）の進化元か。id または name（name_ja）で一致させる。"""
    base = (evolution_card.evolves_from or "").strip()
    if not base:
        return False
    return (
        getattr(field_card, "id", None) == base
        or (getattr(field_card, "name", "") or "").strip() == base
    )


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
    types = getattr(p.active, "attached_energy_types", [])
    if not _can_pay_energy_cost(
        p.active.attached_energy, types,
        atk.energy_cost, getattr(atk, "energy_cost_typed", None),
    ):
        return False
    # こんらん：ワザを使う前にコイン。裏 → 自分に30ダメージ、攻撃失敗
    if getattr(p.active, "special_state", None) == "confusion":
        if not _flip_coin():
            self_before = p.active.hp
            p.active.hp -= 30
            state.log(f"{state.player_name(state.current_player)}: こんらんでコインが裏 → 自分に 30 ダメージ（HP {self_before} → {max(0, p.active.hp)}）、攻撃失敗")
            if p.active and p.active.hp <= 0:
                state.log(f"{state.player_name(state.current_player)} のバトル場のポケモンがこんらんの自傷できぜつ！")
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
    # 弱点：守備側の弱点タイプと攻撃側ポケモンのタイプが一致するとダメージ 2 倍
    defender_card = opp.active.card
    attacker_card = pokemon_card
    if (
        getattr(defender_card, "weakness", None)
        and getattr(attacker_card, "pokemon_type", None)
        and defender_card.weakness == attacker_card.pokemon_type
    ):
        damage *= 2
        state.log(f"{state.player_name(state.current_player)}: 弱点一致！ダメージが 2 倍（{damage // 2} → {damage}）")
    # 抵抗力：守備側の抵抗力タイプと攻撃側ポケモンのタイプが一致するとダメージ -30（0 未満なら 0）
    if (
        getattr(defender_card, "resistance", None)
        and getattr(attacker_card, "pokemon_type", None)
        and defender_card.resistance == attacker_card.pokemon_type
    ):
        before_r = damage
        damage = max(0, damage - 30)
        state.log(f"{state.player_name(state.current_player)}: 抵抗力一致！ダメージが -30（{before_r} → {damage}）")
    # ポケモンのどうぐ（つけているポケモンが受けるワザのダメージを軽減）
    tool = getattr(opp.active, "attached_tool", None)
    if tool and getattr(tool, "is_tool", False) and getattr(tool, "tool_damage_reduce", 0) > 0:
        cond = getattr(tool, "tool_condition_type", None)
        if cond is None or getattr(defender_card, "pokemon_type", None) == cond:
            before_t = damage
            damage = max(0, damage - getattr(tool, "tool_damage_reduce", 0))
            state.log(f"{state.player_name(state.current_player)}: {opp.active.card.name} の {tool.name} でダメージ -{before_t - damage}（{before_t} → {damage}）")
    if atk.name == "しっぺがえし" and opp.knockouts_suffered == 2:
        damage += 90
        state.shippegaeshi_120_used = True
        state.log(f"{state.player_name(state.current_player)}: 「{atk.name}」で相手に {damage} ダメージ（相手あと1きぜつで負けのため 90 ダメージ追加、相手 HP {opp_before} → {max(0, opp_before - damage)}）")
    else:
        state.log(f"{state.player_name(state.current_player)}: 「{atk.name}」で相手に {damage} ダメージ（相手 HP {opp_before} → {max(0, opp_before - damage)}）")
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
                if bench_target_self:
                    state.log(f"{state.player_name(state.current_player)} のベンチの {bp_name} がきぜつ！")
                else:
                    state.log(f"{state.player_name(state.opponent())} のベンチの {bp_name} がきぜつ！（{opp.knockouts_suffered + 1} 回目）")
                    if _handle_opponent_ko(opp, state, koed_bench_bp):
                        return True

    self_dmg = getattr(atk, "self_damage", 0)
    if self_dmg > 0 and p.active:
        self_before = p.active.hp
        p.active.hp -= self_dmg
        state.log(f"{state.player_name(state.current_player)}: 反動で自分に {self_dmg} ダメージ（自分 HP {self_before} → {max(0, p.active.hp)}）")

    # 技の状態異常を付与（相手または自分。ねむり・マヒ・こんらんは上書き、どく・やけどは重複可）
    # status_effect_on_coin_heads が True のときはコイン表のときだけ付与、False なら確定で付与
    status_effect = getattr(atk, "status_effect", None)
    if status_effect:
        target_bp = p.active if getattr(atk, "status_effect_target", "opponent") == "self" else (opp.active if opp.active else None)
        if target_bp:
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

    if opp.active and opp.active.hp <= 0:
        koed_active = opp.active
        state.log(f"{state.player_name(state.opponent())} のバトル場の {koed_active.card.name} がきぜつ！（{opp.knockouts_suffered + 1} 回目）")
        opp.active = None
        if _handle_opponent_ko(opp, state, koed_active):
            return True

    if p.active and p.active.hp <= 0:
        state.log(f"{state.player_name(state.current_player)} のバトル場のポケモンが反動できぜつ！")
        p.active = None
        _promote_from_bench(p, state, state.current_player)
    return True


def _choose_best_attack_index(state: GameState, p: PlayerState, opp: PlayerState) -> int | None:
    """出せる技のうちダメージが最大のインデックスを返す。出せなければ None。"""
    if not p.active or not p.active.card.attacks:
        return None
    best_idx = None
    best_dmg = -1
    types = getattr(p.active, "attached_energy_types", [])
    for idx, atk in enumerate(p.active.card.attacks):
        if not _can_pay_energy_cost(
            p.active.attached_energy, types,
            atk.energy_cost, getattr(atk, "energy_cost_typed", None),
        ):
            continue
        dmg = _attack_damage_for_eval(atk)
        if atk.name == "しっぺがえし" and opp.knockouts_suffered == 2:
            dmg += 90
        if dmg > best_dmg:
            best_dmg = dmg
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
        # 進化後のいずれかの技を出せるなら付与不要
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


def _try_attach_energy_auto(state: GameState) -> bool:
    """自動ターン用：エネルギーを 1 枚付与するなら True を返す。"""
    p = state.active_player_state()
    # 最初の番（先行 1 ターン目）のみ進化できない（ルール A-05）
    can_evolve_this_turn = not _is_first_player_first_turn(state)
    energy_hand_idx = next((i for i, c in enumerate(p.hand) if is_energy(c)), None)
    if energy_hand_idx is None or not p.active:
        return False

    if p.active.attached_energy == 0:
        attach_energy(state, energy_hand_idx)
        return True
    if can_evolve_this_turn and _should_attach_for_evolution(p):
        attach_energy(state, energy_hand_idx)
        return True

    # バトル場に 1 つ付与すれば、より強い技で相手をきぜつできるならバトル場に付与する
    opp = state.defending_player_state()
    if opp.active:
        energy_card = p.hand[energy_hand_idx]
        new_type = getattr(energy_card, "energy_type", None) or "colorless"
        sim_count = p.active.attached_energy + 1
        sim_types = getattr(p.active, "attached_energy_types", []) + [new_type]
        for atk in p.active.card.attacks:
            if not _can_pay_energy_cost(
                sim_count, sim_types,
                atk.energy_cost, getattr(atk, "energy_cost_typed", None),
            ):
                continue
            base_dmg = _attack_damage_for_eval(atk)
            if atk.name == "しっぺがえし" and opp.knockouts_suffered == 2:
                base_dmg += 90
            effective = _effective_damage_to_defender(
                p.active.card, opp.active, base_dmg
            )
            if effective >= opp.active.hp:
                attach_energy(state, energy_hand_idx)
                return True

    energy_needed_active = _energy_needed_for_active(p)
    if _can_active_use_any_attack(p) and p.bench:
        bench_candidates = [
            (bi, b.attached_energy, b.card.attacks)
            for bi, b in enumerate(p.bench)
        ]
        max_need = 2
        need_more = [
            (bi, max_need - en, max((a.energy_cost for a in atks), default=0))
            for bi, en, atks in bench_candidates
        ]
        best_bench = max(need_more, key=lambda x: (x[1] > 0, x[2]), default=None)
        if best_bench and best_bench[1] > 0:
            attach_energy(state, energy_hand_idx, bench_index=best_bench[0])
            return True
    if not _can_active_use_any_attack(p) or not p.bench:
        attach_energy(state, energy_hand_idx)
        return True
    if p.bench:
        attach_energy(state, energy_hand_idx, bench_index=0)
        return True
    return False


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
    # どく：ターン終了時に N ダメージ
    poison_dmg = getattr(p.active, "poison_damage", 0)
    if poison_dmg > 0:
        before = p.active.hp
        p.active.hp -= poison_dmg
        state.log(f"{state.player_name(player_index)}: どくで {p.active.card.name} に {poison_dmg} ダメージ（HP {before} → {max(0, p.active.hp)}）")
    if p.active and p.active.hp <= 0:
        koed_poison = p.active
        state.log(f"{state.player_name(player_index)} のバトル場の {koed_poison.card.name} がきぜつ！（どくのダメージ）")
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
    # やけど：ターン終了時に 20 ダメージ、その後コインで表なら回復
    if getattr(p.active, "burn", False):
        before = p.active.hp
        p.active.hp -= 20
        state.log(f"{state.player_name(player_index)}: やけどで {p.active.card.name} に 20 ダメージ（HP {before} → {max(0, p.active.hp)}）")
        if _flip_coin():
            p.active.burn = False
            state.log(f"{state.player_name(player_index)}: {p.active.card.name} のやけどが治った（コイン：表）")
        else:
            state.log(f"{state.player_name(player_index)}: {p.active.card.name} のやけどは継続（コイン：裏）")
    # マヒ：自分の番の終わりに解除（今ターン終了するプレイヤーのみ）
    if player_index == state.current_player and getattr(p.active, "special_state", None) == "paralysis":
        p.active.special_state = None
        state.log(f"{state.player_name(player_index)}: {p.active.card.name} のマヒが解けた")
    if p.active and p.active.hp <= 0:
        koed_status = p.active
        state.log(f"{state.player_name(player_index)} のバトル場の {koed_status.card.name} がきぜつ！（状態異常のダメージ）")
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
    # 特殊状態の確認（どく・やけど・ねむり・マヒの順、おたがい同時）
    _apply_poison_burn_paralysis_for_active(state, state.current_player)
    if not _check_game_end(state):
        _apply_poison_burn_paralysis_for_active(state, state.opponent())
        _check_game_end(state)
    # 進化・ベンチ出しフラグをリセット
    for pl in state.players:
        if pl.active:
            pl.active.evolved_this_turn = False
            pl.active.put_on_bench_this_turn = False
        for bp in pl.bench:
            bp.evolved_this_turn = False
            bp.put_on_bench_this_turn = False
    state.log(f"{state.player_name(state.current_player)} のターン終了")
    state.current_player = state.opponent()
    state.turn_count += 1
    state.log("")


def run_turn_auto(state: GameState) -> bool:
    """
    現在のプレイヤーが「可能な行動を順番に実行」する。
    順序：0. ベンチにポケモン出す、1. にげる（条件満たすとき）、2. サポート、3. きずぐすり、4. ポケモンいれかえ、5. エネルギー付与、6. 進化、7. 攻撃。
    何もできなければ False を返す。True = ターン内で何かした。
    """
    p = state.active_player_state()
    opp = state.defending_player_state()
    if not p.active:
        return False
    
    acted = False

    while _put_one_pokemon_on_bench(p, state, state.current_player):
        acted = True

    opp_max_dmg = _opponent_max_damage(state)
    our_max_dmg = _our_max_damage(state)
    would_be_koed = p.active and opp_max_dmg > 0 and p.active.hp <= opp_max_dmg
    can_ko_opponent = opp.active and our_max_dmg >= opp.active.hp
    retreat_cost = getattr(p.active.card, "retreat_cost", 1) if p.active else 0
    can_retreat = p.active and getattr(p.active, "special_state", None) not in ("sleep", "paralysis")
    if can_retreat and p.active and p.bench and would_be_koed and not can_ko_opponent and p.active.attached_energy >= retreat_cost + 100:
        best = max(
            range(len(p.bench)),
            key=lambda i: (p.bench[i].hp, p.bench[i].attached_energy),
            default=None,
        )
        if best is not None and retreat(state, best):
            acted = True
            p = state.active_player_state()

    # サポート（1 ターン 1 枚、先行 1 ターン目は不可）: ネモなど
    if not _is_first_player_first_turn(state) and not state.support_used_this_turn:
        for i, c in enumerate(p.hand):
            if is_support(c) and getattr(c, "effect", None) == "draw_3":
                if use_support(state, i):
                    acted = True
                    p = state.active_player_state()
                break

    if p.active and p.active.hp < p.active.max_hp:
        for i, c in enumerate(p.hand):
            if is_goods(c) and getattr(c, "effect", None) == "heal":
                use_potion(state, i)
                acted = True
                break

    # ポケモンいれかえ（バトル場とベンチを入れ替え）
    # タイミング: ベンチ出し・にげる・サポート・きずぐすりの後、エネルギー付与の前。
    # 選出: ベンチのうち残り HP が最大の 1 体と入れ替える（いれかえ後バトル場に出る＝攻撃の的になるため HP の高い方を出す）。
    if p.active and p.bench:
        for i, c in enumerate(p.hand):
            if is_goods(c) and getattr(c, "effect", None) == "swap_active":
                best_bench = max(range(len(p.bench)), key=lambda b: p.bench[b].hp, default=None)
                if best_bench is not None and use_pokemon_swap(state, i, best_bench):
                    acted = True
                    p = state.active_player_state()
                break

    # ポケモンのどうぐ（1 匹に 1 枚まで。条件を満たすポケモンにつける）
    for i, c in enumerate(p.hand):
        if not is_goods(c) or not getattr(c, "is_tool", False):
            continue
        cond = getattr(c, "tool_condition_type", None)
        if p.active and getattr(p.active, "attached_tool", None) is None:
            if cond is None or getattr(p.active.card, "pokemon_type", None) == cond:
                if attach_tool(state, i, bench_index=None):
                    acted = True
                    p = state.active_player_state()
                    break
        for bi, bp in enumerate(p.bench):
            if getattr(bp, "attached_tool", None) is not None:
                continue
            if cond is not None and getattr(bp.card, "pokemon_type", None) != cond:
                continue
            if attach_tool(state, i, bench_index=bi):
                acted = True
                p = state.active_player_state()
                break
        else:
            continue
        break

    if _try_attach_energy_auto(state):
        acted = True
        p = state.active_player_state()

    # 最初の番（先行 1 ターン目）のみ進化できない（ルール A-05）
    can_evolve = not _is_first_player_first_turn(state)
    if can_evolve:
        for hand_idx, c in enumerate(p.hand):
            if not is_pokemon(c) or not c.evolves_from:
                continue
            if p.active and _can_evolve_onto(p.active.card, c):
                evolve_pokemon(state, hand_idx, bench_index=None)
                acted = True
                break
            for bench_idx, bench_poke in enumerate(p.bench):
                if _can_evolve_onto(bench_poke.card, c):
                    evolve_pokemon(state, hand_idx, bench_index=bench_idx)
                    acted = True
                    break
            if acted:
                break

    is_game_first_turn = state.turn_count == 0
    can_attack = not is_game_first_turn and p.active and getattr(p.active, "special_state", None) not in ("sleep", "paralysis")
    if can_attack:
        best_idx = _choose_best_attack_index(state, p, opp)
        if best_idx is not None:
            attack(state, best_idx)
            acted = True
    
    if not acted:
        state.log(f"{state.player_name(state.current_player)}: 実行できるアクションなし（パス）")
    
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
        # デッキ切れ負けなし時用：ターン数で打ち切り（サイド取得数で勝者を決める）
        if state.turn_count >= MAX_TURNS_SAFETY:
            p0, p1 = state.players[0], state.players[1]
            taken0 = PRIZE_COUNT - len(p0.prize_pile)
            taken1 = PRIZE_COUNT - len(p1.prize_pile)
            state.winner = 0 if taken0 >= taken1 else 1
            if state.log_fn:
                state.log(f"{MAX_TURNS_SAFETY} ターンで打ち切り（サイド取得で判定）\n")
                state.log("========== ゲーム終了 ==========\n")
            return state.winner
