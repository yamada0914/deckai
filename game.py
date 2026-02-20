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

from card import PokemonCard, is_energy, is_item, is_pokemon, is_support
from deck import STARTING_HAND_SIZE, create_deck, format_deck_recipe

BENCH_SIZE = 5
WIN_KO_COUNT = 3
MAX_TURNS = 200

# 状態異常：ねむり・マヒ・こんらんはどれか1つのみ。どく・やけどは重複可。ベンチに下がると全て解除。
SpecialState = Literal["sleep", "paralysis", "confusion"]


@dataclass
class BattlePokemon:
    """バトル場／ベンチに出すポケモン（HP・エネルギー・状態異常を管理）"""
    card: PokemonCard
    attached_energy: int = 0
    # 付いているエネルギーをタイプごとに（1 つずつ）。長さは attached_energy と一致。無色＝任意として消費可能。
    attached_energy_types: list = field(default_factory=list)
    # 状態異常：ねむり・マヒ・こんらんはどれか1つのみ
    special_state: SpecialState | None = None
    poison_damage: int = 0  # 0 = どくにかかっていない。10/20/30 で毎ターン末尾にダメージ
    burn: bool = False
    # このターンに進化したか（そのターンに 1 進化したポケモンは同じターンに 2 進化できない）
    evolved_this_turn: bool = False

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
) -> GameState:
    """デッキを組んで初手 4 枚をセット。deck0 / deck1 で使用デッキを指定（0=A, 1=B, 2=C）。"""
    if seed is not None:
        random.seed(seed)
    state = GameState(log_fn=log_fn, deck_indices=(deck0, deck1))
    state.log("========== ゲーム開始 ==========")
    for i in range(2):
        state.log(f"{state.player_name(i)} デッキ: [{format_deck_recipe(state.deck_indices[i])}]")
    for i in range(2):
        state.players[i].deck = create_deck(state.deck_indices[i])
        random.shuffle(state.players[i].deck)
        state.log(f"{state.player_name(i)}: デッキ {len(state.players[i].deck)} 枚をシャッフル")
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
        msg = f"再シャッフル後、初手 4 枚ドロー → 手札 [{', '.join(hand_names)}]" if retry_count > 0 else f"初手 4 枚ドロー → 手札 [{', '.join(hand_names)}]"
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
            player.bench.append(BattlePokemon(card=p))
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


def _handle_opponent_ko(opp: PlayerState, state: GameState) -> bool:
    """相手のきぜつを 1 回加算し、勝ちなら state.winner をセットして True、否則ベンチから繰り出して False。"""
    opp.knockouts_suffered += 1
    if opp.knockouts_suffered >= WIN_KO_COUNT:
        state.winner = state.current_player
        return True
    _promote_from_bench(opp, state, state.opponent())
    return False


def _check_game_end(state: GameState) -> bool:
    """勝敗が決まっていれば state.winner をセットして True を返す。3 回きぜつで負け、またはバトル場・ベンチともにいないと負け。"""
    for i in range(2):
        p = state.players[i]
        if p.knockouts_suffered >= WIN_KO_COUNT:
            state.winner = 1 - i
            state.log(f"{state.player_name(i)} のポケモンが {WIN_KO_COUNT} 回きぜつ → {state.player_name(state.winner)} の勝ち")
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
        # state.log(f"{state.player_name(state.current_player)}: デッキが空のためドローなし（手札 {len(p.hand)} 枚）")
        state.log(f"{state.player_name(state.current_player)}: デッキが空のためドローできず → 負け")
        state.winner = state.opponent()
        return
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
    if not is_item(card) or getattr(card, "effect", None) != "heal":
        return False
    amount = getattr(card, "heal_amount", 20)
    before = p.active.hp
    p.active.hp = min(p.active.hp + amount, p.active.max_hp)
    p.hand.pop(hand_index)
    state.log(f"{state.player_name(state.current_player)}: きずぐすりを使用 → バトル場のポケモンを {before} → {p.active.hp} に回復")
    return True


def use_pokemon_swap(state: GameState, hand_index: int, bench_index: int) -> bool:
    """ポケモンいれかえを使用して自分のバトル場のポケモンとベンチの指定 1 体を入れ替える。"""
    p = state.active_player_state()
    if not p.active or bench_index < 0 or bench_index >= len(p.bench):
        return False
    if hand_index < 0 or hand_index >= len(p.hand):
        return False
    card = p.hand[hand_index]
    if not is_item(card) or getattr(card, "effect", None) != "swap_active":
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
    state.log(f"{log_prefix}{old_name} を {evolved.name} に進化（HP {target.hp}/{target.max_hp}）")


def evolve_pokemon(state: GameState, hand_index: int, bench_index: int | None = None) -> bool:
    """
    手札の進化ポケモンで、バトル場またはベンチのポケモンを進化させる。
    bench_index=None ならバトル場、数値ならベンチのそのインデックス。
    """
    p = state.active_player_state()
    if hand_index < 0 or hand_index >= len(p.hand):
        return False
    evolution_card = p.hand[hand_index]
    if not is_pokemon(evolution_card) or not evolution_card.evolves_from:
        return False

    if bench_index is None:
        if not p.active or p.active.card.id != evolution_card.evolves_from:
            return False
        # 2 進化：そのターンに 1 進化していたら同じターンには 2 進化できない
        if getattr(p.active.card, "evolves_from", None) is not None and getattr(p.active, "evolved_this_turn", False):
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
    if bench_pokemon.card.id != evolution_card.evolves_from:
        return False
    # 2 進化：そのターンに 1 進化していたら同じターンには 2 進化できない
    if getattr(bench_pokemon.card, "evolves_from", None) is not None and getattr(bench_pokemon, "evolved_this_turn", False):
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
    if atk.name == "しっぺがえし" and opp.knockouts_suffered == 2:
        damage += 90
        state.shippegaeshi_120_used = True
        state.log(f"{state.player_name(state.current_player)}: 「{atk.name}」で相手に {damage} ダメージ（相手あと1きぜつで負けのため 90 ダメージ追加、相手 HP {opp_before} → {max(0, opp_before - damage)}）")
    else:
        state.log(f"{state.player_name(state.current_player)}: 「{atk.name}」で相手に {damage} ダメージ（相手 HP {opp_before} → {max(0, opp_before - damage)}）")
    opp.active.hp -= damage
    bench_dmg = getattr(atk, "bench_damage", 0)
    if bench_dmg > 0 and opp.bench:
        bench_target = opp.bench[0]
        bench_before = bench_target.hp
        bench_target.hp -= bench_dmg
        state.log(f"{state.player_name(state.current_player)}: 「{atk.name}」で相手のベンチの {bench_target.card.name} に {bench_dmg} ダメージ（HP {bench_before} → {max(0, bench_target.hp)}）")
        if bench_target.hp <= 0:
            state.log(f"{state.player_name(state.opponent())} のベンチの {bench_target.card.name} がきぜつ！（{opp.knockouts_suffered + 1} 回目）")
            opp.bench.pop(0)
            if _handle_opponent_ko(opp, state):
                return True

    self_dmg = getattr(atk, "self_damage", 0)
    if self_dmg > 0 and p.active:
        self_before = p.active.hp
        p.active.hp -= self_dmg
        state.log(f"{state.player_name(state.current_player)}: 反動で自分に {self_dmg} ダメージ（自分 HP {self_before} → {max(0, p.active.hp)}）")

    # 技の状態異常を付与（相手または自分。ねむり・マヒ・こんらんは上書き、どく・やけどは重複可）
    status_effect = getattr(atk, "status_effect", None)
    if status_effect:
        target_bp = p.active if getattr(atk, "status_effect_target", "opponent") == "self" else (opp.active if opp.active else None)
        if target_bp:
            _apply_status(
                state, target_bp, status_effect,
                poison_damage=getattr(atk, "poison_damage_if_poison", 10),
                log_prefix=f"{state.player_name(state.current_player)}: 「{atk.name}」→ ",
            )

    if opp.active and opp.active.hp <= 0:
        state.log(f"{state.player_name(state.opponent())} のバトル場のポケモンがきぜつ！（{opp.knockouts_suffered + 1} 回目）")
        opp.active = None
        if _handle_opponent_ko(opp, state):
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
        if not (is_pokemon(c) and getattr(c, "evolves_from", None) == p.active.card.id):
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
        if not (is_pokemon(c) and getattr(c, "evolves_from", None) == p.active.card.id):
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
    can_evolve_this_turn = state.turn_count > 1
    energy_hand_idx = next((i for i, c in enumerate(p.hand) if is_energy(c)), None)
    if energy_hand_idx is None or not p.active:
        return False

    if p.active.attached_energy == 0:
        attach_energy(state, energy_hand_idx)
        return True
    if can_evolve_this_turn and _should_attach_for_evolution(p):
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


def end_turn(state: GameState) -> None:
    """ターン終了：どく・やけどダメージ、やけど回復コイン、マヒ解除。その後相手に交代、ターン数増加。"""
    p = state.active_player_state()
    if p.active:
        # どく：ターン終了時に N ダメージ
        poison_dmg = getattr(p.active, "poison_damage", 0)
        if poison_dmg > 0:
            before = p.active.hp
            p.active.hp -= poison_dmg
            state.log(f"{state.player_name(state.current_player)}: どくで {p.active.card.name} に {poison_dmg} ダメージ（HP {before} → {max(0, p.active.hp)}）")
        if p.active and p.active.hp <= 0:
            state.log(f"{state.player_name(state.current_player)} のバトル場の {p.active.card.name} がきぜつ！（どくのダメージ）")
            p.active = None
            _promote_from_bench(p, state, state.current_player)
        if p.active:
            # やけど：ターン終了時に 20 ダメージ、その後コインで表なら回復
            if getattr(p.active, "burn", False):
                before = p.active.hp
                p.active.hp -= 20
                state.log(f"{state.player_name(state.current_player)}: やけどで {p.active.card.name} に 20 ダメージ（HP {before} → {max(0, p.active.hp)}）")
                if _flip_coin():
                    p.active.burn = False
                    state.log(f"{state.player_name(state.current_player)}: {p.active.card.name} のやけどが治った（コイン：表）")
                else:
                    state.log(f"{state.player_name(state.current_player)}: {p.active.card.name} のやけどは継続（コイン：裏）")
            # マヒ：自分の番の終わりに解除
            if getattr(p.active, "special_state", None) == "paralysis":
                p.active.special_state = None
                state.log(f"{state.player_name(state.current_player)}: {p.active.card.name} のマヒが解けた")
            if p.active and p.active.hp <= 0:
                state.log(f"{state.player_name(state.current_player)} のバトル場の {p.active.card.name} がきぜつ！（状態異常のダメージ）")
                p.active = None
                _promote_from_bench(p, state, state.current_player)
    # 進化フラグをリセット（次のターンからは「そのターンに 1 進化した」扱いにならない）
    for pl in state.players:
        if pl.active:
            pl.active.evolved_this_turn = False
        for bp in pl.bench:
            bp.evolved_this_turn = False
    state.log(f"{state.player_name(state.current_player)} のターン終了")
    state.current_player = state.opponent()
    state.turn_count += 1
    state.log("")


def run_turn_auto(state: GameState) -> bool:
    """
    現在のプレイヤーが「可能な行動を順番に実行」する。
    順序：0. 手札のポケモンをベンチに出す（5 体まで）、1. きずぐすり、2. エネルギー付与、3. 進化、4. 攻撃。
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
            if is_item(c) and getattr(c, "effect", None) == "heal":
                use_potion(state, i)
                acted = True
                break

    # ポケモンいれかえ（バトル場とベンチを入れ替え）
    if p.active and p.bench:
        for i, c in enumerate(p.hand):
            if is_item(c) and getattr(c, "effect", None) == "swap_active":
                best_bench = max(range(len(p.bench)), key=lambda b: p.bench[b].hp, default=None)
                if best_bench is not None and use_pokemon_swap(state, i, best_bench):
                    acted = True
                    p = state.active_player_state()
                break

    if _try_attach_energy_auto(state):
        acted = True
        p = state.active_player_state()

    is_first_turn_for_either = state.turn_count <= 1
    can_evolve = not is_first_turn_for_either
    if can_evolve:
        for hand_idx, c in enumerate(p.hand):
            if not is_pokemon(c) or not c.evolves_from:
                continue
            base_id = c.evolves_from
            if p.active and p.active.card.id == base_id:
                evolve_pokemon(state, hand_idx, bench_index=None)
                acted = True
                break
            for bench_idx, bench_poke in enumerate(p.bench):
                if bench_poke.card.id == base_id:
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


def run_game_auto(state: GameState, max_turns: int | None = None) -> int | None:
    """
    自動でターンを進め、勝者が決まるまで実行。
    デッキ切れ・3 きぜつ等で必ず決着。戻り値: 0 or 1 = 勝者。
    """
    if max_turns is None:
        max_turns = MAX_TURNS
    if _check_game_end(state):
        if state.log_fn:
            state.log("========== ゲーム終了 ==========\n")
        return state.winner
    while state.turn_count < max_turns:
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
    return None
