"""
ゲーム状態・定数・セットアップ・共通ヘルパー。

ポケカ風ルール: 初手 7 枚、ベンチ最大 5 体、相手を 3 回きぜつで勝ち。
"""
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Literal

from card import PokemonCard, get_card_by_id, is_basic_pokemon, is_energy, is_goods, is_pokemon, is_stage2_pokemon, is_support
from deck import STARTING_HAND_SIZE, create_deck, create_deck_from_deck_code, get_deck_name

BENCH_SIZE = 5
WIN_KO_COUNT = 3
PRIZE_COUNT = 6
MAX_TURNS_SAFETY = 200
MAX_EVOLVE_ROUNDS_PER_TURN = 20
MAX_BALL_USES_PER_TURN = 30
MAX_TURN_ACTION_ROUNDS = 10

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
    """状態異常を全て解除（ベンチに下がったときなど）。かそくづき等の「次の自分の番で使えない」も解除。"""
    bp.special_state = None
    bp.poison_damage = 0
    bp.burn = False
    bp.protected_next_opponent_turn = False
    bp.damage_reduction_next_turn = 0
    bp.disabled_attack_name = None


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
    energy_attached_this_turn: bool = False
    this_turn_attack_name: str | None = None
    this_turn_attack_actor_id: str | None = None
    last_turn_attack_name: list = field(default_factory=lambda: [None, None])
    last_turn_attack_actor_id: list = field(default_factory=lambda: [None, None])
    our_ko_by_damage_last_turn: list = field(default_factory=lambda: [False, False])
    drawn_this_turn: list = field(default_factory=list)
    record_frame_fn: Callable[["GameState"], None] | None = field(default=None, repr=False)
    turn_when_disabled_attack: list = field(default_factory=lambda: [None, None])

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


def _is_first_player_first_turn(state: GameState) -> bool:
    """先行の 1 ターン目か（サポートは原則使用不可）。"""
    return state.turn_count == 0 and state.current_player == state.first_player


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
            if getattr(c, "name", "") in existing_names:
                continue
            p = c.copy()
            bp = BattlePokemon(card=p)
            bp.put_on_bench_this_turn = True
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
    """
    バトル場が空のとき、ベンチから 1 体をバトル場に出す。
    次の自分のターンに相手のバトル場をきぜつさせられるポケモンがいればそれを優先し、いなければ HP が一番高いポケモンを選ぶ。
    """
    if player.active is not None or not player.bench:
        return False
    opp = state.players[1 - player_index]
    best_idx = None
    if opp.active and opp.active.hp > 0:
        from .damage import _max_effective_damage_for_attacker
        can_ko_indices = [
            i for i in range(len(player.bench))
            if _max_effective_damage_for_attacker(state, player.bench[i], opp.active, player_index) >= opp.active.hp
        ]
        if can_ko_indices:
            best_idx = max(can_ko_indices, key=lambda i: player.bench[i].hp)
    if best_idx is None:
        best_idx = max(range(len(player.bench)), key=lambda i: player.bench[i].hp)
    player.active = player.bench.pop(best_idx)
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
    state.log(
        f"{state.player_name(taker_index)}: サイドを 1 枚とる → {_card_label(card)}（手札 {len(p.hand)} 枚、サイド残り {len(p.prize_pile)} 枚）"
    )
    if len(p.prize_pile) == 0:
        state.winner = taker_index
        state.log(f"{state.player_name(taker_index)}: サイドをすべてとり終えた → 勝ち！")
        return True
    return False


def _log_prize_count_for_ko(state: GameState, taker_index: int, koed_bp: BattlePokemon, prize_count: int) -> None:
    """サイドを複数枚とるときのログ（メガ/ex）を出す。"""
    if prize_count > 1:
        state.log(
            f"{state.player_name(taker_index)}: {koed_bp.card.name} は "
            f"{'メガ' if prize_count == 3 else 'ex'} のため、サイドを {prize_count} 枚とる"
        )


def _handle_own_active_ko(
    state: GameState, player_index: int, koed_bp: BattlePokemon, reason: str
) -> bool:
    """
    自分のバトル場ポケモンがきぜつしたときの処理。
    トラッシュ送り・相手のサイド取得・ベンチ繰り出し。戻り値: 勝敗がついたら True。
    """
    p = state.players[player_index]
    state.log(f"{state.player_name(player_index)} のバトル場の {koed_bp.card.name} がきぜつ！（{reason}）")
    p.discard.append(koed_bp.card)
    tool = getattr(koed_bp, "attached_tool", None)
    if tool:
        p.discard.append(tool)
    p.knockouts_suffered += 1
    p.active = None
    prize_count = _prizes_for_ko(koed_bp)
    taker = 1 - player_index
    _log_prize_count_for_ko(state, taker, koed_bp, prize_count)
    for _ in range(prize_count):
        if _take_prize(state, taker):
            return True
    if state.winner is None:
        _promote_from_bench(p, state, player_index)
    return False


def _handle_opponent_ko(opp: PlayerState, state: GameState, koed_bp: BattlePokemon) -> bool:
    """相手のきぜつを 1 回加算し、攻撃側がサイドをとる（ex なら 2 枚、それ以外は 1 枚）。サイド 0 なら勝ち、否則ベンチから繰り出す。きぜつしたポケモンはすぐにトラッシュに送る。"""
    opp.discard.append(koed_bp.card)
    tool = getattr(koed_bp, "attached_tool", None)
    if tool:
        opp.discard.append(tool)
    if opp.active is koed_bp:
        opp.active = None
    opp.knockouts_suffered += 1
    state._record_frame()
    prize_count = _prizes_for_ko(koed_bp)
    _log_prize_count_for_ko(state, state.current_player, koed_bp, prize_count)
    for _ in range(prize_count):
        if _take_prize(state, state.current_player):
            return True
    state._record_frame()
    _promote_from_bench(opp, state, state.opponent())
    state._record_frame()
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
    """ターン開始：ドロー 1 枚、サポート・エネルギー付与未使用にリセット。ねむりならコインで解除判定。"""
    state.support_used_this_turn = False
    state.energy_attached_this_turn = False
    state.our_ko_by_damage_last_turn[state.current_player] = False
    state.drawn_this_turn = []
    p = state.active_player_state()
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
