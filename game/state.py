"""
ゲーム状態・定数・セットアップ・共通ヘルパー。

ポケカ風ルール: 初手 7 枚、ベンチ最大 5 体、相手を 3 回きぜつで勝ち。
"""
import copy
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Literal

from card import PokemonCard, get_card_by_id, is_basic_pokemon, is_energy, is_goods, is_pokemon, is_stadium, is_stage2_pokemon, is_support
from deck import STARTING_HAND_SIZE, create_deck, create_deck_from_deck_code, get_deck_name
from .deck_strategies import get_allow_duplicate_bench_ids, get_priority_setup_pokemon_ids
from .weights import GameWeights, get_promote_weight

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
    retreat_locked: bool = False

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
            drawn.append(self.deck.pop(0))
        return drawn

    def has_active_pokemon(self) -> bool:
        return self.active is not None and self.active.hp > 0


def _card_label(card) -> str:
    return getattr(card, "name", card.__class__.__name__)


def get_effective_max_hp(state: "GameState", card: PokemonCard) -> int:
    """
    スタジアム等を反映した実効最大 HP。例：グラビティーマウンテン場のとき 2 進化は -30。
    """
    base = getattr(card, "max_hp", 0) or 0
    stadium = getattr(state, "stadium", None)
    if stadium is not None and is_stadium(stadium):
        sid = (getattr(stadium, "id", "") or "").strip()
        sname = (getattr(stadium, "name", "") or "").strip()
        if sid == "gurabiteiimaunten" or "グラビティーマウンテン" in sname:
            if getattr(card, "evolution_stage", None) == "stage2" or is_stage2_pokemon(card):
                return max(0, base - 30)
    return base


def card_from_discard_to_hand(card) -> object:
    """
    トラッシュから手札に加える直前に使う。
    ポケモンはトラッシュ上でも場で受けた hp が残るため、 copy して HP を最大に戻す。
    エネルギー・トレーナー等はそのまま返す。
    """
    if is_pokemon(card) and hasattr(card, "copy"):
        to_hand = card.copy()
        cap = getattr(to_hand, "max_hp", None) or getattr(card, "max_hp", None) or getattr(to_hand, "hp", 0)
        to_hand.hp = cap
        return to_hand
    return card


def _basic_energy_id(energy_type: str) -> str:
    """エネルギータイプ名から基本エネルギーカード ID を返す。"""
    _TYPE_TO_ID = {
        "lightning": "basic-energy-lightning",
        "fighting": "basic-energy-fighting",
        "psychic": "kihonchixyouenerugi",
        "fire": "kihonhonooenerugi",
        "darkness": "kihonakuenerugi",
    }
    return _TYPE_TO_ID.get(energy_type, "basic-energy")


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
    # このターンに自分の山札を切ったか（暗号マニアの解読を、切札系の後に回すため）
    own_deck_shuffled_this_turn: bool = False
    ability_used_this_turn: bool = False
    ability_declared_this_turn: str | None = None
    stadium_played_this_turn: bool = False
    fighting_damage_plus_30_count_this_turn: int = 0
    energy_attached_this_turn: bool = False
    retreat_used_this_turn: bool = False
    this_turn_attack_name: str | None = None
    this_turn_attack_actor_id: str | None = None
    last_turn_attack_name: list = field(default_factory=lambda: [None, None])
    last_turn_attack_actor_id: list = field(default_factory=lambda: [None, None])
    our_ko_by_damage_last_turn: list = field(default_factory=lambda: [False, False])
    drawn_this_turn: list = field(default_factory=list)
    record_frame_fn: Callable[["GameState"], None] | None = field(default=None, repr=False)
    turn_when_disabled_attack: list = field(default_factory=lambda: [None, None])
    weights: GameWeights | None = None
    weights_by_player: list = field(default_factory=lambda: [None, None], repr=False)
    stadium: object | None = None
    goods_locked_next_turn: int | None = None  # むずむずかふん: グッズ使用ロック対象プレイヤー
    briar_extra_prize: bool = False  # ブライア: テラスタルKO時サイド+1
    stadium_played_by: int | None = None
    choice_log: list = field(default_factory=list, repr=False)
    use_attack_minimax: bool = True
    frame_index: int = 0
    q_energy_attach_model_path: str | None = None
    q_energy_attach_model_path_by_player: list = field(default_factory=lambda: [None, None], repr=False)
    q_energy_attach_lambda: float = 0.3
    pi_energy_attach_model_path: str | None = None
    pi_energy_attach_model_path_by_player: list = field(default_factory=lambda: [None, None], repr=False)
    pi_energy_attach_lambda: float = 0.1
    pi_attack_model_path: str | None = None
    pi_attack_model_path_by_player: list = field(default_factory=lambda: [None, None], repr=False)
    pi_attack_lambda: float = 0.1
    value_attack_model_path: str | None = None
    value_attack_model_path_by_player: list = field(default_factory=lambda: [None, None], repr=False)
    value_attack_lambda: float = 0.1
    # V(s) — minimax 末端評価を補正する状態価値モデル
    state_value_model_path: str | None = None
    state_value_model_path_by_player: list = field(default_factory=lambda: [None, None], repr=False)
    state_value_lambda: float = 0.3
    online_eval_enabled_by_player: list = field(default_factory=lambda: [True, True], repr=False)
    # 学習ループの global_step を外から代入する想定（推奨）。λ ウォームアップに使う。
    total_env_steps: int = 0
    # True のときだけ _log_choice で total_env_steps を +1（ローカル単体試合用。RL では False のまま）
    increment_total_env_steps_on_log: bool = False
    # True のときは両プレイヤーとも policy_rules_only（従来）。混合対戦は rules_only_policy_by_player を使う。
    rules_only_policy: bool = False
    rules_only_policy_by_player: list[bool] = field(default_factory=lambda: [False, False])
    # True のとき後攻プレイヤーのみ NN を使い、先行は rules_only 扱い（ゲーム開始後に first_player が確定してから有効）
    nn_second_player_only: bool = False
    # 攻撃直前のにげ: π(s) を logits に加算（action_dim = 1 + BENCH_SIZE）
    pi_retreat_before_attack_model_path: str | None = None
    # NN 側の最大係数（実効値はウォームアップで 0→この値まで線形）
    pi_retreat_before_attack_lambda: float = 0.5
    pi_retreat_before_attack_lambda_warmup_steps: int = 0
    # "linear" | "sigmoid"（後者は λ_max * sigmoid((step/warmup)*6 - 3)）
    pi_retreat_before_attack_lambda_schedule: str = "linear"
    # 互換用。攻撃前にげの λ ウォームアップは total_env_steps を使用する
    pi_retreat_before_attack_training_step: int = 0
    # True ならマスク付き argmax。False なら softmax サンプル（探索用）
    retreat_before_attack_deterministic: bool = True
    # 合法行動から一様ランダムに上書きする確率（軽い ε-greedy、開始値）
    retreat_before_attack_epsilon: float = 0.0
    # ε の終端値（policy_schedule_horizon_steps と組み合わせて線形／指数減衰）
    retreat_before_attack_epsilon_end: float = 0.0
    retreat_before_attack_epsilon_schedule: str = "linear"
    retreat_before_attack_epsilon_decay_k: float = 3.0
    # total_env_steps / horizon で progress∈[0,1]。0 なら ε・ヒューリスティック減衰なし
    policy_schedule_horizon_steps: int = 0
    # ヒューリスティック logits に掛ける係数: max(floor, 1 - progress * heuristic_decay)
    retreat_before_attack_heuristic_decay: float = 0.7
    retreat_before_attack_heuristic_scale_floor: float = 0.3
    # True のとき π とヒューリスティック合成前に |·| 平均でスケール合わせ（合成 logits の暴れ防止）
    retreat_before_attack_normalize_logits: bool = True

    def get_weights_for_player(self, player_index: int) -> GameWeights | None:
        """そのプレイヤーの選択に使う重み。weights_by_player が指定されていればそれ、否则 weights。"""
        w = self.weights_by_player[player_index]
        return w if w is not None else self.weights

    def log(self, msg: str) -> None:
        if self.log_fn:
            self.log_fn(msg)

    def _record_frame(self) -> None:
        if self.record_frame_fn:
            self.record_frame_fn(self)
        self.frame_index += 1

    def opponent(self) -> int:
        return 1 - self.current_player

    def active_player_state(self) -> PlayerState:
        return self.players[self.current_player]

    def defending_player_state(self) -> PlayerState:
        return self.players[self.opponent()]

    def player_name(self, i: int) -> str:
        return _player_name(self, i)

    def copy_for_simulation(self) -> "GameState":
        """minimax 等のシミュレーション用に状態を deep copy する。log / record_frame は無効化する。"""
        c = copy.deepcopy(self)
        c.log_fn = None
        c.record_frame_fn = None
        c.use_attack_minimax = False
        c.use_energy_attack_lookahead = False
        return c


def _log_choice(
    state: "GameState",
    choice_type: str,
    *,
    card_id: str | None = None,
    attack_name: str | None = None,
    can_kill: bool | None = None,
    player: int | None = None,
    retreat_policy_action_id: int | None = None,
    policy_action_id: int | None = None,
    retreat_policy_mask: list[bool] | None = None,
    retreat_policy_logits_pre_mask: list[float] | None = None,
    retreat_policy_logits: list[float] | None = None,
    pi_retreat_lambda_effective: float | None = None,
    retreat_policy_entropy: float | None = None,
    retreat_policy_epsilon_greedy: bool | None = None,
    retreat_policy_top2_gap: float | None = None,
    retreat_policy_schedule_progress: float | None = None,
    retreat_policy_heuristic_scale: float | None = None,
    retreat_policy_epsilon_effective: float | None = None,
    # サポートポリシー
    support_policy_action_id: int | None = None,
    support_policy_mask: list[bool] | None = None,
    support_policy_logits_pre_mask: list[float] | None = None,
    support_policy_logits: list[float] | None = None,
    pi_support_lambda_effective: float | None = None,
    support_policy_entropy: float | None = None,
    support_policy_epsilon_greedy: bool | None = None,
    support_policy_top2_gap: float | None = None,
    support_policy_schedule_progress: float | None = None,
    support_policy_heuristic_scale: float | None = None,
    support_policy_epsilon_effective: float | None = None,
    # エネルギー付与ポリシー
    energy_policy_action_id: int | None = None,
    energy_policy_mask: list[bool] | None = None,
    energy_policy_logits: list[float] | None = None,
    energy_policy_logits_pre_mask: list[float] | None = None,
    energy_policy_top2_gap: float | None = None,
    energy_policy_entropy: float | None = None,
    energy_policy_epsilon_greedy: bool | None = None,
    energy_policy_heuristic_scale: float | None = None,
    energy_policy_schedule_progress: float | None = None,
    energy_policy_epsilon_effective: float | None = None,
    pi_energy_policy_lambda_effective: float | None = None,
    energy_type: str | None = None,
    energy_before: int | None = None,
    energy_after: int | None = None,
    can_attack_after: bool | None = None,
    can_kill_after: bool | None = None,
    energy_lookahead_used: bool | None = None,
    energy_lookahead_score: float | None = None,
) -> None:
    """学習用に選択を記録する。選択直後の盤面評価（手番側有利ほど正）を eval に乗せる。"""
    p = state.current_player if player is None else player
    entry = {
        "type": choice_type,
        "player": p,
        "turn": state.turn_count,
    }
    # この選択の直前に記録されたフレームインデックス（state_before_action）を保存する。
    # frame_index は _record_frame 呼び出しごとに 1 ずつ増える設計なので、直前のスナップショットは frame_index - 1。
    # 初期状態では setup_game の最後で 1 回だけ _record_frame が呼ばれるため、最初の選択でも 0 以上になる。
    prev_frame = state.frame_index - 1
    if prev_frame >= 0:
        entry["frame_index"] = prev_frame
    if card_id is not None:
        entry["card_id"] = card_id
    if attack_name is not None:
        entry["attack_name"] = attack_name
    if can_kill is not None:
        entry["can_kill"] = int(bool(can_kill))
    if retreat_policy_action_id is not None:
        entry["retreat_policy_action_id"] = int(retreat_policy_action_id)
    if policy_action_id is not None:
        entry["policy_action_id"] = int(policy_action_id)
    if retreat_policy_mask is not None:
        entry["retreat_policy_mask"] = [bool(x) for x in retreat_policy_mask]
    if retreat_policy_logits_pre_mask is not None:
        entry["retreat_policy_logits_pre_mask"] = [float(x) for x in retreat_policy_logits_pre_mask]
    if retreat_policy_logits is not None:
        entry["retreat_policy_logits"] = [float(x) for x in retreat_policy_logits]
    if pi_retreat_lambda_effective is not None:
        entry["pi_retreat_lambda_effective"] = float(pi_retreat_lambda_effective)
    if retreat_policy_entropy is not None:
        entry["retreat_policy_entropy"] = float(retreat_policy_entropy)
    if retreat_policy_epsilon_greedy is not None:
        entry["retreat_policy_epsilon_greedy"] = bool(retreat_policy_epsilon_greedy)
    if retreat_policy_top2_gap is not None:
        v = float(retreat_policy_top2_gap)
        entry["retreat_policy_top2_gap"] = v
        entry["top2_gap"] = v
    if retreat_policy_schedule_progress is not None:
        entry["retreat_policy_schedule_progress"] = float(retreat_policy_schedule_progress)
    if retreat_policy_heuristic_scale is not None:
        entry["retreat_policy_heuristic_scale"] = float(retreat_policy_heuristic_scale)
    if retreat_policy_epsilon_effective is not None:
        entry["retreat_policy_epsilon_effective"] = float(retreat_policy_epsilon_effective)
    # サポートポリシー
    if support_policy_action_id is not None:
        entry["support_policy_action_id"] = int(support_policy_action_id)
    if policy_action_id is None and support_policy_action_id is not None:
        entry["policy_action_id"] = int(support_policy_action_id)
    if support_policy_mask is not None:
        entry["support_policy_mask"] = [bool(x) for x in support_policy_mask]
    if support_policy_logits_pre_mask is not None:
        entry["support_policy_logits_pre_mask"] = [float(x) for x in support_policy_logits_pre_mask]
    if support_policy_logits is not None:
        entry["support_policy_logits"] = [float(x) for x in support_policy_logits]
    if pi_support_lambda_effective is not None:
        entry["pi_support_lambda_effective"] = float(pi_support_lambda_effective)
    if support_policy_entropy is not None:
        entry["support_policy_entropy"] = float(support_policy_entropy)
    if support_policy_epsilon_greedy is not None:
        entry["support_policy_epsilon_greedy"] = bool(support_policy_epsilon_greedy)
    if support_policy_top2_gap is not None:
        v = float(support_policy_top2_gap)
        entry["support_policy_top2_gap"] = v
        if "top2_gap" not in entry:
            entry["top2_gap"] = v
    if support_policy_schedule_progress is not None:
        entry["support_policy_schedule_progress"] = float(support_policy_schedule_progress)
    if support_policy_heuristic_scale is not None:
        entry["support_policy_heuristic_scale"] = float(support_policy_heuristic_scale)
    if support_policy_epsilon_effective is not None:
        entry["support_policy_epsilon_effective"] = float(support_policy_epsilon_effective)
    # エネルギー付与ポリシー
    if energy_policy_action_id is not None:
        entry["energy_policy_action_id"] = int(energy_policy_action_id)
    if energy_policy_mask is not None:
        entry["energy_policy_mask"] = [bool(x) for x in energy_policy_mask]
    if energy_policy_logits_pre_mask is not None:
        entry["energy_policy_logits_pre_mask"] = [float(x) for x in energy_policy_logits_pre_mask]
    if energy_policy_logits is not None:
        entry["energy_policy_logits"] = [float(x) for x in energy_policy_logits]
    if energy_policy_top2_gap is not None:
        v = float(energy_policy_top2_gap)
        entry["energy_policy_top2_gap"] = v
        if "top2_gap" not in entry:
            entry["top2_gap"] = v
    if energy_policy_entropy is not None:
        entry["energy_policy_entropy"] = float(energy_policy_entropy)
    if energy_policy_epsilon_greedy is not None:
        entry["energy_policy_epsilon_greedy"] = bool(energy_policy_epsilon_greedy)
    if energy_policy_heuristic_scale is not None:
        entry["energy_policy_heuristic_scale"] = float(energy_policy_heuristic_scale)
    if energy_policy_schedule_progress is not None:
        entry["energy_policy_schedule_progress"] = float(energy_policy_schedule_progress)
    if energy_policy_epsilon_effective is not None:
        entry["energy_policy_epsilon_effective"] = float(energy_policy_epsilon_effective)
    if pi_energy_policy_lambda_effective is not None:
        entry["pi_energy_policy_lambda_effective"] = float(pi_energy_policy_lambda_effective)
    if energy_type is not None:
        entry["energy_type"] = str(energy_type)
    if energy_before is not None:
        entry["energy_before"] = int(energy_before)
    if energy_after is not None:
        entry["energy_after"] = int(energy_after)
    if can_attack_after is not None:
        entry["can_attack_after"] = bool(can_attack_after)
    if can_kill_after is not None:
        entry["can_kill_after"] = bool(can_kill_after)
    if energy_lookahead_used is not None:
        entry["energy_lookahead_used"] = bool(energy_lookahead_used)
    if energy_lookahead_score is not None:
        entry["energy_lookahead_score"] = float(energy_lookahead_score)
    try:
        from .evaluate import evaluate_board
        entry["eval"] = evaluate_board(state, p)
    except Exception:
        pass
    state.choice_log.append(entry)
    if getattr(state, "increment_total_env_steps_on_log", False):
        state.total_env_steps = int(getattr(state, "total_env_steps", 0)) + 1


def rules_only_for_player(state: GameState, player_index: int | None = None) -> bool:
    """
    そのプレイヤーが「ルールのみベースライン」か。
    player_index 省略時は手番（current_player）。旧 pkl に rules_only_policy_by_player が無いときは rules_only_policy にフォールバック。
    nn_second_player_only=True のとき、先行プレイヤーは rules_only 扱い（後攻のみ NN）。
    """
    pl = int(state.current_player if player_index is None else player_index)
    if getattr(state, "nn_second_player_only", False):
        # 先行プレイヤーは rules_only、後攻は NN を使う
        return pl == state.first_player
    by = getattr(state, "rules_only_policy_by_player", None)
    if by is not None and len(by) > pl:
        return bool(by[pl])
    return bool(getattr(state, "rules_only_policy", False))


def setup_game(
    seed: int | None = None,
    log_fn: Callable[[str], None] | None = None,
    record_frame_fn: Callable[["GameState"], None] | None = None,
    deck0: int = 0,
    deck1: int = 1,
    deck_code0: str | None = None,
    deck_code1: str | None = None,
    weights: GameWeights | None = None,
    weights_player0: GameWeights | None = None,
    weights_player1: GameWeights | None = None,
    use_attack_minimax: bool = True,
    use_energy_attack_lookahead: bool = True,
    q_energy_attach_model_path: str | None = None,
    q_energy_attach_model_path_p0: str | None = None,
    q_energy_attach_model_path_p1: str | None = None,
    q_energy_attach_lambda: float = 0.3,
    pi_energy_attach_model_path: str | None = None,
    pi_energy_attach_model_path_p0: str | None = None,
    pi_energy_attach_model_path_p1: str | None = None,
    pi_energy_attach_lambda: float = 0.1,
    pi_attack_model_path: str | None = None,
    pi_attack_model_path_p0: str | None = None,
    pi_attack_model_path_p1: str | None = None,
    pi_attack_lambda: float = 0.1,
    value_attack_model_path: str | None = None,
    value_attack_model_path_p0: str | None = None,
    value_attack_model_path_p1: str | None = None,
    value_attack_lambda: float = 0.1,
    state_value_model_path: str | None = None,
    state_value_model_path_p0: str | None = None,
    state_value_model_path_p1: str | None = None,
    state_value_lambda: float = 0.3,
    online_eval_enabled_p0: bool | None = None,
    online_eval_enabled_p1: bool | None = None,
    pi_retreat_before_attack_model_path: str | None = None,
    pi_retreat_before_attack_lambda: float = 0.5,
    pi_retreat_before_attack_lambda_warmup_steps: int = 0,
    pi_retreat_before_attack_training_step: int = 0,
    retreat_before_attack_deterministic: bool = True,
    retreat_before_attack_epsilon: float = 0.0,
    retreat_before_attack_epsilon_end: float = 0.0,
    retreat_before_attack_epsilon_schedule: str = "linear",
    retreat_before_attack_epsilon_decay_k: float = 3.0,
    policy_schedule_horizon_steps: int = 0,
    retreat_before_attack_heuristic_decay: float = 0.7,
    retreat_before_attack_heuristic_scale_floor: float = 0.3,
    retreat_before_attack_normalize_logits: bool = True,
    total_env_steps: int = 0,
    increment_total_env_steps_on_log: bool = False,
    pi_retreat_before_attack_lambda_schedule: str = "linear",
    # サポートポリシー
    pi_support_model_path: str | None = None,
    pi_support_model_path_p0: str | None = None,
    pi_support_model_path_p1: str | None = None,
    pi_support_lambda: float = 0.5,
    q_support_model_path: str | None = None,
    q_support_model_path_p0: str | None = None,
    q_support_model_path_p1: str | None = None,
    q_support_lambda: float = 3.0,
    q_support_lambda_dynamic: bool = False,
    q_support_lambda_attack_ready: float = 0.3,
    q_support_lambda_setup: float = 1.2,
    pi_support_lambda_warmup_steps: int = 0,
    pi_support_lambda_schedule: str = "linear",
    support_deterministic: bool = True,
    support_epsilon: float = 0.0,
    support_epsilon_end: float = 0.0,
    support_epsilon_schedule: str = "linear",
    support_epsilon_decay_k: float = 3.0,
    support_heuristic_decay: float = 0.7,
    support_heuristic_scale_floor: float = 0.3,
    support_normalize_logits: bool = True,
    # エネルギーポリシー（energy_policy.py）
    pi_energy_policy_model_path: str | None = None,
    pi_energy_policy_lambda: float = 0.5,
    pi_energy_policy_lambda_warmup_steps: int = 0,
    pi_energy_policy_lambda_schedule: str = "linear",
    energy_policy_deterministic: bool = True,
    energy_policy_epsilon: float = 0.0,
    energy_policy_epsilon_end: float = 0.0,
    energy_policy_epsilon_schedule: str = "linear",
    energy_policy_epsilon_decay_k: float = 3.0,
    energy_policy_heuristic_decay: float = 0.7,
    energy_policy_heuristic_scale_floor: float = 0.3,
    energy_policy_normalize_logits: bool = True,
    rules_only_policy: bool = False,
    rules_only_player0: bool = False,
    rules_only_player1: bool = False,
    nn_second_player_only: bool = False,
) -> GameState:
    """
    デッキを組んで初手 7 枚をセット。
    deck0 / deck1 で固定デッキ番号（0=A, 1=B, 2=C, 3=D, 4=E）。
    deck_code0 / deck_code1 を指定すると read_cards_result.json のそのデッキコードの一覧でデッキを組む（未指定なら deck0 / deck1 を使用）。
    weights_player0 / weights_player1 を指定すると、そのプレイヤーだけにその重みを適用する（重みあり vs 重みなしの対戦に使う）。
    rules_only_policy=True のときは従来どおり両プレイヤーにルールのみを強制する。
    rules_only_policy=False のとき rules_only_player0 / rules_only_player1 で片側だけルールのみ（混合対戦）にできる。
    詳細は game/policy_rules_only.py。
    total_env_steps は学習ループの global_step を代入する（increment_total_env_steps_on_log=False 推奨）。
    """
    if seed is not None:
        random.seed(seed)
    if rules_only_policy:
        ro0, ro1 = True, True
    else:
        ro0 = bool(rules_only_player0)
        ro1 = bool(rules_only_player1)
    ro_both = ro0 and ro1
    ro_any = ro0 or ro1

    base_w = weights if weights is not None else GameWeights()
    eff_weights = GameWeights() if ro_both else base_w
    eff_minimax = False if ro_both else use_attack_minimax
    eff_en_lookahead = False if ro_both else use_energy_attack_lookahead
    eff_q = None if ro_both else q_energy_attach_model_path
    eff_pi_e = None if ro_both else pi_energy_attach_model_path
    eff_pi_a = None if ro_both else pi_attack_model_path
    eff_val = None if ro_both else value_attack_model_path
    # デッキコード指定時は registered_decks.json から正しいインデックスを解決
    from deck import find_deck_index_by_code
    resolved_deck0 = (find_deck_index_by_code(deck_code0) if deck_code0 else None) or deck0
    resolved_deck1 = (find_deck_index_by_code(deck_code1) if deck_code1 else None) or deck1
    state = GameState(
        log_fn=log_fn,
        record_frame_fn=record_frame_fn,
        deck_indices=(resolved_deck0, resolved_deck1),
        weights=eff_weights,
        use_attack_minimax=eff_minimax,
        rules_only_policy=ro_both,
        rules_only_policy_by_player=[ro0, ro1],
        q_energy_attach_model_path=eff_q,
        pi_energy_attach_model_path=eff_pi_e,
        pi_attack_model_path=eff_pi_a,
    )
    state.use_energy_attack_lookahead = eff_en_lookahead
    if q_energy_attach_model_path_p0 is not None or q_energy_attach_model_path_p1 is not None:
        state.q_energy_attach_model_path_by_player[0] = q_energy_attach_model_path_p0
        state.q_energy_attach_model_path_by_player[1] = q_energy_attach_model_path_p1
    state.q_energy_attach_lambda = float(q_energy_attach_lambda)
    if pi_energy_attach_model_path_p0 is not None or pi_energy_attach_model_path_p1 is not None:
        state.pi_energy_attach_model_path_by_player[0] = pi_energy_attach_model_path_p0
        state.pi_energy_attach_model_path_by_player[1] = pi_energy_attach_model_path_p1
    state.pi_energy_attach_lambda = float(pi_energy_attach_lambda)
    if pi_attack_model_path_p0 is not None or pi_attack_model_path_p1 is not None:
        state.pi_attack_model_path_by_player[0] = pi_attack_model_path_p0
        state.pi_attack_model_path_by_player[1] = pi_attack_model_path_p1
    state.pi_attack_lambda = float(pi_attack_lambda)
    if value_attack_model_path_p0 is not None or value_attack_model_path_p1 is not None:
        state.value_attack_model_path_by_player[0] = value_attack_model_path_p0
        state.value_attack_model_path_by_player[1] = value_attack_model_path_p1
    state.value_attack_lambda = float(value_attack_lambda)
    if value_attack_model_path is not None and (value_attack_model_path_p0 is None and value_attack_model_path_p1 is None):
        state.value_attack_model_path = value_attack_model_path
    if state_value_model_path_p0 is not None or state_value_model_path_p1 is not None:
        state.state_value_model_path_by_player[0] = None if ro_both else state_value_model_path_p0
        state.state_value_model_path_by_player[1] = None if ro_both else state_value_model_path_p1
    elif state_value_model_path is not None:
        state.state_value_model_path = None if ro_both else state_value_model_path
    state.state_value_lambda = float(state_value_lambda)
    state.pi_retreat_before_attack_model_path = None if ro_both else pi_retreat_before_attack_model_path
    state.pi_retreat_before_attack_lambda = float(pi_retreat_before_attack_lambda)
    state.pi_retreat_before_attack_lambda_warmup_steps = int(pi_retreat_before_attack_lambda_warmup_steps)
    state.pi_retreat_before_attack_training_step = int(pi_retreat_before_attack_training_step)
    state.retreat_before_attack_deterministic = bool(retreat_before_attack_deterministic)
    state.retreat_before_attack_epsilon = float(retreat_before_attack_epsilon)
    state.retreat_before_attack_epsilon_end = float(retreat_before_attack_epsilon_end)
    state.retreat_before_attack_epsilon_schedule = str(retreat_before_attack_epsilon_schedule)
    state.retreat_before_attack_epsilon_decay_k = float(retreat_before_attack_epsilon_decay_k)
    state.policy_schedule_horizon_steps = int(policy_schedule_horizon_steps)
    state.retreat_before_attack_heuristic_decay = float(retreat_before_attack_heuristic_decay)
    state.retreat_before_attack_heuristic_scale_floor = float(retreat_before_attack_heuristic_scale_floor)
    state.retreat_before_attack_normalize_logits = bool(retreat_before_attack_normalize_logits)
    state.total_env_steps = int(total_env_steps)
    state.increment_total_env_steps_on_log = bool(increment_total_env_steps_on_log)
    state.pi_retreat_before_attack_lambda_schedule = str(pi_retreat_before_attack_lambda_schedule)
    # サポートポリシー
    state.pi_support_model_path = None if ro_both else pi_support_model_path
    state.pi_support_model_path_by_player = [
        None if ro_both else pi_support_model_path_p0,
        None if ro_both else pi_support_model_path_p1,
    ]
    state.pi_support_lambda = float(pi_support_lambda)
    state.q_support_model_path = None if ro_both else q_support_model_path
    state.q_support_model_path_by_player = [
        None if ro_both else q_support_model_path_p0,
        None if ro_both else q_support_model_path_p1,
    ]
    state.q_support_lambda = float(q_support_lambda)
    state.q_support_lambda_dynamic = bool(q_support_lambda_dynamic)
    state.q_support_lambda_attack_ready = float(q_support_lambda_attack_ready)
    state.q_support_lambda_setup = float(q_support_lambda_setup)
    state.pi_support_lambda_warmup_steps = int(pi_support_lambda_warmup_steps)
    state.pi_support_lambda_schedule = str(pi_support_lambda_schedule)
    state.support_deterministic = bool(support_deterministic)
    state.support_epsilon = float(support_epsilon)
    state.support_epsilon_end = float(support_epsilon_end)
    state.support_epsilon_schedule = str(support_epsilon_schedule)
    state.support_epsilon_decay_k = float(support_epsilon_decay_k)
    state.support_heuristic_decay = float(support_heuristic_decay)
    state.support_heuristic_scale_floor = float(support_heuristic_scale_floor)
    state.support_normalize_logits = bool(support_normalize_logits)
    # エネルギーポリシー
    state.pi_energy_policy_model_path = None if ro_both else pi_energy_policy_model_path
    state.pi_energy_policy_lambda = float(pi_energy_policy_lambda)
    state.pi_energy_policy_lambda_warmup_steps = int(pi_energy_policy_lambda_warmup_steps)
    state.pi_energy_policy_lambda_schedule = str(pi_energy_policy_lambda_schedule)
    state.energy_policy_deterministic = bool(energy_policy_deterministic)
    state.energy_policy_epsilon = float(energy_policy_epsilon)
    state.energy_policy_epsilon_end = float(energy_policy_epsilon_end)
    state.energy_policy_epsilon_schedule = str(energy_policy_epsilon_schedule)
    state.energy_policy_epsilon_decay_k = float(energy_policy_epsilon_decay_k)
    state.energy_policy_heuristic_decay = float(energy_policy_heuristic_decay)
    state.energy_policy_heuristic_scale_floor = float(energy_policy_heuristic_scale_floor)
    state.energy_policy_normalize_logits = bool(energy_policy_normalize_logits)
    if online_eval_enabled_p0 is not None:
        state.online_eval_enabled_by_player[0] = bool(online_eval_enabled_p0)
    if online_eval_enabled_p1 is not None:
        state.online_eval_enabled_by_player[1] = bool(online_eval_enabled_p1)
    if weights_player0 is not None:
        state.weights_by_player[0] = weights_player0
    if weights_player1 is not None:
        state.weights_by_player[1] = weights_player1
    if ro_both:
        state.weights = GameWeights()
        state.weights_by_player = [None, None]
        state.q_energy_attach_model_path = None
        state.q_energy_attach_model_path_by_player = [None, None]
        state.pi_energy_attach_model_path = None
        state.pi_energy_attach_model_path_by_player = [None, None]
        state.pi_attack_model_path = None
        state.pi_attack_model_path_by_player = [None, None]
        state.value_attack_model_path = None
        state.value_attack_model_path_by_player = [None, None]
        state.state_value_model_path = None
        state.state_value_model_path_by_player = [None, None]
        state.pi_retreat_before_attack_model_path = None
        state.pi_support_model_path = None
        state.pi_energy_policy_model_path = None
        state.use_attack_minimax = False
        state.use_energy_attack_lookahead = False
        state.rules_only_policy = True
        state.rules_only_policy_by_player = [True, True]
        state.nn_second_player_only = False
    elif ro_any:
        for i, roi in enumerate((ro0, ro1)):
            if not roi:
                continue
            state.weights_by_player[i] = GameWeights()
            state.q_energy_attach_model_path_by_player[i] = None
            state.pi_energy_attach_model_path_by_player[i] = None
            state.pi_attack_model_path_by_player[i] = None
            state.value_attack_model_path_by_player[i] = None
            state.state_value_model_path_by_player[i] = None
    state.nn_second_player_only = bool(nn_second_player_only) and not ro_both
    state.log("========== ゲーム開始 ==========")
    state.log(
        f"方策: プレイヤー0={'ルールのみ' if ro0 else 'フル（重み・NN・minimax 可）'} / "
        f"プレイヤー1={'ルールのみ' if ro1 else 'フル（重み・NN・minimax 可）'}"
        + (" / 後攻のみNN" if nn_second_player_only and not ro_both else "")
    )
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
        _put_one_pokemon_active(state, state.players[i], i)
        if state.players[i].active:
            state.log(f"{state.player_name(i)}: バトル場に {state.players[i].active.card.name} を出す（HP {state.players[i].active.max_hp}）")
        # ゲーム開始時にベンチにポケモンを置かない
        # （相手に情報を与えない、ニャースex等は出すとおくのてキャッチが使えなくなる等デメリットのみ）
        # _fill_bench_from_hand(state.players[i], state, i, log_each=False, bench_fill_context="game_start")
        if state.players[i].bench:
            bench_names = [b.card.name for b in state.players[i].bench]
            state.log(f"{state.player_name(i)}: ベンチに {len(state.players[i].bench)} 体 [{', '.join(bench_names)}]")

    for i in range(2):
        for _ in range(PRIZE_COUNT):
            if state.players[i].deck:
                state.players[i].prize_pile.append(state.players[i].deck.pop(0))
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


def _is_basic_rioru_card(card) -> bool:
    """進化前のリオル（ルカリオ系は含めない）。"""
    if not card:
        return False
    n = (getattr(card, "name", "") or "").strip()
    cid = (getattr(card, "id", "") or "").strip().lower()
    return n == "リオル" or cid.startswith("rioru")


def _is_rioru_line_card(card) -> bool:
    """リオルライン全体（リオル・ルカリオ・メガルカリオex等）か。"""
    if not card:
        return False
    n = (getattr(card, "name", "") or "").strip()
    cid = (getattr(card, "id", "") or "").strip().lower()
    if n in ("リオル", "ルカリオ", "メガルカリオex", "メガルカリオ"):
        return True
    return cid.startswith("rioru") or cid.startswith("rukario") or cid.startswith("mrukario")


def _count_basic_rioru_on_field(player: PlayerState) -> int:
    """バトル場＋ベンチにいるたねのリオルの数。"""
    n = 0
    if player.active and _is_basic_rioru_card(player.active.card):
        n += 1
    for bp in player.bench or []:
        if bp and bp.card and _is_basic_rioru_card(bp.card):
            n += 1
    return n


def _count_rioru_line_on_field(player: PlayerState) -> int:
    """バトル場＋ベンチにいるリオルライン全体（リオル＋進化後含む）の数。"""
    n = 0
    if player.active and _is_rioru_line_card(player.active.card):
        n += 1
    for bp in player.bench or []:
        if bp and bp.card and _is_rioru_line_card(bp.card):
            n += 1
    return n


def _field_has_basic_rioru_ko_next_opponent_turn(state: "GameState", player_index: int) -> bool:
    """次の相手の攻撃で、場のリオルのどれかがきぜつしうる（最大受けダメージ以上のダメージを受ける）なら True。"""
    from .damage import _opponent_max_effective_damage

    if state.current_player != player_index:
        return False
    p = state.players[player_index]
    opp_max = _opponent_max_effective_damage(state)
    if opp_max <= 0:
        return False
    if p.active and _is_basic_rioru_card(p.active.card):
        hp = p.active.hp
        if hp is not None and hp > 0 and hp <= opp_max:
            return True
    for bp in p.bench or []:
        if not bp or not bp.card or not _is_basic_rioru_card(bp.card):
            continue
        hp = bp.hp
        if hp is not None and hp > 0 and hp <= opp_max:
            return True
    return False


def _may_add_basic_rioru_to_field(
    state: "GameState",
    player: PlayerState,
    player_index: int,
    *,
    bench_fill_context: str | None,
) -> bool:
    """
    手札からリオルを場に出してよいか。
    リオルライン全体（リオル＋メガルカリオex等の進化後含む）で原則 2 体まで。
    3 体目のルカリオラインはあまり使わない上、マクノシタを置くスペースがなくなるため制限。
    すでに 2 体いるときは、次相手ターンにどれかが落ちそうなときだけ 3 体目可。
    ゲーム開始時のベンチ処理では current_player が片側に固定されていないため、3 体目は出さない。
    """
    # リオルライン全体（進化後含む）でカウント
    line_cnt = _count_rioru_line_on_field(player)
    if line_cnt >= 3:
        return False
    if line_cnt <= 1:
        return True
    # これ以上出すと 3 体目
    if bench_fill_context == "game_start":
        return False
    if state.current_player != player_index:
        return False
    return _field_has_basic_rioru_ko_next_opponent_turn(state, player_index)


def _put_one_pokemon_active(state: "GameState", player: PlayerState, player_index: int) -> bool:
    """手札からたねポケモン（進化ポケモンでない）1 体をバトル場に出す。
    デッキ戦略で優先したいタネが手札にあればそれを優先する。
    ベンチと同じ条件（エンジン1体制限、リオルライン2体制限、縛られリスク回避）を適用。
    ただしバトル場は必ず1体出す必要があるので、全部弾かれたらフォールバック。
    """
    deck_index = state.deck_indices[player_index] if state.deck_indices else 0
    priority_ids = get_priority_setup_pokemon_ids(deck_index)
    allow_duplicate_ids = get_allow_duplicate_bench_ids(deck_index)

    def _card_matches_id(c, pid: str) -> bool:
        cid = (getattr(c, "id", "") or "").strip()
        if not cid or not pid:
            return False
        if cid == pid:
            return True
        if cid.startswith(pid + "-") or cid.startswith(pid + "_"):
            return True
        return False

    # ベンチと同じフィルタを適用した候補を作る
    _ENGINE_SINGLE_NAMES = {"ルナトーン", "ソルロック"}
    # バトル場はまだ空なので既存名はベンチのみ（ゲーム開始時はベンチも空）
    existing_names = {getattr(bp.card, "name", "") for bp in player.bench}
    active_name = getattr(player.active.card, "name", "") if player.active else ""
    all_field_names = existing_names | ({active_name} if active_name else set())

    def _is_acceptable(c) -> bool:
        if not is_pokemon(c) or getattr(c, "evolves_from", None):
            return False
        c_name = (getattr(c, "name", "") or "").strip()
        cid = getattr(c, "id", "") or ""
        # 同名重複チェック
        if c_name in existing_names and cid not in allow_duplicate_ids:
            return False
        # エンジン1体制限
        if c_name in _ENGINE_SINGLE_NAMES and c_name in all_field_names:
            return False
        # リオルライン制限
        if _is_basic_rioru_card(c) and not _may_add_basic_rioru_to_field(
            state, player, player_index, bench_fill_context="game_start"
        ):
            return False
        return True

    # 1) 優先リスト順でフィルタ適用
    cands_priority = []
    for pid in priority_ids:
        for i, c in enumerate(player.hand):
            if _is_acceptable(c) and _card_matches_id(c, pid):
                cands_priority.append((i, c))

    # 2) それ以外のたね
    cands_other = []
    priority_hand_idxs = {i for i, _ in cands_priority}
    for i, c in enumerate(player.hand):
        if i in priority_hand_idxs:
            continue
        if _is_acceptable(c):
            cands_other.append((i, c))

    candidates = cands_priority + cands_other

    # フィルタで全部弾かれたらフォールバック（バトル場は必ず出す必要がある）
    if not candidates:
        for i, c in enumerate(player.hand):
            if is_pokemon(c) and not getattr(c, "evolves_from", None):
                candidates = [(i, c)]
                break

    if candidates:
        i, c = candidates[0]
        p = c.copy()
        player.active = BattlePokemon(card=p)
        player.hand.pop(i)
        return True
    return False


def _put_one_pokemon_on_bench(
    player: PlayerState,
    state: GameState,
    player_index: int,
    *,
    log: bool = True,
    bench_fill_context: str | None = None,
) -> bool:
    """手札からたねポケモン（進化ポケモンでない）1 体をベンチに出す（ベンチが 5 体未満のとき）。デッキ戦略で優先ポケモンがあればそれを先に出す。"""
    if len(player.bench) >= BENCH_SIZE:
        return False
    # ドラパルトデッキ: 手札が少ない+ニャースexが山札にある → ベンチ1枠温存
    # ニャースex(おくのてキャッチ→サポートサーチ)でドロー手段を確保するため
    from .deck_strategies import is_dragapult_deck_for_player as _is_drapa_bench
    if _is_drapa_bench(state, player_index) and len(player.bench) >= BENCH_SIZE - 1:
        _hand_size = len(player.hand)
        _has_support = any(is_support(c) for c in player.hand)
        _nyarth_in_deck = any(
            (getattr(c, "name", "") or "").strip() == "ニャースex" for c in player.deck
        )
        if _hand_size <= 3 and not _has_support and _nyarth_in_deck:
            return False  # ベンチ1枠をニャースex用に温存
    deck_index = state.deck_indices[player_index] if state.deck_indices else 0
    priority_ids = set(get_priority_setup_pokemon_ids(deck_index))
    allow_duplicate_ids = get_allow_duplicate_bench_ids(deck_index)
    existing_names = {getattr(bp.card, "name", "") for bp in player.bench}
    # バトル場も含めた既存ポケモン名（同名制限用）
    active_name = getattr(player.active.card, "name", "") if player.active else ""
    all_field_names = existing_names | ({active_name} if active_name else set())
    # エンジンペア（ルナトーン・ソルロック）は各1体で十分。
    # 2体目は基本的に不要（ベンチ枠の無駄）。終盤の圧縮用途は将来学習で対応。
    _ENGINE_SINGLE_NAMES = {"ルナトーン", "ソルロック"}
    cands_priority: list[tuple[int, object]] = []
    cands_other: list[tuple[int, object]] = []
    # ドラパルトデッキ: サポート使用済みターンではニャースexを出さない
    # （おくのてキャッチで取ったサポートは今ターン使えない。ベンチ枠はドラパルトexラインに回す）
    # ただしサポート未使用なら出す（おくのてキャッチ→サポートサーチ→即使用の連携が可能）
    from .deck_strategies import is_dragapult_deck_for_player as _is_drapa_nyarth
    _drapa_skip_nyarth = False
    if _is_drapa_nyarth(state, player_index):
        if state.support_used_this_turn:
            _drapa_skip_nyarth = True
        else:
            # サポート未使用でも、おくのてキャッチで取ったサポートが活用できないならスキップ
            # アカマツ: ファントムダイブに届く(ドラパルトexがエネ1以上)時のみ価値あり
            # リーリエの決心: 手札が多い(6枚以上)なら不要
            # → どちらも活用できないならニャースexを出す意味がない
            _has_drapa_with_energy = any(
                (getattr(bp.card, "name", "") or "").strip() == "ドラパルトex"
                and (getattr(bp, "attached_energy", 0) or 0) >= 1
                for bp in ([player.active] if player.active else []) + list(player.bench or [])
            )
            _hand_is_large = len(player.hand) >= 6
            if not _has_drapa_with_energy and _hand_is_large:
                _drapa_skip_nyarth = True

    for i, c in enumerate(player.hand):
        if is_pokemon(c) and not c.evolves_from:
            cid = getattr(c, "id", "") or ""
            c_name = getattr(c, "name", "") or ""
            if c_name in existing_names and cid not in allow_duplicate_ids:
                continue
            if c_name in _ENGINE_SINGLE_NAMES and c_name in all_field_names:
                continue
            if _is_basic_rioru_card(c) and not _may_add_basic_rioru_to_field(
                state, player, player_index, bench_fill_context=bench_fill_context
            ):
                continue
            # ドラパルトデッキ: サポート使用済みターンはニャースexを温存
            if _drapa_skip_nyarth and c_name == "ニャースex":
                continue
            # ドラパルトデッキ: キチキギスexは必要になるまで出さない
            # さかてにとるは前ターンに自分のポケモンがきぜつしていないと使えない
            # 序盤に出してもボスの指令で呼ばれてサイド2枚献上するだけ
            if _is_drapa_bench(state, player_index) and c_name == "キチキギスex":
                _any_ko = getattr(state, "any_ko_by_opponent_last_turn", [False, False])
                _our_ko = getattr(state, "our_ko_by_damage_last_turn", [False, False])
                _ko_happened = (_any_ko[player_index] if len(_any_ko) > player_index else False) or \
                               (_our_ko[player_index] if len(_our_ko) > player_index else False)
                if not _ko_happened:
                    continue  # きぜつしていない → さかてにとる使えない → 出す意味なし
            # ドラパルトデッキ: マシマシラはベンチに出さない（ただし種切れ防止で他にたねがなければ出す）
            if _is_drapa_bench(state, player_index) and c_name == "マシマシラ":
                _has_other_basic_in_hand = any(
                    is_pokemon(hc) and not getattr(hc, "evolves_from", None)
                    and (getattr(hc, "name", "") or "") != "マシマシラ"
                    and hc is not c
                    for hi, hc in enumerate(player.hand) if hi != i
                )
                _field_pokemon_count = len(player.bench) + (1 if player.active else 0)
                if _has_other_basic_in_hand or _field_pokemon_count >= 2:
                    continue  # 他にたねがあるか場に2体以上いるなら出さない
            # ドラパルトデッキ: ヨマワルラインは場に合計1体まで
            if _is_drapa_bench(state, player_index) and c_name == "ヨマワル":
                _yoma_line_names = {"ヨマワル", "サマヨール", "ヨノワール"}
                _yoma_count = sum(
                    1 for bp in player.bench
                    if getattr(bp.card, "name", "") in _yoma_line_names
                ) + (1 if player.active and getattr(player.active.card, "name", "") in _yoma_line_names else 0)
                if _yoma_count >= 1:
                    continue
            if cid in priority_ids:
                cands_priority.append((i, c))
            else:
                cands_other.append((i, c))
    for i, c in cands_priority + cands_other:
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
    player: PlayerState,
    state: GameState,
    player_index: int,
    *,
    log_each: bool = True,
    bench_fill_context: str | None = None,
) -> None:
    """手札のポケモンをベンチに出す（最大 BENCH_SIZE まで）。"""
    while len(player.bench) < BENCH_SIZE and _put_one_pokemon_on_bench(
        player, state, player_index, log=log_each, bench_fill_context=bench_fill_context
    ):
        pass


_PROMOTE_CAN_KO_SCALE = 10000
_PROMOTE_EVOLUTION_READY_SCALE = 3000
_PROMOTE_DAMAGE_POTENTIAL_SCALE = 10
_PROMOTE_ZERO_RETREAT_SCALE = 5000
_PROMOTE_SIDE_RACE_PENALTY = -8000      # 出すと相手にサイド取り切られるリスク
_PROMOTE_SIDE_RACE_WIN_OVERRIDE = 15000  # 出して倒せば勝てるなら出す
_PROMOTE_HARIYAMA_BONUS = 3000           # 非ルールアタッカーとして優先


def _promote_from_bench(player: PlayerState, state: GameState, player_index: int) -> bool:
    """
    バトル場が空のとき、ベンチから 1 体をバトル場に出す。
    次の自分のターンに相手をきぜつさせられるポケモンを最優先し、手札（進化・エネルギー）を見て
    次ターン強い攻撃ができそうな候補を優先する。にげるコスト 0 のポケモンがいればそれを優先し、
    次のターンですぐに逃げて本命を出せるようにする。否則は HP が高いほう。重みで補正。

    サイドレース意識:
      相手がこのポケモンを倒すとサイドを取り切れる場合、ペナルティを与える。
      ただし出して相手を倒せば勝てる場合は例外（オーバーライド）。
      ハリテヤマ（非ルール・高火力）は中盤のアタッカーとしてボーナス。
    """
    if player.active is not None or not player.bench:
        return False
    opp = state.players[1 - player_index]
    from .damage import _max_effective_damage_for_attacker, _max_effective_damage_if_attach
    from .evolution import _can_evolve_onto
    opp_prizes = len(opp.prize_pile)
    our_prizes = len(player.prize_pile)

    scored = []
    for i, bp in enumerate(player.bench):
        raw_rc = getattr(bp.card, "retreat_cost", 1)
        tool = getattr(bp, "attached_tool", None)
        retreat_cost_effective = max(0, raw_rc - (2 if tool and (getattr(tool, "id", "") or "") == "fuusen" else 0))
        zero_retreat_bonus = _PROMOTE_ZERO_RETREAT_SCALE if retreat_cost_effective == 0 else 0

        can_ko = (
            opp.active is not None
            and opp.active.hp > 0
            and _max_effective_damage_for_attacker(state, bp, opp.active, player_index) >= opp.active.hp
        )
        # 技コストが高い＋にげコスト2以上 → バトル場に縛られて何もできない。
        # 相手にボスの指令で縛られるのと同じ状況を自分から作ってしまう。
        from .attack import get_legal_attack_indices_for_attacker
        can_attack_now = bool(get_legal_attack_indices_for_attacker(state, player, opp, bp))
        stuck_penalty = 0
        if not can_attack_now and retreat_cost_effective >= 2:
            # メインアタッカー（メガルカリオex等）は次ターンにエネ1で攻撃できるので軽減
            bp_name_stuck = (getattr(bp.card, "name", "") or "").strip()
            if bp_name_stuck in ("メガルカリオex", "メガルカリオ", "ルカリオ"):
                stuck_penalty = -2000  # 1ターン我慢で済む
            elif bp_name_stuck == "ドラパルトex":
                stuck_penalty = -2000  # メインアタッカーなので軽減
            else:
                stuck_penalty = -8000  # 縛られリスク大

        base = (_PROMOTE_CAN_KO_SCALE if can_ko else 0) + zero_retreat_bonus + bp.hp + stuck_penalty

        evolution_bonus = 0.0
        for c in player.hand:
            if is_pokemon(c) and getattr(c, "evolves_from", None) and _can_evolve_onto(bp.card, c):
                evolution_bonus = _PROMOTE_EVOLUTION_READY_SCALE
                break
        # 手札に進化先がなくても、山札にあれば将来アタッカーになれる。
        # ルナトーン等のサポート役よりリオル等の進化元を優先。
        if evolution_bonus == 0.0:
            bp_name = (getattr(bp.card, "name", "") or "").strip()
            if bp_name == "リオル":
                has_evo_in_deck = any(
                    is_pokemon(c) and _can_evolve_onto(bp.card, c)
                    for c in player.deck
                )
                if has_evo_in_deck:
                    evolution_bonus = _PROMOTE_EVOLUTION_READY_SCALE * 0.5

        damage_potential = 0
        types = getattr(bp, "attached_energy_types", [])
        for c in player.hand:
            if not is_energy(c):
                continue
            etype = getattr(c, "energy_type", None) or "colorless"
            dmg = _max_effective_damage_if_attach(
                state, bp.card, bp.attached_energy, list(types), etype, opp.active, player_index
            )
            damage_potential = max(damage_potential, dmg)
        base += damage_potential * _PROMOTE_DAMAGE_POTENTIAL_SCALE
        base += evolution_bonus

        # --- サイドレース意識 ---
        prizes_given = _prizes_for_ko(bp)  # このポケモンが倒されたとき相手が取るサイド数
        bp_name = (getattr(bp.card, "name", "") or "").strip()

        # 相手がこのポケモンを倒すとサイド取り切れる → 出したくない
        if opp_prizes <= prizes_given:
            if can_ko:
                # ただし倒せば勝てるなら出す
                ko_prizes = _prizes_for_ko(opp.active) if opp.active else 1
                if our_prizes <= ko_prizes:
                    base += _PROMOTE_SIDE_RACE_WIN_OVERRIDE  # 勝確 → 出す
                else:
                    base += _PROMOTE_SIDE_RACE_PENALTY  # 倒せるが勝てない → リスキー
            else:
                base += _PROMOTE_SIDE_RACE_PENALTY  # 倒せない → 出さない
        # 相手残りサイド1-2枚 → メガルカリオex を押し付ける
        # （相手はメガルカリオex を倒してもサイド3枚で取り切れない → 余分な苦労をかける）
        elif opp_prizes <= 2 and prizes_given >= 3:
            base += 4000  # 相手がサイド取り切れない状況でメガを押し付ける

        # ハリテヤマ（非ルール・高火力アタッカー）はボーナス
        if bp_name == "ハリテヤマ":
            base += _PROMOTE_HARIYAMA_BONUS
        # メガルカリオex/ルカリオ系はメインアタッカーとしてボーナス
        # エネがなくても次ターンに付ければ攻撃できる
        if bp_name in ("メガルカリオex", "メガルカリオ", "ルカリオ"):
            base += 5000

        # ドラパルトex デッキ: メインアタッカーとしてボーナス
        from .deck_strategies import is_dragapult_deck_for_player
        if is_dragapult_deck_for_player(state, player_index):
            _bp_energy = getattr(bp, "attached_energy", 0) or 0
            if bp_name == "ドラパルトex":
                base += 6000  # メインアタッカー
                # エネルギーが多いほどファントムダイブに近い → さらにボーナス
                if _bp_energy >= 2:
                    base += 4000  # あと1枚でファントムダイブ
                elif _bp_energy >= 1:
                    base += 2000  # ジェットヘッド撃てる
            elif bp_name == "スボミー":
                # スボミーは逃げ0+グッズロック+サイド1枚 → 壁として常に優先
                # いつでも逃げられるので前に出しても安全
                base += 4000
            elif bp_name == "ドロンチ":
                base += 2000
            # ex サポートポケモンは前に出さない（倒されると2サイド献上）
            # キチキギスex/ニャースex は攻撃力がなく、KOされるだけ
            if bp_name in ("キチキギスex", "ニャースex"):
                base -= 12000  # ex 2枚サイド献上を強く回避
            # エネ付きサポートポケモンは前に出さない（やられてエネ失う）
            # ドラメシヤ/ドロンチ（進化先にエネ引き継ぎ）やアタッカーは除外
            _drapa_support_names = {"マシマシラ", "ヨマワル", "サマヨール"}
            if bp_name in _drapa_support_names and _bp_energy > 0:
                base -= 4000  # サポート役のエネ付きは温存

        score = base + get_promote_weight(state.get_weights_for_player(player_index), bp.card)
        scored.append((i, score))
    best_idx = max(scored, key=lambda x: x[1])[0]
    promoted_card = player.bench[best_idx].card
    _log_choice(state, "promote", card_id=getattr(promoted_card, "id", None) or getattr(promoted_card, "name", ""), player=player_index)
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
    # きぜつポケモンに付いていたエネルギーをトラッシュに送る
    _put_energy_cards_in_discard(p, getattr(koed_bp, "attached_energy_types", []) or [], state)
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
    # きぜつポケモンに付いていたエネルギーをトラッシュに送る
    _put_energy_cards_in_discard(opp, getattr(koed_bp, "attached_energy_types", []) or [], state)
    if opp.active is koed_bp:
        opp.active = None
    opp.knockouts_suffered += 1
    state._record_frame()
    prize_count = _prizes_for_ko(koed_bp)
    # ブライア: テラスタルポケモンのKOでサイド+1
    if getattr(state, "briar_extra_prize", False):
        p_atk = state.players[state.current_player]
        if p_atk.active and getattr(p_atk.active.card, "is_terastal", False):
            prize_count += 1
            state.log(f"{state.player_name(state.current_player)}: ブライアの効果でサイドを 1 枚多くとる")
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


def mark_own_deck_shuffled(state: GameState) -> None:
    """現在のプレイヤーの山札を切った直後に呼ぶ（暗号マニアのタイミング用）。"""
    state.own_deck_shuffled_this_turn = True
    # デッキを検索した → サイドの中身を推測可能（消去法）
    if not hasattr(state, "deck_searched_by_player"):
        state.deck_searched_by_player = [False, False]
    state.deck_searched_by_player[state.current_player] = True


def start_turn(state: GameState) -> None:
    """ターン開始：ドロー 1 枚、サポート・スタジアム・エネルギー付与未使用にリセット。ねむりならコインで解除判定。"""
    state.support_used_this_turn = False
    state.own_deck_shuffled_this_turn = False
    state.ability_used_this_turn = False
    state.ability_declared_this_turn = None
    state.stadium_played_this_turn = False
    state.fighting_damage_plus_30_count_this_turn = 0
    state.briar_damage_bonus = 0  # 旧互換用（使わない）
    state.briar_extra_prize = False
    if not hasattr(state, "any_ko_by_opponent_last_turn"):
        state.any_ko_by_opponent_last_turn = [False, False]
    state._sakatenitori_used_this_turn = False
    state._adrenabrain_used_this_turn = False
    state._okunote_used_this_turn = False
    state._teisatsushirei_used_this_turn = False
    state._teisatsushirei_used_ids_this_turn = set()  # 各ドロンチのID別トラッキングもリセット
    # かげしばり: ターン開始時に自分のバトル場の retreat_locked を解除
    p_now = state.players[state.current_player]
    if p_now.active and getattr(p_now.active, "retreat_locked", False):
        p_now.active.retreat_locked = False
    # ブライア: extra prize リセット
    state.briar_extra_prize = False
    state.energy_attached_this_turn = False
    state.retreat_used_this_turn = False
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
