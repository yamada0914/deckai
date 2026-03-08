"""ターン実行用：ベンチ選択（キャッチャー対象・生贄・1 エネルギーで KO 可能判定）。"""
from card import is_energy

from .damage import _max_effective_damage_for_attacker, _max_effective_damage_if_attach
from .state import GameState
from .weights import get_catcher_target_weight


def _min_energy_for_any_attack(card) -> int:
    """そのカードの技のうち、必要エネルギー数が最小の値を返す（技が遅い＝大きいほど良い）。"""
    if not getattr(card, "attacks", None):
        return 0
    return max(
        (len(getattr(a, "energy_cost_typed", []) or []) or getattr(a, "energy_cost", 0) for a in card.attacks),
        default=0,
    )


_CATCHER_THREAT_HP_SCALE = 1.0
_CATCHER_THREAT_EX_BONUS = 5000.0


def _catcher_threat_score(card) -> float:
    """相手ベンチの 1 体がどれだけ脅威か。HP が高く・ex なら高くなる。"""
    hp = getattr(card, "max_hp", 0) or 0
    is_ex = getattr(card, "is_ex", False) or ("ex" in (getattr(card, "name", "") or ""))
    return hp * _CATCHER_THREAT_HP_SCALE + (_CATCHER_THREAT_EX_BONUS if is_ex else 0.0)


def _best_opponent_bench_index_for_catcher(state: GameState) -> int | None:
    """
    ポケモンキャッチャーで引きたい相手ベンチのインデックスを返す。
    脅威（HP が高い・ex）を優先して引き出し、同程度なら逃げにくく・技が遅いポケモンを優先し、重みで補正する。
    """
    opp = state.defending_player_state()
    if not opp.bench or not opp.active:
        return None
    weights = state.get_weights_for_player(state.current_player)
    best_idx = None
    best_score = -1.0
    for i, bp in enumerate(opp.bench):
        card = bp.card
        threat = _catcher_threat_score(card)
        retreat = getattr(card, "retreat_cost", 0)
        min_energy = _min_energy_for_any_attack(card)
        raw = retreat * 1000 + min_energy
        score = threat + raw + get_catcher_target_weight(weights, card)
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx


_SACRIFICE_POKEMON_NAMES = frozenset({"カラミンゴ"})


def _best_sacrifice_bench_index(state: GameState) -> int | None:
    """
    生贄に出すベンチのインデックスを返す。
    まず _SACRIFICE_POKEMON_NAMES に名前があるポケモンを優先し、いなければ
    「今のエネルギーでは相手を倒せない」ベンチのうち max_hp が最小の 1 体を選ぶ（カード追加に強い）。
    """
    p = state.active_player_state()
    opp = state.defending_player_state()
    if not p.bench:
        return None
    for i, bp in enumerate(p.bench):
        if getattr(bp.card, "name", "") in _SACRIFICE_POKEMON_NAMES:
            return i
    if not opp.active:
        return None
    cannot_ko = []
    for i, bp in enumerate(p.bench):
        dmg = _max_effective_damage_for_attacker(state, bp, opp.active, state.current_player)
        if dmg < opp.active.hp:
            cannot_ko.append((i, getattr(bp.card, "max_hp", 60)))
    if not cannot_ko:
        return None
    return min(cannot_ko, key=lambda x: x[1])[0]


def _has_bench_that_can_ko_after_one_attach(state: GameState) -> bool:
    """
    手札のエネルギーを 1 つ付けたときに相手をきぜつさせられるベンチが 1 体でもいれば True。
    カード名に依存せず、どのデッキでも使える。
    """
    p = state.active_player_state()
    opp = state.defending_player_state()
    if not opp.active or not p.bench:
        return False
    types_to_try = list({getattr(c, "energy_type", "colorless") for c in p.hand if is_energy(c)} or ["colorless"])
    for bp in p.bench:
        for et in types_to_try:
            dmg = _max_effective_damage_if_attach(
                state,
                bp.card,
                bp.attached_energy,
                getattr(bp, "attached_energy_types", []),
                et,
                opp.active,
                state.current_player,
            )
            if dmg >= opp.active.hp:
                return True
    return False
