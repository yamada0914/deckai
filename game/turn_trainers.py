"""ターン実行用：サポート・グッズ・どうぐの試行順と 1 回試行。"""
from card import is_goods, is_support

from .state import GameState, PlayerState, _log_choice
from .trainers import attach_tool, use_potion, use_support, use_trainer_goods
from .weights import get_goods_use_weight, get_support_use_weight, get_tool_attach_weight

_HAND_REFRESH_SUPPORT_IDS = ("tanpankozou", "hakasenokenkyuu", "hakasenokenkyuufutouhakase", "jixyajjiman", "kihada")
_SUPPORT_IDS_NO_DISCARD = ("nemo", "nemokako", "nemomirai", "kihada")
_SUPPORT_IDS_DISCARD_ALL = ("hakasenokenkyuu", "hakasenokenkyuufutouhakase")


def _try_attach_one_tool(state: GameState) -> bool:
    """手札からどうぐを 1 枚、バトル場またはベンチにつける。つけたら True。重みで付ける先を選ぶ。"""
    p = state.active_player_state()
    weights = state.get_weights_for_player(state.current_player)
    candidates = []
    for i, c in enumerate(p.hand):
        if not is_goods(c) or not getattr(c, "is_tool", False):
            continue
        cond = getattr(c, "tool_condition_type", None)
        if p.active and getattr(p.active, "attached_tool", None) is None:
            if cond is None or getattr(p.active.card, "pokemon_type", None) == cond:
                candidates.append((i, None, p.active.card))
        for bi, bp in enumerate(p.bench):
            if getattr(bp, "attached_tool", None) is not None:
                continue
            if cond is not None and getattr(bp.card, "pokemon_type", None) != cond:
                continue
            candidates.append((i, bi, bp.card))
    if not candidates:
        return False
    best = max(
        candidates,
        key=lambda x: (get_tool_attach_weight(weights, x[2]), x[1] is None),
    )
    hand_i, bench_idx, _ = best
    if attach_tool(state, hand_i, bench_index=bench_idx):
        target_card = p.active.card if bench_idx is None else p.bench[bench_idx].card
        tid = getattr(target_card, "id", None) or getattr(target_card, "name", "")
        if tid:
            _log_choice(state, "tool_attach", card_id=tid)
        return True
    return False


def _support_try_order(p: PlayerState, state: GameState) -> list[int]:
    """
    サポートを試す順序。重み (w_support_use) のみで決める。同点は手札インデックスで安定ソート。
    固定ロジックは使わず、博士の研究の前後なども重みで制御する。
    """
    support_indices = [(i, p.hand[i]) for i in range(len(p.hand)) if is_support(p.hand[i])]
    if not support_indices:
        return []
    weights = state.get_weights_for_player(state.current_player)
    support_indices.sort(key=lambda x: (-get_support_use_weight(weights, x[1]), x[0]))
    return [i for i, c in support_indices]


def _try_support_no_discard_only(state: GameState) -> bool:
    """手札を捨てないサポート（キハダ・ネモ等）を 1 枚だけ試す。使ったら True。エネルギー付与前にドローしたいとき用。"""
    if state.support_used_this_turn:
        return False
    p = state.active_player_state()
    weights = state.get_weights_for_player(state.current_player)
    no_discard = [
        (i, c)
        for i, c in enumerate(p.hand)
        if is_support(c) and getattr(c, "id", "") in _SUPPORT_IDS_NO_DISCARD
    ]
    if not no_discard:
        return False
    no_discard.sort(key=lambda x: -get_support_use_weight(weights, x[1]))
    for i, c in no_discard:
        if use_support(state, i):
            support_card_id = getattr(c, "id", None) or getattr(c, "name", "")
            if support_card_id:
                _log_choice(state, "support", card_id=support_card_id)
            return True
    return False


def _try_erekijienereta(state: GameState) -> bool:
    """エレキジェネレーターを手札から 1 枚だけ試す。ベンチ・山札があれば使用。使ったら True。"""
    p = state.active_player_state()
    if not p.bench or not p.deck:
        return False
    for i, c in enumerate(p.hand):
        if not is_goods(c):
            continue
        cid = getattr(c, "id", None) or ""
        if cid != "erekijienereta" and (getattr(c, "name", "") or "") != "エレキジェネレーター":
            continue
        if use_trainer_goods(state, i):
            _log_choice(state, "goods", card_id=cid or "erekijienereta")
            return True
    return False


def _try_goods_before_hand_refresh(state: GameState) -> bool:
    """手札刷新系サポートの前にどうぐ・グッズを 1 回試す。使ったら True。重みで試す順を決める。"""
    p = state.active_player_state()
    if _try_attach_one_tool(state):
        return True
    weights = state.get_weights_for_player(state.current_player)
    goods_list = [
        (i, c)
        for i, c in enumerate(p.hand)
        if is_goods(c)
        and getattr(c, "effect", None) != "swap_active"
        and not getattr(c, "is_tool", False)
        and not (getattr(c, "id", None) == "haipaboru" and state.turn_count < 2)
    ]
    goods_list.sort(key=lambda x: -get_goods_use_weight(weights, x[1]))
    for i, c in goods_list:
        if getattr(c, "id", None) == "potion" and getattr(c, "effect", None) == "heal":
            used = use_potion(state, i)
        elif getattr(c, "effect", None) == "heal":
            used = False
        else:
            used = use_trainer_goods(state, i)
        if used:
            gid = getattr(c, "id", None) or getattr(c, "name", "")
            if gid:
                _log_choice(state, "goods", card_id=gid)
            return True
    return False
