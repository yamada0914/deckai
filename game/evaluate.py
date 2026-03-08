"""
盤面評価（中級者 AI 用）。
サイド・手札・エネルギー・「次のターンで KO できる/される」を考慮したスコア。
"""
from .state import GameState, PRIZE_COUNT
from .damage import _max_effective_damage_for_attacker

PRIZE_WEIGHT = 3.0
HAND_WEIGHT = 0.5
ENERGY_WEIGHT = 0.8
OPP_CAN_KO_PENALTY = 2.0
MY_CAN_KO_BONUS = 2.0


def _my_energy_total(state: GameState, for_player: int) -> int:
    """指定プレイヤーのバトル場＋ベンチの付与エネルギー合計。"""
    p = state.players[for_player]
    total = 0
    if p.active:
        total += getattr(p.active, "attached_energy", 0)
    for bp in p.bench:
        total += getattr(bp, "attached_energy", 0)
    return total


def evaluate_board(state: GameState, for_player: int) -> float:
    """
    指定プレイヤー視点の盤面スコア。
    サイド取得数・手札・エネルギー・「次ターンで KO できる/される」を重み付きで加算。
    """
    my = state.players[for_player]
    opp = state.players[1 - for_player]

    my_prizes_taken = PRIZE_COUNT - len(my.prize_pile)
    opp_prizes_taken = PRIZE_COUNT - len(opp.prize_pile)
    score = PRIZE_WEIGHT * (my_prizes_taken - opp_prizes_taken)

    score += HAND_WEIGHT * len(my.hand)
    score += ENERGY_WEIGHT * _my_energy_total(state, for_player)

    my_active = my.active
    opp_active = opp.active
    my_can_ko = False
    opp_can_ko = False
    if my_active and opp_active and my_active.hp > 0 and opp_active.hp > 0:
        my_dmg = _max_effective_damage_for_attacker(state, my_active, opp_active, for_player)
        my_can_ko = my_dmg >= opp_active.hp
        opp_dmg = _max_effective_damage_for_attacker(state, opp_active, my_active, 1 - for_player)
        opp_can_ko = opp_dmg >= my_active.hp

    if opp_can_ko:
        score -= OPP_CAN_KO_PENALTY
    if my_can_ko:
        score += MY_CAN_KO_BONUS

    return score
