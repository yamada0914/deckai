"""
盤面評価（サイドレース + 次ターンKO指向）。

優先順位:
  1. サイド差（最重要）
  2. 今ターンKO可否 × サイド枚数（EX=2倍）
  3. 次ターンKO可否 × サイド枚数
  4. ダメージレース（KO圏外でも詰めているほど有利）
  5. 手札・エネルギー（補助的な小ボーナス）
"""
from card import is_energy
from .state import GameState, PRIZE_COUNT, BattlePokemon, _prizes_for_ko
from .damage import _max_effective_damage_for_attacker, _max_effective_damage_if_attach

# --- 重みパラメータ ---
PRIZE_WEIGHT = 4.0          # サイド差 1枚あたり
MY_CAN_KO_BONUS = 2.5       # 今ターンKO可能（×サイド枚数）
OPP_CAN_KO_PENALTY = 2.0    # 相手が今ターンKO可能（×サイド枚数）
NEXT_TURN_KO_BONUS = 1.5    # 次ターンKO圏内（×サイド枚数）
NEXT_TURN_KO_THREATENED = 1.2  # 相手が次ターンKO圏内（×サイド枚数）
DAMAGE_RACE_WEIGHT = 0.8    # ダメージレース比率（0~1）
HAND_WEIGHT = 0.2           # 手札枚数
ENERGY_WEIGHT = 0.2         # 付与エネルギー合計


def _my_energy_total(state: GameState, for_player: int) -> int:
    """指定プレイヤーのバトル場＋ベンチの付与エネルギー合計。"""
    p = state.players[for_player]
    total = 0
    if p.active:
        total += getattr(p.active, "attached_energy", 0)
    for bp in p.bench:
        total += getattr(bp, "attached_energy", 0)
    return total


def _can_ko_with_one_more_energy(
    state: GameState,
    attacker_bp: BattlePokemon,
    defender_bp: BattlePokemon,
    player_index: int,
) -> bool:
    """
    エネルギーを1枚追加したとき（次ターン）にKOできるか。
    手札にエネルギーがあればそのタイプを、なければポケモン自身のタイプを仮定する。
    """
    p = state.players[player_index]

    # 手札のエネルギータイプを収集
    energy_types: set[str] = set()
    for c in p.hand:
        if is_energy(c):
            etype = getattr(c, "energy_type", None) or "colorless"
            energy_types.add(etype)

    # 手札にエネルギーがない場合はポケモン自身のタイプを仮定
    if not energy_types:
        ptype = getattr(attacker_bp.card, "pokemon_type", None)
        energy_types = {ptype} if ptype else {"colorless"}

    current_types = getattr(attacker_bp, "attached_energy_types", [])
    for etype in energy_types:
        dmg = _max_effective_damage_if_attach(
            state,
            attacker_bp.card,
            attacker_bp.attached_energy,
            list(current_types),
            etype,
            defender_bp,
            player_index,
        )
        if dmg >= defender_bp.hp:
            return True
    return False


def evaluate_board(state: GameState, for_player: int) -> float:
    """
    指定プレイヤー視点の盤面スコア。
    サイドレース・KO脅威・次ターンKO・ダメージレースを重み付きで加算。
    """
    my = state.players[for_player]
    opp = state.players[1 - for_player]

    # 1. サイド差（最重要）
    my_prizes_taken = PRIZE_COUNT - len(my.prize_pile)
    opp_prizes_taken = PRIZE_COUNT - len(opp.prize_pile)
    score = PRIZE_WEIGHT * (my_prizes_taken - opp_prizes_taken)

    # 2. KO脅威 / 次ターンKO / ダメージレース
    if my.active and opp.active and my.active.hp > 0 and opp.active.hp > 0:
        my_dmg = _max_effective_damage_for_attacker(state, my.active, opp.active, for_player)
        opp_dmg = _max_effective_damage_for_attacker(state, opp.active, my.active, 1 - for_player)

        my_can_ko = my_dmg >= opp.active.hp
        opp_can_ko = opp_dmg >= my.active.hp
        opp_prize_val = _prizes_for_ko(opp.active)  # EX=2, メガ=3, それ以外=1
        my_prize_val = _prizes_for_ko(my.active)

        # 今ターンKO
        if my_can_ko:
            score += MY_CAN_KO_BONUS * opp_prize_val
        elif _can_ko_with_one_more_energy(state, my.active, opp.active, for_player):
            # 次ターンKO圏内（今ターンKO済みでない場合のみ加算）
            score += NEXT_TURN_KO_BONUS * opp_prize_val

        if opp_can_ko:
            score -= OPP_CAN_KO_PENALTY * my_prize_val
        elif _can_ko_with_one_more_energy(state, opp.active, my.active, 1 - for_player):
            score -= NEXT_TURN_KO_THREATENED * my_prize_val

        # ダメージレース（KO圏外でも詰めているほど有利）
        if not my_can_ko and opp.active.hp > 0:
            score += DAMAGE_RACE_WEIGHT * (my_dmg / opp.active.hp)
        if not opp_can_ko and my.active.hp > 0:
            score -= DAMAGE_RACE_WEIGHT * (opp_dmg / my.active.hp)

    # 3. 手札（補助）
    score += HAND_WEIGHT * len(my.hand)

    # 4. エネルギー（補助: KO可能性が上がれば自然に評価されるため小さく）
    score += ENERGY_WEIGHT * _my_energy_total(state, for_player)

    # 5. ドラパルトexデッキ: バトル場ドロンチが唯一のドラパルトラインでベンチに他ラインなし → 大ペナルティ
    # KOされるとアタッカーラインが全滅し勝ち筋がなくなる
    from .deck_strategies import is_dragapult_deck_for_player as _is_drapa_eval
    if _is_drapa_eval(state, for_player) and my.active:
        _active_name = (getattr(my.active.card, "name", "") or "").strip()
        _drapa_line_names = {"ドラメシヤ", "ドロンチ", "ドラパルトex"}
        if _active_name == "ドロンチ":
            _bench_has_drapa_line = any(
                (getattr(bp.card, "name", "") or "").strip() in _drapa_line_names
                for bp in my.bench
            )
            if not _bench_has_drapa_line:
                score -= 10.0  # 大きなペナルティ（PRIZE_WEIGHT=4.0基準で2.5サイド分相当）

    return score
