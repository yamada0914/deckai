"""
サポート選択ポリシー。

行動空間（固定次元 1 + len(KNOWN_SUPPORT_IDS)）:
  0: NOOP（このターンはサポートを使わない）
  1..N: KNOWN_SUPPORT_IDS[i-1] に対応するサポートを使う

合成: logits = heuristic_scale * heuristic + λ_eff * NN
マスク: 手札にないカード ID は非合法（NOOP は常に合法）

ε-greedy: support_epsilon → support_epsilon_end へ線形／指数減衰（horizon 指定時）

学習用ログ type: support_policy（NOOP 含む全分岐）。
turn.py の「メインサポートスロット」（エネ付与後）からのみ呼ぶ。
_try_support_no_discard_only（エネ付与前の kihada/nemo 枠）は置き換えない。
"""

from __future__ import annotations

import math
import random
from typing import Any

from card import is_energy, is_support

from .encoders import encode_state_basic
from .state import BENCH_SIZE, GameState, PlayerState, _log_choice, rules_only_for_player
from .turn_trainers import (
    _SUPPORT_ID_ANGOUMANIANOKAIDOKU,
    _SUPPORT_IDS_HAND_REFRESH_FIRST,
    hand_has_remaining_shuffle_effect_for_angou,
)
from .weights import get_support_use_weight

# 既知のサポート card_id 固定リスト（順序変更禁止 — モデルの action dim に対応）
# 新カード追加時はリスト末尾に追加し、既存モデルを再学習すること。
KNOWN_SUPPORT_IDS: tuple[str, ...] = (
    "tanpankozou",
    "hakasenokenkyuu",
    "hakasenokenkyuufutouhakase",   # turn_trainers.py で使われている表記
    "hakasenokenkixyuufutouhakase", # card/data.py の定義上の表記
    "jixyajjiman",
    "kihada",
    "nemo",
    "nemokako",
    "nemomirai",
    "riirienokesshin",
    "zeiyu",
    "angoumanianokaidoku",
    "bosunoshirei",
    "mitsurunoomoiyari",
    "hikari",
    "akamatsu",
    "buraia",
    "meinohagemashi",
)

# action id: 0=NOOP, 1..N=KNOWN_SUPPORT_IDS[i-1]
SUPPORT_ACTION_DIM: int = 1 + len(KNOWN_SUPPORT_IDS)

_SUPPORT_ID_TO_ACTION: dict[str, int] = {sid: i + 1 for i, sid in enumerate(KNOWN_SUPPORT_IDS)}

_NOOP_BIAS = -0.05
_HAND_REFRESH_BIAS = 5.0
_ANGOU_DEFER_PENALTY = -10.0
_MASK_LOGIT_ADD = 1e4


def support_action_id_from_card_id(card_id: str) -> int:
    """カード ID → action id。未知のカードは 0（NOOP 扱い）。"""
    return _SUPPORT_ID_TO_ACTION.get(card_id, 0)


def legal_support_mask(state: GameState) -> list[bool]:
    """
    各 action の合法性マスク。
    - Action 0 (NOOP): 常に合法
    - Action 1..N: 対応カードが手札にあれば合法（use_support 側の詳細条件は実行時チェック）
    """
    mask = [True] + [False] * len(KNOWN_SUPPORT_IDS)
    p = state.active_player_state()
    hand_ids: set[str] = {getattr(c, "id", "") or "" for c in (p.hand or []) if is_support(c)}
    for i, sid in enumerate(KNOWN_SUPPORT_IDS):
        if sid in hand_ids:
            mask[1 + i] = True
    return mask


def _can_ko_opponent_active(state: GameState) -> bool:
    """自分のバトル場ポケモンが相手のバトル場を KO できるかを返す。"""
    from .damage import _our_max_effective_damage
    opp = state.defending_player_state()
    if not opp.active:
        return False
    return _our_max_effective_damage(state) >= opp.active.hp


def _can_ko_any_bench(state: GameState) -> bool:
    """ボスの指令で引き出せば KO できるベンチポケモンがいるかを返す。"""
    from .damage import _max_effective_damage_for_attacker
    p = state.active_player_state()
    opp = state.defending_player_state()
    if not p.active:
        return False
    for bp in (opp.bench or []):
        if bp is None:
            continue
        dmg = _max_effective_damage_for_attacker(state, p.active, bp, state.current_player)
        if dmg >= bp.hp:
            return True
    return False


def heuristic_logits_support(state: GameState) -> list[float]:
    """
    条件付きヒューリスティック — 盤面の状況に応じてスコアを計算する。

    各サポートカードに状況依存のバイアスを与える。
    NNが未学習の初期でも妥当な判断になるよう設計。

    リーリエの決心: 手札が少ないほど高スコア（ドロー価値）
    ボスの指令:     KO 確定時に最高スコア、なければ低め
    ジャッジマン:   相手の手札が多いとき / 自分の手札が少ないとき
    ゼイユ:         手札が少ないとき（5枚になるまでドロー）
    暗号マニア:     シャッフル系が残っていなければ高スコア
    ミツル:         ベンチに出したばかりのポケモンがいるとき
    """
    p = state.active_player_state()
    opp = state.defending_player_state()
    weights = state.get_weights_for_player(state.current_player)
    angou_defer_needed = hand_has_remaining_shuffle_effect_for_angou(state, p)

    hand_size = len(p.hand or [])
    opp_hand_size = len(opp.hand or [])
    turn = int(getattr(state, "turn_count", 0))
    early_game = turn <= 4

    # KO 判定（ボスの指令用、遅延評価）
    ko_active = None
    ko_bench = None

    logits = [_NOOP_BIAS] + [0.0] * len(KNOWN_SUPPORT_IDS)

    for i, sid in enumerate(KNOWN_SUPPORT_IDS):
        card = None
        for c in (p.hand or []):
            if is_support(c) and (getattr(c, "id", "") or "") == sid:
                card = c
                break
        if card is None:
            logits[1 + i] = -_MASK_LOGIT_ADD
            continue

        score = 0.0

        # ---- リーリエの決心（ドロー系） ----
        if sid == "riirienokesshin":
            # 手札3枚以下で最強、5枚以上では弱い
            if hand_size <= 2:
                score += 5.0
            elif hand_size <= 4:
                score += 3.0
            else:
                score += 0.5

        # ---- 博士の研究系（ドロー系） ----
        elif sid in ("hakasenokenkyuu", "hakasenokenkyuufutouhakase", "hakasenokenkixyuufutouhakase"):
            if hand_size <= 3:
                score += 4.0
            elif hand_size <= 5:
                score += 2.0
            else:
                score += 0.3

        # ---- ジャッジマン（妨害＋自分リセット） ----
        elif sid == "jixyajjiman":
            # 相手の手札が多いとき妨害効果大
            if opp_hand_size >= 6:
                score += 3.0
            elif opp_hand_size >= 4:
                score += 1.5
            # 自分の手札が少ないときもリフレッシュ効果
            if hand_size <= 2:
                score += 1.5
            elif hand_size <= 4:
                score += 0.5

        # ---- ボスの指令（相手ベンチを引っ張る） ----
        elif sid == "bosunoshirei":
            # 相手バトル場を既に倒せるなら、ボスを温存（そのまま殴ればよい）
            from .damage import _max_effective_damage_for_attacker
            can_ko_active = False
            opp_protected = getattr(opp.active, "protected_next_opponent_turn", False) if opp.active else False
            # なぐってかくれる等で相手がダメージ無効 → ボスの指令でベンチを引っ張る最優先
            if opp_protected and opp.bench:
                score += 15.0  # 他のサポートより大幅に優先
            elif opp_protected and not opp.bench:
                score += -1.0  # ベンチがいないなら意味なし
            else:
                if p.active and opp.active and opp.active.hp is not None and opp.active.hp > 0 and not opp_protected:
                    our_dmg = _max_effective_damage_for_attacker(state, p.active, opp.active, state.current_player)
                    can_ko_active = our_dmg >= opp.active.hp
                if can_ko_active:
                    score += -5.0  # バトル場を倒せるのにボスは無駄遣い
                else:
                    # ベンチに倒せるポケモンがいるなら高スコア
                    if ko_bench is None:
                        ko_bench = _can_ko_any_bench(state)
                    if ko_bench:
                        score += 5.0
                        has_weak_bench = any(
                            bp is not None and bp.hp <= 70
                            for bp in (opp.bench or [])
                        )
                        if has_weak_bench and early_game:
                            score += 1.0
                        # サイド残り少ない時のボスの指令プレッシャーボーナス
                        # ゼイユ/リーリエ等のドローサポートより優先させて捨てられないようにする
                        our_prizes = len(p.prize_pile) if hasattr(p, "prize_pile") else 99
                        if our_prizes <= 3:
                            score += 10.0  # サイド残り少ない＋KO可能 → 勝利に直結
                        elif our_prizes <= 4:
                            score += 3.0
                    else:
                        # ベンチのKO可能ポケモンがいなくても、
                        # 相手バトル場が強力で自分が次ターンKOされるなら、
                        # ボスで弱いベンチポケモンを引っ張って時間稼ぎ
                        from .damage import _opponent_max_effective_damage
                        _opp_dmg = _opponent_max_effective_damage(state)
                        _would_be_koed = p.active and _opp_dmg > 0 and p.active.hp <= _opp_dmg
                        # 相手がダメージを与えてくる（毎ターン削られる）
                        _opp_is_attacking = _opp_dmg > 0
                        _has_weak_bench_to_stall = opp.bench and any(
                            bp is not None
                            and getattr(bp, "attached_energy", 0) == 0
                            and getattr(bp.card, "retreat_cost", 1) >= 1
                            for bp in opp.bench
                        )
                        if _would_be_koed and _has_weak_bench_to_stall:
                            score += 5.0  # 次ターンKO → 緊急ボス
                        elif _opp_is_attacking and _has_weak_bench_to_stall:
                            score += 3.0  # 毎ターンダメージ → 時間稼ぎボス
                        elif opp.bench:
                            score += 0.5
                        else:
                            score += -1.0

        # ---- ゼイユ（5枚になるまでドロー） ----
        elif sid == "zeiyu":
            draw_count = max(0, 5 - hand_size)
            if draw_count >= 3:
                score += 3.5
            elif draw_count >= 2:
                score += 2.0
            elif draw_count >= 1:
                score += 0.8
            else:
                score += -5.0  # 手札5枚以上 → 引く枚数なし、デッキを減らすだけ
            # Fix K: ボスの指令が手札にあり、サイド残り少ない → ゼイユで捨てると勝ちを逃す
            _hand_has_boss = any(
                (getattr(c, "id", "") or "") == "bosunoshirei"
                for c in (p.hand or []) if c is not card
            )
            if _hand_has_boss:
                our_prizes = len(p.prize_pile) if hasattr(p, "prize_pile") else 99
                if our_prizes <= 4:
                    score -= 10.0
            # メガルカリオexが手札にあり場にいない → ゼイユで捨てると取り返しがつかない
            _hand_has_mega = any(
                "メガルカリオ" in (getattr(c, "name", "") or "")
                for c in (p.hand or []) if c is not card  # ゼイユ自身は除外
            )
            if _hand_has_mega:
                _mega_on_field = False
                if p.active and "メガルカリオ" in (getattr(p.active.card, "name", "") or ""):
                    _mega_on_field = True
                if not _mega_on_field:
                    for bp in (p.bench or []):
                        if "メガルカリオ" in (getattr(bp.card, "name", "") or ""):
                            _mega_on_field = True
                            break
                if not _mega_on_field:
                    score -= 20.0  # メガルカリオexをトラッシュに送るのは致命的

        # ---- 暗号マニアの解読（デッキトップ操作） ----
        elif sid == "angoumanianokaidoku":
            if angou_defer_needed:
                score += -8.0  # シャッフル系が残っている → 後回し
            else:
                score += 2.5  # コンボ成立 → 有効

        # ---- ミツルの思いやり（ベンチに出したばかりのポケモンをすぐ進化） ----
        elif sid == "mitsurunoomoiyari":
            has_new_bench = any(
                getattr(bp, "put_on_bench_this_turn", False)
                for bp in (p.bench or []) if bp is not None
            )
            if has_new_bench:
                score += 3.0
            else:
                score += 0.2

        # ---- キハダ / ネモ系（汎用ドロー） ----
        elif sid in ("kihada", "nemo", "nemokako", "nemomirai"):
            if hand_size <= 3:
                score += 2.5
            elif hand_size <= 5:
                score += 1.0
            else:
                score += 0.2

        # ---- タンパン小僧 ----
        elif sid == "tanpankozou":
            if ko_active is None:
                ko_active = _can_ko_opponent_active(state)
            if not ko_active:
                score += 1.0  # 壁を剥がす
            else:
                score += -0.5  # そのまま倒せるなら不要

        # ---- ヒカリ（汎用ドロー） ----
        elif sid == "hikari":
            if hand_size <= 3:
                score += 2.0
            elif hand_size <= 5:
                score += 1.0
            else:
                score += 0.2

        # ---- アカマツ（山札から2タイプのエネルギー取得） ----
        elif sid == "akamatsu":
            from .deck_strategies import is_dragapult_deck_for_player as _is_drapa_sp
            if _is_drapa_sp(state, state.current_player):
                # ドラパルトexデッキ: アカマツは炎+超の確保→ファントムダイブ発動に直結
                # 場にドラパルトex/ドロンチ/ドラメシヤがいて、エネルギーが足りないなら超高スコア
                _has_drapa_target = False
                _drapa_needs_energy = False
                all_bp = ([p.active] if p.active else []) + list(p.bench or [])
                for bp in all_bp:
                    bpn = (getattr(bp.card, "name", "") or "").strip()
                    if bpn in ("ドラパルトex", "ドロンチ", "ドラメシヤ"):
                        _has_drapa_target = True
                        if bpn == "ドラパルトex" and (getattr(bp, "attached_energy", 0) or 0) < 3:
                            _drapa_needs_energy = True
                        elif bpn in ("ドロンチ", "ドラメシヤ") and (getattr(bp, "attached_energy", 0) or 0) < 2:
                            _drapa_needs_energy = True
                if _drapa_needs_energy:
                    score += 8.0  # エネ加速最優先
                elif _has_drapa_target:
                    score += 4.0  # ターゲットがいるので有用
                else:
                    score += 1.0
            else:
                score += 1.5  # 非ドラパルトデッキでも最低限有用

        # ---- メイのはげまし（トラッシュからStage2にエネ2枚） ----
        elif sid == "meinohagemashi":
            from .deck_strategies import is_dragapult_deck_for_player as _is_drapa_sp2
            if _is_drapa_sp2(state, state.current_player):
                # 条件チェック: サイドが相手より多い（遅れている）
                _behind_on_prizes = len(p.prize_pile) > len(opp.prize_pile)
                # Stage2がいるか
                _has_stage2 = any(
                    getattr(bp.card, "evolution_stage", "") == "stage2"
                    for bp in (([p.active] if p.active else []) + list(p.bench or []))
                )
                # トラッシュにエネがあるか
                _has_trash_energy = any(is_energy(c) for c in (p.discard or []))

                if _behind_on_prizes and _has_stage2 and _has_trash_energy:
                    score += 10.0  # 条件完全一致 → 最優先
                elif _behind_on_prizes and _has_stage2:
                    score += 3.0  # エネがトラッシュにないが、この先溜まるかも
                elif _behind_on_prizes:
                    score += 1.0  # Stage2がまだいない
                else:
                    score += 0.0  # 条件不成立
            else:
                score += 1.0

        # ---- ブライア（テラスタルKOでサイド+1） ----
        elif sid == "buraia":
            if len(opp.prize_pile) == 2:
                score += 3.0  # 条件一致
            else:
                score += -5.0  # 条件不一致 → 使えない

        # 学習重み加算
        score += float(get_support_use_weight(weights, card))
        logits[1 + i] = score

    return logits


# ---- ユーティリティ（retreat_before_attack.py と同パターン） ----

def _apply_legal_mask(logits: list[float], mask: list[bool]) -> list[float]:
    return [logits[i] + (float(mask[i]) - 1.0) * _MASK_LOGIT_ADD for i in range(len(logits))]


def _top2_gap_masked(logits_masked: list[float], mask: list[bool]) -> float:
    vals = [logits_masked[i] for i in range(len(logits_masked)) if i < len(mask) and mask[i]]
    if len(vals) < 2:
        return float("inf") if len(vals) == 1 else 0.0
    vals.sort(reverse=True)
    return float(vals[0] - vals[1])


def _entropy_from_logits(logits: list[float]) -> float:
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps)
    if s <= 0:
        return 0.0
    ent = 0.0
    for e in exps:
        prob = e / s
        if prob > 0.0:
            ent -= prob * math.log(prob + 1e-12)
    return ent


def _mean_abs(xs: list[float]) -> float:
    if not xs:
        return 0.0
    return sum(abs(x) for x in xs) / float(len(xs)) + 1e-6


def _combine_heuristic_and_nn_logits(
    h_scaled: list[float],
    nn_logits: list[float],
    lam: float,
    *,
    normalize: bool,
) -> list[float]:
    n = len(h_scaled)
    if lam <= 0.0:
        return list(h_scaled)
    if not normalize:
        return [
            h_scaled[i] + lam * (float(nn_logits[i]) if i < len(nn_logits) else 0.0)
            for i in range(n)
        ]
    ma_h = _mean_abs(h_scaled)
    nn_f = [float(nn_logits[i]) if i < len(nn_logits) else 0.0 for i in range(n)]
    ma_n = _mean_abs(nn_f)
    return [h_scaled[i] / ma_h + lam * nn_f[i] / ma_n for i in range(n)]


def _sample_from_logits(logits: list[float]) -> int:
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps)
    if s <= 0:
        return 0
    r = random.random() * s
    acc = 0.0
    for i, e in enumerate(exps):
        acc += e
        if r <= acc:
            return i
    return len(exps) - 1


def _sigmoid(x: float) -> float:
    if x >= 30.0:
        return 1.0
    if x <= -30.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def policy_schedule_progress(state: GameState) -> float:
    h = int(getattr(state, "policy_schedule_horizon_steps", 0))
    if h <= 0:
        return 0.0
    return min(1.0, float(getattr(state, "total_env_steps", 0)) / float(h))


def effective_heuristic_scale(state: GameState) -> float:
    if int(getattr(state, "policy_schedule_horizon_steps", 0)) <= 0:
        return 1.0
    p = policy_schedule_progress(state)
    decay = float(getattr(state, "support_heuristic_decay", 0.7))
    floor_ = float(getattr(state, "support_heuristic_scale_floor", 0.3))
    return max(floor_, 1.0 - p * decay)


def effective_pi_support_lambda(state: GameState) -> float:
    """
    policy_schedule_horizon_steps > 0 のときは ε・ヒューリスティックと同じ progress を使用。
    linear: λ_max * min(1, step/warmup)
    sigmoid: λ_max * sigmoid((step/warmup)*6 - 3)
    warmup_steps==0 のときはスケジュールなし（常に λ_max）。
    """
    lam_max = float(getattr(state, "pi_support_lambda", 0.5))
    horizon = int(getattr(state, "policy_schedule_horizon_steps", 0))
    if horizon > 0:
        p = policy_schedule_progress(state)
        sched = str(getattr(state, "pi_support_lambda_schedule", "linear")).lower()
        if sched == "sigmoid":
            return lam_max * _sigmoid(p * 6.0 - 3.0)
        return lam_max * min(1.0, p)
    warmup = int(getattr(state, "pi_support_lambda_warmup_steps", 0))
    step = int(getattr(state, "total_env_steps", 0))
    if warmup <= 0:
        return lam_max
    t = float(step) / float(warmup)
    sched = str(getattr(state, "pi_support_lambda_schedule", "linear")).lower()
    if sched == "sigmoid":
        return lam_max * _sigmoid(t * 6.0 - 3.0)
    return lam_max * min(1.0, t)


def effective_support_epsilon(state: GameState) -> float:
    """ε0 から ε_end へ線形または指数減衰（horizon なしは ε0 のみ）。"""
    eps0 = float(getattr(state, "support_epsilon", 0.0))
    eps1 = float(getattr(state, "support_epsilon_end", 0.0))
    if int(getattr(state, "policy_schedule_horizon_steps", 0)) <= 0:
        return eps0
    p = policy_schedule_progress(state)
    sched = str(getattr(state, "support_epsilon_schedule", "linear")).lower()
    if sched == "exp":
        k = float(getattr(state, "support_epsilon_decay_k", 3.0))
        return eps1 + (eps0 - eps1) * math.exp(-k * p)
    return eps1 + (eps0 - eps1) * (1.0 - p)


def support_policy_decision(state: GameState) -> tuple[int, dict[str, Any]]:
    """
    サポート選択の決定。

    Returns
    -------
    (action_id, debug_dict)
      action_id: 0=NOOP, 1..N=KNOWN_SUPPORT_IDS[action_id-1] を使う
    """
    mask = legal_support_mask(state)
    h_raw = heuristic_logits_support(state)
    h_scale = effective_heuristic_scale(state)
    h_scaled = [float(x) * h_scale for x in h_raw]
    lam_eff = effective_pi_support_lambda(state)

    # Q-model（回帰）でサポート選択
    if rules_only_for_player(state):
        q_path = None
    else:
        q_by = getattr(state, "q_support_model_path_by_player", [None, None])
        q_path = q_by[state.current_player] if q_by[state.current_player] is not None else getattr(state, "q_support_model_path", None)

    if q_path:
        from .encoders import encode_state_v2, _can_attack_now

        # Advantage モデル (.meta.json に action_dim がある) か Q-value モデルかを自動判別
        _meta_path = str(q_path).replace(".pt", ".meta.json")
        _is_advantage = False
        try:
            import json as _json
            with open(_meta_path, "r") as _mf:
                _meta = _json.load(_mf)
            if "action_dim" in _meta and "state_dim" in _meta:
                _is_advantage = True
        except (FileNotFoundError, _json.JSONDecodeError):
            pass

        if _is_advantage:
            from .q_models import load_advantage_model_pt
            q_model = load_advantage_model_pt(q_path)
        else:
            from .q_models import load_q_value_model_pt
            q_model = load_q_value_model_pt(q_path)

        vec = encode_state_v2(state, state.current_player)
        q_vals_raw = q_model.predict_all(vec)

        # z-score 正規化
        n_q = len(q_vals_raw)
        q_mean = sum(q_vals_raw) / n_q if n_q > 0 else 0.0
        q_var = sum((v - q_mean) ** 2 for v in q_vals_raw) / n_q if n_q > 0 else 0.0
        q_std = q_var ** 0.5
        q_normed = [(v - q_mean) / (q_std + 1e-8) for v in q_vals_raw]

        # 状態依存 λ: 攻撃可能なら小さく（ヒューリスティック優先）、
        # セットアップ段階なら大きく（Qを信頼してドロー・進化を促す）
        if bool(getattr(state, "q_support_lambda_dynamic", False)):
            cur_p = state.active_player_state()
            attack_ready = _can_attack_now(cur_p.active) > 0.0 if cur_p.active else False
            if attack_ready:
                q_w = float(getattr(state, "q_support_lambda_attack_ready", 0.3))
            else:
                q_w = float(getattr(state, "q_support_lambda_setup", 1.2))
        else:
            q_w = float(getattr(state, "q_support_lambda", 3.0))

        logits = [
            h_scaled[i] + q_w * (q_normed[i] if i < len(q_normed) else 0.0)
            for i in range(len(h_scaled))
        ]
    else:
        # π-model（ポリシー）でサポート選択（従来）
        if rules_only_for_player(state):
            pi_path = None
        else:
            by_player = getattr(state, "pi_support_model_path_by_player", [None, None])
            pi_path = by_player[state.current_player] if by_player[state.current_player] is not None else getattr(state, "pi_support_model_path", None)
        if pi_path:
            from .pi_models import load_pi_model_pt
            from .encoders import encode_state_v2

            pi = load_pi_model_pt(pi_path)
            if pi.encoder_name == "opening":
                vec = encode_state_v2(state, state.current_player)
            else:
                vec = encode_state_basic(state, state.current_player)
            nn_logits = pi.predict_logits_one(vec)
            do_norm = bool(getattr(state, "support_normalize_logits", True))
            logits = _combine_heuristic_and_nn_logits(h_scaled, list(nn_logits), lam_eff, normalize=do_norm)
        else:
            logits = list(h_scaled)

    logits_pre_mask = list(logits)
    logits_masked = _apply_legal_mask(logits, mask)
    entropy = _entropy_from_logits(logits_masked)
    top2_gap = _top2_gap_masked(logits_masked, mask)
    sched_p = policy_schedule_progress(state)
    eps_eff = effective_support_epsilon(state)
    deterministic = getattr(state, "support_deterministic", True)

    if deterministic:
        idx = int(max(range(len(logits_masked)), key=lambda i: logits_masked[i]))
    else:
        idx = _sample_from_logits(logits_masked)

    eps_used = False
    if eps_eff > 0.0 and random.random() < eps_eff:
        idx = _sample_from_logits(logits_masked)
        eps_used = True

    debug: dict[str, Any] = {
        "mask": mask,
        "logits_pre_mask": logits_pre_mask,
        "logits_masked": logits_masked,
        "pi_support_lambda_effective": lam_eff,
        "support_policy_entropy": entropy,
        "support_policy_epsilon_greedy": eps_used,
        "support_policy_top2_gap": top2_gap,
        "support_policy_schedule_progress": sched_p,
        "support_policy_heuristic_scale": h_scale,
        "support_policy_epsilon_effective": eps_eff,
    }
    return idx, debug


def _log_support_policy_choice(
    state: GameState,
    idx: int,
    dbg: dict[str, Any],
    *,
    card_id: str | None,
) -> None:
    _log_choice(
        state,
        "support_policy",
        card_id=card_id,
        support_policy_action_id=idx,
        support_policy_mask=dbg["mask"],
        support_policy_logits_pre_mask=dbg["logits_pre_mask"],
        support_policy_logits=dbg["logits_masked"],
        pi_support_lambda_effective=dbg["pi_support_lambda_effective"],
        support_policy_entropy=dbg["support_policy_entropy"],
        support_policy_epsilon_greedy=dbg["support_policy_epsilon_greedy"],
        support_policy_top2_gap=dbg.get("support_policy_top2_gap"),
        support_policy_schedule_progress=dbg.get("support_policy_schedule_progress"),
        support_policy_heuristic_scale=dbg.get("support_policy_heuristic_scale"),
        support_policy_epsilon_effective=dbg.get("support_policy_epsilon_effective"),
    )


def try_support_policy(state: GameState) -> bool:
    """
    メインサポートスロット（エネ付与後）でサポートを 1 枚選んで使う。

    - NOOP (idx==0) → 何もせず False を返す（choice_log に記録する）
    - それ以外 → 対応カードを use_support で実行
      - 失敗時は logits 降順で次の合法 action へ fallback
      - 全 fallback 失敗 → NOOP として False を返す
    - 手札にサポートが 1 枚もなければ呼ばない（呼び出し元で確認すること）

    Returns True なら使用成功。choice_log は常に記録（NOOP 含む）。
    """
    from .trainers import use_support

    idx, dbg = support_policy_decision(state)

    if idx == 0:
        _log_support_policy_choice(state, 0, dbg, card_id=None)
        return False

    # 選ばれた action id のカードを手札から使う
    target_sid = KNOWN_SUPPORT_IDS[idx - 1]
    p = state.active_player_state()
    for hand_i, c in enumerate(p.hand or []):
        if not is_support(c) or (getattr(c, "id", "") or "") != target_sid:
            continue
        if use_support(state, hand_i):
            _log_support_policy_choice(state, idx, dbg, card_id=target_sid)
            return True
        break  # 条件不成立：次候補へ

    # use_support 失敗 → logits 降順で fallback
    logits_masked = dbg["logits_masked"]
    mask = dbg["mask"]
    ordered = sorted(range(len(logits_masked)), key=lambda i: -logits_masked[i])
    for fallback_idx in ordered:
        if fallback_idx == idx:
            continue  # すでに試した
        if not mask[fallback_idx]:
            continue
        if fallback_idx == 0:
            # NOOP が最善
            _log_support_policy_choice(state, 0, dbg, card_id=None)
            return False
        fallback_sid = KNOWN_SUPPORT_IDS[fallback_idx - 1]
        p = state.active_player_state()
        for hand_i, c in enumerate(p.hand or []):
            if not is_support(c) or (getattr(c, "id", "") or "") != fallback_sid:
                continue
            if use_support(state, hand_i):
                _log_support_policy_choice(state, fallback_idx, dbg, card_id=fallback_sid)
                return True
            break

    # 全 fallback 失敗 → NOOP
    _log_support_policy_choice(state, 0, dbg, card_id=None)
    return False
