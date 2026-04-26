"""
エネルギー付与ポリシー。

行動空間（固定次元 BENCH_SIZE + 1 = 6）:
  0: バトル場（active）
  1..BENCH_SIZE: ベンチ [0..BENCH_SIZE-1]

NOOP なし: 「付与するか否か」は turn_energy.py 側が決める。
このポリシーは「どこに付与するか」だけを担当。

⚠️ 将来の拡張メモ:
  現在 action = target_index のみ。
  特殊エネが増えた場合は action = (target_index, energy_type) に拡張すること。
  → energy_type は既にログに記録済みなので、拡張時は action_dim を増やすだけで対応可。

合成: logits = heuristic_scale * heuristic + λ_eff * NN
マスク: 候補にないポケモン index は非合法
ε-greedy: energy_policy_epsilon → energy_policy_epsilon_end へ線形／指数減衰

学習用ログ type: energy_attach（既存 type を継承）
extra fields: energy_policy_action_id, energy_policy_mask, energy_policy_logits,
              energy_policy_logits_pre_mask, energy_policy_top2_gap,
              energy_policy_entropy, energy_policy_epsilon_greedy,
              energy_policy_heuristic_scale, energy_policy_schedule_progress,
              energy_policy_epsilon_effective, pi_energy_policy_lambda_effective,
              energy_type, energy_before, energy_after, can_attack_after, can_kill_after
"""

from __future__ import annotations

import math
import random
from typing import Any

from .state import BENCH_SIZE, GameState, _log_choice, _is_first_player_first_turn, rules_only_for_player
from .weights import get_energy_attach_weight

# action 0 = active, action 1..BENCH_SIZE = bench[i-1]
ENERGY_ATTACH_ACTION_DIM: int = 1 + BENCH_SIZE

_DMG_SCALE = 0.1
_ACTIVE_BONUS = 1.0
_MASK_LOGIT_ADD = 1e4


# ---- ポケモン判定ヘルパー ----

def _is_lunatone(card) -> bool:
    n = (getattr(card, "name", "") or "").strip()
    cid = (getattr(card, "id", "") or "").strip()
    return n == "ルナトーン" or cid.startswith("runaton")


def _is_solrock(card) -> bool:
    n = (getattr(card, "name", "") or "").strip()
    cid = (getattr(card, "id", "") or "").strip()
    return n == "ソルロック" or cid.startswith("sorurokku")


def _is_makunoshita(card) -> bool:
    n = (getattr(card, "name", "") or "").strip()
    cid = (getattr(card, "id", "") or "").strip().lower()
    return n == "マクノシタ" or cid.startswith("makunoshita")


def _is_rioru(card) -> bool:
    name = (getattr(card, "name", "") or "").strip()
    cid = (getattr(card, "id", "") or "").strip()
    return name == "リオル" or cid.startswith("rioru")


def _is_lucario_stage_attacker(card) -> bool:
    if not card:
        return False
    n = (getattr(card, "name", "") or "").strip()
    cid = (getattr(card, "id", "") or "").strip().lower()
    if cid.startswith("rioru"):
        return False
    if n in ("ルカリオ", "メガルカリオ", "メガルカリオex"):
        return True
    return cid.startswith("rukario") or cid.startswith("mrukario")


# ---- action id ヘルパー ----

def energy_attach_action_id(bench_index_or_none: int | None) -> int:
    return 0 if bench_index_or_none is None else 1 + bench_index_or_none


def bench_index_from_action_id(action_id: int) -> int | None:
    return None if action_id == 0 else action_id - 1


# ---- マスク ----

def legal_energy_attach_mask(candidates: list[tuple]) -> list[bool]:
    mask = [False] * ENERGY_ATTACH_ACTION_DIM
    for target, _ in candidates:
        aid = energy_attach_action_id(target)
        if 0 <= aid < ENERGY_ATTACH_ACTION_DIM:
            mask[aid] = True
    return mask


# ---- ヒューリスティック logit ----

def heuristic_logits_energy_attach(
    state: GameState,
    candidates: list[tuple],
    energy_card,
) -> list[float]:
    """
    turn_energy.py の旧 _energy_attach_rule_score を logit に変換。
    tuple (w, dmg, is_active) → scalar = w + dmg * DMG_SCALE + ACTIVE_BONUS
    候補にない action は -MASK_LOGIT_ADD で実質無効化。
    """
    p = state.active_player_state()
    opp = state.defending_player_state()
    weights = state.get_weights_for_player(state.current_player)
    new_type = getattr(energy_card, "energy_type", None) or "colorless"
    is_first_turn = _is_first_player_first_turn(state)
    second_player_first_turn = state.turn_count == 0 and state.current_player != state.first_player

    opp_active_is_lucario_mirror = bool(
        opp.active and opp.active.card and _is_lucario_stage_attacker(opp.active.card)
    )
    lunatone_in_field = (
        bool(p.active and _is_lunatone(p.active.card))
        or any(_is_lunatone(bp.card) for bp in (p.bench or []))
    )
    solrock_in_field = (
        bool(p.active and _is_solrock(p.active.card))
        or any(_is_solrock(bp.card) for bp in (p.bench or []))
    )
    fightish_attach = (
        new_type == "fighting"
        or (getattr(energy_card, "id", "") or "") == "rokkutoukenenerugi"
        or (getattr(energy_card, "name", "") or "").strip() == "ロック闘エネルギー"
    )

    logits = [-_MASK_LOGIT_ADD] * ENERGY_ATTACH_ACTION_DIM

    for target, dmg in candidates:
        aid = energy_attach_action_id(target)
        if not (0 <= aid < ENERGY_ATTACH_ACTION_DIM):
            continue

        card = p.active.card if target is None else p.bench[target].card
        bp_ref = p.active if target is None else p.bench[target]
        bp_en = int(getattr(bp_ref, "attached_energy", 0) or 0)

        w = float(get_energy_attach_weight(weights, card, is_first_player=(state.current_player == state.first_player)))

        # バトル場のポケモンが次ターンKOされる+攻撃できない → エネを付けても無駄
        # ベンチに付けて育てる方が得
        _opp = state.defending_player_state()
        if target is None and p.active and _opp and _opp.active and p.bench:
            from .damage import _opponent_max_effective_damage
            _opp_dmg = _opponent_max_effective_damage(state)
            _will_be_koed = _opp_dmg > 0 and (p.active.hp or 0) <= _opp_dmg
            _active_can_attack = bool(dmg > 0)  # 付与後のダメージが0以上か
            if _will_be_koed and not _active_can_attack:
                w -= 5000.0  # KOされる+攻撃不可 → ベンチに回す

        if is_first_turn and _is_rioru(card):
            w += 2000.0
        elif _is_rioru(card):
            w += 1750.0
            if new_type == "fighting":
                w += 350.0
            # メガルカリオexが手札にある → リオルにエネを付けて進化後すぐ攻撃できるようにする
            has_mega_in_hand = any(
                "メガルカリオ" in (getattr(c, "name", "") or "")
                for c in (p.hand or [])
            )
            if has_mega_in_hand:
                w += 500.0

        # ルナトーンはサポート役。エネを付ける意味がほぼない。
        # 既にエネが付いている場合はさらに強くペナルティ。
        if _is_lunatone(card):
            w -= 5000.0
            if bp_en >= 1:
                w -= 10000.0  # 2枚目以降は絶対付けない

        if fightish_attach:
            if _is_makunoshita(card) and bp_en >= 1:
                w += 950.0

        if _is_solrock(card) and not is_first_turn:
            # Fix L: リオルがベンチにエネ0でいて、メガルカリオexが手札にある場合、
            # ソルロックよりリオルにエネを回すべき。また既にエネ1のソルロックには追加不要。
            _rioru_needs_energy = any(
                _is_rioru(bp.card) and int(getattr(bp, "attached_energy", 0) or 0) == 0
                for bp in (p.bench or [])
            ) or (p.active and _is_rioru(p.active.card) and int(getattr(p.active, "attached_energy", 0) or 0) == 0)
            _has_mega_in_hand = any(
                "メガルカリオ" in (getattr(c, "name", "") or "")
                for c in (p.hand or [])
            )
            if _rioru_needs_energy and _has_mega_in_hand:
                w = -9999.0  # リオルにエネを回すべき
            elif bp_en >= 1:
                w = -5000.0  # ソルロックはコスモビームにエネ1で十分。2枚目以降は絶対付けない
            elif not opp_active_is_lucario_mirror:
                w += 1800.0
                if new_type == "fighting":
                    w += 400.0
            elif new_type == "fighting":
                w += 200.0

        if opp_active_is_lucario_mirror and _is_lucario_stage_attacker(card):
            w += 3000.0
            if new_type == "fighting":
                w += 500.0

        # 同名ポケモンが場に複数いる場合、HPが高い方にエネを付ける
        # 手負い（ダメージを受けた）ポケモンは倒されやすいので、無傷の方に投資すべき
        if _is_lucario_stage_attacker(card):
            card_name = (getattr(card, "name", "") or "").strip()
            bp_hp = getattr(bp_ref, "hp", 0) or 0
            # 同名のベンチ/アクティブでHP高いのがいるか
            _same_name_higher_hp = False
            all_field = ([p.active] if p.active else []) + list(p.bench or [])
            for _fp in all_field:
                if _fp is bp_ref:
                    continue
                if (getattr(_fp.card, "name", "") or "").strip() == card_name:
                    if (getattr(_fp, "hp", 0) or 0) > bp_hp:
                        _same_name_higher_hp = True
                        break
            if _same_name_higher_hp:
                w -= 3000.0  # HPが低い方にはエネを付けない

        # メガルカリオex/ルカリオ系: エネを付ければ新しい技が撃てるようになるならボーナス。
        # 例: エネ1→エネ2でメガブレイブ（270ダメ）が解放される。
        if _is_lucario_stage_attacker(card) and not opp_active_is_lucario_mirror:
            from .state import _can_pay_energy_cost
            types_after = list(getattr(bp_ref, "attached_energy_types", []) or []) + [new_type]
            en_after = bp_en + 1
            for atk in (card.attacks or []):
                can_before = _can_pay_energy_cost(
                    bp_en, getattr(bp_ref, "attached_energy_types", []),
                    atk.energy_cost, getattr(atk, "energy_cost_typed", None),
                )
                can_after = _can_pay_energy_cost(
                    en_after, types_after,
                    atk.energy_cost, getattr(atk, "energy_cost_typed", None),
                )
                if not can_before and can_after:
                    w += 2500.0  # 新技が解放される
                    break

        if (
            lunatone_in_field and solrock_in_field and second_player_first_turn
            and _is_solrock(card) and not opp_active_is_lucario_mirror
            and getattr(opp.active, "hp", None) is not None
            and opp.active.hp > 0 and dmg >= opp.active.hp
        ):
            w += 1200.0

        # ---- ドラパルトexデッキ固有ロジック ----
        from .deck_strategies import is_dragapult_deck_for_player
        if is_dragapult_deck_for_player(state, state.current_player):
            card_name = (getattr(card, "name", "") or "").strip()

            # ドラパルトex: メインアタッカー。ファントムダイブは炎+超の2エネ必要
            if card_name == "ドラパルトex":
                types_on = list(getattr(bp_ref, "attached_energy_types", []) or [])
                # 悪エネはファントムダイブに使えない → 絶対付けない
                if new_type == "darkness":
                    w -= 15000.0
                else:
                    w += 3000.0
                    # エネの色が合っていればさらにボーナス
                    has_fire = "fire" in types_on
                    has_psychic = "psychic" in types_on
                    if new_type == "fire" and not has_fire:
                        w += 2000.0  # 炎がまだ付いてない → 炎優先
                    elif new_type == "psychic" and not has_psychic:
                        w += 2000.0  # 超がまだ付いてない → 超優先
                    elif new_type == "fire" and has_fire:
                        w -= 15000.0  # 炎が既にある → 絶対に付けない
                    elif new_type == "psychic" and has_psychic:
                        w -= 15000.0  # 超が既にある → 絶対に付けない
                    # 既にエネが1つ付いている=あと1枚でファントムダイブ → 集中投資
                    if bp_en == 1 and ((new_type == "fire" and not has_fire) or (new_type == "psychic" and not has_psychic)):
                        w += 4000.0  # あと1枚でファントムダイブ → 最優先
                # 新技が解放されるならボーナス（バトル場なら即攻撃可能で超大ボーナス）
                from .state import _can_pay_energy_cost
                types_after = types_on + [new_type]
                en_after = bp_en + 1
                for atk in (card.attacks or []):
                    can_before = _can_pay_energy_cost(
                        bp_en, types_on, atk.energy_cost, getattr(atk, "energy_cost_typed", None),
                    )
                    can_after = _can_pay_energy_cost(
                        en_after, types_after, atk.energy_cost, getattr(atk, "energy_cost_typed", None),
                    )
                    if not can_before and can_after:
                        if target is None:
                            w += 10000.0  # バトル場 → 即ファントムダイブ可能！最大ボーナス
                        else:
                            w += 6000.0  # ベンチ → 次ターンで前に出せる
                        break
                # 既にファントムダイブが撃てる → 追加エネ不要（他のドラパルトexに回す）
                if bp_en >= 2 and has_fire and has_psychic:
                    w -= 3000.0

            # ドロンチ/ドラメシヤ: 進化前。エネを付ける価値はあるがドラパルトexより低い
            # エネは進化後も引き継がれるのでファントムダイブの準備になる
            elif card_name in ("ドロンチ", "ドラメシヤ"):
                # 悪エネは進化後もファントムダイブに使えない → 付けない
                if new_type == "darkness":
                    w -= 10000.0
                else:
                    # 手札にドラパルトex/ふしぎなアメがある → 進化後すぐ攻撃できるようにエネ付け
                    has_drapaex_or_ame = any(
                        (getattr(c, "name", "") or "") in ("ドラパルトex",)
                        or (getattr(c, "id", "") or "") == "fushiginaame"
                        for c in (p.hand or [])
                    )
                    # ドラパルトexが場にいてエネ不足なら、ドロンチ/ドラメシヤよりドラパルトex優先
                    _drapa_on_field_needing = any(
                        (getattr(bp2.card, "name", "") or "").strip() == "ドラパルトex"
                        and (getattr(bp2, "attached_energy", 0) or 0) < 2
                        for bp2 in ([p.active] if p.active else []) + list(p.bench or [])
                    )
                    if _drapa_on_field_needing:
                        w -= 2000.0  # ドラパルトexにエネを回すべき
                    elif has_drapaex_or_ame:
                        w += 2000.0
                        if new_type in ("fire", "psychic"):
                            w += 1000.0
                    else:
                        w += 500.0

            # マシマシラ: アドレナブレインに悪エネが必要
            elif card_name == "マシマシラ":
                if new_type == "darkness":
                    w += 800.0
                else:
                    w -= 2000.0  # 悪以外はマシマシラに付けても意味が薄い

            # サポート役にはエネを付けない（絶対に付けない）
            # 例外: ニャースexがバトル場で悪エネを逃げ用に付ける（ベンチにアタッカーがいる場合）
            elif card_name in ("スボミー", "キチキギスex", "ニャースex"):
                if (
                    card_name == "ニャースex"
                    and target is None  # バトル場
                    and new_type == "darkness"  # 悪エネ（ドラパルトexに不要なので無駄にならない）
                    and bp_en == 0  # まだエネが付いていない
                    and any(
                        (getattr(bp2.card, "name", "") or "").strip() in ("ドラパルトex", "ドロンチ", "ドラメシヤ")
                        for bp2 in (p.bench or [])
                    )
                ):
                    w += 500.0  # 逃げ用に悪エネを付ける（ドラパルトexに不要なエネなので無駄にならない）
                else:
                    w -= 15000.0

            # ヨマワルライン: カースドボムは自爆なのでエネ不要（絶対に付けない）
            elif card_name in ("ヨマワル", "サマヨール", "ヨノワール"):
                w -= 15000.0

        logits[aid] = w + dmg * _DMG_SCALE + (_ACTIVE_BONUS if target is None else 0.0)

    return logits


# ---- ユーティリティ（support_policy.py と同パターン） ----

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


def top2_action_ids_from_logits(logits_masked: list[float], mask: list[bool]) -> list[int]:
    """マスク合法の中で logit が高い順に最大 2 つの action_id を返す。"""
    legal = [(i, logits_masked[i]) for i in range(len(logits_masked)) if i < len(mask) and mask[i]]
    legal.sort(key=lambda x: -x[1])
    return [a[0] for a in legal[:2]]


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


# ---- スケジュール ----

def policy_schedule_progress(state: GameState) -> float:
    h = int(getattr(state, "policy_schedule_horizon_steps", 0))
    if h <= 0:
        return 0.0
    return min(1.0, float(getattr(state, "total_env_steps", 0)) / float(h))


def effective_heuristic_scale(state: GameState) -> float:
    if int(getattr(state, "policy_schedule_horizon_steps", 0)) <= 0:
        return 1.0
    p = policy_schedule_progress(state)
    decay = float(getattr(state, "energy_policy_heuristic_decay", 0.7))
    floor_ = float(getattr(state, "energy_policy_heuristic_scale_floor", 0.3))
    return max(floor_, 1.0 - p * decay)


def effective_pi_energy_lambda(state: GameState) -> float:
    """
    linear: λ_max * min(1, step/warmup)
    sigmoid: λ_max * sigmoid((step/warmup)*6 - 3)
    warmup_steps==0 → 常に λ_max
    """
    lam_max = float(getattr(state, "pi_energy_policy_lambda", 0.5))
    horizon = int(getattr(state, "policy_schedule_horizon_steps", 0))
    if horizon > 0:
        p = policy_schedule_progress(state)
        sched = str(getattr(state, "pi_energy_policy_lambda_schedule", "linear")).lower()
        if sched == "sigmoid":
            return lam_max * _sigmoid(p * 6.0 - 3.0)
        return lam_max * min(1.0, p)
    warmup = int(getattr(state, "pi_energy_policy_lambda_warmup_steps", 0))
    step = int(getattr(state, "total_env_steps", 0))
    if warmup <= 0:
        return lam_max
    t = float(step) / float(warmup)
    sched = str(getattr(state, "pi_energy_policy_lambda_schedule", "linear")).lower()
    if sched == "sigmoid":
        return lam_max * _sigmoid(t * 6.0 - 3.0)
    return lam_max * min(1.0, t)


def effective_energy_epsilon(state: GameState) -> float:
    eps0 = float(getattr(state, "energy_policy_epsilon", 0.0))
    eps1 = float(getattr(state, "energy_policy_epsilon_end", 0.0))
    if int(getattr(state, "policy_schedule_horizon_steps", 0)) <= 0:
        return eps0
    p = policy_schedule_progress(state)
    sched = str(getattr(state, "energy_policy_epsilon_schedule", "linear")).lower()
    if sched == "exp":
        k = float(getattr(state, "energy_policy_epsilon_decay_k", 3.0))
        return eps1 + (eps0 - eps1) * math.exp(-k * p)
    return eps1 + (eps0 - eps1) * (1.0 - p)


# ---- メイン決定関数 ----

def energy_attach_policy_decision(
    state: GameState,
    candidates: list[tuple],
    energy_card,
) -> tuple[int, dict[str, Any]]:
    """
    エネルギー付与先を決定する。

    Parameters
    ----------
    candidates : [(bench_index_or_None, dmg), ...]
    energy_card : 付与するエネルギーカード

    Returns
    -------
    (action_id, debug_dict)
      action_id: 0=active, 1-5=bench[i-1]
    """
    mask = legal_energy_attach_mask(candidates)
    h_raw = heuristic_logits_energy_attach(state, candidates, energy_card)
    h_scale = effective_heuristic_scale(state)
    h_scaled = [float(x) * h_scale for x in h_raw]
    lam_eff = effective_pi_energy_lambda(state)

    # Q-value モデル（prize_delta 回帰）が指定された場合はそちらを優先
    # per-player モデルがあればそちらを使い、なければ共通モデル
    if not rules_only_for_player(state):
        by_player = getattr(state, "q_energy_attach_model_path_by_player", [None, None])
        q_path = (by_player[state.current_player] if by_player else None) or getattr(state, "q_energy_attach_model_path", None)
    else:
        q_path = None
    pi_path = None if rules_only_for_player(state) else getattr(state, "pi_energy_policy_model_path", None)
    q_vals_raw: list[float] | None = None  # lookahead との合算用に生の Q 値を保持
    if q_path:
        from .encoders import encode_state_v2

        # Advantage モデルか Q-value モデルかを自動判別
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
            qm = load_advantage_model_pt(q_path)
        else:
            from .q_models import load_q_value_model_pt
            qm = load_q_value_model_pt(q_path)

        vec = encode_state_v2(state, state.current_player)
        q_vals_raw = qm.predict_all(vec)
        q_scale = float(getattr(state, "q_energy_policy_scale", 10.0))
        nn_logits = [v * q_scale for v in q_vals_raw]
        do_norm = bool(getattr(state, "energy_policy_normalize_logits", True))
        logits = _combine_heuristic_and_nn_logits(h_scaled, nn_logits, lam_eff, normalize=do_norm)
    elif pi_path:
        from .pi_models import load_pi_model_pt
        from .encoders import encode_state_basic, encode_state_v2

        pi = load_pi_model_pt(pi_path)
        if pi.encoder_name == "opening":
            vec = encode_state_v2(state, state.current_player)
        else:
            vec = encode_state_basic(state, state.current_player)
        nn_logits = pi.predict_logits_one(vec)
        do_norm = bool(getattr(state, "energy_policy_normalize_logits", True))
        logits = _combine_heuristic_and_nn_logits(h_scaled, list(nn_logits), lam_eff, normalize=do_norm)
    else:
        logits = list(h_scaled)

    logits_pre_mask = list(logits)
    logits_masked = _apply_legal_mask(logits, mask)
    entropy = _entropy_from_logits(logits_masked)
    top2_gap = _top2_gap_masked(logits_masked, mask)
    sched_p = policy_schedule_progress(state)
    eps_eff = effective_energy_epsilon(state)
    deterministic = getattr(state, "energy_policy_deterministic", True)

    if deterministic:
        action_id = int(max(range(len(logits_masked)), key=lambda i: logits_masked[i]))
    else:
        action_id = _sample_from_logits(logits_masked)

    eps_used = False
    if eps_eff > 0.0 and random.random() < eps_eff:
        action_id = _sample_from_logits(logits_masked)
        eps_used = True

    debug: dict[str, Any] = {
        "mask": mask,
        "logits_pre_mask": logits_pre_mask,
        "logits_masked": logits_masked,
        "top2_gap": top2_gap,
        "entropy": entropy,
        "eps_used": eps_used,
        "heuristic_scale": h_scale,
        "q_vals_raw": q_vals_raw,  # None or list[float] (predicted prize_delta / scale)
        "sched_p": sched_p,
        "eps_eff": eps_eff,
        "lam_eff": lam_eff,
    }
    return action_id, debug


# ---- ログ extras 生成 ----

def _make_energy_log_extras(
    state: GameState,
    action_id: int,
    debug: dict[str, Any],
    energy_card,
    *,
    target_bench_index: int | None,
    candidates_dmg_by_action: dict[int, float],
    lookahead_used: bool | None = None,
    lookahead_score: float | None = None,
) -> dict[str, Any]:
    p = state.active_player_state()
    opp = state.defending_player_state()
    bp = p.active if target_bench_index is None else (
        p.bench[target_bench_index] if target_bench_index < len(p.bench) else None
    )
    before = int(getattr(bp, "attached_energy", 0) or 0) if bp else 0
    after = before + 1

    # can_attack_after: 付与後のエネルギー数でどれかの技コストを満たすか（数のみで近似）
    can_atk = False
    if bp:
        attacks = getattr(getattr(bp, "card", None), "attacks", []) or []
        min_cost = min((getattr(a, "energy_cost", 0) for a in attacks), default=999)
        can_atk = after >= min_cost

    dmg_val = float(candidates_dmg_by_action.get(action_id, 0.0))
    opp_hp = getattr(getattr(opp, "active", None), "hp", None)
    can_kill = bool(opp_hp is not None and opp_hp > 0 and dmg_val >= opp_hp)

    return {
        "energy_policy_action_id": action_id,
        "energy_policy_mask": debug["mask"],
        "energy_policy_logits_pre_mask": debug["logits_pre_mask"],
        "energy_policy_logits": debug["logits_masked"],
        "energy_policy_top2_gap": debug["top2_gap"],
        "energy_policy_entropy": debug["entropy"],
        "energy_policy_epsilon_greedy": debug["eps_used"],
        "energy_policy_heuristic_scale": debug["heuristic_scale"],
        "energy_policy_schedule_progress": debug["sched_p"],
        "energy_policy_epsilon_effective": debug["eps_eff"],
        "pi_energy_policy_lambda_effective": debug["lam_eff"],
        "energy_type": str(getattr(energy_card, "energy_type", None) or "colorless"),
        "energy_before": before,
        "energy_after": after,
        "can_attack_after": can_atk,
        "can_kill_after": can_kill,
        "energy_lookahead_used": bool(lookahead_used) if lookahead_used is not None else False,
        "energy_lookahead_score": float(lookahead_score) if lookahead_score is not None else None,
    }


# ---- エントリポイント ----

def pick_energy_attach_by_policy(
    state: GameState,
    candidates: list[tuple],
    energy_hand_idx: int,
    energy_card,
) -> tuple[int | None, dict[str, Any]]:
    """
    エネルギー付与先を決定して返す。実際の attach は呼び出し元で行う。

    Parameters
    ----------
    candidates : [(bench_index_or_None, dmg), ...]
    energy_hand_idx : 手札中エネルギーカードのインデックス（将来の拡張用）
    energy_card : 付与するエネルギーカード

    Returns
    -------
    (bench_index_or_none, log_extras)
      bench_index_or_none: None=active, 0-4=bench index
      log_extras: _log_choice に渡せる kwargs（完全ログ）
    """
    if not candidates:
        return None, {}

    cdba: dict[int, float] = {
        energy_attach_action_id(t): float(d) for t, d in candidates
    }

    action_id, debug = energy_attach_policy_decision(state, candidates, energy_card)
    debug["candidates_dmg_by_action"] = cdba  # _make_energy_log_extras 用（debug 内に保持）

    bench_index = bench_index_from_action_id(action_id)
    extras = _make_energy_log_extras(
        state, action_id, debug, energy_card,
        target_bench_index=bench_index,
        candidates_dmg_by_action=cdba,
    )

    return bench_index, extras
