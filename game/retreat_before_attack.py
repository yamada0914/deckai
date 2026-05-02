"""
攻撃フェーズ直前の「にげ／交代」を列挙型ポリシーで選ぶ。

行動空間（固定次元 1 + BENCH_SIZE）:
  0: NOOP（にげない）
  1..BENCH_SIZE: ベンチ (i-1) へにげる（合法なら）

合成: logits = heuristic_scale * heuristic + lambda_eff * NN（既定で |·| 平均正規化後に合成: retreat_before_attack_normalize_logits）。
lambda_eff: policy_schedule_horizon_steps > 0 のときは ε・ヒューリスティックと同じ progress。未設定時のみ従来の warmup 比。
total_env_steps は学習の global_step を外から代入（increment_total_env_steps_on_log=True なら _log_choice ごと +1、単体試合用）。

ε: retreat_before_attack_epsilon → epsilon_end へ線形／指数減衰（horizon 指定時）。

マスク: logits[i] + (float(mask[i]) - 1.0) * MASK_ADD（softmax 前のみ）

学習用ログ type: retreat_before_attack（NOOP 含む全分岐）。entropy / epsilon_greedy / top2_gap / schedule 付き。
"""

from __future__ import annotations

import math
import random
from typing import Any

from .attack import get_legal_attack_indices
from .deck_strategies import is_dragapult_deck_for_player, DRAPA_LINE_NAMES, DRAPA_SUPPORT_NAMES
from .damage import (
    _max_effective_damage_for_attacker,
    _opponent_max_effective_damage,
    _our_max_effective_damage,
)
from .encoders import encode_state_basic
from .state import BENCH_SIZE, GameState, PlayerState, _log_choice, rules_only_for_player
from .weights import get_retreat_target_weight

RETREAT_BEFORE_ATTACK_ACTION_DIM = 1 + BENCH_SIZE

_NOOP_BIAS = -0.01
_RETREAT_BASE_PENALTY = -0.02
_BIAS_TWIN = 5.0
_BIAS_SOLROCK = 4.9
_BIAS_MEGA_PROTECT = 6.0

# 非法マスク: logits + (0 - 1) * ADD = -ADD（分岐なしで同値化）
_MASK_LOGIT_ADD = 1e4


def _card_key(card) -> str:
    return (getattr(card, "id", None) or getattr(card, "name", "") or "").strip()


def _is_solrock(card) -> bool:
    n = (getattr(card, "name", "") or "").strip()
    cid = (getattr(card, "id", "") or "").strip()
    return n == "ソルロック" or cid.startswith("sorurokku")


def _solrock_bench_index(p: PlayerState) -> int | None:
    for i, bp in enumerate(p.bench or []):
        if bp and bp.card and _is_solrock(bp.card):
            return i
    return None


def _retreat_energy_cost(p: PlayerState) -> int:
    if not p.active:
        return 999
    raw = getattr(p.active.card, "retreat_cost", 1)
    tool = getattr(p.active, "attached_tool", None)
    if tool and (getattr(tool, "id", "") or "") == "fuusen":
        return max(0, raw - 2)
    return max(0, raw)


def _can_retreat_to_bench(state: GameState, bench_index: int) -> bool:
    if getattr(state, "retreat_used_this_turn", False):
        return False
    p = state.active_player_state()
    if not p.active or bench_index < 0 or bench_index >= len(p.bench):
        return False
    if getattr(p.active, "special_state", None) in ("sleep", "paralysis"):
        return False
    cost = _retreat_energy_cost(p)
    return p.active.attached_energy >= cost


def legal_retreat_before_attack_mask(state: GameState) -> list[bool]:
    mask = [True] + [False] * BENCH_SIZE
    for bi in range(BENCH_SIZE):
        if bi < len(state.active_player_state().bench) and _can_retreat_to_bench(state, bi):
            mask[1 + bi] = True
    return mask


def _twin_preferred_bench(state: GameState) -> int | None:
    if getattr(state, "retreat_used_this_turn", False):
        return None
    p = state.active_player_state()
    opp = state.defending_player_state()
    if not p.active or not opp.active or not p.bench:
        return None
    if getattr(p.active, "special_state", None) in ("sleep", "paralysis"):
        return None
    legal = get_legal_attack_indices(state, p, opp)
    if not legal:
        return None
    opp_max = _opponent_max_effective_damage(state)
    if opp_max <= 0 or p.active.hp > opp_max:
        return None
    active_max = _our_max_effective_damage(state)
    if active_max <= 0:
        return None
    if not p.bench:
        return None
    # にげるとエネが減る。にげた後にKO可能な技が撃てなくなるなら、にげない方がいい。
    raw_rc = getattr(p.active.card, "retreat_cost", 1)
    tool = getattr(p.active, "attached_tool", None)
    eff_rc = max(0, raw_rc - (2 if tool and (getattr(tool, "id", "") or "") == "fuusen" else 0))
    active_key = _card_key(p.active.card)
    weights = state.get_weights_for_player(state.current_player)
    best_bi: int | None = None
    best_tuple: tuple[int, float] | None = None
    for bi, bp in enumerate(p.bench):
        if _card_key(bp.card) != active_key:
            continue
        if bp.hp <= p.active.hp:
            continue
        bench_max = _max_effective_damage_for_attacker(state, bp, opp.active, state.current_player)
        if bench_max < active_max:
            continue
        if not _can_retreat_to_bench(state, bi):
            continue
        # にげコスト > 0 の場合、にげた後のアクティブ（元ベンチ）で相手を倒せるか確認。
        # ベンチの同名ポケモンが自力で倒せるなら、にげても攻撃力は落ちない。
        # 倒せない場合でも、ダメージを受けた方を温存する意味があるのでにげる。
        # ただし、にげコストでアクティブ側のエネが減り攻撃不能になる場合はスキップ。
        if eff_rc > 0 and opp.active and opp.active.hp is not None and opp.active.hp > 0:
            if active_max >= opp.active.hp and bench_max < opp.active.hp:
                # アクティブなら倒せるがベンチでは倒せない → にげない
                continue
        w = float(get_retreat_target_weight(weights, bp.card))
        t = (bp.hp, w)
        if best_tuple is None or t > best_tuple:
            best_tuple = t
            best_bi = bi
    return best_bi


def _solrock_preferred_bench(state: GameState) -> int | None:
    if getattr(state, "retreat_used_this_turn", False):
        return None
    p = state.active_player_state()
    opp = state.defending_player_state()
    if not p.active or not opp.active or not p.bench:
        return None
    if getattr(p.active, "special_state", None) in ("sleep", "paralysis"):
        return None
    if _is_solrock(p.active.card):
        return None
    sol_bi = _solrock_bench_index(p)
    if sol_bi is None:
        return None
    sol_dmg = _max_effective_damage_for_attacker(state, p.bench[sol_bi], opp.active, state.current_player)
    if sol_dmg < opp.active.hp:
        return None
    active_best = _our_max_effective_damage(state)
    if active_best >= opp.active.hp:
        return None
    if not _can_retreat_to_bench(state, sol_bi):
        return None
    return sol_bi


def _mega_protect_preferred_bench(state: GameState) -> int | None:
    """メガルカリオex (3 prize) が KO 圏内にいるとき、より安全なベンチに退避する。

    非常に保守的な条件:
      - active が is_mega (3 prizes)
      - にげるコストが 0（ふうせんあり等）。エネを捨ててまで退避しない。
      - 相手の「現在のバトル場」が既に KO 可能な打点を持っている
      - 相手のサイド残り <= 3（KO で相手が勝つ）
      - 退避先は攻撃可能な 1-prize ポケモンで、相手を KO できるもの
    """
    if getattr(state, "retreat_used_this_turn", False):
        return None
    p = state.active_player_state()
    opp = state.defending_player_state()
    if not p.active or not opp.active or not p.bench:
        return None
    if getattr(p.active, "special_state", None) in ("sleep", "paralysis"):
        return None
    # active が 3-prize (メガ) でなければ不要
    if not getattr(p.active.card, "is_mega", False):
        return None
    # にげるコストが 0 でなければスキップ（エネロスが大きすぎる）
    if _retreat_energy_cost(p) > 0:
        return None
    # 相手のサイド残り <= 3 でなければ不要（KO されても即負けしない）
    opp_remaining = len(opp.prize_pile)
    if opp_remaining > 3:
        return None
    # 相手の現在のバトル場が KO 可能か（保守的: ベンチからの脅威は考慮しない）
    opp_current_max = _opponent_max_effective_damage(state)
    if opp_current_max <= 0 or p.active.hp > opp_current_max:
        return None
    # 退避先を探す: 相手を KO できる 1-prize ポケモンを優先
    best_bi: int | None = None
    best_score = -999.0
    for bi, bp in enumerate(p.bench):
        if not _can_retreat_to_bench(state, bi):
            continue
        bp_prize = 3 if getattr(bp.card, "is_mega", False) else (2 if getattr(bp.card, "is_ex", False) else 1)
        if bp_prize >= 2:
            continue
        bench_dmg = _max_effective_damage_for_attacker(state, bp, opp.active, state.current_player)
        score = 0.0
        if bench_dmg > 0:
            score += 3.0
        if bench_dmg >= opp.active.hp:
            score += 5.0
        if score > best_score:
            best_score = score
            best_bi = bi
    return best_bi


def heuristic_logits_retreat_before_attack(state: GameState) -> list[float]:
    logits = [_NOOP_BIAS] + [_RETREAT_BASE_PENALTY] * BENCH_SIZE
    twin_bi = _twin_preferred_bench(state)
    sol_bi = _solrock_preferred_bench(state)
    mega_bi = _mega_protect_preferred_bench(state)
    if twin_bi is not None and 0 <= twin_bi < BENCH_SIZE:
        logits[1 + twin_bi] += _BIAS_TWIN
    if sol_bi is not None and 0 <= sol_bi < BENCH_SIZE:
        logits[1 + sol_bi] += _BIAS_SOLROCK
    if mega_bi is not None and 0 <= mega_bi < BENCH_SIZE:
        logits[1 + mega_bi] += _BIAS_MEGA_PROTECT

    # ドラパルトexデッキ: エネルギーを貯めたドラパルトexのにげるは大きなペナルティ
    # ファントムダイブ用のfire/psychicエネルギーを失うのは致命的
    if is_dragapult_deck_for_player(state, state.current_player):
        p = state.active_player_state()
        if p.active:
            active_name = (getattr(p.active.card, "name", "") or "").strip()
            active_energy = getattr(p.active, "attached_energy", 0) or 0
            active_types = list(getattr(p.active, "attached_energy_types", []) or [])
            if active_name == "ドラパルトex" and active_energy >= 2:
                # ファントムダイブ用のfire/psychicが付いている → にげるとエネロス
                has_fire = "fire" in active_types
                has_psychic = "psychic" in active_types
                if has_fire or has_psychic:
                    # にげるペナルティを大幅に増やす（NOOP を強く推奨）
                    logits[0] += 10.0  # NOOPを強く推奨
                    for bi in range(BENCH_SIZE):
                        logits[1 + bi] -= 8.0  # にげるペナルティ
            elif active_name in ("キチキギスex", "ニャースex", "ヨマワル", "サマヨール") and active_energy == 0:
                # サポートポケモンがバトル場でエネなし → ベンチのドラパルトexに交代したい
                opp = state.defending_player_state()
                for bi, bp in enumerate(p.bench):
                    if bi >= BENCH_SIZE:
                        break
                    bp_name = (getattr(bp.card, "name", "") or "").strip()
                    if bp_name == "ドラパルトex":
                        bp_energy = getattr(bp, "attached_energy", 0) or 0
                        if bp_energy >= 1:  # 攻撃可能なドラパルトex
                            logits[1 + bi] += 5.0  # 攻撃可能なドラパルトexに交代推奨
                    elif bp_name == "ドロンチ":
                        logits[1 + bi] += 2.0  # ドロンチも壁としてはex より安全

            # キチキギスex/ニャースex がベンチにいる → にげ先として選ばない (2サイド献上リスク)
            for bi, bp in enumerate(p.bench):
                if bi >= BENCH_SIZE:
                    break
                bp_name = (getattr(bp.card, "name", "") or "").strip()
                if bp_name in ("キチキギスex", "ニャースex"):
                    logits[1 + bi] -= 8.0  # 前に出すと2サイド献上リスク

    return logits


def _sigmoid(x: float) -> float:
    if x >= 30.0:
        return 1.0
    if x <= -30.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def policy_schedule_progress(state: GameState) -> float:
    """total_env_steps / policy_schedule_horizon_steps、horizon<=0 なら 0。"""
    h = int(getattr(state, "policy_schedule_horizon_steps", 0))
    if h <= 0:
        return 0.0
    return min(1.0, float(getattr(state, "total_env_steps", 0)) / float(h))


def effective_heuristic_scale(state: GameState) -> float:
    """max(floor, 1 - progress * heuristic_decay)。horizon なしは 1.0。"""
    if int(getattr(state, "policy_schedule_horizon_steps", 0)) <= 0:
        return 1.0
    p = policy_schedule_progress(state)
    decay = float(getattr(state, "retreat_before_attack_heuristic_decay", 0.7))
    floor = float(getattr(state, "retreat_before_attack_heuristic_scale_floor", 0.3))
    return max(floor, 1.0 - p * decay)


def effective_retreat_epsilon(state: GameState) -> float:
    """ε0 から ε_end へ線形または指数減衰（horizon なしは ε0 のみ）。"""
    eps0 = float(getattr(state, "retreat_before_attack_epsilon", 0.0))
    eps1 = float(getattr(state, "retreat_before_attack_epsilon_end", 0.0))
    if int(getattr(state, "policy_schedule_horizon_steps", 0)) <= 0:
        return eps0
    p = policy_schedule_progress(state)
    sched = str(getattr(state, "retreat_before_attack_epsilon_schedule", "linear")).lower()
    if sched == "exp":
        k = float(getattr(state, "retreat_before_attack_epsilon_decay_k", 3.0))
        return eps1 + (eps0 - eps1) * math.exp(-k * p)
    return eps1 + (eps0 - eps1) * (1.0 - p)


def effective_pi_retreat_lambda(state: GameState) -> float:
    """
    policy_schedule_horizon_steps > 0 のときは ε・ヒューリスティックと同じ progress を使用（統一スケジュール）。
    linear: λ_max * min(1, step/warmup)
    sigmoid: λ_max * sigmoid((step/warmup)*6 - 3)  … 初期ほぼ 0、中盤で立ち上がり
    warmup_steps==0 のときはスケジュールなし（常に λ_max）。
    """
    lam_max = float(getattr(state, "pi_retreat_before_attack_lambda", 0.5))
    horizon = int(getattr(state, "policy_schedule_horizon_steps", 0))
    if horizon > 0:
        p = policy_schedule_progress(state)
        sched = str(getattr(state, "pi_retreat_before_attack_lambda_schedule", "linear")).lower()
        if sched == "sigmoid":
            return lam_max * _sigmoid(p * 6.0 - 3.0)
        return lam_max * min(1.0, p)
    warmup = int(getattr(state, "pi_retreat_before_attack_lambda_warmup_steps", 0))
    step = int(getattr(state, "total_env_steps", 0))
    if warmup <= 0:
        return lam_max
    t = float(step) / float(warmup)
    sched = str(getattr(state, "pi_retreat_before_attack_lambda_schedule", "linear")).lower()
    if sched == "sigmoid":
        return lam_max * _sigmoid(t * 6.0 - 3.0)
    return lam_max * min(1.0, t)


def _apply_legal_mask(logits: list[float], mask: list[bool]) -> list[float]:
    """masked_logits[i] = logits[i] + (float(mask[i]) - 1.0) * ADD（その後 softmax / argmax）。"""
    return [logits[i] + (float(mask[i]) - 1.0) * _MASK_LOGIT_ADD for i in range(len(logits))]


def _top2_gap_masked(logits_masked: list[float], mask: list[bool]) -> float:
    """合法マスク内の top1 と top2 の logit 差（迷いの指標。合法が 1 なら inf）。"""
    vals = [logits_masked[i] for i in range(len(logits_masked)) if i < len(mask) and mask[i]]
    if len(vals) < 2:
        return float("inf") if len(vals) == 1 else 0.0
    vals.sort(reverse=True)
    return float(vals[0] - vals[1])


def _entropy_from_logits(logits: list[float]) -> float:
    """マスク済み分布のシャノンエントロピー（自然対数、nat）。"""
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps)
    if s <= 0:
        return 0.0
    ent = 0.0
    for e in exps:
        p = e / s
        if p > 0.0:
            ent -= p * math.log(p + 1e-12)
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
    """
    合成: h + λ * nn。normalize 時は各ベクトルを |·|_mean で割ってから足す（スケール差の吸収）。
    """
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
    out: list[float] = []
    for i in range(n):
        hn = h_scaled[i] / ma_h
        ni = nn_f[i] / ma_n
        out.append(hn + lam * ni)
    return out


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


def retreat_before_attack_decision(state: GameState) -> tuple[int, dict[str, Any]]:
    mask = legal_retreat_before_attack_mask(state)
    h_raw = heuristic_logits_retreat_before_attack(state)
    h_scale = effective_heuristic_scale(state)
    h_scaled = [float(x) * h_scale for x in h_raw]
    lam_eff = effective_pi_retreat_lambda(state)
    pi_path = None if rules_only_for_player(state) else getattr(state, "pi_retreat_before_attack_model_path", None)
    if pi_path:
        from .pi_models import load_pi_model_pt

        pi = load_pi_model_pt(pi_path)
        vec = encode_state_basic(state, state.current_player)
        nn_logits = pi.predict_logits_one(vec)
        do_norm = bool(getattr(state, "retreat_before_attack_normalize_logits", True))
        logits = _combine_heuristic_and_nn_logits(
            h_scaled,
            list(nn_logits),
            lam_eff,
            normalize=do_norm,
        )
    else:
        logits = list(h_scaled)
    logits_pre_mask = list(logits)
    logits_masked = _apply_legal_mask(logits, mask)
    entropy = _entropy_from_logits(logits_masked)
    top2_gap = _top2_gap_masked(logits_masked, mask)
    sched_p = policy_schedule_progress(state)
    eps_eff = effective_retreat_epsilon(state)
    deterministic = getattr(state, "retreat_before_attack_deterministic", True)
    if deterministic:
        idx = int(max(range(len(logits_masked)), key=lambda i: logits_masked[i]))
    else:
        idx = _sample_from_logits(logits_masked)

    eps_used = False
    if eps_eff > 0.0 and random.random() < eps_eff:
        # 一様ランダムではなく、マスク済み方策分布からサンプル（弱い NN も含む logits_masked）
        idx = _sample_from_logits(logits_masked)
        eps_used = True

    debug: dict[str, Any] = {
        "mask": mask,
        "logits_pre_mask": logits_pre_mask,
        "logits_masked": logits_masked,
        "pi_retreat_lambda_effective": lam_eff,
        "retreat_policy_entropy": entropy,
        "retreat_policy_epsilon_greedy": eps_used,
        "retreat_policy_top2_gap": top2_gap,
        "retreat_policy_schedule_progress": sched_p,
        "retreat_policy_heuristic_scale": h_scale,
        "retreat_policy_epsilon_effective": eps_eff,
    }
    return idx, debug


def select_retreat_before_attack_action_index(state: GameState) -> int:
    return retreat_before_attack_decision(state)[0]


def _log_retreat_before_attack_choice(state: GameState, idx: int, dbg: dict[str, Any], *, card_id: str | None) -> None:
    _log_choice(
        state,
        "retreat_before_attack",
        card_id=card_id,
        retreat_policy_action_id=idx,
        policy_action_id=idx,
        retreat_policy_mask=dbg["mask"],
        retreat_policy_logits_pre_mask=dbg["logits_pre_mask"],
        retreat_policy_logits=dbg["logits_masked"],
        pi_retreat_lambda_effective=dbg["pi_retreat_lambda_effective"],
        retreat_policy_entropy=dbg["retreat_policy_entropy"],
        retreat_policy_epsilon_greedy=dbg["retreat_policy_epsilon_greedy"],
        retreat_policy_top2_gap=dbg.get("retreat_policy_top2_gap"),
        retreat_policy_schedule_progress=dbg.get("retreat_policy_schedule_progress"),
        retreat_policy_heuristic_scale=dbg.get("retreat_policy_heuristic_scale"),
        retreat_policy_epsilon_effective=dbg.get("retreat_policy_epsilon_effective"),
    )


def try_retreat_before_attack_policy(state: GameState) -> bool:
    """
    _do_attack_phase 内（合法攻撃が確定した直後）からのみ呼ぶこと。

    呼び出し側で turn_count / 状態異常 / 合法攻撃の有無は既に確認済みとする。
    ここで再度 get_legal_attack_indices で弾くと、条件の微妙なずれでログが一切付かないことがあるため、
    冗長な早期 return は行わない。
    NOOP も含め常に choice_log に記録する。にげ実行時のみ True。
    """
    from .turn import retreat
    from .state import _prizes_for_ko, PRIZE_COUNT

    p = state.active_player_state()
    if not p.active:
        return False

    # 勝ち確チェック: 今の攻撃で相手を倒してサイドを取り切れるなら、にげずに攻撃すべき。
    # にげるとエネを失って強い技が撃てなくなるリスクがある。
    opp = state.defending_player_state()
    if opp.active and opp.active.hp is not None and opp.active.hp > 0:
        our_dmg = _our_max_effective_damage(state)
        if our_dmg >= opp.active.hp:
            opp_prize_value = _prizes_for_ko(opp.active)
            our_remaining = len(p.prize_pile)
            if our_remaining <= opp_prize_value:
                # この攻撃で勝てる → にげない
                return False

    idx, dbg = retreat_before_attack_decision(state)

    if idx == 0:
        _log_retreat_before_attack_choice(state, 0, dbg, card_id=None)
        return False

    bench_i = idx - 1
    if bench_i < 0 or bench_i >= len(p.bench) or not dbg["mask"][idx]:
        _log_retreat_before_attack_choice(state, idx, dbg, card_id=None)
        return False

    twin_bi = _twin_preferred_bench(state)
    sol_bi = _solrock_preferred_bench(state)
    mega_bi = _mega_protect_preferred_bench(state)
    if bench_i == twin_bi:
        state.log(
            f"{state.player_name(state.current_player)}: 次の相手の攻撃できぜつ確定のため、"
            f"HP の多い同じポケモン（ベンチ）に入れ替えてから攻撃する"
        )
    elif bench_i == sol_bi:
        state.log(
            f"{state.player_name(state.current_player)}: バトル場のワザでは相手をきぜつできないが、"
            f"ベンチのソルロックなら倒せるため入れ替えてから攻撃する"
        )
    elif bench_i == mega_bi:
        state.log(
            f"{state.player_name(state.current_player)}: メガルカリオexがKO圏内のため、"
            f"サイド3枚を守るためベンチに退避してから攻撃する"
        )

    rc = getattr(p.bench[bench_i].card, "id", None) or getattr(p.bench[bench_i].card, "name", "")
    _log_retreat_before_attack_choice(state, idx, dbg, card_id=rc)
    return retreat(state, bench_i)
