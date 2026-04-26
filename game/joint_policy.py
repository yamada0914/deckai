"""
サポート＋エネルギーの結合Q値ポリシー。

action = (support_id, energy_target) を同時に決定する。
Q(state, support_onehot_15 + energy_onehot_6) → value

推論時: 合法な全組み合わせ（最大 15×6=90 通り）を評価し、最善を選択。
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .encoders import encode_state_opening
from .q_models import load_q_value_model_pt
from .state import GameState
from .support_policy import KNOWN_SUPPORT_IDS, SUPPORT_ACTION_DIM, legal_support_mask

ENERGY_ACTION_DIM = 6  # active(0) + bench(0..4)
JOINT_ACTION_DIM = SUPPORT_ACTION_DIM * ENERGY_ACTION_DIM  # 15 * 6 = 90


def _get_joint_q_path(state: GameState) -> str | None:
    """現在のプレイヤーの joint Q-model パスを返す。"""
    by_player = getattr(state, "joint_q_model_path_by_player", [None, None])
    return by_player[state.current_player] if by_player[state.current_player] else None


def joint_policy_decision(
    state: GameState,
    legal_energy_targets: list[int] | None = None,
) -> tuple[int, int, dict[str, Any]]:
    """
    結合 Q-model でサポートとエネルギー付与先を同時に決定する。

    Parameters
    ----------
    state : GameState
    legal_energy_targets : list[int] | None
        合法なエネルギー付与先のインデックス (0=active, 1..5=bench)。
        None の場合は全 6 候補が合法とする。

    Returns
    -------
    (support_action_id, energy_action_id, debug_dict)
    """
    q_path = _get_joint_q_path(state)
    if q_path is None:
        return 0, 0, {"joint_q": False}

    q_model = load_q_value_model_pt(q_path)
    vec = encode_state_opening(state, state.current_player)
    state_arr = np.asarray(vec, dtype=np.float32)

    # 合法マスク
    sup_mask = legal_support_mask(state)
    if legal_energy_targets is None:
        ene_mask = [True] * ENERGY_ACTION_DIM
    else:
        ene_mask = [i in legal_energy_targets for i in range(ENERGY_ACTION_DIM)]

    # 全 90 組み合わせの Q 値をバッチ計算
    # action encoding: combined_id = sup_id * 6 + ene_id
    n_actions = JOINT_ACTION_DIM  # 90
    state_rep = np.tile(state_arr, (n_actions, 1))  # (90, state_dim)
    action_onehot = np.eye(n_actions, dtype=np.float32)  # (90, 90)
    x = np.concatenate([state_rep, action_onehot], axis=1)  # (90, state_dim+90)

    import torch
    xt = torch.from_numpy(x).to(q_model.device)
    with torch.no_grad():
        q_vals = q_model.model(xt).detach().cpu().numpy()  # (90,)

    # 合法組み合わせから最善を選択
    best_q = float("-inf")
    best_sup = 0
    best_ene = 0
    for sid in range(SUPPORT_ACTION_DIM):
        if not sup_mask[sid]:
            continue
        for eid in range(ENERGY_ACTION_DIM):
            if not ene_mask[eid]:
                continue
            combined = sid * ENERGY_ACTION_DIM + eid
            q = float(q_vals[combined])
            if q > best_q:
                best_q = q
                best_sup = sid
                best_ene = eid

    debug = {
        "joint_q": True,
        "best_q": best_q,
        "support_action_id": best_sup,
        "energy_action_id": best_ene,
    }
    return best_sup, best_ene, debug
