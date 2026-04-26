"""
学習した value モデルを推論するユーティリティ。

- ValueNet / LoadedValueModel: V(s, a) — attack 選択用
- StateValueNet / LoadedStateValueModel: V(s) — 盤面状態の価値（minimax 末端評価用）

価値ラベルは choice_log に保存されている eval（行動直後の盤面評価）を使う想定。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np

import torch
import torch.nn as nn


class ValueNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 64):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state: torch.Tensor, action_onehot: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action_onehot], dim=-1)
        return self.net(x).squeeze(-1)


@dataclass(frozen=True)
class LoadedValueModel:
    state_dim: int
    action_dim: int
    model: ValueNet
    device: torch.device

    def predict_one(self, state_vec: list[float], action_id: int) -> float:
        s = np.asarray(state_vec, dtype=np.float32)
        a = np.zeros((self.action_dim,), dtype=np.float32)
        if 0 <= action_id < self.action_dim:
            a[action_id] = 1.0
        st = torch.from_numpy(s).to(self.device).to(torch.float32)
        at = torch.from_numpy(a).to(self.device).to(torch.float32)
        with torch.no_grad():
            v = self.model(st.unsqueeze(0), at.unsqueeze(0))[0]
        return float(v.item())


_CACHE: Dict[str, LoadedValueModel] = {}


# ---------------------------------------------------------------------------
# V(s) — state-only value model（minimax 末端評価用）
# ---------------------------------------------------------------------------

class StateValueNet(nn.Module):
    def __init__(self, state_dim: int, hidden: int = 64):
        super().__init__()
        self.state_dim = state_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)


@dataclass(frozen=True)
class LoadedStateValueModel:
    state_dim: int
    model: StateValueNet
    device: torch.device

    def predict_one(self, state_vec: list[float]) -> float:
        s = torch.from_numpy(np.asarray(state_vec, dtype=np.float32)).to(self.device)
        with torch.no_grad():
            v = self.model(s.unsqueeze(0))[0]
        return float(v.item())


_STATE_VALUE_CACHE: Dict[str, LoadedStateValueModel] = {}


def load_state_value_model_pt(
    path: str | Path, *, device: torch.device | None = None
) -> LoadedStateValueModel:
    path = str(path)
    if path in _STATE_VALUE_CACHE:
        return _STATE_VALUE_CACHE[path]
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    state_dim = int(ckpt["state_dim"])
    hidden = int(ckpt.get("hidden", 64))
    dev = device or _default_device()
    model = StateValueNet(state_dim=state_dim, hidden=hidden)
    model.load_state_dict(ckpt["state_dict"])
    model.to(dev)
    model.eval()
    loaded = LoadedStateValueModel(state_dim=state_dim, model=model, device=dev)
    _STATE_VALUE_CACHE[path] = loaded
    return loaded


def _default_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_value_model_pt(path: str | Path, *, device: torch.device | None = None) -> LoadedValueModel:
    path = str(path)
    if path in _CACHE:
        return _CACHE[path]
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    state_dim = int(ckpt["state_dim"])
    action_dim = int(ckpt["action_dim"])
    hidden = int(ckpt.get("hidden", 64))
    dev = device or _default_device()
    model = ValueNet(state_dim=state_dim, action_dim=action_dim, hidden=hidden)
    model.load_state_dict(ckpt["state_dict"])
    model.to(dev)
    model.eval()
    loaded = LoadedValueModel(state_dim=state_dim, action_dim=action_dim, model=model, device=dev)
    _CACHE[path] = loaded
    return loaded

