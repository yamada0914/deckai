"""
学習した Q(s,a)=P(win|state,action) の推論ユーティリティ。

- `.pt`（PyTorch）を優先して読み込む。
- 実行時のオーバーヘッドを抑えるため、パスごとにモデルをキャッシュする。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

import torch
import torch.nn as nn


class QNet(nn.Module):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # logits


@dataclass(frozen=True)
class LoadedQModel:
    state_dim: int
    action_dim: int
    model: QNet
    device: torch.device

    def predict_one(self, state_vec: list[float], action_id: int) -> float:
        """
        1 サンプルの win probability を返す。
        """
        s = np.asarray(state_vec, dtype=np.float32)
        a = np.zeros((self.action_dim,), dtype=np.float32)
        if 0 <= action_id < self.action_dim:
            a[action_id] = 1.0
        x = np.concatenate([s, a], axis=0)
        xt = torch.from_numpy(x).to(self.device).to(torch.float32)
        with torch.no_grad():
            logit = self.model(xt.unsqueeze(0))[0]
            p = torch.sigmoid(logit).item()
        return float(p)


@dataclass(frozen=True)
class LoadedQValueModel:
    """回帰 Q-value モデル: Q(s,a) → expected prize_delta / scale。sigmoid なし。"""

    state_dim: int
    action_dim: int
    model: QNet
    device: torch.device

    def predict_all(self, state_vec: list[float]) -> list[float]:
        """全 action_dim の Q 値をまとめて返す（バッチ推論）。"""
        s = np.asarray(state_vec, dtype=np.float32)
        s_rep = np.tile(s, (self.action_dim, 1))  # (action_dim, state_dim)
        a_mat = np.eye(self.action_dim, dtype=np.float32)  # (action_dim, action_dim)
        x = np.concatenate([s_rep, a_mat], axis=1)  # (action_dim, state_dim+action_dim)
        xt = torch.from_numpy(x).to(self.device)
        with torch.no_grad():
            vals = self.model(xt).detach().cpu().tolist()
        return [float(v) for v in vals]


_CACHE: Dict[str, LoadedQModel] = {}
_QVAL_CACHE: Dict[str, "LoadedQValueModel"] = {}


def _default_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_q_value_model_pt(path: str | Path, *, device: torch.device | None = None) -> "LoadedQValueModel":
    """回帰 Q-value モデル（prize_delta 予測）を読み込む。sigmoid なし。"""
    path = str(path)
    if path in _QVAL_CACHE:
        return _QVAL_CACHE[path]
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    state_dim = int(ckpt["state_dim"])
    action_dim = int(ckpt["action_dim"])
    hidden = int(ckpt.get("hidden", 64))
    dev = device or _default_device()
    model = QNet(state_dim=state_dim, action_dim=action_dim, hidden=hidden)
    model.load_state_dict(ckpt["state_dict"])
    model.to(dev)
    model.eval()
    loaded = LoadedQValueModel(state_dim=state_dim, action_dim=action_dim, model=model, device=dev)
    _QVAL_CACHE[path] = loaded
    return loaded


class _AdvantageNet(torch.nn.Module):
    """state → 全action の advantage を出力（train_support_advantage.py と同構造）。"""
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, action_dim),
        )
    def forward(self, state):
        return self.net(state)


@dataclass(frozen=True)
class LoadedAdvantageModel:
    """Advantage モデル: A(s) → 全action のスコア。LoadedQValueModel と同じ predict_all インターフェース。"""
    state_dim: int
    action_dim: int
    model: _AdvantageNet
    device: torch.device

    def predict_all(self, state_vec: list[float]) -> list[float]:
        s = torch.from_numpy(np.asarray(state_vec, dtype=np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            vals = self.model(s)[0].detach().cpu().tolist()
        return [float(v) for v in vals]


_ADV_CACHE: Dict[str, LoadedAdvantageModel] = {}


def load_advantage_model_pt(path: str | Path, *, device: torch.device | None = None) -> LoadedAdvantageModel:
    """Advantage モデルを読み込む。meta.json から state_dim/action_dim/hidden を取得。"""
    import json
    path = str(path)
    if path in _ADV_CACHE:
        return _ADV_CACHE[path]
    meta_path = path.replace(".pt", ".meta.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)
    state_dim = int(meta["state_dim"])
    action_dim = int(meta["action_dim"])
    hidden = int(meta.get("hidden", 128))
    dev = device or _default_device()
    model = _AdvantageNet(state_dim=state_dim, action_dim=action_dim, hidden=hidden)
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt)
    model.to(dev)
    model.eval()
    loaded = LoadedAdvantageModel(state_dim=state_dim, action_dim=action_dim, model=model, device=dev)
    _ADV_CACHE[path] = loaded
    return loaded


def load_q_model_pt(path: str | Path, *, device: torch.device | None = None) -> LoadedQModel:
    path = str(path)
    if path in _CACHE:
        return _CACHE[path]
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    state_dim = int(ckpt["state_dim"])
    action_dim = int(ckpt["action_dim"])
    hidden = int(ckpt.get("hidden", 64))
    dev = device or _default_device()
    model = QNet(state_dim=state_dim, action_dim=action_dim, hidden=hidden)
    model.load_state_dict(ckpt["state_dict"])
    model.to(dev)
    model.eval()
    loaded = LoadedQModel(state_dim=state_dim, action_dim=action_dim, model=model, device=dev)
    _CACHE[path] = loaded
    return loaded

