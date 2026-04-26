"""
学習した policy π(a|s) の推論ユーティリティ。

- `.pt`（PyTorch）を読み込み、パスごとにモデルをキャッシュする。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np

import torch
import torch.nn as nn


class PiNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 64):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # logits


@dataclass(frozen=True)
class LoadedPiModel:
    state_dim: int
    action_dim: int
    model: PiNet
    device: torch.device
    state_mean: "np.ndarray | None" = None
    state_std: "np.ndarray | None" = None
    encoder_name: str = "basic"  # "basic" | "opening"

    def _normalize(self, s: np.ndarray) -> np.ndarray:
        if self.state_mean is not None and self.state_std is not None:
            return (s - self.state_mean) / (self.state_std + 1e-8)
        return s

    def predict_logits_one(self, state_vec: list[float]) -> list[float]:
        s = self._normalize(np.asarray(state_vec, dtype=np.float32))
        xt = torch.from_numpy(s).to(self.device).to(torch.float32)
        with torch.no_grad():
            logits = self.model(xt.unsqueeze(0))[0]
        return [float(x) for x in logits.detach().cpu().tolist()]

    def predict_probs_one(self, state_vec: list[float]) -> list[float]:
        s = self._normalize(np.asarray(state_vec, dtype=np.float32))
        xt = torch.from_numpy(s).to(self.device).to(torch.float32)
        with torch.no_grad():
            logits = self.model(xt.unsqueeze(0))[0]
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        return [float(x) for x in probs.tolist()]


_CACHE: Dict[str, LoadedPiModel] = {}


def _default_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_pi_model_pt(path: str | Path, *, device: torch.device | None = None) -> LoadedPiModel:
    path = str(path)
    if path in _CACHE:
        return _CACHE[path]
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    state_dim = int(ckpt["state_dim"])
    action_dim = int(ckpt["action_dim"])
    hidden = int(ckpt.get("hidden", 64))
    dev = device or _default_device()
    model = PiNet(state_dim=state_dim, action_dim=action_dim, hidden=hidden)
    model.load_state_dict(ckpt["state_dict"])
    model.to(dev)
    model.eval()
    state_mean = np.asarray(ckpt["state_mean"], dtype=np.float32) if "state_mean" in ckpt else None
    state_std = np.asarray(ckpt["state_std"], dtype=np.float32) if "state_std" in ckpt else None
    encoder_name = str(ckpt.get("encoder_name", "basic"))
    loaded = LoadedPiModel(
        state_dim=state_dim, action_dim=action_dim,
        model=model, device=dev,
        state_mean=state_mean, state_std=state_std,
        encoder_name=encoder_name,
    )
    _CACHE[path] = loaded
    return loaded

