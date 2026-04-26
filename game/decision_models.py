"""学習済みモデルでボスのターゲット選択・エネルギー配置をスコアリング。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
_BOSS_MODEL = None
_ENERGY_MODEL = None


def _load_boss_model():
    global _BOSS_MODEL
    if _BOSS_MODEL is not None:
        return True
    try:
        import torch, torch.nn as nn
        model_dir = _REPO_ROOT / "models" / "boss_target"
        if not (model_dir / "model.pt").exists():
            return False

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.state_enc = nn.Sequential(nn.Linear(181, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
                self.scorer = nn.Sequential(nn.Linear(64 + 7, 64), nn.ReLU(), nn.Linear(64, 1))
            def forward(self, s, t):
                bs, mt, _ = t.shape
                se = self.state_enc(s).unsqueeze(1).expand(-1, mt, -1)
                return self.scorer(torch.cat([se, t], dim=-1)).squeeze(-1)

        m = M()
        m.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True))
        m.eval()
        _BOSS_MODEL = m
        return True
    except Exception:
        return False


def _load_energy_model():
    global _ENERGY_MODEL
    if _ENERGY_MODEL is not None:
        return True
    try:
        import torch, torch.nn as nn
        model_dir = _REPO_ROOT / "models" / "energy_decision"
        if not (model_dir / "model.pt").exists():
            return False

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.state_enc = nn.Sequential(nn.Linear(181, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
                self.scorer = nn.Sequential(nn.Linear(64 + 5, 64), nn.ReLU(), nn.Linear(64, 1))
            def forward(self, s, c):
                bs, mc, _ = c.shape
                se = self.state_enc(s).unsqueeze(1).expand(-1, mc, -1)
                return self.scorer(torch.cat([se, c], dim=-1)).squeeze(-1)

        m = M()
        m.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True))
        m.eval()
        _ENERGY_MODEL = m
        return True
    except Exception:
        return False


_NAME_CATEGORIES = {
    "メガルカリオex": 0.9, "メガルカリオ": 0.85, "ルカリオ": 0.8,
    "リオル": 0.7, "ハリテヤマ": 0.5, "マクノシタ": 0.4,
    "ソルロック": 0.3, "ルナトーン": 0.2,
}


def score_boss_targets(state, player_idx: int) -> Optional[list[tuple[int, float]]]:
    """ボスの指令のベンチターゲットをモデルでスコアリング。
    Returns: [(bench_idx, score), ...] or None if model unavailable.
    """
    if not _load_boss_model():
        return None
    import torch
    from game.encoders import encode_state_v2
    from game.damage import _max_effective_damage_for_attacker
    from game.state import _prizes_for_ko

    p = state.players[player_idx]
    opp = state.players[1 - player_idx]
    if not opp.bench:
        return None

    state_vec = encode_state_v2(state, player_idx)
    targets = []
    indices = []

    # Active (bench_idx = -1 means "don't use boss")
    if opp.active and opp.active.hp and opp.active.hp > 0:
        dmg = _max_effective_damage_for_attacker(state, p.active, opp.active, player_idx) if p.active else 0
        targets.append([
            (opp.active.hp or 0) / 300.0,
            float(dmg >= opp.active.hp if opp.active.hp else False),
            _prizes_for_ko(opp.active) / 3.0,
            getattr(opp.active.card, "retreat_cost", 1) / 3.0,
            getattr(opp.active, "attached_energy", 0) / 3.0,
            (dmg or 0) / 300.0,
            1.0,
        ])
        indices.append(-1)

    for i, bp in enumerate(opp.bench):
        if not bp or not bp.hp or bp.hp <= 0:
            continue
        dmg = _max_effective_damage_for_attacker(state, p.active, bp, player_idx) if p.active else 0
        targets.append([
            (bp.hp or 0) / 300.0,
            float(dmg >= bp.hp),
            _prizes_for_ko(bp) / 3.0,
            getattr(bp.card, "retreat_cost", 1) / 3.0,
            getattr(bp, "attached_energy", 0) / 3.0,
            (dmg or 0) / 300.0,
            0.0,
        ])
        indices.append(i)

    if not targets:
        return None

    with torch.no_grad():
        s = torch.tensor([state_vec], dtype=torch.float32)
        t = torch.tensor([targets], dtype=torch.float32)
        scores = _BOSS_MODEL(s, t)[0].tolist()

    return [(idx, sc) for idx, sc in zip(indices, scores)]


def score_energy_candidates(state, player_idx: int, candidates: list[tuple]) -> Optional[list[float]]:
    """エネルギー配置候補をモデルでスコアリング。
    candidates: [(bench_idx_or_none, dmg), ...]
    Returns: [score, ...] or None.
    """
    if not _load_energy_model():
        return None
    import torch
    from game.encoders import encode_state_v2

    p = state.players[player_idx]
    state_vec = encode_state_v2(state, player_idx)

    feats = []
    for bench_idx, dmg in candidates:
        if bench_idx is None:
            card = p.active.card if p.active else None
            bp = p.active
        else:
            bp = p.bench[bench_idx]
            card = bp.card
        name = (getattr(card, "name", "") or "") if card else ""
        feats.append([
            1.0 if bench_idx is None else 0.0,
            (getattr(bp, "attached_energy", 0) or 0) / 3.0 if bp else 0.0,
            (dmg or 0) / 300.0,
            (getattr(bp, "hp", 0) or 0) / 300.0 if bp else 0.0,
            _NAME_CATEGORIES.get(name, 0.1),
        ])

    with torch.no_grad():
        s = torch.tensor([state_vec], dtype=torch.float32)
        f = torch.tensor([feats], dtype=torch.float32)
        scores = _ENERGY_MODEL(s, f)[0].tolist()

    return scores[:len(candidates)]
