"""
判断ポイントデータから学習モデルを訓練する。

ボスのターゲット選択とエネルギー配置の2つのモデルを訓練。
勝ち試合と負け試合の判断の違いから最適な選択を学ぶ。

使い方:
  python scripts/train_decisions.py
  python scripts/train_decisions.py --data datasets/mistakes.jsonl --epochs 100
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

STATE_DIM = 181  # encode_state_v2


# ---------------------------------------------------------------------------
# ボスのターゲット選択モデル
# ---------------------------------------------------------------------------

BOSS_TARGET_FEAT_DIM = 7  # hp, can_ko, prizes, retreat_cost, energy, dmg, is_active

class BossTargetModel(nn.Module):
    """各ターゲットのスコアを予測。高いスコア = 引っ張るべき。"""
    def __init__(self, state_dim=STATE_DIM, target_feat_dim=BOSS_TARGET_FEAT_DIM, hidden=64):
        super().__init__()
        self.state_enc = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.scorer = nn.Sequential(
            nn.Linear(hidden + target_feat_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state_vec, target_feats):
        """
        state_vec: (batch, state_dim)
        target_feats: (batch, max_targets, target_feat_dim)
        -> scores: (batch, max_targets)
        """
        bs, mt, _ = target_feats.shape
        se = self.state_enc(state_vec).unsqueeze(1).expand(-1, mt, -1)
        combined = torch.cat([se, target_feats], dim=-1)
        return self.scorer(combined).squeeze(-1)


def _boss_target_to_vec(t: dict, is_active: bool = False) -> list[float]:
    return [
        (t.get("hp", 0) or 0) / 300.0,
        float(t.get("can_ko", False)),
        t.get("prizes", 1) / 3.0,
        t.get("retreat_cost", 1) / 3.0,
        t.get("energy", 0) / 3.0,
        (t.get("dmg", 0) or 0) / 300.0,
        1.0 if is_active else 0.0,
    ]


# ---------------------------------------------------------------------------
# エネルギー配置モデル
# ---------------------------------------------------------------------------

ENERGY_CAND_FEAT_DIM = 5  # is_active, current_energy, dmg_after, hp, name_category

class EnergyAttachModel(nn.Module):
    """各候補のスコアを予測。高いスコア = エネを付けるべき。"""
    def __init__(self, state_dim=STATE_DIM, cand_feat_dim=ENERGY_CAND_FEAT_DIM, hidden=64):
        super().__init__()
        self.state_enc = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.scorer = nn.Sequential(
            nn.Linear(hidden + cand_feat_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state_vec, cand_feats):
        bs, mc, _ = cand_feats.shape
        se = self.state_enc(state_vec).unsqueeze(1).expand(-1, mc, -1)
        combined = torch.cat([se, cand_feats], dim=-1)
        return self.scorer(combined).squeeze(-1)


_NAME_CATEGORIES = {
    "メガルカリオex": 0.9, "メガルカリオ": 0.85, "ルカリオ": 0.8,
    "リオル": 0.7, "ハリテヤマ": 0.5, "マクノシタ": 0.4,
    "ソルロック": 0.3, "ルナトーン": 0.2,
}

def _energy_cand_to_vec(c: dict) -> list[float]:
    is_active = 1.0 if c["target"] == "active" else 0.0
    name_cat = _NAME_CATEGORIES.get(c.get("name", ""), 0.1)
    return [
        is_active,
        c.get("current_energy", 0) / 3.0,
        (c.get("dmg_after", 0) or 0) / 300.0,
        (c.get("hp", 0) or 0) / 300.0,
        name_cat,
    ]


# ---------------------------------------------------------------------------
# 学習
# ---------------------------------------------------------------------------

def train_boss_model(samples: list[dict], args):
    """ボスのターゲット選択モデルを訓練。

    学習方式: 勝ち試合のデータを「正解」、負け試合のデータを「不正解」として
    各ターゲットのスコアを学習。
    - 勝ち: 各ターゲットの特徴量 → ターゲットの価値（KO可能+サイド効率が高い = 高スコア）
    - 負け: 同じ状態で違う判断 → 低スコア
    """
    if len(samples) < 20:
        print("Boss: データ不足、スキップ")
        return

    # ターゲット特徴量+ラベル作成
    max_targets = max(len(s["bench_targets"]) + 1 for s in samples)  # +1 for active

    random.shuffle(samples)
    split = max(5, int(len(samples) * 0.1))
    val_s = samples[:split]
    train_s = samples[split:]

    def _to_tensors(slist):
        states, feats, labels = [], [], []
        for s in slist:
            states.append(s["state"])
            # ターゲット: active + bench
            targets = [s["active_target"]] + s["bench_targets"]
            f = [_boss_target_to_vec(t, i == 0) for i, t in enumerate(targets)]
            # ラベル: 勝ち試合ではKO可能+高サイドを高スコア
            # 負け試合ではKO不可+低サイドを低スコア
            won = s["won"]
            lbl = []
            for t in targets:
                score = 0.0
                if t.get("can_ko"):
                    score += t.get("prizes", 1) * 0.3
                score += t.get("retreat_cost", 0) * 0.1  # 逃げコスト高い = 縛れる
                score -= t.get("energy", 0) * 0.1  # エネ多い = 危険（KO不可時）
                if won:
                    lbl.append(score)
                else:
                    lbl.append(-score)  # 負け試合は逆
            while len(f) < max_targets:
                f.append([0.0] * BOSS_TARGET_FEAT_DIM)
                lbl.append(-99.0)  # padding
            feats.append(f[:max_targets])
            labels.append(lbl[:max_targets])
        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(feats, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.float32),
        )

    s_t, f_t, l_t = _to_tensors(train_s)
    s_v, f_v, l_v = _to_tensors(val_s)

    model = BossTargetModel()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    best_val = float("inf")

    for epoch in range(args.epochs):
        model.train()
        indices = list(range(len(train_s)))
        random.shuffle(indices)
        total_loss = 0.0
        n_batches = 0
        for i in range(0, len(indices), args.batch_size):
            batch = indices[i:i + args.batch_size]
            scores = model(s_t[batch], f_t[batch])
            labels = l_t[batch]
            mask = (labels > -50).float()
            loss = ((scores - labels) ** 2 * mask).sum() / mask.sum().clamp(min=1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        model.eval()
        with torch.no_grad():
            v_scores = model(s_v, f_v)
            v_mask = (l_v > -50).float()
            val_loss = ((v_scores - l_v) ** 2 * v_mask).sum() / v_mask.sum().clamp(min=1)

        if (epoch + 1) % 20 == 0:
            print(f"  Boss Epoch {epoch+1}: train={total_loss/max(n_batches,1):.4f} val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            model_dir = _REPO_ROOT / "models" / "boss_target"
            model_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), model_dir / "model.pt")
            with open(model_dir / "meta.json", "w") as mf:
                json.dump({
                    "state_dim": STATE_DIM,
                    "target_feat_dim": BOSS_TARGET_FEAT_DIM,
                    "hidden": 64,
                    "max_targets": max_targets,
                    "n_train": len(train_s),
                    "n_val": len(val_s),
                    "best_val_loss": float(best_val),
                }, mf, indent=2)

    print(f"  Boss完了: val_loss={best_val:.4f}, {len(train_s)}件")


def train_energy_model(samples: list[dict], args):
    """エネルギー配置モデルを訓練。"""
    if len(samples) < 20:
        print("Energy: データ不足、スキップ")
        return

    max_cands = max(len(s["candidates"]) for s in samples)

    random.shuffle(samples)
    split = max(5, int(len(samples) * 0.1))
    val_s = samples[:split]
    train_s = samples[split:]

    def _to_tensors(slist):
        states, feats, labels = [], [], []
        for s in slist:
            states.append(s["state"])
            cands = s["candidates"]
            f = [_energy_cand_to_vec(c) for c in cands]
            # ラベル: ダメージ効率 × 勝敗方向
            won = s["won"]
            lbl = []
            for c in cands:
                score = (c.get("dmg_after", 0) or 0) / 300.0
                # メガルカリオex/リオルへの付与を重視
                name_bonus = _NAME_CATEGORIES.get(c.get("name", ""), 0.0) * 0.5
                score += name_bonus
                if won:
                    lbl.append(score)
                else:
                    lbl.append(-score * 0.5)  # 負けは弱いペナルティ
            while len(f) < max_cands:
                f.append([0.0] * ENERGY_CAND_FEAT_DIM)
                lbl.append(-99.0)
            feats.append(f[:max_cands])
            labels.append(lbl[:max_cands])
        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(feats, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.float32),
        )

    s_t, f_t, l_t = _to_tensors(train_s)
    s_v, f_v, l_v = _to_tensors(val_s)

    model = EnergyAttachModel()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    best_val = float("inf")

    for epoch in range(args.epochs):
        model.train()
        indices = list(range(len(train_s)))
        random.shuffle(indices)
        total_loss = 0.0
        n_batches = 0
        for i in range(0, len(indices), args.batch_size):
            batch = indices[i:i + args.batch_size]
            scores = model(s_t[batch], f_t[batch])
            labels = l_t[batch]
            mask = (labels > -50).float()
            loss = ((scores - labels) ** 2 * mask).sum() / mask.sum().clamp(min=1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        model.eval()
        with torch.no_grad():
            v_scores = model(s_v, f_v)
            v_mask = (l_v > -50).float()
            val_loss = ((v_scores - l_v) ** 2 * v_mask).sum() / v_mask.sum().clamp(min=1)

        if (epoch + 1) % 20 == 0:
            print(f"  Energy Epoch {epoch+1}: train={total_loss/max(n_batches,1):.4f} val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            model_dir = _REPO_ROOT / "models" / "energy_decision"
            model_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), model_dir / "model.pt")
            with open(model_dir / "meta.json", "w") as mf:
                json.dump({
                    "state_dim": STATE_DIM,
                    "cand_feat_dim": ENERGY_CAND_FEAT_DIM,
                    "hidden": 64,
                    "max_cands": max_cands,
                    "n_train": len(train_s),
                    "n_val": len(val_s),
                    "best_val_loss": float(best_val),
                }, mf, indent=2)

    print(f"  Energy完了: val_loss={best_val:.4f}, {len(train_s)}件")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=_REPO_ROOT / "datasets" / "mistakes.jsonl")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    with open(args.data) as f:
        all_data = [json.loads(line) for line in f]

    boss = [r for r in all_data if r["type"] == "boss_target"]
    energy = [r for r in all_data if r["type"] == "energy_attach"]

    print(f"データ: boss={len(boss)}, energy={len(energy)}")
    print("\n--- Boss Target Model ---")
    train_boss_model(boss, args)
    print("\n--- Energy Attach Model ---")
    train_energy_model(energy, args)


if __name__ == "__main__":
    main()
