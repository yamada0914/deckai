"""
Support Q を Advantage 形式で学習する。

A(s,a) = win - V(s) を教師信号として、Q(s,a) ≈ V(s) + A(s,a) を学習。
V(s) は事前学習済みモデルから推論。

入力 JSONL (selfplay_dataset.py --choice-type support_advantage):
  {"state": [...], "action_id": int, "win": 0|1, "turn": int, "hadouzuki_used": 0|1, ...}

出力: .pt モデル + .meta.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


def _default_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class SupportAdvantageNet(nn.Module):
    """Q(s, a) = V(s) + A(s, a) のうち A(s,a) 部分を学習するネットワーク。"""
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128):
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

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """全action の advantage を返す (batch, action_dim)"""
        return self.net(state)


class SupportAdvantageDataset(Dataset):
    def __init__(self, path: Path, v_model_path: Path, accident_weight: float = 0.3):
        from game.value_models import load_state_value_model_pt

        self.rows: list[dict] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))
        if not self.rows:
            raise ValueError(f"dataset が空: {path}")

        self.state_dim = len(self.rows[0]["state"])

        # V(s) を推論して advantage = win - V(s) を計算
        v_model = load_state_value_model_pt(str(v_model_path))
        for r in self.rows:
            state_vec = r["state"]
            v_s = float(v_model.predict_one(state_vec))
            r["advantage"] = float(r["win"]) - v_s
            # 事故試合の重み
            r["sample_weight"] = accident_weight if not r.get("hadouzuki_used", 1) else 1.0

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        s = np.asarray(r["state"], dtype=np.float32)
        action_id = int(r["action_id"])
        advantage = np.float32(r["advantage"])
        weight = np.float32(r["sample_weight"])
        turn = int(r.get("turn", 0))
        return s, action_id, advantage, weight, turn


def main() -> None:
    parser = argparse.ArgumentParser(description="Support Q (Advantage形式) を学習")
    parser.add_argument("--data", type=Path, required=True, help="入力 JSONL")
    parser.add_argument("--v-model", type=Path, required=True, help="V(s) モデル .pt")
    parser.add_argument("--out", type=Path, default=_REPO_ROOT / "models" / "support_advantage.pt")
    parser.add_argument("--meta-out", type=Path, default=_REPO_ROOT / "models" / "support_advantage.meta.json")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--accident-weight", type=float, default=0.3, help="はどうづき未使用試合の重み")
    parser.add_argument("--no-mps", action="store_true")
    args = parser.parse_args()

    from game.support_policy import SUPPORT_ACTION_DIM

    data_path = args.data if args.data.is_absolute() else (_REPO_ROOT / args.data)
    v_model_path = args.v_model if args.v_model.is_absolute() else (_REPO_ROOT / args.v_model)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    device = torch.device("cpu") if args.no_mps else _default_device()
    print(f"device={device}")

    ds = SupportAdvantageDataset(data_path, v_model_path, accident_weight=args.accident_weight)
    n = len(ds)
    print(f"data: {n} rows, state_dim={ds.state_dim}, action_dim={SUPPORT_ACTION_DIM}")

    # train/val split
    idxs = np.arange(n)
    np.random.shuffle(idxs)
    n_val = int(n * args.val_split)
    val_set = set(map(int, idxs[:n_val].tolist()))
    train_set = set(range(n)) - val_set

    class _Sub(Dataset):
        def __init__(self, base, keep):
            self.base = base
            self.keep = sorted(list(keep))
        def __len__(self):
            return len(self.keep)
        def __getitem__(self, i):
            return self.base[self.keep[i]]

    train_ds = _Sub(ds, train_set)
    val_ds = _Sub(ds, val_set) if n_val > 0 else None

    def _collate(batch):
        states, actions, advantages, weights, turns = zip(*batch)
        return (
            torch.from_numpy(np.stack(states)),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(advantages, dtype=torch.float32),
            torch.tensor(weights, dtype=torch.float32),
        )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=_collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=_collate) if val_ds else None

    model = SupportAdvantageNet(
        state_dim=ds.state_dim,
        action_dim=SUPPORT_ACTION_DIM,
        hidden=args.hidden,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        count = 0
        for s_batch, a_batch, adv_batch, w_batch in train_loader:
            s_batch = s_batch.to(device)
            a_batch = a_batch.to(device)
            adv_batch = adv_batch.to(device)
            w_batch = w_batch.to(device)

            all_q = model(s_batch)  # (batch, action_dim)
            q_selected = all_q.gather(1, a_batch.unsqueeze(1)).squeeze(1)  # (batch,)

            # weighted MSE: advantage を予測
            loss = (w_batch * (q_selected - adv_batch) ** 2).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * s_batch.size(0)
            count += s_batch.size(0)

        train_loss = total_loss / max(count, 1)

        # validation
        val_str = ""
        if val_loader:
            model.eval()
            val_loss_total = 0.0
            val_count = 0
            with torch.no_grad():
                for s_batch, a_batch, adv_batch, w_batch in val_loader:
                    s_batch = s_batch.to(device)
                    a_batch = a_batch.to(device)
                    adv_batch = adv_batch.to(device)
                    w_batch = w_batch.to(device)
                    all_q = model(s_batch)
                    q_selected = all_q.gather(1, a_batch.unsqueeze(1)).squeeze(1)
                    loss = (w_batch * (q_selected - adv_batch) ** 2).mean()
                    val_loss_total += loss.item() * s_batch.size(0)
                    val_count += s_batch.size(0)
            val_loss = val_loss_total / max(val_count, 1)
            val_str = f"  val_loss={val_loss:.6f}"

        print(f"epoch {epoch}/{args.epochs}  train_loss={train_loss:.6f}{val_str}")

    # save
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f"saved: {args.out}")

    meta = {
        "state_dim": ds.state_dim,
        "action_dim": SUPPORT_ACTION_DIM,
        "hidden": args.hidden,
        "epochs": args.epochs,
        "data_rows": n,
        "accident_weight": args.accident_weight,
    }
    with open(args.meta_out, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"saved: {args.meta_out}")


if __name__ == "__main__":
    main()
