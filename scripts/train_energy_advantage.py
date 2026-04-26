"""
Energy Q を Advantage 形式で学習する。

A(s,a) = win - V(s) を教師信号。
action_dim = BENCH_SIZE + 1 = 6 (0=active, 1-5=bench[0-4])

入力 JSONL (selfplay_dataset.py --choice-type energy_attach):
  {"state": [...], "action_id": int, "win": 0|1, "turn": int, ...}
"""
from __future__ import annotations

import argparse
import json
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


class EnergyAdvantageNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class EnergyAdvantageDataset(Dataset):
    def __init__(self, path: Path, v_model_path: Path):
        from game.value_models import load_state_value_model_pt

        self.rows: list[dict] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))
        if not self.rows:
            raise ValueError(f"dataset empty: {path}")

        self.state_dim = len(self.rows[0]["state"])
        v_model = load_state_value_model_pt(str(v_model_path))
        for r in self.rows:
            v_s = float(v_model.predict_one(r["state"]))
            r["advantage"] = float(r["win"]) - v_s

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        s = np.asarray(r["state"], dtype=np.float32)
        return s, int(r["action_id"]), np.float32(r["advantage"])


def main():
    parser = argparse.ArgumentParser(description="Energy Q (Advantage) を学習")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--v-model", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=_REPO_ROOT / "models" / "energy_advantage.pt")
    parser.add_argument("--meta-out", type=Path, default=_REPO_ROOT / "models" / "energy_advantage.meta.json")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--no-mps", action="store_true")
    args = parser.parse_args()

    from game.energy_policy import ENERGY_ATTACH_ACTION_DIM

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    device = torch.device("cpu") if args.no_mps else _default_device()
    ds = EnergyAdvantageDataset(
        args.data if args.data.is_absolute() else _REPO_ROOT / args.data,
        args.v_model if args.v_model.is_absolute() else _REPO_ROOT / args.v_model,
    )
    n = len(ds)
    action_dim = ENERGY_ATTACH_ACTION_DIM
    print(f"device={device} data={n} state_dim={ds.state_dim} action_dim={action_dim}")

    idxs = np.arange(n)
    np.random.shuffle(idxs)
    n_val = int(n * args.val_split)
    val_set = set(map(int, idxs[:n_val].tolist()))
    train_set = set(range(n)) - val_set

    class _Sub(Dataset):
        def __init__(self, base, keep):
            self.base, self.keep = base, sorted(list(keep))
        def __len__(self):
            return len(self.keep)
        def __getitem__(self, i):
            return self.base[self.keep[i]]

    def _collate(batch):
        ss, aa, advs = zip(*batch)
        return torch.from_numpy(np.stack(ss)), torch.tensor(aa, dtype=torch.long), torch.tensor(advs, dtype=torch.float32)

    train_loader = DataLoader(_Sub(ds, train_set), batch_size=args.batch_size, shuffle=True, collate_fn=_collate)
    val_loader = DataLoader(_Sub(ds, val_set), batch_size=args.batch_size, shuffle=False, collate_fn=_collate) if n_val else None

    model = EnergyAdvantageNet(ds.state_dim, action_dim, args.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    for ep in range(1, args.epochs + 1):
        model.train()
        t_loss, t_n = 0.0, 0
        for s, a, adv in train_loader:
            s, a, adv = s.to(device), a.to(device), adv.to(device)
            q = model(s).gather(1, a.unsqueeze(1)).squeeze(1)
            loss = ((q - adv) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            t_loss += loss.item() * s.size(0); t_n += s.size(0)

        v_str = ""
        if val_loader:
            model.eval()
            v_loss, v_n = 0.0, 0
            with torch.no_grad():
                for s, a, adv in val_loader:
                    s, a, adv = s.to(device), a.to(device), adv.to(device)
                    q = model(s).gather(1, a.unsqueeze(1)).squeeze(1)
                    v_loss += ((q - adv) ** 2).mean().item() * s.size(0); v_n += s.size(0)
            v_str = f"  val={v_loss/max(v_n,1):.6f}"

        if ep % 5 == 0 or ep == 1:
            print(f"epoch {ep}/{args.epochs}  train={t_loss/max(t_n,1):.6f}{v_str}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out)
    meta = {"state_dim": ds.state_dim, "action_dim": action_dim, "hidden": args.hidden, "epochs": args.epochs, "data_rows": n}
    with open(args.meta_out, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
