"""
V(s) — 盤面状態の価値モデルを PyTorch で学習する（minimax 末端評価用）。

入力 JSONL（export_state_value_dataset.py）:
  {"state":[...], "value": float, ...}
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _default_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _turn_weight(turn: int, *, mode: str, max_turn: int = 200) -> float:
    if mode == "none":
        return 1.0
    t = max(0.0, min(1.0, float(turn) / float(max_turn)))
    v = t if mode == "linear" else math.sqrt(t)
    return 0.2 + 0.8 * v


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


class JsonlStateValueDataset(Dataset):
    def __init__(self, path: Path):
        self.rows: list[dict] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.rows.append(json.loads(line))
        if not self.rows:
            raise ValueError(f"dataset が空です: {path}")
        self.state_dim = len(self.rows[0]["state"])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        s = np.asarray(r["state"], dtype=np.float32)
        # "value" or "win" (selfplay_dataset の state_value 形式対応)
        value = float(r.get("value", r.get("win", 0.0)))
        turn = int(r.get("turn", 0))
        return s, np.float32(value), turn


@dataclass
class TrainMeta:
    data: str
    state_dim: int
    hidden: int
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    seed: int
    device: str
    turn_weight: str
    val_split: float


def main() -> None:
    parser = argparse.ArgumentParser(description="V(s) 盤面価値モデルを学習")
    parser.add_argument("--data", type=Path, required=True, help="入力 JSONL")
    parser.add_argument(
        "--out",
        type=Path,
        default=_REPO_ROOT / "models" / "state_value_mlp.pt",
        help="出力 .pt",
    )
    parser.add_argument(
        "--meta-out",
        type=Path,
        default=_REPO_ROOT / "models" / "state_value_mlp.meta.json",
        help="出力 meta JSON",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--turn-weight",
        type=str,
        default="none",
        choices=["none", "linear", "sqrt"],
    )
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--no-mps", action="store_true", help="cpu で実行")
    args = parser.parse_args()

    data_path = args.data if args.data.is_absolute() else (_REPO_ROOT / args.data)
    if not data_path.is_file():
        print(f"エラー: データセットが見つかりません: {data_path}", file=sys.stderr)
        sys.exit(1)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cpu") if args.no_mps else _default_device()

    ds = JsonlStateValueDataset(data_path)
    n = len(ds)
    idxs = np.arange(n)
    np.random.shuffle(idxs)
    n_val = int(n * float(args.val_split))
    val_idxs = set(map(int, idxs[:n_val].tolist()))

    class _Subset(Dataset):
        def __init__(self, base: Dataset, keep: set[int]):
            self.base = base
            self.keep = sorted(list(keep))

        def __len__(self):
            return len(self.keep)

        def __getitem__(self, i):
            return self.base[self.keep[i]]

    train_keep = set(range(n)) - val_idxs
    train_ds = _Subset(ds, train_keep)
    val_ds = _Subset(ds, val_idxs) if n_val > 0 else None

    def _collate(batch):
        s, y, turns = zip(*batch)
        s_t = torch.from_numpy(np.stack(s)).to(torch.float32)
        y_t = torch.from_numpy(np.asarray(y, dtype=np.float32)).to(torch.float32)
        tw = np.asarray([_turn_weight(int(t), mode=args.turn_weight) for t in turns], dtype=np.float32)
        w_t = torch.from_numpy(tw).to(torch.float32)
        return s_t, y_t, w_t

    train_loader = DataLoader(
        train_ds, batch_size=int(args.batch_size), shuffle=True, num_workers=0, collate_fn=_collate
    )
    val_loader = (
        DataLoader(val_ds, batch_size=int(args.batch_size), shuffle=False, num_workers=0, collate_fn=_collate)
        if val_ds
        else None
    )

    model = StateValueNet(state_dim=ds.state_dim, hidden=int(args.hidden)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    mse = nn.MSELoss(reduction="none")

    def _run(loader: DataLoader, train: bool) -> tuple[float, float]:
        model.train() if train else model.eval()
        loss_sum = 0.0
        count = 0
        err_sum = 0.0
        for s, y, w in loader:
            s = s.to(device)
            y = y.to(device)
            w = w.to(device)
            pred = model(s)
            loss_vec = mse(pred, y)
            loss = (loss_vec * w).mean()
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            with torch.no_grad():
                loss_sum += loss.item() * y.shape[0]
                count += y.shape[0]
                err_sum += torch.abs(pred - y).sum().item()
        mae = err_sum / max(1, count)
        return loss_sum / max(1, count), mae

    print(
        f"device={device} n_train={len(train_ds)} n_val={len(val_ds) if val_ds else 0} state_dim={ds.state_dim}"
    )
    for ep in range(1, int(args.epochs) + 1):
        tr_loss, tr_mae = _run(train_loader, train=True)
        if val_loader:
            va_loss, va_mae = _run(val_loader, train=False)
            print(
                f"epoch {ep}/{args.epochs} train loss={tr_loss:.6f} mae={tr_mae:.4f}  val loss={va_loss:.6f} mae={va_mae:.4f}"
            )
        else:
            print(f"epoch {ep}/{args.epochs} train loss={tr_loss:.6f} mae={tr_mae:.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "state_dim": ds.state_dim, "hidden": int(args.hidden)},
        args.out,
    )
    print(f"saved: {args.out}")

    meta = TrainMeta(
        data=str(data_path),
        state_dim=ds.state_dim,
        hidden=int(args.hidden),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        seed=int(args.seed),
        device=str(device),
        turn_weight=str(args.turn_weight),
        val_split=float(args.val_split),
    )
    args.meta_out.parent.mkdir(parents=True, exist_ok=True)
    args.meta_out.write_text(json.dumps(asdict(meta), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {args.meta_out}")


if __name__ == "__main__":
    main()
