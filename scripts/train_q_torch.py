"""
JSONL dataset から Q(s, a)=P(win | state, action) を PyTorch で学習する。

入力（1 行 JSON）:
  {"state":[...], "action_id":2, "win":1, "turn": 12, ...}

モデル:
  x = state + onehot(action_id)
  p = sigmoid(MLP(x))

出力:
  - モデル重み: models/q_energy_attach_mlp.pt
  - メタ情報:   models/q_energy_attach_mlp.meta.json
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
    if mode == "linear":
        v = t
    else:
        v = math.sqrt(t)
    return 0.2 + 0.8 * v


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

    def forward(self, state: torch.Tensor, action_onehot: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action_onehot], dim=-1)
        return self.net(x).squeeze(-1)  # logits


class JsonlQDataset(Dataset):
    def __init__(self, path: Path, action_dim: int):
        self.path = path
        self.action_dim = action_dim
        self.rows: list[dict] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.rows.append(json.loads(line))
        if not self.rows:
            raise ValueError(
                f"dataset が空です: {path}\n"
                "対戦 pkl に frame_index が無い（古い記録）だと export で行が全部スキップされます。"
                "run_training_games.py で新規に対戦を記録してから export_policy_dataset.py を再実行してください。"
            )
        self.state_dim = len(self.rows[0]["state"])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        state = np.asarray(r["state"], dtype=np.float32)
        action_id = int(r["action_id"])
        win = 1.0 if int(r.get("win", 0)) == 1 else 0.0
        turn = int(r.get("turn", 0))
        # onehot は collate で作るよりここで作る方が単純
        a = np.zeros((self.action_dim,), dtype=np.float32)
        if 0 <= action_id < self.action_dim:
            a[action_id] = 1.0
        return state, a, np.float32(win), turn


@dataclass
class TrainMeta:
    data: str
    state_dim: int
    action_dim: int
    hidden: int
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    seed: int
    device: str
    turn_weight: str
    pos_weight: float
    val_split: float


def main() -> None:
    parser = argparse.ArgumentParser(description="Q(s,a)=P(win|state,action) を PyTorch MLP で学習")
    parser.add_argument("--data", type=Path, required=True, help="入力 JSONL（例: datasets/policy_energy_attach.jsonl）")
    parser.add_argument("--out", type=Path, default=_REPO_ROOT / "models" / "q_energy_attach_mlp.pt", help="出力 .pt")
    parser.add_argument("--meta-out", type=Path, default=_REPO_ROOT / "models" / "q_energy_attach_mlp.meta.json", help="出力 meta JSON")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--action-dim", type=int, default=6, help="energy_attach は 6=active+bench5")
    parser.add_argument("--turn-weight", type=str, default="none", choices=["none", "linear", "sqrt"])
    parser.add_argument("--pos-weight", type=float, default=1.0, help="win=1 の重み（BCE の pos_weight 相当）")
    parser.add_argument("--val-split", type=float, default=0.1, help="検証用に取り分ける割合")
    parser.add_argument("--no-mps", action="store_true", help="MPS を使わない（cpu で実行）")
    args = parser.parse_args()

    # 相対パスはリポジトリルート基準に解決（どこから実行しても同じファイルを指す）
    data_path = args.data if args.data.is_absolute() else (_REPO_ROOT / args.data)
    if not data_path.is_file():
        print(f"エラー: データセットが見つかりません: {data_path}", file=sys.stderr)
        print("先に以下で JSONL を作成してください:", file=sys.stderr)
        print("  python scripts/export_policy_dataset.py --battles-dir battles/train_belt --out datasets/policy_energy_attach.jsonl", file=sys.stderr)
        sys.exit(1)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cpu") if args.no_mps else _default_device()

    try:
        ds = JsonlQDataset(data_path, action_dim=int(args.action_dim))
    except ValueError as e:
        if "dataset が空です" in str(e):
            print(str(e), file=sys.stderr)
            sys.exit(1)
        raise
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
        states, actions, wins, turns = zip(*batch)
        s = torch.from_numpy(np.stack(states)).to(torch.float32)
        a = torch.from_numpy(np.stack(actions)).to(torch.float32)
        y = torch.from_numpy(np.asarray(wins, dtype=np.float32)).to(torch.float32)
        tw = np.asarray([_turn_weight(int(t), mode=args.turn_weight) for t in turns], dtype=np.float32)
        w = torch.from_numpy(tw).to(torch.float32)
        return s, a, y, w

    train_loader = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=True, num_workers=0, collate_fn=_collate)
    val_loader = DataLoader(val_ds, batch_size=int(args.batch_size), shuffle=False, num_workers=0, collate_fn=_collate) if val_ds else None

    model = QNet(state_dim=ds.state_dim, action_dim=int(args.action_dim), hidden=int(args.hidden)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    # pos_weight は BCEWithLogitsLoss の仕様上、y=1 の項だけ倍率が掛かる（クラス不均衡/勝ちを強めたいとき）
    bce = nn.BCEWithLogitsLoss(reduction="none", pos_weight=torch.tensor(float(args.pos_weight), device=device))

    def _run(loader: DataLoader, train: bool) -> tuple[float, float]:
        if train:
            model.train()
        else:
            model.eval()
        loss_sum = 0.0
        correct_sum = 0.0
        count = 0
        for s, a, y, w in loader:
            s = s.to(device)
            a = a.to(device)
            y = y.to(device)
            w = w.to(device)
            logits = model(s, a)
            loss_vec = bce(logits, y)
            loss = (loss_vec * w).mean()
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            with torch.no_grad():
                p = torch.sigmoid(logits)
                pred = (p >= 0.5).to(torch.float32)
                correct_sum += (pred == y).to(torch.float32).sum().item()
                loss_sum += loss.item() * y.shape[0]
                count += y.shape[0]
        return (loss_sum / max(1, count), correct_sum / max(1, count))

    print(f"device={device}  n_train={len(train_ds)}  n_val={len(val_ds) if val_ds else 0}  state_dim={ds.state_dim}  action_dim={args.action_dim}")
    for ep in range(1, int(args.epochs) + 1):
        tr_loss, tr_acc = _run(train_loader, train=True)
        if val_loader:
            va_loss, va_acc = _run(val_loader, train=False)
            print(f"epoch {ep}/{args.epochs}  train loss={tr_loss:.6f} acc={tr_acc:.3f}  val loss={va_loss:.6f} acc={va_acc:.3f}")
        else:
            print(f"epoch {ep}/{args.epochs}  train loss={tr_loss:.6f} acc={tr_acc:.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "state_dim": ds.state_dim, "action_dim": int(args.action_dim), "hidden": int(args.hidden)}, args.out)

    meta = TrainMeta(
        data=str(data_path),
        state_dim=ds.state_dim,
        action_dim=int(args.action_dim),
        hidden=int(args.hidden),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        seed=int(args.seed),
        device=str(device),
        turn_weight=str(args.turn_weight),
        pos_weight=float(args.pos_weight),
        val_split=float(args.val_split),
    )
    args.meta_out.write_text(json.dumps(asdict(meta), ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"saved: {args.out}")
    print(f"saved: {args.meta_out}")


if __name__ == "__main__":
    main()

