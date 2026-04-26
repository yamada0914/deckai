"""
TD 学習（Fitted Q-Iteration）で Q(s,a) を更新する。

TD ターゲット:
  target(s,a) = r/scale + γ * max_a' Q(s', a')   （非終端）
  target(s,a) = r/scale                             （終端: 次ターンなし）

r         = prize_delta（1ターン分のサイドレース報酬）
s'        = 同プレイヤーの次ターン冒頭の状態
γ（gamma）= 割引率（既定 0.9）

手順:
  1. JSONL を (battle_id, player) 別にグループ化・turn でソート
  2. 現在の Q モデルで max_Q(s') を計算 → TD ターゲット確定
  3. MSE で再学習（Fitted Q-Iteration）
  4. --iters 回繰り返す

例:
  python scripts/train_q_td_torch.py \\
    --data datasets/q_stepC_buffer.jsonl \\
    --init-model models/q_loop_stepC/q_iter05.pt \\
    --out-dir models/q_td \\
    --iters 5 --gamma 0.9 --tail-rows 200000
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
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
        return self.net(x).squeeze(-1)


def _load_ckpt(path: Path, device: torch.device) -> tuple[QNet, dict]:
    ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
    state_dim = int(ckpt["state_dim"])
    action_dim = int(ckpt["action_dim"])
    hidden = int(ckpt.get("hidden", 64))
    model = QNet(state_dim=state_dim, action_dim=action_dim, hidden=hidden)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt


def _predict_all_actions_batch(
    model: QNet,
    states: np.ndarray,  # (N, state_dim)
    action_dim: int,
    device: torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    """全 N 状態 × action_dim の Q 値を返す (N, action_dim)。バッチ推論。"""
    N = states.shape[0]
    results = np.zeros((N, action_dim), dtype=np.float32)
    a_eye = np.eye(action_dim, dtype=np.float32)  # (action_dim, action_dim)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        s_batch = states[start:end]  # (B, state_dim)
        B = s_batch.shape[0]
        # expand: (B, action_dim, state_dim + action_dim)
        s_rep = np.repeat(s_batch[:, None, :], action_dim, axis=1)  # (B, A, D)
        a_rep = np.tile(a_eye[None, :, :], (B, 1, 1))               # (B, A, A)
        x = np.concatenate([s_rep, a_rep], axis=2)                  # (B, A, D+A)
        x_flat = x.reshape(B * action_dim, -1)
        xt = torch.from_numpy(x_flat).to(device)
        with torch.no_grad():
            vals = model(xt).detach().cpu().numpy()  # (B*A,)
        results[start:end] = vals.reshape(B, action_dim)
    return results


def _soft_update(model: QNet, target: QNet, tau: float) -> None:
    """target = τ * model + (1-τ) * target（EMA ソフト更新）。tau=1.0 でハードコピー。"""
    with torch.no_grad():
        for p, pt in zip(model.parameters(), target.parameters()):
            pt.data.copy_(tau * p.data + (1.0 - tau) * pt.data)


def _build_td_samples(
    rows: list[dict],
    model_target: QNet,          # ターゲットネット（固定）で bootstrap
    *,
    gamma: float,
    action_dim: int,
    value_scale: float,
    device: torch.device,
) -> list[tuple[np.ndarray, int, float]]:
    """
    (state, action_id, td_target) のリストを返す。
    """
    # (battle_id, player) でグループ化
    groups: dict[tuple, list] = defaultdict(list)
    for row in rows:
        key = (row["battle_id"], int(row.get("player", 0)))
        groups[key].append(row)

    # turn でソート
    for key in groups:
        groups[key].sort(key=lambda r: int(r.get("turn", 0)))

    # 次状態配列を収集してバッチ推論
    # (i, next_state_vec) のリスト → max_Q を一括計算
    samples_raw: list[tuple[np.ndarray, int, float, int | None]] = []
    # index_in_next_states: samples_raw の i 番目の next_states インデックス（None = 終端）
    next_state_list: list[np.ndarray] = []

    for seq in groups.values():
        for i, row in enumerate(seq):
            state = np.asarray(row["state"], dtype=np.float32)
            action_id = int(row["action_id"])
            r = float(row.get("prize_delta", 0)) / value_scale
            if i + 1 < len(seq):
                next_state = np.asarray(seq[i + 1]["state"], dtype=np.float32)
                ns_idx = len(next_state_list)
                next_state_list.append(next_state)
            else:
                ns_idx = None  # 終端
            samples_raw.append((state, action_id, r, ns_idx))

    # バッチ推論で max_Q(s') を計算
    max_next_q = np.zeros(len(next_state_list), dtype=np.float32)
    if next_state_list:
        ns_arr = np.stack(next_state_list, axis=0)  # (M, state_dim)
        q_all = _predict_all_actions_batch(model_target, ns_arr, action_dim, device)
        max_next_q = q_all.max(axis=1)  # (M,)

    # TD ターゲット確定
    samples: list[tuple[np.ndarray, int, float]] = []
    for state, action_id, r, ns_idx in samples_raw:
        if ns_idx is not None:
            target = r + gamma * float(max_next_q[ns_idx])
        else:
            target = r
        samples.append((state, action_id, target))

    return samples


class TDDataset(Dataset):
    def __init__(self, samples: list[tuple[np.ndarray, int, float]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s, a, t = self.samples[idx]
        a_oh = np.zeros(6, dtype=np.float32)
        if 0 <= a < 6:
            a_oh[a] = 1.0
        return s, a_oh, np.float32(t)


def _collate(batch):
    s, a, t = zip(*batch)
    return (
        torch.from_numpy(np.stack(s)),
        torch.from_numpy(np.stack(a)),
        torch.tensor(t, dtype=torch.float32),
    )


def _train_one_iter(
    samples: list[tuple],
    model: QNet,
    opt: torch.optim.Optimizer,
    *,
    epochs: int,
    batch_size: int,
    val_split: float,
    device: torch.device,
) -> tuple[float, float]:
    n = len(samples)
    idxs = list(range(n))
    random.shuffle(idxs)
    n_val = int(n * val_split)
    val_idx = set(idxs[:n_val])

    class _Sub(Dataset):
        def __init__(self, base, keep):
            self.base = base; self.keep = sorted(keep)
        def __len__(self): return len(self.keep)
        def __getitem__(self, i): return self.base[self.keep[i]]

    ds = TDDataset(samples)
    tr_ds = _Sub(ds, set(range(n)) - val_idx)
    va_ds = _Sub(ds, val_idx) if n_val > 0 else None
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, collate_fn=_collate, num_workers=0)
    va_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=False, collate_fn=_collate, num_workers=0) if va_ds else None
    mse = nn.MSELoss(reduction="none")

    final_tr_mae = final_va_mae = 0.0
    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = tr_mae = tr_n = 0
        for s, a, t in tr_loader:
            s, a, t = s.to(device), a.to(device), t.to(device)
            x = torch.cat([s, a], dim=-1)
            pred = model(x)
            loss = mse(pred, t).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            with torch.no_grad():
                tr_loss += loss.item() * s.shape[0]
                tr_mae += torch.abs(pred - t).sum().item()
                tr_n += s.shape[0]
        final_tr_mae = tr_mae / max(1, tr_n)
        if va_loader:
            model.eval()
            va_mae = va_n = 0
            with torch.no_grad():
                for s, a, t in va_loader:
                    s, a, t = s.to(device), a.to(device), t.to(device)
                    x = torch.cat([s, a], dim=-1)
                    pred = model(x)
                    va_mae += torch.abs(pred - t).sum().item()
                    va_n += s.shape[0]
            final_va_mae = va_mae / max(1, va_n)
            print(f"  epoch {ep:>2}/{epochs}  train_mae={final_tr_mae:.4f}  val_mae={final_va_mae:.4f}")
        else:
            print(f"  epoch {ep:>2}/{epochs}  train_mae={final_tr_mae:.4f}")

    return final_tr_mae, final_va_mae


def main() -> None:
    parser = argparse.ArgumentParser(description="TD 学習（Fitted Q-Iteration）で Q-model を更新")
    parser.add_argument("--data", type=Path, required=True, help="入力 JSONL（prize_delta フィールド必須）")
    parser.add_argument("--init-model", type=Path, required=True, help="初期 Q-model .pt（TD bootstrap に使用）")
    parser.add_argument("--out-dir", type=Path, default=_REPO_ROOT / "models" / "q_td", help="出力ディレクトリ")
    parser.add_argument("--iters", type=int, default=5, help="Fitted Q-Iteration 回数（既定 5）")
    parser.add_argument("--gamma", type=float, default=0.9, help="割引率 γ（既定 0.9）")
    parser.add_argument("--value-scale", type=float, default=3.0, help="prize_delta 正規化スケール")
    parser.add_argument("--tail-rows", type=int, default=0, help="末尾 N 行だけ使用（0 で全件）")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10, help="1 イテレーション当たりのエポック数")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-mps", action="store_true")
    parser.add_argument(
        "--target-tau",
        type=float,
        default=0.05,
        help="ターゲットネットのソフト更新係数 τ（0<τ≤1。1.0=ハードコピー、既定 0.05）",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cpu") if args.no_mps else _default_device()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # データ読み込み
    print(f"データ読み込み: {args.data}")
    rows: list[dict] = []
    with args.data.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if args.tail_rows > 0 and len(rows) > args.tail_rows:
        rows = rows[-args.tail_rows:]
    print(f"  使用行数: {len(rows):,}  (battle_id 種類: {len(set(r['battle_id'] for r in rows)):,})")

    # 初期モデル読み込み
    model, ckpt = _load_ckpt(args.init_model, device)
    action_dim = model.action_dim
    state_dim = model.state_dim
    print(f"初期モデル: {args.init_model}  state_dim={state_dim}  action_dim={action_dim}  device={device}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ターゲットネット: 初期は model のハードコピー
    import copy
    model_target = copy.deepcopy(model)
    model_target.eval()
    tau = float(args.target_tau)
    print(f"ターゲットネット: τ={tau}  (1.0=ハードコピー毎iter)")

    for it in range(1, args.iters + 1):
        print(f"\n{'='*55}")
        print(f"  TD iter {it}/{args.iters}  γ={args.gamma}  τ={tau}")
        print(f"{'='*55}")

        # TD ターゲット計算（固定ターゲットネットで bootstrap）
        print("  TD ターゲット計算中 ...")
        model_target.eval()
        samples = _build_td_samples(
            rows, model_target,
            gamma=args.gamma,
            action_dim=action_dim,
            value_scale=args.value_scale,
            device=device,
        )
        targets = [t for _, _, t in samples]
        print(f"  サンプル数: {len(samples):,}  target: min={min(targets):.3f} mean={sum(targets)/len(targets):.3f} max={max(targets):.3f}")

        # 学習
        print("  学習中 ...")
        tr_mae, va_mae = _train_one_iter(
            samples, model, opt,
            epochs=args.epochs,
            batch_size=args.batch_size,
            val_split=args.val_split,
            device=device,
        )

        # ターゲットネットをソフト更新（学習後に model → target に少し寄せる）
        _soft_update(model, model_target, tau)

        # 保存
        out_path = args.out_dir / f"q_td_iter{it:02d}.pt"
        torch.save({
            "state_dict": model.state_dict(),
            "state_dim": state_dim,
            "action_dim": action_dim,
            "hidden": args.hidden,
        }, out_path)
        print(f"  saved: {out_path}  tr_mae={tr_mae:.4f}  val_mae={va_mae:.4f}")

    print(f"\n完了。最終モデル: {args.out_dir}/q_td_iter{args.iters:02d}.pt")


if __name__ == "__main__":
    main()
