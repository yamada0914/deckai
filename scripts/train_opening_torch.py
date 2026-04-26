"""
序盤特化エネルギー付与 policy を学習する。

データ: datasets/opening.jsonl（simulate_opening.py が生成）
損失:   BCEWithLogitsLoss
  各サンプルの (s, a) に対して:
    logit[a] ← P(opening_success | state=s, action=a) のスコア
  → action に対応する 1 次元 logit だけ BCE で訓練。

モデル出力: f(s) → [q_0, ..., q_5]（action_dim 次元 logits）
推論:      argmax sigmoid(q) で最良行動を選ぶ。

出力:
  models/pi_opening_energy.pt      モデル重み
  models/pi_opening_energy.meta.json  メタ情報

使い方:
  .venv/bin/python scripts/train_opening_torch.py
  .venv/bin/python scripts/train_opening_torch.py --epochs 50 --hidden 128
  .venv/bin/python scripts/train_opening_torch.py --role 先行   # 先行だけ学習
  .venv/bin/python scripts/train_opening_torch.py --role 後攻   # 後攻だけ学習
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

# action_id 空間: 0=active, 1..BENCH_SIZE=ベンチ
_ACTION_DIM = 6


def _default_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ────────────────────────────────────────────────
# モデル（PiNet と同形。pi_models.py の PiNet と共用可）
# ────────────────────────────────────────────────
class OpeningNet(nn.Module):
    """f(s) → logits [action_dim]。sigmoid すると各行動の成功確率推定。"""
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
        return self.net(x)  # [B, action_dim]


# ────────────────────────────────────────────────
# Dataset
# ────────────────────────────────────────────────
class OpeningDataset(Dataset):
    def __init__(self, rows: list[dict], *, state_mean: np.ndarray, state_std: np.ndarray):
        states, action_ids, successes = [], [], []
        for r in rows:
            s = np.asarray(r["state"], dtype=np.float32)
            s = (s - state_mean) / (state_std + 1e-8)
            states.append(s)
            action_ids.append(int(r["action_id"]))
            successes.append(float(r["success"]))
        self.states = torch.from_numpy(np.stack(states))        # [N, state_dim]
        self.action_ids = torch.tensor(action_ids, dtype=torch.long)  # [N]
        self.successes = torch.tensor(successes, dtype=torch.float32) # [N]

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int):
        return self.states[idx], self.action_ids[idx], self.successes[idx]


def _load_rows(path: Path, *, role: str | None) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("type") != "energy_attach":
                continue
            if role is not None and r.get("role") != role:
                continue
            rows.append(r)
    return rows


def _compute_norm(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    states = np.asarray([r["state"] for r in rows], dtype=np.float32)
    return states.mean(axis=0), states.std(axis=0)


# ────────────────────────────────────────────────
# 学習
# ────────────────────────────────────────────────
@dataclass
class TrainConfig:
    data: Path
    out: Path
    meta_out: Path
    role: str | None
    epochs: int
    batch_size: int
    lr: float
    hidden: int
    val_ratio: float
    seed: int


def train(cfg: TrainConfig) -> None:
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = _default_device()
    print(f"device: {device}")

    rows = _load_rows(cfg.data, role=cfg.role)
    if not rows:
        print(f"データが 0 件です。--data {cfg.data} と --role {cfg.role} を確認してください。")
        raise SystemExit(1)

    print(f"データ: {len(rows)} 行  role={cfg.role or '全て'}")

    # 正規化パラメータ（全データから算出）
    state_mean, state_std = _compute_norm(rows)
    state_dim = len(rows[0]["state"])

    # train / val split
    random.shuffle(rows)
    n_val = max(1, int(len(rows) * cfg.val_ratio))
    val_rows = rows[:n_val]
    train_rows = rows[n_val:]

    train_ds = OpeningDataset(train_rows, state_mean=state_mean, state_std=state_std)
    val_ds = OpeningDataset(val_rows, state_mean=state_mean, state_std=state_std)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    model = OpeningNet(state_dim=state_dim, action_dim=_ACTION_DIM, hidden=cfg.hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    print(f"train={len(train_rows)}  val={len(val_rows)}  state_dim={state_dim}  hidden={cfg.hidden}")
    print(f"epochs={cfg.epochs}  batch={cfg.batch_size}  lr={cfg.lr}")
    print()

    best_val_loss = float("inf")
    best_state_dict = None

    for epoch in range(1, cfg.epochs + 1):
        # ── train ──
        model.train()
        total_loss = 0.0
        n_batches = 0
        for states_b, actions_b, successes_b in train_loader:
            states_b = states_b.to(device)
            actions_b = actions_b.to(device)
            successes_b = successes_b.to(device)

            logits = model(states_b)                            # [B, action_dim]
            chosen_logits = logits.gather(1, actions_b.unsqueeze(1)).squeeze(1)  # [B]
            loss = loss_fn(chosen_logits, successes_b)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        train_loss = total_loss / max(1, n_batches)

        # ── val ──
        model.eval()
        val_loss_total = 0.0
        val_correct = 0
        val_n = 0
        with torch.no_grad():
            for states_b, actions_b, successes_b in val_loader:
                states_b = states_b.to(device)
                actions_b = actions_b.to(device)
                successes_b = successes_b.to(device)

                logits = model(states_b)
                chosen_logits = logits.gather(1, actions_b.unsqueeze(1)).squeeze(1)
                val_loss_total += loss_fn(chosen_logits, successes_b).item() * len(states_b)
                preds = (chosen_logits >= 0.0).float()
                val_correct += (preds == successes_b).sum().item()
                val_n += len(states_b)

        val_loss = val_loss_total / max(1, val_n)
        val_acc = val_correct / max(1, val_n)

        mark = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            mark = " ★"

        if epoch % 5 == 0 or epoch == 1:
            print(f"epoch {epoch:3d}/{cfg.epochs}  train_loss={train_loss:.4f}"
                  f"  val_loss={val_loss:.4f}  val_acc={val_acc:.3f}{mark}")

    print(f"\nbest val_loss={best_val_loss:.4f}")

    # ── 保存 ──
    cfg.out.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "state_dict": best_state_dict,
        "state_dim": state_dim,
        "action_dim": _ACTION_DIM,
        "hidden": cfg.hidden,
        "state_mean": state_mean.tolist(),
        "state_std": state_std.tolist(),
        "encoder_name": "opening",
    }
    torch.save(ckpt, cfg.out)
    print(f"モデル保存: {cfg.out}")

    meta = {
        "data": str(cfg.data),
        "role": cfg.role,
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "state_dim": state_dim,
        "action_dim": _ACTION_DIM,
        "hidden": cfg.hidden,
        "epochs": cfg.epochs,
        "best_val_loss": best_val_loss,
    }
    cfg.meta_out.parent.mkdir(parents=True, exist_ok=True)
    cfg.meta_out.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"メタ情報: {cfg.meta_out}")


# ────────────────────────────────────────────────
# 推論デモ
# ────────────────────────────────────────────────
def eval_demo(model_path: Path, data_path: Path, *, role: str | None, n_samples: int = 10) -> None:
    """学習済みモデルでサンプル推論を表示する。"""
    ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
    state_dim = int(ckpt["state_dim"])
    action_dim = int(ckpt["action_dim"])
    hidden = int(ckpt.get("hidden", 64))
    state_mean = np.asarray(ckpt["state_mean"], dtype=np.float32)
    state_std = np.asarray(ckpt["state_std"], dtype=np.float32)

    model = OpeningNet(state_dim=state_dim, action_dim=action_dim, hidden=hidden)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    rows = _load_rows(data_path, role=role)
    random.shuffle(rows)
    rows = rows[:n_samples]

    print(f"\n── 推論デモ（{len(rows)} サンプル、role={role or '全て'}）──")
    correct = 0
    for r in rows:
        s = np.asarray(r["state"], dtype=np.float32)
        s = (s - state_mean) / (state_std + 1e-8)
        with torch.no_grad():
            logits = model(torch.from_numpy(s).unsqueeze(0))[0]
        probs = torch.sigmoid(logits).tolist()
        chosen = int(r["action_id"])
        pred_action = int(torch.sigmoid(logits).argmax().item())
        pred_success = probs[chosen] >= 0.5
        success = bool(r["success"])
        ok = pred_success == success
        if ok:
            correct += 1
        print(f"  role={r['role']} turn={r['turn']} action={chosen}"
              f"  P(success|a={chosen})={probs[chosen]:.3f}"
              f"  best_action={pred_action}(p={probs[pred_action]:.3f})"
              f"  actual={int(success)}  {'✓' if ok else '✗'}")
    print(f"サンプル正答率: {correct}/{len(rows)}")


# ────────────────────────────────────────────────
# main
# ────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="序盤 energy_attach policy を BCEWithLogitsLoss で学習する")
    parser.add_argument("--data", type=Path, default=_REPO_ROOT / "datasets" / "opening.jsonl")
    parser.add_argument("--out", type=Path, default=_REPO_ROOT / "models" / "pi_opening_energy.pt")
    parser.add_argument("--role", type=str, default=None, choices=["先行", "後攻"],
                        help="先行/後攻のどちらか一方だけ学習（省略時は両方）")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--demo", action="store_true", help="学習後に推論デモを表示する")
    args = parser.parse_args()

    # role ありの場合はファイル名にも反映する
    if args.role:
        suffix = "sensaki" if args.role == "先行" else "kokou"
        out = args.out.with_name(args.out.stem + f"_{suffix}" + args.out.suffix)
        meta_out = out.with_suffix(".meta.json")
    else:
        out = args.out
        meta_out = out.with_suffix(".meta.json")

    cfg = TrainConfig(
        data=args.data,
        out=out,
        meta_out=meta_out,
        role=args.role,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden=args.hidden,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    train(cfg)

    if args.demo:
        eval_demo(out, args.data, role=args.role)


if __name__ == "__main__":
    main()
