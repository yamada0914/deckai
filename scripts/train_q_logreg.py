"""
JSONL dataset から Q(s, a)=P(win | state, action) を学習する（標準ライブラリのみ）。

背景:
  この環境では torch が未導入、numpy の import も不安定なため、
  まずは pure Python のロジスティック回帰（SGD）で学習を回せるようにする。

入力（1 行 JSON）:
  {"state":[...], "action_id":2, "win":1, "turn": 12, ...}

モデル:
  x = state + onehot(action_id)
  p = sigmoid(w·x + b)

出力（JSON）:
  {"state_dim":15, "action_dim":6, "w":[...], "b":0.0}
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _sigmoid(z: float) -> float:
    # 数値安定化
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _dot(w: list[float], x: list[float]) -> float:
    return sum(wi * xi for wi, xi in zip(w, x))


def _onehot(action_id: int, action_dim: int) -> list[float]:
    v = [0.0] * action_dim
    if 0 <= action_id < action_dim:
        v[action_id] = 1.0
    return v


def _iter_rows(jsonl_path: Path):
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _turn_weight(turn: int, *, mode: str, max_turn: int = 200) -> float:
    """
    終盤を強く学習したいときの重み。
    - none: 1.0
    - linear: turn/max_turn を 0.2〜1.0 にスケール
    - sqrt: sqrt(turn/max_turn) を 0.2〜1.0 にスケール
    """
    if mode == "none":
        return 1.0
    t = max(0.0, min(1.0, float(turn) / float(max_turn)))
    if mode == "linear":
        v = t
    else:
        v = math.sqrt(t)
    return 0.2 + 0.8 * v


def main() -> None:
    parser = argparse.ArgumentParser(description="Q(s,a)=P(win|state,action) をロジスティック回帰で学習（pure Python）")
    parser.add_argument("--data", type=Path, required=True, help="入力 JSONL（例: datasets/policy_energy_attach.jsonl）")
    parser.add_argument("--out", type=Path, default=_REPO_ROOT / "models" / "q_energy_attach_logreg.json", help="出力モデル JSON")
    parser.add_argument("--epochs", type=int, default=3, help="エポック数")
    parser.add_argument("--lr", type=float, default=0.05, help="学習率")
    parser.add_argument("--l2", type=float, default=0.0, help="L2 正則化（例: 1e-4）")
    parser.add_argument("--seed", type=int, default=0, help="乱数 seed")
    parser.add_argument("--action-dim", type=int, default=6, help="action の次元（energy_attach は 6=active+bench5）")
    parser.add_argument("--turn-weight", type=str, default="none", choices=["none", "linear", "sqrt"], help="turn による重み付け")
    parser.add_argument("--pos-weight", type=float, default=1.0, help="win=1 の重み（負け行動を学びすぎる場合に増やす）")
    args = parser.parse_args()

    random.seed(args.seed)

    # まず全行をメモリに読み（軽量）: state は 15 次元想定で数十万行でも大きくなりにくい
    rows = list(_iter_rows(args.data))
    if not rows:
        print(f"データが空です: {args.data}", file=sys.stderr)
        raise SystemExit(1)

    state_dim = len(rows[0]["state"])
    action_dim = int(args.action_dim)
    x_dim = state_dim + action_dim

    # 初期化
    w = [0.0] * x_dim
    b = 0.0

    # シャッフル用 index
    idxs = list(range(len(rows)))

    def _make_x(r) -> list[float]:
        s = r["state"]
        a = int(r["action_id"])
        return list(map(float, s)) + _onehot(a, action_dim)

    for ep in range(1, args.epochs + 1):
        random.shuffle(idxs)
        loss_sum = 0.0
        n = 0
        for j in idxs:
            r = rows[j]
            y = 1.0 if int(r.get("win", 0)) == 1 else 0.0
            x = _make_x(r)
            z = _dot(w, x) + b
            p = _sigmoid(z)

            # 重み付け
            tw = _turn_weight(int(r.get("turn", 0)), mode=args.turn_weight)
            cw = args.pos_weight if y == 1.0 else 1.0
            weight = tw * cw

            # BCE: -[y log p + (1-y) log(1-p)]
            eps = 1e-12
            loss = -(y * math.log(p + eps) + (1.0 - y) * math.log(1.0 - p + eps))
            loss_sum += weight * loss
            n += 1

            # 勾配（BCE + sigmoid の組み合わせ）: (p - y) * x
            g = weight * (p - y)
            lr = args.lr
            if args.l2 > 0.0:
                # L2: 0.5*l2*||w||^2 → grad += l2*w
                for i in range(x_dim):
                    w[i] -= lr * (g * x[i] + args.l2 * w[i])
            else:
                for i in range(x_dim):
                    w[i] -= lr * (g * x[i])
            b -= lr * g

        print(f"epoch {ep}/{args.epochs}  loss={loss_sum/max(1,n):.6f}  n={n}")

    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "w": w,
        "b": b,
        "meta": {
            "data": str(args.data),
            "epochs": args.epochs,
            "lr": args.lr,
            "l2": args.l2,
            "seed": args.seed,
            "turn_weight": args.turn_weight,
            "pos_weight": args.pos_weight,
        },
    }
    out_path.write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()

