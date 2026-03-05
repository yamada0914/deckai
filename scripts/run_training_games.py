"""
指定したデッキ組み合わせで対戦を N 回ずつ記録し、その後 train_weights で重みを学習する。

  汎用的な定石を作る（デッキが増えても使う）:
    --matchup all で複数組み合わせを混ぜ、--min-samples 5 以上でノイズを削る。
    python scripts/run_training_games.py --matchup all --n 500 --scale 12 --min-samples 5

  特定対戦だけ強くする:
    python scripts/run_training_games.py --matchup 5v5 --n 1000 --out-dir battles/train_5v5

  詳細: docs/weight_usage_tips.md
"""
import sys
import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_BATTLES_SUBDIR = "train_weights"


def _run_record(battle_id: str, deck0: int, deck1: int, seed: int | None, fast: bool = True) -> bool:
    cmd = [
        sys.executable,
        str(_REPO_ROOT / "scripts" / "record_game.py"),
        "--id", battle_id,
        "--deck0", str(deck0),
        "--deck1", str(deck1),
    ]
    if seed is not None:
        cmd.extend(["--seed", str(seed)])
    if fast:
        cmd.append("--fast")
    timeout_sec = 60 if fast else 300
    result = subprocess.run(cmd, cwd=str(_REPO_ROOT), capture_output=True, text=True, timeout=timeout_sec)
    return result.returncode == 0


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="対戦を記録してから重みを学習する")
    parser.add_argument("--out-dir", type=Path, default=_REPO_ROOT / "battles" / _BATTLES_SUBDIR, help="対戦 pkl を保存するディレクトリ")
    parser.add_argument("--n", type=int, default=500, help="各組み合わせの試合数")
    parser.add_argument("--matchup", type=str, default="all", choices=["all", "5v6", "5v5", "6v6"], help="学習する対戦だけ記録（all=5v6+5v5+6v6、5v5=コライドン同士のみなど）")
    parser.add_argument("--only-train", action="store_true", help="記録は行わず、既存の --out-dir から学習のみ実行")
    parser.add_argument("--weights-out", type=Path, default=_REPO_ROOT / "weights" / "weights.json", help="出力する重み JSON")
    parser.add_argument("--scale", type=float, default=10.0, help="train_weights の --scale（大きくすると重みが強く効く）")
    parser.add_argument("--min-samples", type=int, default=1, help="train_weights の --min-samples（出現回数が少ない選択は重みにしない）")
    parser.add_argument("--fast", action="store_true", default=True, help="記録時に minimax をオフにして短時間で実行（デフォルト ON）")
    parser.add_argument("--no-fast", action="store_false", dest="fast", help="記録時に minimax を有効（1 試合あたり時間がかかる）")
    args = parser.parse_args()

    battles_dir = args.out_dir
    n = args.n

    all_configs = [("5v6", 5, 6), ("5v5", 5, 5), ("6v6", 6, 6)]
    if args.matchup == "all":
        configs = all_configs
    else:
        configs = [c for c in all_configs if c[0] == args.matchup]

    if not args.only_train:
        total = len(configs) * n
        done = 0
        prefix = battles_dir.name if battles_dir != _REPO_ROOT / "battles" else "train_weights"
        for label, d0, d1 in configs:
            for i in range(n):
                bid = f"{prefix}/train_{label}_{i:04d}" if prefix != "battles" else f"train_{label}_{i:04d}"
                ok = _run_record(bid, d0, d1, seed=None, fast=args.fast)
                done += 1
                if done % 50 == 0 or not ok:
                    status = "OK" if ok else "FAIL"
                    print(f"[{done}/{total}] {bid} {status}")
                if not ok:
                    print(f"  stderr: (run with same --id to see error)")
        print(f"記録完了: {total} 試合 → {battles_dir}")

    # train_weights
    train_cmd = [
        sys.executable,
        str(_REPO_ROOT / "scripts" / "train_weights.py"),
        "--battles-dir", str(battles_dir),
        "--output", str(args.weights_out),
        "--scale", str(args.scale),
        "--min-samples", str(args.min_samples),
    ]
    print("学習実行:", " ".join(train_cmd))
    subprocess.run(train_cmd, cwd=str(_REPO_ROOT))
    print(f"重みを保存しました: {args.weights_out}")


if __name__ == "__main__":
    main()
