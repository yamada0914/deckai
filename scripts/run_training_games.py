"""
指定したデッキ組み合わせで対戦を N 回ずつ記録し、その後 train_weights で重みを学習する。

  汎用的な定石を作る（デッキが増えても使う）:
    --matchup all で複数組み合わせを混ぜ、--min-samples 5 以上でノイズを削る。
    python scripts/run_training_games.py --matchup all --n 500 --scale 12 --min-samples 5

  特定対戦だけ強くする:
    python scripts/run_training_games.py --matchup 5v5 --n 1000 --out-dir battles/train_5v5

  特定デッキ（デッキコード）で自己対戦して最適行動を学習:
    python scripts/run_training_games.py --deck-code 8DcaYc-IR2SmJ-x8D8G8 --n 500 --out-dir battles/train_belt --weights-out weights/weights.json

  ルールのみベースラインでデータを取り直す（1 から学習し直す用）:
    python scripts/run_training_games.py --deck-code ... --n 500 --rules-only --out-dir battles/train_rules_only

  詳細: docs/weight_usage_tips.md
"""
import sys
import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_BATTLES_SUBDIR = "train_weights"


def _run_record(
    battle_id: str,
    deck0: int,
    deck1: int,
    seed: int | None,
    fast: bool = True,
    deck_code0: str | None = None,
    deck_code1: str | None = None,
    rules_only: bool = False,
    rules_only_player0: bool = False,
    rules_only_player1: bool = False,
    nn_second_only: bool = False,
    no_lookahead: bool = True,
    energy_epsilon: float = 0.0,
    support_epsilon: float = 0.0,
    pi_energy_policy_model: str | None = None,
    pi_energy_policy_lambda: float = 0.5,
    pi_support_model: str | None = None,
    pi_support_lambda: float = 0.5,
) -> bool:
    cmd = [
        sys.executable,
        str(_REPO_ROOT / "scripts" / "record_game.py"),
        "--id", battle_id,
        "--deck0", str(deck0),
        "--deck1", str(deck1),
    ]
    if deck_code0:
        cmd.extend(["--deck-code0", deck_code0])
    if deck_code1:
        cmd.extend(["--deck-code1", deck_code1])
    if seed is not None:
        cmd.extend(["--seed", str(seed)])
    if fast:
        cmd.append("--fast")
    if rules_only:
        cmd.append("--rules-only")
    if rules_only_player0:
        cmd.append("--rules-only-player0")
    if rules_only_player1:
        cmd.append("--rules-only-player1")
    if nn_second_only:
        cmd.append("--nn-second-only")
    if no_lookahead:
        cmd.append("--no-lookahead")
    if energy_epsilon > 0.0:
        cmd.extend(["--energy-epsilon", str(energy_epsilon)])
    if support_epsilon > 0.0:
        cmd.extend(["--support-epsilon", str(support_epsilon)])
    if pi_energy_policy_model:
        cmd.extend(["--pi-energy-policy-model", pi_energy_policy_model])
        cmd.extend(["--pi-energy-policy-lambda", str(pi_energy_policy_lambda)])
    if pi_support_model:
        cmd.extend(["--pi-support-model", pi_support_model])
        cmd.extend(["--pi-support-lambda", str(pi_support_lambda)])
    timeout_sec = 60 if fast else 300
    result = subprocess.run(cmd, cwd=str(_REPO_ROOT), capture_output=True, text=True, timeout=timeout_sec)
    return result.returncode == 0


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="対戦を記録してから重みを学習する")
    parser.add_argument("--out-dir", type=Path, default=_REPO_ROOT / "battles" / _BATTLES_SUBDIR, help="対戦 pkl を保存するディレクトリ")
    parser.add_argument("--n", type=int, default=500, help="各組み合わせの試合数")
    parser.add_argument("--matchup", type=str, default="all", choices=["all", "5v6", "5v5", "6v6"], help="学習する対戦だけ記録（all=5v6+5v5+6v6、5v5=コライドン同士のみなど）。--deck-code 指定時は無視")
    parser.add_argument("--deck-code", type=str, default=None, metavar="CODE", help="指定したデッキコードで同デッキ対戦（自己対戦）を N 回記録して学習（例: 8DcaYc-IR2SmJ-x8D8G8）")
    parser.add_argument("--only-train", action="store_true", help="記録は行わず、既存の --out-dir から学習のみ実行")
    parser.add_argument("--weights-out", type=Path, default=_REPO_ROOT / "weights" / "weights.json", help="出力する重み JSON")
    parser.add_argument("--scale", type=float, default=10.0, help="train_weights の --scale（大きくすると重みが強く効く）")
    parser.add_argument("--min-samples", type=int, default=1, help="train_weights の --min-samples（出現回数が少ない選択は重みにしない）")
    parser.add_argument("--fast", action="store_true", default=True, help="記録時に minimax をオフにして短時間で実行（デフォルト ON）")
    parser.add_argument("--no-fast", action="store_false", dest="fast", help="記録時に minimax を有効（1 試合あたり時間がかかる）")
    parser.add_argument(
        "--rules-only",
        action="store_true",
        help="記録時に両者ルールのみ（record_game.py --rules-only と同じ）",
    )
    parser.add_argument(
        "--rules-only-player0",
        action="store_true",
        help="記録時にプレイヤー 0 のみルールのみ（record_game.py と同フラグ）",
    )
    parser.add_argument(
        "--rules-only-player1",
        action="store_true",
        help="記録時にプレイヤー 1 のみルールのみ",
    )
    parser.add_argument(
        "--nn-second-only",
        action="store_true",
        help="後攻のみ NN を使う（先行は rules_only）",
    )
    parser.add_argument(
        "--no-lookahead",
        action="store_true",
        default=True,
        help="lookahead をオフ（デフォルト ON）",
    )
    parser.add_argument(
        "--with-lookahead",
        action="store_false",
        dest="no_lookahead",
        help="lookahead を有効にする",
    )
    parser.add_argument(
        "--energy-epsilon",
        type=float,
        default=0.0,
        metavar="P",
        help="エネルギー付与の ε 探索確率（例: 0.3）",
    )
    parser.add_argument(
        "--support-epsilon",
        type=float,
        default=0.0,
        metavar="P",
        help="サポートポリシーの ε 探索確率",
    )
    parser.add_argument(
        "--pi-energy-policy-model",
        type=str,
        default=None,
        metavar="PATH",
        help="学習済みエネルギー付与ポリシー NN モデル .pt",
    )
    parser.add_argument(
        "--pi-energy-policy-lambda",
        type=float,
        default=0.5,
        metavar="L",
    )
    parser.add_argument(
        "--pi-support-model",
        type=str,
        default=None,
        metavar="PATH",
        help="学習済みサポートポリシー NN モデル .pt",
    )
    parser.add_argument(
        "--pi-support-lambda",
        type=float,
        default=0.5,
        metavar="L",
    )
    args = parser.parse_args()

    battles_dir = args.out_dir
    n = args.n
    deck_code = (args.deck_code or "").strip() or None

    if deck_code:
        configs = [("deck_code", 0, 0, deck_code, deck_code)]
    else:
        all_configs = [("5v6", 5, 6, None, None), ("5v5", 5, 5, None, None), ("6v6", 6, 6, None, None)]
        if args.matchup == "all":
            configs = all_configs
        else:
            configs = [c for c in all_configs if c[0] == args.matchup]

    if not args.only_train:
        total = len(configs) * n
        done = 0
        prefix = battles_dir.name if battles_dir != _REPO_ROOT / "battles" else "train_weights"
        for label, d0, d1, code0, code1 in configs:
            for i in range(n):
                bid = f"{prefix}/train_{label}_{i:04d}" if prefix != "battles" else f"train_{label}_{i:04d}"
                ok = _run_record(
                    bid,
                    d0,
                    d1,
                    seed=None,
                    fast=args.fast,
                    deck_code0=code0,
                    deck_code1=code1,
                    rules_only=bool(args.rules_only),
                    rules_only_player0=bool(args.rules_only_player0),
                    rules_only_player1=bool(args.rules_only_player1),
                    nn_second_only=bool(args.nn_second_only),
                    no_lookahead=bool(args.no_lookahead),
                    energy_epsilon=float(args.energy_epsilon),
                    support_epsilon=float(args.support_epsilon),
                    pi_energy_policy_model=args.pi_energy_policy_model,
                    pi_energy_policy_lambda=float(args.pi_energy_policy_lambda),
                    pi_support_model=args.pi_support_model,
                    pi_support_lambda=float(args.pi_support_lambda),
                )
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
