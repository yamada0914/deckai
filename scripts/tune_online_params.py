"""
online 評価の係数をグリッド探索して、比較勝率が最も高い組み合わせを見つける。
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


WINRATE_RE = re.compile(r"重みあり側の勝ち:\s+\d+\s+勝\s+/\s+\d+\s+試合\s+\(([\d.]+)%\)")


def run_once(repo: Path, env_overrides: dict[str, str], deck_code: str, n: int) -> float | None:
    env = os.environ.copy()
    env.update(env_overrides)
    cmd = [
        sys.executable,
        "scripts/compare_weighted_vs_unweighted.py",
        "--weights",
        "weights/weights_online.json",
        "--deck-code",
        deck_code,
        "--n",
        str(n),
        "--no-minimax",
        "--online-for-weighted-only",
    ]
    p = subprocess.run(cmd, cwd=repo, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    m = WINRATE_RE.search(p.stdout)
    if not m:
        print("----- command output -----")
        print(p.stdout)
        return None
    return float(m.group(1))


def main() -> None:
    parser = argparse.ArgumentParser(description="online 係数の自動チューニング")
    parser.add_argument("--deck-code", required=True, help="比較対象デッキコード")
    parser.add_argument("--n", type=int, default=300, help="各試行の試合数")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parent.parent

    # まず影響が大きい係数を中心に探索（件数を抑えて実行時間を短縮）
    pos_list = [100.0, 150.0, 200.0]
    neg_list = [30.0, 60.0, 90.0]
    counter_list = [0.85, 0.9]
    floor_list = [0.85, 0.9]

    best: tuple[float, dict[str, str]] | None = None
    tried = 0
    for pos in pos_list:
        for neg in neg_list:
            for counter in counter_list:
                for floor in floor_list:
                    overrides = {
                        "DECKAI_ONLINE_FINISHER_POS_BONUS": str(pos),
                        "DECKAI_ONLINE_FINISHER_NEG_PENALTY": str(neg),
                        "DECKAI_ONLINE_COUNTER_PENALTY_MULT": str(counter),
                        "DECKAI_ONLINE_ENERGY_FLOOR_BASE": str(floor),
                        "DECKAI_ONLINE_HP_FLOOR_BASE": str(floor),
                    }
                    wr = run_once(repo, overrides, args.deck_code, args.n)
                    tried += 1
                    if wr is None:
                        continue
                    print(f"[{tried:02d}] winrate={wr:.1f}% overrides={overrides}")
                    if best is None or wr > best[0]:
                        best = (wr, overrides)

    if best is None:
        raise SystemExit("有効な結果が得られませんでした。")

    print("\n=== BEST ===")
    print(f"winrate={best[0]:.1f}%")
    for k, v in best[1].items():
        print(f"{k}={v}")


if __name__ == "__main__":
    main()

