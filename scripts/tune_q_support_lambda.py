"""
q_support_lambda のグリッドサーチ。

エネQ + サポートQ 複合で、サポートQの合成係数を変えてA/Bテスト。
各 λ vs ヒューリスティック（モデルなし）の勝率を比較する。

例:
  python scripts/tune_q_support_lambda.py --games 300
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


def _run_ab(
    n_games: int,
    combos: list[tuple[int, int]],
    q_path: str | None,
    qs_path: str | None,
    q_support_lambda: float,
) -> float:
    from game import setup_game, run_game_auto

    wins = 0
    total = 0
    for i in range(n_games):
        deck0, deck1 = combos[i % len(combos)]
        s = setup_game(
            deck0=deck0, deck1=deck1,
            use_attack_minimax=True, use_energy_attack_lookahead=True,
            q_energy_attach_model_path_p0=q_path,
            q_support_model_path_p0=qs_path,
            q_support_lambda=q_support_lambda,
        )
        w = run_game_auto(s)
        if w == 0:
            wins += 1
        total += 1

        s2 = setup_game(
            deck0=deck1, deck1=deck0,
            use_attack_minimax=True, use_energy_attack_lookahead=True,
            q_energy_attach_model_path_p1=q_path,
            q_support_model_path_p1=qs_path,
            q_support_lambda=q_support_lambda,
        )
        w2 = run_game_auto(s2)
        if w2 == 1:
            wins += 1
        total += 1
    return wins / max(1, total)


def main() -> None:
    parser = argparse.ArgumentParser(description="q_support_lambda グリッドサーチ")
    parser.add_argument("--games", type=int, default=300, help="各条件の片側試合数（既定 300、合計 ×2）")
    parser.add_argument("--decks", type=str, default="5,5:5,6:6,6")
    parser.add_argument("--q-model", type=Path, default=_REPO_ROOT / "models" / "q_energy_attach_best.pt")
    parser.add_argument("--q-support-model", type=Path, default=_REPO_ROOT / "models" / "q_support" / "q_support_iter05.pt")
    parser.add_argument("--lambdas", type=str, default="0.0,0.3,0.5,0.8,1.0,1.5,2.0,3.0",
                        help="テストする λ 値（カンマ区切り）")
    args = parser.parse_args()

    combos: list[tuple[int, int]] = []
    for part in args.decks.split(":"):
        a, b = part.strip().split(",")
        combos.append((int(a.strip()), int(b.strip())))

    q_path = str(args.q_model) if args.q_model.is_file() else None
    qs_path = str(args.q_support_model) if args.q_support_model.is_file() else None
    lambdas = [float(x.strip()) for x in args.lambdas.split(",")]

    print(f"q_support_lambda グリッドサーチ")
    print(f"  エネQ: {q_path or '(なし)'}")
    print(f"  サポートQ: {qs_path or '(なし)'}")
    print(f"  λ候補: {lambdas}")
    print(f"  試合数: {args.games}×2 / 条件")
    print()

    results: list[tuple[float, float]] = []
    for lam in lambdas:
        t0 = time.time()
        wr = _run_ab(args.games, combos, q_path, qs_path, lam)
        elapsed = time.time() - t0
        results.append((lam, wr))
        bar = "█" * int(wr * 40) + "░" * (40 - int(wr * 40))
        print(f"  λ={lam:<4}  {wr*100:5.1f}%  {bar}  ({elapsed:.0f}s)")

    print()
    print("=== 結果まとめ ===")
    best_lam, best_wr = max(results, key=lambda x: x[1])
    for lam, wr in results:
        marker = " ★" if lam == best_lam else ""
        print(f"  λ={lam:<4}  →  {wr*100:.1f}%{marker}")
    print(f"\n  ベスト: λ={best_lam}  ({best_wr*100:.1f}%)")


if __name__ == "__main__":
    main()
