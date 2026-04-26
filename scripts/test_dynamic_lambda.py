"""
状態依存 q_support_lambda のテスト。

静的 λ（固定） vs 動的 λ（攻撃可能/セットアップで分岐）を比較。

例:
  python scripts/test_dynamic_lambda.py --games 400
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
    *,
    dynamic: bool,
    static_lambda: float = 0.8,
    lambda_attack_ready: float = 0.3,
    lambda_setup: float = 1.2,
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
            q_support_lambda=static_lambda,
            q_support_lambda_dynamic=dynamic,
            q_support_lambda_attack_ready=lambda_attack_ready,
            q_support_lambda_setup=lambda_setup,
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
            q_support_lambda=static_lambda,
            q_support_lambda_dynamic=dynamic,
            q_support_lambda_attack_ready=lambda_attack_ready,
            q_support_lambda_setup=lambda_setup,
        )
        w2 = run_game_auto(s2)
        if w2 == 1:
            wins += 1
        total += 1
    return wins / max(1, total)


def main() -> None:
    parser = argparse.ArgumentParser(description="静的 vs 動的 λ テスト")
    parser.add_argument("--games", type=int, default=400)
    parser.add_argument("--decks", type=str, default="5,5:5,6:6,6")
    parser.add_argument("--q-model", type=Path, default=_REPO_ROOT / "models" / "q_energy_attach_best.pt")
    parser.add_argument("--q-support-model", type=Path, default=_REPO_ROOT / "models" / "q_support" / "q_support_iter05.pt")
    args = parser.parse_args()

    combos: list[tuple[int, int]] = []
    for part in args.decks.split(":"):
        a, b = part.strip().split(",")
        combos.append((int(a.strip()), int(b.strip())))

    q_path = str(args.q_model) if args.q_model.is_file() else None
    qs_path = str(args.q_support_model) if args.q_support_model.is_file() else None

    print(f"静的 vs 動的 λ テスト ({args.games}×2 試合 / 条件)")
    print()

    # 静的 λ=0.8
    print("--- 静的 λ=0.8 ---")
    t0 = time.time()
    wr_static = _run_ab(args.games, combos, q_path, qs_path, dynamic=False, static_lambda=0.8)
    print(f"  → {wr_static*100:.1f}%  ({time.time()-t0:.0f}s)")

    # 動的 λ: attack=0.3, setup=1.2
    print("\n--- 動的 λ: attack=0.3, setup=1.2 ---")
    t0 = time.time()
    wr_dyn_a = _run_ab(args.games, combos, q_path, qs_path, dynamic=True,
                       lambda_attack_ready=0.3, lambda_setup=1.2)
    print(f"  → {wr_dyn_a*100:.1f}%  ({time.time()-t0:.0f}s)")

    # 動的 λ: attack=0.2, setup=1.5（より極端）
    print("\n--- 動的 λ: attack=0.2, setup=1.5 ---")
    t0 = time.time()
    wr_dyn_b = _run_ab(args.games, combos, q_path, qs_path, dynamic=True,
                       lambda_attack_ready=0.2, lambda_setup=1.5)
    print(f"  → {wr_dyn_b*100:.1f}%  ({time.time()-t0:.0f}s)")

    # 動的 λ: attack=0.5, setup=1.0（穏やか）
    print("\n--- 動的 λ: attack=0.5, setup=1.0 ---")
    t0 = time.time()
    wr_dyn_c = _run_ab(args.games, combos, q_path, qs_path, dynamic=True,
                       lambda_attack_ready=0.5, lambda_setup=1.0)
    print(f"  → {wr_dyn_c*100:.1f}%  ({time.time()-t0:.0f}s)")

    # 動的 λ: attack=0.0, setup=1.5（攻撃時は完全にヒューリスティック）
    print("\n--- 動的 λ: attack=0.0, setup=1.5 ---")
    t0 = time.time()
    wr_dyn_d = _run_ab(args.games, combos, q_path, qs_path, dynamic=True,
                       lambda_attack_ready=0.0, lambda_setup=1.5)
    print(f"  → {wr_dyn_d*100:.1f}%  ({time.time()-t0:.0f}s)")

    print("\n=== まとめ ===")
    results = [
        ("静的 λ=0.8",                     wr_static),
        ("動的 attack=0.3 / setup=1.2",   wr_dyn_a),
        ("動的 attack=0.2 / setup=1.5",   wr_dyn_b),
        ("動的 attack=0.5 / setup=1.0",   wr_dyn_c),
        ("動的 attack=0.0 / setup=1.5",   wr_dyn_d),
    ]
    best_label, best_wr = max(results, key=lambda x: x[1])
    for label, wr in results:
        marker = " ★" if label == best_label else ""
        print(f"  {label:.<35s} {wr*100:.1f}%{marker}")


if __name__ == "__main__":
    main()
