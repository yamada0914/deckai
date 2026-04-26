"""
エネルギー Q-model + サポート Q-model の複合 A/B テスト。

モデル有り vs ヒューリスティック（モデル無し）の勝率を測定する。
両サイド（P0/P1）入れ替えて公平に比較。

テスト項目:
  1. エネQ + サポートQ 複合 vs ヒューリスティック
  2. エネQ のみ vs ヒューリスティック
  3. サポートQ のみ vs ヒューリスティック
  4. 条件付きヒューリスティックのみ vs 旧ヒューリスティック（ベースライン改善確認）

例:
  python scripts/ab_test_combined.py --games 400
  python scripts/ab_test_combined.py --games 400 --q-model models/q_energy_attach_best.pt --q-support-model models/q_support/q_support_iter05.pt
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
    setup_kw_model: dict,
    setup_kw_baseline: dict | None = None,
) -> float:
    """model 側の勝率を返す (0.0〜1.0)。"""
    from game import setup_game, run_game_auto

    if setup_kw_baseline is None:
        setup_kw_baseline = {}
    wins = 0
    total = 0
    for i in range(n_games):
        deck0, deck1 = combos[i % len(combos)]
        # P0=MODEL, P1=BASELINE
        kw = dict(deck0=deck0, deck1=deck1, use_attack_minimax=True, use_energy_attack_lookahead=True)
        kw_p0 = {k + "_p0" if not k.startswith("use_") else k: v for k, v in setup_kw_model.items()}
        kw_p1 = {k + "_p1" if not k.startswith("use_") else k: v for k, v in setup_kw_baseline.items()}
        # Flatten per-player args
        for k, v in setup_kw_model.items():
            kw[k + "_p0"] = v
        for k, v in setup_kw_baseline.items():
            kw[k + "_p1"] = v
        s = setup_game(**kw)
        w = run_game_auto(s)
        if w == 0:
            wins += 1
        total += 1

        # P0=BASELINE, P1=MODEL
        kw2 = dict(deck0=deck1, deck1=deck0, use_attack_minimax=True, use_energy_attack_lookahead=True)
        for k, v in setup_kw_baseline.items():
            kw2[k + "_p0"] = v
        for k, v in setup_kw_model.items():
            kw2[k + "_p1"] = v
        s2 = setup_game(**kw2)
        w2 = run_game_auto(s2)
        if w2 == 1:
            wins += 1
        total += 1
    return wins / max(1, total)


def main() -> None:
    parser = argparse.ArgumentParser(description="エネルギーQ + サポートQ 複合 A/B テスト")
    parser.add_argument("--games", type=int, default=400, help="片側あたりの試合数（既定 400、合計 ×2）")
    parser.add_argument("--decks", type=str, default="5,5:5,6:6,6")
    parser.add_argument(
        "--q-model", type=Path,
        default=_REPO_ROOT / "models" / "q_energy_attach_best.pt",
        help="エネルギー Q-model パス",
    )
    parser.add_argument(
        "--q-support-model", type=Path,
        default=_REPO_ROOT / "models" / "q_support" / "q_support_iter05.pt",
        help="サポート Q-model パス",
    )
    parser.add_argument("--q-support-lambda", type=float, default=3.0, help="サポートQの合成係数")
    args = parser.parse_args()

    from game import setup_game, run_game_auto

    combos: list[tuple[int, int]] = []
    for part in args.decks.split(":"):
        a, b = part.strip().split(",")
        combos.append((int(a.strip()), int(b.strip())))

    q_path = str(args.q_model) if args.q_model.is_file() else None
    qs_path = str(args.q_support_model) if args.q_support_model.is_file() else None

    print(f"A/B テスト: モデル有り vs ヒューリスティック")
    print(f"  エネルギー Q:  {q_path or '(なし)'}")
    print(f"  サポート Q:    {qs_path or '(なし)'}")
    print(f"  デッキ: {args.decks}  試合数: {args.games}×2")
    print()

    results: dict[str, float] = {}

    # --- 1. エネQ + サポートQ 複合 ---
    if q_path and qs_path:
        print("--- エネQ + サポートQ 複合 vs ヒューリスティック ---")
        t0 = time.time()
        wins = 0
        total = 0
        for i in range(args.games):
            deck0, deck1 = combos[i % len(combos)]
            s = setup_game(
                deck0=deck0, deck1=deck1,
                use_attack_minimax=True, use_energy_attack_lookahead=True,
                q_energy_attach_model_path_p0=q_path,
                q_support_model_path_p0=qs_path,
                q_support_lambda=args.q_support_lambda,
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
                q_support_lambda=args.q_support_lambda,
            )
            w2 = run_game_auto(s2)
            if w2 == 1:
                wins += 1
            total += 1

            if (i + 1) % 100 == 0 or (i + 1) == args.games:
                print(f"  [{i+1:>4}/{args.games}] model={wins}({wins/total*100:.1f}%)  {time.time()-t0:.0f}s")

        wr = wins / total * 100
        results["エネQ + サポートQ 複合"] = wr
        print(f"  → {wr:.1f}%\n")

    # --- 2. エネQ のみ ---
    if q_path:
        print("--- エネQ のみ vs ヒューリスティック ---")
        t0 = time.time()
        wins = 0
        total = 0
        for i in range(args.games):
            deck0, deck1 = combos[i % len(combos)]
            s = setup_game(
                deck0=deck0, deck1=deck1,
                use_attack_minimax=True, use_energy_attack_lookahead=True,
                q_energy_attach_model_path_p0=q_path,
            )
            w = run_game_auto(s)
            if w == 0:
                wins += 1
            total += 1

            s2 = setup_game(
                deck0=deck1, deck1=deck0,
                use_attack_minimax=True, use_energy_attack_lookahead=True,
                q_energy_attach_model_path_p1=q_path,
            )
            w2 = run_game_auto(s2)
            if w2 == 1:
                wins += 1
            total += 1

        wr = wins / total * 100
        results["エネQ のみ"] = wr
        print(f"  → {wr:.1f}%  ({time.time()-t0:.0f}s)\n")

    # --- 3. サポートQ のみ ---
    if qs_path:
        print("--- サポートQ のみ vs ヒューリスティック ---")
        t0 = time.time()
        wins = 0
        total = 0
        for i in range(args.games):
            deck0, deck1 = combos[i % len(combos)]
            s = setup_game(
                deck0=deck0, deck1=deck1,
                use_attack_minimax=True, use_energy_attack_lookahead=True,
                q_support_model_path_p0=qs_path,
                q_support_lambda=args.q_support_lambda,
            )
            w = run_game_auto(s)
            if w == 0:
                wins += 1
            total += 1

            s2 = setup_game(
                deck0=deck1, deck1=deck0,
                use_attack_minimax=True, use_energy_attack_lookahead=True,
                q_support_model_path_p1=qs_path,
                q_support_lambda=args.q_support_lambda,
            )
            w2 = run_game_auto(s2)
            if w2 == 1:
                wins += 1
            total += 1

        wr = wins / total * 100
        results["サポートQ のみ"] = wr
        print(f"  → {wr:.1f}%  ({time.time()-t0:.0f}s)\n")

    # --- まとめ ---
    print("=== まとめ ===")
    for label, wr in results.items():
        print(f"  {label:.<28s} {wr:.1f}%")
    print(f"  {'ヒューリスティック（基準）':.<28s} 50.0%")


if __name__ == "__main__":
    main()
