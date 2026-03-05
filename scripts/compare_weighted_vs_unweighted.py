"""
同じデッキで「重みあり」vs「重みなし」を N 回対戦する。

  先行・後攻の偏りを除くため、--fair（既定）で「重みありをプレイヤー0」と「重みありをプレイヤー1」を
  半々で実行し、「重みあり側の勝率」を出す。

  例:
    python scripts/compare_weighted_vs_unweighted.py --deck 5 --n 1000
    python scripts/compare_weighted_vs_unweighted.py --deck 5 --n 500 --no-minimax  # 重みの差が出やすい
"""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import argparse
from game import load_weights, setup_game, run_game_auto
from deck import get_deck_name


def main() -> None:
    parser = argparse.ArgumentParser(description="同じデッキで重みあり vs 重みなしを N 回対戦")
    parser.add_argument("--deck", type=int, default=5, help="両者とも使うデッキ番号（5=コライドン, 6=ミライドンなど）")
    parser.add_argument("--n", type=int, default=100, help="試合数（--fair 時は半分ずつに割り振る）")
    parser.add_argument("--weights", type=Path, default=_REPO_ROOT / "weights" / "weights.json", help="重み JSON")
    parser.add_argument("--no-fair", action="store_true", help="先行偏りを取らず、重みありを常にプレイヤー0 にする")
    parser.add_argument(
        "--no-minimax",
        action="store_true",
        help="技選択で minimax（2 手読み）を使わず 1 手評価のみにする。重みの差が出やすくなる",
    )
    parser.add_argument(
        "--weight-scale",
        type=float,
        default=1.0,
        metavar="X",
        help="重みを X 倍して使う（例: 2.0 で強く効かせる）。既定 1.0",
    )
    args = parser.parse_args()

    deck = args.deck
    n = args.n
    weights = (
        load_weights(args.weights, scale=args.weight_scale)
        if args.weights.is_file()
        else None
    )
    name = get_deck_name(deck)
    fair = not args.no_fair
    use_minimax = not args.no_minimax

    if fair:
        print(f"対戦: 重みあり {name} vs 重みなし {name}、計 {n} 回（先行偏りを除くため重みの配置を半々で実行）")
    else:
        print(f"対戦: 重みあり {name} (プレイヤー0) vs 重みなし {name} (プレイヤー1)、{n} 回")
    if not use_minimax:
        print("（技選択は 1 手評価のみ・minimax なし）")
    if args.weight_scale != 1.0:
        print(f"（重みを {args.weight_scale} 倍で適用）")
    print()

    def _setup(weights_p0, weights_p1):
        return setup_game(
            seed=None,
            log_fn=None,
            deck0=deck,
            deck1=deck,
            weights_player0=weights_p0,
            weights_player1=weights_p1,
            use_attack_minimax=use_minimax,
        )

    wins_weighted_side = 0
    wins_p0 = 0
    wins_p1 = 0

    if fair:
        half = n // 2
        for _ in range(half):
            state = _setup(weights, None)
            winner = run_game_auto(state)
            if winner == 0:
                wins_weighted_side += 1
            wins_p0 += 1 if winner == 0 else 0
            wins_p1 += 1 if winner == 1 else 0
        for _ in range(n - half):
            state = _setup(None, weights)
            winner = run_game_auto(state)
            if winner == 1:
                wins_weighted_side += 1
            wins_p0 += 1 if winner == 0 else 0
            wins_p1 += 1 if winner == 1 else 0
        print(f"重みあり側の勝ち: {wins_weighted_side} 勝 / {n} 試合 ({100 * wins_weighted_side / n:.1f}%)")
        print(f"（参考: プレイヤー0 勝ち {wins_p0} / プレイヤー1 勝ち {wins_p1}）")
    else:
        for _ in range(n):
            state = _setup(weights, None)
            winner = run_game_auto(state)
            if winner == 0:
                wins_weighted_side += 1
            wins_p0 += 1 if winner == 0 else 0
            wins_p1 += 1 if winner == 1 else 0
        print(f"重みあり（プレイヤー0）: {wins_weighted_side} 勝 ({100 * wins_weighted_side / n:.1f}%)")
        print(f"重みなし（プレイヤー1）: {wins_p1} 勝 ({100 * wins_p1 / n:.1f}%)")


if __name__ == "__main__":
    main()
