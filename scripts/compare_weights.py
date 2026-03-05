"""
重みあり vs 重みなしで同じ対戦を繰り返し、勝率の差を表示する。

  学習の効果を「デッキごとの勝率変化」で確認できる。
  例: python scripts/compare_weights.py --n 300 --deck0 5 --deck1 6
"""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import argparse
from game import load_weights, setup_game, run_game_auto
from deck import get_deck_name


def run_n_games(n: int, deck0: int, deck1: int, weights_path: Path | None) -> tuple[int, int]:
    """n 試合を実行し、(deck0 の勝ち数, deck1 の勝ち数) を返す。"""
    weights = load_weights(weights_path) if weights_path and weights_path.is_file() else None
    wins = [0, 0]
    for _ in range(n):
        state = setup_game(seed=None, log_fn=None, deck0=deck0, deck1=deck1, weights=weights)
        winner = run_game_auto(state)
        wins[winner] += 1
    return wins[0], wins[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="重みあり vs なしで勝率を比較する")
    parser.add_argument("--n", type=int, default=300, help="各条件での試合数")
    parser.add_argument("--deck0", type=int, default=5, help="プレイヤー0 のデッキ番号")
    parser.add_argument("--deck1", type=int, default=6, help="プレイヤー1 のデッキ番号")
    parser.add_argument("--weights", type=Path, default=_REPO_ROOT / "weights" / "weights.json", help="重み JSON")
    args = parser.parse_args()

    n = args.n
    d0, d1 = args.deck0, args.deck1
    name0 = get_deck_name(d0)
    name1 = get_deck_name(d1)

    print(f"対戦: {name0} (deck{d0}) vs {name1} (deck{d1})、各 {n} 試合\n")

    print("重みなしで実行中...")
    w0_no, w1_no = run_n_games(n, d0, d1, None)
    print(f"  重みなし: {name0} {w0_no} 勝 ({100*w0_no/n:.1f}%) / {name1} {w1_no} 勝 ({100*w1_no/n:.1f}%)")

    print("重みありで実行中...")
    w0_yes, w1_yes = run_n_games(n, d0, d1, args.weights)
    print(f"  重みあり: {name0} {w0_yes} 勝 ({100*w0_yes/n:.1f}%) / {name1} {w1_yes} 勝 ({100*w1_yes/n:.1f}%)")

    print()
    diff0 = (w0_yes - w0_no) / n * 100
    diff1 = (w1_yes - w1_no) / n * 100
    print(f"差（重みあり − 重みなし）: {name0} {diff0:+.1f}%pt / {name1} {diff1:+.1f}%pt")
    print(f"（重みは両プレイヤーに同じように適用されます）")


if __name__ == "__main__":
    main()
