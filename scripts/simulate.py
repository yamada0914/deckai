"""
対戦シミュレーション
デッキ A（オタチ・オオタチ・モトトカゲ）、B（メグロコ・ワルビル）、C（ズピカ・ハラバリー）で、
初手 7 枚・先行ドローあり・サイド 6 枚取り切りで勝敗。自動対戦を繰り返し、勝率などを表示する。
"""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import random
from pathlib import Path

from game import load_weights, setup_game, run_game_auto
from deck import get_deck_count, get_deck_name


def run_simulation(
    n_games: int = 1000,
    seed: int | None = None,
    log_first: bool = True,
    deck0: int = 0,
    deck1: int = 1,
    log_when_deck_loses: int | None = None,
    weights_path: Path | None = None,
    weight_scale: float = 1.0,
) -> dict:
    """
    n_games 回対戦し、結果を返す。deck0 / deck1 で使用デッキを指定（0=A, 1=B, 2=C, 3=D, 4=E）。
    log_when_deck_loses にデッキ番号を指定すると、そのデッキが負けた最初の 1 試合のログを表示する。
    """
    if seed is not None:
        random.seed(seed)
    first_win = 0
    second_win = 0
    wins = [0, 0]
    shippegaeshi_120_games = 0
    logged_loss = False
    for game_idx in range(n_games):
        log_buffer: list[str] = []
        if log_when_deck_loses is not None:
            log_fn = log_buffer.append
        elif log_first and game_idx == 0:
            log_fn = print
        else:
            log_fn = None
        weights = (
            load_weights(weights_path, scale=weight_scale)
            if weights_path and weights_path.is_file()
            else None
        )
        state = setup_game(seed=None, log_fn=log_fn, deck0=deck0, deck1=deck1, weights=weights)
        first_player = state.current_player
        winner = run_game_auto(state)
        if state.shippegaeshi_120_used:
            shippegaeshi_120_games += 1
        wins[winner] += 1
        if winner == first_player:
            first_win += 1
        else:
            second_win += 1
        if log_when_deck_loses is not None and not logged_loss:
            deck_lost = (deck0 == log_when_deck_loses and winner == 1) or (deck1 == log_when_deck_loses and winner == 0)
            if deck_lost:
                print(f"\n===== {get_deck_name(log_when_deck_loses)}が負けた試合のログ =====\n")
                for line in log_buffer:
                    print(line)
                logged_loss = True
    return {
        "first_win": first_win,
        "second_win": second_win,
        "wins": wins,
        "deck0": deck0,
        "deck1": deck1,
        "shippegaeshi_120_games": shippegaeshi_120_games,
    }


def main() -> None:
    n = 1000
    deck0 = 0
    deck1 = 1
    if len(sys.argv) > 1:
        try:
            n = int(sys.argv[1])
        except ValueError:
            pass
    max_deck = get_deck_count() - 1
    if len(sys.argv) > 2:
        try:
            deck0 = int(sys.argv[2])
            deck0 = max(0, min(max_deck, deck0))
        except ValueError:
            pass
    if len(sys.argv) > 3:
        try:
            deck1 = int(sys.argv[3])
            deck1 = max(0, min(max_deck, deck1))
        except ValueError:
            pass
    weights_path: Path | None = None
    weight_scale = 1.0
    if "--weights" in sys.argv:
        i = sys.argv.index("--weights")
        if i + 1 < len(sys.argv):
            weights_path = Path(sys.argv[i + 1])
    if "--weight-scale" in sys.argv:
        i = sys.argv.index("--weight-scale")
        if i + 1 < len(sys.argv):
            try:
                weight_scale = float(sys.argv[i + 1])
            except ValueError:
                pass
    if "--help" in sys.argv or "-h" in sys.argv:
        print("使い方: python scripts/simulate.py [対戦数] [デッキ0] [デッキ1] [--weights 重み.json] [--weight-scale X]")
        print("  デッキ: 0=オタチ, 1=ワニ, 2=カエル, 3=ワルビアル, 4=ジバコイル, 5以降=登録デッキ")
        print("  例: python scripts/simulate.py 1000 3 4  → 1000 回、ワルビアル vs ジバコイル")
        print("  例: python scripts/simulate.py 100 --weights weights/weights.json  → 重み付きで 100 回")
        print("  例: python scripts/simulate.py 100 --weights weights/weights.json --weight-scale 2  → 重みを 2 倍で適用")
        return
    if deck0 == deck1:
        print(f"{get_deck_name(deck0)}どうし、{n} 回シミュレートします。")
    else:
        print(f"{get_deck_name(deck0)} vs {get_deck_name(deck1)}、{n} 回シミュレートします。")
    log_wani_loss = deck0 == 1 and deck1 == 2
    print("（1 回目のみアクションのログを表示します）\n" if not log_wani_loss else "（ワニが負けた最初の 1 試合のログを表示します）\n")
    results = run_simulation(
        n_games=n,
        log_first=not log_wani_loss,
        deck0=deck0,
        deck1=deck1,
        log_when_deck_loses=1 if log_wani_loss else None,
        weights_path=weights_path,
        weight_scale=weight_scale,
    )
    first_win = results["first_win"]
    second_win = results["second_win"]
    wins = results["wins"]
    d0, d1 = results["deck0"], results["deck1"]
    print(f"先手の勝ち: {first_win} 回 ({100 * first_win / n:.1f} %)")
    print(f"後手の勝ち: {second_win} 回 ({100 * second_win / n:.1f} %)")
    if d0 != d1:
        print(f"{get_deck_name(d0)}の勝ち: {wins[0]} 回 ({100 * wins[0] / n:.1f} %)")
        print(f"{get_deck_name(d1)}の勝ち: {wins[1]} 回 ({100 * wins[1] / n:.1f} %)")


if __name__ == "__main__":
    main()
