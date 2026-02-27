"""
毎回 1 試合を実行し、ログと状態スナップショットを保存する。

対戦 ID でログ・pkl・動画を紐付け。ID 未指定時は日時（YYYYMMDD_HHMMSS）で自動採番。
  1 対戦ごとに battles/<id>/ フォルダに battle.log, battle_states.pkl を保存。
  python record_game.py                    # 日時で ID 自動 → battles/20260227_143052/ など
  python record_game.py --id my_match      # 対戦 ID を手動指定する場合のみ
  python make_video.py --battle-id <id>   # 表示された ID で battles/<id>/battle.mp4 を生成
"""
import argparse
import copy
import pickle
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
BATTLES_DIR = _PROJECT_ROOT / "battles"


def _run_and_record(
    seed: int | None = None,
    deck0: int = 5,
    deck1: int = 6,
    battle_id: str | None = None,
    log_path: Path | None = None,
    states_path: Path | None = None,
) -> tuple[list, str]:
    """
    1 試合を実行し、ログを書きつつターン開始ごとの状態をリストに溜める。
    戻り値: 状態のリスト（pickle 保存も行う）
    """
    from game import (
        _check_game_end,
        end_turn,
        run_turn_auto,
        setup_game,
        start_turn,
    )

    if seed is not None:
        import random
        random.seed(seed)

    bid = battle_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    battle_dir = BATTLES_DIR / bid
    log_path = log_path or battle_dir / "battle.log"
    states_path = states_path or battle_dir / "battle_states.pkl"

    states: list = []
    log_snapshots: list[list[str]] = []
    log_lines: list[str] = []

    def log_fn(msg: str) -> None:
        log_lines.append(msg)

    def record_frame(s) -> None:
        states.append(copy.deepcopy(s))
        log_snapshots.append(list(log_lines))

    log_lines.append(f"対戦 ID: {bid}")
    log_lines.append("")

    state = setup_game(seed=seed, log_fn=log_fn, record_frame_fn=record_frame, deck0=deck0, deck1=deck1)

    while True:
        start_turn(state)
        if state.winner is not None:
            state.log("========== ゲーム終了 ==========\n")
            break
        run_turn_auto(state)
        if _check_game_end(state):
            state.log("========== ゲーム終了 ==========\n")
            state._record_frame()
            break
        end_turn(state)
        if state.turn_count >= 200:
            from game import MAX_TURNS_SAFETY
            state.winner = 0 if (6 - len(state.players[0].prize_pile)) >= (6 - len(state.players[1].prize_pile)) else 1
            if state.log_fn:
                state.log(f"{MAX_TURNS_SAFETY} ターンで打ち切り（サイド取得で判定）\n")
            state.log("========== ゲーム終了 ==========\n")
            states.append(copy.deepcopy(state))
            log_snapshots.append(list(log_lines))
            break

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print(f"対戦 ID: {bid}")
    print(f"ログを保存しました: {log_path}")

    for s in states:
        s.log_fn = None
        s.record_frame_fn = None
    states_path.parent.mkdir(parents=True, exist_ok=True)
    with open(states_path, "wb") as f:
        pickle.dump({"states": states, "log_snapshots": log_snapshots}, f)
    print(f"状態スナップショットを保存しました: {states_path} ({len(states)} フレーム)")
    print(f"動画生成: python make_video.py --battle-id {bid}")

    return states, bid


def main() -> None:
    parser = argparse.ArgumentParser(description="1 試合を実行しログと状態を保存する（動画用）")
    parser.add_argument("--id", "--battle-id", dest="battle_id", type=str, default=None, metavar="ID", help="対戦 ID（省略時は日時 YYYYMMDD_HHMMSS で自動）")
    parser.add_argument("--seed", type=int, default=None, help="乱数シード（指定すると同じ進行を再現可能）")
    parser.add_argument("--deck0", type=int, default=5, help="プレイヤー0 のデッキ番号（5=コライドン）")
    parser.add_argument("--deck1", type=int, default=6, help="プレイヤー1 のデッキ番号（6=ミライドン）")
    parser.add_argument("--log", type=Path, default=None, metavar="PATH", help="ログ出力先（省略時は battles/<ID>/battle.log）")
    parser.add_argument("--states", type=Path, default=None, metavar="PATH", help="状態 pickle 出力先（省略時は battles/<ID>/battle_states.pkl）")
    args = parser.parse_args()

    _run_and_record(
        seed=args.seed,
        deck0=args.deck0,
        deck1=args.deck1,
        battle_id=args.battle_id,
        log_path=args.log,
        states_path=args.states,
    )


if __name__ == "__main__":
    main()
