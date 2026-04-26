"""
1コマンドで対戦記録→動画生成。

  python scripts/quick_battle.py                          # ランダムseed
  python scripts/quick_battle.py --seed 42                # seed指定
  python scripts/quick_battle.py --seed 42 --log          # ログも表示
  python scripts/quick_battle.py --deck0-code CODE --deck1-code CODE
"""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import argparse
import random
from datetime import datetime


def main() -> None:
    parser = argparse.ArgumentParser(description="対戦記録→動画生成を1コマンドで実行")
    parser.add_argument("--seed", type=int, default=None, help="乱数シード（省略でランダム）")
    parser.add_argument("--id", type=str, default=None, help="対戦ID（省略で自動）")
    parser.add_argument("--deck0", type=int, default=5, help="プレイヤー0デッキ番号")
    parser.add_argument("--deck1", type=int, default=6, help="プレイヤー1デッキ番号")
    parser.add_argument("--deck0-code", type=str, default=None, help="プレイヤー0デッキコード")
    parser.add_argument("--deck1-code", type=str, default=None, help="プレイヤー1デッキコード")
    parser.add_argument("--fps", type=float, default=0.33, help="動画FPS（デフォルト0.33）")
    parser.add_argument("--log", action="store_true", help="バトルログも標準出力に表示")
    parser.add_argument("--no-video", action="store_true", help="動画生成をスキップ")
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else random.randint(0, 99999)
    bid = args.id or datetime.now().strftime("%Y%m%d_%H%M%S")

    from scripts.record_game import _run_and_record
    states, bid = _run_and_record(
        seed=seed,
        deck0=args.deck0,
        deck1=args.deck1,
        deck_code0=args.deck0_code,
        deck_code1=args.deck1_code,
        battle_id=bid,
    )

    if args.log:
        log_path = _REPO_ROOT / "battles" / bid / "battle.log"
        if log_path.is_file():
            print("\n" + log_path.read_text(encoding="utf-8"))

    if args.no_video:
        return

    # 動画生成
    import pickle
    import subprocess
    from board_render import render_board_frame

    battle_dir = _REPO_ROOT / "battles" / bid
    states_path = battle_dir / "battle_states.pkl"
    output_mp4 = battle_dir / "battle.mp4"
    frames_dir = battle_dir / "frames"

    with open(states_path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict):
        states = data.get("states", [])
        log_snapshots = data.get("log_snapshots", [])
    else:
        states = data
        log_snapshots = []

    # キーフレーム間引き
    def _sig(state):
        parts = [state.turn_count, state.current_player, state.winner]
        for p in state.players:
            if p.active is None:
                parts.append(None)
            else:
                parts.append((getattr(p.active.card, "id", ""), p.active.hp))
            parts.append(tuple((getattr(bp.card, "id", ""), bp.hp) for bp in p.bench))
            parts.append(len(p.hand))
            parts.append(len(p.deck))
            parts.append(len(p.discard))
            parts.append(len(p.prize_pile))
        return tuple(parts)

    key_indices = [0]
    prev = _sig(states[0])
    for i in range(1, len(states)):
        s = _sig(states[i])
        if s != prev:
            key_indices.append(i)
            prev = s

    images_dir = _REPO_ROOT / "card_images"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for f in frames_dir.glob("frame_*.png"):
        f.unlink()

    # ffmpegにrawvideoパイプで直接渡す（中間ファイル不要、最速）
    import io
    import numpy as np

    # 最初のフレームでサイズを取得
    first_state = states[key_indices[0]]
    first_log = "\n".join(log_snapshots[key_indices[0]]) if key_indices[0] < len(log_snapshots) else ""
    first_img = render_board_frame(first_state, images_dir, log_text=first_log or None)
    fw, fh = first_img.size

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(args.fps),
        "-f", "rawvideo",
        "-pixel_format", "rgb24",
        "-video_size", f"{fw}x{fh}",
        "-i", "-",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "ultrafast",
        str(output_mp4),
    ]
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        proc.stdin.write(np.array(first_img.convert("RGB")).tobytes())
        for out_i, state_i in enumerate(key_indices[1:], 1):
            state = states[state_i]
            log_text = "\n".join(log_snapshots[state_i]) if state_i < len(log_snapshots) else ""
            img = render_board_frame(state, images_dir, log_text=log_text or None)
            proc.stdin.write(np.array(img.convert("RGB")).tobytes())
        proc.stdin.close()
        proc.wait()
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd)
        print(f"動画: {output_mp4}  ({len(key_indices)} フレーム)")
    except (FileNotFoundError, subprocess.CalledProcessError):
        # フォールバック: ファイル経由
        for out_i, state_i in enumerate(key_indices):
            state = states[state_i]
            out_png = frames_dir / f"frame_{out_i:04d}.png"
            log_text = "\n".join(log_snapshots[state_i]) if state_i < len(log_snapshots) else ""
            render_board_frame(state, images_dir, output_path=out_png, log_text=log_text or None)
        pattern = frames_dir / "frame_%04d.png"
        fallback_cmd = [
            "ffmpeg", "-y",
            "-framerate", str(args.fps),
            "-i", str(pattern),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            str(output_mp4),
        ]
        subprocess.run(fallback_cmd, check=True, capture_output=True)
        print(f"動画: {output_mp4}  ({len(key_indices)} フレーム, fallback)")


if __name__ == "__main__":
    main()
