"""
記録した状態スナップショットから盤面フレームを描画し、動画にまとめる。

  オプションなしなら battles/ 内の最新シミュレーションで動画を作成。
  python scripts/make_video.py                              # 最新対戦 → battles/<最新ID>/battle.mp4
  python scripts/make_video.py --battle-id 20250226_143052   # 指定した対戦で動画作成
"""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import argparse
import pickle
import subprocess
from board_render import render_board_frame

BATTLES_DIR = _REPO_ROOT / "battles"


def _board_signature(state) -> tuple:
    """盤面の見た目を決める要素だけを取り出し、同一フレームの間引きに使う。"""
    parts: list = [state.turn_count, state.current_player, state.winner]
    for p in state.players:
        if p.active is None:
            parts.append(None)
        else:
            parts.append((getattr(p.active.card, "id", ""), p.active.hp))
        bench = tuple((getattr(bp.card, "id", ""), bp.hp) for bp in p.bench)
        parts.append(bench)
        parts.append(len(p.hand))
        parts.append(len(p.deck))
        parts.append(len(p.discard))
        parts.append(len(p.prize_pile))
    return tuple(parts)


def _key_frame_indices(states: list) -> list[int]:
    """盤面が前フレームと変わったインデックスだけを返す（先頭は必ず含む）。"""
    if not states:
        return []
    indices = [0]
    prev_sig = _board_signature(states[0])
    for i in range(1, len(states)):
        sig = _board_signature(states[i])
        if sig != prev_sig:
            indices.append(i)
            prev_sig = sig
    return indices


def _latest_battle_id() -> str | None:
    """battles/ 内で battle_states.pkl がある対戦のうち、最も新しい対戦 ID を返す。なければ None。"""
    if not BATTLES_DIR.is_dir():
        return None
    latest_id = None
    latest_mtime = 0.0
    for sub in BATTLES_DIR.iterdir():
        if not sub.is_dir():
            continue
        pkl = sub / "battle_states.pkl"
        if pkl.is_file():
            m = pkl.stat().st_mtime
            if m > latest_mtime:
                latest_mtime = m
                latest_id = sub.name
    return latest_id


def main() -> None:
    parser = argparse.ArgumentParser(description="状態スナップショットから盤面動画を生成する")
    parser.add_argument("--battle-id", type=str, default=None, metavar="ID", help="対戦 ID（指定時は battles/<ID>/ 内の pkl → battle.mp4, フレームは frames/）")
    parser.add_argument("--states", type=Path, default=None, metavar="PATH", help="状態 pickle（--battle-id 未指定時のデフォルト: battle_states.pkl）")
    parser.add_argument("-o", "--output", type=Path, default=None, metavar="PATH", help="出力動画（--battle-id 未指定時のデフォルト: battle.mp4）")
    parser.add_argument("--fps", type=float, default=0.33, help="動画の FPS（デフォルト: 0.33 = 1 フレーム約 3 秒表示。小さくするほどフレーム間の間隔が長い）")
    parser.add_argument("--frames-dir", type=Path, default=None, help="フレーム画像を残すディレクトリ（省略時: battle-id なら battles/<ID>/frames/）")
    args = parser.parse_args()

    if args.states is not None:
        states_path = args.states
        output_mp4 = args.output or _REPO_ROOT / "battle.mp4"
        frames_dir = args.frames_dir or _REPO_ROOT / "frames"
        effective_battle_id = None
    elif args.battle_id:
        battle_dir = BATTLES_DIR / args.battle_id
        states_path = battle_dir / "battle_states.pkl"
        output_mp4 = args.output or battle_dir / "battle.mp4"
        frames_dir = args.frames_dir or battle_dir / "frames"
        effective_battle_id = args.battle_id
    else:
        latest_id = _latest_battle_id()
        if latest_id is None:
            fallback = _REPO_ROOT / "battle_states.pkl"
            if fallback.is_file():
                states_path = fallback
                output_mp4 = args.output or _REPO_ROOT / "battle.mp4"
                frames_dir = args.frames_dir or _REPO_ROOT / "frames"
                effective_battle_id = None
            else:
                print("エラー: battles/ に対戦がありません。先に python scripts/record_game.py を実行してください。", file=sys.stderr)
                sys.exit(1)
        else:
            battle_dir = BATTLES_DIR / latest_id
            states_path = battle_dir / "battle_states.pkl"
            output_mp4 = args.output or battle_dir / "battle.mp4"
            frames_dir = args.frames_dir or battle_dir / "frames"
            effective_battle_id = latest_id
            print(f"最新の対戦を使用: {latest_id}")
    if args.output is not None:
        output_mp4 = args.output
    if not states_path.is_file():
        print(f"エラー: 状態ファイルが見つかりません: {states_path}", file=sys.stderr)
        print("先に python scripts/record_game.py を実行するか、--battle-id で対戦 ID を指定してください。", file=sys.stderr)
        sys.exit(1)

    with open(states_path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict):
        states = data.get("states", [])
        log_snapshots = data.get("log_snapshots", [])
    else:
        states = data
        log_snapshots = []
    if not states:
        print("エラー: 状態が 0 件です。", file=sys.stderr)
        sys.exit(1)

    images_dir = _REPO_ROOT / "card_images"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for f in frames_dir.glob("frame_*.png"):
        f.unlink()

    key_indices = _key_frame_indices(states)
    for out_i, state_i in enumerate(key_indices):
        state = states[state_i]
        out_png = frames_dir / f"frame_{out_i:04d}.png"
        log_text = "\n".join(log_snapshots[state_i]) if state_i < len(log_snapshots) else ""
        render_board_frame(state, images_dir, output_path=out_png, log_text=log_text or None)
    print(f"フレームを {len(key_indices)} 枚出力しました（元 {len(states)} 枚から間引き）: {frames_dir}/")

    if effective_battle_id:
        print(f"対戦 ID: {effective_battle_id}")
    pattern = frames_dir / "frame_%04d.png"
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(args.fps),
        "-i", str(pattern),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(output_mp4),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"動画を保存しました: {output_mp4}")
    except FileNotFoundError:
        print("ffmpeg が見つかりません。インストール後、次で動画を作成できます:", file=sys.stderr)
        print(f"  ffmpeg -y -framerate {args.fps} -i \"{frames_dir}/frame_%04d.png\" -c:v libx264 -pix_fmt yuv420p {output_mp4}", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"エラー: ffmpeg が失敗しました: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
