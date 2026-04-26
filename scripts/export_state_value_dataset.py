"""
battle_states.pkl から V(s) 用データセットを書き出す。

ラベル = ゲーム結果（勝ち +1 / 負け -1）に割引を掛けた値。

  value = (gamma ** (T - t)) * result

  result : +1（勝ち）/ -1（負け）/ 0（引き分け・winner なし）
  T      : ゲーム総ターン数
  t      : その選択のターン番号
  gamma  : 割引率（--gamma、デフォルト 0.99）

gamma=1.0 にすると割引なし（純粋なゲーム結果のみ）。

対象: type が --choice-types に含まれる全エントリ（デフォルト: attack, energy_attach）

出力 JSONL:
  {"state":[...], "value": float, "player": 0|1, "type":"attack"|..., "turn":..., "battle_id":"..."}
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from game.encoders import encode_state_basic, encode_state_drapa


def _iter_battle_pkls(battles_dir: Path) -> list[Path]:
    return sorted(battles_dir.glob("*/battle_states.pkl"))


def _battle_id_from_pkl(pkl_path: Path, battles_dir: Path) -> str:
    try:
        return str(pkl_path.parent.relative_to(battles_dir))
    except Exception:
        return pkl_path.parent.name


def main() -> None:
    parser = argparse.ArgumentParser(description="battle_states.pkl から V(s) データセットを書き出す")
    parser.add_argument(
        "--battles-dir",
        type=Path,
        default=_REPO_ROOT / "battles" / "train_weights",
        help="*/battle_states.pkl を探すディレクトリ",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_REPO_ROOT / "datasets" / "state_value.jsonl",
        help="出力 JSONL",
    )
    parser.add_argument(
        "--choice-types",
        type=str,
        default="attack,energy_attach",
        help="カンマ区切りの対象 type（例: attack,energy_attach）",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="割引率（0〜1。1.0 で割引なし。デフォルト 0.99）",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="basic",
        choices=["basic", "drapa"],
        help="エンコーダ（basic=21次元, drapa=41次元ドラパルト特化）",
    )
    args = parser.parse_args()

    target_types = {t.strip() for t in args.choice_types.split(",") if t.strip()}
    gamma = float(args.gamma)

    pkls = _iter_battle_pkls(args.battles_dir)
    if not pkls:
        print(f"pkl が見つかりません: {args.battles_dir}/*/battle_states.pkl", file=sys.stderr)
        raise SystemExit(1)

    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_rows = 0
    n_skipped = 0
    n_no_frame = 0
    n_no_winner = 0

    with out_path.open("w", encoding="utf-8") as f_out:
        for pkl_path in pkls:
            with pkl_path.open("rb") as f:
                data = pickle.load(f)

            states = data.get("states", [])
            if not states:
                continue

            last = states[-1]
            winner = getattr(last, "winner", None)
            if winner not in (0, 1):
                n_no_winner += 1
                continue

            # ゲーム総ターン数（割引の基準）
            game_length = int(getattr(last, "turn_count", len(states)))

            choice_log = getattr(last, "choice_log", []) or []
            battle_id = _battle_id_from_pkl(pkl_path, args.battles_dir)

            for e in choice_log:
                if e.get("type") not in target_types:
                    continue

                player = int(e.get("player", 0))
                frame_index = e.get("frame_index", None)
                if frame_index is None:
                    n_no_frame += 1
                    n_skipped += 1
                    continue
                if frame_index < 0 or frame_index >= len(states):
                    n_skipped += 1
                    continue

                turn = int(e.get("turn", 0))
                result = 1.0 if player == winner else -1.0
                steps_to_end = max(0, game_length - turn)
                value = (gamma ** steps_to_end) * result

                s_before = states[frame_index]
                row = {
                    "state": (encode_state_drapa if args.encoder == "drapa" else encode_state_basic)(s_before, player),
                    "value": float(value),
                    "player": player,
                    "type": str(e.get("type", "")),
                    "turn": turn,
                    "battle_id": battle_id,
                }
                f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_rows += 1

    print(f"出力: {out_path}")
    print(f"rows: {n_rows}  skipped: {n_skipped}  no_frame_index: {n_no_frame}  no_winner: {n_no_winner}")


if __name__ == "__main__":
    main()
