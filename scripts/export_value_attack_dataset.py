"""
battle_states.pkl（states + choice_log）から、attack 用 value データセットを作る。

学習目的（1 手価値）:
  入力  : state_before_action, action_id(attack index)
  目的値: choice_log に保存された eval（行動直後の盤面評価）

出力 JSONL:
  {"state":[...], "action_id": int, "value": float, "player": 0|1, "type":"attack", "turn":..., "battle_id":"..."}

注意:
  - choice_log.type == "attack" のみ対象
  - 候補が 1 個以下のターンは除外（policy の飽和を防ぐ）
  - action_id は attacks 配列の index（0..max_attacks-1）に揃える
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from game.encoders import encode_state_basic


def _iter_battle_pkls(battles_dir: Path) -> list[Path]:
    return sorted(battles_dir.glob("*/battle_states.pkl"))


def _battle_id_from_pkl(pkl_path: Path, battles_dir: Path) -> str:
    try:
        return str(pkl_path.parent.relative_to(battles_dir))
    except Exception:
        return pkl_path.parent.name


def main() -> None:
    parser = argparse.ArgumentParser(description="battle_states.pkl から attack value データセットを書き出す")
    parser.add_argument("--battles-dir", type=Path, default=_REPO_ROOT / "battles" / "train_weights", help="*/battle_states.pkl を探すディレクトリ")
    parser.add_argument("--out", type=Path, default=_REPO_ROOT / "datasets" / "value_attack.jsonl", help="出力 JSONL")
    parser.add_argument("--max-attacks", type=int, default=4, help="最大技数（action_dim）。これを超える技 index は無視する")
    parser.add_argument("--choice-type", type=str, default="attack", choices=["attack"], help="対象の choice_log.type")
    args = parser.parse_args()

    pkls = _iter_battle_pkls(args.battles_dir)
    if not pkls:
        print(f"pkl が見つかりません: {args.battles_dir}/*/battle_states.pkl", file=sys.stderr)
        raise SystemExit(1)

    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_rows = 0
    n_skipped = 0
    n_no_frame = 0
    n_no_action = 0
    n_no_eval = 0

    max_attacks = int(args.max_attacks)

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
                continue

            choice_log = getattr(last, "choice_log", []) or []
            if not choice_log:
                continue

            battle_id = _battle_id_from_pkl(pkl_path, args.battles_dir)

            for e in choice_log:
                if e.get("type") != args.choice_type:
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

                s_before = states[frame_index]
                active = getattr(s_before.players[player], "active", None)
                attacks = getattr(getattr(active, "card", None), "attacks", []) or []
                if len(attacks) <= 1:
                    n_skipped += 1
                    continue

                attack_name = e.get("attack_name", "") or ""
                action_id = None
                for i, atk in enumerate(attacks):
                    if getattr(atk, "name", "") == attack_name:
                        action_id = i
                        break
                if action_id is None:
                    n_no_action += 1
                    n_skipped += 1
                    continue
                if not (0 <= action_id < max_attacks):
                    n_skipped += 1
                    continue

                value = e.get("eval", None)
                if value is None:
                    n_no_eval += 1
                    n_skipped += 1
                    continue

                row = {
                    "state": encode_state_basic(s_before, player),
                    "action_id": int(action_id),
                    "value": float(value),
                    "player": player,
                    "type": "attack",
                    "turn": int(e.get("turn", 0)),
                    "battle_id": battle_id,
                }
                f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_rows += 1

    print(f"出力: {out_path}")
    print(
        f"rows: {n_rows}  skipped: {n_skipped}  no_frame_index: {n_no_frame}  no_action_id: {n_no_action}  no_eval: {n_no_eval}"
    )


if __name__ == "__main__":
    main()

