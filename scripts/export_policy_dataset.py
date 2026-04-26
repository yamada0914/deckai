"""
battle_states.pkl（states + choice_log）から、学習用データセットを作る。

出力形式（JSONL）:
  {"state": [...], "action_id": int, "win": 0|1, "player": 0|1, "type": "...", "turn": int, "battle_id": "..."}

  retreat_before_attack の行では top2_gap / retreat_policy_entropy / retreat_policy_schedule_progress / total_env_steps を付与（ train_pi_torch の重み付け用）。
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from game.encoders import encode_state_basic, encode_state_opening, energy_attach_action_id_from_target_card_id


def _iter_battle_pkls(battles_dir: Path) -> list[Path]:
    return sorted(battles_dir.glob("*/battle_states.pkl"))


def _battle_id_from_pkl(pkl_path: Path, battles_dir: Path) -> str:
    try:
        return str(pkl_path.parent.relative_to(battles_dir))
    except Exception:
        return pkl_path.parent.name


def main() -> None:
    parser = argparse.ArgumentParser(description="battle_states.pkl から (state, action, win, success) データセットを書き出す")
    parser.add_argument("--battles-dir", type=Path, default=_REPO_ROOT / "battles" / "train_weights", help="*/battle_states.pkl を探すディレクトリ")
    parser.add_argument("--out", type=Path, default=_REPO_ROOT / "datasets" / "policy_energy_attach.jsonl", help="出力 JSONL")
    parser.add_argument(
        "--choice-type",
        type=str,
        default="energy_attach",
        choices=["energy_attach", "retreat_before_attack", "support_policy"],
        help="対象の choice_log.type",
    )
    parser.add_argument(
        "--battle-id",
        type=str,
        default=None,
        metavar="ID",
        help="この対戦フォルダだけ処理する（例: 20260322_142219）。retreat_before_attack の有無を 1 試合で確認するときに便利",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="opening",
        choices=["basic", "opening"],
        help="状態エンコーダー（basic=21次元 / opening=105次元、既定）",
    )
    args = parser.parse_args()

    if args.battle_id:
        single = args.battles_dir / args.battle_id / "battle_states.pkl"
        if not single.is_file():
            print(f"見つかりません: {single}", file=sys.stderr)
            raise SystemExit(1)
        pkls = [single.resolve()]
    else:
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
    n_pkls = len(pkls)
    n_battles_ok = 0
    type_counts: Counter[str] = Counter()

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

            n_battles_ok += 1
            for e in choice_log:
                t = e.get("type")
                type_counts[str(t if t is not None else "<none>")] += 1

            battle_id = _battle_id_from_pkl(pkl_path, args.battles_dir)
            _encode = encode_state_opening if args.encoder == "opening" else encode_state_basic

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

                win = 1 if winner == player else 0
                if args.choice_type == "energy_attach":
                    # energy_policy_action_id があればそちらを優先（ε-greedy 収集データ）
                    aid = e.get("energy_policy_action_id")
                    if aid is not None:
                        action_id = int(aid)
                    else:
                        target_card_id = e.get("card_id", "") or ""
                        action_id = energy_attach_action_id_from_target_card_id(s_before, player, target_card_id)
                    if action_id is None:
                        n_no_action += 1
                        n_skipped += 1
                        continue
                    row = {
                        "state": _encode(s_before, player),
                        "action_id": int(action_id),
                        "win": win,
                        "success": win,
                        "player": player,
                        "type": args.choice_type,
                        "turn": int(e.get("turn", 0)),
                        "battle_id": battle_id,
                    }
                    # top2_gap（ε-greedy で収集したデータに付与される）
                    tg = e.get("energy_policy_top2_gap")
                    if tg is not None:
                        row["top2_gap"] = float(tg)
                elif args.choice_type == "support_policy":
                    aid = e.get("support_policy_action_id")
                    if aid is None:
                        n_no_action += 1
                        n_skipped += 1
                        continue
                    action_id = int(aid)
                    row = {
                        "state": _encode(s_before, player),
                        "action_id": action_id,
                        "win": win,
                        "success": win,
                        "player": player,
                        "type": args.choice_type,
                        "turn": int(e.get("turn", 0)),
                        "battle_id": battle_id,
                    }
                    tg = e.get("support_policy_top2_gap")
                    if tg is not None:
                        row["top2_gap"] = float(tg)
                else:  # retreat_before_attack
                    aid = e.get("retreat_policy_action_id", e.get("policy_action_id"))
                    if aid is None:
                        n_no_action += 1
                        n_skipped += 1
                        continue
                    action_id = int(aid)
                    tg = e.get("top2_gap")
                    if tg is None:
                        tg = e.get("retreat_policy_top2_gap")
                    row = {
                        "state": _encode(s_before, player),
                        "action_id": action_id,
                        "win": win,
                        "success": win,
                        "player": player,
                        "type": args.choice_type,
                        "turn": int(e.get("turn", 0)),
                        "battle_id": battle_id,
                        "total_env_steps": int(getattr(s_before, "total_env_steps", 0)),
                    }
                    if tg is not None:
                        row["top2_gap"] = float(tg)
                    rpe = e.get("retreat_policy_entropy")
                    if rpe is not None:
                        row["retreat_policy_entropy"] = float(rpe)
                    rsp = e.get("retreat_policy_schedule_progress")
                    if rsp is not None:
                        row["retreat_policy_schedule_progress"] = float(rsp)

                f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_rows += 1

    print(f"出力: {out_path}")
    print(f"rows: {n_rows}  skipped: {n_skipped}  no_frame_index: {n_no_frame}  no_action_id: {n_no_action}")
    print(f"battle_states.pkl: {n_pkls} 件  勝者確定・choice_log あり: {n_battles_ok} 件")

    if n_rows == 0:
        print("", file=sys.stderr)
        print("警告: 出力行が 0 件です。", file=sys.stderr)
        print(f"  --choice-type {args.choice_type!r} に一致する choice_log がありません。", file=sys.stderr)
        print("  考えられる原因:", file=sys.stderr)
        print("    · battles-dir に pkl が無い、または別フォルダを指定する必要がある", file=sys.stderr)
        print("    · 古い対戦では retreat_before_attack が未記録（攻撃前にげのログ追加前の pkl）", file=sys.stderr)
        print("    · 試合中に一度も「攻撃前にげ」分岐に入っていない（該当局面が無い）", file=sys.stderr)
        if args.battle_id and args.choice_type == "retreat_before_attack":
            print("", file=sys.stderr)
            print(f"  --battle-id {args.battle_id!r} を指定中: このフォルダの pkl が古いと、", file=sys.stderr)
            print("    attack はあっても retreat_before_attack は 1 件もありません（当時は未実装／未記録）。", file=sys.stderr)
            print("    getattr(attached_tool) 修正後に record_game で取り直した対戦 ID を指定してください。", file=sys.stderr)
        if args.choice_type == "retreat_before_attack" and type_counts.get("retreat", 0) > 0:
            print("", file=sys.stderr)
            print("  ※ type に retreat はありますが、別経路のログです（例: にげ処理の通常記録）。", file=sys.stderr)
            print("    学習用の retreat_before_attack は try_retreat_before_attack_policy 経由でのみ付きます。", file=sys.stderr)
            print("    現行コードで record_game / run_training_games から対戦を取り直すと記録されます。", file=sys.stderr)
        print("  対処: record_game / run_training_games で対戦を取り直すか、battles 直下など pkl があるパスを指定してください。", file=sys.stderr)
        if type_counts:
            print(f"  今回スキャンした choice_log.type の集計（上位 20）: {type_counts.most_common(20)}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
