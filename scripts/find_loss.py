"""
負けパターンを探して動画を生成する。

使い方:
  python scripts/find_loss.py                     # 各デッキ1つずつ
  python scripts/find_loss.py --count 3            # 各デッキ3つずつ
  python scripts/find_loss.py --deck1 7            # コライドンのみ
  python scripts/find_loss.py --min-turns 10       # 10ターン以上の負けのみ
"""
from __future__ import annotations

import argparse
import pickle
import shutil
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


def main():
    parser = argparse.ArgumentParser(description="負けパターンを探して動画を生成")
    parser.add_argument("--count", type=int, default=1, help="各デッキから見つける負け数")
    parser.add_argument("--deck0", type=int, default=5, help="自分のデッキ番号")
    parser.add_argument("--deck1", type=int, nargs="*", default=[7], help="相手のデッキ番号")
    parser.add_argument("--min-turns", type=int, default=8, help="最低ターン数（初手事故除外）")
    parser.add_argument("--max-seed", type=int, default=2000, help="探索する最大seed")
    parser.add_argument("--no-video", action="store_true", help="動画生成をスキップ")
    args = parser.parse_args()

    deck_names = {5: "unfair_lucario", 6: "belt_lucario", 7: "koraidon", 8: "miraidon"}

    from scripts.record_game import _run_and_record

    results = []

    for deck1 in args.deck1:
        name = deck_names.get(deck1, f"deck{deck1}")
        found = 0
        for seed in range(args.max_seed):
            if found >= args.count:
                break
            bid = f"_scan_{seed}_{name}"
            import io, contextlib
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                _, _ = _run_and_record(seed=seed, deck0=args.deck0, deck1=deck1, battle_id=bid)
            sp = Path(f"battles/{bid}/battle_states.pkl")
            with open(sp, "rb") as f:
                data = pickle.load(f)
            states = data.get("states", data) if isinstance(data, dict) else data
            winner = states[-1].winner if states else None
            turn_count = states[-1].turn_count if states else 0
            shutil.rmtree(f"battles/{bid}", ignore_errors=True)

            if seed > 0 and seed % 50 == 0:
                print(f"  scanning... seed {seed}/{args.max_seed}", end="\r", flush=True)
            if winner != 0 and turn_count >= args.min_turns:
                found += 1
                # 動画生成
                vid_id = f"loss_{name}_{seed}"
                if args.no_video:
                    _, _ = _run_and_record(seed=seed, deck0=args.deck0, deck1=deck1, battle_id=vid_id)
                    print(f"[LOSS] seed={seed} vs {name} turns={turn_count} → battles/{vid_id}/battle.log")
                else:
                    import subprocess
                    print(f"\n[FOUND] seed={seed} vs {name} turns={turn_count} — 動画生成中...")
                    subprocess.run(
                        [sys.executable, "scripts/quick_battle.py",
                         "--seed", str(seed), "--deck0", str(args.deck0),
                         "--deck1", str(deck1), "--id", vid_id],
                        cwd=str(_REPO_ROOT),
                    )
                    # 負け確認
                    log_path = _REPO_ROOT / "battles" / vid_id / "battle.log"
                    if log_path.exists():
                        last_lines = log_path.read_text().strip().split("\n")[-3:]
                        is_loss = any("勝ち" in l and "アンフェアルカリオ" not in l for l in last_lines)
                        if is_loss:
                            mp4 = _REPO_ROOT / "battles" / vid_id / "battle.mp4"
                            print(f"[LOSS] seed={seed} vs {name} turns={turn_count} → {mp4}")
                            results.append((seed, deck1, name, turn_count, str(mp4)))
                        else:
                            print(f"[SKIP] seed={seed} vs {name} → 動画生成時は勝ち（乱数ずれ）")
                            found -= 1  # やり直し
                            shutil.rmtree(f"battles/{vid_id}", ignore_errors=True)

    if results:
        print(f"\n=== {len(results)} 件の負け動画 ===")
        for seed, d1, name, turns, mp4 in results:
            print(f"  seed={seed} vs {name} ({turns}T) → {mp4}")


if __name__ == "__main__":
    main()
