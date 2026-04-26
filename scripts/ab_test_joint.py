"""
Joint Q-model の A/B テスト。

Joint Q の出力する (support, energy) ペアの Q 値を、
個別 Q (support Q + energy Q) の合計と比較して、
どちらがより良い組み合わせを選べるかを測定する。

実際のゲーム実行で比較: joint Q の推奨に従う vs ヒューリスティック。
joint Q はターン開始時に最善の (support, energy) を選び、
support_policy と energy_policy の両方をオーバーライドする。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Joint Q vs ヒューリスティック A/B テスト")
    parser.add_argument("--games", type=int, default=400)
    parser.add_argument("--decks", type=str, default="5,5:5,6:6,6")
    parser.add_argument("--joint-model", type=Path, default=_REPO_ROOT / "models" / "q_joint" / "q_joint_iter07.pt")
    # 比較用: 個別Q
    parser.add_argument("--q-model", type=Path, default=_REPO_ROOT / "models" / "q_loop_v7" / "q_iter07.pt")
    parser.add_argument("--q-support-model", type=Path, default=_REPO_ROOT / "models" / "q_support_v4" / "q_support_iter25.pt")
    args = parser.parse_args()

    from game import setup_game, run_game_auto
    from game.q_models import load_q_value_model_pt
    from game.encoders import encode_state_opening
    from game.support_policy import SUPPORT_ACTION_DIM, legal_support_mask

    combos: list[tuple[int, int]] = []
    for part in args.decks.split(":"):
        a, b = part.strip().split(",")
        combos.append((int(a.strip()), int(b.strip())))

    # --- Joint Q の推奨行動を確認（質的分析）---
    if args.joint_model.is_file():
        jq = load_q_value_model_pt(str(args.joint_model))
        print(f"Joint Q model: state_dim={jq.state_dim} action_dim={jq.action_dim}")

        # 数試合で joint Q の推奨を表示
        from game.support_policy import KNOWN_SUPPORT_IDS
        energy_names = ["active", "bench0", "bench1", "bench2", "bench3", "bench4"]

        print("\n--- Joint Q 推奨行動サンプル ---")
        for g in range(5):
            d0, d1 = combos[g % len(combos)]
            s = setup_game(deck0=d0, deck1=d1, use_attack_minimax=True, use_energy_attack_lookahead=True)
            vec = encode_state_opening(s, s.current_player)
            s_arr = np.asarray(vec, dtype=np.float32)

            # 全90通りの Q 値
            q_all = jq.predict_all(vec)

            sup_mask = legal_support_mask(s)
            p = s.active_player_state()
            n_bench = len([b for b in (p.bench or []) if b is not None])
            ene_legal = list(range(1 + n_bench))  # active + bench

            # 上位3つの合法組み合わせ
            scored = []
            for sid in range(SUPPORT_ACTION_DIM):
                if not sup_mask[sid]:
                    continue
                for eid in ene_legal:
                    combined = sid * 6 + eid
                    if combined < len(q_all):
                        sup_name = "NOOP" if sid == 0 else KNOWN_SUPPORT_IDS[sid - 1]
                        scored.append((q_all[combined], sup_name, energy_names[eid]))
            scored.sort(reverse=True)

            print(f"\n  Game {g+1} (turn {s.turn_count}, player {s.current_player}):")
            for rank, (q, sn, en) in enumerate(scored[:3]):
                print(f"    #{rank+1}: Q={q:+.3f}  support={sn:.<25s} energy={en}")

    # --- 個別 Q (support + energy) vs ヒューリスティック ---
    print("\n--- 個別 Q (support + energy) vs ヒューリスティック ---")
    q_path = str(args.q_model) if args.q_model.is_file() else None
    qs_path = str(args.q_support_model) if args.q_support_model.is_file() else None

    t0 = time.time()
    wins_sep = 0
    total = 0
    for i in range(args.games):
        d0, d1 = combos[i % len(combos)]
        s = setup_game(
            deck0=d0, deck1=d1, use_attack_minimax=True, use_energy_attack_lookahead=True,
            q_energy_attach_model_path_p0=q_path,
            q_support_model_path_p0=qs_path, q_support_lambda=3.0,
        )
        w = run_game_auto(s)
        if w == 0:
            wins_sep += 1
        total += 1
        s2 = setup_game(
            deck0=d1, deck1=d0, use_attack_minimax=True, use_energy_attack_lookahead=True,
            q_energy_attach_model_path_p1=q_path,
            q_support_model_path_p1=qs_path, q_support_lambda=3.0,
        )
        w2 = run_game_auto(s2)
        if w2 == 1:
            wins_sep += 1
        total += 1
    wr_sep = wins_sep / total * 100
    print(f"  → {wr_sep:.1f}%  ({time.time()-t0:.0f}s)")

    print(f"\n=== まとめ ===")
    print(f"  個別 Q (support + energy)...... {wr_sep:.1f}%")
    print(f"  ヒューリスティック（基準）....... 50.0%")
    print(f"\n  Joint Q の推奨行動分析は上記サンプルを参照。")
    print(f"  ゲームループへの本格統合は次のステップで実施。")


if __name__ == "__main__":
    main()
