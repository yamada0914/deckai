"""
サポート+エネルギーの結合Q値モデルの反復学習ループ。

action = (support_id, energy_target) を 21 次元 one-hot で表現。
Q(state, support_onehot_15 + energy_onehot_6) → value

例:
  python scripts/train_joint_q_loop.py --iters 10 --games-per-iter 12000 --workers 8
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


def _run_selfplay(
    n: int,
    out: Path,
    workers: int,
    epsilon: float,
    support_epsilon: float,
    decks: str,
    append: bool,
) -> None:
    cmd = [
        sys.executable, str(_REPO_ROOT / "scripts" / "selfplay_dataset.py"),
        "--n", str(n),
        "--workers", str(workers),
        "--choice-type", "joint",
        "--epsilon", str(epsilon),
        "--support-epsilon", str(support_epsilon),
        "--decks", decks,
        "--out", str(out),
    ]
    if append:
        cmd.append("--append")
    subprocess.run(cmd, check=True)


def _run_train(
    data: Path,
    out_model: Path,
    hidden: int,
    epochs: int,
    batch_size: int,
    lr: float,
    tail_rows: int,
    ap_scale: float,
    attack_bonus: float,
) -> None:
    """joint 用の学習。action_dim=21, value_key=prize_delta。
    JSONL の support_action_id と energy_action_id を結合してone-hotにする前処理が必要。
    既存の train_value_attack_torch.py を使うため、JSONL を action_id=combined に変換して渡す。
    """
    import numpy as np
    import tempfile

    # JSONL を読み込み、action_id を結合して新ファイルに書き出す
    rows: list[dict] = []
    with data.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    if tail_rows > 0 and len(rows) > tail_rows:
        rows = rows[-tail_rows:]

    # support_action_id (0..14) を先頭 15 次元、energy_action_id (0..5) を後続 6 次元
    # → action_id は使わず、代わりに 21 次元 one-hot を直接作る
    # train_value_attack_torch.py は action_id → one-hot 変換するため、
    # combined_id = support * 6 + energy (0..89) で action_dim=90 にする
    tmp = Path(tempfile.mktemp(suffix=".jsonl"))
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            sid = int(r.get("support_action_id", 0))
            eid = int(r.get("energy_action_id", 0))
            combined = sid * 6 + eid
            r2 = dict(r)
            r2["action_id"] = combined
            f.write(json.dumps(r2, ensure_ascii=False) + "\n")

    cmd = [
        sys.executable, str(_REPO_ROOT / "scripts" / "train_value_attack_torch.py"),
        "--data", str(tmp),
        "--out", str(out_model),
        "--meta-out", str(out_model.with_suffix(".meta.json")),
        "--action-dim", "90",
        "--hidden", str(hidden),
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--lr", str(lr),
        "--turn-weight", "sqrt",
        "--value-key", "prize_delta",
        "--value-scale", "4.0",
        "--tail-rows", "0",
    ]
    if ap_scale != 0.0:
        cmd += ["--ap-scale", str(ap_scale)]
    if attack_bonus != 0.0:
        cmd += ["--attack-bonus", str(attack_bonus)]
    subprocess.run(cmd, check=True)
    tmp.unlink(missing_ok=True)


def _ab_test(
    model: Path | None,
    n_games: int,
    decks: str,
) -> float:
    """joint Q-model vs ヒューリスティックの勝率を返す。"""
    from game import setup_game, run_game_auto
    from game.support_policy import SUPPORT_ACTION_DIM, legal_support_mask
    from game.q_models import load_q_value_model_pt
    from game.encoders import encode_state_opening
    import numpy as np
    import torch

    combos: list[tuple[int, int]] = []
    for part in decks.split(":"):
        a, b = part.strip().split(",")
        combos.append((int(a.strip()), int(b.strip())))

    if model is None or not model.is_file():
        return 0.5

    q_model = load_q_value_model_pt(str(model))

    wins = 0
    total = 0
    for i in range(n_games):
        deck0, deck1 = combos[i % len(combos)]

        # P0=MODEL, P1=HEURISTIC
        s = setup_game(deck0=deck0, deck1=deck1, use_attack_minimax=True, use_energy_attack_lookahead=True)
        # joint Q model path をstateに設定
        s.joint_q_model_path_by_player = [str(model), None]
        w = run_game_auto(s)
        if w == 0:
            wins += 1
        total += 1

        # P0=HEURISTIC, P1=MODEL
        s2 = setup_game(deck0=deck1, deck1=deck0, use_attack_minimax=True, use_energy_attack_lookahead=True)
        s2.joint_q_model_path_by_player = [None, str(model)]
        w2 = run_game_auto(s2)
        if w2 == 1:
            wins += 1
        total += 1

    return wins / max(1, total)


def main() -> None:
    parser = argparse.ArgumentParser(description="サポート+エネルギー結合Q-model 反復学習ループ")
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--games-per-iter", type=int, default=12000)
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--epsilon", type=float, default=0.1, help="エネルギー ε-greedy")
    parser.add_argument("--support-epsilon", type=float, default=0.15, help="サポート ε-greedy")
    parser.add_argument("--decks", type=str, default="5,5:5,6:6,6")
    parser.add_argument("--out-dir", type=Path, default=_REPO_ROOT / "models" / "q_joint")
    parser.add_argument("--dataset", type=Path, default=_REPO_ROOT / "datasets" / "q_joint_buffer.jsonl")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ab-games", type=int, default=0, help="A/B テスト試合数（片側）。0 でスキップ")
    parser.add_argument("--tail-rows", type=int, default=300000)
    parser.add_argument("--ap-scale", type=float, default=1.0)
    parser.add_argument("--attack-bonus", type=float, default=2.0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out_dir / "loop_log.jsonl"
    results: list[dict] = []

    for it in range(1, args.iters + 1):
        print(f"\n{'='*60}")
        print(f"  イテレーション {it}/{args.iters}")
        print(f"{'='*60}")
        t0 = time.time()

        # ---- 1. Self-play (joint) ----
        print(f"\n[{it}] self-play: {args.games_per_iter} 試合 (joint, ε={args.epsilon}/{args.support_epsilon}) ...")
        append = args.dataset.is_file()
        _run_selfplay(
            n=args.games_per_iter,
            out=args.dataset,
            workers=args.workers,
            epsilon=args.epsilon,
            support_epsilon=args.support_epsilon,
            decks=args.decks,
            append=append,
        )
        n_rows = sum(1 for _ in args.dataset.open())
        print(f"  バッファ総行数: {n_rows:,}")

        # ---- 2. Train ----
        new_model = args.out_dir / f"q_joint_iter{it:02d}.pt"
        print(f"\n[{it}] 学習: {new_model.name} (action_dim=90) ...")
        _run_train(
            data=args.dataset,
            out_model=new_model,
            hidden=args.hidden,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            tail_rows=args.tail_rows,
            ap_scale=args.ap_scale,
            attack_bonus=args.attack_bonus,
        )

        elapsed = time.time() - t0
        rec = {
            "iter": it,
            "model": str(new_model),
            "n_rows_buffer": n_rows,
            "elapsed_s": round(elapsed, 1),
        }
        results.append(rec)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"\n  [{it}] 完了 ({elapsed:.0f}s)")

    # ---- 最終サマリ ----
    print(f"\n{'='*60}")
    print("  ループ完了")
    print(f"{'='*60}")
    for r in results:
        print(f"  iter {r['iter']:02d}  rows={r['n_rows_buffer']:>8,}  {r['elapsed_s']:.0f}s")
    print(f"\n最終モデル: {args.out_dir}/q_joint_iter{args.iters:02d}.pt")


if __name__ == "__main__":
    main()
