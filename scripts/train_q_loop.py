"""
Q-model の反復学習ループ（self-play → 学習 → self-play → ...）。

各イテレーションで:
  1. 現在の Q-model で self-play → JSONL 追記（リプレイバッファ蓄積）
  2. 蓄積データで Q-model を再学習
  3. A/B テスト（前世代 vs 今世代）でスコアを記録
  4. 勝率が改善していれば best_model を更新

例:
  python scripts/train_q_loop.py --iters 5 --games-per-iter 8000 --workers 8
  python scripts/train_q_loop.py --iters 3 --games-per-iter 5000 --start-model models/q_energy_attach_mlp.pt
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
    decks: str,
    q_model: Path | None,
    append: bool,
    q_support_model: str | None = None,
) -> None:
    cmd = [
        sys.executable, str(_REPO_ROOT / "scripts" / "selfplay_dataset.py"),
        "--n", str(n),
        "--workers", str(workers),
        "--epsilon", str(epsilon),
        "--decks", decks,
        "--out", str(out),
    ]
    if append:
        cmd.append("--append")
    if q_model is not None:
        cmd += ["--q-model", str(q_model)]
    if q_support_model is not None:
        cmd += ["--q-support-model", str(q_support_model)]
    subprocess.run(cmd, check=True)


def _run_train(
    data: Path,
    out_model: Path,
    hidden: int,
    epochs: int,
    batch_size: int,
    lr: float,
    tail_rows: int = 0,
    ap_scale: float = 0.0,
    attack_bonus: float = 0.0,
) -> None:
    cmd = [
        sys.executable, str(_REPO_ROOT / "scripts" / "train_value_attack_torch.py"),
        "--data", str(data),
        "--out", str(out_model),
        "--meta-out", str(out_model.with_suffix(".meta.json")),
        "--action-dim", "6",
        "--hidden", str(hidden),
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--lr", str(lr),
        "--turn-weight", "sqrt",
        "--value-key", "prize_delta",
        "--value-scale", "3.0",
    ]
    if tail_rows > 0:
        cmd += ["--tail-rows", str(tail_rows)]
    if ap_scale != 0.0:
        cmd += ["--ap-scale", str(ap_scale)]
    if attack_bonus != 0.0:
        cmd += ["--attack-bonus", str(attack_bonus)]
    subprocess.run(cmd, check=True)


def _ab_test(
    model_a: Path | None,
    model_b: Path | None,
    n_games: int,
    decks: str,
) -> float:
    """model_b の model_a に対する勝率を返す（0.0〜1.0）。"""
    from game import setup_game, run_game_auto

    combos: list[tuple[int, int]] = []
    for part in decks.split(":"):
        a, b = part.strip().split(",")
        combos.append((int(a.strip()), int(b.strip())))

    path_a = str(model_a) if model_a else None
    path_b = str(model_b) if model_b else None

    wins_b = 0
    total = 0
    for i in range(n_games):
        deck0, deck1 = combos[i % len(combos)]

        # 試合1: P0=B, P1=A
        s = setup_game(deck0=deck0, deck1=deck1,
                       use_attack_minimax=True, use_energy_attack_lookahead=True)
        s.q_energy_attach_model_path_by_player[0] = path_b
        s.q_energy_attach_model_path_by_player[1] = path_a
        w = run_game_auto(s)
        if w == 0:
            wins_b += 1
        total += 1

        # 試合2: P0=A, P1=B
        s2 = setup_game(deck0=deck1, deck1=deck0,
                        use_attack_minimax=True, use_energy_attack_lookahead=True)
        s2.q_energy_attach_model_path_by_player[0] = path_a
        s2.q_energy_attach_model_path_by_player[1] = path_b
        w2 = run_game_auto(s2)
        if w2 == 1:
            wins_b += 1
        total += 1

    return wins_b / max(1, total)


def main() -> None:
    parser = argparse.ArgumentParser(description="Q-model 反復学習ループ")
    parser.add_argument("--iters", type=int, default=5, help="反復回数（既定 5）")
    parser.add_argument("--games-per-iter", type=int, default=8000, help="1 イテレーション当たりの self-play 試合数（既定 8000）")
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1), help="並列ワーカー数")
    parser.add_argument("--epsilon", type=float, default=0.1, help="ε-greedy 確率（既定 0.1）")
    parser.add_argument("--decks", type=str, default="5,5:5,6:6,6", help="デッキ組み合わせ")
    parser.add_argument("--start-model", type=Path, default=_REPO_ROOT / "models" / "q_energy_attach_mlp.pt",
                        help="初期 Q-model パス（既存モデルから開始）")
    parser.add_argument("--out-dir", type=Path, default=_REPO_ROOT / "models" / "q_loop",
                        help="各世代のモデル保存ディレクトリ")
    parser.add_argument("--dataset", type=Path, default=_REPO_ROOT / "datasets" / "q_loop_buffer.jsonl",
                        help="リプレイバッファ JSONL（イテレーションをまたいで追記）")
    parser.add_argument("--hidden", type=int, default=128, help="MLP 隠れ層サイズ")
    parser.add_argument("--epochs", type=int, default=15, help="学習エポック数")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ab-games", type=int, default=200, help="A/B テスト試合数（片側）。0 でスキップ")
    parser.add_argument("--tail-rows", type=int, default=0,
                        help="学習時に使う末尾 N 行（最新データ優先。0 で全件）")
    parser.add_argument("--no-start-data", action="store_true",
                        help="初回 self-play を既存モデルなし（ヒューリスティック）で実行する")
    parser.add_argument("--q-support-model", type=str, default=None,
                        help="self-play 時に使うサポート Q-model（サポート後の状態でエネQを学習するため）")
    parser.add_argument("--ap-scale", type=float, default=0.0,
                        help="action_potential への重み（既定 0.0=無効）")
    parser.add_argument("--attack-bonus", type=float, default=0.0,
                        help="エネ貼り後に攻撃可能になった場合のボーナス（既定 0.0=無効）")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    current_model: Path | None = None
    if not args.no_start_data and args.start_model.is_file():
        current_model = args.start_model
        print(f"初期モデル: {current_model}")
    else:
        print("初期モデルなし（ヒューリスティックで初回 self-play）")

    results: list[dict] = []
    log_path = args.out_dir / "loop_log.jsonl"

    for it in range(1, args.iters + 1):
        print(f"\n{'='*60}")
        print(f"  イテレーション {it}/{args.iters}")
        print(f"{'='*60}")
        t0 = time.time()

        # ---- 1. Self-play ----
        print(f"\n[{it}] self-play: {args.games_per_iter} 試合 ...")
        append = args.dataset.is_file()
        _run_selfplay(
            n=args.games_per_iter,
            out=args.dataset,
            workers=args.workers,
            epsilon=args.epsilon,
            decks=args.decks,
            q_model=current_model,
            append=append,
            q_support_model=args.q_support_model,
        )
        n_rows = sum(1 for _ in args.dataset.open())
        print(f"  バッファ総行数: {n_rows:,}")

        # ---- 2. Train ----
        new_model = args.out_dir / f"q_iter{it:02d}.pt"
        print(f"\n[{it}] 学習: {new_model.name} ...")
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

        # ---- 3. A/B テスト ----
        win_rate = None
        if args.ab_games > 0:
            print(f"\n[{it}] A/B テスト: 新モデル vs 旧モデル ({args.ab_games*2} 試合) ...")
            win_rate = _ab_test(
                model_a=current_model,
                model_b=new_model,
                n_games=args.ab_games,
                decks=args.decks,
            )
            print(f"  新モデル勝率: {win_rate*100:.1f}%  旧モデル勝率: {(1-win_rate)*100:.1f}%")

        elapsed = time.time() - t0
        rec = {
            "iter": it,
            "model": str(new_model),
            "n_rows_buffer": n_rows,
            "win_rate_vs_prev": win_rate,
            "elapsed_s": round(elapsed, 1),
        }
        results.append(rec)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"\n  [{it}] 完了 ({elapsed:.0f}s)  ログ: {log_path}")

        current_model = new_model

    # ---- 最終サマリ ----
    print(f"\n{'='*60}")
    print("  ループ完了")
    print(f"{'='*60}")
    for r in results:
        wr = f"{r['win_rate_vs_prev']*100:.1f}%" if r["win_rate_vs_prev"] is not None else "  --  "
        print(f"  iter {r['iter']:02d}  勝率(vs前世代)={wr:>7}  rows={r['n_rows_buffer']:>8,}  {r['elapsed_s']:.0f}s")
    print(f"\n最終モデル: {current_model}")
    print(f"ログ: {log_path}")


if __name__ == "__main__":
    main()
