"""
インプロセス self-play でデータセットを直接生成する。

pkl を経由しないため高速（deepcopy なし・サブプロセスなし）。
ε-greedy（--epsilon / --support-epsilon）でランダム探索を混ぜてデータを多様化する。
--workers で並列実行（既定: CPU 数）。

出力形式（JSONL）:
  {"state": [...], "action_id": int, "win": 0|1, "player": 0|1,
   "type": "energy_attach"|"support_policy", "turn": int, "battle_id": str,
   "epsilon_greedy": bool}

例:
  python scripts/selfplay_dataset.py --n 25000 --out datasets/selfplay_energy.jsonl
  python scripts/selfplay_dataset.py --n 10000 --decks 5,5 --epsilon 0.15 --workers 4
  python scripts/selfplay_dataset.py --n 10000 --decks 5,5:5,6:6,6 --choice-type support_policy --support-epsilon 0.3 --out datasets/selfplay_support.jsonl
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


def _compute_action_potential(state, player: int) -> float:
    """
    次ターン開始時点での行動可能性スコア（0〜5）。
    デッキ構築原則「毎ターン エネ貼る / サポ使う / 攻撃する」を報酬に反映。

      +1: アクティブポケモンが攻撃できる（最小コストを満たしている）
      +1: 手札に進化カードあり、対応ポケモンが場にいる
      +1: エネルギーを貼れる対象がいる（アクティブまたはベンチにポケモンがいる）
      +1: 手札にサポートがある（毎ターンサポート原則）
      +1: ベンチに育成中の次アタッカーがいる（バックアップ原則：エネ1枚以上）
    record_frame のコールバック内で呼ぶ想定。
    """
    from game.encoders import _can_attack_now
    from card import is_support

    p = state.players[player]
    active = p.active
    if active is None:
        return 0.0

    score = 0.0

    # 1. 攻撃できるか（アクティブポケモンの最小技コストを満たすか）
    if _can_attack_now(active) > 0.0:
        score += 1.0

    # 2. 進化できるか（手札の進化カードが場のポケモンに対応しているか）
    field_pokemon = [active] + [b for b in (p.bench or []) if b is not None]
    field_ids = {getattr(bp.card, "id", "") for bp in field_pokemon if bp is not None}
    for c in (p.hand or []):
        evolves_from = getattr(c, "evolves_from", None)
        if evolves_from and evolves_from in field_ids:
            score += 1.0
            break

    # 3. エネルギーを貼れる対象がいるか（ポケモンが 1 体以上いれば常に貼れる）
    if field_pokemon:
        score += 1.0

    # 4. 毎ターンサポート原則: 次ターン使えるサポートが手札にある
    has_support = any(is_support(c) for c in (p.hand or []))
    if has_support:
        score += 1.0

    # 5. バックアップ原則: ベンチに育成中の次アタッカーがいる（エネ1枚以上）
    backup_ready = False
    for bp in (p.bench or []):
        if bp is None:
            continue
        if getattr(bp, "attached_energy", 0) >= 1:
            backup_ready = True
            break
    if backup_ready:
        score += 1.0

    return score


def _run_one_game(
    deck0: int,
    deck1: int,
    epsilon: float,
    game_idx: int,
    q_model_path: str | None = None,
    choice_type: str = "energy_attach",
    support_epsilon: float = 0.2,
    q_support_model_path: str | None = None,
) -> tuple[list[dict], int | None]:
    """
    1 試合を実行し、指定タイプのエントリのリストと勝者を返す。

    choice_type="energy_attach" : エネルギー付与の選択ログを収集
    choice_type="support_policy": サポートカード選択のログを収集

    frame 記録時に即エンコードするため deepcopy は不要。
    returns: (rows, winner)  rows = [{state, action_id, win, ...}, ...]
    """
    from game import run_game_auto, setup_game
    from game.encoders import encode_state_v2 as encode_state_opening

    # フレームごとに (p0ベクトル, p1ベクトル, p0残サイド, p1残サイド, turn_count) を保存
    frame_data: list[tuple] = []

    def record_frame(s) -> None:
        frame_data.append((
            encode_state_opening(s, 0),
            encode_state_opening(s, 1),
            len(s.players[0].prize_pile),
            len(s.players[1].prize_pile),
            int(s.turn_count),
            _compute_action_potential(s, 0),   # idx 5: P0 行動可能性
            _compute_action_potential(s, 1),   # idx 6: P1 行動可能性
        ))

    state = setup_game(
        deck0=deck0,
        deck1=deck1,
        record_frame_fn=record_frame,
        use_attack_minimax=True,
        use_energy_attack_lookahead=True,
        energy_policy_epsilon=epsilon if choice_type in ("energy_attach", "joint") else 0.0,
        energy_policy_epsilon_end=epsilon if choice_type in ("energy_attach", "joint") else 0.0,
        q_energy_attach_model_path=q_model_path,
        support_epsilon=support_epsilon if choice_type in ("support_policy", "joint") else 0.0,
        support_epsilon_end=support_epsilon if choice_type in ("support_policy", "joint") else 0.0,
        q_support_model_path=q_support_model_path,
    )

    winner = run_game_auto(state)
    if winner is None:
        return [], None

    choice_log = getattr(state, "choice_log", []) or []
    rows: list[dict] = []

    # turn_count → 最初のフレームインデックス のマップ（prize_delta 計算用）
    turn_to_first_frame: dict[int, int] = {}
    for fi, fd in enumerate(frame_data):
        tc = fd[4]
        if tc not in turn_to_first_frame:
            turn_to_first_frame[tc] = fi

    def _compute_prize_delta(frame_idx: int, player: int) -> tuple[list, int, float]:
        """(state_vec, prize_delta, action_potential_next) を返す共通ロジック。
        action_potential_next: 次ターン開始フレームの行動可能性スコア（0〜3）。
        """
        fd_now = frame_data[frame_idx]
        state_vec = fd_now[player]
        prizes_p0_now = fd_now[2]
        prizes_p1_now = fd_now[3]
        turn_now = fd_now[4]

        next_own_turn = turn_now + 2
        fi_next = turn_to_first_frame.get(next_own_turn)
        if fi_next is not None:
            fd_next = frame_data[fi_next]
            prizes_p0_next = fd_next[2]
            prizes_p1_next = fd_next[3]
            ap_next = float(fd_next[5 + player]) if len(fd_next) > 5 else 0.0
        else:
            prizes_p0_next = 0 if winner == 0 else prizes_p0_now
            prizes_p1_next = 0 if winner == 1 else prizes_p1_now
            ap_next = 0.0

        p0_took = prizes_p0_now - prizes_p0_next
        p1_took = prizes_p1_now - prizes_p1_next
        prize_delta = (p0_took - p1_took) if player == 0 else (p1_took - p0_took)
        return state_vec, int(prize_delta), ap_next

    if choice_type == "energy_attach":
        for entry in choice_log:
            if entry.get("type") != "energy_attach":
                continue
            frame_idx = entry.get("frame_index")
            if frame_idx is None or frame_idx < 0 or frame_idx >= len(frame_data):
                continue
            action_id = entry.get("energy_policy_action_id")
            if action_id is None:
                continue

            player = int(entry.get("player", 0))
            state_vec, prize_delta, ap_next = _compute_prize_delta(frame_idx, player)

            # エネ貼り後に攻撃可能になったか（即時効果）
            can_attack_after = int(bool(entry.get("can_attack_after", False)))

            rows.append({
                "state": state_vec,
                "action_id": int(action_id),
                "win": 1 if winner == player else 0,
                "prize_delta": prize_delta,
                "action_potential": ap_next,
                "can_attack_after": can_attack_after,
                "player": player,
                "type": "energy_attach",
                "turn": int(entry.get("turn", 0)),
                "battle_id": f"sp_{game_idx}",
                "epsilon_greedy": bool(entry.get("energy_policy_epsilon_greedy", False)),
            })

    elif choice_type == "support_policy":
        for entry in choice_log:
            if entry.get("type") != "support_policy":
                continue
            frame_idx = entry.get("frame_index")
            if frame_idx is None or frame_idx < 0 or frame_idx >= len(frame_data):
                continue
            action_id = entry.get("support_policy_action_id")
            if action_id is None:
                continue

            player = int(entry.get("player", 0))
            state_vec, prize_delta, ap_next = _compute_prize_delta(frame_idx, player)

            rows.append({
                "state": state_vec,
                "action_id": int(action_id),
                "win": 1 if winner == player else 0,
                "prize_delta": prize_delta,
                "action_potential": ap_next,
                "player": player,
                "type": "support_policy",
                "turn": int(entry.get("turn", 0)),
                "battle_id": f"sp_{game_idx}",
                "epsilon_greedy": bool(entry.get("support_policy_epsilon_greedy", False)),
            })

    elif choice_type == "joint":
        # サポート + エネルギーの結合行動を同一ターンからペアリング
        # (battle_id, player, turn) でグループ化
        from collections import defaultdict
        by_turn: dict[tuple, dict] = defaultdict(dict)
        for entry in choice_log:
            etype = entry.get("type")
            if etype not in ("support_policy", "energy_attach"):
                continue
            key = (int(entry.get("player", 0)), int(entry.get("turn", 0)))
            by_turn[key][etype] = entry

        for (player, turn), entries in by_turn.items():
            sup_entry = entries.get("support_policy")
            ene_entry = entries.get("energy_attach")
            if not ene_entry:
                continue  # エネ付与なしのターンはスキップ

            # state はサポート前のフレーム（判断時点）
            # サポートがあればそのフレーム、なければエネのフレーム
            frame_idx = (sup_entry or ene_entry).get("frame_index")
            if frame_idx is None or frame_idx < 0 or frame_idx >= len(frame_data):
                continue

            support_action_id = int(sup_entry.get("support_policy_action_id", 0)) if sup_entry else 0
            energy_action_id = int(ene_entry.get("energy_policy_action_id", 0))
            can_attack_after = int(bool(ene_entry.get("can_attack_after", False)))

            state_vec, prize_delta, ap_next = _compute_prize_delta(frame_idx, player)

            rows.append({
                "state": state_vec,
                "support_action_id": support_action_id,
                "energy_action_id": energy_action_id,
                "win": 1 if winner == player else 0,
                "prize_delta": prize_delta,
                "action_potential": ap_next,
                "can_attack_after": can_attack_after,
                "player": player,
                "type": "joint",
                "turn": turn,
                "battle_id": f"sp_{game_idx}",
            })

    # state_value: 全フレームの状態+勝敗を出力（V(s)学習用）
    if choice_type == "state_value":
        for fi, fd in enumerate(frame_data):
            turn = fd[4]
            for player in (0, 1):
                state_vec = fd[player]
                win = 1 if winner == player else 0
                # prize_delta（2ターン先との差）
                _, prize_delta, ap_next = _compute_prize_delta(fi, player)
                # はどうづき使用フラグ（事故試合判定用）
                hadouzuki_used = any(
                    e.get("type") == "attack" and e.get("attack_name") == "はどうづき" and int(e.get("player", -1)) == player
                    for e in choice_log
                )
                rows.append({
                    "state": state_vec,
                    "win": win,
                    "prize_delta": prize_delta,
                    "action_potential": ap_next,
                    "hadouzuki_used": int(hadouzuki_used),
                    "player": player,
                    "type": "state_value",
                    "turn": turn,
                    "battle_id": f"sp_{game_idx}",
                })

    # support_advantage: サポート使用ログから Advantage学習用データを生成
    # card_id → KNOWN_SUPPORT_IDS の action_id に変換
    if choice_type == "support_advantage":
        from game.support_policy import KNOWN_SUPPORT_IDS
        _sid_to_action = {sid: i + 1 for i, sid in enumerate(KNOWN_SUPPORT_IDS)}

        for entry in choice_log:
            etype = entry.get("type", "")
            if etype not in ("support", "support_policy"):
                continue
            frame_idx = entry.get("frame_index")
            if frame_idx is None or frame_idx < 0 or frame_idx >= len(frame_data):
                continue
            card_id = entry.get("card_id", "")
            # support_policy タイプは action_id が直接記録されている
            if etype == "support_policy" and entry.get("support_policy_action_id") is not None:
                action_id = int(entry["support_policy_action_id"])
            else:
                action_id = _sid_to_action.get(card_id, 0)

            player = int(entry.get("player", 0))
            state_vec, prize_delta, ap_next = _compute_prize_delta(frame_idx, player)
            win = 1 if winner == player else 0

            # はどうづき使用フラグ（事故試合判定用）
            hadouzuki_used = any(
                e.get("type") == "attack" and e.get("attack_name") == "はどうづき" and int(e.get("player", -1)) == player
                for e in choice_log
            )

            rows.append({
                "state": state_vec,
                "action_id": int(action_id),
                "card_id": card_id,
                "win": win,
                "prize_delta": prize_delta,
                "action_potential": ap_next,
                "hadouzuki_used": int(hadouzuki_used),
                "player": player,
                "type": "support_advantage",
                "turn": int(entry.get("turn", 0)),
                "battle_id": f"sp_{game_idx}",
            })

    return rows, winner


def _worker(args_tuple: tuple) -> tuple[list[dict], int | None]:
    """multiprocessing.Pool 用ラッパー。"""
    deck0, deck1, epsilon, game_idx, q_model_path, choice_type, support_epsilon, q_support_model_path = args_tuple
    return _run_one_game(deck0, deck1, epsilon, game_idx, q_model_path, choice_type, support_epsilon, q_support_model_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-play でエネルギー付与データセットを生成")
    parser.add_argument("--n", type=int, default=25000, help="総試合数（既定 25000）")
    parser.add_argument(
        "--decks",
        type=str,
        default="5,5:5,6:6,6",
        help="デッキ組み合わせ（コロン区切り、例: 5,5:5,6:6,6）。均等に割り振る",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.1,
        help="エネルギー付与の ε-greedy 確率（既定 0.1 = 10%%）",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_REPO_ROOT / "datasets" / "selfplay_energy_attach.jsonl",
        help="出力 JSONL ファイル",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="既存ファイルに追記する（既定は上書き）",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="並列ワーカー数（既定: CPU 数）",
    )
    parser.add_argument(
        "--q-model",
        type=str,
        default=None,
        help="self-play に使う Q-value モデルパス（省略時はヒューリスティックのみ）",
    )
    parser.add_argument(
        "--choice-type",
        type=str,
        default="energy_attach",
        choices=["energy_attach", "support_policy", "joint", "state_value", "support_advantage"],
        help="収集する選択タイプ（既定: energy_attach）",
    )
    parser.add_argument(
        "--support-epsilon",
        type=float,
        default=0.2,
        help="サポートポリシーの ε-greedy 確率（--choice-type support_policy 時のみ有効、既定 0.2）",
    )
    parser.add_argument(
        "--q-support-model",
        type=str,
        default=None,
        help="サポート Q-model パス（--choice-type support_policy 時に使用）",
    )
    args = parser.parse_args()

    # デッキ組み合わせをパース
    combos: list[tuple[int, int]] = []
    for part in args.decks.split(":"):
        a, b = part.strip().split(",")
        combos.append((int(a.strip()), int(b.strip())))

    n = args.n
    epsilon = args.epsilon
    out_path = args.out
    workers = min(args.workers, n)
    q_model_path: str | None = args.q_model
    choice_type: str = args.choice_type
    support_epsilon: float = args.support_epsilon
    q_support_model_path: str | None = args.q_support_model
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if args.append else "w"
    print(f"self-play 開始: {n} 試合 / choice-type={choice_type} / デッキ={args.decks} / workers={workers}")
    if choice_type == "energy_attach":
        print(f"  energy ε={epsilon}  Q-model: {q_model_path or '(なし=ヒューリスティック)'}")
    else:
        print(f"  support ε={support_epsilon}  Q-support: {q_support_model_path or '(なし)'}")
    print(f"出力: {out_path}  (モード: {'追記' if args.append else '上書き'})")
    print()

    # ジョブリスト生成
    jobs = [
        (combos[i % len(combos)][0], combos[i % len(combos)][1], epsilon, i, q_model_path, choice_type, support_epsilon, q_support_model_path)
        for i in range(n)
    ]

    start = time.time()
    total_rows = 0
    total_wins = [0, 0]
    skipped = 0
    done = 0

    with out_path.open(mode, encoding="utf-8") as f_out:
        if workers == 1:
            # シングルスレッド
            for job in jobs:
                rows, winner = _worker(job)
                if winner is None:
                    skipped += 1
                else:
                    total_wins[winner] += 1
                    for row in rows:
                        f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    total_rows += len(rows)
                done += 1
                if done % 1000 == 0 or done == n:
                    elapsed = time.time() - start
                    ms = elapsed / done * 1000
                    eta = (n - done) * ms / 1000
                    print(f"[{done:>6}/{n}] {ms:.1f}ms/試合  rows={total_rows:,}  ETA {eta:.0f}s")
        else:
            # マルチプロセス
            with multiprocessing.Pool(processes=workers) as pool:
                for rows, winner in pool.imap_unordered(_worker, jobs, chunksize=10):
                    if winner is None:
                        skipped += 1
                    else:
                        total_wins[winner] += 1
                        for row in rows:
                            f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                        total_rows += len(rows)
                    done += 1
                    if done % 1000 == 0 or done == n:
                        elapsed = time.time() - start
                        ms = elapsed / done * 1000
                        eta = (n - done) * ms / 1000
                        print(f"[{done:>6}/{n}] {ms:.1f}ms/試合(実効)  rows={total_rows:,}  "
                              f"P0勝:{total_wins[0]}  P1勝:{total_wins[1]}  ETA {eta:.0f}s")

    elapsed = time.time() - start
    print()
    print(f"完了: {n - skipped} 試合 / {elapsed:.1f}秒 ({elapsed / (n or 1) * 1000:.1f}ms/試合)")
    print(f"rows: {total_rows:,}  skipped: {skipped}")
    print(f"出力: {out_path}")


if __name__ == "__main__":
    main()
