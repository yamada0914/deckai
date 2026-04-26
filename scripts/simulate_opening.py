"""
序盤特化データセット生成スクリプト。

ルカリオデッキの序盤「理想ムーブ」達成率を学習データとして書き出す。

成功条件:
  先行プレイヤー: 先行2ターン目（turn_count=2）でルカリオが攻撃した
  後攻プレイヤー: 後攻1ターン目（turn_count=1）でソルロックが攻撃した

出力形式（JSONL）:
  {"state": [...], "action_id": int, "success": 0|1, "player": 0|1,
   "type": "energy_attach", "turn": int, "role": "先行"|"後攻"}

使い方:
  python scripts/simulate_opening.py --n 5000
  python scripts/simulate_opening.py --n 10000 --seed 42 --out datasets/opening.jsonl
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from game import _check_game_end, end_turn, run_turn_auto, setup_game, start_turn
from game.encoders import encode_state_opening, energy_attach_action_id_from_target_card_id

# 先行で何ターン目（turn_count）に理想攻撃を期待するか
_LUCARIO_TARGET_TURN = 2   # 先行2ターン目 = turn_count 2
# 後攻で何ターン目（turn_count）に理想攻撃を期待するか
_SOLROCK_TARGET_TURN = 1   # 後攻1ターン目 = turn_count 1
# 何ターン進めるか（先行T1=0、後攻T1=1、先行T2=2 の合計3ターン）
_SIM_TURNS = 3


def _is_lucario_attacker(card_id: str | None) -> bool:
    """card_id がルカリオ系かどうか。"""
    if not card_id:
        return False
    cid = card_id.lower()
    return "rukario" in cid


def _is_solrock_attacker(card_id: str | None) -> bool:
    """card_id がソルロックかどうか。"""
    if not card_id:
        return False
    cid = card_id.lower()
    return "sorurokku" in cid


def simulate_one_game(
    deck0: int,
    deck1: int,
    deck_code0: str | None,
    deck_code1: str | None,
    collect_second_player: bool,
    model_path: str | None = None,
    model_lambda: float = 1.0,
) -> list[dict]:
    """
    1試合の序盤（先行T1〜先行T2）を動かし、energy_attach 選択のデータを返す。

    collect_second_player=True のとき後攻プレイヤーのデータも収集する。
    """
    states: list = []

    def record_frame(s) -> None:
        states.append(copy.deepcopy(s))

    use_nn = model_path is not None
    state = setup_game(
        seed=None,
        log_fn=None,
        record_frame_fn=record_frame,
        deck0=deck0,
        deck1=deck1,
        deck_code0=deck_code0,
        deck_code1=deck_code1,
        use_attack_minimax=False,
        rules_only_policy=not use_nn,
        pi_energy_policy_model_path=model_path,
        pi_energy_policy_lambda=model_lambda,
        energy_policy_heuristic_scale_floor=0.0 if use_nn else 0.3,
    )

    first_player = state.first_player
    second_player = 1 - first_player

    # 先行プレイヤーがルカリオで攻撃したか（turn_count=2）
    lucario_attacked = False
    # 後攻プレイヤーがソルロックで攻撃したか（turn_count=1）
    solrock_attacked = False

    for _ in range(_SIM_TURNS):
        start_turn(state)
        if state.winner is not None:
            break
        run_turn_auto(state)

        # run_turn_auto 直後（end_turn 前）に今ターンの攻撃を確認する
        cur_tc = state.turn_count
        cp = state.current_player
        actor = state.this_turn_attack_actor_id

        if cur_tc == _LUCARIO_TARGET_TURN and cp == first_player:
            lucario_attacked = _is_lucario_attacker(actor)

        if cur_tc == _SOLROCK_TARGET_TURN and cp == second_player:
            solrock_attacked = _is_solrock_attacker(actor)

        if _check_game_end(state):
            break
        end_turn(state)

    # choice_log から energy_attach を抽出してラベルを付ける
    rows: list[dict] = []
    for e in (state.choice_log or []):
        if e.get("type") != "energy_attach":
            continue

        player = int(e.get("player", 0))
        turn = int(e.get("turn", 0))

        # 先行プレイヤー: ターン 0（先行T1）と 2（先行T2）が対象
        if player == first_player:
            if turn not in (0, _LUCARIO_TARGET_TURN):
                continue
            success = 1 if lucario_attacked else 0
            role = "先行"
        # 後攻プレイヤー: ターン 1（後攻T1）が対象（collect_second_player=True のとき）
        elif player == second_player and collect_second_player:
            if turn != _SOLROCK_TARGET_TURN:
                continue
            success = 1 if solrock_attacked else 0
            role = "後攻"
        else:
            continue

        frame_index = e.get("frame_index")
        if frame_index is None or frame_index < 0 or frame_index >= len(states):
            continue
        s_before = states[frame_index]

        target_card_id = e.get("card_id", "") or ""
        action_id = energy_attach_action_id_from_target_card_id(s_before, player, target_card_id)
        if action_id is None:
            continue

        rows.append({
            "state": encode_state_opening(s_before, player),
            "action_id": int(action_id),
            "success": success,
            "player": player,
            "type": "energy_attach",
            "turn": turn,
            "role": role,
        })

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="序盤特化データセットを生成する")
    parser.add_argument("--n", type=int, default=5000, help="シミュレーション試合数")
    parser.add_argument("--deck0", type=int, default=5, help="プレイヤー0のデッキ番号（デフォルト 5=アンフェアルカリオ）")
    parser.add_argument("--deck1", type=int, default=5, help="プレイヤー1のデッキ番号（デフォルト 5=アンフェアルカリオ）")
    parser.add_argument("--deck-code0", type=str, default=None, metavar="CODE", help="プレイヤー0のデッキコード（--deck0 より優先）")
    parser.add_argument("--deck-code1", type=str, default=None, metavar="CODE", help="プレイヤー1のデッキコード（--deck1 より優先）")
    parser.add_argument("--out", type=Path, default=_REPO_ROOT / "datasets" / "opening.jsonl", help="出力 JSONL パス")
    parser.add_argument("--seed", type=int, default=None, help="乱数シード（再現性用）")
    parser.add_argument("--model-path", type=str, default=None, metavar="PATH",
                        help="使用する NN モデル（.pt）。省略時は rules_only ベースライン")
    parser.add_argument("--model-lambda", type=float, default=1.0,
                        help="NN の影響度（大きいほど NN 優先）")
    parser.add_argument(
        "--second-player",
        action="store_true",
        default=True,
        help="後攻プレイヤー（ソルロック）のデータも収集する（デフォルト ON）",
    )
    parser.add_argument(
        "--no-second-player",
        action="store_false",
        dest="second_player",
        help="後攻プレイヤーのデータを収集しない",
    )
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    out_path: Path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_games = 0
    n_rows = 0
    n_success_first = 0
    n_total_first = 0
    n_success_second = 0
    n_total_second = 0

    from deck import get_deck_name
    print(f"デッキ0: {args.deck0}={get_deck_name(args.deck0)}  デッキ1: {args.deck1}={get_deck_name(args.deck1)}")
    print(f"モデル: {args.model_path or 'rules_only（NN なし）'}")
    print(f"試合数: {args.n}  出力: {out_path}")
    print()

    with out_path.open("w", encoding="utf-8") as f_out:
        for i in range(args.n):
            rows = simulate_one_game(
                deck0=args.deck0,
                deck1=args.deck1,
                deck_code0=args.deck_code0,
                deck_code1=args.deck_code1,
                collect_second_player=args.second_player,
                model_path=args.model_path,
                model_lambda=args.model_lambda,
            )
            for row in rows:
                f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_rows += 1
                if row["role"] == "先行":
                    n_total_first += 1
                    n_success_first += row["success"]
                else:
                    n_total_second += 1
                    n_success_second += row["success"]
            n_games += 1

            if (i + 1) % 500 == 0:
                rate1 = n_success_first / max(1, n_total_first)
                rate2 = n_success_second / max(1, n_total_second)
                print(f"  {i + 1}/{args.n} 試合  rows={n_rows}"
                      f"  先行成功率={rate1:.1%}({n_success_first}/{n_total_first})"
                      f"  後攻成功率={rate2:.1%}({n_success_second}/{n_total_second})")

    print()
    print(f"完了: {n_games} 試合  総行数: {n_rows}")
    print(f"先行ルカリオ攻撃成功率: {n_success_first/max(1,n_total_first):.1%}"
          f"  ({n_success_first}/{n_total_first})")
    if args.second_player:
        print(f"後攻ソルロック攻撃成功率: {n_success_second/max(1,n_total_second):.1%}"
              f"  ({n_success_second}/{n_total_second})")
    print(f"出力: {out_path}")


if __name__ == "__main__":
    main()
