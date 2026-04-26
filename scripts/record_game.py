"""
毎回 1 試合を実行し、ログと状態スナップショットを保存する。

対戦 ID でログ・pkl・動画を紐付け。ID 未指定時は日時（YYYYMMDD_HHMMSS）で自動採番。
  1 対戦ごとに battles/<id>/ フォルダに battle.log, battle_states.pkl を保存。
  python scripts/record_game.py                    # 日時で ID 自動
  python scripts/record_game.py --id my_match       # 対戦 ID を手動指定
  python scripts/make_video.py --battle-id <id>     # 動画生成
"""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import argparse
import copy
import pickle
from datetime import datetime

BATTLES_DIR = _REPO_ROOT / "battles"


def _run_and_record(
    seed: int | None = None,
    deck0: int = 5,
    deck1: int = 6,
    deck_code0: str | None = None,
    deck_code1: str | None = None,
    battle_id: str | None = None,
    log_path: Path | None = None,
    states_path: Path | None = None,
    weights_path: Path | None = None,
    weight_scale: float = 1.0,
    fast: bool = False,
    rules_only_policy: bool = False,
    rules_only_player0: bool = False,
    rules_only_player1: bool = False,
    nn_second_player_only: bool = False,
    use_energy_attack_lookahead: bool = False,
    state_value_model_path: str | None = None,
    state_value_lambda: float = 0.3,
    total_env_steps: int = 0,
    increment_total_env_steps_on_log: bool = True,
    pi_retreat_before_attack_lambda_schedule: str = "linear",
    policy_schedule_horizon_steps: int = 0,
    retreat_before_attack_epsilon: float = 0.0,
    retreat_before_attack_epsilon_end: float = 0.0,
    retreat_before_attack_epsilon_schedule: str = "linear",
    retreat_before_attack_epsilon_decay_k: float = 3.0,
    retreat_before_attack_heuristic_decay: float = 0.7,
    retreat_before_attack_heuristic_scale_floor: float = 0.3,
    retreat_before_attack_normalize_logits: bool = True,
    energy_policy_epsilon: float = 0.0,
    energy_policy_epsilon_end: float = 0.0,
    support_epsilon: float = 0.0,
    support_epsilon_end: float = 0.0,
    pi_energy_policy_model_path: str | None = None,
    pi_energy_policy_lambda: float = 0.5,
    pi_support_model_path: str | None = None,
    pi_support_lambda: float = 0.5,
) -> tuple[list, str]:
    """
    1 試合を実行し、ログを書きつつターン開始ごとの状態をリストに溜める。
    戻り値: 状態のリスト（pickle 保存も行う）
    """
    from game import (
        _check_game_end,
        end_turn,
        load_weights,
        run_turn_auto,
        setup_game,
        start_turn,
    )

    if seed is not None:
        import random
        random.seed(seed)

    bid = battle_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    battle_dir = BATTLES_DIR / bid
    log_path = log_path or battle_dir / "battle.log"
    states_path = states_path or battle_dir / "battle_states.pkl"

    states: list = []
    log_snapshots: list[list[str]] = []
    log_lines: list[str] = []

    def log_fn(msg: str) -> None:
        log_lines.append(msg)

    def record_frame(s) -> None:
        states.append(copy.deepcopy(s))
        log_snapshots.append(list(log_lines))

    log_lines.append(f"対戦 ID: {bid}")
    log_lines.append("")

    if rules_only_policy:
        weights = None
        use_mm = False
    else:
        weights = (
            load_weights(weights_path, scale=weight_scale)
            if weights_path and weights_path.is_file()
            else None
        )
        # 混合対戦ではフル側が minimax / 重みを使う（--fast でオフ可）
        use_mm = not fast
    state = setup_game(
        seed=seed,
        log_fn=log_fn,
        record_frame_fn=record_frame,
        deck0=deck0,
        deck1=deck1,
        deck_code0=deck_code0,
        deck_code1=deck_code1,
        weights=weights,
        use_attack_minimax=use_mm,
        use_energy_attack_lookahead=use_energy_attack_lookahead,
        rules_only_policy=rules_only_policy,
        rules_only_player0=rules_only_player0,
        rules_only_player1=rules_only_player1,
        nn_second_player_only=nn_second_player_only,
        state_value_model_path=state_value_model_path,
        state_value_lambda=state_value_lambda,
        total_env_steps=total_env_steps,
        increment_total_env_steps_on_log=increment_total_env_steps_on_log,
        pi_retreat_before_attack_lambda_schedule=pi_retreat_before_attack_lambda_schedule,
        policy_schedule_horizon_steps=policy_schedule_horizon_steps,
        retreat_before_attack_epsilon=retreat_before_attack_epsilon,
        retreat_before_attack_epsilon_end=retreat_before_attack_epsilon_end,
        retreat_before_attack_epsilon_schedule=retreat_before_attack_epsilon_schedule,
        retreat_before_attack_epsilon_decay_k=retreat_before_attack_epsilon_decay_k,
        retreat_before_attack_heuristic_decay=retreat_before_attack_heuristic_decay,
        retreat_before_attack_heuristic_scale_floor=retreat_before_attack_heuristic_scale_floor,
        retreat_before_attack_normalize_logits=retreat_before_attack_normalize_logits,
        energy_policy_epsilon=energy_policy_epsilon,
        energy_policy_epsilon_end=energy_policy_epsilon_end,
        support_epsilon=support_epsilon,
        support_epsilon_end=support_epsilon_end,
        pi_energy_policy_model_path=pi_energy_policy_model_path,
        pi_energy_policy_lambda=pi_energy_policy_lambda,
        pi_support_model_path=pi_support_model_path,
        pi_support_lambda=pi_support_lambda,
    )

    while True:
        start_turn(state)
        if state.winner is not None:
            state.log(f"========== ゲーム終了: {state.player_name(state.winner)} の勝ち（プレイヤー{state.winner}） ==========\n")
            break
        run_turn_auto(state)
        if _check_game_end(state):
            state.log(f"========== ゲーム終了: {state.player_name(state.winner)} の勝ち（プレイヤー{state.winner}） ==========\n")
            state._record_frame()
            break
        end_turn(state)
        if state.turn_count >= 200:
            from game import MAX_TURNS_SAFETY
            state.winner = 0 if (6 - len(state.players[0].prize_pile)) >= (6 - len(state.players[1].prize_pile)) else 1
            if state.log_fn:
                state.log(f"{MAX_TURNS_SAFETY} ターンで打ち切り（サイド取得で判定）\n")
            state.log(f"========== ゲーム終了: {state.player_name(state.winner)} の勝ち（プレイヤー{state.winner}） ==========\n")
            states.append(copy.deepcopy(state))
            log_snapshots.append(list(log_lines))
            break

    winner_name = state.player_name(state.winner) if state.winner is not None else "不明"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print(f"対戦 ID: {bid}  結果: {winner_name} の勝ち（プレイヤー{state.winner}）")
    print(f"ログを保存しました: {log_path}")

    for s in states:
        s.log_fn = None
        s.record_frame_fn = None
    states_path.parent.mkdir(parents=True, exist_ok=True)
    with open(states_path, "wb") as f:
        pickle.dump({"states": states, "log_snapshots": log_snapshots}, f)
    print(f"状態スナップショットを保存しました: {states_path} ({len(states)} フレーム)")
    print(f"動画生成: python scripts/make_video.py --battle-id {bid}")

    return states, bid


def main() -> None:
    parser = argparse.ArgumentParser(description="1 試合を実行しログと状態を保存する（動画用）")
    parser.add_argument("--id", "--battle-id", dest="battle_id", type=str, default=None, metavar="ID", help="対戦 ID（省略時は日時 YYYYMMDD_HHMMSS で自動）")
    parser.add_argument("--seed", type=int, default=None, help="乱数シード（指定すると同じ進行を再現可能）")
    parser.add_argument("--deck0", type=int, default=5, help="プレイヤー0 のデッキ番号（5=コライドン）。--deck-code0 指定時は無視")
    parser.add_argument("--deck1", type=int, default=6, help="プレイヤー1 のデッキ番号（6=ミライドン）。--deck-code1 指定時は無視")
    parser.add_argument("--deck-code0", type=str, default=None, metavar="CODE", help="プレイヤー0 のデッキコード（例: 8DcaYc-IR2SmJ-x8D8G8）")
    parser.add_argument("--deck-code1", type=str, default=None, metavar="CODE", help="プレイヤー1 のデッキコード")
    parser.add_argument("--log", type=Path, default=None, metavar="PATH", help="ログ出力先（省略時は battles/<ID>/battle.log）")
    parser.add_argument("--states", type=Path, default=None, metavar="PATH", help="状態 pickle 出力先（省略時は battles/<ID>/battle_states.pkl）")
    parser.add_argument("--weights", type=Path, default=None, metavar="PATH", help="学習した重み JSON（指定するとその重みで選択が補正される）")
    parser.add_argument("--weight-scale", type=float, default=1.0, metavar="X", help="重みを X 倍して使う（既定 1.0）")
    parser.add_argument("--fast", action="store_true", help="攻撃選択で minimax をオフにして短時間で記録（学習用）")
    parser.add_argument(
        "--rules-only",
        action="store_true",
        help="両プレイヤーともルールのみ（空の GameWeights・NN 無効・minimax オフなど）。片側のみは --rules-only-player0 / --rules-only-player1",
    )
    parser.add_argument(
        "--rules-only-player0",
        action="store_true",
        help="プレイヤー 0 だけルールのみ（1 はフル方策。--weights 可・--fast で minimax オフ）",
    )
    parser.add_argument(
        "--rules-only-player1",
        action="store_true",
        help="プレイヤー 1 だけルールのみ（0 はフル方策）",
    )
    parser.add_argument(
        "--nn-second-only",
        action="store_true",
        help="後攻プレイヤーのみ NN を使い、先行は rules_only 扱い（データ収集・推論効率化）",
    )
    parser.add_argument(
        "--no-lookahead",
        action="store_true",
        help="エネルギー付与の lookahead をオフにする（デフォルト OFF、--fast と組み合わせ推奨）",
    )
    parser.add_argument(
        "--energy-epsilon",
        type=float,
        default=0.0,
        metavar="P",
        help="エネルギー付与ポリシーの ε-greedy 探索確率（例: 0.3 でランダム混ぜデータ収集）",
    )
    parser.add_argument(
        "--energy-epsilon-end",
        type=float,
        default=0.0,
        metavar="P",
        help="エネルギー付与 ε の終端値（horizon 指定時に減衰）",
    )
    parser.add_argument(
        "--support-epsilon",
        type=float,
        default=0.0,
        metavar="P",
        help="サポートポリシーの ε-greedy 探索確率",
    )
    parser.add_argument(
        "--support-epsilon-end",
        type=float,
        default=0.0,
        metavar="P",
        help="サポート ε の終端値",
    )
    parser.add_argument(
        "--pi-energy-policy-model",
        type=str,
        default=None,
        metavar="PATH",
        help="学習済みエネルギー付与ポリシー NN モデル .pt（例: models/pi_energy_policy.pt）",
    )
    parser.add_argument(
        "--pi-energy-policy-lambda",
        type=float,
        default=0.5,
        metavar="L",
        help="エネルギーポリシー NN の混合係数（ヒューリスティック + L * NN）",
    )
    parser.add_argument(
        "--pi-support-model",
        type=str,
        default=None,
        metavar="PATH",
        help="学習済みサポートポリシー NN モデル .pt（例: models/pi_support.pt）",
    )
    parser.add_argument(
        "--pi-support-lambda",
        type=float,
        default=0.5,
        metavar="L",
        help="サポートポリシー NN の混合係数",
    )
    parser.add_argument("--state-value-model", type=str, default=None, metavar="PATH", help="V(s) モデル .pt のパス（minimax 末端評価を補正）")
    parser.add_argument("--state-value-lambda", type=float, default=0.3, metavar="A", help="V(s) の混合係数（0=evaluate_board のみ、1=V(s) のみ。既定 0.3）")
    parser.add_argument("--total-env-steps", type=int, default=0, help="学習 global_step を渡す（攻撃前にげの λ 用。既定 0）")
    parser.add_argument(
        "--no-increment-env-steps-on-log",
        action="store_true",
        help="_log_choice で total_env_steps を増やさない（RL で global_step を外から同期するとき）",
    )
    parser.add_argument(
        "--retreat-lambda-schedule",
        type=str,
        default="linear",
        choices=["linear", "sigmoid"],
        help="攻撃前にげの λ ウォームアップ形状",
    )
    parser.add_argument(
        "--policy-schedule-horizon-steps",
        type=int,
        default=0,
        metavar="N",
        help="total_env_steps に対するスケジュール長（0 で ε・ヒューリスティック減衰なし）",
    )
    parser.add_argument(
        "--retreat-epsilon",
        type=float,
        default=0.0,
        metavar="P",
        help="攻撃前にげの ε-greedy 開始確率",
    )
    parser.add_argument(
        "--retreat-epsilon-end",
        type=float,
        default=0.0,
        metavar="P",
        help="ε の終端（horizon 指定時に線形／指数で減衰）",
    )
    parser.add_argument(
        "--retreat-epsilon-schedule",
        type=str,
        default="linear",
        choices=["linear", "exp"],
        help="ε の減衰形状",
    )
    parser.add_argument("--retreat-epsilon-decay-k", type=float, default=3.0, help="ε が exp 減衰のときの k")
    parser.add_argument(
        "--retreat-heuristic-decay",
        type=float,
        default=0.7,
        help="ヒューリスティック係数 1 - progress*decay の decay（最大で 1 からこれだけ下げる）",
    )
    parser.add_argument(
        "--retreat-heuristic-scale-floor",
        type=float,
        default=0.3,
        help="ヒューリスティック係数の下限",
    )
    parser.add_argument(
        "--no-retreat-normalize-logits",
        action="store_true",
        help="攻撃前にげで h と NN logits の |·| 平均正規化をオフ（従来の h + λ*NN 合成）",
    )
    args = parser.parse_args()

    _run_and_record(
        seed=args.seed,
        deck0=args.deck0,
        deck1=args.deck1,
        deck_code0=args.deck_code0,
        deck_code1=args.deck_code1,
        battle_id=args.battle_id,
        log_path=args.log,
        states_path=args.states,
        weights_path=args.weights,
        weight_scale=args.weight_scale,
        fast=args.fast,
        rules_only_policy=bool(args.rules_only),
        rules_only_player0=bool(args.rules_only_player0),
        rules_only_player1=bool(args.rules_only_player1),
        nn_second_player_only=bool(args.nn_second_only),
        use_energy_attack_lookahead=not bool(args.no_lookahead),
        state_value_model_path=args.state_value_model,
        state_value_lambda=float(args.state_value_lambda),
        total_env_steps=int(args.total_env_steps),
        increment_total_env_steps_on_log=not bool(args.no_increment_env_steps_on_log),
        pi_retreat_before_attack_lambda_schedule=str(args.retreat_lambda_schedule),
        policy_schedule_horizon_steps=int(args.policy_schedule_horizon_steps),
        retreat_before_attack_epsilon=float(args.retreat_epsilon),
        retreat_before_attack_epsilon_end=float(args.retreat_epsilon_end),
        retreat_before_attack_epsilon_schedule=str(args.retreat_epsilon_schedule),
        retreat_before_attack_epsilon_decay_k=float(args.retreat_epsilon_decay_k),
        retreat_before_attack_heuristic_decay=float(args.retreat_heuristic_decay),
        retreat_before_attack_heuristic_scale_floor=float(args.retreat_heuristic_scale_floor),
        retreat_before_attack_normalize_logits=not bool(args.no_retreat_normalize_logits),
        energy_policy_epsilon=float(args.energy_epsilon),
        energy_policy_epsilon_end=float(args.energy_epsilon_end),
        support_epsilon=float(args.support_epsilon),
        support_epsilon_end=float(args.support_epsilon_end),
        pi_energy_policy_model_path=args.pi_energy_policy_model,
        pi_energy_policy_lambda=float(args.pi_energy_policy_lambda),
        pi_support_model_path=args.pi_support_model,
        pi_support_lambda=float(args.pi_support_lambda),
    )


if __name__ == "__main__":
    main()
