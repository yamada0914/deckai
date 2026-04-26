"""
記録した対戦 pkl から choice_log の評価値 (eval) を集計し、重みを更新して JSON で保存する。

  複数の battle_states.pkl を読み、選択ごとの「選択直後の盤面評価」の平均から重みを計算する。
  全体平均を 0 とし、平均評価が高い選択ほど正の重みになる。eval がない古い pkl は集計対象外。

  既定では技選択 (attack) のみ学習する。--choice-type energy_attach でエネルギー付与先のみ、--all-choices で全選択を学習。

  昔の対戦 pkl（例: battles/train_belt）にも energy_attach の choice_log が入っているので、同じデータでエネルギー付与先の学習ができる。

  例:
    python scripts/train_weights.py
    python scripts/train_weights.py --battles-dir battles/train_belt --choice-type energy_attach --output weights/weights.json
    python scripts/train_weights.py --all-choices
"""
import argparse
import pickle
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from game.weights import GameWeights, load_weights, save_weights


def _choice_key(entry: dict, *, first_player: int | None = None) -> str | None:
    """choice_log の 1 件から重みのキーを返す。attack のときは card_id|attack_name|can_kill。
    energy_attach かつ first_player が判明している場合は "first|card_id" / "second|card_id" で先行/後攻を区別する。
    """
    t = entry.get("type")
    card_id = entry.get("card_id", "")
    if t == "attack":
        can_kill = int(bool(entry.get("can_kill", 0)))
        return f"{card_id}|{entry.get('attack_name', '')}|{can_kill}"
    if t == "energy_attach" and card_id:
        if first_player is not None:
            player = entry.get("player")
            prefix = "first" if player == first_player else "second"
            return f"{prefix}|{card_id}"
        return card_id
    if t in ("retreat", "swap", "promote", "catcher", "support", "goods", "evolve_onto", "tool_attach", "haipaboru_discard") and card_id:
        return card_id
    return None


def _aggregate_from_pkl(pkl_path: Path) -> tuple[dict[str, list[float]], int]:
    """
    1 つの pkl から choice_log を読み、キーごとに「選択直後の盤面評価 (eval)」のリストを返す。
    eval がないエントリは無視する。戻り値: (key -> [eval, ...], 試合数 1)
    """
    with pkl_path.open("rb") as f:
        data = pickle.load(f)
    states = data.get("states", [])
    if not states:
        return ({}, 0)
    last = states[-1]
    choice_log = getattr(last, "choice_log", [])

    first_player = getattr(last, "first_player", None)

    agg: dict[str, list[float]] = {}
    for e in choice_log:
        if "eval" not in e:
            continue
        k = _choice_key(e, first_player=first_player)
        if k is None:
            continue
        typ = e.get("type")
        if typ is None:
            continue
        sk = f"{typ}:{k}"
        if sk not in agg:
            agg[sk] = []
        agg[sk].append(float(e["eval"]))
    return (agg, 1)


def _merge_aggregates(acc: dict[str, list[float]], one: dict[str, list[float]]) -> None:
    """one の集計を acc にマージする（in-place）。"""
    for k, evals in one.items():
        acc.setdefault(k, []).extend(evals)


def _aggregates_to_weights(
    agg: dict[str, list[float]],
    scale: float = 0.5,
    min_samples: int = 1,
    smooth_samples: int = 0,
    attack_only: bool = True,
) -> GameWeights:
    """
    キー "type:key" ごとの [eval] から GameWeights を構築する。
    全体平均を baseline とし、weight = (mean_eval - baseline) * scale。
    smooth_samples > 0 のとき、サンプル数が少ないキーは重みを抑える（n / smooth_samples を掛ける）。
    attack_only が True のときは技選択 (attack) のみ重みを付け、他は空のままにする。
    """
    all_evals: list[float] = []
    for evals in agg.values():
        all_evals.extend(evals)
    baseline = sum(all_evals) / len(all_evals) if all_evals else 0.0

    w_energy_attach = {}
    w_retreat_target = {}
    w_swap_target = {}
    w_promote = {}
    w_attack = {}
    w_catcher_target = {}
    w_support_use = {}
    w_goods_use = {}
    w_evolve_onto = {}
    w_tool_attach = {}
    w_haipaboru_discard = {}

    for full_key, evals in agg.items():
        if attack_only and not full_key.startswith("attack:"):
            continue
        n = len(evals)
        if n < min_samples:
            continue
        if ":" not in full_key:
            continue
        typ, key = full_key.split(":", 1)
        mean_eval = sum(evals) / n
        weight = (mean_eval - baseline) * scale
        if smooth_samples > 0:
            weight *= min(1.0, n / smooth_samples)

        if typ == "energy_attach":
            w_energy_attach[key] = weight
        elif typ == "retreat":
            w_retreat_target[key] = weight
        elif typ == "swap":
            w_swap_target[key] = weight
        elif typ == "promote":
            w_promote[key] = weight
        elif typ == "attack":
            w_attack[key] = weight
        elif typ == "catcher":
            w_catcher_target[key] = weight
        elif typ == "support":
            w_support_use[key] = weight
        elif typ == "goods":
            w_goods_use[key] = weight
        elif typ == "evolve_onto":
            w_evolve_onto[key] = weight
        elif typ == "tool_attach":
            w_tool_attach[key] = weight
        elif typ == "haipaboru_discard":
            w_haipaboru_discard[key] = weight

    return GameWeights(
        w_energy_attach=w_energy_attach,
        w_retreat_target=w_retreat_target,
        w_swap_target=w_swap_target,
        w_promote=w_promote,
        w_attack=w_attack,
        w_catcher_target=w_catcher_target,
        w_support_use=w_support_use,
        w_goods_use=w_goods_use,
        w_evolve_onto=w_evolve_onto,
        w_tool_attach=w_tool_attach,
        w_haipaboru_discard=w_haipaboru_discard,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="対戦 pkl から重みを学習し JSON で保存する")
    parser.add_argument(
        "--battles-dir",
        type=Path,
        default=_REPO_ROOT / "battles",
        metavar="DIR",
        help="対戦 pkl が入ったディレクトリ（配下の */battle_states.pkl を検索）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_REPO_ROOT / "weights" / "weights.json",
        metavar="PATH",
        help="出力する重み JSON のパス",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.5,
        help="評価値の差を重みに変換するときのスケール（平均より 1 点高い選択 → +scale）",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=1,
        help="重みを付けるのに必要な最小選択回数",
    )
    parser.add_argument(
        "--smooth-samples",
        type=int,
        default=0,
        metavar="N",
        help="サンプル数が少ないキーで重みを抑える（重みに min(1, n/N) を掛ける）。0 で無効。例: 20 で 5 回しか出てない選択は 1/4 の重みになる",
    )
    parser.add_argument(
        "--merge",
        type=Path,
        default=None,
        metavar="PATH",
        help="既存の重み JSON を読み込み、学習した重みとマージする（同じキーは学習で上書き）",
    )
    parser.add_argument(
        "--all-choices",
        action="store_true",
        help="技選択以外も学習する（既定は技選択 attack のみ）",
    )
    parser.add_argument(
        "--choice-type",
        type=str,
        default=None,
        choices=["attack", "energy_attach", "retreat", "swap", "promote", "catcher", "support", "goods", "evolve_onto", "tool_attach", "haipaboru_discard"],
        metavar="TYPE",
        help="この選択タイプのみ学習する（例: energy_attach＝エネルギー付与先）。既存の pkl をそのまま使える",
    )
    args = parser.parse_args()

    battles_dir = args.battles_dir
    pkl_files = list(battles_dir.glob("*/battle_states.pkl"))
    if not pkl_files:
        print(f"対戦 pkl が見つかりません: {battles_dir}/*/battle_states.pkl")
        sys.exit(1)

    acc: dict[str, list[float]] = {}
    game_count = 0
    for pkl_path in sorted(pkl_files):
        one, n = _aggregate_from_pkl(pkl_path)
        _merge_aggregates(acc, one)
        game_count += n

    if args.all_choices:
        choice_filter = None
        attack_only = False
    elif args.choice_type:
        choice_filter = args.choice_type
        attack_only = False
    else:
        choice_filter = "attack"
        attack_only = True
    if choice_filter:
        acc = {k: v for k, v in acc.items() if k.startswith(choice_filter + ":")}

    total_evals = sum(len(v) for v in acc.values())
    if total_evals == 0:
        print("警告: 評価値 (eval) が 1 件もありません。新しく記録した対戦 pkl を使ってください。", file=sys.stderr)
        if choice_filter or attack_only:
            print("（--all-choices で全選択、--choice-type energy_attach でエネルギー付与先のみ集計）", file=sys.stderr)

    weights = _aggregates_to_weights(
        acc,
        scale=args.scale,
        min_samples=args.min_samples,
        smooth_samples=args.smooth_samples,
        attack_only=attack_only,
    )

    if args.merge and args.merge.is_file():
        base = load_weights(args.merge)
        weights = GameWeights(
            w_energy_attach={**base.w_energy_attach, **weights.w_energy_attach},
            w_retreat_target={**base.w_retreat_target, **weights.w_retreat_target},
            w_swap_target={**base.w_swap_target, **weights.w_swap_target},
            w_promote={**base.w_promote, **weights.w_promote},
            w_attack={**base.w_attack, **weights.w_attack},
            w_catcher_target={**base.w_catcher_target, **weights.w_catcher_target},
            w_support_use={**base.w_support_use, **weights.w_support_use},
            w_goods_use={**base.w_goods_use, **weights.w_goods_use},
            w_evolve_onto={**base.w_evolve_onto, **weights.w_evolve_onto},
            w_tool_attach={**base.w_tool_attach, **weights.w_tool_attach},
            w_haipaboru_discard={**base.w_haipaboru_discard, **weights.w_haipaboru_discard},
        )

    save_weights(weights, args.output)
    if choice_filter:
        key_count = len(getattr(weights, f"w_{choice_filter}", {}))
        print(f"重みを保存しました: {args.output}（{choice_filter} のみ学習、対戦数: {game_count}、キー数: {key_count}）")
    elif attack_only:
        print(f"重みを保存しました: {args.output}（技選択のみ学習、対戦数: {game_count}、attack キー数: {len(weights.w_attack)}）")
        print("全選択を学習する場合は --all-choices、エネルギー付与先のみは --choice-type energy_attach を付けてください。")
    else:
        print(f"重みを保存しました: {args.output}（対戦数: {game_count}、キー数: "
              f"energy_attach={len(weights.w_energy_attach)} retreat={len(weights.w_retreat_target)} "
              f"swap={len(weights.w_swap_target)} promote={len(weights.w_promote)} attack={len(weights.w_attack)} "
              f"catcher={len(weights.w_catcher_target)} support={len(weights.w_support_use)} goods={len(weights.w_goods_use)} "
              f"evolve_onto={len(weights.w_evolve_onto)} tool_attach={len(weights.w_tool_attach)} haipaboru_discard={len(weights.w_haipaboru_discard)}）")


if __name__ == "__main__":
    main()
