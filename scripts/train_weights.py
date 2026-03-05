"""
記録した対戦 pkl から choice_log と勝者を集計し、重みを更新して JSON で保存する。

  複数の battle_states.pkl を読み、選択ごとの勝率から重みを計算する。
  勝率 0.5 を 0 とし、勝率が高い選択ほど正の重みになる。

  例:
    python scripts/train_weights.py
    python scripts/train_weights.py --battles-dir battles --output weights/weights.json --scale 10
"""
import argparse
import pickle
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from game.weights import GameWeights, load_weights, save_weights


def _choice_key(entry: dict) -> str | None:
    """choice_log の 1 件から重みのキーを返す。attack のときは card_id|attack_name。"""
    t = entry.get("type")
    card_id = entry.get("card_id", "")
    if t == "attack":
        return f"{card_id}|{entry.get('attack_name', '')}"
    if t in ("energy_attach", "retreat", "swap", "promote", "catcher", "support", "goods", "evolve_onto", "tool_attach", "haipaboru_discard") and card_id:
        return card_id
    return None


def _aggregate_from_pkl(pkl_path: Path) -> tuple[dict[str, list[bool]], int]:
    """
    1 つの pkl から choice_log と winner を読み、キーごとに (勝ったか) のリストを返す。
    戻り値: (key -> [won, ...], 試合数 1)
    """
    with pkl_path.open("rb") as f:
        data = pickle.load(f)
    states = data.get("states", [])
    if not states:
        return ({}, 0)
    last = states[-1]
    winner = getattr(last, "winner", None)
    choice_log = getattr(last, "choice_log", [])
    if winner not in (0, 1):
        return ({}, 1)  # 引き分けなどは集計に使わない

    agg: dict[str, list[bool]] = {}
    for e in choice_log:
        k = _choice_key(e)
        if k is None:
            continue
        typ = e.get("type")
        if typ is None:
            continue
        won = e.get("player") == winner
        key_for_weight = (typ, k)  # (type, key) で区別
        sk = f"{typ}:{k}"
        if sk not in agg:
            agg[sk] = []
        agg[sk].append(won)
    return (agg, 1)


def _merge_aggregates(acc: dict[str, list[bool]], one: dict[str, list[bool]]) -> None:
    """one の集計を acc にマージする（in-place）。"""
    for k, wins in one.items():
        acc.setdefault(k, []).extend(wins)


def _aggregates_to_weights(
    agg: dict[str, list[bool]],
    scale: float = 10.0,
    min_samples: int = 1,
    smooth_samples: int = 0,
) -> GameWeights:
    """
    キー "type:key" ごとの [won] から GameWeights を構築する。
    weight = (win_rate - 0.5) * 2 * scale。勝率 0.5 で 0、1 で +scale、0 で -scale。
    smooth_samples > 0 のとき、サンプル数が少ないキーは重みを抑える（n/ smooth_samples を掛ける）。
    これで「数回しか出てないのに ±10 」になるのを防ぎ、中間の値が出やすくなる。
    """
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

    for full_key, wins in agg.items():
        n = len(wins)
        if n < min_samples:
            continue
        if ":" not in full_key:
            continue
        typ, key = full_key.split(":", 1)
        win_rate = sum(wins) / n
        weight = (win_rate - 0.5) * 2 * scale
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
        default=10.0,
        help="勝率を重みに変換するときのスケール（勝率 1 → +scale、0 → -scale）",
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
    args = parser.parse_args()

    battles_dir = args.battles_dir
    pkl_files = list(battles_dir.glob("*/battle_states.pkl"))
    if not pkl_files:
        print(f"対戦 pkl が見つかりません: {battles_dir}/*/battle_states.pkl")
        sys.exit(1)

    acc: dict[str, list[bool]] = {}
    game_count = 0
    for pkl_path in sorted(pkl_files):
        one, n = _aggregate_from_pkl(pkl_path)
        _merge_aggregates(acc, one)
        game_count += n

    weights = _aggregates_to_weights(
        acc,
        scale=args.scale,
        min_samples=args.min_samples,
        smooth_samples=args.smooth_samples,
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
    print(f"重みを保存しました: {args.output}（対戦数: {game_count}、キー数: "
          f"energy_attach={len(weights.w_energy_attach)} retreat={len(weights.w_retreat_target)} "
          f"swap={len(weights.w_swap_target)} promote={len(weights.w_promote)} attack={len(weights.w_attack)} "
          f"catcher={len(weights.w_catcher_target)} support={len(weights.w_support_use)} goods={len(weights.w_goods_use)} "
          f"evolve_onto={len(weights.w_evolve_onto)} tool_attach={len(weights.w_tool_attach)} haipaboru_discard={len(weights.w_haipaboru_discard)}）")


if __name__ == "__main__":
    main()
