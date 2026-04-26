"""
同じデッキまたは異なるデッキで「重みあり」vs「重みなし」を N 回対戦する。

  先行・後攻の偏りを除くため、--fair（既定）で「重みあり側」の配置を半々で実行し、「重みあり側の勝率」を出す。
  学習に --deck-code を使った場合は、比較も同じ --deck-code を指定すること（デッキが違うと重みが効かない）。

  例:
    同じデッキで比較:
      python scripts/compare_weighted_vs_unweighted.py --deck 5 --n 1000
    デッキコードで学習した重みを同じデッキで比較:
      python scripts/compare_weighted_vs_unweighted.py --deck-code 8DcaYc-IR2SmJ-x8D8G8 --n 1000
    重みの差を出しやすくする（技選択は 1 手評価のみ）:
      python scripts/compare_weighted_vs_unweighted.py --deck-code 8DcaYc-IR2SmJ-x8D8G8 --n 1000 --no-minimax
"""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import argparse
from game import load_weights, setup_game, run_game_auto
from deck import get_deck_name


def main() -> None:
    parser = argparse.ArgumentParser(description="重みあり vs 重みなしを N 回対戦（同じデッキまたは異なるデッキ可）")
    parser.add_argument("--deck", type=int, default=5, help="両者とも使うデッキ番号（--deck-code / --deck0/--deck1 未指定時のみ）")
    parser.add_argument("--deck-code", type=str, default=None, metavar="CODE", help="学習に使ったデッキコードで両者とも対戦（学習と同じデッキで比較しないと重みが効かない）")
    parser.add_argument("--deck0", type=int, default=None, metavar="N", help="プレイヤー0 のデッキ番号。--deck1 と揃えると異なるデッキ対戦")
    parser.add_argument("--deck1", type=int, default=None, metavar="N", help="プレイヤー1 のデッキ番号。重みあり側のデッキにするなら --weights-for 1（既定）")
    parser.add_argument("--weights-for", type=int, default=1, choices=(0, 1), metavar="0|1", help="重みを渡すプレイヤー（0 または 1）。既定 1")
    parser.add_argument("--n", type=int, default=100, help="試合数（--fair 時は半分ずつに割り振る）")
    parser.add_argument("--weights", type=Path, default=_REPO_ROOT / "weights" / "weights.json", help="重み JSON")
    parser.add_argument("--q-energy-attach", type=Path, default=None, metavar="PATH", help="energy_attach の learned Q モデル (.pt)。指定すると付与先は top2 を Q で順位付けして選ぶ")
    parser.add_argument("--q-for-weighted-only", action="store_true", help="Q を重みあり側だけに渡す（重みあり+Q vs 重みなし・Q なしで効果を見る）")
    parser.add_argument("--q-lambda", type=float, default=0.3, metavar="X", help="rule + X*Q の X。Q が弱いときは 0.1〜0.2 が安定しやすい。既定 0.3")
    parser.add_argument("--pi-energy-attach", type=Path, default=None, metavar="PATH", help="energy_attach の learned policy モデル (.pt)。指定すると付与先は top2 を policy で順位付けして選ぶ")
    parser.add_argument("--pi-for-weighted-only", action="store_true", help="policy を重みあり側だけに渡す（重みあり+policy vs 重みなし・policy なしで効果を見る）")
    parser.add_argument("--pi-lambda", type=float, default=0.1, metavar="X", help="rule + X*log(pi) の X。既定 0.1")
    parser.add_argument("--value-attack", type=Path, default=None, metavar="PATH", help="attack の learned value モデル (.pt)。指定すると技を value でスコアリングして選ぶ")
    parser.add_argument("--value-attack-for-weighted-only", action="store_true", help="value を重みあり側だけに渡す（重みあり+value vs 重みなし・value なしで効果を見る）")
    parser.add_argument("--value-attack-lambda", type=float, default=0.1, metavar="X", help="rule + X*value の X。既定 0.1")
    parser.add_argument("--state-value-model", type=Path, default=None, metavar="PATH", help="V(s) モデル .pt のパス（minimax 末端評価を補正）")
    parser.add_argument("--state-value-for-weighted-only", action="store_true", help="V(s) を重みあり側だけに渡す（V(s) あり vs なし の A/B 比較用）")
    parser.add_argument("--state-value-lambda", type=float, default=0.3, metavar="A", help="V(s) の混合係数。既定 0.3")
    parser.add_argument("--no-fair", action="store_true", help="先行偏りを取らず、重みありを常にプレイヤー0 にする")
    parser.add_argument(
        "--no-minimax",
        action="store_true",
        help="技選択で minimax（2 手読み）を使わず 1 手評価のみにする。重みの差が出やすくなる",
    )
    parser.add_argument(
        "--weight-scale",
        type=float,
        default=1.0,
        metavar="X",
        help="重みを X 倍して使う（例: 2.0 で強く効かせる）。既定 1.0",
    )
    parser.add_argument("--disable-online-eval", action="store_true", help="online 評価（is_online）を両者で無効化する")
    parser.add_argument("--online-for-weighted-only", action="store_true", help="online 評価を重みあり側だけ有効化する（A/B 比較用）")
    args = parser.parse_args()

    deck_code = (args.deck_code or "").strip() or None
    if deck_code:
        deck0 = deck1 = 0
        deck_code0 = deck_code1 = deck_code
        name0 = name1 = f"デッキコード({deck_code[:12]}...)" if len(deck_code) > 12 else f"デッキコード({deck_code})"
    elif args.deck0 is not None and args.deck1 is not None:
        deck0, deck1 = args.deck0, args.deck1
        deck_code0 = deck_code1 = None
        name0 = get_deck_name(deck0)
        name1 = get_deck_name(deck1)
    else:
        deck0 = deck1 = args.deck
        deck_code0 = deck_code1 = None
        name0 = name1 = get_deck_name(deck0)
    n = args.n
    weights_for = args.weights_for
    weights = (
        load_weights(args.weights, scale=args.weight_scale)
        if args.weights.is_file()
        else None
    )
    fair = not args.no_fair
    use_minimax = not args.no_minimax

    if deck0 == deck1:
        if fair:
            print(f"対戦: 重みあり {name0} vs 重みなし {name0}、計 {n} 回（先行偏りを除くため重みの配置を半々で実行）")
        else:
            print(f"対戦: 重みあり {name0} (プレイヤー0) vs 重みなし {name0} (プレイヤー1)、{n} 回")
    else:
        weighted_name = name1 if weights_for == 1 else name0
        unweighted_name = name0 if weights_for == 1 else name1
        if fair:
            print(f"対戦: 重みなし {name0} vs 重みあり {name1}、計 {n} 回（先行偏りを除くため重みあり側の配置を半々で実行）")
        else:
            print(f"対戦: プレイヤー0 {name0}（重み{'あり' if weights_for == 0 else 'なし'}） vs プレイヤー1 {name1}（重み{'あり' if weights_for == 1 else 'なし'}）、{n} 回")
    if not use_minimax:
        print("（技選択は 1 手評価のみ・minimax なし）")
    if args.weight_scale != 1.0:
        print(f"（重みを {args.weight_scale} 倍で適用）")
    if args.disable_online_eval:
        print("（online 評価は両者で無効）")
    elif args.online_for_weighted_only:
        print("（online 評価は重みあり側のみ有効）")
    if args.q_energy_attach and args.q_for_weighted_only:
        print("（Q は重みあり側のみ使用）")
    if args.q_energy_attach and args.q_lambda != 0.3:
        print(f"（Q 補正: rule + {args.q_lambda}*Q）")
    if args.pi_energy_attach and args.pi_for_weighted_only:
        print("（policy は重みあり側のみ使用）")
    if args.pi_energy_attach and args.pi_lambda != 0.1:
        print(f"（policy 補正: rule + {args.pi_lambda}*log(pi)）")
    if args.value_attack and args.value_attack_for_weighted_only:
        print("（value は重みあり側のみ使用）")
    if args.value_attack and args.value_attack_lambda != 0.1:
        print(f"（value 補正: rule + {args.value_attack_lambda}*value）")
    print()

    q_path_str = str(args.q_energy_attach) if args.q_energy_attach and args.q_energy_attach.is_file() else None
    pi_path_str = str(args.pi_energy_attach) if args.pi_energy_attach and args.pi_energy_attach.is_file() else None
    value_attack_path_str = str(args.value_attack) if args.value_attack and args.value_attack.is_file() else None
    state_value_path_str = str(args.state_value_model) if args.state_value_model and args.state_value_model.is_file() else None

    def _setup(d0: int, d1: int, code0: str | None, code1: str | None, weights_p0, weights_p1):
        if args.q_for_weighted_only and q_path_str:
            q_global = None
            q_p0 = q_path_str if weights_p0 is not None else None
            q_p1 = q_path_str if weights_p1 is not None else None
        elif q_path_str:
            q_global = q_path_str
            q_p0 = q_p1 = None
        else:
            q_global = q_p0 = q_p1 = None

        if args.pi_for_weighted_only and pi_path_str:
            pi_global = None
            pi_p0 = pi_path_str if weights_p0 is not None else None
            pi_p1 = pi_path_str if weights_p1 is not None else None
        elif pi_path_str:
            pi_global = pi_path_str
            pi_p0 = pi_p1 = None
        else:
            pi_global = pi_p0 = pi_p1 = None

        if args.value_attack_for_weighted_only and value_attack_path_str:
            value_attack_global = None
            value_attack_p0 = value_attack_path_str if weights_p0 is not None else None
            value_attack_p1 = value_attack_path_str if weights_p1 is not None else None
        elif value_attack_path_str:
            value_attack_global = value_attack_path_str
            value_attack_p0 = value_attack_p1 = None
        else:
            value_attack_global = value_attack_p0 = value_attack_p1 = None

        if args.disable_online_eval:
            online_p0, online_p1 = False, False
        elif args.online_for_weighted_only:
            online_p0 = weights_p0 is not None
            online_p1 = weights_p1 is not None
        else:
            online_p0 = online_p1 = None
        return setup_game(
            seed=None,
            log_fn=None,
            deck0=d0,
            deck1=d1,
            deck_code0=code0,
            deck_code1=code1,
            weights_player0=weights_p0,
            weights_player1=weights_p1,
            use_attack_minimax=use_minimax,
            q_energy_attach_model_path=q_global,
            q_energy_attach_model_path_p0=q_p0,
            q_energy_attach_model_path_p1=q_p1,
            q_energy_attach_lambda=args.q_lambda,
            pi_energy_attach_model_path=pi_global,
            pi_energy_attach_model_path_p0=pi_p0,
            pi_energy_attach_model_path_p1=pi_p1,
            pi_energy_attach_lambda=args.pi_lambda,
            value_attack_model_path=value_attack_global,
            value_attack_model_path_p0=value_attack_p0,
            value_attack_model_path_p1=value_attack_p1,
            value_attack_lambda=args.value_attack_lambda,
            state_value_model_path=None if args.state_value_for_weighted_only else state_value_path_str,
            state_value_model_path_p0=state_value_path_str if (args.state_value_for_weighted_only and weights_p0 is not None) else None,
            state_value_model_path_p1=state_value_path_str if (args.state_value_for_weighted_only and weights_p1 is not None) else None,
            state_value_lambda=args.state_value_lambda,
            online_eval_enabled_p0=online_p0,
            online_eval_enabled_p1=online_p1,
        )

    wins_weighted_side = 0
    wins_p0 = 0
    wins_p1 = 0

    if fair and deck0 != deck1:
        half = n // 2
        for _ in range(half):
            state = _setup(deck0, deck1, deck_code0, deck_code1, weights if weights_for == 0 else None, weights if weights_for == 1 else None)
            winner = run_game_auto(state)
            if winner == weights_for:
                wins_weighted_side += 1
            wins_p0 += 1 if winner == 0 else 0
            wins_p1 += 1 if winner == 1 else 0
        for _ in range(n - half):
            state = _setup(deck1, deck0, deck_code1, deck_code0, weights if weights_for == 1 else None, weights if weights_for == 0 else None)
            winner = run_game_auto(state)
            if winner == (1 - weights_for):
                wins_weighted_side += 1
            wins_p0 += 1 if winner == 0 else 0
            wins_p1 += 1 if winner == 1 else 0
        print(f"重みあり側の勝ち: {wins_weighted_side} 勝 / {n} 試合 ({100 * wins_weighted_side / n:.1f}%)")
        print(f"（参考: プレイヤー0 勝ち {wins_p0} / プレイヤー1 勝ち {wins_p1}）")
    elif fair:
        half = n // 2
        for _ in range(half):
            state = _setup(deck0, deck1, deck_code0, deck_code1, weights, None)
            winner = run_game_auto(state)
            if winner == 0:
                wins_weighted_side += 1
            wins_p0 += 1 if winner == 0 else 0
            wins_p1 += 1 if winner == 1 else 0
        for _ in range(n - half):
            state = _setup(deck0, deck1, deck_code0, deck_code1, None, weights)
            winner = run_game_auto(state)
            if winner == 1:
                wins_weighted_side += 1
            wins_p0 += 1 if winner == 0 else 0
            wins_p1 += 1 if winner == 1 else 0
        print(f"重みあり側の勝ち: {wins_weighted_side} 勝 / {n} 試合 ({100 * wins_weighted_side / n:.1f}%)")
        print(f"（参考: プレイヤー0 勝ち {wins_p0} / プレイヤー1 勝ち {wins_p1}）")
    else:
        for _ in range(n):
            state = _setup(deck0, deck1, deck_code0, deck_code1, weights if weights_for == 0 else None, weights if weights_for == 1 else None)
            winner = run_game_auto(state)
            if winner == 0 and weights_for == 0:
                wins_weighted_side += 1
            elif winner == 1 and weights_for == 1:
                wins_weighted_side += 1
            wins_p0 += 1 if winner == 0 else 0
            wins_p1 += 1 if winner == 1 else 0
        print(f"重みあり側の勝ち: {wins_weighted_side} 勝 / {n} 試合 ({100 * wins_weighted_side / n:.1f}%)")
        print(f"（参考: プレイヤー0 勝ち {wins_p0} / プレイヤー1 勝ち {wins_p1}）")


if __name__ == "__main__":
    main()
