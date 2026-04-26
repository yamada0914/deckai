"""
学習用ベースライン「ルールのみポリシー」。

`setup_game(..., rules_only_policy=True)` で **両者** ルールのみにする。
`rules_only_policy=False` かつ `rules_only_player0` / `rules_only_player1` で **片側だけ** ルールのみ（混合対戦）にもできる。

次を **そのプレイヤーについて** 無効化・単純化する（混合時はルールのみ側のみ）:

- JSON の `GameWeights`（空の重みに置き換え）
- エネルギー付与の π / Q モデル
- 攻撃の π / value モデル
- 攻撃選択の minimax（`use_attack_minimax=False`）
- `deck_strategies.get_fetch_bonus_for_card`（デッキ固有の取得ボーナス）
- エネ貼りの「コード内固定ヒューリスティック＋学習重み」合成スコア →
  **付与後の最大有効ダメージが最大**の候補のみ（同点はバトル場を優先）

残るもの: ゲームルール上の合法手生成（例: マクノシタ エネ 0 は手張りしない等）、
トレーナー・ボール・攻撃の既存ルールベース処理の大部分。

1 から学習し直すときの教師データ生成に使う。
"""

from __future__ import annotations


def pick_energy_attach_candidate(
    candidates: list[tuple],
) -> tuple:
    """
    エネ貼り候補から 1 つを選ぶ。

    Parameters
    ----------
    candidates :
        (bench_index_or_None, max_effective_damage_if_attach) のリスト。
        bench_index_or_None が None ならバトル場。

    Returns
    -------
    選んだタプルそのもの。
    """
    if not candidates:
        raise ValueError("candidates が空です")
    # 有効ダメージ降順、同点はバトル場（target is None）を優先
    return max(candidates, key=lambda it: (it[1], 1 if it[0] is None else 0))
