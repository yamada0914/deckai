# 中級者 AI 設計（評価関数 × 2 手読み）

固定ロジックを増やさず、「読み」と「価値判断」で中級者化する方針のメモ。

## 方針

- **やらないこと**: 固定ロジックの強化（複雑化・デバッグ困難・伸びしろが消える）
- **やること**: 評価関数の拡張と浅い minimax（depth=2）

## 実装済み

### 1. 盤面評価 `evaluate_board(state, for_player)`

`game/evaluate.py`。指定プレイヤー視点のスコアを返す。

- サイド: `+3 * (自分の取得数 - 相手の取得数)`
- 手札: `+0.5 * 手札枚数`
- エネルギー: `+0.8 * (バトル場＋ベンチの付与エネルギー合計)`
- 相手が次ターンで KO 可能: `-2`
- 自分が次ターンで KO 可能: `+2`

重み（`PRIZE_WEIGHT`, `HAND_WEIGHT` など）は定数で、実験で調整可能。

### 2. シミュレーション用コピー

`GameState.copy_for_simulation()` で deep copy し、`log_fn` と `record_frame_fn` を無効化。minimax で使用。

### 3. 攻撃選択の 2 手読み

`turn.py` の `_choose_best_attack_index_minimax`:

1. 合法な技インデックス一覧を取得（`get_legal_attack_indices`）
2. 各技について: 状態をコピー → その技で攻撃 → 相手ターン（`end_turn` → `start_turn` → `run_turn_auto`）→ 終局なら ±大スコア、否則 `evaluate_board(clone, me)`
3. スコア最大の技を採用。候補が 0 件 or フォールバック時は従来の `_choose_best_attack_index`（1 手の有効ダメージ＋KO ボーナス＋重み）を使用。

## 今後の拡張案

- 評価重みの学習・チューニング（既存の `weights` とは別の「盤面評価用重み」）
- 山札残り・進化ライン完成度などの項を評価に追加
- 「博士の研究は手札が弱いときだけ」など、期待値に基づくカード使用判断（現状は候補生成の固定順のまま）
