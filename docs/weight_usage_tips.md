# 重みをうまく利用するためのコツ

## 学習のやり方（手順）

1. **対戦を記録する**  
   各試合で「どの選択をしたか」が `choice_log` に残るので、その試合結果と一緒に pkl に保存する。
   - **一括でたくさん回す**: `python scripts/run_training_games.py --matchup all --n 500`  
     → `battles/train_weights/` 配下に `train_5v6_0000`, `train_5v6_0001`, … のように `battle_states.pkl` が溜まる。
   - **1 試合だけ手動で**: `python scripts/record_game.py`  
     → `battles/<日時>/battle_states.pkl` ができる。

2. **記録した pkl から重みを計算する**  
   `train_weights.py` が「選択ごとの盤面評価」を集計し、評価が高い選択ほど正の重みになるよう変換する。**現在は技選択 (attack) のみ学習**する（`--all-choices` で他も学習可能）。
   ```bash
   python scripts/train_weights.py --battles-dir battles/train_weights --output weights/weights.json --scale 10
   ```
   - 既存の重みに上乗せしたい: `--merge weights/weights.json` を付ける。
   - 出現回数が少ない選択を重みから外したい: `--min-samples 5` など。

3. **学習した重みでプレイする**  
   - record_game: `python scripts/record_game.py --weights weights/weights.json`
   - シミュレーション: `scripts/simulate.py` や `compare_weighted_vs_unweighted.py` で `--weights` を指定。

---

目的に応じて二つの使い方があります。

- **汎用的な定石** … デッキが増えても共通で使える「どの対戦でも効く選択の傾向」を作る（推奨）。
- **特定対戦で強くする** … あるデッキ・あるマッチアップだけを強くしたいときに使う。

---

# 汎用的な定石を作りたい場合（デッキが 10 個などに増えても使う）

特定デッキに寄せず、「多くの状況で勝ちにつながる選択」を重みにしたいときは、**いろいろなデッキ組み合わせで学習**する。

## 方針

- **多くの組み合わせで記録する**  
  例: いまは 5v6・5v5・6v6 の 3 パターン。将来デッキが 10 個なら「全ペア」や「ランダムに選んだペア」をたくさん回す。  
  → 特定マッチアップに偏らず、「勝った試合でよく選ばれていた手」が重みに乗る。
- **スケールはやや控えめに**  
  `--scale 10〜15` 程度。強くしすぎると「ある対戦でだけ有効だった手」が効きすぎる。
- **min-samples で定石だけ残す**  
  `--min-samples 5` や `10` にすると、**多くの試合で繰り返し選ばれた選択**だけが重みになる。  
  → レアな状況限定の手は重みに乗りにくく、汎用的な定石に近づく。
- **試合数は多めに**  
  組み合わせが多いぶん、各選択の出現回数を稼ぐため、総試合数は 2000〜5000 回以上あると安心。

## コマンド例（いまの 3 組み合わせで定石を作る）

```bash
# 5v6・5v5・6v6 をすべて使って学習（既定の all）。スケール控えめ・min-samples でノイズ削減
python scripts/run_training_games.py --matchup all --n 500 --scale 12 --min-samples 5 --weights-out weights/weights.json
```

将来デッキが 10 個になったら、`run_training_games.py` で「全ペア」や「ランダムペアを N 回」を回し、同じように `--matchup all` 相当で学習すれば、定石的な重みを更新できる。

## 定石の評価のしかた

- 特定デッキに絞らず、**複数デッキで重みあり vs 重みなし**を試す。  
  例: デッキ 5・6・7 でそれぞれ 500 回ずつ `compare_weighted_vs_unweighted` を回し、重みあり側の勝率がおおむね 50% を上回るかを見る。
- 差は数％程度でも、「いろいろな対戦で一貫してプラス」なら定石として機能していると考えてよい。

---

# 重みありが 50% に届かない・効いている感じがしないとき

## 想定される理由

1. **試合数のばらつき**  
   100 試合だと標準誤差は約 5% 程度。45% は「本当は 50%」の範囲内のこともある。**300〜500 試合**で見ると傾向がはっきりする。

2. **学習と評価のマッチアップが違う**  
   `--matchup all`（5v5・5v6・6v6 混在）で学習すると、コライドン同士ミラー（5v5）だけの評価では「対ミライドン用」の選択が混ざり、ミラーで効きにくい。  
   → **評価したい対戦（例: 5v5）だけ**で記録・学習し直すと差が出やすい。

3. **技選択は minimax が主で重みは補助**  
   攻撃は「きぜつ優先 → 有効ダメージ」の固定ロジックが強く、重みはその上に足しているだけ。エネルギー付与・サポート順・にげる先などは重みで決まるが、**試合の帰結を大きく左右するのは技選択**なので、重みの効きは限定的に見える。

4. **スケールが小さい**  
   `--scale 10` だと、有効ダメージ（数十〜数百）に比べて重みの差（±10 程度）は小さい。技選択で「重みでひっくり返る」ことは少ない。  
   → スケールを **20〜30** にすると、学習した傾向が強く出る（過学習リスクは増える）。

5. **ノイズや過学習**  
   サンプルが少ない選択に極端な重みが付くと、悪手を選びやすくなる。`--min-samples 5` や `--smooth-samples 20` で抑える。

## 試すとよいこと

| 目的 | やること |
|------|----------|
| 差をはっきり見たい | `--n 300` や `500` で再計測 |
| コライドン同士で効かせたい | `--matchup 5v5 --n 1000` で記録し直し、その pkl だけで学習 |
| 重みを強く効かせたい | 学習時に `--scale 25` などに上げる |
| 定石だけにしたい | `--min-samples 5`、`--smooth-samples 20` で学習 |

---

# 重みをもっと実感したいとき

## 1. 比較で「重みの効き」を見る

- **`--no-minimax`** を付けると技選択も重みで決まり、差が出やすい。  
  `python scripts/compare_weighted_vs_unweighted.py --deck 5 --n 500 --no-minimax`
- **`--weight-scale 2`** で重みを 2 倍にして比較すると、効きの強さを感じやすい。  
  （学習し直さず、読み込み時に倍率をかけるだけ）

## 2. 重みの倍率を変える（再学習なし）

**`--weight-scale X`** で、同じ JSON を「X 倍」して使える。

- 比較: `python scripts/compare_weighted_vs_unweighted.py --deck 5 --n 300 --weight-scale 2`
- 記録: `python scripts/record_game.py --weights weights/weights.json --weight-scale 1.5`
- シミュレーション: `python scripts/simulate.py 100 --weights weights/weights.json --weight-scale 2`

1.5 や 2 にすると選択が重みに引っ張られやすくなり、実感しやすい。やりすぎると過学習気味の重みが効きすぎるので、様子を見て調整。

## 3. 学習時に強くする

`train_weights.py` で **`--scale 25`** などにすると、最初から大きい重みが付く。  
`--weight-scale` と組み合わせる場合は、どちらか一方を強くする程度にするとよい。

---

# 特定の対戦で強くしたい場合

## 1. 「評価したい対戦」と同じ組み合わせで学習する

**いま**: 5v6・5v5・6v6 を混ぜて学習している  
→ コライドン同士のミラーでは、対ミライドン用の選択が混ざり、効果が薄くなりやすい。

**おすすめ**: 重みを効かせたい対戦だけを学習する。

- **コライドン同士**で重みを効かせたい → **5v5 だけ**で記録・学習する。
- コライドン vs ミライドンで効かせたい → **5v6 だけ**で学習する。

```bash
# コライドン同士 1000 回だけ記録して学習（別フォルダに保存）
python scripts/run_training_games.py --matchup 5v5 --n 1000 --out-dir battles/train_5v5 --weights-out weights/weights_5v5.json
```

---

## 2. 重みの強さ（スケール）を大きくする

**いま**: `--scale 10`（勝率 1 → +10、0 → -10）。有效ダメージは数十〜数百なので、重みが選択をひっくり返すことは少ない。

**おすすめ**: スケールを 20〜50 にして、学習した傾向を強く効かせる。

```bash
python scripts/train_weights.py --battles-dir battles/train_5v5 --output weights/weights.json --scale 30
```

---

## 3. ノイズを減らす（出現回数が少ない選択は重みにしない）

**いま**: 1 回でも出た選択に重みを付けている。試合数が少ないと勝率がぶれやすい。

**おすすめ**: `--min-samples 5` や `10` で、ある程度選ばれた選択だけを重みにする。

```bash
python scripts/train_weights.py --battles-dir battles/train_5v5 --output weights/weights.json --scale 20 --min-samples 5
```

---

## 4. 試合数を増やす

選択ごとの勝率を安定させるため、**同じ組み合わせで 1000 回以上**あるとよい。  
（コライドン同士だけなら 1500 回でも 30 分程度）

---

## 5. 効果の計測は「公平モード」で

先行・後攻の偏りを除くため、`compare_weighted_vs_unweighted.py` は**公平モード（既定）**のまま使う。

```bash
python scripts/compare_weighted_vs_unweighted.py --deck 5 --n 1000 --weights weights/weights_5v5.json
```

---

## 推奨フロー例（コライドン同士で重みを効かせる）

1. **5v5 だけ 1000 回記録して学習**（スケールやや強め・min-samples でノイズ削減）
   ```bash
   python scripts/run_training_games.py --matchup 5v5 --n 1000 --out-dir battles/train_5v5 --weights-out weights/weights_5v5.json --scale 25 --min-samples 5
   ```
2. **重みあり vs 重みなしを 1000 回（公平モード）**
   ```bash
   python scripts/compare_weighted_vs_unweighted.py --deck 5 --n 1000 --weights weights/weights_5v5.json
   ```

同じデッキ同士のミラーはもともとドロー・乱数の影響が大きいので、差は数％程度になることもあります。それでも「同じ条件で学習・評価」にすると、重みの有無の差は出やすくなります。

---

# 重みが ±10 ばかりになる場合

学習式は「勝率 100% → +scale、0% → -scale」なので、**その選択で常に勝った／常に負けた**と出ると ±10 だけになる。サンプル数が少ないと 100%／0% になりやすい。

**対処**:
- **`--smooth-samples N`**（例: 20）を付けると、サンプル数が少ないキーは重みに `min(1, n/N)` を掛ける。  
  例: 5 回しか出てない選択は 5/20 = 0.25 倍になり、+10 → +2.5 のように中間の値になる。
- 試合数を増やすと、同じ選択が勝つ試合・負ける試合の両方に現れ、勝率が 0.6 などになり ±10 以外の重みが付く。

```bash
python scripts/train_weights.py --battles-dir battles/train_weights --output weights/weights.json --scale 10 --smooth-samples 20
```

---

# 共通で使えるオプションの目安

| 目的 | --matchup | --scale | --min-samples |
|------|-----------|---------|---------------|
| 汎用の定石（多くのデッキで使う） | all | 10〜15 | 5〜10 |
| 特定マッチアップを強くする | 5v5 など | 20〜30 | 5 |

---

# 重みまわり 調査・今後の候補リスト

## 現在学習しているもの

| 項目 | 説明 |
|------|------|
| 技選択 | どのワザを出すか（`w_attack`）。現在ここだけ学習。 |

## 将来用（実装・記録はあるが学習対象外）

エネルギー付与先・にげる・いれかえ・繰り出し・キャッチャー対象・サポート／グッズ順・進化・どうぐ・ハイパーボール捨て などはゲーム内で重みを参照するが、学習は `--all-choices` を付けたときのみ行う。

## 未実装・今後の候補

強化学習や報酬の細分化などは未対応。

## その他

- **デッキが 10 個になったとき**: `run_training_games.py` で全ペア or ランダムペアを回し、定石用重みを更新する拡張。
- **学習方法**: いまは「選択ごとの勝率」の線形変換のみ。強化学習や報酬の細分化は未対応。
