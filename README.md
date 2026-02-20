# deckAi

ポケモンカード風デッキ対戦のシミュレーションと、カード画像からのデータ取り込みを行うプロジェクトです。

## カード画像から data.py まで（read_cards → update_cards_from_json）

画像フォルダに置いたポケモンカードを読み取り、その結果で `card/data.py` を更新する流れです。

```
カード画像（フォルダ）
    ↓ read_cards.py（OpenAI Vision API で解析）
read_cards_result.json
    ↓ update_cards_from_json.py（マーカー内ブロックを置換）
card/data.py が更新される
```

### 1. 準備

- Python 3 と依存パッケージを入れる  
  `pip install -r requirements.txt`
- OpenAI API キーを用意し、`.env` に書く  
  `cp .env.example .env` のあと、`.env` の `OPENAI_API_KEY=` にキーを設定する。
- カード画像を入れるフォルダを用意する（例: `card_images`）。  
  対応形式: `.png`, `.jpg`, `.jpeg`, `.webp`

### 2. 画像を読み取る（read_cards.py）

```bash
# デフォルトで card_images を参照し、結果を read_cards_result.json に保存
python read_cards.py

# フォルダ・出力先を指定する場合
python read_cards.py --folder ./card_images -o read_cards_result.json
```

- 結果は **必ずファイルに保存** される（未指定時は `read_cards_result.json`）。
- **追加モード**: 既存の JSON があるときは、新規のカードだけが追加される。  
  同じ `_source_image`（ファイル名）が既にある画像は **解析しない**（API を呼ばない）。  
  同じ `id` が既にあるカードは上書きせずスキップ。同じ `id` で内容が違うときだけ「データの食い違い」を stderr に表示する。

### 3. data.py を更新する（update_cards_from_json.py）

**実行すると card/data.py の「JSON から生成」マーカーで囲まれたブロックが、read_cards_result.json の内容で置き換わります。** 手でコピーする必要はありません。

```bash
# data.py を更新する（既定）
python update_cards_from_json.py

# 更新せず、内容だけ確認する
python update_cards_from_json.py --dry-run

# data.py は触らず、生成コードだけファイルに出力する
python update_cards_from_json.py -o card/data_generated.py
```

- **更新される箇所**: カード定義（PokemonCard）、`CARD_ID_TO_NAME`、`_CARD_REGISTRY` のうち、data.py 内で「# ----- JSON から生成 -----」で囲まれた部分だけ。
- **こんらんなど**: 説明文に「こんらん」と「このポケモン」が含まれる技は、自動で `status_effect='confusion'`, `status_effect_target='self'` が付きます。

### まとめ

1. カード画像を `card_images` などに置く。  
2. `python read_cards.py` で `read_cards_result.json` を生成・追加する。  
3. `python update_cards_from_json.py` で `card/data.py` を更新する。  
4. デッキやシミュレーションでは、`card/data.py` のカード ID（例: `zupika-svd-041`）を参照する。
