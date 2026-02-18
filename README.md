# deckAi

ポケモンカード風デッキ対戦のシミュレーションと、カード画像からのデータ取り込みを行うプロジェクトです。

## カード画像から data.py まで（read_cards → update_cards_from_json）

画像フォルダに置いたポケモンカードを読み取り、その結果で `card/data.py` を更新する流れです。

```
カード画像（フォルダ）
    ↓ read_cards.py（OpenAI Vision API で解析）
read_cards_result.json
    ↓ update_cards_from_json.py（Python 定義を生成・マージ）
card/data.py
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

```bash
# デフォルトで read_cards_result.json を読み、card/data.py を更新
python update_cards_from_json.py

# 入力 JSON と data.py を指定する場合
python update_cards_from_json.py --json read_cards_result.json --data card/data.py
```

- **JSON にあるカード**: 同じ `id` が既にあればその定義を置き換え、なければ `# 基本エネルギー` の直前に追加する。
- **JSON にないポケモン**（オタチ・オオタチなど）は既存の定義をそのまま残す。
- **エネルギー・アイテム**（BASIC_ENERGY, POTION 等）は変更しない。
- `CARD_ID_TO_NAME` と `_CARD_REGISTRY` に、JSON にあってレジストリに無い `id` を追加する。

### まとめ

1. カード画像を `card_images` などに置く。  
2. `python read_cards.py` で `read_cards_result.json` を生成・追加する。  
3. `python update_cards_from_json.py` で `card/data.py` を更新する。  
4. デッキやシミュレーションでは、`card/data.py` のカード ID（例: `zupika-svd-041`）を参照する。
