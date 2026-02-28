# deckAi

ポケモンカード風ルールに基づくデッキ対戦のシミュレーションと、カード画像からのデータ取り込みを行うプロジェクトです。

## 概要

- **カードデータ**: 画像を OpenAI Vision API で解析し、JSON → `card/data.py` へ反映するパイプライン
- **対戦シミュレーション**: 複数デッキで自動対戦を繰り返し、勝率などを集計
- **対戦記録・動画**: 1 試合を実行してログと状態を保存し、盤面画像をフレーム化して MP4 を生成

## ディレクトリ構成

```
deckAi/
├── README.md
├── requirements.txt          # pip 用依存
├── environment.yml           # Conda 用（任意）
├── card/                     # カード定義・マスタ
│   ├── __init__.py
│   ├── model.py              # 型（PokemonCard, Attack 等）
│   └── data.py               # 実データ（一部 JSON から生成）
├── deck.py                   # デッキレシピ・生成
├── game/                     # ゲーム状態・ターン進行・AI 手番（パッケージ）
│   ├── __init__.py
│   ├── state.py              # 状態・セットアップ・共通ヘルパー
│   ├── damage.py             # ダメージ計算
│   ├── evolution.py          # 進化
│   ├── trainers.py           # トレーナー（グッズ・サポート・どうぐ）
│   ├── attack.py             # 攻撃
│   └── turn.py               # ターン進行・run_turn_auto
├── board_render.py           # 盤面を 1 枚画像に描画
├── read_cards.py             # 画像 or デッキコード → JSON（ルートで実行）
├── update_cards_from_json.py # JSON → card/data.py 更新（ルートで実行）
├── scripts/                   # シミュレーション・記録・動画用 CLI
│   ├── simulate.py           # 対戦シミュレーション（勝率等）
│   ├── record_game.py        # 1 試合を記録（ログ + pkl）
│   └── make_video.py         # pkl から盤面フレーム → MP4
├── tests/                    # テスト（pytest）
│   ├── conftest.py           # 共通フィクスチャ
│   ├── test_fushiginaame.py  # ふしぎなアメの挙動
│   ├── test_rules_01_play_supplement.py   # 遊びかた補足に基づくルール
│   ├── test_rules_02_card_descriptions.py # カード説明文に基づくルール
│   └── test_rules_advanced.py            # 上級ルールに基づくルール
├── read_cards_data/          # カード画像マッピング用 JSON
│   ├── pokemon.json
│   ├── trainers.json
│   └── energy.json
├── read_cards_result.json    # read_cards の出力（画像解析結果）
├── registered_decks.json     # 登録デッキ一覧
├── battles/                  # 対戦記録（record_game で作成）
│   └── <id>/
│       ├── battle.log
│       ├── battle_states.pkl
│       └── (make_video で battle.mp4)
└── rules/                    # ルール参照（上級ルールガイド整理）
    ├── README.md
    ├── 01_play_supplement.md
    └── 02_card_descriptions.md
```

**実行の注意**: すべて **プロジェクトルート（deckAi/）をカレントにした状態**で実行してください。シミュレーション・記録・動画は `python scripts/simulate.py` のように `scripts/` 内のスクリプトを指定します。

## セットアップ

- **Python**: 3.10 以上を推奨
- **依存**: `pip install -r requirements.txt`  
  （または Conda: `conda env create -f environment.yml`）
- **OpenAI API**（read_cards のみ）: `.env` に `OPENAI_API_KEY=` を設定  
  （`.env.example` をコピーして編集）

## 使い方

### A. カード画像から data.py まで（read_cards → update_cards_from_json）

画像フォルダに置いたポケモンカードを読み取り、その結果で `card/data.py` を更新する流れです。

```
カード画像（フォルダ）
    ↓ read_cards.py（OpenAI Vision API で解析）
read_cards_result.json
    ↓ update_cards_from_json.py（マーカー内ブロックを置換）
card/data.py が更新される
```

#### 1. 準備

- 依存パッケージを入れる: `pip install -r requirements.txt`
- OpenAI API キーを `.env` に書く（`cp .env.example .env` のあと編集）
- カード画像用フォルダを用意（例: `card_images`）。対応形式: `.png`, `.jpg`, `.jpeg`, `.webp`

#### 2. 画像を読み取る（read_cards.py）

```bash
# デフォルトで card_images を参照し、結果を read_cards_result.json に保存
python read_cards.py

# デッキコードで一覧取得＋画像を card_images にダウンロード
python read_cards.py gLn9ng-Cr455W-iNnLLN

# フォルダ・出力先を指定する場合
python read_cards.py --folder ./card_images -o read_cards_result.json
```

- 結果は **必ずファイルに保存** される（未指定時は `read_cards_result.json`）
- **追加モード**: 既存 JSON があるときは新規カードだけ追加。同じ `_source_image` は解析スキップ、同じ `id` は上書きせずスキップ（内容が違うときだけ stderr に表示）

#### 3. data.py を更新する（update_cards_from_json.py）

**実行すると card/data.py の「JSON から生成」マーカーで囲まれたブロックが、read_cards_result.json の内容で置き換わります。**

```bash
# data.py を更新する（既定）
python update_cards_from_json.py

# 更新せず、内容だけ確認する
python update_cards_from_json.py --dry-run

# data.py は触らず、生成コードだけファイルに出力する
python update_cards_from_json.py -o card/data_generated.py
```

- **更新される箇所**: カード定義（PokemonCard）、`CARD_ID_TO_NAME`、`_CARD_REGISTRY` のうち、「# ----- JSON から生成 -----」で囲まれた部分のみ
- 説明文に「こんらん」と「このポケモン」が含まれる技は、自動で `status_effect='confusion'`, `status_effect_target='self'` が付きます

#### まとめ（A）

1. カード画像を `card_images` などに置く  
2. `python read_cards.py` で `read_cards_result.json` を生成・追加  
3. `python update_cards_from_json.py` で `card/data.py` を更新  
4. デッキ・シミュレーションでは `card/data.py` のカード ID（例: `zupika-svd-041`）を参照

---

### B. 対戦シミュレーション（scripts/simulate.py）

デッキ A〜E および登録デッキで自動対戦を繰り返し、勝率などを表示します。初手 7 枚・先行ドローあり・サイド 6 枚取り切りで勝敗判定です。

```bash
# 既定: 1000 回、デッキ 0 vs 1、1 回目のみログ表示
python scripts/simulate.py

# 対戦数・デッキ番号を指定
python scripts/simulate.py 500 3 4   # 500 回、デッキ 3 vs 4

# 使い方
python scripts/simulate.py --help
```

- デッキ番号: 0=オタチ, 1=ワニ, 2=カエル, 3=ワルビアル, 4=ジバコイル, 5 以降=登録デッキ（`registered_decks.json`）

---

### C. 対戦記録と動画（record_game.py → make_video.py）

1 試合を実行してログと状態スナップショットを保存し、必要に応じて盤面動画（MP4）を生成します。

#### 1. 試合を記録する（scripts/record_game.py）

```bash
# 対戦 ID を日時で自動採番 → battles/20260227_143052/ など
python scripts/record_game.py

# 対戦 ID を手動指定
python scripts/record_game.py --id my_match
```

- 各対戦ごとに `battles/<id>/` に `battle.log` と `battle_states.pkl` が保存されます

#### 2. 動画を生成する（scripts/make_video.py）

```bash
# 最新の対戦で動画作成 → battles/<最新ID>/battle.mp4
python scripts/make_video.py

# 指定した対戦 ID で動画作成
python scripts/make_video.py --battle-id 20260227_143052

# FPS や出力先の指定
python scripts/make_video.py --battle-id <id> --fps 1.0 -o out.mp4
```

- 記録した状態を 1 フレーム 1 枚の画像に描画し、ffmpeg で MP4 にまとめます

---

### D. テスト（pytest）

ルートで pytest を実行し、ゲームルールやカード効果の挙動を検証します。

```bash
# 全テストを実行
python -m pytest tests/ -v

# 短い出力で実行
python -m pytest tests/ -q

# 特定のファイルだけ実行
python -m pytest tests/test_fushiginaame.py -v
python -m pytest tests/test_rules_01_play_supplement.py -v
```

| ファイル | 役割 |
|----------|------|
| `tests/conftest.py` | 共通フィクスチャ（最小 GameState など） |
| `tests/test_fushiginaame.py` | ふしぎなアメで 1 進化とばし進化する挙動を検証 |
| `tests/test_rules_01_play_supplement.py` | rules/01_play_supplement.md に基づくルール（ワザ・にげる・ベンチ・進化・グッズ・サポート・エネルギー・きぜつ・勝敗・セットアップ等） |
| `tests/test_rules_02_card_descriptions.py` | rules/02_card_descriptions.md に基づくルール（カード説明文優先・ダメージ計算・効果・用語） |
| `tests/test_rules_advanced.py` | rules/advanced_rule.md（上級ルール）に基づくルールテスト |

---

## スクリプト一覧

| ファイル | 役割 |
|----------|------|
| `read_cards.py` | カード画像 or デッキコードから JSON を生成（OpenAI Vision 使用） |
| `update_cards_from_json.py` | JSON で `card/data.py` の生成ブロックを更新 |
| `scripts/simulate.py` | 指定デッキで N 回対戦し勝率などを表示 |
| `scripts/record_game.py` | 1 試合を実行しログ・状態を `battles/<id>/` に保存 |
| `scripts/make_video.py` | 保存した状態から盤面フレームを描画し MP4 を出力 |

## コアモジュール

| ファイル | 役割 |
|----------|------|
| `card/` | カード型（model）とマスタデータ（data）、get_card_by_id 等 |
| `deck.py` | デッキレシピ（A〜E）、登録デッキ、create_deck / デッキコードからの生成 |
| `game/` | GameState、ターン進行、ドロー・進化・ワザ・アイテム等の処理、AI 手番（state / damage / evolution / trainers / attack / turn） |
| `board_render.py` | GameState を 1 枚画像に描画（動画用フレーム生成） |

## ルール参照

`rules/` に公式の「上級プレイヤー用ルールガイド」を整理したドキュメントがあります。遊び方の補足やカード説明文の解釈は [rules/README.md](rules/README.md) を参照してください。
