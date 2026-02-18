#!/usr/bin/env python3
"""
フォルダ内のポケモンカード画像を読み取り、カード情報を JSON で出力するコマンド。

使い方:
  python read_cards.py [フォルダパス]   # 結果は read_cards_result.json に保存
  python read_cards.py --folder ./card_images -o cards.json  # 出力先を指定

環境変数 OPENAI_API_KEY が必要です。.env に書いておくと自動で読み込みます（.env は Git に含めないこと）。
画像は OpenAI Vision API (gpt-4o) で解析します。
"""
import argparse
import base64
import json
import re
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
_project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(_project_root))

# 出力先のデフォルトファイル（-o 未指定時は必ずここに保存する）
DEFAULT_OUTPUT_FILE = _project_root / "read_cards_result.json"

# .env があれば読み込み（OPENAI_API_KEY を Git に上げずに管理するため）
try:
    from dotenv import load_dotenv
    load_dotenv(_project_root / ".env")
except ImportError:
    pass

# サポートする画像拡張子
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

# ポケモンタイプの id 一覧（画像から選んでもらう用）
POKEMON_TYPE_IDS = [
    "grass", "fire", "water", "lightning", "psychic",
    "fighting", "darkness", "metal", "fairy",
    "colorless", "dragon",
]

# 技のコストで使うタイプ id（エネルギー 9 種 + 無色）
ENERGY_COST_TYPE_IDS = [
    "grass", "fire", "water", "lightning", "psychic",
    "fighting", "darkness", "metal", "fairy", "colorless",
]

CARD_EXTRACTION_PROMPT = """この画像はポケモンカードです。以下のルールに従い、JSON のみを 1 つ返してください。説明文は不要です。

- カード上部: 名前（日本語）、HP、ポケモンのタイプ（HP の横のシンボル）。
- 技: 各技の「名前」「必要なエネルギー数（合計）」「エネルギーコストの内訳（タイプ指定がある場合）」「ダメージ」「自分への反動ダメージ」「相手ベンチへの追加ダメージ」「説明（任意）」。
  - エネルギーコストは技名の左のシンボルを左からそのまま配列にすること。同じタイプが2つなら ["fighting", "fighting"] のように同じ id を2回並べる。闘2なら無色は含めない。雷1＋無色1のときだけ ["lightning", "colorless"]。全て無色（星マークのみ）のときだけ energy_cost_typed は null。
- 弱点: 下部の「弱点」のタイプ。なければ null。
- にげる: にげるコスト（捨てるエネルギー数）。無色は任意で 1 つと数える。
- 進化: 進化元のポケモン名（例: ズピカ）があれば、そのカード id 用のローマ字（例: zupika）。たねポケモンなら null。
- カード左下: (1) レギュレーション（そのカードが使える環境の目安。例: G）、(2) 収録パック/デッキの識別（例: SVD）、(3) カード番号（例: 064/139）。

ポケモンタイプは: """ + ", ".join(POKEMON_TYPE_IDS) + """
技のエネルギーコストのタイプは: """ + ", ".join(ENERGY_COST_TYPE_IDS) + """

返す JSON の形式（必ずこのキーのみ、余計なキーは入れない）:
{
  "name_ja": "日本語名",
  "hp": 数値,
  "pokemon_type": "上記のポケモンタイプ id",
  "weakness": "上記のポケモンタイプ id または null",
  "retreat_cost": 数値,
  "evolves_from": "進化元の id または null",
  "regulation": "レギュレーション（使用可能な環境の目安）。例: G。読み取れなければ null",
  "set_code": "収録されているパックやデッキの識別。例: SVD。読み取れなければ null",
  "card_number": "カード番号（例: 064/139）。読み取れなければ null",
  "attacks": [
    {
      "name": "技名",
      "energy_cost": 数値,
      "energy_cost_typed": 画像の通り（例: 闘2→["fighting","fighting"]、雷1無色1→["lightning","colorless"]）。全て無色なら null,
      "damage": 数値,
      "self_damage": 0,
      "bench_damage": 0,
      "description": "説明"
    }
  ]
}
※ energy_cost_typed は技にタイプ指定があるときだけ配列で。全て無色なら null。
"""


def list_image_paths(folder: Path) -> list[Path]:
    """フォルダ内の画像ファイルのパスを拡張子でソートして返す。"""
    if not folder.is_dir():
        return []
    paths = [p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(paths, key=lambda p: p.name)


def image_to_base64_url(path: Path) -> str:
    """画像ファイルを base64 データ URL に変換する。"""
    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    ext = path.suffix.lower()
    mime = "image/png" if ext == ".png" else "image/jpeg" if ext in (".jpg", ".jpeg") else "image/webp"
    return f"data:{mime};base64,{b64}"


def extract_json_from_response(text: str) -> dict | None:
    """レスポンステキストから JSON ブロックを 1 つ取り出す。"""
    # コードブロックがあればその中身を使う
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass
    # 全体を JSON として試す
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    # 最初の { から最後の } までを切り出す
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
    return None


def romanji_from_filename(filename: str) -> str | None:
    """ファイル名からローマ字部分を抽出（例: 043743_P_ZUPIKA.jpg -> zupika）。_P_ の後ろの英字を返す。"""
    if not filename:
        return None
    stem = Path(filename).stem
    m = re.search(r"_P_([A-Za-z0-9]+)$", stem)
    if m:
        return m.group(1).lower()
    # _P_ が無い場合は拡張子を除いた末尾の英字ブロック（例: KARAMINGO.jpg -> karamingo）
    m = re.search(r"([A-Za-z]+)(?:_?\d*)?$", stem, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    return None


def name_ja_to_id(name_ja: str) -> str:
    """日本語名をカード id 用のローマ字に変換（pykakasi）。ファイル名から取れないときのフォールバック用。"""
    s = (name_ja or "").strip()
    if not s:
        return "unknown"
    if s.isascii():
        return s.lower().replace(" ", "-")
    try:
        import pykakasi
        kks = pykakasi.kakasi()
        result = kks.convert(s)
        romaji = "".join(item.get("hepburn", "") for item in result)
        if not romaji or not romaji.isascii():
            romaji = "unknown"
        else:
            romaji = re.sub(r"[^a-zA-Z0-9]+", "-", romaji.lower()).strip("-") or "unknown"
        return romaji
    except Exception:
        return re.sub(r"[^a-zA-Z0-9]", "", s.lower()) or "unknown"


def make_card_id(data: dict, source_filename: str | None = None) -> str:
    """
    カードを一意に識別する id を生成する。
    ファイル名（source_filename または data の _source_image）からローマ字を取得し、
    set_code と card_number があれば「名前-set_code-番号」に。無ければ名前のみ。
    """
    name_id = None
    fname = source_filename or data.get("_source_image")
    if fname:
        name_id = romanji_from_filename(fname)
    if not name_id:
        name_id = name_ja_to_id(data.get("name_ja", ""))
    set_code = data.get("set_code")
    card_number = data.get("card_number")
    if set_code and card_number:
        # 番号は "041/139" の左側だけ使う（英数字のみ）
        num_part = card_number.split("/")[0].strip() if "/" in str(card_number) else str(card_number)
        num_part = re.sub(r"[^0-9a-zA-Z]", "", num_part) or "0"
        set_part = re.sub(r"[^0-9a-zA-Z]", "", str(set_code).lower()) or "unknown"
        return f"{name_id}-{set_part}-{num_part}"
    return name_id


def read_card_from_image(image_path: Path, *, openai_client=None) -> dict | None:
    """1 枚の画像からカード情報を抽出する。"""
    try:
        from openai import OpenAI
    except ImportError:
        print("openai パッケージが必要です: pip install openai", file=sys.stderr)
        return None

    try:
        client = openai_client or OpenAI()
    except Exception as e:
        if "api_key" in str(e).lower() or "openai_api_key" in str(e).lower():
            print("環境変数 OPENAI_API_KEY を設定してください。", file=sys.stderr)
        raise
    image_url = image_to_base64_url(image_path)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": CARD_EXTRACTION_PROMPT},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        max_tokens=1024,
    )
    text = response.choices[0].message.content or ""
    data = extract_json_from_response(text)
    if not data:
        return None
    # id は main で _source_image を付けたあと make_card_id で設定（ファイル名のローマ字を使用）
    # 進化元が日本語名なら id に変換
    ev = data.get("evolves_from")
    if ev and not ev.isascii():
        data["evolves_from"] = name_ja_to_id(ev)
    # タイプがリスト外なら null に（API の typo 対策）
    if data.get("pokemon_type") and data["pokemon_type"] not in POKEMON_TYPE_IDS:
        data["pokemon_type"] = None
    if data.get("weakness") and data["weakness"] not in POKEMON_TYPE_IDS:
        data["weakness"] = None
    return data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="フォルダ内のポケモンカード画像を読み取り、JSON で出力する",
    )
    parser.add_argument(
        "folder",
        nargs="?",
        default="card_images",
        help="画像が入ったフォルダ（デフォルト: card_images）",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help=f"出力先 JSON ファイル（省略時は {DEFAULT_OUTPUT_FILE.name} に保存）",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="進捗を stderr に出さない",
    )
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.exists():
        print(f"エラー: フォルダが存在しません: {folder}", file=sys.stderr)
        sys.exit(1)
    if not folder.is_dir():
        print(f"エラー: ディレクトリではありません: {folder}", file=sys.stderr)
        sys.exit(1)

    paths = list_image_paths(folder)
    if not paths:
        print(f"エラー: 画像ファイルがありません: {folder}", file=sys.stderr)
        sys.exit(1)

    if not args.quiet:
        print(f"画像を {len(paths)} 件検出しました。", file=sys.stderr)

    # 既存の結果を読み込む（追加モードのため）
    output_path = Path(args.output) if args.output else DEFAULT_OUTPUT_FILE
    existing_cards: list[dict] = []
    if output_path.exists():
        try:
            existing = json.loads(output_path.read_text(encoding="utf-8"))
            existing_cards = existing.get("cards", [])
        except (json.JSONDecodeError, OSError):
            pass
    existing_by_id = {c["id"]: c for c in existing_cards if c.get("id")}
    existing_sources = {c.get("_source_image") for c in existing_cards if c.get("_source_image")}

    def card_content_for_compare(c: dict) -> dict:
        """比較用（_source_image は無視）。"""
        return {k: v for k, v in c.items() if k != "_source_image"}

    to_add: list[dict] = []
    for i, path in enumerate(paths, 1):
        if path.name in existing_sources:
            if not args.quiet:
                print(f"[{i}/{len(paths)}] {path.name} は既存のためスキップ", file=sys.stderr)
            continue
        if not args.quiet:
            print(f"[{i}/{len(paths)}] {path.name} を解析中...", file=sys.stderr)
        data = read_card_from_image(path)
        if not data:
            if not args.quiet:
                print(f"  スキップ（解析できませんでした）", file=sys.stderr)
            continue
        data["_source_image"] = path.name
        data["id"] = make_card_id(data)
        cid = data.get("id")
        if cid in existing_by_id:
            # 同じ id が既にある：内容が違うときだけ報告
            if card_content_for_compare(data) != card_content_for_compare(existing_by_id[cid]):
                print(f"データの食い違い（id={cid}）: 既存のレコードと画像の内容が異なります。", file=sys.stderr)
            continue
        to_add.append(data)

    result_cards = existing_cards + to_add
    result_cards.sort(key=lambda c: c.get("name_ja", ""))
    out = {"cards": result_cards}
    json_str = json.dumps(out, ensure_ascii=False, indent=2)
    output_path.write_text(json_str, encoding="utf-8")
    if not args.quiet:
        added = len(to_add)
        print(f"結果を保存しました: {output_path}（追加 {added} 件、既存 {len(existing_cards)} 件）", file=sys.stderr)


if __name__ == "__main__":
    main()
