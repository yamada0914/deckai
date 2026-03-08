#!/usr/bin/env python3
"""
フォルダ内のポケモンカード画像を読み取り、カード情報を JSON で出力するコマンド。

使い方:
  python read_cards.py gLn9ng-Cr455W-iNnLLN   # デッキコードで一覧取得＋画像を card_images にダウンロード
  python read_cards.py gLn9ng-Cr455W-iNnLLN card_images/example   # 画像フォルダを指定（テスト用）
  python read_cards.py --images-only フォルダ --replace-existing   # 指定フォルダ内の画像だけ再解析し、既存 id は上書き（例: ルナトーン・ハリテヤマのみ入れたフォルダで再解析）

環境変数 OPENAI_API_KEY が必要です。.env に書いておくと自動で読み込みます（.env は Git に含めないこと）。
画像は OpenAI Vision API (gpt-4o) で解析します。
"""
import argparse
import base64
import json
import os
import re
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(_project_root))

DEFAULT_OUTPUT_FILE = _project_root / "read_cards_result.json"
CARD_DATA_DIR = "read_cards_data"
CARD_FILES_BY_SLOT = {
    "pokemon": f"{CARD_DATA_DIR}/pokemon.json",
    "trainers": f"{CARD_DATA_DIR}/trainers.json",
    "energy": f"{CARD_DATA_DIR}/energy.json",
}

try:
    from dotenv import load_dotenv
    load_dotenv(_project_root / ".env")
except ImportError:
    pass

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

POKEMON_TYPE_IDS = [
    "grass", "fire", "water", "lightning", "psychic",
    "fighting", "darkness", "metal", "fairy",
    "colorless", "dragon",
]

ENERGY_COST_TYPE_IDS = [
    "grass", "fire", "water", "lightning", "psychic",
    "fighting", "darkness", "metal", "fairy", "colorless",
]

ENERGY_TYPE_IDS = ["grass", "fire", "water", "lightning", "psychic", "fighting", "darkness", "metal", "fairy"]

CARD_EXTRACTION_PROMPT = """この画像はポケモンカードゲームのカードです。ポケモン・トレーナー（アイテム/サポート）・エネルギーのいずれかです。
まずカードの種類を判定し、以下のルールに従って JSON のみを 1 つ返してください。説明文は不要です。

【共通】
- まずカード左上付近（名前・「たね」/進化表記・HP の有無など）を見て、ポケモンかトレーナーかエネルギーかを判定する。
- card_type は "pokemon" / "goods" / "support" / "tool" / "energy" のいずれか 1 つ。グッズは "goods"、サポートは "support"、ポケモンのどうぐは "tool"、基本・特殊エネルギーは "energy"。
- ポケモン（card_type: "pokemon"）の場合のみ、カード左下の regulation（例: G）、set_code（例: SVD）、card_number（例: 064/139）を読み取り JSON に含める。トレーナー・エネルギーの場合はこの 3 項目は読み取らず含めない（出力トークン節約）。

【ポケモン（card_type: "pokemon"）】
- 名前、HP、ポケモンのタイプ（HP 横のシンボル）。技は「名前」「エネルギー数」「エネルギーコストの内訳」「ダメージ」「反動」「ベンチダメージ」「説明」。弱点・にげる・進化元。
- **特性（ability）**: カードに「特性」とラベルされたブロック（特性名＋説明文）があれば必ず読み取り、JSON の "ability" に含める。無ければ "ability": null。
- エネルギーコスト: 闘2→["fighting","fighting"]、雷1無色1→["lightning","colorless"]。全て無色なら energy_cost_typed は null。
- evolves_from: カード左上に「たね」と書いてある場合は必ず null。1進化・2進化の場合は「〇〇から進化」の進化元ポケモンの日本語名（name_ja）で返す（例: レアコイル）。id 用ローマ字（例: rarecoil）でも可。たねなら null。
- evolution_stage: カード左上の進化表記に合わせて "basic"（たね） / "stage1"（1進化） / "stage2"（2進化） のいずれか 1 つを必ず返す。

【トレーナー・グッズ（card_type: "goods"）】
- 名前、効果の説明文（description）。きずぐすり・ポケモンいれかえなど、1 ターンに何枚でも使えるカード。

【トレーナー・ポケモンのどうぐ（card_type: "tool"）】
- 名前、効果の説明文（description）。ポケモン 1 匹に 1 枚つけるカード（例: 岩のむねあて）。

【トレーナー・サポート（card_type: "support"）】
- 名前、効果の説明文（description）。ネモ・博士の研究など、1 ターンに 1 枚まで使えるカード。

【エネルギー（card_type: "energy"）】
- 名前。基本エネルギーなら energy_type にタイプ id（草→grass、炎→fire、水→water、雷→lightning、超→psychic、闘→fighting、悪→darkness、鋼→metal、妖→fairy）。特殊エネルギー（無色や複数タイプを付与する等）なら energy_type は null。

ポケモンタイプ id: """ + ", ".join(POKEMON_TYPE_IDS) + """
技のエネルギーコスト id: """ + ", ".join(ENERGY_COST_TYPE_IDS) + """
エネルギーカードのタイプ id: """ + ", ".join(ENERGY_TYPE_IDS) + """

返す JSON は card_type に応じて次のいずれか。必ずこれらのキーのみにし、余計なキーは入れない。

ポケモンの場合:
{
  "card_type": "pokemon",
  "name_ja": "日本語名",
  "hp": 数値,
  "pokemon_type": "上記のポケモンタイプ id",
  "weakness": "上記 id または null",
  "retreat_cost": 数値,
  "evolves_from": "進化元の name_ja（日本語名）または id 用ローマ字、たねなら null",
  "evolution_stage": "basic または stage1 または stage2（たね→basic、1進化→stage1、2進化→stage2）",
  "regulation": "G などまたは null",
  "set_code": "SVD などまたは null",
  "card_number": "064/139 などまたは null",
  "ability": null または { "name": "特性名", "description": "特性の説明文" }（カードに特性ブロックが無い場合は null）,
  "attacks": [ { "name": "技名", "energy_cost": 数値, "energy_cost_typed": 配列または null, "damage": 数値, "self_damage": 0, "bench_damage": 0, "description": "説明" } ]
}

グッズの場合:
{
  "card_type": "goods",
  "name_ja": "日本語名",
  "description": "効果の説明文"
}

サポートの場合:
{
  "card_type": "support",
  "name_ja": "日本語名",
  "description": "効果の説明文"
}

エネルギーの場合:
{
  "card_type": "energy",
  "name_ja": "日本語名",
  "energy_type": "grass/fire/water/lightning/psychic/fighting/darkness/metal/fairy のいずれか、または特殊エネルギーなら null"
}
"""


DECK_RESULT_URL = "https://www.pokemon-card.com/deck/result.html/deckID/{deck_code}/"
DECK_IMAGE_BASE_URL = "https://www.pokemon-card.com"
EXAMPLE_DECK_CODE = "gLn9ng-Cr455W-iNnLLN"

SELENIUM_WAIT_SECONDS = 30


def _fetch_deck_page_html_with_selenium(url: str, _log_err: bool = False) -> tuple[str | None, str | None]:
    """
    Selenium で公式デッキページを開き、JS 描画後の HTML を返す。
    戻り値: (html, None) 成功時。(None, "原因") 失敗時。タイムアウトで空の一覧の場合は (html, "timeout")。
    _log_err が True のときは stderr に原因を出力する。
    """
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from webdriver_manager.chrome import ChromeDriverManager
    except ImportError as e:
        if _log_err:
            print(f"Selenium 取得失敗（import）: {e}", file=sys.stderr)
        return (None, "import")
    options = Options()
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    driver = None
    try:
        driver_path = ChromeDriverManager().install()
        if "THIRD_PARTY_NOTICES" in driver_path:
            parent = Path(driver_path).parent
            alt = parent / "chromedriver"
            if alt.is_file():
                driver_path = str(alt)
        if os.path.isfile(driver_path):
            os.chmod(driver_path, 0o755)
        service = Service(driver_path)
        driver = webdriver.Chrome(service=service, options=options)
        import time
        driver.get(url)
        if _log_err:
            print("ブラウザでページを開きました。id=\"cardImagesView\" 内に「枚」の文字が現れるまで最大 30 秒待ちます…", file=sys.stderr)
        time.sleep(2)
        try:
            WebDriverWait(driver, SELENIUM_WAIT_SECONDS).until(
                EC.text_to_be_present_in_element((By.ID, "cardImagesView"), "枚")
            )
        except Exception as e:
            html = driver.page_source if driver else None
            if _log_err:
                print(f"Selenium 待機タイムアウト（{SELENIUM_WAIT_SECONDS} 秒以内に id=\"cardImagesView\" 内に「枚」が現れませんでした）: {e}", file=sys.stderr)
            return (html, "timeout")
        if _log_err:
            print("「枚」を検出しました。ページの HTML を取得してカード名・枚数を抜き出します。", file=sys.stderr)
        time.sleep(1)
        return (driver.page_source, None)
    except Exception as e:
        if _log_err:
            print(f"Selenium 取得失敗: {e}", file=sys.stderr)
        return (None, "exception")
    finally:
        if driver:
            driver.quit()


def get_deck_page_html(
    deck_code: str,
    html_path: Path | None = None,
    log_selenium_errors: bool = False,
) -> tuple[str | None, str | None]:
    """
    デッキ結果ページの HTML を 1 つ取得する。
    - html_path が指定されていればそのファイルを読む。
    - そうでなければ Selenium で公式 URL を開く。
    戻り値: (html, None) 成功時。(None, "原因") 失敗時。
    """
    deck_code = (deck_code or "").strip()
    if html_path and html_path.is_file():
        try:
            return (html_path.read_text(encoding="utf-8", errors="replace"), None)
        except OSError:
            return (None, "read_error")
    if not deck_code:
        return (None, "no_input")
    url = DECK_RESULT_URL.format(deck_code=deck_code)
    return _fetch_deck_page_html_with_selenium(url, _log_err=log_selenium_errors)


def fetch_deck_list(
    deck_code: str,
    html_path: Path | None = None,
    log_selenium_errors: bool = False,
) -> list[dict]:
    """
    デッキ一覧（カード名・枚数）を取得する。
    get_deck_page_html で HTML を取得し、_parse_deck_page_html でパースする。
    戻り値: [ {"name_ja": "メグロコ", "count": 3}, ... ]
    """
    html, reason = get_deck_page_html(deck_code, html_path, log_selenium_errors)
    if not html:
        return []
    cards = _parse_deck_page_html(html)
    if not cards and reason == "timeout" and log_selenium_errors:
        print("（ヒント: 画像一覧が表示されるまで待つか、もう少し待ってから再実行してください）", file=sys.stderr)
    return cards


def _parse_deck_page_html(html: str) -> list[dict]:
    """
    デッキ結果ページの HTML からカード名・枚数のリストを抽出する。
    まず cardImagesView（画像一覧）内の img alt と「N枚」をパースし、
    取れなければ cardListView（リスト）の linkCursor と「N枚」でパースする。
    """
    section = re.search(r'id="cardImagesView"[^>]*>([\s\S]*?)</section>', html)
    if section:
        block = section.group(1)
        names = re.findall(r'<img[^>]*\salt="([^"]+)"', block)
        counts = re.findall(r'<span>(\d+)枚</span>', block)
        if len(names) == len(counts) and names:
            return [
                {"name_ja": name.strip(), "count": int(c)}
                for name, c in zip(names, counts)
            ]
    section = re.search(r'id="cardListView"[^>]*>([\s\S]*?)</section>', html)
    if not section:
        return []
    block = section.group(1)
    names = re.findall(r'class="linkCursor">([^<]+)<br>', block)
    counts = re.findall(r'<span>(\d+)枚</span>', block)
    if len(names) != len(counts) or not names:
        return []
    return [
        {"name_ja": name.strip(), "count": int(c)}
        for name, c in zip(names, counts)
    ]


def _extract_card_image_urls(html: str) -> list[tuple[str, str]]:
    """
    デッキ結果ページの HTML（cardImagesView）からカード画像の URL と保存用ファイル名を抽出する。
    戻り値: [(full_url, filename), ...]。同じカードは 1 回だけ（URL の重複は除く）。
    """
    section = re.search(r'id="cardImagesView"[^>]*>([\s\S]*?)</section>', html)
    if not section:
        return []
    block = section.group(1)
    urls = re.findall(r'<img[^>]*\ssrc="([^"]+)"', block)
    seen: set[str] = set()
    result: list[tuple[str, str]] = []
    for raw in urls:
        raw = raw.strip()
        if not raw or raw in seen:
            continue
        seen.add(raw)
        if raw.startswith("//"):
            full = "https:" + raw
        elif raw.startswith("/"):
            full = DECK_IMAGE_BASE_URL.rstrip("/") + raw
        else:
            full = raw
        path = full.split("?")[0]
        filename = path.split("/")[-1] if "/" in path else "image.jpg"
        if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            filename += ".jpg"
        result.append((full, filename))
    return result


def download_deck_images(html: str, folder: Path, quiet: bool = False) -> int:
    """
    デッキ結果ページの HTML から画像 URL を抽出し、指定フォルダに保存する。
    戻り値: ダウンロードしたファイル数。
    """
    urls_and_names = _extract_card_image_urls(html)
    if not urls_and_names:
        return 0
    folder.mkdir(parents=True, exist_ok=True)
    import urllib.request
    count = 0
    for url, filename in urls_and_names:
        path = folder / filename
        if path.exists():
            continue
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as res:
                path.write_bytes(res.read())
            count += 1
            if not quiet:
                print(f"  保存: {filename}", file=sys.stderr)
        except Exception as e:
            if not quiet:
                print(f"  取得失敗 {filename}: {e}", file=sys.stderr)
    return count


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
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
    return None


def romanji_from_filename(filename: str) -> str | None:
    """ファイル名からローマ字部分を抽出。_P_ / _T_ / _E_ の後ろの英字を返す（例: 043743_P_ZUPIKA.jpg -> zupika, 043834_T_TANPANKOZOU.jpg -> tanpankozou）。"""
    if not filename:
        return None
    stem = Path(filename).stem
    m = re.search(r"_[PTE]_([A-Za-z0-9]+)$", stem, re.IGNORECASE)
    if m:
        return m.group(1).lower()
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
    トレーナー（goods / support）は同名＝同一効果のため名前のみで id を決める。
    ポケモン・エネルギーは同名でも別カードがあり得るため、set_code と card_number があれば「名前-set_code-番号」にする。
    """
    card_type = (data.get("card_type") or "pokemon").strip().lower()
    if card_type in ("goods", "support", "tool"):
        name_id = name_ja_to_id(data.get("name_ja", "")) or ""
        if not name_id or name_id == "unknown":
            fname = source_filename or data.get("_source_image")
            if fname:
                name_id = romanji_from_filename(fname) or ""
        return name_id or "unknown"

    name_id = None
    fname = source_filename or data.get("_source_image")
    if fname:
        name_id = romanji_from_filename(fname)
    if not name_id or name_id == "unknown":
        name_id = name_ja_to_id(data.get("name_ja", "")) or None
    set_code = data.get("set_code")
    card_number = data.get("card_number")
    if set_code and card_number and name_id and name_id != "unknown":
        num_part = card_number.split("/")[0].strip() if "/" in str(card_number) else str(card_number)
        num_part = re.sub(r"[^0-9a-zA-Z]", "", num_part) or "0"
        set_part = re.sub(r"[^0-9a-zA-Z]", "", str(set_code).lower()) or "unknown"
        return f"{name_id}-{set_part}-{num_part}"
    return name_id or "unknown"


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
    card_type = (data.get("card_type") or "pokemon").strip().lower()
    if card_type == "item":
        card_type = "goods"
    if card_type not in ("pokemon", "goods", "support", "tool", "energy"):
        card_type = "pokemon"
    data["card_type"] = card_type

    if card_type == "pokemon":
        if data.get("pokemon_type") and data["pokemon_type"] not in POKEMON_TYPE_IDS:
            data["pokemon_type"] = None
        if data.get("weakness") and data["weakness"] not in POKEMON_TYPE_IDS:
            data["weakness"] = None
        if not data.get("attacks"):
            data["attacks"] = []
    elif card_type in ("goods", "support"):
        if "description" not in data:
            data["description"] = ""
    elif card_type == "energy":
        et = data.get("energy_type")
        if et and et not in ENERGY_TYPE_IDS:
            data["energy_type"] = None
    return data


def _fetch_deck_and_download(deck_code: str, folder: Path, quiet: bool) -> list[dict]:
    """デッキページを取得し、一覧をパースして画像をダウンロードする。戻り値は deck_lists。"""
    deck_lists: list[dict] = []
    html, _ = get_deck_page_html(deck_code, None, log_selenium_errors=not quiet)
    if not html:
        if not quiet:
            print(f"デッキ {deck_code}: 一覧を取得できませんでした（Selenium 未導入・タイムアウト・またはページ構造変更の可能性）。", file=sys.stderr)
        return deck_lists
    deck_cards = _parse_deck_page_html(html)
    if deck_cards:
        deck_lists.append({"deck_code": deck_code, "cards": deck_cards})
        if not quiet:
            print(f"デッキ {deck_code}: {len(deck_cards)} 種類のカードを取得しました。", file=sys.stderr)
    elif not quiet:
        print(f"デッキ {deck_code}: 一覧を取得できませんでした（cardImagesView/cardListView をパースできませんでした）。", file=sys.stderr)
    if not quiet:
        print("カード画像をダウンロードしています…", file=sys.stderr)
    n = download_deck_images(html, folder, quiet=quiet)
    if not quiet:
        print(f"カード画像を {n} 枚保存しました: {folder}", file=sys.stderr)
    return deck_lists


def _group_cards_by_type(cards: list[dict]) -> dict[str, list[dict]]:
    """カードを card_type でグループ化する。pokemon / goods / support / tool / energy。tool は trainers に含める。"""
    groups: dict[str, list[dict]] = {"pokemon": [], "goods": [], "support": [], "tool": [], "energy": []}
    for c in cards:
        t = (c.get("card_type") or "pokemon").strip().lower()
        if t not in groups:
            t = "pokemon"
        groups[t].append(c)
    for key in groups:
        groups[key].sort(key=lambda x: x.get("name_ja", ""))
    return groups


_ENERGY_OMIT_KEYS = frozenset({"regulation", "set_code", "card_number"})
_TRAINER_OMIT_KEYS = frozenset({"regulation", "set_code", "card_number"})


def _energy_card_for_save(card: dict) -> dict:
    """エネルギー用に regulation / set_code / card_number を除いたコピーを返す。"""
    return {k: v for k, v in card.items() if k not in _ENERGY_OMIT_KEYS}


def _trainer_card_for_save(card: dict) -> dict:
    """トレーナー（グッズ・サポート）用に regulation / set_code / card_number を除いたコピーを返す。"""
    return {k: v for k, v in card.items() if k not in _TRAINER_OMIT_KEYS}


def _cards_to_flat_list(cards: list | dict) -> list[dict]:
    """cards が配列ならそのまま、card_type でグループ化された dict なら 1 つの配列に展開して返す。"""
    if isinstance(cards, list):
        return cards
    if isinstance(cards, dict):
        order = ("pokemon", "goods", "support", "energy")
        return [c for k in order for c in (cards.get(k) or [])]
    return []


def _save_deck_lists_only(deck_lists: list[dict], output_path: Path, quiet: bool, message: str = "デッキ一覧のみ保存しました") -> None:
    """デッキ一覧をマージして JSON に書き出す。既存の cards は消さず、分割ファイルがあればそのまま残す。"""
    existing_cards, existing_deck_lists = _load_existing_result(output_path)
    final_deck_lists = _merge_deck_lists(existing_deck_lists, deck_lists)
    groups = _group_cards_by_type(existing_cards)
    base_dir = output_path.parent
    (base_dir / CARD_DATA_DIR).mkdir(parents=True, exist_ok=True)
    trainers_raw = (groups.get("goods") or []) + (groups.get("support") or []) + (groups.get("tool") or [])
    trainers_raw.sort(key=lambda x: x.get("name_ja", ""))
    trainers_list = [_trainer_card_for_save(c) for c in trainers_raw]
    energy_list = [_energy_card_for_save(c) for c in (groups.get("energy") or [])]
    for slot, items in [("pokemon", groups.get("pokemon") or []), ("trainers", trainers_list), ("energy", energy_list)]:
        (base_dir / CARD_FILES_BY_SLOT[slot]).write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    out = {"deck_lists": final_deck_lists, "card_files": CARD_FILES_BY_SLOT}
    output_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    if not quiet:
        print(f"{message}: {output_path}", file=sys.stderr)


def _load_cards_from_split_files(base_dir: Path, card_files: dict[str, str]) -> list[dict]:
    """card_files で指定された JSON を読み、1 つの配列にマージして返す。"""
    order = ("pokemon", "trainers", "energy")
    result: list[dict] = []
    for key in order:
        name = card_files.get(key)
        if not name:
            continue
        path = base_dir / name
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            items = data if isinstance(data, list) else data.get("cards", data.get("items", []))
            if isinstance(items, list):
                result.extend(items)
        except (json.JSONDecodeError, OSError):
            pass
    return result


def _load_existing_result(output_path: Path) -> tuple[list[dict], list[dict]]:
    """既存の read_cards 結果を読む。戻り値は (existing_cards, existing_deck_lists)。
    単一ファイルの cards / card_files 分割の両方に対応。"""
    if not output_path.exists():
        return ([], [])
    try:
        data = json.loads(output_path.read_text(encoding="utf-8"))
        deck_lists = data.get("deck_lists", [])
        cards_raw = data.get("cards")
        card_files = data.get("card_files")
        if card_files and isinstance(card_files, dict):
            cards = _load_cards_from_split_files(output_path.parent, card_files)
        else:
            cards = _cards_to_flat_list(cards_raw) if cards_raw is not None else []
        return (cards, deck_lists)
    except (json.JSONDecodeError, OSError):
        return ([], [])


def _merge_deck_lists(existing: list[dict], new: list[dict]) -> list[dict]:
    """既存のデッキ一覧に今回の分をマージ（同じ deck_code は上書き）。"""
    by_code = {d["deck_code"]: d for d in existing if d.get("deck_code")}
    for d in new:
        by_code[d["deck_code"]] = d
    return list(by_code.values())


def _process_new_images(
    paths: list[Path], existing_cards: list[dict], quiet: bool, *, replace_existing: bool = False
) -> tuple[list[dict], list[dict], int]:
    """
    画像パスを Vision API で解析する。
    replace_existing が True のときは既存スキップをせず全画像を解析し、id が既存と一致したら既存を上書きする。
    戻り値: (既存カードリスト（上書き反映済み）, 新規追加分のカード辞書リスト, 上書きした件数)。
    """

    def content_for_compare(c: dict) -> dict:
        return {k: v for k, v in c.items() if k != "_source_image"}

    current = list(existing_cards)
    existing_by_id = {c["id"]: c for c in current if c.get("id")}
    existing_sources = {c.get("_source_image") for c in current if c.get("_source_image")}
    to_add: list[dict] = []
    replaced_count = 0
    for i, path in enumerate(paths, 1):
        if not replace_existing:
            if path.name in existing_sources:
                if not quiet:
                    print(f"[{i}/{len(paths)}] {path.name} は既存のためスキップ", file=sys.stderr)
                continue
            candidate_id = romanji_from_filename(path.name)
            if candidate_id and candidate_id in existing_by_id:
                if not quiet:
                    print(f"[{i}/{len(paths)}] {path.name} は既存のためスキップ（id={candidate_id}）", file=sys.stderr)
                continue
        if not quiet:
            print(f"[{i}/{len(paths)}] {path.name} を解析中...", file=sys.stderr)
        data = read_card_from_image(path)
        if not data:
            if not quiet:
                print("  スキップ（解析できませんでした）", file=sys.stderr)
            continue
        data["_source_image"] = path.name
        data["id"] = make_card_id(data)
        cid = data.get("id")
        if cid in existing_by_id:
            if replace_existing:
                idx = next((j for j, c in enumerate(current) if c.get("id") == cid), -1)
                if idx >= 0:
                    current[idx] = data
                    existing_by_id[cid] = data
                    replaced_count += 1
                    if not quiet:
                        print(f"  既存を上書き: id={cid}", file=sys.stderr)
            else:
                if content_for_compare(data) != content_for_compare(existing_by_id[cid]):
                    print(f"データの食い違い（id={cid}）: 既存のレコードと画像の内容が異なります。", file=sys.stderr)
            continue
        to_add.append(data)
        existing_by_id[cid] = data
    return (current, to_add, replaced_count)


def _save_full_result(
    existing_cards: list[dict], to_add: list[dict], deck_lists: list[dict], output_path: Path, quiet: bool, *, replaced: int = 0
) -> None:
    """カード一覧をポケモン / トレーナーズ / エネルギーで 3 ファイルに分けて保存。メイン JSON には deck_lists と card_files のみ。"""
    result_cards = existing_cards + to_add
    groups = _group_cards_by_type(result_cards)
    base_dir = output_path.parent
    (base_dir / CARD_DATA_DIR).mkdir(parents=True, exist_ok=True)
    trainers_raw = (groups.get("goods") or []) + (groups.get("support") or []) + (groups.get("tool") or [])
    trainers_raw.sort(key=lambda x: x.get("name_ja", ""))
    trainers_list = [_trainer_card_for_save(c) for c in trainers_raw]
    energy_list = [_energy_card_for_save(c) for c in (groups.get("energy") or [])]
    slot_data = [
        ("pokemon", groups.get("pokemon") or []),
        ("trainers", trainers_list),
        ("energy", energy_list),
    ]
    for slot, items in slot_data:
        path = base_dir / CARD_FILES_BY_SLOT[slot]
        path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    out = {"deck_lists": deck_lists, "card_files": CARD_FILES_BY_SLOT}
    output_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    if not quiet:
        msg = f"結果を保存しました: {output_path}（ポケモン/トレーナーズ/エネルギー別ファイル + 追加 {len(to_add)} 件、既存 {len(existing_cards)} 件）"
        if replaced:
            msg += f"、上書き {replaced} 件"
        print(msg, file=sys.stderr)


def register_deck(result_path: Path, *, quiet: bool = False) -> int:
    """
    read_cards_result.json の deck_lists を registered_decks.json に登録する。
    戻り値: 登録したデッキ数。
    """
    if not result_path.is_file():
        if not quiet:
            print(f"エラー: ファイルがありません: {result_path}", file=sys.stderr)
        return 0
    try:
        data = json.loads(result_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        if not quiet:
            print(f"エラー: 読み取り失敗: {e}", file=sys.stderr)
        return 0
    deck_lists = data.get("deck_lists", [])
    if not deck_lists:
        if not quiet:
            print("デッキ一覧が空です。", file=sys.stderr)
        return 0
    DECK_CODE_DISPLAY_NAMES = {
        "gLn9ng-Cr455W-iNnLLN": "コライドンデッキ",
        "fbkfFF-napJOD-fFVdVk": "ミライドンデッキ",
    }
    registered = []
    for d in deck_lists:
        code = (d.get("deck_code") or "").strip()
        if not code:
            continue
        name = (d.get("name") or DECK_CODE_DISPLAY_NAMES.get(code) or code)[:40]
        registered.append({"deck_code": code, "name": name})
    try:
        from deck import REGISTERED_DECKS_PATH, clear_registered_decks_cache
        REGISTERED_DECKS_PATH.write_text(json.dumps(registered, ensure_ascii=False, indent=2), encoding="utf-8")
        clear_registered_decks_cache()
    except OSError as e:
        if not quiet:
            print(f"エラー: 書き込み失敗: {e}", file=sys.stderr)
        return 0
    if not quiet:
        print(f"デッキを {len(registered)} 件登録しました: {REGISTERED_DECKS_PATH}", file=sys.stderr)
        for i, r in enumerate(registered):
            print(f"  {i}: {r.get('name', r.get('deck_code', ''))} ({r.get('deck_code', '')})", file=sys.stderr)
    return len(registered)


def main() -> None:
    if len(sys.argv) >= 2 and sys.argv[1] == "register-deck":
        parser_reg = argparse.ArgumentParser(
            description="read_cards_result.json の deck_lists を registered_decks.json に登録し、シミュレーションで使えるようにする。",
        )
        parser_reg.add_argument(
            "-o", "--output",
            default=None,
            help=f"read_cards_result.json のパス（省略時は {DEFAULT_OUTPUT_FILE}）",
        )
        parser_reg.add_argument("-q", "--quiet", action="store_true", help="進捗を出さない")
        args_reg = parser_reg.parse_args(sys.argv[2:])
        result_path = Path(args_reg.output) if args_reg.output else DEFAULT_OUTPUT_FILE
        n = register_deck(result_path, quiet=args_reg.quiet)
        sys.exit(0 if n > 0 else 1)

    parser = argparse.ArgumentParser(
        description="公式デッキコードでカード一覧を取得し、カード画像をダウンロードする。画像からカード情報を JSON で出力する。",
    )
    parser.add_argument(
        "deck_code",
        nargs="?",
        default=None,
        metavar="DECK_CODE",
        help="公式デッキコード（必須）。例: gLn9ng-Cr455W-iNnLLN",
    )
    parser.add_argument(
        "folder",
        nargs="?",
        default="card_images",
        metavar="folder",
        help="画像フォルダ（デフォルト: card_images）。テスト時は例: card_images/example",
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
    parser.add_argument(
        "--images-only",
        action="store_true",
        help="スクレイピング・ダウンロードをスキップし、指定フォルダ内の画像だけを Vision で解析する（デッキコード不要）",
    )
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="既存スキップをせず全画像を解析し、同じ id の既存カードがあれば上書きする（ルナトーン・ハリテヤマなど再解析用）",
    )
    args = parser.parse_args()

    images_only = getattr(args, "images_only", False)
    deck_code = (args.deck_code or "").strip()

    if images_only:
        if deck_code and ("/" in deck_code or Path(deck_code).is_dir()):
            folder = Path(deck_code)
        else:
            folder = Path("card_images/example")
    else:
        folder = Path(args.folder)

    if not images_only and not deck_code:
        print("エラー: デッキコードを引数に指定してください。例: python read_cards.py gLn9ng-Cr455W-iNnLLN", file=sys.stderr)
        sys.exit(1)

    if folder.exists() and not folder.is_dir():
        print(f"エラー: ディレクトリではありません: {folder}", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output) if args.output else DEFAULT_OUTPUT_FILE

    if images_only:
        deck_lists = []
        if not folder.exists():
            print(f"エラー: フォルダが存在しません: {folder}", file=sys.stderr)
            sys.exit(1)
    else:
        deck_lists = _fetch_deck_and_download(deck_code, folder, args.quiet)

    if not folder.exists():
        if deck_lists:
            _save_deck_lists_only(deck_lists, output_path, args.quiet)
            return
        print(f"エラー: フォルダが存在しません: {folder}", file=sys.stderr)
        sys.exit(1)

    paths = list_image_paths(folder)
    if not paths:
        if deck_lists:
            _save_deck_lists_only(deck_lists, output_path, args.quiet, message="画像なし。デッキ一覧のみ保存しました")
            return
        print(f"エラー: 画像ファイルがありません: {folder}", file=sys.stderr)
        sys.exit(1)

    if not args.quiet:
        print(f"画像を {len(paths)} 件検出しました。", file=sys.stderr)

    existing_cards, existing_deck_lists = _load_existing_result(output_path)
    final_deck_lists = _merge_deck_lists(existing_deck_lists, deck_lists)
    replace_existing = getattr(args, "replace_existing", False)
    existing_after, to_add, replaced_count = _process_new_images(
        paths, existing_cards, args.quiet, replace_existing=replace_existing
    )
    _save_full_result(existing_after, to_add, final_deck_lists, output_path, args.quiet, replaced=replaced_count)


if __name__ == "__main__":
    main()
