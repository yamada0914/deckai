"""
デッキ定義（レシピ・デッキ生成・表示）。

デッキ A/B/C/D/E のレシピ、create_deck / format_deck_recipe を提供。
デッキコードで取得した一覧（read_cards_result.json の deck_lists）からもデッキを生成できる。
"""
import json
from pathlib import Path

from card import CARD_ID_TO_NAME, get_card_by_id, get_trainer_id_by_name

_PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_READ_CARDS_JSON = _PROJECT_ROOT / "read_cards_result.json"

_recipe_cache: dict[tuple[str, str], dict[str, int]] = {}


def _load_cards_from_split_files(base_dir: Path, card_files: dict) -> list:
    """card_files で指定された JSON を読み、1 つの配列にマージして返す。"""
    order = ("pokemon", "trainers", "energy")
    result: list = []
    for key in order:
        name = card_files.get(key)
        if not name:
            continue
        p = base_dir / name
        if not p.is_file():
            continue
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
            items = raw if isinstance(raw, list) else raw.get("cards", raw.get("items", []))
            if isinstance(items, list):
                result.extend(items)
        except (json.JSONDecodeError, OSError):
            pass
    return result


def clear_recipe_cache() -> None:
    """load_recipe_from_deck_code のキャッシュを空にする。read_cards_result.json を更新したあとで再読み込みしたいときに呼ぶ。"""
    _recipe_cache.clear()

DECK_RECIPE_A = {"otachi": 3, "ootachi": 2, "mototokage": 2, "basic-energy": 6, "nemo": 1}
DECK_RECIPE_B = {"meguroko": 3, "warubiru": 2, "mototokage": 2, "basic-energy": 6, "potion": 1}
DECK_RECIPE_C = {"zupika-svd-041": 3, "harabarii-svd-042": 2, "mototokage": 2, "basic-energy": 4, "basic-energy-lightning": 2, "potion": 1, "nemo": 4}
DECK_RECIPE_D = {"meguroko-svd-062": 3, "warubiru-svd-063": 2, "warubiaru-svd-064": 2, "rikukurage-svd-066": 2, "nonokurage-svd-065": 3, "gakegani-svd-067": 2, "basic-energy-fighting": 8, "nemo": 4, "potion": 2, "pokemon_irekae": 2}
DECK_RECIPE_E = {"jibakoil-svd-038": 2, "rarecoil-svd-037": 2, "coil-svd-036": 3, "zupika-svd-041": 3, "harabarii-svd-042": 2, "karamingo-svg-029": 2, "basic-energy-lightning": 8, "nemo": 4, "potion": 2, "pokemon_irekae": 2}

DECK_RECIPES = [DECK_RECIPE_A, DECK_RECIPE_B, DECK_RECIPE_C, DECK_RECIPE_D, DECK_RECIPE_E]
DECK_SIZE = 13
STARTING_HAND_SIZE = 4


def format_deck_recipe(deck_index: int) -> str:
    """デッキ番号（0=A, 1=B, 2=C, 3=D, 4=E）のレシピを「カード名枚数, ...」の文字列で返す。"""
    recipe = DECK_RECIPES[deck_index % len(DECK_RECIPES)]
    names = [f"{CARD_ID_TO_NAME.get(cid, cid)}{n}" for cid, n in recipe.items()]
    return "、".join(names)


def create_deck(deck_index: int = 0) -> list:
    """指定デッキを生成（0=A, 1=B, 2=C, 3=D, 4=E）。各カードにユニーク instance_id を付与。"""
    recipe = DECK_RECIPES[deck_index % len(DECK_RECIPES)]
    return create_deck_from_recipe(recipe)


def create_deck_from_recipe(recipe: dict) -> list:
    """レシピ { card_id: 枚数 } からデッキを生成。各カードにユニーク instance_id を付与。"""
    deck = []
    uid = 0
    for card_id, count in recipe.items():
        for _ in range(count):
            deck.append(get_card_by_id(card_id, f"card-{uid}"))
            uid += 1
    return deck


def load_recipe_from_deck_code(
    deck_code: str,
    json_path: Path | str | None = None,
) -> dict | None:
    """
    read_cards_result.json の deck_lists から指定 deck_code のデッキを探し、
    カード名（name_ja）を cards の id に解決してレシピ { card_id: 枚数 } を返す。
    見つからないか解釈できない場合は None。同一 deck_code はキャッシュして 2 回目以降は JSON を読まない。
    """
    path = Path(json_path) if json_path else DEFAULT_READ_CARDS_JSON
    path_str = str(path.resolve())
    cache_key = (deck_code.strip(), path_str)
    if cache_key in _recipe_cache:
        return dict(_recipe_cache[cache_key])
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    deck_lists = data.get("deck_lists", [])
    card_files = data.get("card_files")
    if card_files and isinstance(card_files, dict):
        cards = _load_cards_from_split_files(path.parent, card_files)
    else:
        raw_cards = data.get("cards", [])
        if isinstance(raw_cards, dict):
            cards = [c for k in ("pokemon", "goods", "support", "energy") for c in (raw_cards.get(k) or [])]
        else:
            cards = raw_cards
    name_to_id: dict[str, str] = {}
    for c in cards:
        name = (c.get("name_ja") or "").strip()
        cid = c.get("id")
        if not name:
            continue
        ctype = (c.get("card_type") or "pokemon").strip().lower()
        if ctype in ("goods", "support", "tool"):
            cid = get_trainer_id_by_name(name) or cid
        if cid and name not in name_to_id:
            name_to_id[name] = cid
    deck_item = None
    for d in deck_lists:
        if (d.get("deck_code") or "").strip() == deck_code.strip():
            deck_item = d
            break
    if not deck_item:
        return None
    recipe: dict[str, int] = {}
    for item in deck_item.get("cards", []):
        name = (item.get("name_ja") or "").strip()
        count = int(item.get("count") or 0)
        if not name or count <= 0:
            continue
        cid = name_to_id.get(name)
        if not cid:
            continue
        recipe[cid] = recipe.get(cid, 0) + count
    if recipe:
        _recipe_cache[cache_key] = recipe
    return recipe if recipe else None


def create_deck_from_deck_code(
    deck_code: str,
    json_path: Path | str | None = None,
) -> list | None:
    """
    デッキコードで取得した一覧（JSON の deck_lists）からデッキを生成する。
    レシピが取得できなければ None。取得できれば create_deck_from_recipe と同じ形式のリスト。
    """
    recipe = load_recipe_from_deck_code(deck_code, json_path)
    if not recipe:
        return None
    return create_deck_from_recipe(recipe)
