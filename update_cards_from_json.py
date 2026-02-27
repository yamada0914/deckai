#!/usr/bin/env python3
"""
read_cards_result.json を読み取り、card/data.py の該当ブロックを追加・更新する。

やっていること:
  1. read_cards_result.json（read_cards.py で画像から取り出したカード情報）を読む
  2. 各カードを PokemonCard(...) と Attack(...) の Python コードに変換
  3. card/data.py 内の「JSON から生成」マーカーで囲まれたブロックを、この内容で置き換える

使い方:
  python update_cards_from_json.py              # data.py を直接更新（既定）
  python update_cards_from_json.py -o out.py    # ファイルにのみ出力（data.py は触らない）
  python update_cards_from_json.py --dry-run    # 更新内容を表示するが data.py は書き換えない
"""
import argparse
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_JSON_PATH = PROJECT_ROOT / "read_cards_result.json"
DATA_PY_PATH = PROJECT_ROOT / "card" / "data.py"

CARD_BLOCK_START = "# ----- 以下 JSON から生成（update_cards_from_json.py で上書き） -----"
CARD_BLOCK_END = "# ----- 以上 JSON から生成 -----"
REGISTRY_MARKER_START = "    # ----- JSON から生成 -----"
REGISTRY_MARKER_END = "    # ----- 以上 JSON から生成 -----"


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


def id_to_var_name(card_id: str) -> str:
    """カード id（例: karamingo-svg-029）を変数名（KARAMINGO_SVG_029）に変換。"""
    return card_id.replace("-", "_").upper()


def py_str(s: str | None) -> str:
    """文字列を Python のリテラルとしてエスケープ。"""
    if s is None:
        return "None"
    return repr(s)


def py_list_str(arr: list | None) -> str:
    """文字列リストを Python ソースに。"""
    if arr is None:
        return "None"
    return "[" + ", ".join(repr(x) for x in arr) + "]"


def attack_from_json(a: dict) -> str:
    """1 つの技を Attack(...) のソース文字列に。JSON の明示フィールドを優先し、なければ説明から status_effect / コイン技を推測する。"""
    name = a.get("name", "")
    energy_cost = a.get("energy_cost", 0)
    damage = a.get("damage", 0)
    self_damage = a.get("self_damage", 0)
    bench_damage = a.get("bench_damage", 0)
    description = a.get("description", "") or ""
    cost_typed = a.get("energy_cost_typed")
    coin_flips = a.get("coin_flips", 0) or 0
    damage_per_coin = a.get("damage_per_coin", 0) or 0
    if not (coin_flips > 0 and damage_per_coin > 0) and description and "コイン" in description and "オモテ" in description and "×" in description:
        m_n = re.search(r"コインを?(\d+)回", description)
        m_d = re.search(r"×(\d+)", description)
        if m_n and m_d:
            coin_flips = int(m_n.group(1))
            damage_per_coin = int(m_d.group(1))
    if coin_flips > 0 and damage_per_coin > 0:
        damage = 0
    status_effect = a.get("status_effect")
    status_effect_target = a.get("status_effect_target")
    if not status_effect and description and "こんらん" in description and "このポケモン" in description:
        status_effect = "confusion"
        status_effect_target = "self"
    if not status_effect and description and "マヒ" in description:
        status_effect = "paralysis"
    status_effect_on_coin_heads = a.get("status_effect_on_coin_heads", False)
    if not status_effect_on_coin_heads and status_effect and description and "コイン" in description and "オモテなら" in description:
        status_effect_on_coin_heads = True
    bench_damage_count = a.get("bench_damage_count", 1)
    bench_damage_target = a.get("bench_damage_target", "opponent")
    if bench_damage_target == "opponent" and description and "自分のベンチ" in description:
        bench_damage_target = "self"
    if bench_damage_count == 1 and bench_damage_target == "self" and description and "全員" in description:
        bench_damage_count = 0
    parts = [
        py_str(name),
        str(energy_cost),
        str(damage),
        str(self_damage),
        str(bench_damage),
        py_str(description),
    ]
    if bench_damage_count != 1:
        parts.append(f"bench_damage_count={bench_damage_count}")
    if bench_damage_target != "opponent":
        parts.append(f"bench_damage_target={py_str(bench_damage_target)}")
    if cost_typed is not None:
        parts.append(f"energy_cost_typed={py_list_str(cost_typed)}")
    if coin_flips > 0 and damage_per_coin > 0:
        parts.append(f"coin_flips={coin_flips}")
        parts.append(f"damage_per_coin={damage_per_coin}")
    if status_effect:
        parts.append(f"status_effect={py_str(status_effect)}")
        if status_effect_target:
            parts.append(f"status_effect_target={py_str(status_effect_target)}")
        if status_effect_on_coin_heads:
            parts.append("status_effect_on_coin_heads=True")
    return "Attack(" + ", ".join(parts) + ")"


def _pokemon_card_from_json(card: dict) -> str:
    """ポケモンカードを PokemonCard(...) のソース文字列に。"""
    card_id = card.get("id", "unknown")
    name_ja = card.get("name_ja", "")
    hp = card.get("hp", 60)
    attacks = card.get("attacks", [])
    evolves_from = card.get("evolves_from")
    evolution_stage = card.get("evolution_stage")  # basic / stage1 / stage2（画像認識で設定）
    if evolution_stage not in ("basic", "stage1", "stage2"):
        # JSON に無い場合は evolves_from が無ければたねとみなす
        evolution_stage = "basic" if not evolves_from else None
    retreat_cost = card.get("retreat_cost", 1)
    pokemon_type = card.get("pokemon_type")
    weakness = card.get("weakness")
    resistance = card.get("resistance")
    lines = [
        f"# {name_ja}（{card_id}）",
        f"{id_to_var_name(card_id)} = PokemonCard(",
        f"    id={py_str(card_id)},",
        f"    name={py_str(name_ja)},",
        f"    hp={hp},",
        f"    max_hp={hp},",
        f"    attacks=[",
    ]
    for a in attacks:
        lines.append("        " + attack_from_json(a) + ",")
    lines.append("    ],")
    lines.append(f"    evolves_from={py_str(evolves_from)},")
    if evolution_stage is not None:
        lines.append(f"    evolution_stage={py_str(evolution_stage)},")
    lines.append(f"    retreat_cost={retreat_cost},")
    lines.append(f"    pokemon_type={py_str(pokemon_type)},")
    if weakness is not None:
        lines.append(f"    weakness={py_str(weakness)},")
    if resistance is not None:
        lines.append(f"    resistance={py_str(resistance)},")
    is_ex = card.get("is_ex") if "is_ex" in card else ("ex" in (name_ja or ""))
    is_mega = card.get("is_mega") if "is_mega" in card else ("メガ" in (name_ja or ""))
    if is_ex:
        lines.append("    is_ex=True,")
    if is_mega:
        lines.append("    is_mega=True,")
    lines.append(")")
    return "\n".join(lines)


def _goods_card_from_json(card: dict) -> str:
    """グッズカードを GoodsCard(...) のソース文字列に。"""
    card_id = card.get("id", "unknown")
    name_ja = card.get("name_ja", "")
    description = card.get("description", "") or ""
    return "\n".join([
        f"# {name_ja}（{card_id}）",
        f"{id_to_var_name(card_id)} = GoodsCard(",
        f"    id={py_str(card_id)},",
        f"    name={py_str(name_ja)},",
        f"    description={py_str(description)},",
        ")",
    ])


def _tool_card_from_json(card: dict) -> str:
    """ポケモンのどうぐを GoodsCard(..., is_tool=True, ...) のソース文字列に。"""
    card_id = card.get("id", "unknown")
    name_ja = card.get("name_ja", "")
    description = card.get("description", "") or ""
    reduce_val = card.get("tool_damage_reduce")
    cond_type = card.get("tool_condition_type")
    if reduce_val is None and name_ja == "岩のむねあて":
        reduce_val, cond_type = 30, "fighting"
    lines = [
        f"# {name_ja}（{card_id}）",
        f"{id_to_var_name(card_id)} = GoodsCard(",
        f"    id={py_str(card_id)},",
        f"    name={py_str(name_ja)},",
        f"    effect={py_str('tool')},",
        f"    description={py_str(description)},",
        "    is_tool=True,",
        f"    tool_damage_reduce={reduce_val if reduce_val is not None else 0},",
        f"    tool_condition_type={py_str(cond_type)},",
        ")",
    ]
    return "\n".join(lines)


def _support_card_from_json(card: dict) -> str:
    """サポートカードを SupportCard(...) のソース文字列に。"""
    card_id = card.get("id", "unknown")
    name_ja = card.get("name_ja", "")
    description = card.get("description", "") or ""
    return "\n".join([
        f"# {name_ja}（{card_id}）",
        f"{id_to_var_name(card_id)} = SupportCard(",
        f"    id={py_str(card_id)},",
        f"    name={py_str(name_ja)},",
        f"    description={py_str(description)},",
        ")",
    ])


def _energy_card_from_json(card: dict) -> str:
    """エネルギーカードを EnergyCard(...) のソース文字列に。"""
    card_id = card.get("id", "unknown")
    name_ja = card.get("name_ja", "")
    energy_type = card.get("energy_type")
    return "\n".join([
        f"# {name_ja}（{card_id}）",
        f"{id_to_var_name(card_id)} = EnergyCard(",
        f"    id={py_str(card_id)},",
        f"    name={py_str(name_ja)},",
        f"    energy_type={py_str(energy_type)},",
        ")",
    ])


def card_from_json(card: dict) -> str:
    """card_type に応じて PokemonCard / GoodsCard / SupportCard / EnergyCard のソース文字列に。"""
    card_type = (card.get("card_type") or "pokemon").strip().lower()
    if card_type in ("goods", "item"):  # "item" は旧形式の互換
        return _goods_card_from_json(card)
    if card_type == "tool":
        return _tool_card_from_json(card)
    if card_type == "support":
        return _support_card_from_json(card)
    if card_type == "energy":
        return _energy_card_from_json(card)
    return _pokemon_card_from_json(card)


def generate_registry_entries(cards: list[dict]) -> tuple[str, str]:
    """CARD_ID_TO_NAME と _CARD_REGISTRY のエントリ行を生成。"""
    name_lines = []
    reg_lines = []
    for card in cards:
        cid = card.get("id")
        if not cid:
            continue
        name_ja = card.get("name_ja", cid)
        var = id_to_var_name(cid)
        name_lines.append(f'    "{cid}": {py_str(name_ja)},')
        reg_lines.append(f'    "{cid}": {var},')
    return "\n".join(name_lines), "\n".join(reg_lines)


def update_data_py(cards: list[dict], dry_run: bool = False) -> str:
    """
    card/data.py のマーカーで囲まれた 3 ブロックを、JSON から生成した内容で置き換える。
    戻り値: 更新後の data.py 全文。dry_run 時も同じ内容を返す（書き込みはしない）。
    """
    if not DATA_PY_PATH.is_file():
        raise SystemExit(f"card/data.py が見つかりません: {DATA_PY_PATH}")

    body_lines = []
    for card in cards:
        body_lines.append(card_from_json(card))
        body_lines.append("")
    cards_block = "\n".join(body_lines)

    name_entries, reg_entries = generate_registry_entries(cards)
    name_block = name_entries
    reg_block = reg_entries

    text = DATA_PY_PATH.read_text(encoding="utf-8")

    card_pattern = re.escape(CARD_BLOCK_START) + r"\n.*?" + re.escape(CARD_BLOCK_END)
    card_repl = f"{CARD_BLOCK_START}\n{cards_block}\n{CARD_BLOCK_END}"
    if not re.search(card_pattern, text, re.DOTALL):
        raise SystemExit("card/data.py にカード用マーカーが見つかりません。")
    text = re.sub(card_pattern, card_repl, text, count=1, flags=re.DOTALL)

    name_pattern = re.escape(REGISTRY_MARKER_START) + r"\n(.*?)" + re.escape(REGISTRY_MARKER_END)
    name_repl = f"{REGISTRY_MARKER_START}\n{name_block}\n{REGISTRY_MARKER_END}"
    if not re.search(name_pattern, text, re.DOTALL):
        raise SystemExit("card/data.py の CARD_ID_TO_NAME にマーカーが見つかりません。")
    text = re.sub(name_pattern, name_repl, text, count=1, flags=re.DOTALL)

    reg_pattern = re.escape(REGISTRY_MARKER_START) + r"\n(.*?)" + re.escape(REGISTRY_MARKER_END)
    reg_repl = f"{REGISTRY_MARKER_START}\n{reg_block}\n{REGISTRY_MARKER_END}"
    matches = list(re.finditer(reg_pattern, text, re.DOTALL))
    if len(matches) < 2:
        raise SystemExit("card/data.py の _CARD_REGISTRY にマーカーが見つかりません。")
    text = text[: matches[1].start()] + reg_repl + text[matches[1].end() :]

    if not dry_run:
        DATA_PY_PATH.write_text(text, encoding="utf-8")
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="read_cards_result.json で card/data.py を更新")
    parser.add_argument("--json", "-j", type=Path, default=DEFAULT_JSON_PATH, help="入力 JSON パス")
    parser.add_argument("-o", "--output", type=Path, default=None, help="このファイルにのみ出力（data.py は更新しない）")
    parser.add_argument("--dry-run", action="store_true", help="更新内容を表示するが data.py は書き換えない")
    args = parser.parse_args()

    path = args.json
    if not path.is_file():
        raise SystemExit(f"ファイルが見つかりません: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    card_files = data.get("card_files")
    if card_files and isinstance(card_files, dict):
        cards = _load_cards_from_split_files(path.parent, card_files)
    else:
        raw_cards = data.get("cards", [])
        if isinstance(raw_cards, dict):
            cards = [c for k in ("pokemon", "goods", "support", "energy") for c in (raw_cards.get(k) or [])]
        else:
            cards = raw_cards or []
    if not cards:
        raise SystemExit("cards が空です")

    if args.output is not None:
        out_lines = [
            '"""',
            "read_cards_result.json から生成。",
            '"""',
            "from card.model import Attack, EnergyCard, GoodsCard, PokemonCard, SupportCard",
            "",
        ]
        for card in cards:
            out_lines.append(card_from_json(card))
            out_lines.append("")
        name_entries, reg_entries = generate_registry_entries(cards)
        out_lines.append("# ----- CARD_ID_TO_NAME 用 -----")
        out_lines.append(name_entries)
        out_lines.append("")
        out_lines.append("# ----- _CARD_REGISTRY 用 -----")
        out_lines.append(reg_entries)
        args.output.write_text("\n".join(out_lines), encoding="utf-8")
        print(f"出力しました: {args.output}", file=__import__("sys").stderr)
        return

    update_data_py(cards, dry_run=args.dry_run)
    if args.dry_run:
        print("--dry-run のため card/data.py は書き換えていません。", file=__import__("sys").stderr)
    else:
        print(f"更新しました: {DATA_PY_PATH}", file=__import__("sys").stderr)


if __name__ == "__main__":
    main()
