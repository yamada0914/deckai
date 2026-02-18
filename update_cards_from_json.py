#!/usr/bin/env python3
"""
read_cards_result.json を読み取り、card/data.py 用の PokemonCard 定義（Python ソース）を生成する。

使い方:
  python update_cards_from_json.py                    # 標準出力に生成
  python update_cards_from_json.py -o generated.py    # ファイルに出力
  python update_cards_from_json.py --json path.json   # 入力 JSON を指定

生成されたコードは card/data.py に手で貼るか、インポートしてレジストリに追加する。
"""
import argparse
import json
import re
from pathlib import Path

# プロジェクトルート
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_JSON_PATH = PROJECT_ROOT / "read_cards_result.json"


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
    """1 つの技を Attack(...) のソース文字列に。"""
    name = a.get("name", "")
    energy_cost = a.get("energy_cost", 0)
    damage = a.get("damage", 0)
    self_damage = a.get("self_damage", 0)
    bench_damage = a.get("bench_damage", 0)
    description = a.get("description", "") or ""
    cost_typed = a.get("energy_cost_typed")
    parts = [
        py_str(name),
        str(energy_cost),
        str(damage),
        str(self_damage),
        str(bench_damage),
        py_str(description),
    ]
    if cost_typed is not None:
        parts.append(f"energy_cost_typed={py_list_str(cost_typed)}")
    return "Attack(" + ", ".join(parts) + ")"


def card_from_json(card: dict) -> str:
    """1 枚のカードを PokemonCard(...) のソース文字列に。"""
    card_id = card.get("id", "unknown")
    name_ja = card.get("name_ja", "")
    hp = card.get("hp", 60)
    attacks = card.get("attacks", [])
    evolves_from = card.get("evolves_from")
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
    lines.append(f"    retreat_cost={retreat_cost},")
    lines.append(f"    pokemon_type={py_str(pokemon_type)},")
    if weakness is not None:
        lines.append(f"    weakness={py_str(weakness)},")
    if resistance is not None:
        lines.append(f"    resistance={py_str(resistance)},")
    lines.append(")")
    return "\n".join(lines)


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


def main() -> None:
    parser = argparse.ArgumentParser(description="read_cards_result.json から card/data.py 用のコードを生成")
    parser.add_argument("--json", "-j", type=Path, default=DEFAULT_JSON_PATH, help="入力 JSON パス")
    parser.add_argument("-o", "--output", type=Path, default=None, help="出力先ファイル（未指定なら標準出力）")
    parser.add_argument("--registry", action="store_true", help="CARD_ID_TO_NAME と _CARD_REGISTRY の追記用も出力")
    args = parser.parse_args()

    path = args.json
    if not path.is_file():
        raise SystemExit(f"ファイルが見つかりません: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    cards = data.get("cards", [])
    if not cards:
        raise SystemExit("cards が空です")

    out_lines = [
        '"""',
        "read_cards_result.json から生成。update_cards_from_json.py で再生成できます。",
        "card/data.py に貼るか、このファイルを data から import してレジストリに追加してください。",
        '"""',
        "from card.model import Attack, PokemonCard",
        "",
    ]
    for card in cards:
        out_lines.append(card_from_json(card))
        out_lines.append("")

    if args.registry:
        name_entries, reg_entries = generate_registry_entries(cards)
        out_lines.append("# ----- 以下を CARD_ID_TO_NAME に追記 -----")
        out_lines.append(name_entries)
        out_lines.append("")
        out_lines.append("# ----- 以下を _CARD_REGISTRY に追記 -----")
        out_lines.append(reg_entries)

    text = "\n".join(out_lines)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
        print(f"出力しました: {args.output}", file=__import__("sys").stderr)
    else:
        print(text)


if __name__ == "__main__":
    main()
