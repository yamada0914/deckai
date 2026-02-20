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

# プロジェクトルート
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_JSON_PATH = PROJECT_ROOT / "read_cards_result.json"
DATA_PY_PATH = PROJECT_ROOT / "card" / "data.py"

# data.py 内の置き換えブロック用マーカー（data.py に同じ文字列が入っていること）
CARD_BLOCK_START = "# ----- 以下 JSON から生成（update_cards_from_json.py で上書き） -----"
CARD_BLOCK_END = "# ----- 以上 JSON から生成 -----"
REGISTRY_MARKER_START = "    # ----- JSON から生成 -----"
REGISTRY_MARKER_END = "    # ----- 以上 JSON から生成 -----"


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
    """1 つの技を Attack(...) のソース文字列に。説明から status_effect / コイン技を推測して付与する。"""
    name = a.get("name", "")
    energy_cost = a.get("energy_cost", 0)
    damage = a.get("damage", 0)
    self_damage = a.get("self_damage", 0)
    bench_damage = a.get("bench_damage", 0)
    description = a.get("description", "") or ""
    cost_typed = a.get("energy_cost_typed")
    # 説明からコイン技を推測（「コインを2回投げ、オモテの数×100ダメージ」など）
    coin_flips = 0
    damage_per_coin = 0
    if description and "コイン" in description and "オモテ" in description and "×" in description:
        m_n = re.search(r"コインを?(\d+)回", description)
        m_d = re.search(r"×(\d+)", description)
        if m_n and m_d:
            coin_flips = int(m_n.group(1))
            damage_per_coin = int(m_d.group(1))
            damage = 0  # コイン技のときはダメージはコインで決まる
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
    if coin_flips > 0 and damage_per_coin > 0:
        parts.append(f"coin_flips={coin_flips}")
        parts.append(f"damage_per_coin={damage_per_coin}")
    # 説明から状態異常を推測（「このポケモンをこんらんにする」など）
    if description and "こんらん" in description and "このポケモン" in description:
        parts.append("status_effect='confusion'")
        parts.append("status_effect_target='self'")
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

    # カード定義ブロック（開始・終了マーカーを含めて置換）
    card_pattern = re.escape(CARD_BLOCK_START) + r"\n.*?" + re.escape(CARD_BLOCK_END)
    card_repl = f"{CARD_BLOCK_START}\n{cards_block}\n{CARD_BLOCK_END}"
    if not re.search(card_pattern, text, re.DOTALL):
        raise SystemExit("card/data.py にカード用マーカーが見つかりません。")
    text = re.sub(card_pattern, card_repl, text, count=1, flags=re.DOTALL)

    # CARD_ID_TO_NAME 内の JSON ブロック（マーカー間の行だけ置換）
    name_pattern = re.escape(REGISTRY_MARKER_START) + r"\n(.*?)" + re.escape(REGISTRY_MARKER_END)
    name_repl = f"{REGISTRY_MARKER_START}\n{name_block}\n{REGISTRY_MARKER_END}"
    if not re.search(name_pattern, text, re.DOTALL):
        raise SystemExit("card/data.py の CARD_ID_TO_NAME にマーカーが見つかりません。")
    text = re.sub(name_pattern, name_repl, text, count=1, flags=re.DOTALL)

    # _CARD_REGISTRY 内の JSON ブロック
    reg_pattern = re.escape(REGISTRY_MARKER_START) + r"\n(.*?)" + re.escape(REGISTRY_MARKER_END)
    reg_repl = f"{REGISTRY_MARKER_START}\n{reg_block}\n{REGISTRY_MARKER_END}"
    # 2 つ目のマーカーペア（_CARD_REGISTRY 内）を置換
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
    cards = data.get("cards", [])
    if not cards:
        raise SystemExit("cards が空です")

    if args.output is not None:
        # ファイルに出力するだけ（従来の挙動）
        out_lines = [
            '"""',
            "read_cards_result.json から生成。",
            '"""',
            "from card.model import Attack, PokemonCard",
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

    # data.py を直接更新
    update_data_py(cards, dry_run=args.dry_run)
    if args.dry_run:
        print("--dry-run のため card/data.py は書き換えていません。", file=__import__("sys").stderr)
    else:
        print(f"更新しました: {DATA_PY_PATH}", file=__import__("sys").stderr)


if __name__ == "__main__":
    main()
