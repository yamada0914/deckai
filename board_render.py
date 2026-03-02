"""
盤面を 1 枚の画像として描画する。

GameState を受け取り、上下で向き合う形（上: 相手側 / 下: 自分側）で
中央に横線、サイド・ベンチ・バトル場・山札・手札（カード画像）を描画する。
"""
from pathlib import Path
import json
import textwrap

from PIL import Image, ImageDraw, ImageFont

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from game import GameState

_PROJECT_ROOT = Path(__file__).resolve().parent
_READ_CARDS_DATA = _PROJECT_ROOT / "read_cards_data"
CARD_BACK_IMAGE = _PROJECT_ROOT / "card_images" / "basic" / "20230809083754-0.jpeg"
_CARD_IMAGE_FILES: dict[str, str] | None = None


def _load_card_id_to_image() -> dict[str, str]:
    """read_cards_data の JSON から card id → _source_image のマッピングを構築する。"""
    global _CARD_IMAGE_FILES
    if _CARD_IMAGE_FILES is not None:
        return _CARD_IMAGE_FILES
    out: dict[str, str] = {}
    for name in ("pokemon.json", "trainers.json", "energy.json"):
        path = _READ_CARDS_DATA / name
        if not path.is_file():
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            items = raw if isinstance(raw, list) else raw.get("cards", raw.get("items", []))
            if not isinstance(items, list):
                continue
            for c in items:
                cid = c.get("id")
                src = c.get("_source_image")
                if cid and src:
                    out[cid] = src
        except (json.JSONDecodeError, OSError):
            continue
    if "kihontouenerugi" in out:
        out.setdefault("basic-energy-fighting", out["kihontouenerugi"])
    if "kihonkaminarienerugi" in out:
        out.setdefault("basic-energy-lightning", out["kihonkaminarienerugi"])
    if "basic-energy-fighting" in out:
        out.setdefault("basic-energy", out["basic-energy-fighting"])
    if "nemokako" in out:
        out.setdefault("nemo", out["nemokako"])
    if "pokemonirekae" in out:
        out.setdefault("pokemon_irekae", out["pokemonirekae"])
    if "karamingo-svd-109" in out:
        out.setdefault("karamingo-svg-029", out["karamingo-svd-109"])
    _CARD_IMAGE_FILES = out
    return out


def get_card_image_path(card_id: str, images_dir: Path | str) -> Path | None:
    """
    カード ID に対応する画像ファイルの Path を返す。
    見つからなければ None。images_dir は card_images など画像フォルダのパス。
    """
    mapping = _load_card_id_to_image()
    filename = mapping.get(card_id) if card_id else None
    if not filename:
        return None
    folder = Path(images_dir)
    p = folder / filename
    return p if p.is_file() else None


BOARD_WIDTH = 900
BOARD_HEIGHT = 980
PRIZE_SLOTS = 6
BENCH_SLOTS = 5
MARGIN = 24
PRIZE_TO_BENCH = 12
BENCH_GAP = 10
TOOL_OFFSET = 8
TOOL_VERTICAL_OVERHANG = 18
CARD_W_BENCH = 90
CARD_H_BENCH = 124
MIN_HAND_CARD_W = 20
LOG_PANEL_WIDTH = 350
LOG_LINE_HEIGHT = 20
LOG_FONT_SIZE = 14
BG_COLOR = (30, 80, 30)
SLOT_COLOR = (60, 100, 60)
CARD_BACK_COLOR = (80, 60, 40)
TEXT_COLOR = (255, 255, 255)
LABEL_COLOR = (240, 240, 200)

_STATUS_MARKER_FILES = {
    "poison": _PROJECT_ROOT / "card_images" / "basic" / "pk-svb-032.webp",
    "burn": _PROJECT_ROOT / "card_images" / "basic" / "38559.jpg",
}
_STATUS_MARKER_CACHE: dict[str, Image.Image] | None = None


def _simplify_log_text_for_panel(log_text: str | None) -> str:
    """
    盤面左側のログパネル用に、詳細なログ文字列から簡易表示用テキストを生成する。
    - プレイヤー名などのプレフィックス「XX: 」を削除
    - 「→」「（」以降の細かい情報（手札枚数、HP の推移など）はカット
    - 先頭・区切り線（==========, ----------）はそのまま残す
    """
    if not log_text:
        return ""
    lines = [ln.rstrip() for ln in log_text.split("\n") if ln.strip()]
    simplified: list[str] = []
    for ln in lines:
        stripped = ln.strip()
        if not stripped:
            continue
        if stripped.startswith("==========") or stripped.startswith("----------"):
            simplified.append(stripped)
            continue
        body = stripped.split(": ", 1)[1] if ": " in stripped else stripped
        for sep in ("→", "（"):
            if sep in body:
                body = body.split(sep, 1)[0].rstrip()
        simplified.append(body)
    return "\n".join(simplified)


def _draw_placeholder(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int, label: str = "") -> None:
    """カードがないスロット用の角丸四角とテキストを描画する。"""
    draw.rounded_rectangle([x, y, x + w, y + h], radius=8, fill=SLOT_COLOR, outline=(100, 140, 100))
    if label:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc", 14)
        except OSError:
            font = ImageFont.load_default()
        tw = draw.textlength(label, font=font)
        if tw > w - 8:
            label = label[: min(len(label), 8)] + "…"
        draw.text((x + (w - draw.textlength(label, font=font)) // 2, y + (h - 16) // 2), label, fill=TEXT_COLOR, font=font)


def _paste_card_image(bg: Image.Image, card_path: Path | None, x: int, y: int, w: int, h: int, name: str) -> None:
    """カード画像をリサイズして bg に貼り付ける。なければプレースホルダーを描画。"""
    draw = ImageDraw.Draw(bg)
    if card_path and card_path.is_file():
        try:
            img = Image.open(card_path).convert("RGB")
            img = img.resize((w, h), Image.Resampling.LANCZOS)
            bg.paste(img, (x, y))
        except Exception:
            _draw_placeholder(draw, x, y, w, h, name)
    else:
        _draw_placeholder(draw, x, y, w, h, name)


def _load_status_markers() -> dict[str, Image.Image]:
    """どく・やけど用のマーカー画像を読み込んでキャッシュする。"""
    global _STATUS_MARKER_CACHE
    if _STATUS_MARKER_CACHE is not None:
        return _STATUS_MARKER_CACHE
    out: dict[str, Image.Image] = {}
    for key, path in _STATUS_MARKER_FILES.items():
        try:
            if path.is_file():
                img = Image.open(path).convert("RGBA")
                out[key] = img
        except Exception:
            continue
    _STATUS_MARKER_CACHE = out
    return out


def _draw_status_markers(
    bg: Image.Image,
    bp: object,
    x: int,
    y: int,
    w: int,
    h: int,
) -> None:
    """
    どく・やけど状態のときに、カード上にマーカーを重ねて表示する。
    - どく: 紫マーカー
    - やけど: オレンジマーカー
    """
    markers_raw = _load_status_markers()
    has_poison = getattr(bp, "poison_damage", 0) > 0
    has_burn = getattr(bp, "burn", False)
    keys: list[str] = []
    if has_poison and "poison" in markers_raw:
        keys.append("poison")
    if has_burn and "burn" in markers_raw:
        keys.append("burn")
    if not keys:
        return

    size = max(14, min(28, w // 3))
    pad = 4
    for idx, key in enumerate(keys):
        icon = markers_raw.get(key)
        if not icon:
            continue
        iw, ih = icon.size
        side = min(iw, ih)
        left = (iw - side) // 2
        top = (ih - side) // 2
        cropped = icon.crop((left, top, left + side, top + side))
        resized = cropped.resize((size, size), Image.Resampling.LANCZOS)
        px = x + pad + idx * (size + 2)
        py = y + h - size - pad
        bg.paste(resized.convert("RGB"), (px, py), resized)


def _draw_log_panel(
    draw: ImageDraw.ImageDraw,
    log_text: str,
    panel_width: int,
    height: int,
) -> None:
    """左側にログテキストを描画する。収まらない行は末尾のみ表示。"""
    try:
        font = ImageFont.truetype("/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc", LOG_FONT_SIZE)
    except OSError:
        font = ImageFont.load_default()
    pad = 8
    max_lines = max(1, (height - pad * 2) // LOG_LINE_HEIGHT)
    lines = [ln.rstrip() for ln in (log_text or "").split("\n")]
    normalized: list[str] = []
    for ln in lines:
        if not ln.strip():
            if normalized and normalized[-1] != "":
                normalized.append("")
            continue
        normalized.append(ln)
    show_lines = normalized[-max_lines:] if len(normalized) > max_lines else normalized
    y = pad
    for ln in show_lines:
        if y + LOG_LINE_HEIGHT > height - pad:
            break
        draw.text((pad, y), ln[:28], fill=TEXT_COLOR, font=font)
        y += LOG_LINE_HEIGHT


def _draw_pokemon_with_tool(
    bg: Image.Image,
    draw: ImageDraw.ImageDraw,
    bp: object,
    x: int,
    y: int,
    w: int,
    h: int,
    images_dir: Path,
) -> None:
    """ポケモンカードを描画する。どうぐがついていればその上にはみ出すように重ねて表示する。

    状態異常に応じてカードの向きを変える（実カードの扱いに合わせた演出）:
    - ねむり / マヒ: カードを横向きにする
    - こんらん: カードを逆さまにする
    """
    tool = getattr(bp, "attached_tool", None)
    if tool:
        tool_path = get_card_image_path(getattr(tool, "id", ""), images_dir)
        tool_name = getattr(tool, "name", "どうぐ")
        _paste_card_image(bg, tool_path, x, y - TOOL_VERTICAL_OVERHANG, w, h, tool_name)

    extra_bottom = 8
    card_layer = Image.new("RGBA", (w, h + extra_bottom), (0, 0, 0, 0))
    card_draw = ImageDraw.Draw(card_layer)
    path = get_card_image_path(getattr(bp.card, "id", ""), images_dir)
    _paste_card_image(card_layer, path, 0, 0, w, h, bp.card.name)
    _draw_hp_bar(card_draw, 0, 0, w, h, bp.hp, bp.max_hp)
    _draw_energy_on_card(card_layer, card_draw, 0, 0, w, h, getattr(bp, "attached_energy_types", []), images_dir)
    _draw_status_markers(card_layer, bp, 0, 0, w, h)

    status = getattr(bp, "special_state", None)
    paste_img: Image.Image = card_layer
    px, py = x, y
    if status == "confusion":
        paste_img = card_layer.rotate(180, expand=True)
        rw, rh = paste_img.size
        cx = x + w // 2
        cy = y + h // 2
        px = cx - rw // 2
        py = cy - rh // 2
    elif status in ("sleep", "paralysis"):
        paste_img = card_layer.rotate(90, expand=True)
        rw, rh = paste_img.size
        cx = x + w // 2
        cy = y + h // 2
        px = cx - rw // 2
        py = cy - rh // 2

    bg.paste(paste_img.convert("RGB"), (px, py), paste_img)


def _draw_hp_bar(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int, hp: int, max_hp: int) -> None:
    """カード右上に HP を表示（グレー角丸枠・大きな数値・その下に緑/グレーの HP バー）。"""
    if max_hp <= 0:
        return
    ratio = max(0, min(1, hp / max_hp))
    pad = 6
    box_w = min(44, w // 2)
    num_h = 14 if w < 100 else 18
    bar_h = 5
    box_h = num_h + 2 + bar_h
    rx, ry = x + w - box_w - pad, y + pad
    draw.rounded_rectangle([rx, ry, rx + box_w, ry + box_h], radius=4, fill=(70, 70, 70), outline=(100, 100, 100))
    try:
        font = ImageFont.truetype("/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc", num_h)
    except OSError:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc", num_h)
        except OSError:
            font = ImageFont.load_default()
    num_text = str(hp)
    tw = draw.textlength(num_text, font=font)
    num_x = rx + (box_w - tw) // 2
    for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1), (0, -1), (0, 1), (-1, 0), (1, 0)]:
        draw.text((num_x + dx, ry + 1 + dy), num_text, fill=(40, 40, 60), font=font)
    draw.text((num_x, ry + 1), num_text, fill=TEXT_COLOR, font=font)
    bar_y = ry + num_h + 2
    bar_w = box_w - 4
    bar_x = rx + 2
    draw.rounded_rectangle([bar_x, bar_y, bar_x + bar_w, bar_y + bar_h], radius=2, fill=(60, 60, 60))
    if ratio > 0:
        draw.rounded_rectangle([bar_x, bar_y, bar_x + int(bar_w * ratio), bar_y + bar_h], radius=2, fill=(60, 200, 80))


_ENERGY_IMAGE_FILES = {
    "fighting": "fighting_energy_icatch-640x360.webp",
    "lightning": "ligtning_energy_icatch-640x360.webp",
    "fire": "fire_energy_icatch-640x360.webp",
    "water": "water_energy_energy_icatch-640x360.webp",
    "grass": "grass_energy_icatch-640x360.webp",
    "psychic": "psychic_energy_icatch-640x360.webp",
    "darkness": "darkness_energy_icatch-640x360.webp",
    "metal": "metal_energy_icatch-640x360.webp",
}

_ENERGY_FALLBACK_COLORS = {
    "fighting": (200, 80, 60),
    "lightning": (240, 220, 60),
    "fire": (220, 100, 50),
    "water": (60, 140, 220),
    "grass": (80, 180, 80),
    "psychic": (180, 80, 160),
    "darkness": (80, 60, 80),
    "metal": (160, 160, 180),
    "fairy": (240, 180, 200),
    "dragon": (120, 80, 180),
    "colorless": (200, 200, 200),
}


def _draw_energy_on_card(
    bg: Image.Image,
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    w: int,
    h: int,
    energy_types: list,
    images_dir: Path | str,
) -> None:
    """カード左下に付与エネルギーを画像（または色円のフォールバック）で表示。"""
    if not energy_types:
        return
    pad = 5
    r = 9 if w < 100 else 12
    gap = 4
    step = 2 * r + gap
    n_max = min(len(energy_types), max(1, (w - 2 * pad) // step))
    energy_types = energy_types[:n_max]
    cx = x + pad + r - 4
    cy = y + h - pad - r + 10
    energy_dir = Path(images_dir) / "energy"
    size = 2 * r

    for i, ty in enumerate(energy_types):
        ox = cx + i * step
        box = (ox - r, cy - r, ox + r, cy + r)
        filename = _ENERGY_IMAGE_FILES.get(ty)
        path = energy_dir / filename if filename else None
        outline_white = (255, 255, 255)
        if path and path.is_file():
            try:
                img = Image.open(path).convert("RGBA")
                scale = min(size / img.width, size / img.height) * 2
                new_w = max(1, int(img.width * scale))
                new_h = max(1, int(img.height * scale))
                img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                alpha = img.split()[3]
                paste_x = ox - r + (size - new_w) // 2
                paste_y = cy - r + (size - new_h) // 2
                bg.paste(img.convert("RGB"), (paste_x, paste_y), alpha)
            except Exception:
                color = _ENERGY_FALLBACK_COLORS.get(ty, (180, 180, 180))
                draw.ellipse([box[0], box[1], box[2], box[3]], fill=(50, 50, 50), outline=outline_white)
                draw.ellipse([box[0] + 2, box[1] + 2, box[2] - 2, box[3] - 2], fill=color, outline=outline_white)
        else:
            color = _ENERGY_FALLBACK_COLORS.get(ty, (180, 180, 180))
            draw.ellipse([box[0], box[1], box[2], box[3]], fill=(50, 50, 50), outline=outline_white)
            draw.ellipse([box[0] + 2, box[1] + 2, box[2] - 2, box[3] - 2], fill=color, outline=outline_white)
        draw.ellipse([box[0], box[1], box[2], box[3]], outline=outline_white)


def _draw_card_back(bg: Image.Image, draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int) -> None:
    """裏向きカード（サイド・手札用）を描画する。指定画像があれば貼り付け、なければ簡易模様。"""
    if CARD_BACK_IMAGE.is_file():
        _paste_card_image(bg, CARD_BACK_IMAGE, x, y, w, h, "裏")
    else:
        draw.rounded_rectangle([x, y, x + w, y + h], radius=6, fill=CARD_BACK_COLOR, outline=(120, 90, 50))
        draw.line([x + 4, y + 4, x + w - 4, y + h - 4], fill=(100, 80, 50), width=2)
        draw.line([x + w - 4, y + 4, x + 4, y + h - 4], fill=(100, 80, 50), width=2)


def _render_prize_stack(
    bg: Image.Image,
    draw: ImageDraw.ImageDraw,
    prize_pile: list,
    start_px: int,
    y_bottom: int,
    prize_w: int,
    prize_h: int,
    step: int,
    images_dir: Path,
) -> None:
    """サイドを表向きで重ねて描画。下（画面下・1 枚目が奥）から上（画面上・最後の枚が手前）に重なる。お互い同じ。"""
    n = len(prize_pile)
    for i in range(n - 1, -1, -1):
        row, col = divmod(i, 2)
        y = y_bottom - row * step - prize_h
        x = start_px + col * (prize_w // 2)
        card = prize_pile[i]
        card_id = getattr(card, "id", "")
        card_name = getattr(card, "name", "") or getattr(card, "name_ja", "?")
        path = get_card_image_path(card_id, images_dir)
        _paste_card_image(bg, path, x, y, prize_w, prize_h, card_name)


def _render_hand_cards(
    bg: Image.Image,
    draw: ImageDraw.ImageDraw,
    hand: list,
    images_dir: Path,
    hand_y: int,
    zone_center_x: int,
    max_width: int,
) -> None:
    """手札を描画。横幅は max_width 以内に収め、枚数に応じてカードサイズを可変にして全枚表示する。"""
    n_hand = len(hand)
    if n_hand == 0:
        return
    n_show = n_hand
    card_w = (max_width - max(0, n_show - 1) * BENCH_GAP) // n_show
    if card_w < MIN_HAND_CARD_W:
        n_show = max(1, (max_width + BENCH_GAP) // (MIN_HAND_CARD_W + BENCH_GAP))
        n_show = min(n_show, n_hand)
        card_w = (max_width - max(0, n_show - 1) * BENCH_GAP) // n_show
        if card_w < MIN_HAND_CARD_W:
            card_w = MIN_HAND_CARD_W
    else:
        card_w = min(CARD_W_BENCH, card_w)
    card_h = int(card_w * CARD_H_BENCH / CARD_W_BENCH)
    total_hw = n_show * card_w + max(0, n_show - 1) * BENCH_GAP
    start_hx = zone_center_x - total_hw // 2
    for i in range(n_show):
        card = hand[i]
        hx = start_hx + i * (card_w + BENCH_GAP)
        card_id = getattr(card, "id", "")
        name = getattr(card, "name", "?")
        path = get_card_image_path(card_id, images_dir)
        _paste_card_image(bg, path, hx, hand_y, card_w, card_h, name)
    if n_hand > n_show:
        try:
            font_small = ImageFont.truetype("/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc", 12)
        except OSError:
            font_small = ImageFont.load_default()
        draw.text(
            (start_hx + total_hw - 24, hand_y + card_h // 2 - 6),
            f"+{n_hand - n_show}",
            fill=TEXT_COLOR,
            font=font_small,
        )


def render_board_frame(
    state: "GameState",
    images_dir: Path | str,
    output_path: Path | str | None = None,
    width: int = BOARD_WIDTH,
    height: int = BOARD_HEIGHT,
    log_text: str | None = None,
) -> Image.Image:
    """
    現在の GameState から盤面を 1 枚の画像として描画する。

    - 上: 相手側、下: 自分側。中央に横線でバトル場が向き合う。
    - 各側にサイド・山札・ベンチ・バトル場・手札（カード画像）を表示。
    - log_text を渡すと左側にログパネルを描画する。
    """
    images_dir = Path(images_dir)
    board_offset = LOG_PANEL_WIDTH if log_text else 0
    total_width = board_offset + width
    bg = Image.new("RGB", (total_width, height), BG_COLOR)
    draw = ImageDraw.Draw(bg)

    if log_text:
        panel_text = _simplify_log_text_for_panel(log_text)
        draw.rectangle([0, 0, LOG_PANEL_WIDTH, height], fill=(20, 55, 20))
        draw.line([LOG_PANEL_WIDTH, 0, LOG_PANEL_WIDTH, height], fill=(70, 110, 70))
        _draw_log_panel(draw, panel_text, LOG_PANEL_WIDTH, height)

    cx = board_offset + width // 2
    center_y = height // 2
    total_bw = BENCH_SLOTS * CARD_W_BENCH + (BENCH_SLOTS - 1) * BENCH_GAP
    active_x = cx - CARD_W_BENCH // 2
    start_bx = cx - total_bw // 2
    prize_w = CARD_W_BENCH
    prize_h = CARD_H_BENCH
    prize_area_height = 12 + 2 * CARD_H_BENCH
    prize_step = int(CARD_H_BENCH * 0.55)
    prize_row_w = int(1.5 * prize_w) + BENCH_GAP
    start_px_opp = start_bx + total_bw + PRIZE_TO_BENCH
    start_px_self = start_bx - prize_row_w - PRIZE_TO_BENCH
    hand_max_width = total_bw + 2 * CARD_W_BENCH + 2 * BENCH_GAP

    try:
        font_label = ImageFont.truetype("/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc", 20)
        font_small = ImageFont.truetype("/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc", 12)
    except OSError:
        font_label = ImageFont.load_default()
        font_small = ImageFont.load_default()

    draw.rectangle([board_offset, center_y - 2, total_width, center_y + 2], fill=(70, 110, 70))

    turn_n = (state.turn_count // 2) + 1
    is_first_turn = state.current_player == state.first_player
    turn_label = f"先行{turn_n}ターン" if is_first_turn else f"後行{turn_n}ターン"
    turn_text = f"{turn_label} - {state.player_name(state.current_player)}のターン"
    opp_label = state.player_name(1)
    self_label = state.player_name(0)

    if state.current_player == 1:
        try:
            bbox = draw.textbbox((0, 0), turn_text, font=font_label)
            tw = bbox[2] - bbox[0]
            draw.text((cx - tw // 2, 8), turn_text, fill=LABEL_COLOR, font=font_label)
        except (TypeError, AttributeError):
            draw.text((cx - 80, 8), turn_text, fill=LABEL_COLOR, font=font_label)

    opp = state.players[1]
    draw.text((board_offset + MARGIN, 10), opp_label, fill=LABEL_COLOR, font=font_label)
    hand_y_opp = 42
    _render_hand_cards(bg, draw, opp.hand, images_dir, hand_y_opp, cx, hand_max_width)
    draw.text((cx + hand_max_width // 2 + 8, hand_y_opp + CARD_H_BENCH + 4), f"手札 {len(opp.hand)} 枚", fill=TEXT_COLOR, font=font_small, anchor="lt")
    active_y_opp = center_y - CARD_H_BENCH - 16
    bench_y_opp = active_y_opp - 20 - CARD_H_BENCH
    prize_y_bottom_opp = active_y_opp + CARD_H_BENCH
    _render_prize_stack(bg, draw, opp.prize_pile, start_px_opp, prize_y_bottom_opp, prize_w, prize_h, prize_step, images_dir)
    prize_block_top_opp = prize_y_bottom_opp - prize_h - 2 * prize_step
    deck_trash_x_opp = start_bx - CARD_W_BENCH - BENCH_GAP
    deck_y_opp = active_y_opp
    trash_y_opp = bench_y_opp
    draw.text((deck_trash_x_opp, deck_y_opp - 14), f"山札 {len(opp.deck)}", fill=TEXT_COLOR, font=font_small)
    if opp.deck:
        _draw_card_back(bg, draw, deck_trash_x_opp, deck_y_opp, CARD_W_BENCH, CARD_H_BENCH)
    else:
        _draw_placeholder(draw, deck_trash_x_opp, deck_y_opp, CARD_W_BENCH, CARD_H_BENCH, "")
    draw.text((deck_trash_x_opp, trash_y_opp - 14), f"トラッシュ {len(opp.discard)}", fill=TEXT_COLOR, font=font_small)
    if opp.discard:
        top_card = opp.discard[-1]
        path = get_card_image_path(getattr(top_card, "id", ""), images_dir)
        if path and path.is_file():
            try:
                img = Image.open(path).convert("RGB")
                img = img.resize((CARD_W_BENCH, CARD_H_BENCH), Image.Resampling.LANCZOS)
                bg.paste(img, (deck_trash_x_opp, trash_y_opp))
            except Exception:
                _draw_placeholder(draw, deck_trash_x_opp, trash_y_opp, CARD_W_BENCH, CARD_H_BENCH, "")
        else:
            _draw_placeholder(draw, deck_trash_x_opp, trash_y_opp, CARD_W_BENCH, CARD_H_BENCH, "")
    else:
        _draw_placeholder(draw, deck_trash_x_opp, trash_y_opp, CARD_W_BENCH, CARD_H_BENCH, "")
    for i in range(BENCH_SLOTS):
        bx = start_bx + i * (CARD_W_BENCH + BENCH_GAP)
        if i < len(opp.bench):
            bp = opp.bench[i]
            _draw_pokemon_with_tool(bg, draw, bp, bx, bench_y_opp, CARD_W_BENCH, CARD_H_BENCH, images_dir)
        else:
            _draw_placeholder(draw, bx, bench_y_opp, CARD_W_BENCH, CARD_H_BENCH, "")
    if opp.active:
        bp = opp.active
        _draw_pokemon_with_tool(bg, draw, bp, active_x, active_y_opp, CARD_W_BENCH, CARD_H_BENCH, images_dir)
    else:
        _draw_placeholder(draw, active_x, active_y_opp, CARD_W_BENCH, CARD_H_BENCH, "なし")

    self_p = state.players[0]
    active_y_self = center_y + 16
    if self_p.active:
        bp = self_p.active
        _draw_pokemon_with_tool(bg, draw, bp, active_x, active_y_self, CARD_W_BENCH, CARD_H_BENCH, images_dir)
    else:
        _draw_placeholder(draw, active_x, active_y_self, CARD_W_BENCH, CARD_H_BENCH, "なし")
    bench_y_self = active_y_self + CARD_H_BENCH + 28
    for i in range(BENCH_SLOTS):
        bx = start_bx + i * (CARD_W_BENCH + BENCH_GAP)
        if i < len(self_p.bench):
            bp = self_p.bench[i]
            _draw_pokemon_with_tool(bg, draw, bp, bx, bench_y_self, CARD_W_BENCH, CARD_H_BENCH, images_dir)
        else:
            _draw_placeholder(draw, bx, bench_y_self, CARD_W_BENCH, CARD_H_BENCH, "")
    deck_trash_x_self = start_bx + total_bw + BENCH_GAP
    deck_y_self = bench_y_self - CARD_H_BENCH - 8
    trash_y_self = bench_y_self
    draw.text((deck_trash_x_self, deck_y_self - 14), f"山札 {len(self_p.deck)}", fill=TEXT_COLOR, font=font_small)
    if self_p.deck:
        _draw_card_back(bg, draw, deck_trash_x_self, deck_y_self, CARD_W_BENCH, CARD_H_BENCH)
    else:
        _draw_placeholder(draw, deck_trash_x_self, deck_y_self, CARD_W_BENCH, CARD_H_BENCH, "")
    if self_p.discard:
        top_card = self_p.discard[-1]
        path = get_card_image_path(getattr(top_card, "id", ""), images_dir)
        if path and path.is_file():
            try:
                img = Image.open(path).convert("RGB")
                img = img.resize((CARD_W_BENCH, CARD_H_BENCH), Image.Resampling.LANCZOS)
                bg.paste(img, (deck_trash_x_self, trash_y_self))
            except Exception:
                _draw_placeholder(draw, deck_trash_x_self, trash_y_self, CARD_W_BENCH, CARD_H_BENCH, "")
        else:
            _draw_placeholder(draw, deck_trash_x_self, trash_y_self, CARD_W_BENCH, CARD_H_BENCH, "")
    else:
        _draw_placeholder(draw, deck_trash_x_self, trash_y_self, CARD_W_BENCH, CARD_H_BENCH, "")
    draw.text((deck_trash_x_self, trash_y_self + CARD_H_BENCH + 2), f"トラッシュ {len(self_p.discard)}", fill=TEXT_COLOR, font=font_small)
    hand_y_self = bench_y_self + CARD_H_BENCH + 36
    _render_hand_cards(bg, draw, self_p.hand, images_dir, hand_y_self, cx, hand_max_width)
    draw.text((cx + hand_max_width // 2 + 8, hand_y_self - 4), f"手札 {len(self_p.hand)} 枚", fill=TEXT_COLOR, font=font_small, anchor="lb")
    if state.current_player == 0:
        turn_y_below_hand = hand_y_self + CARD_H_BENCH + 12
        try:
            bbox = draw.textbbox((0, 0), turn_text, font=font_label)
            tw = bbox[2] - bbox[0]
            draw.text((cx - tw // 2, turn_y_below_hand), turn_text, fill=LABEL_COLOR, font=font_label)
        except (TypeError, AttributeError):
            draw.text((cx - 80, turn_y_below_hand), turn_text, fill=LABEL_COLOR, font=font_label)
    prize_y_bottom_self = bench_y_self + CARD_H_BENCH
    _render_prize_stack(bg, draw, self_p.prize_pile, start_px_self, prize_y_bottom_self, prize_w, prize_h, prize_step, images_dir)
    prize_block_top_self = prize_y_bottom_self - prize_h - 2 * prize_step
    draw.text((board_offset + MARGIN, height - 28), self_label, fill=LABEL_COLOR, font=font_label)

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        bg.save(out)

    return bg


def run_game_until_turn(state: "GameState", max_turn_count: int) -> None:
    """
    試合を max_turn_count ターンまで進める（または勝者が決まるまで）。
    状態は state をそのまま更新する。
    """
    from game import (
        _check_game_end,
        end_turn,
        run_turn_auto,
        start_turn,
    )

    while state.turn_count < max_turn_count and state.winner is None:
        start_turn(state)
        if state.winner is not None:
            return
        run_turn_auto(state)
        _check_game_end(state)
        if state.winner is not None:
            return
        end_turn(state)


def main() -> None:
    """対戦 1 ゲームをセットアップし、初期盤面を 1 枚描画して保存する。"""
    import argparse

    from game import setup_game

    parser = argparse.ArgumentParser(description="盤面フレームを 1 枚生成する")
    parser.add_argument(
        "--mid-game",
        action="store_true",
        help="試合を数ターン進めた「試合途中」の盤面を 1 枚生成する",
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=5,
        metavar="N",
        help="--mid-game 時に何ターン進めるか（デフォルト: 5）",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        metavar="PATH",
        help="出力画像パス（省略時は frames/frame_001.png または frames/frame_midgame.png）",
    )
    args = parser.parse_args()

    state = setup_game(seed=42, deck0=3, deck1=4)
    images_dir = _PROJECT_ROOT / "card_images"
    out_dir = _PROJECT_ROOT / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mid_game:
        run_game_until_turn(state, args.turns)
        output = Path(args.output) if args.output else out_dir / "frame_midgame.png"
    else:
        output = Path(args.output) if args.output else out_dir / "frame_001.png"

    render_board_frame(state, images_dir, output_path=output)
    print(f"保存しました: {output}")


if __name__ == "__main__":
    main()
