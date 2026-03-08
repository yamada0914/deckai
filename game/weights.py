"""選択肢への重み（機械学習用）。未登録のカードは 0 として既存ロジックに寄せる。"""
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class GameWeights:
    """
    各選択で card_id ごとに加算するバイアス。
    キーが無い場合は 0 として扱い、重みなし＝現状ルールになる。
    """
    w_energy_attach: dict[str, float] = field(default_factory=dict)
    w_retreat_target: dict[str, float] = field(default_factory=dict)
    w_swap_target: dict[str, float] = field(default_factory=dict)
    w_promote: dict[str, float] = field(default_factory=dict)
    w_attack: dict[str, float] = field(default_factory=dict)
    w_catcher_target: dict[str, float] = field(default_factory=dict)
    w_support_use: dict[str, float] = field(default_factory=dict)
    w_goods_use: dict[str, float] = field(default_factory=dict)
    w_evolve_onto: dict[str, float] = field(default_factory=dict)
    w_tool_attach: dict[str, float] = field(default_factory=dict)
    w_haipaboru_discard: dict[str, float] = field(default_factory=dict)
    w_pokepaddo_fetch: dict[str, float] = field(default_factory=dict)
    w_faitogongu_fetch: dict[str, float] = field(default_factory=dict)


def _card_id(card) -> str:
    """カードの識別子。重みのキーに使う。"""
    return getattr(card, "id", None) or getattr(card, "name", "") or ""


def get_energy_attach_weight(weights: "GameWeights | None", card) -> float:
    if weights is None:
        return 0.0
    return weights.w_energy_attach.get(_card_id(card), 0.0)


def get_retreat_target_weight(weights: "GameWeights | None", card) -> float:
    if weights is None:
        return 0.0
    return weights.w_retreat_target.get(_card_id(card), 0.0)


def get_swap_target_weight(weights: "GameWeights | None", card) -> float:
    if weights is None:
        return 0.0
    return weights.w_swap_target.get(_card_id(card), 0.0)


def get_promote_weight(weights: "GameWeights | None", card) -> float:
    if weights is None:
        return 0.0
    return weights.w_promote.get(_card_id(card), 0.0)


def get_attack_weight(weights: "GameWeights | None", card, atk) -> float:
    """技選択の重み。キーは card_id|attack_name。"""
    if weights is None:
        return 0.0
    key = f"{_card_id(card)}|{getattr(atk, 'name', '')}"
    return weights.w_attack.get(key, 0.0)


def get_catcher_target_weight(weights: "GameWeights | None", card) -> float:
    if weights is None:
        return 0.0
    return weights.w_catcher_target.get(_card_id(card), 0.0)


def get_support_use_weight(weights: "GameWeights | None", card) -> float:
    if weights is None:
        return 0.0
    return weights.w_support_use.get(_card_id(card), 0.0)


def get_goods_use_weight(weights: "GameWeights | None", card) -> float:
    if weights is None:
        return 0.0
    return weights.w_goods_use.get(_card_id(card), 0.0)


def get_evolve_onto_weight(weights: "GameWeights | None", card) -> float:
    if weights is None:
        return 0.0
    return weights.w_evolve_onto.get(_card_id(card), 0.0)


def get_tool_attach_weight(weights: "GameWeights | None", card) -> float:
    if weights is None:
        return 0.0
    return weights.w_tool_attach.get(_card_id(card), 0.0)


def get_haipaboru_discard_weight(weights: "GameWeights | None", card) -> float:
    """ハイパーボールで捨てる 2 枚を選ぶときの重み。高いほど捨てやすい。"""
    if weights is None:
        return 0.0
    return weights.w_haipaboru_discard.get(_card_id(card), 0.0)


def get_pokepaddo_fetch_weight(weights: "GameWeights | None", card, deck_index: int) -> float:
    """
    ポケパッドで山札から取ってくるポケモンを選ぶときの重み。高いほど選ばれやすい。
    キーはまず "deck_index|card_id"（デッキごと）、無ければ "card_id"（共通）で参照する。
    """
    if weights is None:
        return 0.0
    cid = _card_id(card)
    key_deck = f"{deck_index}|{cid}"
    if key_deck in weights.w_pokepaddo_fetch:
        return weights.w_pokepaddo_fetch[key_deck]
    return weights.w_pokepaddo_fetch.get(cid, 0.0)


def get_faitogongu_fetch_weight(weights: "GameWeights | None", card, deck_index: int) -> float:
    """
    ファイトゴングで山札から取ってくるカード（闘たね or 基本闘エネルギー）を選ぶときの重み。高いほど選ばれやすい。
    キーはまず "deck_index|card_id"、無ければ "card_id" で参照する。
    """
    if weights is None:
        return 0.0
    cid = _card_id(card)
    key_deck = f"{deck_index}|{cid}"
    if key_deck in weights.w_faitogongu_fetch:
        return weights.w_faitogongu_fetch[key_deck]
    return weights.w_faitogongu_fetch.get(cid, 0.0)


def save_weights(weights: GameWeights, path: str | Path) -> None:
    """重みを JSON で保存する。"""
    path = Path(path)
    data = {
        "w_energy_attach": weights.w_energy_attach,
        "w_retreat_target": weights.w_retreat_target,
        "w_swap_target": weights.w_swap_target,
        "w_promote": weights.w_promote,
        "w_attack": weights.w_attack,
        "w_catcher_target": weights.w_catcher_target,
        "w_support_use": weights.w_support_use,
        "w_goods_use": weights.w_goods_use,
        "w_evolve_onto": weights.w_evolve_onto,
        "w_tool_attach": weights.w_tool_attach,
        "w_haipaboru_discard": weights.w_haipaboru_discard,
        "w_pokepaddo_fetch": weights.w_pokepaddo_fetch,
        "w_faitogongu_fetch": weights.w_faitogongu_fetch,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def scale_weights(weights: GameWeights, factor: float) -> GameWeights:
    """全重みを factor 倍した GameWeights を返す。実感を強くしたいときなどに使う。"""
    if factor == 1.0:
        return weights
    def _scale(d: dict[str, float]) -> dict[str, float]:
        return {k: v * factor for k, v in d.items()}
    return GameWeights(
        w_energy_attach=_scale(weights.w_energy_attach),
        w_retreat_target=_scale(weights.w_retreat_target),
        w_swap_target=_scale(weights.w_swap_target),
        w_promote=_scale(weights.w_promote),
        w_attack=_scale(weights.w_attack),
        w_catcher_target=_scale(weights.w_catcher_target),
        w_support_use=_scale(weights.w_support_use),
        w_goods_use=_scale(weights.w_goods_use),
        w_evolve_onto=_scale(weights.w_evolve_onto),
        w_tool_attach=_scale(weights.w_tool_attach),
        w_haipaboru_discard=_scale(weights.w_haipaboru_discard),
        w_pokepaddo_fetch=_scale(weights.w_pokepaddo_fetch),
        w_faitogongu_fetch=_scale(weights.w_faitogongu_fetch),
    )


def load_weights(path: str | Path, scale: float = 1.0) -> GameWeights:
    """JSON から重みを読み込む。scale に 1 以外を指定すると全重みをその倍率にする（実感を強くしたいとき用）。"""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    w = GameWeights(
        w_energy_attach=data.get("w_energy_attach", {}),
        w_retreat_target=data.get("w_retreat_target", {}),
        w_swap_target=data.get("w_swap_target", {}),
        w_promote=data.get("w_promote", {}),
        w_attack=data.get("w_attack", {}),
        w_catcher_target=data.get("w_catcher_target", {}),
        w_support_use=data.get("w_support_use", {}),
        w_goods_use=data.get("w_goods_use", {}),
        w_evolve_onto=data.get("w_evolve_onto", {}),
        w_tool_attach=data.get("w_tool_attach", {}),
        w_haipaboru_discard=data.get("w_haipaboru_discard", {}),
        w_pokepaddo_fetch=data.get("w_pokepaddo_fetch", {}),
        w_faitogongu_fetch=data.get("w_faitogongu_fetch", {}),
    )
    return scale_weights(w, scale) if scale != 1.0 else w
