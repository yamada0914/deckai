"""
card パッケージ。

- card.model: 型定義（PokemonCard, Attack 等）と is_pokemon / is_energy / is_item
- card.data: マスタデータと get_card_by_id / CARD_ID_TO_NAME

利用側は「from card import PokemonCard, get_card_by_id」のようにパッケージから import すればよい。
"""
from card.data import CARD_ID_TO_NAME, get_card_by_id
from card.model import (
    Attack,
    CardType,
    EnergyCard,
    ItemCard,
    PokemonCard,
    is_energy,
    is_item,
    is_pokemon,
)

__all__ = [
    "Attack",
    "CardType",
    "CARD_ID_TO_NAME",
    "EnergyCard",
    "ItemCard",
    "PokemonCard",
    "get_card_by_id",
    "is_energy",
    "is_item",
    "is_pokemon",
]
