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
    EnergyTypeId,
    EnergyTypeInfo,
    ENERGY_TYPES,
    ItemCard,
    PokemonCard,
    PokemonTypeId,
    PokemonTypeInfo,
    POKEMON_TYPES,
    SupportCard,
    is_energy,
    is_item,
    is_pokemon,
    is_support,
)

__all__ = [
    "Attack",
    "CardType",
    "CARD_ID_TO_NAME",
    "EnergyCard",
    "EnergyTypeId",
    "EnergyTypeInfo",
    "ENERGY_TYPES",
    "ItemCard",
    "PokemonCard",
    "PokemonTypeId",
    "PokemonTypeInfo",
    "POKEMON_TYPES",
    "SupportCard",
    "get_card_by_id",
    "is_energy",
    "is_item",
    "is_pokemon",
    "is_support",
]
