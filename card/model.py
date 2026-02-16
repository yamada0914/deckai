"""
カードの型定義（model）。

ポケモン・エネルギー・アイテムの dataclass と、カード種別判定 is_pokemon / is_energy / is_item を提供。
"""
from dataclasses import dataclass, field
from typing import Literal

CardType = Literal["pokemon", "energy", "item"]


@dataclass
class Attack:
    name: str
    energy_cost: int
    damage: int
    self_damage: int = 0
    bench_damage: int = 0  # 相手のベンチ 1 体への追加ダメージ
    description: str = ""


@dataclass
class PokemonCard:
    id: str
    name: str
    type: CardType = "pokemon"
    hp: int = 110
    max_hp: int = 110
    attacks: list[Attack] = field(default_factory=list)
    evolves_from: str | None = None  # 進化元（例: "meguroko"）
    retreat_cost: int = 1  # にげるために捨てるエネルギー数
    instance_id: str = ""

    def copy(self) -> "PokemonCard":
        return PokemonCard(
            id=self.id,
            name=self.name,
            type=self.type,
            hp=self.hp,
            max_hp=self.max_hp,
            attacks=[Attack(**a.__dict__) for a in self.attacks],
            evolves_from=self.evolves_from,
            retreat_cost=self.retreat_cost,
            instance_id=self.instance_id,
        )


@dataclass
class EnergyCard:
    id: str
    name: str
    type: CardType = "energy"
    provides: int = 1
    instance_id: str = ""


@dataclass
class ItemCard:
    id: str
    name: str
    type: CardType = "item"
    effect: str = "heal"
    heal_amount: int = 20
    description: str = ""
    instance_id: str = ""


def is_pokemon(card) -> bool:
    return getattr(card, "type", None) == "pokemon"


def is_energy(card) -> bool:
    return getattr(card, "type", None) == "energy"


def is_item(card) -> bool:
    return getattr(card, "type", None) == "item"
