"""
カードのマスタデータ。

各カードの実体（OTACHI, MEGUROKO 等）、レジストリ、get_card_by_id / CARD_ID_TO_NAME を提供。
"""
from copy import deepcopy

from card.model import Attack, EnergyCard, ItemCard, PokemonCard

# ----- マスタカード -----

# オタチ（たねポケモン・デッキ A 用）
OTACHI = PokemonCard(
    id="otachi",
    name="オタチ",
    hp=60,
    max_hp=60,
    attacks=[
        Attack("ひらてうち", 1, 20, 0, 0, "20 ダメージ"),
    ],
    evolves_from=None,
)

# オオタチ（オタチから進化・デッキ A 用）
OOTACHI = PokemonCard(
    id="ootachi",
    name="オオタチ",
    hp=120,
    max_hp=120,
    attacks=[
        Attack("ひっかく", 1, 40, 0, 0, "40 ダメージ"),
        Attack("ジェットヘッド", 2, 70, 0, 0, "70 ダメージ"),
    ],
    evolves_from="otachi",
)

# モトトカゲ（たねポケモン・デッキ A 用）
MOTOTOKAGE = PokemonCard(
    id="mototokage",
    name="モトトカゲ",
    hp=110,
    max_hp=110,
    attacks=[
        Attack("しっぽのムチ", 1, 10, 0, 0, "10 ダメージ"),
        Attack("スピードアタック", 2, 50, 0, 0, "50 ダメージ"),
    ],
    evolves_from=None,
)

# メグロコ（たねポケモン・デッキ B 用）
MEGUROKO = PokemonCard(
    id="meguroko",
    name="メグロコ",
    hp=70,
    max_hp=70,
    attacks=[
        Attack("かじる", 1, 210, 0, 0, "10 ダメージ"),
        Attack("ぶつかる", 2, 230, 0, 0, "30 ダメージ"),
    ],
    evolves_from=None,
    retreat_cost=2,
)

# ワルビル（メグロコから進化・デッキ B 用）
WARUBIRU = PokemonCard(
    id="warubiru",
    name="ワルビル",
    hp=100,
    max_hp=100,
    attacks=[
        Attack("しっぺがえし", 1, 230, 0, 0, "30 ダメージ（相手のサイドが残り 1 枚なら 90 ダメージ追加）"),
        Attack("どつく", 2, 260, 0, 0, "60 ダメージ"),
    ],
    evolves_from="meguroko",
    retreat_cost=2,
)

# ズピカ（たねポケモン・デッキ C 用）
ZUPIKA = PokemonCard(
    id="zupika",
    name="ズピカ",
    hp=60,
    max_hp=60,
    attacks=[
        Attack("でんきショック", 2, 40, 10, 0, "40 ダメージ（自分にも 10 ダメージ）"),
    ],
    evolves_from=None,
)

# ハラバリー（ズピカから進化・デッキ C 用）
HARABARII = PokemonCard(
    id="harabarii",
    name="ハラバリー",
    hp=130,
    max_hp=130,
    attacks=[
        Attack("エレキバレット", 2, 70, 0, 30, "70 ダメージ（相手のベンチ 1 体にも 30 ダメージ）"),
    ],
    evolves_from="zupika",
    retreat_cost=2,
)

# 基本エネルギー
BASIC_ENERGY = EnergyCard(id="basic-energy", name="基本エネルギー", provides=1)

# きずぐすり（自分のバトル場のポケモンを 30 回復）
POTION = ItemCard(
    id="potion",
    name="きずぐすり",
    effect="heal",
    heal_amount=30,
    description="自分のバトル場のポケモンを 30 回復する",
)

# ログ用：カード ID → 表示名
CARD_ID_TO_NAME = {
    "otachi": "オタチ",
    "ootachi": "オオタチ",
    "mototokage": "モトトカゲ",
    "meguroko": "メグロコ",
    "warubiru": "ワルビル",
    "zupika": "ズピカ",
    "harabarii": "ハラバリー",
    "basic-energy": "基本エネルギー",
    "potion": "きずぐすり",
}

# カード ID → マスタカード（get_card_by_id 用）
_CARD_REGISTRY: dict[str, PokemonCard | EnergyCard | ItemCard] = {
    "otachi": OTACHI,
    "ootachi": OOTACHI,
    "mototokage": MOTOTOKAGE,
    "meguroko": MEGUROKO,
    "warubiru": WARUBIRU,
    "zupika": ZUPIKA,
    "harabarii": HARABARII,
    "basic-energy": BASIC_ENERGY,
    "potion": POTION,
}


def get_card_by_id(card_id: str, instance_id: str = "") -> PokemonCard | EnergyCard | ItemCard:
    """カード ID からマスタのコピーを生成する。instance_id を付与。"""
    if card_id not in _CARD_REGISTRY:
        raise ValueError(f"Unknown card id: {card_id}")
    c = deepcopy(_CARD_REGISTRY[card_id])
    c.instance_id = instance_id
    return c
