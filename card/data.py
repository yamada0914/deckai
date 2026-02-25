"""
カードのマスタデータ。

各カードの実体（OTACHI, MEGUROKO 等）、レジストリ、get_card_by_id / CARD_ID_TO_NAME を提供。
"""
from copy import deepcopy

from card.model import Attack, EnergyCard, GoodsCard, is_goods, is_support, PokemonCard, SupportCard

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

MEGUROKO = PokemonCard(
    id="meguroko",
    name="メグロコ",
    hp=70,
    max_hp=70,
    attacks=[
        Attack("かじる", 1, 10, 0, 0, "10 ダメージ", energy_cost_typed=["fighting"]),
        Attack("ぶつかる", 2, 30, 0, 0, "30 ダメージ", energy_cost_typed=["fighting", "fighting"]),
    ],
    evolves_from=None,
    retreat_cost=2,
    pokemon_type="fighting",
    weakness="grass",
)

WARUBIRU = PokemonCard(
    id="warubiru",
    name="ワルビル",
    hp=100,
    max_hp=100,
    attacks=[
        Attack("しっぺがえし", 1, 30, 0, 0, "30 ダメージ（相手のサイドが残り 1 枚なら 90 ダメージ追加）", energy_cost_typed=["fighting"]),
        Attack("どつく", 2, 60, 0, 0, "60 ダメージ", energy_cost_typed=["fighting", "fighting"]),
    ],
    evolves_from="meguroko",
    retreat_cost=2,
    pokemon_type="fighting",
    weakness="grass",
)

ZUPIKA = PokemonCard(
    id="zupika",
    name="ズピカ",
    hp=60,
    max_hp=60,
    attacks=[
        Attack("でんきショック", 2, 40, 10, 0, "40 ダメージ（自分にも 10 ダメージ）"),
    ],
    evolves_from=None,
    retreat_cost=1,
    pokemon_type="lightning",
    weakness="fighting",
)

HARABARII = PokemonCard(
    id="harabarii",
    name="ハラバリー",
    hp=130,
    max_hp=130,
    attacks=[
        Attack(
            "エレキバレット", 2, 70, 0, 30,
            "70 ダメージ（相手のベンチ 1 体にも 30 ダメージ）",
            energy_cost_typed=["lightning", "colorless"],
        ),
    ],
    evolves_from="zupika",
    retreat_cost=2,
    pokemon_type="lightning",
    weakness="fighting",
)

# ----- 以下 JSON から生成（update_cards_from_json.py で上書き） -----
# カラミンゴ（karamingo-svg-029）
KARAMINGO_SVG_029 = PokemonCard(
    id='karamingo-svg-029',
    name='カラミンゴ',
    hp=110,
    max_hp=110,
    attacks=[
        Attack('はばたく', 1, 30, 0, 0, ''),
        Attack('きゅうこうか', 3, 110, 20, 0, 'このポケモンにも20ダメージ。'),
    ],
    evolves_from=None,
    retreat_cost=1,
    pokemon_type='colorless',
    weakness='lightning',
    resistance='fighting',
)

# ガケガニ（gakegani-svd-067）
GAKEGANI_SVD_067 = PokemonCard(
    id='gakegani-svd-067',
    name='ガケガニ',
    hp=130,
    max_hp=130,
    attacks=[
        Attack('はさむ', 2, 50, 0, 0, '', energy_cost_typed=['fighting', 'fighting']),
        Attack('アドレナハンマー', 3, 130, 0, 0, 'このポケモンをこんらんにする。', energy_cost_typed=['fighting', 'fighting', 'fighting'], status_effect='confusion', status_effect_target='self'),
    ],
    evolves_from=None,
    retreat_cost=3,
    pokemon_type='fighting',
    weakness='grass',
)

# コイル（coil-svd-036）
COIL_SVD_036 = PokemonCard(
    id='coil-svd-036',
    name='コイル',
    hp=60,
    max_hp=60,
    attacks=[
        Attack('ぶつかる', 1, 10, 0, 0, '', energy_cost_typed=['lightning']),
        Attack('スピードボール', 2, 20, 0, 0, ''),
    ],
    evolves_from=None,
    retreat_cost=1,
    pokemon_type='lightning',
    weakness='fighting',
)

# ジバコイル（jibakoil-svd-038）
JIBAKOIL_SVD_038 = PokemonCard(
    id='jibakoil-svd-038',
    name='ジバコイル',
    hp=170,
    max_hp=170,
    attacks=[
        Attack('マグネリジェクト', 1, 50, 0, 0, 'のぞむなら、相手のバトルポケモンをベンチポケモンと入れ替える。[バトル場に出すポケモンは相手が選ぶ。]', energy_cost_typed=['lightning']),
        Attack('かみなり', 3, 180, 30, 0, 'このポケモンにも30ダメージ。', energy_cost_typed=['lightning', 'colorless', 'colorless']),
    ],
    evolves_from='rarecoil',
    retreat_cost=2,
    pokemon_type='lightning',
    weakness='fighting',
)

# ズピカ（zupika-svd-041）
ZUPIKA_SVD_041 = PokemonCard(
    id='zupika-svd-041',
    name='ズピカ',
    hp=60,
    max_hp=60,
    attacks=[
        Attack('でんげき', 2, 40, 10, 0, 'このポケモンにも10ダメージ。', energy_cost_typed=['lightning', 'colorless']),
    ],
    evolves_from=None,
    retreat_cost=1,
    pokemon_type='lightning',
    weakness='fighting',
)

# ノノクラゲ（nonokurage-svd-065）
NONOKURAGE_SVD_065 = PokemonCard(
    id='nonokurage-svd-065',
    name='ノノクラゲ',
    hp=60,
    max_hp=60,
    attacks=[
        Attack('けとばす', 1, 10, 0, 0, '', energy_cost_typed=['fighting']),
        Attack('どろかけ', 2, 20, 0, 0, ''),
    ],
    evolves_from=None,
    retreat_cost=1,
    pokemon_type='fighting',
    weakness='grass',
)

# ハラバリー（harabarii-svd-042）
HARABARII_SVD_042 = PokemonCard(
    id='harabarii-svd-042',
    name='ハラバリー',
    hp=130,
    max_hp=130,
    attacks=[
        Attack('エレキバレット', 2, 70, 0, 30, '相手のベンチポケモン1匹にも、30ダメージ。[ベンチは弱点・抵抗力を計算しない。]', energy_cost_typed=['lightning', 'colorless']),
    ],
    evolves_from='zupika',
    retreat_cost=2,
    pokemon_type='lightning',
    weakness='fighting',
)

# メグロコ（meguroko-svd-062）
MEGUROKO_SVD_062 = PokemonCard(
    id='meguroko-svd-062',
    name='メグロコ',
    hp=70,
    max_hp=70,
    attacks=[
        Attack('かじる', 1, 10, 0, 0, '', energy_cost_typed=['fighting']),
        Attack('ぶつかる', 2, 30, 0, 0, '', energy_cost_typed=['fighting', 'fighting']),
    ],
    evolves_from=None,
    retreat_cost=2,
    pokemon_type='fighting',
    weakness='grass',
)

# モトトカゲ（mototokage-mc-627）
MOTOTOKAGE_MC_627 = PokemonCard(
    id='mototokage-mc-627',
    name='モトトカゲ',
    hp=110,
    max_hp=110,
    attacks=[
        Attack('しっぽのムチ', 1, 10, 0, 0, ''),
        Attack('スピードアタック', 2, 50, 0, 0, ''),
    ],
    evolves_from=None,
    retreat_cost=1,
    pokemon_type='colorless',
    weakness='fighting',
)

# リククラゲ（rikukurage-svd-066）
RIKUKURAGE_SVD_066 = PokemonCard(
    id='rikukurage-svd-066',
    name='リククラゲ',
    hp=130,
    max_hp=130,
    attacks=[
        Attack('たたく', 2, 40, 0, 0, ''),
        Attack('ダブルウィップ', 3, 0, 0, 0, 'コインを2回投げ、オモテの数×100ダメージ。', energy_cost_typed=['fighting', 'colorless', 'colorless'], coin_flips=2, damage_per_coin=100),
    ],
    evolves_from='nonokurage',
    retreat_cost=2,
    pokemon_type='fighting',
    weakness='grass',
)

# レアコイル（rarecoil-svd-037）
RARECOIL_SVD_037 = PokemonCard(
    id='rarecoil-svd-037',
    name='レアコイル',
    hp=90,
    max_hp=90,
    attacks=[
        Attack('たいあたり', 1, 30, 0, 0, '', energy_cost_typed=['lightning']),
        Attack('エレキボール', 3, 60, 0, 0, '', energy_cost_typed=['lightning', 'colorless', 'colorless']),
    ],
    evolves_from='coil',
    retreat_cost=2,
    pokemon_type='lightning',
    weakness='fighting',
)

# ワルビアル（warubiaru-svd-064）
WARUBIARU_SVD_064 = PokemonCard(
    id='warubiaru-svd-064',
    name='ワルビアル',
    hp=170,
    max_hp=170,
    attacks=[
        Attack('ガブガブバイト', 1, 50, 0, 0, 'ウラが出るまでコインを投げ、オモテの数ぶん、相手のバトルポケモンについているエネルギーを選び、トラッシュする。', energy_cost_typed=['fighting']),
        Attack('じしん', 2, 180, 0, 30, '自分のベンチポケモン全員にも、それぞれ30ダメージ。[ベンチは弱点・抵抗力を計算しない。]', energy_cost_typed=['fighting', 'fighting']),
    ],
    evolves_from='waruvile',
    retreat_cost=3,
    pokemon_type='fighting',
    weakness='grass',
)

# ワルビル（warubiru-svd-063）
WARUBIRU_SVD_063 = PokemonCard(
    id='warubiru-svd-063',
    name='ワルビル',
    hp=100,
    max_hp=100,
    attacks=[
        Attack('しっぺがえし', 1, 30, 0, 0, '相手のサイドの残り枚数が1枚なら、90ダメージ追加。', energy_cost_typed=['fighting']),
        Attack('どつく', 2, 60, 0, 0, '', energy_cost_typed=['fighting', 'fighting']),
    ],
    evolves_from='meguroko',
    retreat_cost=2,
    pokemon_type='fighting',
    weakness='grass',
)

# ----- 以上 JSON から生成 -----

















# 基本エネルギー（無色＝任意の 1 エネルギーとして使用可）
BASIC_ENERGY = EnergyCard(id="basic-energy", name="基本エネルギー", provides=1, energy_type=None)
# 基本雷エネルギー（技コストの「雷」にカウント）
BASIC_ENERGY_LIGHTNING = EnergyCard(id="basic-energy-lightning", name="基本雷エネルギー", provides=1, energy_type="lightning")
# 基本闘エネルギー（技コストの「闘」にカウント）
BASIC_ENERGY_FIGHTING = EnergyCard(id="basic-energy-fighting", name="基本闘エネルギー", provides=1, energy_type="fighting")

# きずぐすり（自分のバトル場のポケモンを 30 回復）
POTION = GoodsCard(
    id="potion",
    name="きずぐすり",
    effect="heal",
    heal_amount=30,
    description="自分のバトル場のポケモンを 30 回復する",
)

# ポケモンいれかえ（自分のバトル場のポケモンとベンチのポケモンを入れ替えるグッズ）
POKEMON_IREKAE = GoodsCard(
    id="pokemon_irekae",
    name="ポケモンいれかえ",
    effect="swap_active",
    description="自分のバトル場のポケモンとベンチのポケモンを 1 体入れ替える",
)

# 岩のむねあて（ポケモンのどうぐ：つけている闘ポケモンが受けるワザのダメージ -30）
IWANOMUNEATE = GoodsCard(
    id="iwanomuneate",
    name="岩のむねあて",
    effect="tool",
    description="このカードをつけている闘ポケモンが、相手のポケモンから受けるワザのダメージは「-30」される。",
    is_tool=True,
    tool_damage_reduce=30,
    tool_condition_type="fighting",
)

# ネモ（サポート：デッキから 3 枚引く）
NEMO = SupportCard(
    id="nemo",
    name="ネモ",
    effect="draw_3",
    draw_count=3,
    description="デッキからカードを 3 枚引く",
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
    "basic-energy-lightning": "基本雷エネルギー",
    "basic-energy-fighting": "基本闘エネルギー",
    "potion": "きずぐすり",
    "pokemon_irekae": "ポケモンいれかえ",
    "iwanomuneate": "岩のむねあて",
    "nemo": "ネモ",

    # ----- JSON から生成 -----
    "karamingo-svg-029": 'カラミンゴ',
    "gakegani-svd-067": 'ガケガニ',
    "coil-svd-036": 'コイル',
    "jibakoil-svd-038": 'ジバコイル',
    "zupika-svd-041": 'ズピカ',
    "nonokurage-svd-065": 'ノノクラゲ',
    "harabarii-svd-042": 'ハラバリー',
    "meguroko-svd-062": 'メグロコ',
    "mototokage-mc-627": 'モトトカゲ',
    "rikukurage-svd-066": 'リククラゲ',
    "rarecoil-svd-037": 'レアコイル',
    "warubiaru-svd-064": 'ワルビアル',
    "warubiru-svd-063": 'ワルビル',
    # ----- 以上 JSON から生成 -----
}

def get_trainer_id_by_name(name_ja: str) -> str | None:
    """トレーナー（グッズ・サポート）の名前からレジストリの id を返す。見つからなければ None。"""
    name = (name_ja or "").strip()
    if not name:
        return None
    for cid, card in _CARD_REGISTRY.items():
        if (is_goods(card) or is_support(card)) and (getattr(card, "name", "") or "").strip() == name:
            return cid
    return None


# カード ID → マスタカード（get_card_by_id 用）
_CARD_REGISTRY: dict[str, PokemonCard | EnergyCard | GoodsCard | SupportCard] = {
    "otachi": OTACHI,
    "ootachi": OOTACHI,
    "mototokage": MOTOTOKAGE,
    "meguroko": MEGUROKO,
    "warubiru": WARUBIRU,
    "zupika": ZUPIKA,
    "harabarii": HARABARII,
    "basic-energy": BASIC_ENERGY,
    "basic-energy-lightning": BASIC_ENERGY_LIGHTNING,
    "basic-energy-fighting": BASIC_ENERGY_FIGHTING,
    "potion": POTION,
    "pokemon_irekae": POKEMON_IREKAE,
    "iwanomuneate": IWANOMUNEATE,
    "nemo": NEMO,

    # ----- JSON から生成 -----
    "karamingo-svg-029": KARAMINGO_SVG_029,
    "gakegani-svd-067": GAKEGANI_SVD_067,
    "coil-svd-036": COIL_SVD_036,
    "jibakoil-svd-038": JIBAKOIL_SVD_038,
    "zupika-svd-041": ZUPIKA_SVD_041,
    "nonokurage-svd-065": NONOKURAGE_SVD_065,
    "harabarii-svd-042": HARABARII_SVD_042,
    "meguroko-svd-062": MEGUROKO_SVD_062,
    "mototokage-mc-627": MOTOTOKAGE_MC_627,
    "rikukurage-svd-066": RIKUKURAGE_SVD_066,
    "rarecoil-svd-037": RARECOIL_SVD_037,
    "warubiaru-svd-064": WARUBIARU_SVD_064,
    "warubiru-svd-063": WARUBIRU_SVD_063,
    # ----- 以上 JSON から生成 -----
}


def get_card_by_id(card_id: str, instance_id: str = "") -> PokemonCard | EnergyCard | GoodsCard | SupportCard:
    """カード ID からマスタのコピーを生成する。instance_id を付与。"""
    if card_id not in _CARD_REGISTRY:
        raise ValueError(f"Unknown card id: {card_id}")
    c = deepcopy(_CARD_REGISTRY[card_id])
    c.instance_id = instance_id
    return c
