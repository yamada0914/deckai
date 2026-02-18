"""
カードの型定義（model）。

ポケモン・エネルギー・アイテムの dataclass、エネルギータイプ一覧（画像認識用の形の情報含む）、
カード種別判定 is_pokemon / is_energy / is_item を提供。

ルール: 無色のエネルギーは技のコスト・にげるコストの両方で、任意の 1 エネルギーとして扱う。
"""
from dataclasses import dataclass, field
from typing import Literal

CardType = Literal["pokemon", "energy", "item", "support"]

# 基本エネルギーカードがあるタイプ（9 種）。画像認識用に各タイプの「形」の説明を保持する。
EnergyTypeId = Literal[
    "grass", "fire", "water", "lightning", "psychic",
    "fighting", "darkness", "metal", "fairy",
]

# ポケモンのタイプ全 11 種（エネルギーカードのない colorless / dragon を含む）
PokemonTypeId = Literal[
    "grass", "fire", "water", "lightning", "psychic",
    "fighting", "darkness", "metal", "fairy",
    "colorless", "dragon",
]


@dataclass(frozen=True)
class EnergyTypeInfo:
    """1 種類のエネルギータイプの情報（表示名と画像認識用の形の説明）。"""
    id: str  # コード用（grass, fire, ...）
    name_ja: str  # 日本語 1 文字（草・炎・水・雷・超・闘・悪・鋼・妖）
    name_en: str
    shape_description_ja: str  # 画像認識用：シンボルの形の説明
    shape_description_en: str  # 同上（英語）


# 9 種の基本エネルギータイプと、それぞれのシンボル形の説明（今後画像認識で参照する）
ENERGY_TYPES: tuple[EnergyTypeInfo, ...] = (
    EnergyTypeInfo(
        id="grass",
        name_ja="草",
        name_en="Grass",
        shape_description_ja="黒い円の中に暗緑色の三つ葉の葉（または植物の芽）の形。縁は緑の輪。",
        shape_description_en="Dark green leaf / three-lobed plant sprout in black circle, green outline.",
    ),
    EnergyTypeInfo(
        id="fire",
        name_ja="炎",
        name_en="Fire",
        shape_description_ja="黒い円の中に赤い炎。三つの上向きに曲がった炎の形。縁はオレンジの輪。",
        shape_description_en="Dark red flame with three upward-curving elements in black circle, orange outline.",
    ),
    EnergyTypeInfo(
        id="water",
        name_ja="水",
        name_en="Water",
        shape_description_ja="黒い円の中に青い水滴（涙型）。縁は青の輪。",
        shape_description_en="Dark blue teardrop / water droplet in black circle, blue outline.",
    ),
    EnergyTypeInfo(
        id="lightning",
        name_ja="雷",
        name_en="Lightning",
        shape_description_ja="黒い円の中に黒い稲妻の形。縁は黄色の輪。",
        shape_description_en="Black lightning bolt in black circle, yellow outline.",
    ),
    EnergyTypeInfo(
        id="psychic",
        name_ja="超",
        name_en="Psychic",
        shape_description_ja="黒い円の中に紫の目の形（虹彩・瞳孔あり）。縁は紫の輪。",
        shape_description_en="Stylized dark purple eye with iris and pupil in black circle, purple outline.",
    ),
    EnergyTypeInfo(
        id="fighting",
        name_ja="闘",
        name_en="Fighting",
        shape_description_ja="黒い円の中に茶色の握り拳（右向き）。縁はオレンジ〜茶の輪。",
        shape_description_en="Dark brown clenched fist facing right in black circle, orange-brown outline.",
    ),
    EnergyTypeInfo(
        id="darkness",
        name_ja="悪",
        name_en="Darkness",
        shape_description_ja="色付き円の中に黒いシンボル。大きな半月状の曲線と、その上部に小さな丸い点。皆既日食や闇夜の月を思わせる形。背景は濃い灰〜黒。",
        shape_description_en="Black symbol in colored circle: large crescent curve with small round dot above it, like eclipse or dark moon. Background dark gray to black.",
    ),
    EnergyTypeInfo(
        id="metal",
        name_ja="鋼",
        name_en="Metal",
        shape_description_ja="黒い円の中に薄い灰の歯車状（または三角に近い金属形、三つの頂点）。縁は灰白の輪。",
        shape_description_en="Light gray gear-like / triangular metallic symbol with three points in black circle.",
    ),
    EnergyTypeInfo(
        id="fairy",
        name_ja="妖",
        name_en="Fairy",
        shape_description_ja="黒い円の中にピンクのハート、または二つの水滴が絡まった形。縁はピンクの輪。",
        shape_description_en="Dark pink heart or two intertwined teardrops in black circle, pink outline.",
    ),
)


@dataclass(frozen=True)
class PokemonTypeInfo:
    """ポケモン 1 タイプの情報（表示名・画像認識用の形・基本エネルギーの有無）。"""
    id: str
    name_ja: str
    name_en: str
    shape_description_ja: str
    shape_description_en: str
    has_basic_energy: bool  # 基本エネルギーカードが存在するか（colorless / dragon は False）


# ポケモンタイプ全 11 種（エネルギー 9 種 + 無色・ドラゴン）。画像認識用の形の説明付き。
def _pokemon_types() -> tuple[PokemonTypeInfo, ...]:
    energy_by_id = {e.id: e for e in ENERGY_TYPES}
    return (
        *tuple(
            PokemonTypeInfo(
                id=e.id,
                name_ja=e.name_ja,
                name_en=e.name_en,
                shape_description_ja=e.shape_description_ja,
                shape_description_en=e.shape_description_en,
                has_basic_energy=True,
            )
            for e in ENERGY_TYPES
        ),
        PokemonTypeInfo(
            id="colorless",
            name_ja="無色",
            name_en="Colorless",
            shape_description_ja="銀〜明るい灰の円に、黒い六芒星（中心から放射状に 6 本の尖った光芒）。",
            shape_description_en="Black six-pointed star (asterisk-like rays) in silver or light gray circle.",
            has_basic_energy=False,
        ),
        PokemonTypeInfo(
            id="dragon",
            name_ja="ドラゴン",
            name_en="Dragon",
            shape_description_ja="金〜茶色の円に、左向きに流れる黒いシンボル。鋭い爪や翼を思わせる形。左上に小さな三角形の切り欠き。",
            shape_description_en="Black symbol flowing left, claw- or wing-like, with small triangular notch upper-left; circle background gold to brown.",
            has_basic_energy=False,
        ),
    )


POKEMON_TYPES: tuple[PokemonTypeInfo, ...] = _pokemon_types()


# 技のエネルギーコスト 1 スロットのタイプ（energy_cost_typed の要素）。無色＝任意の 1 エネルギー。
EnergyCostSlot = Literal[
    "grass", "fire", "water", "lightning", "psychic",
    "fighting", "darkness", "metal", "fairy", "colorless",
]


@dataclass
class Attack:
    name: str
    # 技に必要なエネルギー数（合計）。energy_cost_typed を指定する場合はその長さと一致させる。
    energy_cost: int
    damage: int
    self_damage: int = 0
    bench_damage: int = 0  # 相手のベンチ 1 体への追加ダメージ
    description: str = ""
    # 技に必要なエネルギーをタイプごとに指定（例: ["lightning", "colorless"] = 雷1＋無色1）。省略時は energy_cost だけの「無色のみ」扱い。
    energy_cost_typed: list[EnergyCostSlot] | None = None


@dataclass
class PokemonCard:
    id: str
    name: str
    type: CardType = "pokemon"
    hp: int = 110
    max_hp: int = 110
    attacks: list[Attack] = field(default_factory=list)
    evolves_from: str | None = None  # 進化元（例: "meguroko"）
    retreat_cost: int = 1  # にげるために捨てるエネルギー数。無色は任意の 1 エネルギーでよい。
    instance_id: str = ""
    # ポケモンのタイプ（草・炎・水など）。弱点判定で攻撃側のタイプとして参照する。
    pokemon_type: PokemonTypeId | None = None
    # 弱点。このタイプのポケモンから技を受けると、受けるダメージが 2 倍になる。
    weakness: PokemonTypeId | None = None

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
            pokemon_type=self.pokemon_type,
            weakness=self.weakness,
        )


@dataclass
class EnergyCard:
    id: str
    name: str
    type: CardType = "energy"
    provides: int = 1
    instance_id: str = ""
    # このエネルギーが付与するタイプ。None の場合は無色（任意の 1 エネルギーとして扱う）。
    energy_type: EnergyTypeId | None = None


@dataclass
class ItemCard:
    id: str
    name: str
    type: CardType = "item"
    effect: str = "heal"
    heal_amount: int = 20
    description: str = ""
    instance_id: str = ""


@dataclass
class SupportCard:
    """サポート（1 ターンに 1 枚まで使用可。先行の 1 ターン目は原則使用不可）。"""
    id: str
    name: str
    type: CardType = "support"
    effect: str = "draw_3"
    draw_count: int = 3
    description: str = ""
    instance_id: str = ""


def is_pokemon(card) -> bool:
    return getattr(card, "type", None) == "pokemon"


def is_energy(card) -> bool:
    return getattr(card, "type", None) == "energy"


def is_item(card) -> bool:
    return getattr(card, "type", None) == "item"


def is_support(card) -> bool:
    return getattr(card, "type", None) == "support"
