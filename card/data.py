"""
カードのマスタデータ。

各カードの実体（OTACHI, MEGUROKO 等）、レジストリ、get_card_by_id / CARD_ID_TO_NAME を提供。
"""
from copy import deepcopy

from card.model import Attack, EnergyCard, GoodsCard, is_goods, is_support, PokemonCard, SupportCard

# ----- マスタカード -----

OTACHI = PokemonCard(
    id="otachi",
    name="オタチ",
    hp=60,
    max_hp=60,
    attacks=[
        Attack("ひらてうち", 1, 20, 0, 0, "20 ダメージ"),
    ],
    evolves_from=None,
    evolution_stage="basic",
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
    evolution_stage="stage1",
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
    evolution_stage="basic",
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
    evolution_stage="basic",
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
    evolution_stage="stage1",
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
    evolution_stage="basic",
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
    evolution_stage="stage1",
    retreat_cost=2,
    pokemon_type="lightning",
    weakness="fighting",
)

# ----- 以下 JSON から生成（update_cards_from_json.py で上書き） -----
# ウソッキー（usokki-svd-058）
USOKKI_SVD_058 = PokemonCard(
    id='usokki-svd-058',
    name='ウソッキー',
    hp=110,
    max_hp=110,
    attacks=[
        Attack('なぐってかくれる', 1, 20, 0, 0, 'コインを1回投げオモテなら、次の相手の番、このポケモンはワザのダメージや効果を受けない。', energy_cost_typed=['fighting']),
        Attack('ひじうち', 3, 90, 0, 0, '', energy_cost_typed=['colorless', 'colorless', 'colorless']),
    ],
    evolves_from=None,
    evolution_stage='basic',
    retreat_cost=2,
    pokemon_type='fighting',
    regulation='G',
    weakness='grass',
)

# カイデン（kaiden-svd-044）
KAIDEN_SVD_044 = PokemonCard(
    id='kaiden-svd-044',
    name='カイデン',
    hp=60,
    max_hp=60,
    attacks=[
        Attack('ふきあらす', 1, 0, 0, 0, '相手は相手自身の手札をすべて山札にもどして切る。その後、相手は山札を4枚引く。', energy_cost_typed=['colorless']),
        Attack('はばたく', 3, 40, 0, 0, '', energy_cost_typed=['colorless', 'colorless', 'colorless']),
    ],
    evolves_from=None,
    evolution_stage='basic',
    retreat_cost=1,
    pokemon_type='lightning',
    regulation='G',
    weakness='lightning',
)

# カラミンゴ（karamingo-svd-109）
KARAMINGO_SVD_109 = PokemonCard(
    id='karamingo-svd-109',
    name='カラミンゴ',
    hp=110,
    max_hp=110,
    attacks=[
        Attack('はばたく', 1, 30, 0, 0, '', energy_cost_typed=['colorless']),
        Attack('きゅうこうか', 3, 110, 20, 0, 'このポケモンにも20ダメージ。', energy_cost_typed=['colorless', 'colorless', 'colorless']),
    ],
    evolves_from=None,
    evolution_stage='basic',
    retreat_cost=1,
    pokemon_type='colorless',
    regulation='G',
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
    evolution_stage='basic',
    retreat_cost=3,
    pokemon_type='fighting',
    regulation='G',
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
    evolution_stage='basic',
    retreat_cost=1,
    pokemon_type='lightning',
    regulation='G',
    weakness='fighting',
)

# コライドンex（koraidonex-svd-068）
KORAIDONEX_SVD_068 = PokemonCard(
    id='koraidonex-svd-068',
    name='コライドンex',
    hp=230,
    max_hp=230,
    attacks=[
        Attack('スプリットビーム', 1, 20, 0, 20, '相手のベンチポケモン2匹にも、それぞれ20ダメージ。[ベンチは弱点・抵抗力を計算しない。]', bench_damage_count=2, energy_cost_typed=['fighting']),
        Attack('ガイアプレス', 3, 230, 30, 0, 'このポケモンにも30ダメージ。', energy_cost_typed=['fighting', 'fighting', 'colorless']),
    ],
    evolves_from=None,
    evolution_stage='basic',
    retreat_cost=2,
    pokemon_type='fighting',
    regulation='G',
    weakness='psychic',
    is_ex=True,
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
    evolves_from='レアコイル',
    evolution_stage='stage2',
    retreat_cost=2,
    pokemon_type='lightning',
    regulation='G',
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
    evolution_stage='basic',
    retreat_cost=1,
    pokemon_type='lightning',
    regulation='G',
    weakness='fighting',
)

# ノコッチ（nokotchi-svd-092）
NOKOTCHI_SVD_092 = PokemonCard(
    id='nokotchi-svd-092',
    name='ノコッチ',
    hp=70,
    max_hp=70,
    attacks=[
        Attack('ともだちをさがす', 1, 0, 0, 0, '自分の山札からポケモンを1枚選び、相手に見せて、手札に加える。そして山札を切る。', energy_cost_typed=['colorless']),
        Attack('かみつく', 3, 50, 0, 0, '', energy_cost_typed=['colorless', 'colorless', 'colorless']),
    ],
    evolves_from=None,
    evolution_stage='basic',
    retreat_cost=1,
    pokemon_type='colorless',
    regulation='G',
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
    evolution_stage='basic',
    retreat_cost=1,
    pokemon_type='fighting',
    regulation='G',
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
    evolves_from='ズピカ',
    evolution_stage='stage1',
    retreat_cost=2,
    pokemon_type='lightning',
    regulation='G',
    weakness='fighting',
)

# バチンウニ（bachinuni-svd-040）
BACHINUNI_SVD_040 = PokemonCard(
    id='bachinuni-svd-040',
    name='バチンウニ',
    hp=90,
    max_hp=90,
    attacks=[
        Attack('しびれはり', 1, 20, 0, 0, 'コインを1回投げオモテなら、相手のバトルポケモンをマヒにする。', energy_cost_typed=['lightning'], status_effect='paralysis', status_effect_on_coin_heads=True),
        Attack('ついげきバリバリ', 1, 100, 0, 0, 'このワザは、前の自分の番に、このポケモンが「しびれはり」を使っていなければ使えない。', energy_cost_typed=['lightning']),
    ],
    evolves_from=None,
    evolution_stage='basic',
    retreat_cost=3,
    pokemon_type='lightning',
    regulation='G',
    weakness='fighting',
)

# ピカチュウ（pikachixyuu-svd-034）
PIKACHIXYUU_SVD_034 = PokemonCard(
    id='pikachixyuu-svd-034',
    name='ピカチュウ',
    hp=70,
    max_hp=70,
    attacks=[
        Attack('なきごえ', 1, 0, 0, 0, '次の相手の番、このワザを受けたポケモンが使うワザのダメージは「-20」される。', energy_cost_typed=['colorless']),
        Attack('ピカボルト', 2, 30, 0, 0, '', energy_cost_typed=['lightning', 'colorless']),
    ],
    evolves_from=None,
    evolution_stage='basic',
    retreat_cost=1,
    pokemon_type='lightning',
    regulation='G',
    weakness='fighting',
)

# ミライドンex（miraidonex-svd-046）
MIRAIDONEX_SVD_046 = PokemonCard(
    id='miraidonex-svd-046',
    name='ミライドンex',
    hp=220,
    max_hp=220,
    attacks=[
        Attack('クイックドロー', 1, 20, 0, 0, '自分の山札を2枚引く。', energy_cost_typed=['lightning']),
        Attack('テクノターボ', 3, 150, 0, 0, '自分のトラッシュから「基本 雷エネルギー」を1枚選び、ベンチポケモンにつける。', energy_cost_typed=['lightning', 'lightning', 'colorless']),
    ],
    evolves_from=None,
    evolution_stage='basic',
    retreat_cost=1,
    pokemon_type='lightning',
    regulation='G',
    weakness='fighting',
    is_ex=True,
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
    evolution_stage='basic',
    retreat_cost=2,
    pokemon_type='fighting',
    regulation='G',
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
    evolution_stage='basic',
    retreat_cost=1,
    pokemon_type='colorless',
    regulation='H',
    weakness='fighting',
)

# ライチュウ（raichixyuu-svd-035）
RAICHIXYUU_SVD_035 = PokemonCard(
    id='raichixyuu-svd-035',
    name='ライチュウ',
    hp=140,
    max_hp=140,
    attacks=[
        Attack('エレキチャージ', 1, 0, 0, 0, '自分の山札から「基本エネルギー」を2枚まで選び、このポケモンにつける。そして山札を切る。', energy_cost_typed=['colorless']),
        Attack('10まんボルト', 3, 200, 0, 0, 'このポケモンについているエネルギーを、すべてトラッシュする。', energy_cost_typed=['lightning', 'lightning', 'colorless']),
    ],
    evolves_from='ピカチュウ',
    evolution_stage='stage1',
    retreat_cost=1,
    pokemon_type='lightning',
    regulation='G',
    weakness='fighting',
)

# リオル（rioru-svd-059）
RIORU_SVD_059 = PokemonCard(
    id='rioru-svd-059',
    name='リオル',
    hp=70,
    max_hp=70,
    attacks=[
        Attack('パンチ', 1, 10, 0, 0, '', energy_cost_typed=['fighting']),
        Attack('とつげき', 2, 50, 20, 0, 'このポケモンにも20ダメージ。', energy_cost_typed=['fighting', 'colorless']),
    ],
    evolves_from=None,
    evolution_stage='basic',
    retreat_cost=1,
    pokemon_type='fighting',
    regulation='G',
    weakness='psychic',
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
    evolves_from='ノノクラゲ',
    evolution_stage='stage1',
    retreat_cost=2,
    pokemon_type='fighting',
    regulation='G',
    weakness='grass',
)

# ルカリオ（rukario-svd-060）
RUKARIO_SVD_060 = PokemonCard(
    id='rukario-svd-060',
    name='ルカリオ',
    hp=130,
    max_hp=130,
    attacks=[
        Attack('アベンジナックル', 1, 30, 0, 0, '前の相手の番に、ワザのダメージで、自分の闘ポケモンがきぜつしていたなら、120ダメージ追加。', energy_cost_typed=['fighting']),
        Attack('かそくづき', 3, 120, 0, 0, '次の自分の番、このポケモンは「かそくづき」が使えない。', energy_cost_typed=['colorless', 'colorless', 'colorless']),
    ],
    evolves_from='リオル',
    evolution_stage='stage1',
    retreat_cost=2,
    pokemon_type='fighting',
    regulation='G',
    weakness='psychic',
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
    evolves_from='コイル',
    evolution_stage='stage1',
    retreat_cost=2,
    pokemon_type='lightning',
    regulation='G',
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
        Attack('じしん', 2, 180, 0, 30, '自分のベンチポケモン全員にも、それぞれ30ダメージ。[ベンチは弱点・抵抗力を計算しない。]', bench_damage_count=0, bench_damage_target='self', energy_cost_typed=['fighting', 'fighting']),
    ],
    evolves_from='ワルビル',
    evolution_stage='stage2',
    retreat_cost=3,
    pokemon_type='fighting',
    regulation='G',
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
    evolves_from='メグロコ',
    evolution_stage='stage1',
    retreat_cost=2,
    pokemon_type='fighting',
    regulation='G',
    weakness='grass',
)

# おとどけドローン（otodokedoron）
OTODOKEDORON = GoodsCard(
    id='otodokedoron',
    name='おとどけドローン',
    description='コインを2回投げ、すべてオモテなら、自分の山札から好きなカードを1枚選び、手札に加える。そして山札を切る。',
)

# たんぱんこぞう（tanpankozou）
TANPANKOZOU = SupportCard(
    id='tanpankozou',
    name='たんぱんこぞう',
    description='自分の手札をすべて山札にもどして切る。その後、山札を5枚引く。',
)

# ふしぎなアメ（fushiginaame）
FUSHIGINAAME = GoodsCard(
    id='fushiginaame',
    name='ふしぎなアメ',
    description='自分の手札から2進化ポケモンを1枚選び、そのポケモンへと進化する自分の場のたねポケモンにのせ、1進化をとばして進化させる。（最初の自分の番や、出したばかりのポケモンには使えない。）',
)

# エネルギー回収（unknown）
UNKNOWN = GoodsCard(
    id='unknown',
    name='エネルギー回収',
    description='自分のトラッシュから基本エネルギーを2枚まで選び、相手に見せて、手札に加える。',
)

# エレキジェネレーター（erekijienereta）
EREKIJIENERETA = GoodsCard(
    id='erekijienereta',
    name='エレキジェネレーター',
    description='自分の山札を上から5枚見て、その中から「基本エネルギー」を2枚まで選び、ベンチのポケモンに好きなようにつける。残りのカードは山札にもどして切る。',
)

# キハダ（kihada）
KIHADA = SupportCard(
    id='kihada',
    name='キハダ',
    description='自分の手札を1枚選び、山札の下にもどす。その後、自分の手札が5枚になるように、山札を引く。（自分の手札がこのカード1枚だけなら、このカードは使えない。）',
)

# ジャッジマン（jixyajjiman）
JIXYAJJIMAN = SupportCard(
    id='jixyajjiman',
    name='ジャッジマン',
    description='おたがいのプレイヤーは、それぞれ手札をすべて山札にもどして切る。その後、それぞれ山札を4枚引く。',
)

# スーパーボール（supaboru）
SUPABORU = GoodsCard(
    id='supaboru',
    name='スーパーボール',
    description='自分の山札を上から7枚見て、その中からポケモンを1枚選び、相手に見せて、手札に加える。残りのカードは山札にもどして切る。',
)

# ネモ（nemokako）
NEMOKAKO = SupportCard(
    id='nemokako',
    name='ネモ',
    description='自分の山札を3枚引く。',
)

# ネモ（nemomirai）
NEMOMIRAI = SupportCard(
    id='nemomirai',
    name='ネモ',
    description='自分の山札を3枚引く。',
)

# ハイパーボール（haipaboru）
HAIPABORU = GoodsCard(
    id='haipaboru',
    name='ハイパーボール',
    description='このカードは、自分の手札を2枚トラッシュしなければ使えない。自分の山札からポケモンを1枚選び、相手に見せて、手札に加える。そして山札を切る。',
)

# ポケモンいれかえ（pokemonirekae）
POKEMONIREKAE = GoodsCard(
    id='pokemonirekae',
    name='ポケモンいれかえ',
    description='自分のバトルポケモンをベンチポケモンと入れ替える。',
)

# ポケモンキャッチャー（pokemonkixyatchixya）
POKEMONKIXYATCHIXYA = GoodsCard(
    id='pokemonkixyatchixya',
    name='ポケモンキャッチャー',
    description='コインを1回投げオモテなら、相手のベンチポケモンを1匹選び、バトルポケモンと入れ替える。',
)

# 博士の研究（hakasenokenkyuu）
HAKASENOKENKYUU = SupportCard(
    id='hakasenokenkyuu',
    name='博士の研究',
    description='自分の手札をすべてトラッシュし、山札を7枚引く。',
)

# 博士の研究（hakasenokenkixyuufutouhakase）
HAKASENOKENKIXYUUFUTOUHAKASE = SupportCard(
    id='hakasenokenkixyuufutouhakase',
    name='博士の研究',
    description='自分の手札をすべてトラッシュし、山札を7枚引く。',
)

# 岩のむねあて（iwanomuneate）
IWANOMUNEATE = GoodsCard(
    id='iwanomuneate',
    name='岩のむねあて',
    effect='tool',
    description='このカードをつけている闘ポケモンが、相手のポケモンから受けるワザのダメージは「-30」される。',
    is_tool=True,
    tool_damage_reduce=30,
    tool_condition_type='fighting',
)

# 基本闘エネルギー（kihontouenerugi）
KIHONTOUENERUGI = EnergyCard(
    id='kihontouenerugi',
    name='基本闘エネルギー',
    energy_type='fighting',
)

# 基本雷エネルギー（kihonkaminarienerugi）
KIHONKAMINARIENERUGI = EnergyCard(
    id='kihonkaminarienerugi',
    name='基本雷エネルギー',
    energy_type='lightning',
)

# ----- 以上 JSON から生成 -----

















BASIC_ENERGY = EnergyCard(id="basic-energy", name="基本エネルギー", provides=1, energy_type=None)
BASIC_ENERGY_LIGHTNING = EnergyCard(id="basic-energy-lightning", name="基本雷エネルギー", provides=1, energy_type="lightning")
BASIC_ENERGY_FIGHTING = EnergyCard(id="basic-energy-fighting", name="基本闘エネルギー", provides=1, energy_type="fighting")

POTION = GoodsCard(
    id="potion",
    name="きずぐすり",
    effect="heal",
    heal_amount=30,
    description="自分のバトル場のポケモンを 30 回復する",
)

POKEMON_IREKAE = GoodsCard(
    id="pokemon_irekae",
    name="ポケモンいれかえ",
    effect="swap_active",
    description="自分のバトル場のポケモンとベンチのポケモンを 1 体入れ替える",
)

IWANOMUNEATE = GoodsCard(
    id="iwanomuneate",
    name="岩のむねあて",
    effect="tool",
    description="このカードをつけている闘ポケモンが、相手のポケモンから受けるワザのダメージは「-30」される。",
    is_tool=True,
    tool_damage_reduce=30,
    tool_condition_type="fighting",
)

NEMO = SupportCard(
    id="nemo",
    name="ネモ",
    effect="draw_3",
    draw_count=3,
    description="デッキからカードを 3 枚引く",
)

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
    "usokki-svd-058": 'ウソッキー',
    "kaiden-svd-044": 'カイデン',
    "karamingo-svd-109": 'カラミンゴ',
    "gakegani-svd-067": 'ガケガニ',
    "coil-svd-036": 'コイル',
    "koraidonex-svd-068": 'コライドンex',
    "jibakoil-svd-038": 'ジバコイル',
    "zupika-svd-041": 'ズピカ',
    "nokotchi-svd-092": 'ノコッチ',
    "nonokurage-svd-065": 'ノノクラゲ',
    "harabarii-svd-042": 'ハラバリー',
    "bachinuni-svd-040": 'バチンウニ',
    "pikachixyuu-svd-034": 'ピカチュウ',
    "miraidonex-svd-046": 'ミライドンex',
    "meguroko-svd-062": 'メグロコ',
    "mototokage-mc-627": 'モトトカゲ',
    "raichixyuu-svd-035": 'ライチュウ',
    "rioru-svd-059": 'リオル',
    "rikukurage-svd-066": 'リククラゲ',
    "rukario-svd-060": 'ルカリオ',
    "rarecoil-svd-037": 'レアコイル',
    "warubiaru-svd-064": 'ワルビアル',
    "warubiru-svd-063": 'ワルビル',
    "otodokedoron": 'おとどけドローン',
    "tanpankozou": 'たんぱんこぞう',
    "fushiginaame": 'ふしぎなアメ',
    "unknown": 'エネルギー回収',
    "erekijienereta": 'エレキジェネレーター',
    "kihada": 'キハダ',
    "jixyajjiman": 'ジャッジマン',
    "supaboru": 'スーパーボール',
    "nemokako": 'ネモ',
    "nemomirai": 'ネモ',
    "haipaboru": 'ハイパーボール',
    "pokemonirekae": 'ポケモンいれかえ',
    "pokemonkixyatchixya": 'ポケモンキャッチャー',
    "hakasenokenkyuu": '博士の研究',
    "hakasenokenkixyuufutouhakase": '博士の研究',
    "iwanomuneate": '岩のむねあて',
    "kihontouenerugi": '基本闘エネルギー',
    "kihonkaminarienerugi": '基本雷エネルギー',
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
    "usokki-svd-058": USOKKI_SVD_058,
    "kaiden-svd-044": KAIDEN_SVD_044,
    "karamingo-svd-109": KARAMINGO_SVD_109,
    "gakegani-svd-067": GAKEGANI_SVD_067,
    "coil-svd-036": COIL_SVD_036,
    "koraidonex-svd-068": KORAIDONEX_SVD_068,
    "jibakoil-svd-038": JIBAKOIL_SVD_038,
    "zupika-svd-041": ZUPIKA_SVD_041,
    "nokotchi-svd-092": NOKOTCHI_SVD_092,
    "nonokurage-svd-065": NONOKURAGE_SVD_065,
    "harabarii-svd-042": HARABARII_SVD_042,
    "bachinuni-svd-040": BACHINUNI_SVD_040,
    "pikachixyuu-svd-034": PIKACHIXYUU_SVD_034,
    "miraidonex-svd-046": MIRAIDONEX_SVD_046,
    "meguroko-svd-062": MEGUROKO_SVD_062,
    "mototokage-mc-627": MOTOTOKAGE_MC_627,
    "raichixyuu-svd-035": RAICHIXYUU_SVD_035,
    "rioru-svd-059": RIORU_SVD_059,
    "rikukurage-svd-066": RIKUKURAGE_SVD_066,
    "rukario-svd-060": RUKARIO_SVD_060,
    "rarecoil-svd-037": RARECOIL_SVD_037,
    "warubiaru-svd-064": WARUBIARU_SVD_064,
    "warubiru-svd-063": WARUBIRU_SVD_063,
    "otodokedoron": OTODOKEDORON,
    "tanpankozou": TANPANKOZOU,
    "fushiginaame": FUSHIGINAAME,
    "unknown": UNKNOWN,
    "erekijienereta": EREKIJIENERETA,
    "kihada": KIHADA,
    "jixyajjiman": JIXYAJJIMAN,
    "supaboru": SUPABORU,
    "nemokako": NEMOKAKO,
    "nemomirai": NEMOMIRAI,
    "haipaboru": HAIPABORU,
    "pokemonirekae": POKEMONIREKAE,
    "pokemonkixyatchixya": POKEMONKIXYATCHIXYA,
    "hakasenokenkyuu": HAKASENOKENKYUU,
    "hakasenokenkixyuufutouhakase": HAKASENOKENKIXYUUFUTOUHAKASE,
    "iwanomuneate": IWANOMUNEATE,
    "kihontouenerugi": KIHONTOUENERUGI,
    "kihonkaminarienerugi": KIHONKAMINARIENERUGI,
    # ----- 以上 JSON から生成 -----
}


def get_card_by_id(card_id: str, instance_id: str = "") -> PokemonCard | EnergyCard | GoodsCard | SupportCard:
    """カード ID からマスタのコピーを生成する。instance_id を付与。"""
    if card_id not in _CARD_REGISTRY:
        raise ValueError(f"Unknown card id: {card_id}")
    c = deepcopy(_CARD_REGISTRY[card_id])
    c.instance_id = instance_id
    return c


def get_card_by_name(name_ja: str, instance_id: str = "") -> PokemonCard | EnergyCard | GoodsCard | SupportCard | None:
    """名前（日本語）が一致するカードをレジストリから探しコピーを返す。見つからなければ None。"""
    name_ja = (name_ja or "").strip()
    if not name_ja:
        return None
    for c in _CARD_REGISTRY.values():
        if getattr(c, "name", "") == name_ja:
            out = deepcopy(c)
            out.instance_id = instance_id
            return out
    return None
