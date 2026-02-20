"""
デッキ定義（レシピ・デッキ生成・表示）。

デッキ A/B/C/D/E のレシピ、create_deck / format_deck_recipe を提供。
"""
from card import CARD_ID_TO_NAME, get_card_by_id

# デッキ A（14 枚）: オタチ 3、オオタチ 2、モトトカゲ 2、エネルギー 6、ネモ 1
DECK_RECIPE_A = {"otachi": 3, "ootachi": 2, "mototokage": 2, "basic-energy": 6, "nemo": 1}
# デッキ B（15 枚）: メグロコ 3、ワルビル 2、モトトカゲ 2、エネルギー 6、きずぐすり 1
DECK_RECIPE_B = {"meguroko": 3, "warubiru": 2, "mototokage": 2, "basic-energy": 6, "potion": 1}
# デッキ C（15 枚）: ズピカ 3、ハラバリー 2、モトトカゲ 2、無色 4・雷 2、きずぐすり 1
DECK_RECIPE_C = {"zupika": 3, "harabarii": 2, "mototokage": 2, "basic-energy": 4, "basic-energy-lightning": 2, "potion": 1, "nemo": 4}
# デッキ D（26 枚）: メグロコ 3、ワルビル 2、ワルビアル 2、リククラゲ 2、ノノクラゲ 3、ガケガニ 2、エネルギー 10、ポケモンいれかえ 2
DECK_RECIPE_D = {"meguroko-svd-062": 3, "warubiru-svd-063": 2, "warubiaru-svd-064": 2, "rikukurage-svd-066": 2, "nonokurage-svd-065": 3, "gakegani-svd-067": 2, "basic-energy-fighting": 8, "nemo": 4, "potion": 2, "pokemon_irekae": 2}
# デッキ E（26 枚）: ジバコイル 2, レアコイル 2, コイル 3, ズピカ 3、ハラバリー 2、 カラミンゴ 2、 エネルギー 10、ポケモンいれかえ 2
DECK_RECIPE_E = {"jibakoil-svd-038": 2, "rarecoil-svd-037": 2, "coil-svd-036": 3, "zupika": 3, "harabarii": 2, "karamingo-svg-029": 2, "basic-energy-lightning": 8, "nemo": 4, "potion": 2, "pokemon_irekae": 2}

DECK_RECIPES = [DECK_RECIPE_A, DECK_RECIPE_B, DECK_RECIPE_C, DECK_RECIPE_D, DECK_RECIPE_E]
DECK_SIZE = 13  # 最大デッキ枚数（デッキにより 12 または 13）
STARTING_HAND_SIZE = 4


def format_deck_recipe(deck_index: int) -> str:
    """デッキ番号（0=A, 1=B, 2=C, 3=D, 4=E）のレシピを「カード名枚数, ...」の文字列で返す。"""
    recipe = DECK_RECIPES[deck_index % len(DECK_RECIPES)]
    names = [f"{CARD_ID_TO_NAME.get(cid, cid)}{n}" for cid, n in recipe.items()]
    return "、".join(names)


def create_deck(deck_index: int = 0) -> list:
    """指定デッキを生成（0=A, 1=B, 2=C, 3=D, 4=E）。各カードにユニーク instance_id を付与。"""
    recipe = DECK_RECIPES[deck_index % len(DECK_RECIPES)]
    deck = []
    uid = 0
    for card_id, count in recipe.items():
        for _ in range(count):
            deck.append(get_card_by_id(card_id, f"card-{uid}"))
            uid += 1
    return deck
