"""
デッキごとの戦略（条件・優先度）を定義する。

registered_decks.json の strategy で指定した ID に応じ、ベンチ出しの優先ポケモンや
山札検索時のボーナスなどが変わる。ルカリオデッキ以外ではこれらの条件は適用されない。
"""
from deck import get_deck_strategy

STRATEGY_PRIORITY_SETUP_IDS: dict[str, list[str]] = {
    "lucario": ["sorurokku-mc-372", "runaton"],
}

STRATEGY_FETCH_BONUS_BY_CARD: dict[str, dict[str, float]] = {
    "lucario": {"sorurokku-mc-372": 500.0, "runaton": 500.0},
}


def get_priority_setup_pokemon_ids(deck_index: int) -> list[str]:
    """このデッキの戦略で「先に揃えたい」ポケモンの card id リストを返す。該当なしなら []。"""
    strategy = get_deck_strategy(deck_index)
    if not strategy:
        return []
    return list(STRATEGY_PRIORITY_SETUP_IDS.get(strategy, []))


def get_fetch_bonus_for_card(deck_index: int, card_id: str) -> float:
    """このデッキの戦略で、山札からこのカードを取るときに加算するボーナス。該当なしなら 0。"""
    strategy = get_deck_strategy(deck_index)
    if not strategy:
        return 0.0
    bonuses = STRATEGY_FETCH_BONUS_BY_CARD.get(strategy, {})
    return bonuses.get(card_id, 0.0)
