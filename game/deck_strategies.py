"""
デッキごとの戦略（条件・優先度）を定義する。

registered_decks.json の strategy で指定した ID に応じ、ベンチ出しの優先ポケモンや
山札検索時のボーナスなどが変わる。ルカリオデッキ以外ではこれらの条件は適用されない。
"""
from deck import get_deck_strategy

STRATEGY_PRIORITY_SETUP_IDS: dict[str, list[str]] = {
    "lucario": [
        "sorurokku-mc-372",   # ソルロック最優先（コスモビームで序盤から殴れる）
        "rioru-svd-059",      # リオル（メガルカリオexに進化して殴れる）
        "rioru-sv1a-091",
        "rioru-mil-068",
        "runaton",            # ルナトーン（攻撃力低い、スタートさせたくない）
        # マクノシタは含まない（逃げコスト2で序盤にスタートすると事故る）
    ],
    "dragapult": [
        "subomi-s2a-011",      # スボミー最優先（むずむずかふんでグッズロック）
        "dorameshiya-mc-546",  # ドラメシヤ（ドロンチ→ドラパルトexに進化）
        "yomawaru-mc-308",     # ヨマワル（サマヨール→ヨノワールのカースドボム用）
        # ニャースex/キチキギスex/マシマシラはベンチ枠を圧迫するので序盤は出さない
    ],
}

STRATEGY_FETCH_BONUS_BY_CARD: dict[str, dict[str, float]] = {
    "lucario": {
        "sorurokku-mc-372": 500.0,
        "runaton": 500.0,
        "rioru-svd-059": 300.0,
        "rioru-sv1a-091": 300.0,
        "rioru-mil-068": 300.0,
    },
    "dragapult": {
        "subomi-s2a-011": 500.0,       # スボミー（むずむずかふんでグッズロック）
        "dorameshiya-mc-546": 600.0,   # ドラメシヤ（ドロンチ→ドラパルトexに進化）
        "doronchi-mc-547": 700.0,      # ドロンチ（ていさつしれい、次ターン進化可能）
        "doraparutoex-mc-548": 800.0,  # ドラパルトex（メインアタッカー）
        "yomawaru-mc-308": 400.0,      # ヨマワル（カースドボム用）
        "samayoru-mc-309": 300.0,      # サマヨール（カースドボム50）
        "yonowaru-mc-310": 350.0,      # ヨノワール（カースドボム130）
    },
}

STRATEGY_ALLOW_DUPLICATE_BENCH_IDS: dict[str, list[str]] = {
    "lucario": ["rioru-svd-059", "rioru-sv1a-091", "rioru-mil-068"],
    "dragapult": ["dorameshiya-mc-546", "yomawaru-mc-308"],  # ドラメシヤ3体+ヨマワル1体が理想
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


def get_allow_duplicate_bench_ids(deck_index: int) -> set[str]:
    """このデッキの戦略でベンチに同名を複数出してよい card id の集合。ルカリオデッキではリオルを複数枚候補にできるが、場に出す枚数は state 側で原則 2・例外 3 に制限。"""
    strategy = get_deck_strategy(deck_index)
    if not strategy:
        return set()
    return set(STRATEGY_ALLOW_DUPLICATE_BENCH_IDS.get(strategy, []))


def is_dragapult_deck(deck_index: int) -> bool:
    """ドラパルトexデッキか。"""
    return get_deck_strategy(deck_index) == "dragapult"


def is_dragapult_deck_for_player(state, player_index: int) -> bool:
    """指定プレイヤーがドラパルトexデッキか。"""
    if not hasattr(state, "deck_indices") or not state.deck_indices:
        return False
    idx = state.deck_indices[player_index] if player_index < len(state.deck_indices) else 0
    return is_dragapult_deck(idx)


# ─── ドラパルトexデッキ共通定数 ───
# 各ファイルで重複定義されていたfrozensetを集約

# ドラパルトexの進化ライン（エネ付与・進化の優先対象）
DRAPA_LINE_NAMES: frozenset[str] = frozenset({"ドラパルトex", "ドロンチ", "ドラメシヤ"})

# サポートポケモン（エネ付与を避ける対象）
DRAPA_SUPPORT_NAMES: frozenset[str] = frozenset({
    "スボミー", "キチキギスex", "ニャースex",
    "ヨマワル", "サマヨール", "ヨノワール", "マシマシラ",
})

# エネルギー付与を完全禁止するポケモン
DRAPA_ENERGY_BANNED_NAMES: frozenset[str] = frozenset({"スボミー"})


def is_drapa_line(card_name: str) -> bool:
    """ドラパルトexの進化ライン（ドラメシヤ/ドロンチ/ドラパルトex）か。"""
    return card_name in DRAPA_LINE_NAMES


def is_drapa_support(card_name: str) -> bool:
    """ドラパルトデッキのサポートポケモンか（エネ付与を避ける対象）。"""
    return card_name in DRAPA_SUPPORT_NAMES


def is_drapa_energy_banned(card_name: str) -> bool:
    """エネルギー付与を完全禁止するポケモンか。"""
    return card_name in DRAPA_ENERGY_BANNED_NAMES
