"""
ふしぎなアメの挙動確認。
手札に「ふしぎなアメ」「ワルビアル（2進化）」「ワルビル（1進化）」、
バトル場に「メグロコ（たね）」がある状態でふしぎなアメを使い、
メグロコがワルビアルに 1 進化をとばして進化することを確認する。
"""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from game import GameState, BattlePokemon, use_trainer_goods
from card import get_card_by_id


def test_fushiginaame_evolve_basic_to_stage2():
    """ふしぎなアメでたねポケモンが 2 進化に 1 進化をとばして進化する。"""
    state = GameState(
        current_player=0,
        first_player=0,
        turn_count=2,
        log_fn=lambda _: None,
    )
    p0 = state.players[0]
    p1 = state.players[1]

    p0.hand = [
        get_card_by_id("fushiginaame", "hand-fushiginaame"),
        get_card_by_id("warubiaru-svd-064", "hand-warubiaru"),
        get_card_by_id("warubiru-svd-063", "hand-warubiru"),
    ]
    p0.active = BattlePokemon(
        card=get_card_by_id("meguroko-svd-062", "active-meguroko"),
        put_on_bench_this_turn=False,
    )
    p0.bench = []
    p0.deck = []
    p0.discard = []
    p0.prize_pile = []

    p1.hand = []
    p1.active = None
    p1.bench = []
    p1.deck = []
    p1.discard = []
    p1.prize_pile = []

    hand_index_fushiginaame = 0
    ok = use_trainer_goods(state, hand_index_fushiginaame)

    assert ok, "ふしぎなアメの使用に失敗"
    assert p0.active is not None, "バトル場が空"
    assert p0.active.card.name == "ワルビアル", f"進化後の名前がワルビアルでない: {p0.active.card.name}"
    assert len(p0.hand) == 1, f"手札はワルビル 1 枚のはず: {len(p0.hand)}"
    assert any(getattr(c, "id", "") == "fushiginaame" for c in p0.discard), "ふしぎなアメが捨て札に入っていること"
    assert not any(getattr(c, "name", "") == "ワルビアル" for c in p0.hand), "ワルビアルは場にのったので手札に残っていないこと"


def test_fushiginaame_stage1_not_in_hand():
    """ふしぎなアメで 1 進化が手札にない場合もレジストリの名前検索で使える（ジバコイル→コイル）。"""
    state = GameState(
        current_player=0,
        first_player=0,
        turn_count=2,
        log_fn=lambda _: None,
    )
    p0 = state.players[0]
    p1 = state.players[1]

    # 手札に 2 進化（ジバコイル）とふしぎなアメのみ。レアコイルは手札にない。
    p0.hand = [
        get_card_by_id("fushiginaame", "hand-fushiginaame"),
        get_card_by_id("jibakoil-svd-038", "hand-jibakoil"),
    ]
    # ベンチにたねのコイル（レアコイルの進化元）
    p0.active = None
    p0.bench = [
        BattlePokemon(
            card=get_card_by_id("coil-svd-036", "bench-coil"),
            put_on_bench_this_turn=False,
        ),
    ]
    p0.deck = []
    p0.discard = []
    p0.prize_pile = []

    p1.hand = []
    p1.active = None
    p1.bench = []
    p1.deck = []
    p1.discard = []
    p1.prize_pile = []

    ok = use_trainer_goods(state, 0)  # ふしぎなアメを使用

    assert ok, "ふしぎなアメの使用に失敗（1 進化が手札にない場合は get_card_by_name で解決）"
    assert len(p0.bench) == 1, "ベンチは 1 体のまま"
    assert p0.bench[0].card.name == "ジバコイル", f"コイルがジバコイルに進化していること: {p0.bench[0].card.name}"
    assert any(getattr(c, "id", "") == "fushiginaame" for c in p0.discard), "ふしぎなアメが捨て札に入っていること"
