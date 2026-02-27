"""
ふしぎなアメの挙動確認用スクリプト。
手札に「ふしぎなアメ」「ワルビアル（2進化）」「ワルビル（1進化）」、
バトル場に「メグロコ（たね）」がある状態でふしぎなアメを使い、
メグロコがワルビアルに 1 進化をとばして進化することを確認する。
"""
from game import GameState, PlayerState, BattlePokemon, use_trainer_goods
from card import get_card_by_id

def main() -> None:
    # 状態を組み立てる
    state = GameState(
        current_player=0,
        first_player=0,
        turn_count=1,  # 先行 1 ターン後なのでふしぎなアメ使用可
        log_fn=print,
    )
    p0 = state.players[0]
    p1 = state.players[1]

    # 手札: ふしぎなアメ, ワルビアル(2進化), ワルビル(1進化) … ワルビルで stage1 を名前解決
    fushiginaame = get_card_by_id("fushiginaame", "hand-fushiginaame")
    warubiaru = get_card_by_id("warubiaru-svd-064", "hand-warubiaru")
    warubiru = get_card_by_id("warubiru-svd-063", "hand-warubiru")
    p0.hand = [fushiginaame, warubiaru, warubiru]

    # バトル場: メグロコ（たね）、出したばかりではない
    meguroko = get_card_by_id("meguroko-svd-062", "active-meguroko")
    p0.active = BattlePokemon(card=meguroko, put_on_bench_this_turn=False)
    p0.bench = []
    p0.deck = []
    p0.discard = []
    p0.prize_pile = []

    # 相手は空でよい（アイテム処理だけ見る）
    p1.hand = []
    p1.active = None
    p1.bench = []
    p1.deck = []
    p1.discard = []
    p1.prize_pile = []

    # ふしぎなアメは手札 0 番
    hand_index_fushiginaame = 0
    assert p0.hand[hand_index_fushiginaame].id == "fushiginaame"

    print("--- 実行前 ---")
    print(f"バトル場: {p0.active.card.name} (id={p0.active.card.id})")
    print(f"手札: {[getattr(c, 'name', c.id) for c in p0.hand]}")

    ok = use_trainer_goods(state, hand_index_fushiginaame)

    print("\n--- 実行後 ---")
    print(f"use_trainer_goods 戻り値: {ok}")
    print(f"バトル場: {p0.active.card.name} (id={getattr(p0.active.card, 'id', '')})")
    print(f"手札枚数: {len(p0.hand)}")
    print(f"手札: {[getattr(c, 'name', getattr(c, 'id', '')) for c in p0.hand]}")
    print(f"捨て札: {[getattr(c, 'name', getattr(c, 'id', '')) for c in p0.discard]}")

    # 検証
    assert ok, "ふしぎなアメの使用に失敗"
    assert p0.active is not None, "バトル場が空"
    assert p0.active.card.name == "ワルビアル", f"進化後の名前がワルビアルでない: {p0.active.card.name}"
    assert len(p0.hand) == 1, f"手札はワルビル 1 枚のはず: {len(p0.hand)}"
    assert any(getattr(c, "id", "") == "fushiginaame" for c in p0.discard), "ふしぎなアメが捨て札に入っていること"
    assert not any(getattr(c, "name", "") == "ワルビアル" for c in p0.hand), "ワルビアルは場にのったので手札に残っていないこと"

    print("\nOK: ふしぎなアメでメグロコ → ワルビアルに 1 進化をとばして進化できた。")


if __name__ == "__main__":
    main()
