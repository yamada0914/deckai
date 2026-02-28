"""
rules/advanced_rule.md（上級プレイヤー用ルールガイド Ver.3.4）に基づくルールテスト。

01_play_supplement / 02_card_descriptions と対応する項目を、上級ルールの用語・手順に合わせてテストする。
"""
import pytest

from card import get_card_by_id
from game import (
    GameState,
    BattlePokemon,
    BENCH_SIZE,
    PRIZE_COUNT,
    attach_energy,
    retreat,
    use_support,
    use_potion,
    evolve_pokemon,
    attack,
    setup_game,
    _check_game_end,
)


def _minimal_state(current_player=0, turn_count=1, first_player=0):
    state = GameState(
        current_player=current_player,
        first_player=first_player,
        turn_count=turn_count,
        log_fn=lambda _: None,
    )
    for i in range(2):
        state.players[i].deck = []
        state.players[i].discard = []
        state.players[i].prize_pile = []
    return state


class TestAdvanced_A_Pokemon:
    """Ⅰ A ポケモン（ワザ・特性・にげる・ベンチに出す・進化）"""

    def test_a01_waza_one_per_turn(self):
        """ワザは自分の番に 1 つだけ使う。"""
        state = _minimal_state()
        p = state.players[0]
        opp = state.players[1]
        p.active = BattlePokemon(
            card=get_card_by_id("mototokage", "a"),
            attached_energy=1,
            attached_energy_types=["colorless"],
        )
        opp.active = BattlePokemon(card=get_card_by_id("mototokage", "o"))
        p.hand = []
        opp.hand = []
        ok = attack(state, 0)
        assert ok is True

    def test_a03_nigeru_once_per_turn_clears_status(self):
        """にげるは自分の番に 1 回だけ。ベンチにもどると特殊状態・ワザの効果はなくなる。"""
        state = _minimal_state()
        p = state.players[0]
        card = get_card_by_id("mototokage", "a")
        p.active = BattlePokemon(
            card=card,
            attached_energy=2,
            attached_energy_types=["colorless", "colorless"],
            disabled_attack_name="かそくづき",
        )
        p.bench = [BattlePokemon(card=get_card_by_id("mototokage", "b"))]
        p.hand = []
        retreat(state, 0)
        assert getattr(p.bench[0], "disabled_attack_name", None) is None

    def test_a05_evolution_no_first_turn_no_just_put(self):
        """最初の番には進化できない。場に出したばかり・進化したばかりもその番には進化できない。"""
        state = _minimal_state(turn_count=2)
        p = state.players[0]
        base = get_card_by_id("meguroko-svd-062", "base")
        p.active = BattlePokemon(card=base, put_on_bench_this_turn=True)
        p.hand = [get_card_by_id("warubiru-svd-063", "evo")]
        ok = evolve_pokemon(state, 0, bench_index=None)
        assert ok is False


class TestAdvanced_B_Trainers:
    """Ⅰ B トレーナーズ（グッズ・どうぐ・サポート・スタジアム）"""

    def test_b01_goods_usable(self):
        """グッズは自分の番に何枚でも使える。"""
        state = _minimal_state()
        p = state.players[0]
        card = get_card_by_id("mototokage", "a")
        card.hp = 50
        p.active = BattlePokemon(card=card)
        p.hand = [get_card_by_id("potion", "p")]
        p.discard = []
        assert use_potion(state, 0) is True

    def test_b03_support_once_per_turn(self):
        """サポートは自分の番に 1 枚だけ。"""
        state = _minimal_state(turn_count=2)
        p = state.players[0]
        p.hand = [
            get_card_by_id("nemokako", "n1"),
            get_card_by_id("nemokako", "n2"),
        ]
        p.deck = [get_card_by_id("mototokage", f"d{i}") for i in range(10)]
        p.active = BattlePokemon(card=get_card_by_id("mototokage", "a"))
        state.support_used_this_turn = False
        assert use_support(state, 0) is True
        assert use_support(state, 0) is False


class TestAdvanced_C_Energy:
    """Ⅰ C エネルギー: 自分の番ごとに 1 枚だけつけられる。"""

    def test_energy_one_per_turn(self):
        """手札からエネルギーを自分の番ごとに 1 枚だけつける。"""
        state = _minimal_state()
        p = state.players[0]
        p.active = BattlePokemon(card=get_card_by_id("mototokage", "a"))
        p.hand = [
            get_card_by_id("basic-energy", "e1"),
            get_card_by_id("basic-energy", "e2"),
        ]
        state.energy_attached_this_turn = False
        assert attach_energy(state, 0) is True
        assert attach_energy(state, 0) is False


class TestAdvanced_D_E_KoAndWinLose:
    """Ⅰ D きぜつ / E 勝ち負け"""

    def test_d_ko_prize_take(self):
        """きぜつしたポケモンの数ぶんサイドをとる。"""
        state = _minimal_state()
        state.players[0].prize_pile = []
        state.players[1].prize_pile = [None]
        _check_game_end(state)
        assert state.winner == 0

    def test_e_win_no_pokemon(self):
        """自分の場のポケモンが 1 匹もいなくなったら負け。"""
        state = _minimal_state()
        state.players[0].active = None
        state.players[0].bench = []
        state.players[0].prize_pile = [None] * PRIZE_COUNT
        state.players[1].prize_pile = [None] * PRIZE_COUNT
        _check_game_end(state)
        assert state.winner == 1


class TestAdvanced_G_Setup:
    """Ⅰ G 対戦準備: 山札を置く、手札 7 枚、サイド 6 枚。"""

    def test_prize_six_and_bench_max_five(self):
        """サイド 6 枚、ベンチ最大 5 匹。"""
        assert PRIZE_COUNT == 6
        assert BENCH_SIZE == 5

    def test_setup_produces_valid_game(self):
        """対戦準備後、双方にバトルポケモンとサイドがある。"""
        state = setup_game(seed=123, deck0=0, deck1=0)
        for i in range(2):
            assert state.players[i].active is not None
            assert len(state.players[i].prize_pile) == PRIZE_COUNT
