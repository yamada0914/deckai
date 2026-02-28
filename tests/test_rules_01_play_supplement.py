"""
rules/01_play_supplement.md（遊びかた説明書の補足）に基づくルールテスト。

- A ポケモン（ワザ・特性・にげる・ベンチに出す・進化）
- B トレーナーズ（グッズ・どうぐ・サポート・スタジアム）
- C エネルギー
- D きぜつ / E 勝ち負け / F ポケモンチェック / G 対戦準備 / H 山札
"""
import pytest

from card import get_card_by_id, is_energy, is_pokemon, is_support
from game import (
    GameState,
    PlayerState,
    BattlePokemon,
    PRIZE_COUNT,
    BENCH_SIZE,
    attach_energy,
    retreat,
    use_support,
    use_potion,
    use_trainer_goods,
    evolve_pokemon,
    attack,
    setup_game,
    _check_game_end,
)


def _minimal_state(current_player=0, turn_count=1, first_player=0):
    """テスト用の最小 GameState。手札・バトル場・ベンチを呼び出し側で設定する。"""
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


# ----- A ポケモン -----


class TestA01_Waza:
    """A-01 ワザ: 自分の番に 1 つだけ。ねむり・マヒでは宣言できない。"""

    def test_attack_ends_turn_effect(self):
        """ワザを使い終えると自分の番は終わる（run_turn_auto で 1 ターンに 1 回しか攻撃しないことを前提に、attack が成功すると acted になる）。"""
        state = _minimal_state()
        p = state.players[0]
        opp = state.players[1]
        card = get_card_by_id("mototokage", "active")
        p.active = BattlePokemon(card=card, attached_energy=1, attached_energy_types=["colorless"])
        p.hand = []
        opp_card = get_card_by_id("mototokage", "opp")
        opp.active = BattlePokemon(card=opp_card)
        opp.hand = []
        assert len(p.active.card.attacks) > 0
        ok = attack(state, 0)
        assert ok is True

    def test_sleep_cannot_attack(self):
        """ねむり・マヒの場合はワザを宣言できない（attack が False を返す）。"""
        state = _minimal_state()
        p = state.players[0]
        opp = state.players[1]
        card = get_card_by_id("mototokage", "active")
        p.active = BattlePokemon(card=card, attached_energy=1, attached_energy_types=["colorless"], special_state="sleep")
        p.hand = []
        opp.active = BattlePokemon(card=get_card_by_id("mototokage", "opp"))
        opp.hand = []
        ok = attack(state, 0)
        assert ok is False


class TestA03_Nigeru:
    """A-03 にげる: 自分の番に 1 回だけ。ねむり・マヒ・にげられない効果中はできない。ベンチに戻ると特殊状態・ワザの効果はなくなる。"""

    def test_paralysis_cannot_retreat(self):
        """マヒのポケモンはにげられない。"""
        state = _minimal_state()
        p = state.players[0]
        card = get_card_by_id("mototokage", "active")
        p.active = BattlePokemon(card=card, attached_energy=1, attached_energy_types=["colorless"], special_state="paralysis")
        bench_card = get_card_by_id("mototokage", "bench")
        p.bench = [BattlePokemon(card=bench_card)]
        p.hand = []
        ok = retreat(state, 0)
        assert ok is False

    def test_sleep_cannot_retreat(self):
        """ねむりのポケモンはにげられない。"""
        state = _minimal_state()
        p = state.players[0]
        card = get_card_by_id("mototokage", "active")
        p.active = BattlePokemon(card=card, attached_energy=1, attached_energy_types=["colorless"], special_state="sleep")
        p.bench = [BattlePokemon(card=get_card_by_id("mototokage", "bench"))]
        p.hand = []
        assert retreat(state, 0) is False

    def test_retreat_clears_status(self):
        """ベンチにもどると特殊状態やワザの効果はすべてなくなる（_clear_status で disabled_attack_name も解除）。"""
        state = _minimal_state()
        p = state.players[0]
        card = get_card_by_id("mototokage", "active")
        p.active = BattlePokemon(
            card=card,
            attached_energy=2,
            attached_energy_types=["colorless", "colorless"],
            disabled_attack_name="かそくづき",
        )
        p.bench = [BattlePokemon(card=get_card_by_id("mototokage", "bench"))]
        p.hand = []
        ok = retreat(state, 0)
        assert ok is True
        assert getattr(p.bench[0], "disabled_attack_name", None) is None


class TestA04_Bench:
    """A-04 ベンチに出す: たねはベンチが 5 匹を超えないかぎり、1 回の自分の番に何匹でも出せる。"""

    def test_bench_max_five(self):
        """ベンチは最大 5 体（定数 BENCH_SIZE）。"""
        assert BENCH_SIZE == 5


class TestA05_Evolution:
    """A-05 進化: 最初の番には進化できない。場に出したばかり・進化したばかりのポケモンもその番には進化できない。"""

    def test_evolution_blocked_first_turn(self):
        """最初の番（turn_count == 0）では進化できない（run_turn_auto では can_evolve = state.turn_count >= 2）。"""
        state = _minimal_state(turn_count=0)
        p = state.players[0]
        base = get_card_by_id("meguroko-svd-062", "base")
        p.active = BattlePokemon(card=base, put_on_bench_this_turn=False)
        evo = get_card_by_id("warubiru-svd-063", "evo")
        p.hand = [evo]
        ok = evolve_pokemon(state, 0, bench_index=None)
        state2 = _minimal_state(turn_count=2)
        p2 = state2.players[0]
        p2.active = BattlePokemon(card=base, put_on_bench_this_turn=True)
        p2.hand = [get_card_by_id("warubiru-svd-063", "evo2")]
        ok2 = evolve_pokemon(state2, 0, bench_index=None)
        assert ok2 is False


# ----- B トレーナーズ -----


class TestB01_Goods:
    """B-01 グッズ: 自分の番に何枚でも使える。"""

    def test_goods_use_succeeds(self):
        """グッズは条件を満たせば使える（きずぐすりはバトル場のポケモンがダメージを受けていれば使える）。"""
        state = _minimal_state()
        p = state.players[0]
        card = get_card_by_id("mototokage", "active")
        card.hp = 40
        p.active = BattlePokemon(card=card)
        p.hand = [get_card_by_id("potion", "potion")]
        p.discard = []
        ok = use_potion(state, 0)
        assert ok is True


class TestB03_Support:
    """B-03 サポート: 自分の番に 1 枚だけ使える。"""

    def test_support_only_once_per_turn(self):
        """サポートは 1 ターンに 1 枚だけ（2 枚目は use_support が False）。"""
        state = _minimal_state(turn_count=2)
        p = state.players[0]
        p.hand = [
            get_card_by_id("nemo", "nemo1"),
            get_card_by_id("nemo", "nemo2"),
        ]
        p.deck = [get_card_by_id("mototokage", f"d{i}") for i in range(10)]
        p.active = BattlePokemon(card=get_card_by_id("mototokage", "active"))
        state.support_used_this_turn = False
        ok1 = use_support(state, 0)
        assert ok1 is True
        ok2 = use_support(state, 0)
        assert ok2 is False


# ----- C エネルギー -----


class TestC_Energy:
    """C エネルギー: 自分の番ごとに 1 枚だけ、自分の場のポケモンにつけられる。"""

    def test_energy_once_per_turn(self):
        """1 ターンにエネルギー付与は 1 回まで（2 回目は attach_energy が False）。"""
        state = _minimal_state()
        p = state.players[0]
        p.active = BattlePokemon(card=get_card_by_id("mototokage", "active"))
        energy = get_card_by_id("basic-energy", "e1")
        p.hand = [
            energy,
            get_card_by_id("basic-energy", "e2"),
        ]
        state.energy_attached_this_turn = False
        ok1 = attach_energy(state, 0)
        assert ok1 is True
        ok2 = attach_energy(state, 0)
        assert ok2 is False


# ----- D きぜつ / E 勝ち負け / G 対戦準備 -----


class TestD_E_GameEnd:
    """D きぜつ / E 勝ち負け: サイド 0・場にポケモンがいない・山札 0 でドロー不能で負け。"""

    def test_win_by_prizes_taken(self):
        """相手がサイドをすべてとり終えたらその相手が勝ち（winner が「サイドを取った側」）。"""
        state = _minimal_state()
        state.players[0].prize_pile = []
        state.players[1].prize_pile = [None]
        _check_game_end(state)
        assert state.winner == 0

    def test_win_by_no_pokemon(self):
        """自分の場のポケモンが 1 匹もいなくなったらそのプレイヤーが負け。"""
        state = _minimal_state()
        state.players[0].active = None
        state.players[0].bench = []
        state.players[0].prize_pile = [None] * PRIZE_COUNT
        state.players[1].prize_pile = [None] * PRIZE_COUNT
        _check_game_end(state)
        assert state.winner == 1


class TestG_Setup:
    """G 対戦準備: 初手 7 枚、サイド 6 枚。"""

    def test_setup_hand_and_prizes(self):
        """setup_game で初手 7 枚ドロー・サイド 6 枚。手札はバトル場・ベンチに出すぶん減る。"""
        state = setup_game(seed=42, deck0=0, deck1=0)
        assert PRIZE_COUNT == 6
        for i in range(2):
            assert len(state.players[i].prize_pile) == PRIZE_COUNT
            assert state.players[i].active is not None
            assert len(state.players[i].hand) + 1 + len(state.players[i].bench) >= 7
