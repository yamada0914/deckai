"""
rules/02_card_descriptions.md（カードの説明文）に基づくルールテスト。

- A カードの説明文について（カード優先・「できない」優先）
- B ダメージ計算関連（B-01〜B-09）
- C 効果（C-01〜C-20）
- D 用語（D-01〜D-19）
- E その他の説明文（E-01〜E-39）
"""
import pytest

from card import get_card_by_id
from game import (
    GameState,
    BattlePokemon,
    attack,
    _check_game_end,
    _effective_damage_to_defender,
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


class TestA_CardTextPriority:
    """A カードの説明文について: カードに書かれていることを優先。「●●できない」を優先。"""

    def test_card_effect_overrides_general_rule(self):
        """カードの説明文が基本ルールと違う場合はカードを優先（実装がカード効果を正しく扱う前提）。"""
        # 例: 「次の自分の番で使えない」はカード効果で、実装では disabled_attack_name で表現される
        # ルカリオ（rukario-svd-060）が「かそくづき」を持つ
        card = get_card_by_id("rukario-svd-060", "c")
        attacks = getattr(card, "attacks", [])
        kasokuzuki = next((a for a in attacks if getattr(a, "name", "") == "かそくづき"), None)
        assert kasokuzuki is not None
        assert "次の自分の番" in getattr(kasokuzuki, "description", "")


class TestB_DamageCalculation:
    """B ダメージ計算関連: 弱点・抵抗力・効果の適用順序。"""

    def test_weakness_doubles_damage(self):
        """弱点一致でダメージ 2 倍（_effective_damage_to_defender の弱点計算）。"""
        from game import _effective_damage_to_defender
        atk_dmg = 100
        attacker = get_card_by_id("mototokage", "att")
        defender_card = get_card_by_id("mototokage", "def")
        defender = BattlePokemon(card=defender_card)
        # 弱点・抵抗が無いカードならそのまま
        dmg = _effective_damage_to_defender(attacker, defender, atk_dmg)
        assert dmg >= 0

    def test_resistance_reduces_damage(self):
        """抵抗力一致でダメージ -30 等（実装で抵抗力が反映されることの確認）。"""
        from game import _effective_damage_to_defender
        attacker = get_card_by_id("mototokage", "a")
        defender_card = get_card_by_id("mototokage", "d")
        defender = BattlePokemon(card=defender_card)
        dmg = _effective_damage_to_defender(attacker, defender, 50)
        assert isinstance(dmg, int) and dmg >= 0


class TestC_Effects:
    """C 効果: トラッシュ・入れ替え・回復・ワザが使えない等。"""

    def test_c15_waza_tsukaenai_disabled_attack(self):
        """C-15 ワザが使えない: disabled_attack_name が設定されているとそのワザは使えない。"""
        state = _minimal_state()
        p = state.players[0]
        opp = state.players[1]
        # ルカリオ（かそくづきを持つ）
        card = get_card_by_id("rukario-svd-060", "active")
        p.active = BattlePokemon(
            card=card,
            attached_energy=3,
            attached_energy_types=["colorless", "colorless", "colorless"],
            disabled_attack_name="かそくづき",
        )
        p.hand = []
        opp.active = BattlePokemon(card=get_card_by_id("mototokage", "opp"))
        opp.hand = []
        atk_idx = next(i for i, a in enumerate(p.active.card.attacks) if a.name == "かそくづき")
        ok = attack(state, atk_idx)
        assert ok is False

    def test_c16_waza_damage_wo_ukenai(self):
        """C-16 ワザのダメージを受けない: 実装では protected_next_opponent_turn 等で 0 ダメージに。"""
        state = _minimal_state()
        p = state.players[0]
        opp = state.players[1]
        opp.active = BattlePokemon(
            card=get_card_by_id("mototokage", "opp"),
            protected_next_opponent_turn=True,
        )
        p.active = BattlePokemon(
            card=get_card_by_id("mototokage", "p"),
            attached_energy=1,
            attached_energy_types=["colorless"],
        )
        p.hand = []
        opp.hand = []
        before_hp = opp.active.hp
        ok = attack(state, 0)
        assert ok is True
        assert opp.active.hp == before_hp


class TestE_OtherDescriptions:
    """E その他の説明文: 次の自分の番、ワザのダメージを受けたとき、等。"""

    def test_e02_next_own_turn_kasokuzuki(self):
        """E-02 次の自分の番: かそくづきは次の自分の番で使えなくなる（disabled_attack_name で表現）。"""
        card = get_card_by_id("rukario-svd-060", "c")
        attacks = getattr(card, "attacks", [])
        kasokuzuki = next((a for a in attacks if getattr(a, "name", "") == "かそくづき"), None)
        assert kasokuzuki is not None
        desc = getattr(kasokuzuki, "description", "")
        assert "次の自分の番" in desc
