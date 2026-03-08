"""ダメージ計算（弱点・抵抗・どうぐ・技評価）。"""
from card import PokemonCard

from .state import (
    GameState,
    PlayerState,
    BattlePokemon,
    _can_pay_energy_cost,
)


def _attack_damage_for_eval(atk) -> int:
    """技のダメージを評価用に返す。コイン技の場合は期待値（表0.5×回数×damage_per_coin）。"""
    cf = getattr(atk, "coin_flips", 0)
    dpc = getattr(atk, "damage_per_coin", 0)
    if cf > 0 and dpc > 0:
        return int(cf * 0.5 * dpc)
    return atk.damage


def _effective_damage_to_defender(
    attacker_card: PokemonCard,
    defender: BattlePokemon,
    base_damage: int,
    state: GameState | None = None,
    attacker_bp: BattlePokemon | None = None,
) -> int:
    """弱点・抵抗力・どうぐ・パワープロテイン・マキシマムベルトを考慮した、守備側が受けるダメージを返す。"""
    defender_card = defender.card
    damage = base_damage
    if (
        getattr(defender_card, "weakness", None)
        and getattr(attacker_card, "pokemon_type", None)
        and defender_card.weakness == attacker_card.pokemon_type
    ):
        damage *= 2
    if (
        getattr(defender_card, "resistance", None)
        and getattr(attacker_card, "pokemon_type", None)
        and defender_card.resistance == attacker_card.pokemon_type
    ):
        damage = max(0, damage - 30)
    tool = getattr(defender, "attached_tool", None)
    if tool and getattr(tool, "is_tool", False) and getattr(tool, "tool_damage_reduce", 0) > 0:
        cond = getattr(tool, "tool_condition_type", None)
        if cond is None or getattr(defender_card, "pokemon_type", None) == cond:
            damage = max(0, damage - getattr(tool, "tool_damage_reduce", 0))
    n = getattr(state, "fighting_damage_plus_30_count_this_turn", 0) if state else 0
    if n > 0 and getattr(attacker_card, "pokemon_type", None) == "fighting":
        damage += 30 * n
    if attacker_bp:
        atk_tool = getattr(attacker_bp, "attached_tool", None)
        if atk_tool and (getattr(atk_tool, "id", "") or "") == "makishimamuberuto":
            if getattr(defender_card, "is_ex", False) or ("ex" in (getattr(defender_card, "name", "") or "")):
                damage += 50
    return damage


def _opponent_max_damage(state: GameState) -> int:
    """相手のバトル場ポケモンが今出せる最大ダメージを返す（エネルギーに応じた最強ワザ）。"""
    opp = state.defending_player_state()
    if not opp.active or not opp.active.card.attacks:
        return 0
    max_dmg = 0
    types = getattr(opp.active, "attached_energy_types", [])
    for atk in opp.active.card.attacks:
        if _can_pay_energy_cost(
            opp.active.attached_energy, types,
            atk.energy_cost, getattr(atk, "energy_cost_typed", None),
        ):
            dmg = _attack_damage_for_eval(atk)
            if atk.name == "しっぺがえし" and len(state.defending_player_state().prize_pile) == 1:
                dmg += 90
            max_dmg = max(max_dmg, dmg)
    return max_dmg


def _opponent_max_effective_damage(state: GameState) -> int:
    """相手のバトル場ポケモンが自分のバトル場に与えうる最大の有效ダメージ（弱点・抵抗・どうぐ込み）。"""
    p = state.active_player_state()
    opp = state.defending_player_state()
    if not opp.active or not opp.active.card.attacks or not p.active:
        return 0
    max_dmg = 0
    types = getattr(opp.active, "attached_energy_types", [])
    for atk in opp.active.card.attacks:
        if _can_pay_energy_cost(
            opp.active.attached_energy, types,
            atk.energy_cost, getattr(atk, "energy_cost_typed", None),
        ):
            base = _attack_damage_for_eval(atk)
            if atk.name == "しっぺがえし" and len(opp.prize_pile) == 1:
                base += 90
            eff = _effective_damage_to_defender(opp.active.card, p.active, base, state=state, attacker_bp=opp.active)
            max_dmg = max(max_dmg, eff)
    return max_dmg


def _our_max_damage(state: GameState) -> int:
    """自分のバトル場ポケモンがこのターン出せる最大ダメージを返す。"""
    p = state.active_player_state()
    opp = state.defending_player_state()
    if not p.active or not p.active.card.attacks:
        return 0
    max_dmg = 0
    types = getattr(p.active, "attached_energy_types", [])
    for atk in p.active.card.attacks:
        if _can_pay_energy_cost(
            p.active.attached_energy, types,
            atk.energy_cost, getattr(atk, "energy_cost_typed", None),
        ):
            dmg = _attack_damage_for_eval(atk)
            if atk.name == "しっぺがえし" and len(opp.prize_pile) == 1:
                dmg += 90
            if atk.name == "アベンジナックル" and state.our_ko_by_damage_last_turn[state.current_player]:
                dmg += 120
            max_dmg = max(max_dmg, dmg)
    return max_dmg


def _our_max_effective_damage(state: GameState) -> int:
    """自分のバトル場ポケモンが相手のバトル場に与えうる最大の有效ダメージ（弱点・抵抗・どうぐ込み）。"""
    p = state.active_player_state()
    opp = state.defending_player_state()
    if not p.active or not p.active.card.attacks or not opp.active:
        return 0
    max_dmg = 0
    types = getattr(p.active, "attached_energy_types", [])
    for atk in p.active.card.attacks:
        if _can_pay_energy_cost(
            p.active.attached_energy, types,
            atk.energy_cost, getattr(atk, "energy_cost_typed", None),
        ):
            base = _attack_damage_for_eval(atk)
            if atk.name == "しっぺがえし" and len(opp.prize_pile) == 1:
                base += 90
            if atk.name == "アベンジナックル" and state.our_ko_by_damage_last_turn[state.current_player]:
                base += 120
            eff = _effective_damage_to_defender(p.active.card, opp.active, base, state=state, attacker_bp=p.active)
            max_dmg = max(max_dmg, eff)
    return max_dmg


def _max_effective_damage_for_attacker(
    state: GameState,
    attacker_bp: BattlePokemon,
    defender_bp: BattlePokemon | None,
    player_index: int,
) -> int:
    """任意のバトルポケモンが相手のバトル場に与えうる最大の有效ダメージを返す。"""
    if not attacker_bp.card.attacks or not defender_bp:
        return 0
    max_dmg = 0
    types = getattr(attacker_bp, "attached_energy_types", [])
    opp = state.players[1 - player_index]
    for atk in attacker_bp.card.attacks:
        if not _can_pay_energy_cost(
            attacker_bp.attached_energy, types,
            atk.energy_cost, getattr(atk, "energy_cost_typed", None),
        ):
            continue
        base = _attack_damage_for_eval(atk)
        if atk.name == "しっぺがえし" and len(opp.prize_pile) == 1:
            base += 90
        if atk.name == "アベンジナックル" and state.our_ko_by_damage_last_turn[player_index]:
            base += 120
        eff = _effective_damage_to_defender(attacker_bp.card, defender_bp, base, state=state, attacker_bp=attacker_bp)
        max_dmg = max(max_dmg, eff)
    return max_dmg


def _max_effective_damage_if_attach(
    state: GameState,
    attacker_card,
    attached_count: int,
    attached_types: list,
    extra_type: str,
    defender: BattlePokemon | None,
    current_player: int,
) -> int:
    """
    このポケモンに extra_type を 1 つ付けたとき、
    相手のバトル場（defender）に与えられる最大の有效ダメージを返す。
    """
    if not attacker_card.attacks:
        return 0
    sim_count = attached_count + 1
    sim_types = list(attached_types) + [extra_type]
    max_eff = 0
    opp = state.defending_player_state()
    for atk in attacker_card.attacks:
        if not _can_pay_energy_cost(
            sim_count, sim_types,
            atk.energy_cost, getattr(atk, "energy_cost_typed", None),
        ):
            continue
        base = _attack_damage_for_eval(atk)
        if atk.name == "しっぺがえし" and len(opp.prize_pile) == 1:
            base += 90
        if atk.name == "アベンジナックル" and state.our_ko_by_damage_last_turn[current_player]:
            base += 120
        eff = _effective_damage_to_defender(attacker_card, defender, base) if defender else base
        max_eff = max(max_eff, eff)
    return max_eff
