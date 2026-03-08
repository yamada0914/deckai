"""ターン実行用：エネルギー付与・進化の試行。"""
from card import is_energy, is_pokemon

from .damage import _max_effective_damage_if_attach
from .evolution import _can_evolve_onto, evolve_pokemon
from .state import (
    GameState,
    PlayerState,
    MAX_EVOLVE_ROUNDS_PER_TURN,
    _can_pay_energy_cost,
    _is_first_player_first_turn,
    _log_choice,
)
from .trainers import attach_energy
from .weights import get_energy_attach_weight, get_evolve_onto_weight


def _should_attach_for_evolution(p: PlayerState) -> bool:
    """バトル場のポケモンが手札の進化で必要エネルギーに足りていなければ True。"""
    if not p.active:
        return False
    types = getattr(p.active, "attached_energy_types", [])
    for c in p.hand:
        if not (is_pokemon(c) and _can_evolve_onto(p.active.card, c)):
            continue
        for a in c.attacks:
            if _can_pay_energy_cost(
                p.active.attached_energy, types,
                a.energy_cost, getattr(a, "energy_cost_typed", None),
            ):
                return False
        return True
    return False


def _energy_needed_for_active(p: PlayerState) -> int:
    """バトル場（＋手札の進化）に必要なエネルギーコスト合計を返す。"""
    if not p.active:
        return 0
    need = max((a.energy_cost for a in p.active.card.attacks), default=0)
    for c in p.hand:
        if not (is_pokemon(c) and _can_evolve_onto(p.active.card, c)):
            continue
        need = max(need, max((a.energy_cost for a in c.attacks), default=0))
        break
    return need


def _can_active_use_any_attack(p: PlayerState) -> bool:
    """バトル場のポケモンがどれか 1 つでも技を出せるか（タイプ指定込みで判定）。"""
    if not p.active or not p.active.card.attacks:
        return False
    types = getattr(p.active, "attached_energy_types", [])
    for atk in p.active.card.attacks:
        if _can_pay_energy_cost(
            p.active.attached_energy, types,
            atk.energy_cost, getattr(atk, "energy_cost_typed", None),
        ):
            return True
    return False


def _max_energy_for_pokemon(attacks: list) -> int:
    """そのポケモンの技で必要な最大エネルギー数（個数）を返す。"""
    return max((a.energy_cost for a in attacks), default=0)


def _try_evolve_once(state: GameState) -> bool:
    """進化を 1 回だけ試す（手札の進化でバトル場 or ベンチにのせる）。進化したら True。重みでのせる先を選ぶ。"""
    p = state.active_player_state()
    weights = state.get_weights_for_player(state.current_player)
    candidates = []
    for hand_idx, c in enumerate(p.hand):
        if not is_pokemon(c) or not getattr(c, "evolves_from", None):
            continue
        if p.active and _can_evolve_onto(p.active.card, c):
            candidates.append((hand_idx, None, p.active.card))
        for bench_idx, bench_poke in enumerate(p.bench):
            if _can_evolve_onto(bench_poke.card, c):
                candidates.append((hand_idx, bench_idx, bench_poke.card))
    if not candidates:
        return False
    best = max(
        candidates,
        key=lambda x: (get_evolve_onto_weight(weights, x[2]), x[1] is None),
    )
    hand_idx, bench_idx, target_card = best
    evolve_pokemon(state, hand_idx, bench_index=bench_idx)
    tid = getattr(target_card, "id", None) or getattr(target_card, "name", "")
    if tid:
        _log_choice(state, "evolve_onto", card_id=tid)
    return True


def _try_evolve_rounds_after_hand_change(state: GameState) -> bool:
    """手札が入れ替わった直後に進化可能なら進化する（サポート・グッズで手札が変わった後用）。1 回以上進化したら True。"""
    if state.turn_count < 2:
        return False
    acted = False
    for _ in range(MAX_EVOLVE_ROUNDS_PER_TURN):
        if _try_evolve_once(state):
            acted = True
            state._record_frame()
        else:
            break
    return acted


def _try_attach_energy_auto(state: GameState) -> bool:
    """
    自動ターン用：エネルギーを 1 枚付与するなら True を返す。
    付与先は重み (w_energy_attach) を最優先で選び、同点時は有効ダメージで決める。
    手札の進化に必要なときはバトル場に付与する特別扱いあり。
    技で必要な最大までしか付与しない。
    """
    p = state.active_player_state()
    can_evolve_this_turn = not _is_first_player_first_turn(state)
    energy_hand_idx = next((i for i, c in enumerate(p.hand) if is_energy(c)), None)
    if energy_hand_idx is None or not p.active:
        return False

    energy_card = p.hand[energy_hand_idx]
    new_type = getattr(energy_card, "energy_type", None) or "colorless"
    opp = state.defending_player_state()
    max_active = _max_energy_for_pokemon(p.active.card.attacks)

    if can_evolve_this_turn and _should_attach_for_evolution(p):
        attach_energy(state, energy_hand_idx)
        return True

    candidates = []
    if p.active.attached_energy < max_active:
        dmg = _max_effective_damage_if_attach(
            state,
            p.active.card,
            p.active.attached_energy,
            getattr(p.active, "attached_energy_types", []),
            new_type,
            opp.active,
            state.current_player,
        )
        candidates.append((None, dmg))
    for bi, b in enumerate(p.bench):
        max_b = _max_energy_for_pokemon(b.card.attacks)
        if b.attached_energy >= max_b:
            continue
        dmg = _max_effective_damage_if_attach(
            state,
            b.card,
            b.attached_energy,
            getattr(b, "attached_energy_types", []),
            new_type,
            opp.active,
            state.current_player,
        )
        candidates.append((bi, dmg))

    if not candidates:
        return False

    weights = state.get_weights_for_player(state.current_player)

    def _energy_attach_score(item: tuple) -> tuple:
        target, dmg = item
        card = p.active.card if target is None else p.bench[target].card
        w = get_energy_attach_weight(weights, card)
        return (w, dmg, target is None)

    best = max(candidates, key=_energy_attach_score)
    target_card = p.active.card if best[0] is None else p.bench[best[0]].card
    card_id = getattr(target_card, "id", None) or getattr(target_card, "name", "")
    if best[0] is None:
        attach_energy(state, energy_hand_idx)
        _log_choice(state, "energy_attach", card_id=card_id)
        return True
    attach_energy(state, energy_hand_idx, bench_index=best[0])
    _log_choice(state, "energy_attach", card_id=card_id)
    return True
