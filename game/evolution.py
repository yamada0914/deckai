"""進化（進化適用・進化可否判定・evolve_pokemon）。"""
from card import PokemonCard, is_pokemon

from .state import (
    GameState,
    PlayerState,
    BattlePokemon,
    _clear_status,
)


def _apply_evolution(
    target: BattlePokemon,
    evolution_card: PokemonCard,
    state: GameState,
    log_prefix: str,
) -> None:
    """進化を 1 体に適用（HP・エネルギーを引き継ぎ、カード差し替え）。"""
    old_name = target.card.name
    old_hp = target.hp
    old_max_hp = target.card.max_hp
    old_energy = target.attached_energy
    old_energy_types = getattr(target, "attached_energy_types", [])
    damage_taken = old_max_hp - old_hp
    evolved = evolution_card.copy()
    target.card = evolved
    target.card.max_hp = evolved.max_hp
    target.card.hp = max(0, min(evolved.max_hp, evolved.max_hp - damage_taken))
    target.attached_energy = old_energy
    target.attached_energy_types = list(old_energy_types)
    target.evolved_this_turn = True
    _clear_status(target)
    state.log(f"{log_prefix}{old_name} を {evolved.name} に進化（HP {target.hp}/{target.max_hp}）")


def _can_evolve_onto(field_card, evolution_card) -> bool:
    """場のポケモン（field_card）が進化カード（evolution_card）の進化元か。id または name で一致させる。"""
    base = (evolution_card.evolves_from or "").strip()
    if not base:
        return False
    fid = (getattr(field_card, "id", None) or "").strip()
    fname = (getattr(field_card, "name", "") or getattr(field_card, "name_ja", "") or "").strip()
    if fid == base or fname == base:
        return True
    if base and (fid.startswith(base + "-") or fid.startswith(base + "_")):
        return True
    return False


def evolve_pokemon(state: GameState, hand_index: int, bench_index: int | None = None) -> bool:
    """
    手札の進化ポケモンで、バトル場またはベンチのポケモンを進化させる。
    bench_index=None ならバトル場、数値ならベンチのそのインデックス。
    evolves_from は進化元の id または name_ja（日本語名）でよい。set 付き id のポケモンも名前で一致すれば進化できる。
    """
    p = state.active_player_state()
    if hand_index < 0 or hand_index >= len(p.hand):
        return False
    evolution_card = p.hand[hand_index]
    if not is_pokemon(evolution_card) or not evolution_card.evolves_from:
        return False

    if bench_index is None:
        if not p.active or not _can_evolve_onto(p.active.card, evolution_card):
            return False
        if getattr(p.active.card, "evolves_from", None) is not None and getattr(p.active, "evolved_this_turn", False):
            return False
        if getattr(p.active, "put_on_bench_this_turn", False):
            return False
        _apply_evolution(
            p.active, evolution_card, state,
            f"{state.player_name(state.current_player)}: ",
        )
        p.hand.pop(hand_index)
        return True
    if bench_index < 0 or bench_index >= len(p.bench):
        return False
    bench_pokemon = p.bench[bench_index]
    if not _can_evolve_onto(bench_pokemon.card, evolution_card):
        return False
    if getattr(bench_pokemon.card, "evolves_from", None) is not None and getattr(bench_pokemon, "evolved_this_turn", False):
        return False
    if getattr(bench_pokemon, "put_on_bench_this_turn", False):
        return False
    _apply_evolution(
        bench_pokemon, evolution_card, state,
        f"{state.player_name(state.current_player)}: ベンチの ",
    )
    p.hand.pop(hand_index)
    return True
