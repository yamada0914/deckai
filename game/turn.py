"""ターン実行（にげる・エネルギー付与・ポケモンチェック・run_turn_auto・run_game_auto）。"""
import random

from card import is_energy, is_goods, is_pokemon, is_stadium, is_support

from .attack import (
    _choose_best_attack_index,
    attack,
    attack_has_merit_effect_at_zero_damage,
    get_legal_attack_indices,
)
from .evaluate import evaluate_board
from .damage import (
    _attack_damage_for_eval,
    _effective_damage_to_defender,
    _max_effective_damage_for_attacker,
    _opponent_max_effective_damage,
    _our_max_effective_damage,
)
from .state import (
    GameState,
    PlayerState,
    BENCH_SIZE,
    MAX_BALL_USES_PER_TURN,
    MAX_EVOLVE_ROUNDS_PER_TURN,
    MAX_TURNS_SAFETY,
    MAX_TURN_ACTION_ROUNDS,
    PRIZE_COUNT,
    start_turn,
    _put_one_pokemon_on_bench,
    _is_first_player_first_turn,
    _check_game_end,
    _flip_coin,
    _put_energy_cards_in_discard,
    _clear_status,
    _handle_own_active_ko,
    _log_choice,
)
from .trainers import (
    play_stadium,
    use_pokemon_swap,
    use_potion,
    use_support,
    use_trainer_goods,
    _try_use_ability_runasaikuru,
)
from .turn_bench import (
    _best_opponent_bench_index_for_catcher,
    _best_sacrifice_bench_index,
    _has_bench_that_can_ko_after_one_attach,
)
from .turn_energy import (
    _try_attach_energy_auto,
    _try_evolve_once,
    _try_evolve_rounds_after_hand_change,
)
from .turn_trainers import (
    _HAND_REFRESH_SUPPORT_IDS,
    _support_try_order,
    _try_attach_one_tool,
    _try_erekijienereta,
    _try_goods_before_hand_refresh,
    _try_support_no_discard_only,
)
from .weights import get_goods_use_weight, get_retreat_target_weight, get_swap_target_weight

_BALL_GOODS_IDS = ("supaboru", "haipaboru", "otodokedoron", "pokemonkixyatchixya")


def retreat(state: GameState, bench_index: int) -> bool:
    """バトル場のポケモンとベンチの bench_index 番目を入れ替える（逃げる）。にげるエネルギー分を捨てる。ねむり・マヒ中はにげられない。"""
    p = state.active_player_state()
    if not p.active or bench_index < 0 or bench_index >= len(p.bench):
        return False
    if getattr(p.active, "special_state", None) in ("sleep", "paralysis"):
        state.log(f"{state.player_name(state.current_player)}: {p.active.card.name} は状態異常のためにげられない")
        return False
    old_active = p.active
    cost = getattr(old_active.card, "retreat_cost", 1)
    tool = getattr(old_active, "attached_tool", None)
    if tool and (getattr(tool, "id", "") or "") == "fuusen":
        cost = max(0, cost - 2)
    if old_active.attached_energy < cost:
        return False
    types = getattr(old_active, "attached_energy_types", [])
    if cost > 0:
        discarded_types = types[-cost:] if len(types) >= cost else list(types)
        _put_energy_cards_in_discard(p, discarded_types, state)
        old_active.attached_energy -= cost
        if len(types) >= cost:
            old_active.attached_energy_types = types[:-cost]
        else:
            old_active.attached_energy_types = []
    if cost > 0:
        state.log(
            f"{state.player_name(state.current_player)}: 逃げるために {old_active.card.name} のエネルギーを {cost} 個捨てる"
        )
    state.retreat_used_this_turn = True
    p.active = p.bench[bench_index]
    p.bench[bench_index] = old_active
    _clear_status(old_active)
    state.log(
        f"{state.player_name(state.current_player)}: {old_active.card.name} をベンチに戻し、"
        f"{p.active.card.name} をバトル場に出す（HP {p.active.hp}/{p.active.max_hp}）"
    )
    return True


def _apply_poison_burn_paralysis_for_active(state: GameState, player_index: int) -> None:
    """
    ポケモンチェック（ルール F）：そのプレイヤーのバトル場ポケモンに
    どく・やけどダメージ、やけど回復コイン、マヒ解除を適用する。
    きぜつ時はベンチから繰り出し、勝敗判定は呼び出し元で行う。
    """
    p = state.players[player_index]
    if not p.active:
        return
    poison_dmg = getattr(p.active, "poison_damage", 0)
    if poison_dmg > 0:
        before = p.active.hp
        p.active.hp -= poison_dmg
        state.log(f"{state.player_name(player_index)}: どくで {p.active.card.name} に {poison_dmg} ダメージ（HP {before} → {max(0, p.active.hp)}）")
    if p.active and p.active.hp <= 0:
        koed_poison = p.active
        if _handle_own_active_ko(state, player_index, koed_poison, "どくのダメージ"):
            return
    if not p.active:
        return
    if getattr(p.active, "burn", False):
        before = p.active.hp
        p.active.hp -= 20
        state.log(f"{state.player_name(player_index)}: やけどで {p.active.card.name} に 20 ダメージ（HP {before} → {max(0, p.active.hp)}）")
        if _flip_coin():
            p.active.burn = False
            state.log(f"{state.player_name(player_index)}: {p.active.card.name} のやけどが治った（コイン：表）")
        else:
            state.log(f"{state.player_name(player_index)}: {p.active.card.name} のやけどは継続（コイン：裏）")
    if player_index == state.current_player and getattr(p.active, "special_state", None) == "paralysis":
        p.active.special_state = None
        state.log(f"{state.player_name(player_index)}: {p.active.card.name} のマヒが解けた")
    if p.active and p.active.hp <= 0:
        _handle_own_active_ko(state, player_index, p.active, "状態異常のダメージ")


_MINIMAX_WIN_SCORE = 1e9
_MINIMAX_LOSE_SCORE = -1e9


def _minimax_score_after_attack(clone: GameState, me: int) -> float:
    """攻撃適用済み clone に対して、相手ターン・自分ターンを進めた後のスコアを返す。"""
    if clone.winner is not None:
        return _MINIMAX_WIN_SCORE if clone.winner == me else _MINIMAX_LOSE_SCORE
    end_turn(clone)
    if clone.winner is not None:
        return _MINIMAX_WIN_SCORE if clone.winner == me else _MINIMAX_LOSE_SCORE
    start_turn(clone)
    if clone.winner is not None:
        return _MINIMAX_WIN_SCORE if clone.winner == me else _MINIMAX_LOSE_SCORE
    run_turn_auto(clone)
    _check_game_end(clone)
    if clone.winner is not None:
        return _MINIMAX_WIN_SCORE if clone.winner == me else _MINIMAX_LOSE_SCORE
    return evaluate_board(clone, me)


def _choose_best_attack_index_minimax(state: GameState, p: PlayerState, opp: PlayerState) -> int | None:
    """
    2 手読み（自分が攻撃 → 相手のターン → 盤面評価）で最良の技インデックスを返す。
    候補が 0 件なら None。シミュレーションでは log / record_frame は出さない。
    """
    legal = get_legal_attack_indices(state, p, opp)
    if not legal:
        return None
    me = state.current_player
    best_idx = None
    best_score = _MINIMAX_LOSE_SCORE - 1
    for idx in legal:
        clone = state.copy_for_simulation()
        attack(clone, idx)
        score = _minimax_score_after_attack(clone, me)
        if score > best_score:
            best_score = score
            best_idx = idx
    return best_idx


def end_turn(state: GameState) -> None:
    """
    ターン終了：ポケモンチェック（ルール F）。
    おたがいのポケモンのどく・やけどを確認し、自分の番のマヒ解除。
    その後相手に交代、ターン数増加。
    """
    _apply_poison_burn_paralysis_for_active(state, state.current_player)
    if not _check_game_end(state):
        _apply_poison_burn_paralysis_for_active(state, state.opponent())
        _check_game_end(state)
    for pl in state.players:
        if pl.active:
            pl.active.evolved_this_turn = False
            pl.active.put_on_bench_this_turn = False
        for bp in pl.bench:
            bp.evolved_this_turn = False
            bp.put_on_bench_this_turn = False
    cp = state.current_player
    state.our_ko_by_damage_last_turn[cp] = False
    state.last_turn_attack_name[cp] = state.this_turn_attack_name
    state.last_turn_attack_actor_id[cp] = state.this_turn_attack_actor_id
    state.this_turn_attack_name = None
    state.this_turn_attack_actor_id = None
    leaving = state.players[cp]
    skip_clear_disabled = (
        state.turn_when_disabled_attack[cp] is not None
        and state.turn_count == state.turn_when_disabled_attack[cp]
    )
    if not skip_clear_disabled:
        for bp in ([leaving.active] if leaving.active else []) + list(leaving.bench):
            if bp:
                bp.disabled_attack_name = None
        if state.turn_when_disabled_attack[cp] is not None:
            state.turn_when_disabled_attack[cp] = None
    state.current_player = state.opponent()
    state.turn_count += 1
    state.log("")


def _try_swap_for_survival(state: GameState) -> bool:
    """きぜつ確定で相手を倒せないとき、いれかえグッズで生存できるベンチに替える。替えたら True。"""
    p = state.active_player_state()
    opp = state.defending_player_state()
    opp_max_effective = _opponent_max_effective_damage(state)
    our_max_effective = _our_max_effective_damage(state)
    would_be_koed = p.active and opp_max_effective > 0 and p.active.hp <= opp_max_effective
    can_ko_opponent = opp.active and our_max_effective >= opp.active.hp
    if not (would_be_koed and not can_ko_opponent and p.active and p.bench):
        return False
    _SWAP_SURVIVE_SCALE = 10000
    for i, c in enumerate(p.hand):
        if not is_goods(c) or (getattr(c, "effect", None) != "swap_active" and getattr(c, "id", "") not in ("pokemon_irekae", "pokemonirekae")):
            continue
        weights = state.get_weights_for_player(state.current_player)

        def _swap_score(bi: int) -> float:
            bp = p.bench[bi]
            survives = bp.hp > opp_max_effective
            return (1 if survives else 0) * _SWAP_SURVIVE_SCALE + bp.hp + get_swap_target_weight(weights, bp.card)

        best_bench = max(range(len(p.bench)), key=_swap_score)
        sc = getattr(p.bench[best_bench].card, "id", None) or getattr(p.bench[best_bench].card, "name", "")
        _log_choice(state, "swap", card_id=sc)
        if use_pokemon_swap(state, i, best_bench):
            return True
        break
    return False


def _try_retreat_when_koed(state: GameState) -> bool:
    """きぜつ確定で相手を倒せないとき、にげるエネルギーがあればベンチに逃げる。逃げたら True。"""
    if getattr(state, "retreat_used_this_turn", False):
        return False
    p = state.active_player_state()
    opp = state.defending_player_state()
    opp_max_effective = _opponent_max_effective_damage(state)
    our_max_effective = _our_max_effective_damage(state)
    would_be_koed = p.active and opp_max_effective > 0 and p.active.hp <= opp_max_effective
    can_ko_opponent = opp.active and our_max_effective >= opp.active.hp
    _raw_retreat = getattr(p.active.card, "retreat_cost", 1) if p.active else 0
    retreat_cost = max(0, _raw_retreat - (2 if p.active and (getattr(p.active, "attached_tool", None) and (getattr(p.active.attached_tool, "id", "") or "") == "fuusen") else 0))
    can_retreat = p.active and getattr(p.active, "special_state", None) not in ("sleep", "paralysis")
    has_bench_survivor = p.bench and any(bp.hp > opp_max_effective for bp in p.bench)
    can_ko_from_bench = opp.active and any(
        _max_effective_damage_for_attacker(state, bp, opp.active, state.current_player) >= opp.active.hp for bp in p.bench
    )
    retreat_helps = has_bench_survivor or can_ko_from_bench
    if not (can_retreat and p.active and p.bench and would_be_koed and not can_ko_opponent and retreat_helps and p.active.attached_energy >= retreat_cost):
        return False
    _RETREAT_C_KO, _RETREAT_C_DMG = 1_000_000, 1000
    weights = state.get_weights_for_player(state.current_player)
    best_idx = None
    best_score = -1.0
    for i, bp in enumerate(p.bench):
        dmg = _max_effective_damage_for_attacker(state, bp, opp.active, state.current_player) if opp.active else 0
        can_ko = int(opp.active is not None and dmg >= opp.active.hp)
        raw = can_ko * _RETREAT_C_KO + dmg * _RETREAT_C_DMG + bp.hp
        score = raw + get_retreat_target_weight(weights, bp.card)
        if score > best_score:
            best_score = score
            best_idx = i
    if best_idx is None:
        survivors = [(i, p.bench[i].hp, p.bench[i].attached_energy) for i in range(len(p.bench)) if p.bench[i].hp > opp_max_effective]
        if survivors:
            best_idx = max(survivors, key=lambda x: (x[1], x[2], get_retreat_target_weight(weights, p.bench[x[0]].card)))[0]
        else:
            best_idx = max(
                range(len(p.bench)),
                key=lambda i: (p.bench[i].hp, p.bench[i].attached_energy, get_retreat_target_weight(weights, p.bench[i].card)),
                default=None,
            )
    if best_idx is None:
        return False
    rc = getattr(p.bench[best_idx].card, "id", None) or getattr(p.bench[best_idx].card, "name", "")
    _log_choice(state, "retreat", card_id=rc)
    return retreat(state, best_idx)


def _try_retreat_voluntary(state: GameState) -> bool:
    """にげるコストが 0 のとき（風船など）に、自発的にベンチと入れ替える。1 回だけ試し、替えたら True。"""
    if getattr(state, "retreat_used_this_turn", False):
        return False
    p = state.active_player_state()
    if not p.active or not p.bench:
        return False
    if getattr(p.active, "special_state", None) in ("sleep", "paralysis"):
        return False
    _raw_rc = getattr(p.active.card, "retreat_cost", 1)
    tool = getattr(p.active, "attached_tool", None)
    retreat_cost = max(0, _raw_rc - (2 if tool and (getattr(tool, "id", "") or "") == "fuusen" else 0))
    if retreat_cost > 0 or p.active.attached_energy < retreat_cost:
        return False
    weights = state.get_weights_for_player(state.current_player)
    best_idx = max(
        range(len(p.bench)),
        key=lambda i: (get_retreat_target_weight(weights, p.bench[i].card), p.bench[i].hp, p.bench[i].attached_energy),
    )
    rc = getattr(p.bench[best_idx].card, "id", None) or getattr(p.bench[best_idx].card, "name", "")
    _log_choice(state, "retreat", card_id=rc)
    return retreat(state, best_idx)


def _do_attack_phase(state: GameState, p: PlayerState, opp: PlayerState) -> bool:
    """攻撃可能なら技を選んで攻撃（または生贄にげる）。実行したら True。"""
    is_game_first_turn = state.turn_count == 0
    if is_game_first_turn or not p.active or getattr(p.active, "special_state", None) in ("sleep", "paralysis"):
        return False
    best_idx = _choose_best_attack_index_minimax(state, p, opp) if getattr(state, "use_attack_minimax", True) else None
    if best_idx is None:
        best_idx = _choose_best_attack_index(state, p, opp)
    if best_idx is None:
        return False
    atk = p.active.card.attacks[best_idx]
    ac_id = getattr(p.active.card, "id", None) or getattr(p.active.card, "name", "")
    _log_choice(state, "attack", card_id=ac_id, attack_name=atk.name)
    base_dmg = _attack_damage_for_eval(atk)
    effective_dmg = _effective_damage_to_defender(p.active.card, opp.active, base_dmg, state=state, attacker_bp=p.active) if opp.active else base_dmg
    _raw_rc = getattr(p.active.card, "retreat_cost", 1)
    retreat_cost = max(0, _raw_rc - (2 if getattr(p.active, "attached_tool", None) and (getattr(p.active.attached_tool, "id", "") or "") == "fuusen" else 0))
    can_retreat_here = (
        bool(p.bench)
        and getattr(p.active, "special_state", None) not in ("sleep", "paralysis")
        and p.active.attached_energy >= retreat_cost
    )
    sacrifice_idx = _best_sacrifice_bench_index(state)
    should_retreat_sacrifice = (
        bool(opp.active)
        and effective_dmg <= 0
        and attack_has_merit_effect_at_zero_damage(p.active.card, atk)
        and can_retreat_here
        and sacrifice_idx is not None
        and _has_bench_that_can_ko_after_one_attach(state)
        and not getattr(state, "retreat_used_this_turn", False)
    )
    if should_retreat_sacrifice and retreat(state, sacrifice_idx):
        return True
    attack(state, best_idx)
    return True


def run_turn_auto(state: GameState) -> bool:
    """
    現在のプレイヤーが「可能な行動を順番に実行」する。
    順序：0. ベンチにポケモン出す、1. 進化、2. スタジアム（1 枚まで）、3. 手札を捨てないサポート（キハダ・ネモ）、4. エネルギー付与、5. ボール系グッズ、6. サポート、7. いれかえ、8. にげる、9. どうぐ・グッズ、10. 進化（再）、11. 攻撃。
    サポートまたはグッズを使用した場合は優先順位 1（進化）から手札を再確認する（MAX_TURN_ACTION_ROUNDS 回まで）。
    何もできなければ False を返す。True = ターン内で何かした。
    """
    p = state.active_player_state()
    opp = state.defending_player_state()
    if not p.active:
        return False

    acted = False

    def _try_put_bench_until_full():
        nonlocal p, acted
        put_count = 0
        while put_count < BENCH_SIZE and _put_one_pokemon_on_bench(p, state, state.current_player):
            put_count += 1
            acted = True
            p = state.active_player_state()
            state._record_frame()

    _try_put_bench_until_full()

    action_round = 0
    while action_round < MAX_TURN_ACTION_ROUNDS:
        action_round += 1

        used_fushiginaame = False
        if state.turn_count >= 2:
            for i, c in enumerate(p.hand):
                if not is_goods(c):
                    continue
                if getattr(c, "id", "") != "fushiginaame" and getattr(c, "name", "") != "ふしぎなアメ":
                    continue
                if use_trainer_goods(state, i):
                    acted = True
                    used_fushiginaame = True
                    p = state.active_player_state()
                    state._record_frame()
                    _try_put_bench_until_full()
                    break
            if used_fushiginaame:
                continue

        can_evolve = state.turn_count >= 2
        evolve_rounds = 0
        while can_evolve and evolve_rounds < MAX_EVOLVE_ROUNDS_PER_TURN:
            evolve_rounds += 1
            p = state.active_player_state()
            if _try_evolve_once(state):
                acted = True
                state._record_frame()
            else:
                break

        if not _is_first_player_first_turn(state) and not state.stadium_played_this_turn:
            for i, c in enumerate(p.hand):
                if is_stadium(c) and play_stadium(state, i):
                    acted = True
                    p = state.active_player_state()
                    state._record_frame()
                    break

        if _try_use_ability_runasaikuru(state):
            acted = True
            p = state.active_player_state()
            state._record_frame()
            _try_put_bench_until_full()
            continue

        if not _is_first_player_first_turn(state) and _try_support_no_discard_only(state):
            acted = True
            p = state.active_player_state()
            state._record_frame()
            _try_put_bench_until_full()
            continue

        if not _is_first_player_first_turn(state) and _try_erekijienereta(state):
            acted = True
            p = state.active_player_state()
            state._record_frame()
            _try_put_bench_until_full()
            continue

        if not state.energy_attached_this_turn and _try_attach_energy_auto(state):
            acted = True
            p = state.active_player_state()
            state._record_frame()

        opp_max_effective = _opponent_max_effective_damage(state)
        our_max_effective = _our_max_effective_damage(state)
        would_be_koed = p.active and opp_max_effective > 0 and p.active.hp <= opp_max_effective
        can_ko_opponent = opp.active and our_max_effective >= opp.active.hp

        ball_uses = 0
        while ball_uses < MAX_BALL_USES_PER_TURN:
            ball_uses += 1
            used_ball = False
            ball_candidates = [
                (i, c)
                for i, c in enumerate(p.hand)
                if is_goods(c)
                and not getattr(c, "is_tool", False)
                and getattr(c, "id", "") in _BALL_GOODS_IDS
            ]
            ball_candidates = [
                (i, c)
                for i, c in ball_candidates
                if not (getattr(c, "id", "") == "haipaboru" and state.turn_count < 2)
            ]
            ball_candidates.sort(key=lambda x: (0 if getattr(x[1], "id", "") == "pokemonkixyatchixya" else 1, x[0]))
            for i, c in ball_candidates:
                if getattr(c, "id", "") == "pokemonkixyatchixya":
                    catcher_idx = _best_opponent_bench_index_for_catcher(state)
                    opp = state.defending_player_state()
                    catcher_card_id = (
                        getattr(opp.bench[catcher_idx].card, "id", None) or getattr(opp.bench[catcher_idx].card, "name", "")
                        if catcher_idx is not None and catcher_idx < len(opp.bench)
                        else ""
                    )
                    used = use_trainer_goods(state, i, pokemon_catcher_bench_index=catcher_idx)
                    if used and catcher_card_id:
                        _log_choice(state, "catcher", card_id=catcher_card_id)
                else:
                    used = use_trainer_goods(state, i)
                if used:
                    acted = True
                    used_ball = True
                    p = state.active_player_state()
                    state._record_frame()
                    break
            if not used_ball:
                break
        _try_put_bench_until_full()
        if not _is_first_player_first_turn(state):
            evolve_after_ball = 0
            while evolve_after_ball < MAX_EVOLVE_ROUNDS_PER_TURN and state.turn_count >= 2:
                evolve_after_ball += 1
                p = state.active_player_state()
                if _try_evolve_once(state):
                    acted = True
                    state._record_frame()
                else:
                    break

        has_hand_refresh_support = any(is_support(c) and getattr(c, "id", "") in _HAND_REFRESH_SUPPORT_IDS for c in p.hand)
        if not _is_first_player_first_turn(state) and has_hand_refresh_support and not state.support_used_this_turn:
            if _try_goods_before_hand_refresh(state):
                acted = True
                p = state.active_player_state()
                state._record_frame()
                _try_put_bench_until_full()
                continue

        if not _is_first_player_first_turn(state) and not state.support_used_this_turn:
            used_support = False
            for i in _support_try_order(p, state):
                c = p.hand[i]
                support_card_id = getattr(c, "id", None) or getattr(c, "name", "")
                if use_support(state, i):
                    acted = True
                    used_support = True
                    if support_card_id:
                        _log_choice(state, "support", card_id=support_card_id)
                    p = state.active_player_state()
                    state._record_frame()
                    break
            _try_put_bench_until_full()
            if used_support:
                if _try_evolve_rounds_after_hand_change(state):
                    acted = True
                continue

        if _try_swap_for_survival(state):
            acted = True
            p = state.active_player_state()
            state._record_frame()

        if _try_retreat_when_koed(state):
            acted = True
            p = state.active_player_state()
            state._record_frame()

        if _try_retreat_voluntary(state):
            acted = True
            p = state.active_player_state()
            state._record_frame()

        if _try_attach_one_tool(state):
            acted = True
            p = state.active_player_state()
            state._record_frame()

        used_goods = False
        goods_order = [
            (i, c)
            for i, c in enumerate(p.hand)
            if is_goods(c)
            and getattr(c, "effect", None) != "swap_active"
            and not getattr(c, "is_tool", False)
            and not (getattr(c, "id", None) == "haipaboru" and state.turn_count < 2)
        ]

        def _goods_sort_key(x):
            (i, c) = x
            w = get_goods_use_weight(state.get_weights_for_player(state.current_player), c)
            cid = getattr(c, "id", None) or ""
            if (cid == "erekijienereta" or getattr(c, "name", "") == "エレキジェネレーター") and p.bench and p.deck:
                return (0, -w, i)
            return (1, -w, i)

        goods_order.sort(key=_goods_sort_key)
        p = state.active_player_state()
        for i, c in goods_order:
            try_idx = next((j for j, h in enumerate(p.hand) if h is c), -1)
            if try_idx < 0:
                continue
            if getattr(c, "id", None) == "potion" and getattr(c, "effect", None) == "heal":
                used = use_potion(state, try_idx)
            elif getattr(c, "effect", None) == "heal":
                used = False
            else:
                used = use_trainer_goods(state, try_idx)
            if used:
                acted = True
                used_goods = True
                gid = getattr(c, "id", None) or getattr(c, "name", "")
                if gid:
                    _log_choice(state, "goods", card_id=gid)
                p = state.active_player_state()
                state._record_frame()
                break
        p = state.active_player_state()
        _try_put_bench_until_full()
        if used_goods:
            if _try_evolve_rounds_after_hand_change(state):
                acted = True
            continue

        can_evolve = state.turn_count >= 2
        evolve_rounds2 = 0
        while can_evolve and evolve_rounds2 < MAX_EVOLVE_ROUNDS_PER_TURN:
            evolve_rounds2 += 1
            p = state.active_player_state()
            if _try_evolve_once(state):
                acted = True
                state._record_frame()
            else:
                break

        break

    p = state.active_player_state()
    opp = state.defending_player_state()
    if _do_attack_phase(state, p, opp):
        acted = True
        state._record_frame()

    if not acted:
        state.log(f"{state.player_name(state.current_player)}: 実行できるアクションなし（パス）")
        state._record_frame()

    return acted


def run_game_auto(state: GameState) -> int:
    """
    自動でターンを進め、勝者が決まるまで実行。
    デッキ切れ・サイド取り切り・ポケモン全滅で必ず決着。戻り値: 0 or 1 = 勝者。
    """
    if _check_game_end(state):
        if state.log_fn:
            state.log("========== ゲーム終了 ==========\n")
        return state.winner
    while True:
        start_turn(state)
        if state.winner is not None:
            if state.log_fn:
                state.log("========== ゲーム終了 ==========\n")
            return state.winner
        acted = run_turn_auto(state)
        if _check_game_end(state):
            if state.log_fn:
                state.log("========== ゲーム終了 ==========\n")
            return state.winner
        end_turn(state)
        if state.turn_count >= MAX_TURNS_SAFETY:
            p0, p1 = state.players[0], state.players[1]
            taken0 = PRIZE_COUNT - len(p0.prize_pile)
            taken1 = PRIZE_COUNT - len(p1.prize_pile)
            state.winner = 0 if taken0 >= taken1 else 1
            if state.log_fn:
                state.log(f"{MAX_TURNS_SAFETY} ターンで打ち切り（サイド取得で判定）\n")
                state.log("========== ゲーム終了 ==========\n")
            return state.winner
