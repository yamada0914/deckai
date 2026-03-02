"""ターン実行（にげる・エネルギー付与・ポケモンチェック・run_turn_auto・run_game_auto）。"""
import random

from card import is_energy, is_goods, is_pokemon, is_support

_BALL_GOODS_IDS = ("supaboru", "haipaboru", "otodokedoron", "pokemonkixyatchixya")
_HAND_REFRESH_SUPPORT_IDS = ("tanpankozou", "hakasenokenkyuu", "hakasenokenkyuufutouhakase", "jixyajjiman", "kihada")
# 手札を捨てないサポート（ネモ・キハダ）→ 複数あるとき優先
_SUPPORT_IDS_NO_DISCARD = ("nemo", "nemokako", "nemomirai", "kihada")
# 手札をすべて捨てるサポート（博士の研究）→ 手札 5 枚以上なら勿体無いので後回し
_SUPPORT_IDS_DISCARD_ALL = ("hakasenokenkyuu", "hakasenokenkyuufutouhakase")

from .attack import _choose_best_attack_index, attack
from .damage import (
    _max_effective_damage_for_attacker,
    _max_effective_damage_if_attach,
    _opponent_max_effective_damage,
    _our_max_effective_damage,
)
from .evolution import _can_evolve_onto, evolve_pokemon
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
    _can_pay_energy_cost,
    _handle_own_active_ko,
)
from .trainers import attach_energy, attach_tool, use_pokemon_swap, use_potion, use_support, use_trainer_goods


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
    if old_active.attached_energy < cost:
        return False
    types = getattr(old_active, "attached_energy_types", [])
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
    p.active = p.bench[bench_index]
    p.bench[bench_index] = old_active
    _clear_status(old_active)
    state.log(
        f"{state.player_name(state.current_player)}: {old_active.card.name} をベンチに戻し、"
        f"{p.active.card.name} をバトル場に出す（HP {p.active.hp}/{p.active.max_hp}）"
    )
    return True


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


def _try_attach_one_tool(state: GameState) -> bool:
    """手札からどうぐを 1 枚、バトル場またはベンチにつける。つけたら True。"""
    p = state.active_player_state()
    for i, c in enumerate(p.hand):
        if not is_goods(c) or not getattr(c, "is_tool", False):
            continue
        cond = getattr(c, "tool_condition_type", None)
        if p.active and getattr(p.active, "attached_tool", None) is None:
            if cond is None or getattr(p.active.card, "pokemon_type", None) == cond:
                if attach_tool(state, i, bench_index=None):
                    return True
        for bi, bp in enumerate(p.bench):
            if getattr(bp, "attached_tool", None) is not None:
                continue
            if cond is not None and getattr(bp.card, "pokemon_type", None) != cond:
                continue
            if attach_tool(state, i, bench_index=bi):
                return True
    return False


def _support_try_order(p) -> list[int]:
    """
    サポートを試す順序。手札を捨てないもの優先。
    手札がネモと博士の研究だけのときは博士の研究を先に。手札 5 枚以上なら博士の研究は後回し。
    """
    support_indices = [(i, p.hand[i]) for i in range(len(p.hand)) if is_support(p.hand[i])]
    if not support_indices:
        return []
    n_hand = len(p.hand)
    no_discard = [i for i, c in support_indices if getattr(c, "id", "") in _SUPPORT_IDS_NO_DISCARD]
    discard_all = [i for i, c in support_indices if getattr(c, "id", "") in _SUPPORT_IDS_DISCARD_ALL]
    other = [i for i, c in support_indices if i not in no_discard and i not in discard_all]
    # 手札がネモと博士の研究だけ → 博士の研究を先に（手札刷新）
    if n_hand == 2 and len(no_discard) == 1 and len(discard_all) == 1 and not other:
        return discard_all + no_discard
    # 手札 5 枚以上 → 博士の研究は勿体無いので最後に試す
    if n_hand >= 5:
        return no_discard + other + discard_all
    # 通常: 手札を捨てないものから
    return no_discard + other + discard_all


def _try_goods_before_hand_refresh(state: GameState) -> bool:
    """手札刷新系サポートの前にどうぐ・グッズを 1 回試す。使ったら True。"""
    p = state.active_player_state()
    if _try_attach_one_tool(state):
        return True
    for i, c in enumerate(p.hand):
        if not is_goods(c) or getattr(c, "effect", None) == "swap_active" or getattr(c, "is_tool", False):
            continue
        if getattr(c, "id", None) == "potion" and getattr(c, "effect", None) == "heal":
            used = use_potion(state, i)
        elif getattr(c, "effect", None) == "heal":
            used = False
        else:
            used = use_trainer_goods(state, i)
        if used:
            return True
    return False


def _try_evolve_once(state: GameState) -> bool:
    """進化を 1 回だけ試す（手札の進化でバトル場 or ベンチにのせる）。進化したら True。"""
    p = state.active_player_state()
    for hand_idx, c in enumerate(p.hand):
        if not is_pokemon(c) or not c.evolves_from:
            continue
        if p.active and _can_evolve_onto(p.active.card, c):
            evolve_pokemon(state, hand_idx, bench_index=None)
            return True
        for bench_idx, bench_poke in enumerate(p.bench):
            if _can_evolve_onto(bench_poke.card, c):
                evolve_pokemon(state, hand_idx, bench_index=bench_idx)
                return True
    return False


def _try_attach_energy_auto(state: GameState) -> bool:
    """
    自動ターン用：エネルギーを 1 枚付与するなら True を返す。
    付与先は「このエネルギーを付けたときに相手に与えられる有效ダメージが最大のポケモン」で決める。
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
    energy_needed_active = _energy_needed_for_active(p)

    if p.active.attached_energy == 0:
        attach_energy(state, energy_hand_idx)
        return True
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
    best = max(candidates, key=lambda x: (x[1], x[0] is None))
    if best[0] is None:
        attach_energy(state, energy_hand_idx)
        return True
    attach_energy(state, energy_hand_idx, bench_index=best[0])
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


def run_turn_auto(state: GameState) -> bool:
    """
    現在のプレイヤーが「可能な行動を順番に実行」する。
    順序：0. ベンチにポケモン出す、1. 進化、2. エネルギー付与、3. ボール系グッズ、4. サポート、5. いれかえ、6. にげる、7. どうぐ・グッズ、8. 進化（再）、9. 攻撃。
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

        if not state.energy_attached_this_turn and _try_attach_energy_auto(state):
            acted = True
            p = state.active_player_state()
            state._record_frame()

        opp_max_effective = _opponent_max_effective_damage(state)
        our_max_effective = _our_max_effective_damage(state)
        would_be_koed = p.active and opp_max_effective > 0 and p.active.hp <= opp_max_effective
        can_ko_opponent = opp.active and our_max_effective >= opp.active.hp

        # ボール系どうぐ（スーパーボール等）は先行 1 ターン目でも使用可（制限されるのはサポートのみ）
        ball_uses = 0
        while ball_uses < MAX_BALL_USES_PER_TURN:
            ball_uses += 1
            used_ball = False
            for i, c in enumerate(p.hand):
                if not is_goods(c) or getattr(c, "is_tool", False):
                    continue
                if getattr(c, "id", "") not in _BALL_GOODS_IDS:
                    continue
                if use_trainer_goods(state, i):
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
            for i in _support_try_order(p):
                if use_support(state, i):
                    acted = True
                    used_support = True
                    p = state.active_player_state()
                    state._record_frame()
                    break
            _try_put_bench_until_full()
            if used_support:
                continue

        p = state.active_player_state()
        opp_max_effective = _opponent_max_effective_damage(state)
        our_max_effective = _our_max_effective_damage(state)
        would_be_koed = p.active and opp_max_effective > 0 and p.active.hp <= opp_max_effective
        can_ko_opponent = opp.active and our_max_effective >= opp.active.hp
        if would_be_koed and not can_ko_opponent and p.active and p.bench:
            for i, c in enumerate(p.hand):
                if is_goods(c) and (getattr(c, "effect", None) == "swap_active" or getattr(c, "id", "") in ("pokemon_irekae", "pokemonirekae")):
                    survives = [(bi, p.bench[bi].hp) for bi in range(len(p.bench)) if p.bench[bi].hp > opp_max_effective]
                    if survives:
                        best_bench = max(survives, key=lambda x: x[1])[0]
                    else:
                        best_bench = max(range(len(p.bench)), key=lambda b: p.bench[b].hp, default=None)
                    if best_bench is not None and use_pokemon_swap(state, i, best_bench):
                        acted = True
                        p = state.active_player_state()
                        state._record_frame()
                    break

        p = state.active_player_state()
        opp_max_effective = _opponent_max_effective_damage(state)
        our_max_effective = _our_max_effective_damage(state)
        would_be_koed = p.active and opp_max_effective > 0 and p.active.hp <= opp_max_effective
        can_ko_opponent = opp.active and our_max_effective >= opp.active.hp
        retreat_cost = getattr(p.active.card, "retreat_cost", 1) if p.active else 0
        can_retreat = p.active and getattr(p.active, "special_state", None) not in ("sleep", "paralysis")
        has_bench_survivor = p.bench and any(bp.hp > opp_max_effective for bp in p.bench)
        can_ko_from_bench = opp.active and any(
            _max_effective_damage_for_attacker(state, bp, opp.active, state.current_player) >= opp.active.hp for bp in p.bench
        )
        retreat_helps = has_bench_survivor or can_ko_from_bench
        if can_retreat and p.active and p.bench and would_be_koed and not can_ko_opponent and retreat_helps and p.active.attached_energy >= retreat_cost:
            best_idx = None
            best_score = (-1, -1, -1)
            for i, bp in enumerate(p.bench):
                dmg = _max_effective_damage_for_attacker(state, bp, opp.active, state.current_player) if opp.active else 0
                can_ko = int(opp.active is not None and dmg >= opp.active.hp)
                score = (can_ko, dmg, bp.hp)
                if score > best_score:
                    best_score = score
                    best_idx = i
            if best_idx is None:
                survivors = [(i, p.bench[i].hp, p.bench[i].attached_energy) for i in range(len(p.bench)) if p.bench[i].hp > opp_max_effective]
                if survivors:
                    best_idx = max(survivors, key=lambda x: (x[1], x[2]))[0]
                else:
                    best_idx = max(
                        range(len(p.bench)),
                        key=lambda i: (p.bench[i].hp, p.bench[i].attached_energy),
                        default=None,
                    )
            if best_idx is not None and retreat(state, best_idx):
                acted = True
                p = state.active_player_state()
                state._record_frame()

        if _try_attach_one_tool(state):
            acted = True
            p = state.active_player_state()
            state._record_frame()

        used_goods = False
        for i, c in enumerate(p.hand):
            if not is_goods(c) or getattr(c, "effect", None) == "swap_active" or getattr(c, "is_tool", False):
                continue
            if getattr(c, "id", None) == "potion" and getattr(c, "effect", None) == "heal":
                used = use_potion(state, i)
            elif getattr(c, "effect", None) == "heal":
                used = False
            else:
                used = use_trainer_goods(state, i)
            if used:
                acted = True
                used_goods = True
                p = state.active_player_state()
                state._record_frame()
                break
        p = state.active_player_state()
        _try_put_bench_until_full()
        if used_goods:
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

    is_game_first_turn = state.turn_count == 0
    can_attack = not is_game_first_turn and p.active and getattr(p.active, "special_state", None) not in ("sleep", "paralysis")
    if can_attack:
        best_idx = _choose_best_attack_index(state, p, opp)
        if best_idx is not None:
            attack(state, best_idx)
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
        if not acted:
            end_turn(state)
        else:
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
