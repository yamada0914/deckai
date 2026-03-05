"""ターン実行（にげる・エネルギー付与・ポケモンチェック・run_turn_auto・run_game_auto）。"""
import random

from card import is_energy, is_goods, is_pokemon, is_support

_BALL_GOODS_IDS = ("supaboru", "haipaboru", "otodokedoron", "pokemonkixyatchixya")
# 先行1ターン目など進化不可のターン (turn_count < 2) ではハイパーボールは使わない（次のターンに回す）
_HAND_REFRESH_SUPPORT_IDS = ("tanpankozou", "hakasenokenkyuu", "hakasenokenkyuufutouhakase", "jixyajjiman", "kihada")
# 手札を捨てないサポート（ネモ・キハダ）→ エネルギー付与前に試すブロックで使用
_SUPPORT_IDS_NO_DISCARD = ("nemo", "nemokako", "nemomirai", "kihada")
_SUPPORT_IDS_DISCARD_ALL = ("hakasenokenkyuu", "hakasenokenkyuufutouhakase")

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
    _log_choice,
)
from .trainers import attach_energy, attach_tool, use_pokemon_swap, use_potion, use_support, use_trainer_goods
from .weights import (
    get_catcher_target_weight,
    get_energy_attach_weight,
    get_evolve_onto_weight,
    get_goods_use_weight,
    get_retreat_target_weight,
    get_support_use_weight,
    get_swap_target_weight,
    get_tool_attach_weight,
)


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


def _min_energy_for_any_attack(card) -> int:
    """そのカードの技のうち、必要エネルギー数が最小の値を返す（技が遅い＝大きいほど良い）。"""
    if not getattr(card, "attacks", None):
        return 0
    return max(
        (len(getattr(a, "energy_cost_typed", []) or []) or getattr(a, "energy_cost", 0) for a in card.attacks),
        default=0,
    )


# ポケモンキャッチャーで「脅威」とみなすスコアのスケール（HP・ex で脅威度を付与）
_CATCHER_THREAT_HP_SCALE = 1.0
_CATCHER_THREAT_EX_BONUS = 5000.0


def _catcher_threat_score(card) -> float:
    """相手ベンチの 1 体がどれだけ脅威か。HP が高く・ex なら高くなる。"""
    hp = getattr(card, "max_hp", 0) or 0
    is_ex = getattr(card, "is_ex", False) or ("ex" in (getattr(card, "name", "") or ""))
    return hp * _CATCHER_THREAT_HP_SCALE + (_CATCHER_THREAT_EX_BONUS if is_ex else 0.0)


def _best_opponent_bench_index_for_catcher(state: GameState) -> int | None:
    """
    ポケモンキャッチャーで引きたい相手ベンチのインデックスを返す。
    脅威（HP が高い・ex）を優先して引き出し、同程度なら逃げにくく・技が遅いポケモンを優先し、重みで補正する。
    """
    opp = state.defending_player_state()
    if not opp.bench or not opp.active:
        return None
    weights = state.get_weights_for_player(state.current_player)
    best_idx = None
    best_score = -1.0
    for i, bp in enumerate(opp.bench):
        card = bp.card
        threat = _catcher_threat_score(card)
        retreat = getattr(card, "retreat_cost", 0)
        min_energy = _min_energy_for_any_attack(card)
        raw = retreat * 1000 + min_energy
        score = threat + raw + get_catcher_target_weight(weights, card)
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx


# 生贄に出す候補のカード名（ここに追加すれば種類を増やせる）。いなければ「相手を倒せない・HP 低い」ベンチを候補にする。
_SACRIFICE_POKEMON_NAMES = frozenset({"カラミンゴ"})


def _best_sacrifice_bench_index(state: GameState) -> int | None:
    """
    生贄に出すベンチのインデックスを返す。
    まず _SACRIFICE_POKEMON_NAMES に名前があるポケモンを優先し、いなければ
    「今のエネルギーでは相手を倒せない」ベンチのうち max_hp が最小の 1 体を選ぶ（カード追加に強い）。
    """
    p = state.active_player_state()
    opp = state.defending_player_state()
    if not p.bench:
        return None
    for i, bp in enumerate(p.bench):
        if getattr(bp.card, "name", "") in _SACRIFICE_POKEMON_NAMES:
            return i
    if not opp.active:
        return None
    cannot_ko = []
    for i, bp in enumerate(p.bench):
        dmg = _max_effective_damage_for_attacker(state, bp, opp.active, state.current_player)
        if dmg < opp.active.hp:
            cannot_ko.append((i, getattr(bp.card, "max_hp", 60)))
    if not cannot_ko:
        return None
    return min(cannot_ko, key=lambda x: x[1])[0]


def _has_bench_that_can_ko_after_one_attach(state: GameState) -> bool:
    """
    手札のエネルギーを 1 つ付けたときに相手をきぜつさせられるベンチが 1 体でもいれば True。
    カード名に依存せず、どのデッキでも使える。
    """
    p = state.active_player_state()
    opp = state.defending_player_state()
    if not opp.active or not p.bench:
        return False
    types_to_try = list({getattr(c, "energy_type", "colorless") for c in p.hand if is_energy(c)} or ["colorless"])
    for bp in p.bench:
        for et in types_to_try:
            dmg = _max_effective_damage_if_attach(
                state,
                bp.card,
                bp.attached_energy,
                getattr(bp, "attached_energy_types", []),
                et,
                opp.active,
                state.current_player,
            )
            if dmg >= opp.active.hp:
                return True
    return False


def _max_energy_for_pokemon(attacks: list) -> int:
    """そのポケモンの技で必要な最大エネルギー数（個数）を返す。"""
    return max((a.energy_cost for a in attacks), default=0)


def _try_attach_one_tool(state: GameState) -> bool:
    """手札からどうぐを 1 枚、バトル場またはベンチにつける。つけたら True。重みで付ける先を選ぶ。"""
    p = state.active_player_state()
    weights = state.get_weights_for_player(state.current_player)
    candidates = []
    for i, c in enumerate(p.hand):
        if not is_goods(c) or not getattr(c, "is_tool", False):
            continue
        cond = getattr(c, "tool_condition_type", None)
        if p.active and getattr(p.active, "attached_tool", None) is None:
            if cond is None or getattr(p.active.card, "pokemon_type", None) == cond:
                candidates.append((i, None, p.active.card))
        for bi, bp in enumerate(p.bench):
            if getattr(bp, "attached_tool", None) is not None:
                continue
            if cond is not None and getattr(bp.card, "pokemon_type", None) != cond:
                continue
            candidates.append((i, bi, bp.card))
    if not candidates:
        return False
    best = max(
        candidates,
        key=lambda x: (get_tool_attach_weight(weights, x[2]), x[1] is None),
    )
    hand_i, bench_idx, _ = best
    if attach_tool(state, hand_i, bench_index=bench_idx):
        target_card = p.active.card if bench_idx is None else p.bench[bench_idx].card
        tid = getattr(target_card, "id", None) or getattr(target_card, "name", "")
        if tid:
            _log_choice(state, "tool_attach", card_id=tid)
        return True
    return False


def _support_try_order(p: PlayerState, state: GameState) -> list[int]:
    """
    サポートを試す順序。重み (w_support_use) のみで決める。同点は手札インデックスで安定ソート。
    固定ロジックは使わず、博士の研究の前後なども重みで制御する。
    """
    support_indices = [(i, p.hand[i]) for i in range(len(p.hand)) if is_support(p.hand[i])]
    if not support_indices:
        return []
    weights = state.get_weights_for_player(state.current_player)
    support_indices.sort(key=lambda x: (-get_support_use_weight(weights, x[1]), x[0]))
    return [i for i, c in support_indices]


def _try_support_no_discard_only(state: GameState) -> bool:
    """手札を捨てないサポート（キハダ・ネモ等）を 1 枚だけ試す。使ったら True。エネルギー付与前にドローしたいとき用。"""
    if state.support_used_this_turn:
        return False
    p = state.active_player_state()
    weights = state.get_weights_for_player(state.current_player)
    no_discard = [
        (i, c)
        for i, c in enumerate(p.hand)
        if is_support(c) and getattr(c, "id", "") in _SUPPORT_IDS_NO_DISCARD
    ]
    if not no_discard:
        return False
    no_discard.sort(key=lambda x: -get_support_use_weight(weights, x[1]))
    for i, c in no_discard:
        if use_support(state, i):
            support_card_id = getattr(c, "id", None) or getattr(c, "name", "")
            if support_card_id:
                _log_choice(state, "support", card_id=support_card_id)
            return True
    return False


def _try_erekijienereta(state: GameState) -> bool:
    """エレキジェネレーターを手札から 1 枚だけ試す。ベンチ・山札があれば使用。使ったら True。"""
    p = state.active_player_state()
    if not p.bench or not p.deck:
        return False
    for i, c in enumerate(p.hand):
        if not is_goods(c):
            continue
        cid = getattr(c, "id", None) or ""
        if cid != "erekijienereta" and (getattr(c, "name", "") or "") != "エレキジェネレーター":
            continue
        if use_trainer_goods(state, i):
            _log_choice(state, "goods", card_id=cid or "erekijienereta")
            return True
    return False


def _try_goods_before_hand_refresh(state: GameState) -> bool:
    """手札刷新系サポートの前にどうぐ・グッズを 1 回試す。使ったら True。重みで試す順を決める。"""
    p = state.active_player_state()
    if _try_attach_one_tool(state):
        return True
    weights = state.get_weights_for_player(state.current_player)
    goods_list = [
        (i, c)
        for i, c in enumerate(p.hand)
        if is_goods(c)
        and getattr(c, "effect", None) != "swap_active"
        and not getattr(c, "is_tool", False)
        and not (getattr(c, "id", None) == "haipaboru" and state.turn_count < 2)
    ]
    goods_list.sort(key=lambda x: -get_goods_use_weight(weights, x[1]))
    for i, c in goods_list:
        if getattr(c, "id", None) == "potion" and getattr(c, "effect", None) == "heal":
            used = use_potion(state, i)
        elif getattr(c, "effect", None) == "heal":
            used = False
        else:
            used = use_trainer_goods(state, i)
        if used:
            gid = getattr(c, "id", None) or getattr(c, "name", "")
            if gid:
                _log_choice(state, "goods", card_id=gid)
            return True
    return False


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


# depth=2 minimax 用の終局スコア（勝ち/負け）
_MINIMAX_WIN_SCORE = 1e9
_MINIMAX_LOSE_SCORE = -1e9


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
        if clone.winner is not None:
            score = _MINIMAX_WIN_SCORE if clone.winner == me else _MINIMAX_LOSE_SCORE
        else:
            end_turn(clone)
            if clone.winner is not None:
                score = _MINIMAX_WIN_SCORE if clone.winner == me else _MINIMAX_LOSE_SCORE
            else:
                start_turn(clone)
                if clone.winner is not None:
                    score = _MINIMAX_WIN_SCORE if clone.winner == me else _MINIMAX_LOSE_SCORE
                else:
                    run_turn_auto(clone)
                    _check_game_end(clone)
                    if clone.winner is not None:
                        score = _MINIMAX_WIN_SCORE if clone.winner == me else _MINIMAX_LOSE_SCORE
                    else:
                        score = evaluate_board(clone, me)
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
    state.our_ko_by_damage_last_turn[cp] = False  # アベンジナックル用。「前の相手の番にきぜつ」は自分のターン終了でリセット
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
    順序：0. ベンチにポケモン出す、1. 進化、2. 手札を捨てないサポート（キハダ・ネモ）、3. エネルギー付与、4. ボール系グッズ、5. サポート（博士の研究・たんぱんこぞう等はここで試行）、6. いれかえ、7. にげる、8. どうぐ・グッズ、9. 進化（再）、10. 攻撃。
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

        p = state.active_player_state()
        opp_max_effective = _opponent_max_effective_damage(state)
        our_max_effective = _our_max_effective_damage(state)
        would_be_koed = p.active and opp_max_effective > 0 and p.active.hp <= opp_max_effective
        can_ko_opponent = opp.active and our_max_effective >= opp.active.hp
        if would_be_koed and not can_ko_opponent and p.active and p.bench:
            for i, c in enumerate(p.hand):
                if is_goods(c) and (getattr(c, "effect", None) == "swap_active" or getattr(c, "id", "") in ("pokemon_irekae", "pokemonirekae")):
                    _SWAP_SURVIVE_SCALE = 10000
                    def _swap_score(bi: int) -> float:
                        bp = p.bench[bi]
                        survives = bp.hp > opp_max_effective
                        return (1 if survives else 0) * _SWAP_SURVIVE_SCALE + bp.hp + get_swap_target_weight(state.get_weights_for_player(state.current_player), bp.card)
                    best_bench = max(range(len(p.bench)), key=_swap_score)
                    sc = getattr(p.bench[best_bench].card, "id", None) or getattr(p.bench[best_bench].card, "name", "")
                    _log_choice(state, "swap", card_id=sc)
                    if use_pokemon_swap(state, i, best_bench):
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
            _RETREAT_C_KO, _RETREAT_C_DMG = 1_000_000, 1000
            best_idx = None
            best_score = -1.0
            for i, bp in enumerate(p.bench):
                dmg = _max_effective_damage_for_attacker(state, bp, opp.active, state.current_player) if opp.active else 0
                can_ko = int(opp.active is not None and dmg >= opp.active.hp)
                raw = can_ko * _RETREAT_C_KO + dmg * _RETREAT_C_DMG + bp.hp
                score = raw + get_retreat_target_weight(state.get_weights_for_player(state.current_player), bp.card)
                if score > best_score:
                    best_score = score
                    best_idx = i
            if best_idx is None:
                survivors = [(i, p.bench[i].hp, p.bench[i].attached_energy) for i in range(len(p.bench)) if p.bench[i].hp > opp_max_effective]
                if survivors:
                    best_idx = max(
                        survivors,
                        key=lambda x: (x[1], x[2], get_retreat_target_weight(state.get_weights_for_player(state.current_player), p.bench[x[0]].card)),
                    )[0]
                else:
                    best_idx = max(
                        range(len(p.bench)),
                        key=lambda i: (p.bench[i].hp, p.bench[i].attached_energy, get_retreat_target_weight(state.get_weights_for_player(state.current_player), p.bench[i].card)),
                        default=None,
                    )
            if best_idx is not None:
                rc = getattr(p.bench[best_idx].card, "id", None) or getattr(p.bench[best_idx].card, "name", "")
                _log_choice(state, "retreat", card_id=rc)
                if retreat(state, best_idx):
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

    is_game_first_turn = state.turn_count == 0
    can_attack = not is_game_first_turn and p.active and getattr(p.active, "special_state", None) not in ("sleep", "paralysis")
    if can_attack:
        best_idx = None
        if getattr(state, "use_attack_minimax", True):
            best_idx = _choose_best_attack_index_minimax(state, p, opp)
        if best_idx is None:
            best_idx = _choose_best_attack_index(state, p, opp)
        if best_idx is not None:
            atk = p.active.card.attacks[best_idx]
            ac_id = getattr(p.active.card, "id", None) or getattr(p.active.card, "name", "")
            _log_choice(state, "attack", card_id=ac_id, attack_name=atk.name)
            base_dmg = _attack_damage_for_eval(atk)
            effective_dmg = (
                _effective_damage_to_defender(p.active.card, opp.active, base_dmg)
                if opp.active
                else base_dmg
            )
            retreat_cost = getattr(p.active.card, "retreat_cost", 1)
            can_retreat_here = (
                p.bench
                and getattr(p.active, "special_state", None) not in ("sleep", "paralysis")
                and p.active.attached_energy >= retreat_cost
            )
            sacrifice_idx = _best_sacrifice_bench_index(state)
            if (
                opp.active
                and effective_dmg <= 0
                and attack_has_merit_effect_at_zero_damage(p.active.card, atk)
                and can_retreat_here
                and sacrifice_idx is not None
                and _has_bench_that_can_ko_after_one_attach(state)
            ):
                if retreat(state, sacrifice_idx):
                    acted = True
                    state._record_frame()
                else:
                    attack(state, best_idx)
                    acted = True
                    state._record_frame()
            else:
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
