"""ターン実行（にげる・エネルギー付与・ポケモンチェック・run_turn_auto・run_game_auto）。"""
import math
import random

from card import is_energy, is_goods, is_pokemon, is_stadium, is_support

from .attack import (
    _choose_best_attack_index,
    attack,
    attack_has_merit_effect_at_zero_damage,
    get_legal_attack_indices,
    get_legal_attack_indices_for_attacker,
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
    rules_only_for_player,
)
from .trainers import (
    play_stadium,
    use_pokemon_swap,
    use_potion,
    use_support,
    use_trainer_goods,
    _try_use_ability_runasaikuru,
    _try_use_ability_sakatenitori,
    _try_use_ability_adrenabrain,
    _try_use_ability_okunote_catch,
    _try_use_ability_teisatsushirei,
    _try_use_ability_cursed_bomb,
)
from .turn_bench import (
    _best_opponent_bench_index_for_catcher,
    _best_sacrifice_bench_index,
    _has_bench_that_can_ko_after_one_attach,
)
from .turn_energy import (
    _build_energy_attach_input,
    _should_attach_for_evolution,
    _pick_energy_hand_idx,
    _try_attach_energy_auto,
    _try_evolve_once,
    _try_evolve_rounds_after_hand_change,
)
from .turn_trainers import (
    _HAIPABORU_BEFORE_HAND_REFRESH_SUPPORT_IDS,
    _SUPPORT_IDS_HAND_REFRESH_FIRST,
    _support_try_order,
    _try_angou_before_luna_cycle,
    _try_attach_one_tool,
    _try_erekijienereta,
    _try_goods_before_hand_refresh,
    _try_support_no_discard_only,
)
from .retreat_before_attack import try_retreat_before_attack_policy
from .support_policy import try_support_policy, KNOWN_SUPPORT_IDS
from .weights import get_attack_weight, get_goods_use_weight, get_retreat_target_weight, get_swap_target_weight
from .encoders import encode_state_basic, encode_state_opening, encode_state_v2
from .pi_models import load_pi_model_pt
from .value_models import load_value_model_pt, load_state_value_model_pt

from .state import _prizes_for_ko, PRIZE_COUNT

_BALL_GOODS_IDS = ("faitogongu", "pokepaddo", "supaboru", "haipaboru", "otodokedoron", "pokemonkixyatchixya", "nakayoshipofuin")

# ソルロック攻撃セットアップ（ルナトーン必須）:
# 「次ターンのドローが確約されている」扱いのため、手札に手札補充サポートがある場合のみ
# ソルロックを能動的にバトル場へ出しに行く。
_SOLROCK_NEXT_TURN_DRAW_SUPPORT_IDS = (
    "nemo",
    "nemokako",
    "nemomirai",
    "kihada",
    "tanpankozou",
    "hakasenokenkyuu",
    "hakasenokenkyuufutouhakase",
    "jixyajjiman",
    "riirienokesshin",
    "zeiyu",
)


def _is_lunatone(card) -> bool:
    n = (getattr(card, "name", "") or "").strip()
    cid = (getattr(card, "id", "") or "").strip()
    return n == "ルナトーン" or cid.startswith("runaton")


def _is_solrock(card) -> bool:
    n = (getattr(card, "name", "") or "").strip()
    cid = (getattr(card, "id", "") or "").strip()
    return n == "ソルロック" or cid.startswith("sorurokku-mc-372") or cid.startswith("sorurokku")


def _field_has_pokemon(p: PlayerState, matcher) -> bool:
    if p.active and p.active.card and matcher(p.active.card):
        return True
    for bp in (p.bench or []):
        if bp and bp.card and matcher(bp.card):
            return True
    return False


def _has_draw_support_in_hand_for_solrock(p: PlayerState) -> bool:
    for c in p.hand:
        if is_support(c) and (getattr(c, "id", "") or "") in _SOLROCK_NEXT_TURN_DRAW_SUPPORT_IDS:
            return True
    return False


def _solrock_bench_index(p: PlayerState) -> int | None:
    for i, bp in enumerate(p.bench or []):
        if bp and bp.card and _is_solrock(bp.card):
            return i
    return None


def _try_faitogongu_engine_opener(state: GameState) -> bool:
    """
    手札にファイトゴングが 2 枚以上かつ基本闘エネがあり、場にルナトーン・ソルロックが揃っていないとき、
    ルナサイクルやエネルギー付与より先にファイトゴングを最大 2 回使う（山札からエンジンを取る）。
    """
    if _is_first_player_first_turn(state):
        return False
    if getattr(state, "energy_attached_this_turn", False):
        return False
    if getattr(state, "ability_declared_this_turn", None) == "ルナサイクル":
        return False
    p = state.active_player_state()
    if not p.deck:
        return False

    gong_ix = [
        i
        for i, c in enumerate(p.hand)
        if is_goods(c) and (getattr(c, "id", "") == "faitogongu" or getattr(c, "name", "") == "ファイトゴング")
    ]
    if len(gong_ix) < 2:
        return False
    has_basic_fighting = any(
        is_energy(c)
        and ((getattr(c, "id", "") or "") == "basic-energy-fighting" or getattr(c, "name", "") == "基本闘エネルギー")
        for c in (p.hand or [])
    )
    if not has_basic_fighting:
        return False
    lunatone_in_field = bool(p.active and _is_lunatone(p.active.card)) or any(
        _is_lunatone(bp.card) for bp in (p.bench or [])
    )
    solrock_in_field = bool(p.active and _is_solrock(p.active.card)) or any(
        _is_solrock(bp.card) for bp in (p.bench or [])
    )
    if lunatone_in_field and solrock_in_field:
        return False

    used_any = False
    for _ in range(2):
        p = state.active_player_state()
        indices = [
            i
            for i, c in enumerate(p.hand)
            if is_goods(c) and (getattr(c, "id", "") == "faitogongu" or getattr(c, "name", "") == "ファイトゴング")
        ]
        if not indices:
            break
        idx = indices[0]
        if use_trainer_goods(state, idx):
            used_any = True
            _log_choice(state, "goods", card_id="faitogongu")
        else:
            break
    return used_any


def retreat(state: GameState, bench_index: int) -> bool:
    """バトル場のポケモンとベンチの bench_index 番目を入れ替える（逃げる）。にげるエネルギー分を捨てる。ねむり・マヒ中はにげられない。"""
    p = state.active_player_state()
    if not p.active or bench_index < 0 or bench_index >= len(p.bench):
        return False
    # ドラパルトデッキ: ドロンチをバトル場に出さない（次ターンで即KOされるリスク）
    # ドラパルトexのみバトル場に出す
    from .deck_strategies import is_dragapult_deck_for_player as _is_drapa_ret
    if _is_drapa_ret(state, state.current_player):
        target_name = (getattr(p.bench[bench_index].card, "name", "") or "").strip()
        if target_name == "ドロンチ":
            return False  # ドロンチは前に出さない
    if getattr(p.active, "special_state", None) in ("sleep", "paralysis"):
        state.log(f"{state.player_name(state.current_player)}: {p.active.card.name} は状態異常のためにげられない")
        return False
    if getattr(p.active, "retreat_locked", False):
        state.log(f"{state.player_name(state.current_player)}: {p.active.card.name} は「かげしばり」のためにげられない")
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
    """攻撃適用済み clone に対して、相手ターン・自分ターンを進めた後のスコアを返す。
    state_value_model_path が設定されていれば evaluate_board を V(s) で補正する。
    """
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
    base = evaluate_board(clone, me)
    by_player = getattr(clone, "state_value_model_path_by_player", [None, None])
    sv_path = (
        by_player[me]
        if me < len(by_player) and by_player[me] is not None
        else getattr(clone, "state_value_model_path", None)
    )
    if sv_path:
        sv_model = load_state_value_model_pt(sv_path)
        # モデルのstate_dimに応じてエンコーダを自動選択
        from .encoders import encode_state_drapa
        if sv_model.state_dim <= 25:
            state_vec = encode_state_basic(clone, me)
        elif sv_model.state_dim <= 45:
            state_vec = encode_state_drapa(clone, me)
        else:
            state_vec = encode_state_v2(clone, me)
        v = sv_model.predict_one(state_vec)
        alpha = float(getattr(clone, "state_value_lambda", 0.3))
        alpha = max(0.0, min(1.0, alpha))
        return (1.0 - alpha) * base + alpha * v
    return base


# minimax の技選択で「学習した重み」をどれだけ効かせるか。盤面評価の 1 目安（約 10）と同程度になるよう 5 倍する。
_ATTACK_WEIGHT_SCALE_IN_MINIMAX = 5.0


def _choose_best_attack_index_minimax(state: GameState, p: PlayerState, opp: PlayerState) -> int | None:
    """
    2 手読み（自分が攻撃 → 相手のターン → 盤面評価）で最良の技インデックスを返す。
    候補が 0 件なら None。シミュレーションでは log / record_frame は出さない。
    学習した重みをスコアに加算し、評価が近いときに重みで差をつける（_ATTACK_WEIGHT_SCALE_IN_MINIMAX 倍して効かせる）。
    """
    legal = get_legal_attack_indices(state, p, opp)
    if not legal:
        return None
    me = state.current_player
    weights = state.get_weights_for_player(me)
    best_idx = None
    best_score = _MINIMAX_LOSE_SCORE - 1
    for idx in legal:
        clone = state.copy_for_simulation()
        attack(clone, idx)
        score = _minimax_score_after_attack(clone, me)
        atk = p.active.card.attacks[idx]
        score += _ATTACK_WEIGHT_SCALE_IN_MINIMAX * get_attack_weight(weights, p.active.card, atk)

        atk_name = getattr(atk, "name", "")

        # KO確定ボーナス: この技で相手を倒せるなら大ボーナス。
        # 倒せる技が複数あるなら、エネ加速等の副効果でさらに差をつける。
        if opp.active and opp.active.hp is not None and opp.active.hp > 0:
            from .damage import _max_effective_damage_for_attacker
            eff_dmg = _max_effective_damage_for_attacker(state, p.active, opp.active, me)

            if atk_name == "はどうづき":
                fighting_in_trash = sum(
                    1 for c in p.discard
                    if is_energy(c) and getattr(c, "energy_type", None) == "fighting"
                )
                accel_count = min(fighting_in_trash, 3)
                score += accel_count * 15.0

            # メガブレイブペナルティ: KO確定しないなら使いたくない
            if atk_name == "メガブレイブ" and eff_dmg < opp.active.hp:
                score -= 50.0

            # KOできる技には圧倒的ボーナス（倒せるなら確実に倒す）。
            # minimaxの先読みスコアを上回る必要がある。
            if eff_dmg >= opp.active.hp:
                opp_prize_value = _prizes_for_ko(opp.active)
                score += 100000.0 + opp_prize_value * 10000.0  # KO確定ボーナス

                our_prize_cost = _prizes_for_ko(p.active)
                if our_prize_cost == 1 and opp_prize_value >= 2:
                    score += (opp_prize_value - our_prize_cost) * 20.0

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
    if hasattr(state, "any_ko_by_opponent_last_turn"):
        state.any_ko_by_opponent_last_turn[cp] = False
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
    # むずむずかふん: グッズロックをクリア（ロックされた側のターン終了時）
    if getattr(state, "goods_locked_next_turn", None) == cp:
        state.goods_locked_next_turn = None
    state.current_player = state.opponent()
    state.turn_count += 1
    state.log("")


def _is_lucario_stage_attacker_card_for_swap(card) -> bool:
    """リオル以外のルカリオライン（いれかえの無意味な同名入替の判定用）。"""
    if not card:
        return False
    n = (getattr(card, "name", "") or "").strip()
    cid = (getattr(card, "id", "") or "").strip().lower()
    if cid.startswith("rioru"):
        return False
    if n in ("ルカリオ", "メガルカリオ", "メガルカリオex"):
        return True
    return cid.startswith("rukario") or cid.startswith("mrukario")


def _try_swap_for_survival(state: GameState) -> bool:
    """
    きぜつ確定で相手を倒せないとき、いれかえグッズで生存できるベンチに替える。替えたら True。
    ふうせんでにげ 0 のときは使わない。入れ替え先が技不能で前だけ技が出せるとき・同名ルカリオ系の無意味入替もしない。
    ベンチに次の相手攻撃を耐えて技が出せるルカリオ系がいれば、ソルロック固定よりそちらを前に出す。
    """
    p = state.active_player_state()
    opp = state.defending_player_state()
    opp_max_effective = _opponent_max_effective_damage(state)
    our_max_effective = _our_max_effective_damage(state)
    would_be_koed = p.active and opp_max_effective > 0 and p.active.hp <= opp_max_effective
    can_ko_opponent = opp.active and our_max_effective >= opp.active.hp
    if not (would_be_koed and not can_ko_opponent and p.active and p.bench):
        return False
    _SWAP_SURVIVE_SCALE = 10000
    opp_remaining_prizes = len(opp.prize_pile) if hasattr(opp, "prize_pile") else PRIZE_COUNT
    for i, c in enumerate(p.hand):
        if not is_goods(c) or (getattr(c, "effect", None) != "swap_active" and getattr(c, "id", "") not in ("pokemon_irekae", "pokemonirekae")):
            continue
        weights = state.get_weights_for_player(state.current_player)

        def _swap_score(bi: int) -> float:
            bp = p.bench[bi]
            survives = bp.hp > opp_max_effective
            # サイド管理: 相手の残りサイドが少ないとき、高サイドポケモンを前に出すペナルティ
            prize_pen = _prize_management_penalty(bp, opp_remaining_prizes)
            return (1 if survives else 0) * _SWAP_SURVIVE_SCALE + bp.hp + get_swap_target_weight(weights, bp.card) + prize_pen

        def _best_lucario_bench_for_survival_swap() -> int | None:
            """次の相手ダメージを耐え、技が出せるベンチのルカリオ系がいればそのインデックス（メインアタッカーを前に）。"""
            best_bi: int | None = None
            best_key: tuple = (-1, -1, -1.0)
            for bi, bp in enumerate(p.bench):
                if not _is_lucario_stage_attacker_card_for_swap(bp.card):
                    continue
                hp = bp.hp
                if hp is None or hp <= opp_max_effective:
                    continue
                if not get_legal_attack_indices_for_attacker(state, p, opp, bp):
                    continue
                dmg = 0
                if opp.active and getattr(opp.active, "hp", None) is not None and opp.active.hp > 0:
                    dmg = _max_effective_damage_for_attacker(state, bp, opp.active, state.current_player)
                w = get_swap_target_weight(weights, bp.card)
                key = (dmg, hp, w)
                if key > best_key:
                    best_key = key
                    best_bi = bi
            return best_bi

        # ソルロック前出しはルナ＋ドローサポートがあるときのフォールバック。ルカリオ系が前に立てるならそちらを優先。
        solrock_bi = _solrock_bench_index(p)
        solrock_plan_ok = (
            solrock_bi is not None
            and _field_has_pokemon(p, _is_lunatone)
            and _has_draw_support_in_hand_for_solrock(p)
            and p.bench[solrock_bi].hp > opp_max_effective
        )
        lucario_survivor_bi = _best_lucario_bench_for_survival_swap()
        if lucario_survivor_bi is not None:
            best_bench = lucario_survivor_bi
        elif solrock_plan_ok:
            best_bench = solrock_bi
        else:
            best_bench = max(range(len(p.bench)), key=_swap_score)
        bench_bp = p.bench[best_bench]

        # ふうせん等でにげ 0 ならいれかえでグッズを消費しない（この後のにげでよい）
        if getattr(p.active, "special_state", None) not in ("sleep", "paralysis"):
            raw_rc = getattr(p.active.card, "retreat_cost", 1)
            tool = getattr(p.active, "attached_tool", None)
            eff_retreat = max(0, raw_rc - (2 if tool and (getattr(tool, "id", "") or "") == "fuusen" else 0))
            if eff_retreat == 0 and not getattr(state, "retreat_used_this_turn", False):
                continue

        legal_act = get_legal_attack_indices_for_attacker(state, p, opp, p.active)
        legal_bench = get_legal_attack_indices_for_attacker(state, p, opp, bench_bp)
        # 今の前で技が出せるのに、入れ替え先では出せない → 防戦一方になりやすい。波動などで殴ってから次ターンを狙う。
        if len(legal_act) > 0 and len(legal_bench) == 0:
            continue

        an = (getattr(p.active.card, "name", "") or "").strip()
        bn = (getattr(bench_bp.card, "name", "") or "").strip()
        if (
            an == bn
            and an
            and _is_lucario_stage_attacker_card_for_swap(p.active.card)
            and _is_lucario_stage_attacker_card_for_swap(bench_bp.card)
        ):
            continue

        sc = getattr(bench_bp.card, "id", None) or getattr(bench_bp.card, "name", "")
        _log_choice(state, "swap", card_id=sc)
        if use_pokemon_swap(state, i, best_bench):
            return True
        break
    return False


def _prize_management_penalty(bp, opp_remaining_prizes: int) -> float:
    """
    サイド管理: 相手の残りサイドが少ないとき、メガルカリオex等の高サイドポケモンを
    前に出すペナルティを返す（負の値）。
    「サイドを効率よく取らせない」戦略:
    - 相手の残り≤3 でメガ(3枚)を前に出すのは無駄が出る可能性あり→ペナルティ
    - 相手の残り≤2 でex(2枚)を前に出すと、1枚余分に取らせてしまう場面は許容
    - 相手の残り1 でex/メガを前に出すと、1枚取られるだけで倒せないポケモンの方がよい
    """
    prizes_given = _prizes_for_ko(bp)
    if prizes_given <= 1:
        return 0.0
    # 相手がこのポケモンを倒すと、余分なサイドが出る場合にペナルティ
    wasted = max(0, prizes_given - opp_remaining_prizes)
    if wasted > 0:
        # 相手にとって無駄なサイドが出る＝こちらの損失
        return -wasted * 5000.0
    # 相手の残りサイドが少ないほど、高サイドポケモンは避けたい
    if opp_remaining_prizes <= 3 and prizes_given >= 3:
        return -3000.0
    if opp_remaining_prizes <= 2 and prizes_given >= 2:
        return -1500.0
    return 0.0


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

    # ソルロック攻撃セットアップが成立していて、かつ逃げ先として生存できるなら優先する。
    solrock_bi = _solrock_bench_index(p)
    solrock_plan_ok = (
        solrock_bi is not None
        and _field_has_pokemon(p, _is_lunatone)
        and _has_draw_support_in_hand_for_solrock(p)
        and p.bench[solrock_bi].hp > opp_max_effective
    )

    opp_remaining_prizes = len(opp.prize_pile) if hasattr(opp, "prize_pile") else PRIZE_COUNT

    from .deck_strategies import is_dragapult_deck_for_player as _is_drapa_koed
    _is_drapa_koed_deck = _is_drapa_koed(state, state.current_player)
    if solrock_plan_ok:
        # 条件を満たしているなら、逃げ先はソルロック固定（優先順位の遵守）
        best_idx = solrock_bi
    else:
        best_idx = None
        best_score = -1.0
        for i, bp in enumerate(p.bench):
            dmg = _max_effective_damage_for_attacker(state, bp, opp.active, state.current_player) if opp.active else 0
            can_ko = int(opp.active is not None and dmg >= opp.active.hp)
            raw = can_ko * _RETREAT_C_KO + dmg * _RETREAT_C_DMG + bp.hp
            # サイド管理: 相手の残りサイドが少ないとき、高サイドポケモンを前に出すペナルティ
            raw += _prize_management_penalty(bp, opp_remaining_prizes)
            # ドラパルトexデッキ: 前に出すポケモンの優先度調整
            if _is_drapa_koed_deck:
                bp_name = (getattr(bp.card, "name", "") or "").strip()
                if bp_name in ("キチキギスex", "ニャースex"):
                    raw -= 50000  # exサポート → 絶対出さない（サイド2献上）
                elif bp_name == "ドラメシヤ":
                    raw -= 30000  # 進化の基盤 → 出さない（守りたい）
                elif bp_name in ("サマヨール", "ヨノワール", "マシマシラ"):
                    raw -= 20000  # サポート系
                elif bp_name == "ドロンチ":
                    raw -= 10000  # HP低くてKOされやすい → 前に出さない
                elif bp_name == "ヨマワル":
                    raw += 5000  # 壁として適任
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
    """
    バトル場のポケモンでは相手を倒せないが、ベンチのポケモンなら倒せるときに
    自発的ににげる。にげるコストが払える（エネが足りる or ふうせんでコスト0）場合に発動。
    例: ソルロック（エネ1、にげ1）→メガルカリオex（エネ2）ではどうづき。
    """
    if getattr(state, "retreat_used_this_turn", False):
        return False
    if state.turn_count == 0:
        return False
    p = state.active_player_state()
    opp = state.defending_player_state()
    if not p.active or not p.bench or not opp.active or opp.active.hp is None or opp.active.hp <= 0:
        return False
    if getattr(p.active, "special_state", None) in ("sleep", "paralysis"):
        return False
    raw_rc = getattr(p.active.card, "retreat_cost", 1)
    tool = getattr(p.active, "attached_tool", None)
    eff_retreat = max(0, raw_rc - (2 if tool and (getattr(tool, "id", "") or "") == "fuusen" else 0))
    # にげるコストが払えないなら不可
    if p.active.attached_energy < eff_retreat:
        return False
    # バトル場で攻撃できるなら不要
    legal_active = get_legal_attack_indices(state, p, opp)
    our_dmg = _our_max_effective_damage(state)
    if legal_active and our_dmg >= opp.active.hp:
        return False
    # ベンチに攻撃可能なポケモンがいるか（KOできなくても攻撃できれば交代する価値あり）
    best_idx = None
    best_score = -1
    from .deck_strategies import is_dragapult_deck_for_player as _is_drapa_vol
    _is_drapa_vol_deck = _is_drapa_vol(state, state.current_player)
    for i, bp in enumerate(p.bench):
        if not get_legal_attack_indices_for_attacker(state, p, opp, bp):
            continue
        dmg = _max_effective_damage_for_attacker(state, bp, opp.active, state.current_player)
        # KO可能なら最優先、そうでなくてもバトル場が攻撃不能なら交代
        ko_bonus = 1000000 if dmg >= opp.active.hp else 0
        score = ko_bonus + dmg + bp.hp
        # ドラパルトexデッキ: 前に出すポケモンの優先度調整
        if _is_drapa_vol_deck:
            bp_name = (getattr(bp.card, "name", "") or "").strip()
            if bp_name in ("キチキギスex", "ニャースex"):
                score -= 50000
            elif bp_name == "ドラメシヤ":
                score -= 30000  # 進化の基盤 → 前に出さない
            elif bp_name in ("サマヨール", "ヨノワール", "マシマシラ"):
                score -= 40000
            elif bp_name == "スボミー":
                score += 5000  # 壁+グッズロック → 前に出す候補として優先
            elif bp_name == "ヨマワル":
                score += 3000  # 壁として適任
        if score > best_score:
            best_score = score
            best_idx = i
    # バトル場が攻撃できないなら、ベンチに攻撃可能なポケモンがいれば交代
    # バトル場が攻撃できる場合はKO可能なときだけ交代
    # ドラパルトexデッキ: サポートポケモンがバトル場→常に逃げる（攻撃が弱い/無意味）
    if best_idx is None:
        return False
    _active_is_support = False
    if _is_drapa_vol_deck:
        _an = (getattr(p.active.card, "name", "") or "").strip()
        if _an in ("ヨマワル", "サマヨール", "マシマシラ"):
            _active_is_support = True
    if legal_active and best_score < 1000000 and not _active_is_support:
        return False  # バトル場で攻撃できるのにKOできない交代はしない
    rc = getattr(p.bench[best_idx].card, "id", None) or getattr(p.bench[best_idx].card, "name", "")
    _log_choice(state, "retreat", card_id=rc)
    return retreat(state, best_idx)


def _try_swap_to_attacker(state: GameState) -> bool:
    """
    バトル場が攻撃できない＋ベンチに攻撃可能なポケモンがいるとき、
    いれかえグッズで交代する。にげるコストが払えない場合でも有効。
    """
    p = state.active_player_state()
    opp = state.defending_player_state()
    if not p.active or not p.bench or not opp.active:
        return False
    if state.turn_count == 0:
        return False
    # バトル場で攻撃できるなら不要
    legal_now = get_legal_attack_indices(state, p, opp)
    if legal_now:
        return False
    # いれかえグッズが手札にあるか
    swap_indices = [
        i for i, c in enumerate(p.hand)
        if is_goods(c) and (getattr(c, "effect", None) == "swap_active"
                           or getattr(c, "id", "") in ("pokemon_irekae", "pokemonirekae"))
    ]
    if not swap_indices:
        return False
    # ベンチに攻撃可能なポケモンがいるか
    best_bi = None
    best_dmg = 0
    for bi, bp in enumerate(p.bench):
        if get_legal_attack_indices_for_attacker(state, p, opp, bp):
            dmg = _max_effective_damage_for_attacker(state, bp, opp.active, state.current_player)
            if dmg > best_dmg:
                best_dmg = dmg
                best_bi = bi
    if best_bi is None:
        return False
    if use_pokemon_swap(state, swap_indices[0], best_bi):
        return True
    return False


def _can_win_now_check(state: GameState) -> bool:
    """今すぐ攻撃して勝てるか（サイド取り切り or 種切れ）"""
    p = state.active_player_state()
    opp = state.defending_player_state()
    if not p.active or not opp.active or not opp.active.hp or opp.active.hp <= 0:
        return False
    if getattr(p.active, "special_state", None) in ("sleep", "paralysis"):
        return False
    legal = get_legal_attack_indices(state, p, opp)
    if not legal:
        return False
    dmg = _our_max_effective_damage(state)
    if dmg >= opp.active.hp:
        prize_val = _prizes_for_ko(opp.active)
        if len(p.prize_pile) <= prize_val or not opp.bench:
            return True
    return False


def _do_attack_phase(state: GameState, p: PlayerState, opp: PlayerState) -> bool:
    """攻撃可能なら技を選んで攻撃（または生贄にげる）。実行したら True。"""
    is_game_first_turn = state.turn_count == 0
    if is_game_first_turn or not p.active or getattr(p.active, "special_state", None) in ("sleep", "paralysis"):
        return False
    legal_idxs = get_legal_attack_indices(state, p, opp)
    if not legal_idxs:
        return False

    # なぐってかくれる等でダメージ無効の相手には攻撃しても0ダメージ。
    # ボスの指令やどすこいキャッチャーでベンチを引き出せるなら、そちらを試す。
    # どちらもできなければ攻撃をスキップ（0ダメージを与えるだけ無駄）。
    # なぐってかくれる等でダメージ無効の相手には攻撃しても0ダメージ。
    # ボスの指令が手札にあればメインループに戻してサポート使用に任せる。
    # ベンチ引き出し手段がなければ0ダメージでも攻撃する（副効果があるかもしれない）。
    opp_protected_pre = getattr(opp.active, "protected_next_opponent_turn", False) if opp.active else False
    if opp_protected_pre and opp.bench:
        _switched_protected = False
        # 1. ボスの指令が手札にあり未使用なら使う
        if not state.support_used_this_turn:
            for _hi, _hc in enumerate(p.hand):
                if (getattr(_hc, "id", "") or "") == "bosunoshirei":
                    if use_support(state, _hi):
                        p = state.active_player_state()
                        opp = state.defending_player_state()
                        legal_idxs = get_legal_attack_indices(state, p, opp)
                        if not legal_idxs:
                            return True
                        _switched_protected = True
                    break
        # 2. どすこいキャッチャー（ハリテヤマ進化）でベンチを引き出す
        if not _switched_protected:
            from .evolution import evolve_pokemon, _can_evolve_onto
            for _hi, _hc in enumerate(p.hand):
                if not is_pokemon(_hc):
                    continue
                if (getattr(_hc, "ability_name", None) or "").strip() != "どすこいキャッチャー":
                    continue
                evolved = False
                for _bi, _bp in enumerate(p.bench):
                    if _can_evolve_onto(_bp.card, _hc) and not getattr(_bp, "put_on_bench_this_turn", False):
                        if evolve_pokemon(state, _hi, bench_index=_bi):
                            evolved = True
                            break
                if evolved:
                    p = state.active_player_state()
                    opp = state.defending_player_state()
                    legal_idxs = get_legal_attack_indices(state, p, opp)
                    if not legal_idxs:
                        return True
                    _switched_protected = True
                break
        # ベンチ引き出し手段なし → 0ダメージだが攻撃は実行する（メインループ停止防止）

    # Pattern 2: リオルの無駄パンチ回避
    # リオルがバトル場で最大ダメージ≤20、かつ相手に次ターンKOされる場合、
    # ベンチにより強いポケモンがいるなら退避する（ふうせん・いれかえ経由）。
    if (
        p.active
        and (getattr(p.active.card, "name", "") or "").strip() == "リオル"
        and p.bench
        and opp.active and opp.active.hp is not None and opp.active.hp > 0
    ):
        _riolu_our_dmg = _our_max_effective_damage(state)
        _riolu_opp_dmg = _opponent_max_effective_damage(state)
        _riolu_will_be_koed = _riolu_opp_dmg > 0 and p.active.hp <= _riolu_opp_dmg
        if _riolu_our_dmg <= 20 and _riolu_will_be_koed:
            # ベンチにもっとダメージを出せるポケモンがいるか
            _riolu_best_bench_dmg = 0
            _riolu_best_bi = None
            for _rbi, _rbp in enumerate(p.bench):
                _rdmg = _max_effective_damage_for_attacker(state, _rbp, opp.active, state.current_player)
                if _rdmg > _riolu_best_bench_dmg:
                    _riolu_best_bench_dmg = _rdmg
                    _riolu_best_bi = _rbi
            if _riolu_best_bi is not None and _riolu_best_bench_dmg > _riolu_our_dmg:
                _riolu_retreated = False
                # にげるコスト確認
                if not getattr(state, "retreat_used_this_turn", False):
                    _rrc = getattr(p.active.card, "retreat_cost", 1)
                    _rtool = getattr(p.active, "attached_tool", None)
                    _reff_rc = max(0, _rrc - (2 if _rtool and (getattr(_rtool, "id", "") or "") == "fuusen" else 0))
                    if p.active.attached_energy >= _reff_rc:
                        if retreat(state, _riolu_best_bi):
                            _riolu_retreated = True
                            p = state.active_player_state()
                            opp = state.defending_player_state()
                            legal_idxs = get_legal_attack_indices(state, p, opp)
                            if not legal_idxs:
                                return True
                            state._record_frame()
                # いれかえグッズで退避
                if not _riolu_retreated:
                    _rswap_indices = [
                        _ri for _ri, _rc in enumerate(p.hand)
                        if is_goods(_rc) and (getattr(_rc, "effect", None) == "swap_active"
                                              or getattr(_rc, "id", "") in ("pokemon_irekae", "pokemonirekae"))
                    ]
                    if _rswap_indices:
                        if use_pokemon_swap(state, _rswap_indices[0], _riolu_best_bi):
                            p = state.active_player_state()
                            opp = state.defending_player_state()
                            legal_idxs = get_legal_attack_indices(state, p, opp)
                            if not legal_idxs:
                                return True
                            state._record_frame()

    # 攻撃直前KOチェック: ベンチのアタッカーで相手をKOできるなら逃げて交代。
    # バトル場では倒せないが、ベンチのアタッカーなら倒せるケース。
    if (
        opp.active and opp.active.hp is not None and opp.active.hp > 0
        and not getattr(state, "retreat_used_this_turn", False)
        and getattr(p.active, "special_state", None) not in ("sleep", "paralysis")
    ):
        our_dmg_now = _our_max_effective_damage(state)
        if our_dmg_now < opp.active.hp:
            raw_rc = getattr(p.active.card, "retreat_cost", 1)
            tool = getattr(p.active, "attached_tool", None)
            eff_rc = max(0, raw_rc - (2 if tool and (getattr(tool, "id", "") or "") == "fuusen" else 0))
            if p.active.attached_energy >= eff_rc:
                for bi, bp in enumerate(p.bench):
                    bench_dmg = _max_effective_damage_for_attacker(state, bp, opp.active, state.current_player)
                    if bench_dmg >= opp.active.hp and get_legal_attack_indices_for_attacker(state, p, opp, bp):
                        if retreat(state, bi):
                            p = state.active_player_state()
                            opp = state.defending_player_state()
                            legal_idxs = get_legal_attack_indices(state, p, opp)
                            if not legal_idxs:
                                return True
                            state._record_frame()
                        break

    # 攻撃直前のにげポリシー（choice_log type: retreat_before_attack）。
    # run_turn_auto のメインループ「直後」ではなく、ここで呼ぶ（合法攻撃が確定した経路と必ず一致させる）。
    if try_retreat_before_attack_policy(state):
        p = state.active_player_state()
        opp = state.defending_player_state()
        legal_idxs = get_legal_attack_indices(state, p, opp)
        if not legal_idxs:
            return True
        state._record_frame()

    ro_attack = rules_only_for_player(state)
    # まずルール側でのベスト（minimax または重みベース）を 1 つ決める。ルールのみ手番は minimax しない。
    best_idx_rule = None
    if not ro_attack and getattr(state, "use_attack_minimax", True):
        best_idx_rule = _choose_best_attack_index_minimax(state, p, opp)
    if best_idx_rule is None:
        best_idx_rule = _choose_best_attack_index(state, p, opp)
    if best_idx_rule is None or best_idx_rule not in legal_idxs:
        best_idx_rule = legal_idxs[0]

    # attack policy があれば、rule + λ*log(pi) で候補を再スコアリングする（ルールのみ手番では使わない）。
    by_player_pi = getattr(state, "pi_attack_model_path_by_player", [None, None])
    pi_path = None if ro_attack else (
        by_player_pi[state.current_player]
        if state.current_player < len(by_player_pi) and by_player_pi[state.current_player] is not None
        else getattr(state, "pi_attack_model_path", None)
    )

    def _rule_score_for_attack(idx: int) -> float:
        atk = p.active.card.attacks[idx]
        base = _attack_damage_for_eval(atk)
        ign = getattr(atk, "damage_ignores_weakness_resistance", False)
        eff = (
            _effective_damage_to_defender(
                p.active.card, opp.active, base, state=state, attacker_bp=p.active, ignore_weakness_resistance=ign
            )
            if opp.active
            else base
        )
        return float(eff)

    best_idx = best_idx_rule

    # minimax が低ダメージ技を選んだ場合、ルールベース（高ダメージ技）と比較。
    # デメリット（自傷、次ターン使用不可等）がない技なら高ダメージを常に優先。
    if best_idx is not None and opp.active:
        best_rule = _choose_best_attack_index(state, p, opp)
        if best_rule is not None and best_rule != best_idx:
            mm_atk = p.active.card.attacks[best_idx]
            rule_atk = p.active.card.attacks[best_rule]
            mm_eff = _rule_score_for_attack(best_idx)
            rule_eff = _rule_score_for_attack(best_rule)
            # ルールベースの方がダメージが高い＋デメリットなし → 常に優先
            _rule_has_demerit = (
                getattr(rule_atk, "self_damage", 0) > 0  # 自傷あり
                or "使えない" in (getattr(rule_atk, "description", "") or "")  # 次ターン使用不可
            )
            if rule_eff > mm_eff and not _rule_has_demerit:
                best_idx = best_rule

    # value が指定されていれば value 優先で候補を選ぶ（eval_after を教師にしている想定）。
    by_player_value = getattr(state, "value_attack_model_path_by_player", [None, None])
    value_path = None if ro_attack else (
        by_player_value[state.current_player]
        if state.current_player < len(by_player_value) and by_player_value[state.current_player] is not None
        else getattr(state, "value_attack_model_path", None)
    )
    if value_path:
        value = load_value_model_pt(value_path)
        state_vec = encode_state_basic(state, state.current_player)
        alpha = float(getattr(state, "value_attack_lambda", 0.1))
        alpha = max(0.0, min(1.0, alpha))

        # rule_s と value を min-max 正規化してから混ぜる（スケール差で noise が効きすぎるのを防ぐ）
        rule_scores: list[float] = []
        value_preds: list[float] = []
        for idx in legal_idxs:
            rule_scores.append(_rule_score_for_attack(idx))
            if 0 <= idx < value.action_dim:
                value_preds.append(float(value.predict_one(state_vec, idx)))
            else:
                value_preds.append(0.0)

        if value_preds:
            rmin = min(rule_scores)
            rmax = max(rule_scores)
            vmin = min(value_preds)
            vmax = max(value_preds)
            r_range = max(1e-6, rmax - rmin)
            v_range = max(1e-6, vmax - vmin)

            # value がほぼ一様ならルールに戻す
            if (vmax - vmin) >= 0.05:
                best_score = float("-inf")
                for idx, rs, vp in zip(legal_idxs, rule_scores, value_preds):
                    rule_norm = (rs - rmin) / r_range
                    value_norm = (vp - vmin) / v_range
                    score = (1.0 - alpha) * rule_norm + alpha * value_norm
                    if score > best_score:
                        best_score = score
                        best_idx = idx
    elif pi_path:
        pi = load_pi_model_pt(pi_path)
        state_vec = encode_state_basic(state, state.current_player)
        probs = pi.predict_probs_one(state_vec)
        lambda_pi = getattr(state, "pi_attack_lambda", 0.1)
        eps = 1e-6
        max_p = max(probs) if probs else 0.0
        # policy がほぼ一様（自信なし）のときはルールに任せる。
        if max_p >= 0.4:
            best_score = float("-inf")
            for idx in legal_idxs:
                rule_s = _rule_score_for_attack(idx)
                p_a = float(probs[idx]) if 0 <= idx < len(probs) else 0.0
                score = rule_s + lambda_pi * float(math.log(p_a + eps))
                if score > best_score:
                    best_score = score
                    best_idx = idx

    # KO強制: 倒せる技があるなら確実に倒す（minimax/value/piの結果より優先）。
    # ダメージ無効状態（なぐってかくれる等）の相手にはKO強制しない。
    opp_protected = getattr(opp.active, "protected_next_opponent_turn", False) if opp.active else False
    if opp.active and opp.active.hp is not None and opp.active.hp > 0 and not opp_protected:
        ko_candidates = []
        for idx in legal_idxs:
            atk_check = p.active.card.attacks[idx]
            base_check = _attack_damage_for_eval(atk_check)
            ign_check = getattr(atk_check, "damage_ignores_weakness_resistance", False)
            eff_check = (
                _effective_damage_to_defender(
                    p.active.card, opp.active, base_check, state=state, attacker_bp=p.active, ignore_weakness_resistance=ign_check
                )
                if opp.active
                else base_check
            )
            if eff_check >= opp.active.hp:
                ko_candidates.append((idx, eff_check))
        if ko_candidates:
            # 自傷きぜつで自分も倒れる技を除外（サイド取り切り勝ちの場合のみ許可）
            safe_ko = []
            for idx_ko, eff_ko in ko_candidates:
                atk_ko = p.active.card.attacks[idx_ko]
                self_dmg = getattr(atk_ko, "self_damage", 0)
                would_self_ko = self_dmg > 0 and p.active.hp <= self_dmg
                if would_self_ko:
                    # 自傷きぜつ → サイド取り切りで勝てるなら許可
                    opp_prize_val = _prizes_for_ko(opp.active)
                    if len(p.prize_pile) <= opp_prize_val:
                        safe_ko.append((idx_ko, eff_ko))
                    # それ以外は除外（ベンチなしなら種切れ負け）
                else:
                    safe_ko.append((idx_ko, eff_ko))
            if safe_ko:
                # KO確定技が複数ある場合、追加効果がある技を優先（ファントムダイブのベンチダメカン等）
                # デメリットなし＋ダメージが高い技 ＝ 追加効果がある可能性が高い
                best_idx = max(safe_ko, key=lambda x: x[1])[0]

    if best_idx is None:
        return False
    atk = p.active.card.attacks[best_idx]
    ac_id = getattr(p.active.card, "id", None) or getattr(p.active.card, "name", "")
    base_dmg = _attack_damage_for_eval(atk)
    ign = getattr(atk, "damage_ignores_weakness_resistance", False)
    effective_dmg = (
        _effective_damage_to_defender(
            p.active.card, opp.active, base_dmg, state=state, attacker_bp=p.active, ignore_weakness_resistance=ign
        )
        if opp.active
        else base_dmg
    )
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
    # この技で相手 active をきぜつさせられるか（選択キー用）。
    can_kill = bool(opp.active and opp.active.hp is not None and effective_dmg >= opp.active.hp)
    attack(state, best_idx)
    # choice_log の eval は「行動直後」を教師にしたいので、attack() 実行後に記録する。
    _log_choice(state, "attack", card_id=ac_id, attack_name=atk.name, can_kill=can_kill)
    return True


def _best_attack_score_after_attach(
    state: GameState,
    bench_idx_or_none: int | None,
    energy_hand_idx: int,
) -> float:
    """
    指定ターゲットにエネルギーを付与したクローンで、合法攻撃 top2 のミニマックス最大スコアを返す。
    合法攻撃がなければ evaluate_board を返す。
    """
    from .trainers import attach_energy as _attach_en

    clone = state.copy_for_simulation()
    if bench_idx_or_none is None:
        _attach_en(clone, energy_hand_idx)
    else:
        _attach_en(clone, energy_hand_idx, bench_index=bench_idx_or_none)

    p_cl = clone.active_player_state()
    opp_cl = clone.defending_player_state()
    legal = get_legal_attack_indices(clone, p_cl, opp_cl)

    if not legal:
        return evaluate_board(clone, clone.current_player)

    me = clone.current_player
    weights = clone.get_weights_for_player(me)
    best_score = _MINIMAX_LOSE_SCORE - 1.0

    for idx in legal[:2]:
        clone2 = clone.copy_for_simulation()
        attack(clone2, idx)
        score = _minimax_score_after_attack(clone2, me)
        atk_obj = p_cl.active.card.attacks[idx]
        score += _ATTACK_WEIGHT_SCALE_IN_MINIMAX * get_attack_weight(weights, p_cl.active.card, atk_obj)
        if score > best_score:
            best_score = score

    return best_score


def _try_joint_q_decision(state: GameState) -> bool:
    """
    Joint Q-model で (support, energy) を同時に決定して実行する。
    joint_q_model_path_by_player が設定されているプレイヤーのみ有効。
    成功時は True を返し、サポート使用 + エネルギー付与を両方完了する。
    """
    by_player = getattr(state, "joint_q_model_path_by_player", [None, None])
    q_path = by_player[state.current_player] if state.current_player < len(by_player) else None
    if not q_path:
        return False
    # サポートもエネルギーも両方未使用のときだけ joint 判断
    if state.support_used_this_turn or state.energy_attached_this_turn:
        return False
    if _is_first_player_first_turn(state):
        return False  # 先行1ターン目はサポート不可

    from .joint_policy import joint_policy_decision, ENERGY_ACTION_DIM
    from .support_policy import legal_support_mask
    from .trainers import use_support
    from .state import _log_choice

    p = state.active_player_state()

    # 合法エネルギーターゲット
    n_bench = len([b for b in (p.bench or []) if b is not None])
    legal_ene = list(range(min(1 + n_bench, ENERGY_ACTION_DIM)))

    sup_id, ene_id, dbg = joint_policy_decision(state, legal_energy_targets=legal_ene)

    acted = False

    # --- サポート実行 ---
    if sup_id > 0 and sup_id <= len(KNOWN_SUPPORT_IDS):
        target_sid = KNOWN_SUPPORT_IDS[sup_id - 1]
        p = state.active_player_state()
        for hand_i, c in enumerate(p.hand or []):
            if is_support(c) and (getattr(c, "id", "") or "") == target_sid:
                if use_support(state, hand_i):
                    acted = True
                    p = state.active_player_state()
                    state._record_frame()
                break

    # --- エネルギー実行 ---
    if not state.energy_attached_this_turn:
        from .trainers import attach_energy
        p = state.active_player_state()
        energy_hand_idx = _pick_energy_hand_idx(p, state)
        if energy_hand_idx is not None:
            bidx = ene_id  # 0=active, 1..=bench
            if bidx == 0 and p.active:
                attach_energy(state, energy_hand_idx)  # bench_index=None → active
                acted = True
                state._record_frame()
            elif 0 < bidx <= len(p.bench) and p.bench[bidx - 1] is not None:
                attach_energy(state, energy_hand_idx, bench_index=bidx - 1)
                acted = True
                state._record_frame()
            else:
                # フォールバック: 通常のエネルギー付与
                if _try_attach_energy_auto(state):
                    acted = True
                    state._record_frame()

    return acted


def _try_attach_energy_with_attack_lookahead(state: GameState) -> bool:
    """
    全候補 × 攻撃 top2 の先読みで付与先を選ぶ。
    ヒューリスティックは候補生成（_build_energy_attach_input のフィルタリング）のみに使用し、
    最終決定は lookahead スコア（attach 後の attack minimax 評価）で行う。
    use_energy_attack_lookahead=False、rules_only、または候補が 1 件以下の場合は通常ロジックにフォールバック。
    """
    from .energy_policy import (
        energy_attach_policy_decision,
        bench_index_from_action_id,
        _make_energy_log_extras,
        energy_attach_action_id,
        heuristic_logits_energy_attach,
    )
    from .trainers import attach_energy as _attach_en

    # ドラパルトexデッキ: ファントムダイブ完成即時パス（最優先）
    from .deck_strategies import is_dragapult_deck_for_player as _is_drapa_la
    if _is_drapa_la(state, state.current_player) and not state.energy_attached_this_turn:
        p_la = state.active_player_state()
        _all_bp_la = ([p_la.active] if p_la.active else []) + list(p_la.bench or [])
        for _bp_la in _all_bp_la:
            if (getattr(_bp_la.card, "name", "") or "").strip() != "ドラパルトex":
                continue
            _types_la = list(getattr(_bp_la, "attached_energy_types", []) or [])
            _need_la = None
            if "fire" in _types_la and "psychic" not in _types_la:
                _need_la = "psychic"
            elif "psychic" in _types_la and "fire" not in _types_la:
                _need_la = "fire"
            if _need_la:
                from card import is_energy as _is_e_la
                for _ei_la, _ec_la in enumerate(p_la.hand):
                    if _is_e_la(_ec_la) and getattr(_ec_la, "energy_type", None) == _need_la:
                        _bi_la = None
                        if _bp_la is not p_la.active:
                            for _bi2, _bbp in enumerate(p_la.bench):
                                if _bbp is _bp_la:
                                    _bi_la = _bi2
                                    break
                        if _bi_la is not None:
                            _attach_en(state, _ei_la, bench_index=_bi_la)
                        else:
                            _attach_en(state, _ei_la)
                        return True

    if rules_only_for_player(state):
        return _try_attach_energy_auto(state)

    if not getattr(state, "use_energy_attack_lookahead", True):
        return _try_attach_energy_auto(state)

    p = state.active_player_state()
    can_evolve_this_turn = not _is_first_player_first_turn(state)

    # 進化用即時付与（policy を通さない特殊ケース）
    if can_evolve_this_turn and _should_attach_for_evolution(p):
        return _try_attach_energy_auto(state)

    # 次ターン進化→攻撃の準備付与（_try_attach_energy_auto 側と同じロジック）
    # バトル場またはベンチの進化元にエネを付けておく（進化できないターンのみ）
    if not can_evolve_this_turn and p.active and not state.energy_attached_this_turn:
        from .evolution import _can_evolve_onto as _evo_check
        energy_hand_idx = _pick_energy_hand_idx(p, state)
        if energy_hand_idx is not None:
            targets = [(None, p.active)] + [(bi, bp) for bi, bp in enumerate(p.bench)]
            for bench_idx, bp in targets:
                for c in p.hand:
                    if not (is_pokemon(c) and _evo_check(bp.card, c)):
                        continue
                    has_cheap_attack = any(a.energy_cost <= 1 for a in (c.attacks or []))
                    if has_cheap_attack and bp.attached_energy == 0:
                        if bench_idx is None:
                            _attach_en(state, energy_hand_idx)
                        else:
                            _attach_en(state, energy_hand_idx, bench_index=bench_idx)
                        return True
                    break

    result = _build_energy_attach_input(state)
    if result is None:
        return False
    energy_hand_idx, energy_card, candidates = result

    # ヒューリスティックはログ・NN 混合用に計算するが、最終決定には使わない
    action_id_policy, debug = energy_attach_policy_decision(state, candidates, energy_card)

    # 全候補の action_id（_build_energy_attach_input が生成したもの = 候補生成の結果）
    all_action_ids = [energy_attach_action_id(t) for t, _ in candidates]

    lookahead_used = False
    best_action_id = action_id_policy  # 候補 1 件のときのフォールバック
    best_lookahead_score: float | None = None

    if len(all_action_ids) >= 2:
        # lookahead + Q-value（z-score 正規化済み）の合算スコアで最善を選ぶ
        # q_combine_w: 正規化後 Q 値の重み（lookahead と同スケールで足し合わせ）
        q_combine_w = float(getattr(state, "q_lookahead_combine_weight", 3.0))
        q_vals_raw: list[float] | None = debug.get("q_vals_raw")

        # 6 action 分の Q 値を z-score 正規化（mean/std を action 間で計算）
        if q_vals_raw and len(q_vals_raw) >= 2:
            q_mean = sum(q_vals_raw) / len(q_vals_raw)
            q_var = sum((v - q_mean) ** 2 for v in q_vals_raw) / len(q_vals_raw)
            q_std = q_var ** 0.5
            q_normed = [(v - q_mean) / (q_std + 1e-8) for v in q_vals_raw]
        else:
            q_normed = None

        # ① Q で上位 k 件に絞る（Q モデルがある場合のみ）→ ノイズ行動を排除
        top_k = int(getattr(state, "q_lookahead_top_k", 3))  # 0 = フィルタなし、3 = Q 上位 3 件
        if q_normed and top_k > 0 and len(all_action_ids) > top_k:
            lookahead_candidates = sorted(
                all_action_ids,
                key=lambda a: q_normed[a] if 0 <= a < len(q_normed) else float("-inf"),
                reverse=True,
            )[:top_k]
        else:
            lookahead_candidates = all_action_ids

        # ② 絞り込んだ候補だけ lookahead + Q + ヒューリスティック合算スコアで最善を選ぶ
        h_logits = heuristic_logits_energy_attach(state, candidates, energy_card)
        _H_SCALE_IN_LOOKAHEAD = 0.5  # ヒューリスティックの重み（lookaheadより控えめ）
        best_la_score = float("-inf")
        for aid in lookahead_candidates:
            bidx = bench_index_from_action_id(aid)
            la_score = _best_attack_score_after_attach(state, bidx, energy_hand_idx)
            q_bonus = (q_normed[aid] * q_combine_w) if (q_normed and 0 <= aid < len(q_normed)) else 0.0
            h_bonus = (h_logits[aid] * _H_SCALE_IN_LOOKAHEAD) if 0 <= aid < len(h_logits) else 0.0
            score = la_score + q_bonus + h_bonus
            if score > best_la_score:
                best_la_score = score
                best_action_id = aid
        lookahead_used = best_action_id != action_id_policy
        best_lookahead_score = best_la_score

    bench_index = bench_index_from_action_id(best_action_id)
    cdba: dict[int, float] = {energy_attach_action_id(t): float(d) for t, d in candidates}
    log_extras = _make_energy_log_extras(
        state, best_action_id, debug, energy_card,
        target_bench_index=bench_index,
        candidates_dmg_by_action=cdba,
        lookahead_used=lookahead_used,
        lookahead_score=best_lookahead_score,
    )

    target_card = p.active.card if bench_index is None else p.bench[bench_index].card
    card_id = getattr(target_card, "id", None) or getattr(target_card, "name", "")
    if bench_index is None:
        _attach_en(state, energy_hand_idx)
    else:
        _attach_en(state, energy_hand_idx, bench_index=bench_index)
    _log_choice(state, "energy_attach", card_id=card_id, **log_extras)
    return True


def run_turn_auto(state: GameState) -> bool:
    """
    現在のプレイヤーが「可能な行動を順番に実行」する。
    順序：0. ベンチにポケモン出す、1. 進化、2. スタジアム（1 枚まで）、3. 手札を捨てないサポート（キハダ・ネモ）、4. エレキジェネ、5. ボール系グッズ、6. 進化（ボール後）、7. エネルギー付与（優先度低め・捨て札刷新サポートより前）、8. 手札刷新前グッズ／サポート、9. いれかえ、10. にげる（次相手ターンきぜつ確定かつにげで助かる等は `_try_retreat_when_koed`。「攻撃不可だけ」の無目的交代はしない）、11. どうぐ・グッズ、12. 進化（再）、13. 攻撃。
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
            # ニャースex: おくのてキャッチ（ベンチに出したとき発動）
            # サポート使用済みでも次ターン用にサーチする価値はある
            if p.bench:
                last_bp = p.bench[-1]
                if (getattr(last_bp.card, "name", "") or "").strip() == "ニャースex":
                    if getattr(last_bp, "put_on_bench_this_turn", False):
                        _try_use_ability_okunote_catch(state)
                        p = state.active_player_state()
            state._record_frame()

    # 種切れ即勝ち: 相手ベンチなしでKO可能なら余計なアクションを全てスキップ
    if not _is_first_player_first_turn(state) and state.turn_count > 0:
        _opp_sw = state.defending_player_state()
        if _opp_sw.active and not _opp_sw.bench and _opp_sw.active.hp and _opp_sw.active.hp > 0:
            # 今すぐKOできるか
            if _can_win_now_check(state):
                p = state.active_player_state()
                opp = state.defending_player_state()
                if _do_attack_phase(state, p, opp):
                    return True
            # ファイトゴングでエネ取得→付与→KO��きるか
            _p_sw = state.active_player_state()
            if not state.energy_attached_this_turn and any(is_energy(c) for c in _p_sw.hand) is False:
                _fg_idx = next((i for i, c in enumerate(_p_sw.hand) if getattr(c, "id", "") == "faitogongu"), None)
                if _fg_idx is not None:
                    # ファイトゴングで闘エネまたは基本闘エネを取れるか
                    _has_energy_in_deck = any(
                        is_energy(c) and getattr(c, "energy_type", None) == "fighting"
                        for c in _p_sw.deck
                    )
                    if _has_energy_in_deck:
                        use_trainer_goods(state, _fg_idx)
                        acted = True
                        p = state.active_player_state()
                        state._record_frame()
                        # エネ付与
                        from .turn_energy import _pick_energy_hand_idx
                        eidx = _pick_energy_hand_idx(p, state)
                        if eidx is not None:
                            from .trainers import attach_energy as _sw_attach
                            _sw_attach(state, eidx)
                            p = state.active_player_state()
                            state._record_frame()
                        if _can_win_now_check(state):
                            opp = state.defending_player_state()
                            if _do_attack_phase(state, p, opp):
                                return True

    # KOプラン探索: 確定行動の組み合わせでこのターンKOできる手順があるか
    from .turn_planner import find_ko_plan
    _ko_plan = None
    if not _is_first_player_first_turn(state) and state.turn_count > 0:
        # バトル場から直接KOできる場合はプランナー不要（通常ロジックで倒せる）
        p_check = state.active_player_state()
        opp_check = state.defending_player_state()
        already_can_ko = False
        if p_check.active and opp_check.active and opp_check.active.hp and opp_check.active.hp > 0:
            already_can_ko = _our_max_effective_damage(state) >= opp_check.active.hp
        if not already_can_ko:
            # 軽量チェック: KOプランの可能性があるか（手札/場にアタッカー候補がいるか）
            _has_attacker_potential = False
            _atk_names = {"ソルロック", "メガルカリオex", "メガルカリオ", "ルカリオ", "ハリテヤマ",
                          "ドラパルトex", "ドロンチ", "ドラメシヤ"}
            # 場にアタッカーがいるか
            if p_check.active and getattr(p_check.active.card, "name", "") in _atk_names:
                _has_attacker_potential = True
            for bp in p_check.bench:
                if getattr(bp.card, "name", "") in _atk_names:
                    _has_attacker_potential = True
                    break
            # 手札に進化先があるか
            if not _has_attacker_potential:
                for c in p_check.hand:
                    if is_pokemon(c) and getattr(c, "name", "") in _atk_names:
                        _has_attacker_potential = True
                        break
            if _has_attacker_potential:
                _ko_plan = find_ko_plan(state)
        # KOプランにエネ付与が含まれる場合、ルナサイクルで先にドローした方が得なケースをチェック。
        # サポートなし + エネ1枚 + ファイトゴングorルナサイクルでドロー可能 → KOプランを後回し。
        # ルナサイクル後もKOは取れる（ドローでエネを引ける+手張り）。
        if _ko_plan is not None:
            _ko_has_energy_step = any(s.action == "energy" for s in _ko_plan.steps)
            if _ko_has_energy_step:
                _p_ko = state.active_player_state()
                _ko_energy_count = sum(1 for c in _p_ko.hand if is_energy(c))
                _ko_has_support = any(is_support(c) for c in _p_ko.hand)
                _ko_luna_on_field = _field_has_pokemon(_p_ko, _is_lunatone) and _field_has_pokemon(_p_ko, _is_solrock)
                _ko_luna_via_gong = (
                    not _ko_luna_on_field
                    and _field_has_pokemon(_p_ko, _is_solrock)
                    and any(getattr(c, "id", "") == "faitogongu" for c in _p_ko.hand)
                    and any(_is_lunatone(dc) for dc in _p_ko.deck if is_pokemon(dc))
                )
                _ko_has_tanka = any(getattr(c, "id", "") == "yorunotanka" for c in _p_ko.hand)
                _ko_should_luna_first = (
                    (_ko_luna_on_field or _ko_luna_via_gong)
                    and getattr(state, "ability_declared_this_turn", None) != "ルナサイクル"
                    and (
                        # サポートなし+エネ1枚 → ルナサイクルでドロー優先
                        (not _ko_has_support and _ko_energy_count <= 1)
                        # タンカがあればエネ回収できるのでルナサイクル先でもKO可能
                        or _ko_has_tanka
                    )
                )
                # 種切れ勝ちの場合は遅延しない（即KOで勝てる）
                _opp_ko = state.defending_player_state()
                _opp_no_bench = not _opp_ko.bench
                if _ko_should_luna_first and not _opp_no_bench:
                    _ko_plan = None  # ルナサイクル優先 → KOプランは通常ロジックで後から実行される

        if _ko_plan is not None:
            # KO可能な手順を実行
            from .trainers import attach_energy as _plan_attach
            from .evolution import evolve_pokemon
            state._ko_plan_executing = True
            for step in _ko_plan.steps:
                p_plan = state.active_player_state()
                if step.action == "energy":
                    eidx = next((i for i, c in enumerate(p_plan.hand) if is_energy(c)), None)
                    if eidx is not None:
                        if step.target.startswith("active"):
                            _plan_attach(state, eidx)
                        else:
                            # bench:名前 → ベンチインデックスを探す
                            target_name = step.target.split(":")[-1]
                            bi = next((i for i, bp in enumerate(p_plan.bench) if bp.card.name == target_name), None)
                            if bi is not None:
                                _plan_attach(state, eidx, bench_index=bi)
                            else:
                                _plan_attach(state, eidx)
                elif step.action == "evolve":
                    # bench:旧名→新名 or active→新名
                    if "→" in step.target:
                        evo_name = step.target.split("→")[-1]
                        hi = next((i for i, c in enumerate(p_plan.hand) if is_pokemon(c) and getattr(c, "name", "") == evo_name), None)
                        if hi is not None:
                            if step.target.startswith("active"):
                                evolve_pokemon(state, hi)
                            else:
                                old_name = step.target.split(":")[1].split("→")[0]
                                bi = next((i for i, bp in enumerate(p_plan.bench) if bp.card.name == old_name), None)
                                if bi is not None:
                                    evolve_pokemon(state, hi, bench_index=bi)
                elif step.action == "retreat":
                    target_name = step.target.lstrip("→")
                    bi = next((i for i, bp in enumerate(p_plan.bench) if bp.card.name == target_name), None)
                    if bi is not None:
                        retreat(state, bi)
                elif step.action == "fight_gong":
                    fg_idx = next((i for i, c in enumerate(p_plan.hand) if getattr(c, "id", "") == "faitogongu"), None)
                    if fg_idx is not None:
                        use_trainer_goods(state, fg_idx)
                elif step.action == "bench":
                    # ベンチ出し（トップレベルインポート済みの_put_one_pokemon_on_benchを使う）
                    _put_one_pokemon_on_bench(p_plan, state, state.current_player)
                elif step.action == "tool":
                    # ふうせん等のどうぐ装着
                    from .trainers import attach_tool
                    tool_name = step.target.split("←")[-1]
                    ti = next((i for i, c in enumerate(p_plan.hand) if getattr(c, "name", "") == tool_name and getattr(c, "is_tool", False)), None)
                    if ti is not None:
                        if step.target.startswith("active"):
                            attach_tool(state, ti)
                        else:
                            bp_name = step.target.split(":")[1].split("←")[0]
                            bi = next((i for i, bp in enumerate(p_plan.bench) if bp.card.name == bp_name), None)
                            if bi is not None:
                                attach_tool(state, ti, bench_index=bi)
                elif step.action == "power_protein":
                    pp_idx = next((i for i, c in enumerate(p_plan.hand) if getattr(c, "id", "") == "pawaapurotein"), None)
                    if pp_idx is not None:
                        use_trainer_goods(state, pp_idx)
                elif step.action == "hyper_ball":
                    # プラン指定のポケモンを取るために、一時的に_find_pokemon_for_haipaboruを上書き
                    target_name = step.detail.get("target_name", "")
                    import game.trainers as _gt
                    _orig_find = _gt._find_pokemon_for_haipaboru
                    def _plan_find(p, _tn=target_name):
                        for i, c in enumerate(p.deck):
                            if is_pokemon(c) and getattr(c, "name", "") == _tn:
                                return (i, c)
                        return _orig_find(p)
                    _gt._find_pokemon_for_haipaboru = _plan_find
                    try:
                        hb_idx = next((i for i, c in enumerate(p_plan.hand) if getattr(c, "id", "") == "haipaboru"), None)
                        if hb_idx is not None:
                            use_trainer_goods(state, hb_idx)
                    finally:
                        _gt._find_pokemon_for_haipaboru = _orig_find
                state._record_frame()
            state._ko_plan_executing = False
            # KOプラン実行後、通常ターン処理にフォールスルー。
            # 攻撃前にサポート/グッズを使い切る（ゼイユでドロー等）。
            # 攻撃はメインループ末尾の_do_attack_phaseで実行される。
            p = state.active_player_state()

    # 勝ち確ならベンチ出しも不要 → 即攻撃フェーズへ
    _can_win_now = False
    if not _is_first_player_first_turn(state) and state.turn_count > 0:
        p_wc = state.active_player_state()
        opp_wc = state.defending_player_state()
        if (
            p_wc.active and opp_wc.active
            and opp_wc.active.hp is not None and opp_wc.active.hp > 0
            and getattr(p_wc.active, "special_state", None) not in ("sleep", "paralysis")
            and not getattr(opp_wc.active, "protected_next_opponent_turn", False)
        ):
            legal_wc = get_legal_attack_indices(state, p_wc, opp_wc)
            if legal_wc:
                dmg_wc = _our_max_effective_damage(state)
                if dmg_wc >= opp_wc.active.hp:
                    prize_wc = _prizes_for_ko(opp_wc.active)
                    if len(p_wc.prize_pile) <= prize_wc:
                        _can_win_now = True
                    # 相手ベンチなし → 倒せば種切れ勝ち
                    elif not opp_wc.bench:
                        _can_win_now = True

    # ていさつしれいはメインループ内（サポート使用後）で実行する
    # リーリエの決心→ていさつしれいの順が正しい（リーリエで手札を入れ替えてからていさつしれい）

    if not _can_win_now:
        _try_put_bench_until_full()

    # キチキギスex: さかてにとる（前の相手の番にきぜつしていれば 3 枚ドロー）
    if _try_use_ability_sakatenitori(state):
        acted = True
        p = state.active_player_state()
        state._record_frame()
        _try_put_bench_until_full()

    # マシマシラ: アドレナブレイン（自分のダメカンを相手に移す）
    if _try_use_ability_adrenabrain(state):
        acted = True
        p = state.active_player_state()
        opp = state.defending_player_state()
        state._record_frame()

    action_round = 0
    while action_round < MAX_TURN_ACTION_ROUNDS:
        action_round += 1

        # 勝ち確即攻撃: 今すぐ攻撃してサイドを取り切れるなら、余計なことをせず即breakして攻撃フェーズへ。
        if not _is_first_player_first_turn(state) and state.turn_count > 0:
            p_win = state.active_player_state()
            opp_win = state.defending_player_state()
            if (
                p_win.active and opp_win.active
                and opp_win.active.hp is not None and opp_win.active.hp > 0
                and getattr(p_win.active, "special_state", None) not in ("sleep", "paralysis")
            ):
                legal_win = get_legal_attack_indices(state, p_win, opp_win)
                if legal_win:
                    our_dmg_win = _our_max_effective_damage(state)
                    if our_dmg_win >= opp_win.active.hp:
                        opp_prize_val = _prizes_for_ko(opp_win.active)
                        our_remaining = len(p_win.prize_pile)
                        if our_remaining <= opp_prize_val or not opp_win.bench:
                            break  # 即攻撃フェーズへ��サイド取り切り or 種切れ勝ち）

        used_fushiginaame = False
        if state.turn_count >= 2:
            # ふしぎなアメの前にていさつしれいを使う（ドロンチ→ドラパルトex進化で失われるため）
            _has_ame_in_hand = any(
                is_goods(c) and (getattr(c, "id", "") == "fushiginaame" or getattr(c, "name", "") == "ふしぎなアメ")
                for c in p.hand
            )
            if _has_ame_in_hand:
                while _try_use_ability_teisatsushirei(state):
                    acted = True
                    p = state.active_player_state()
                    state._record_frame()
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
        # 進化前にていさつしれいを使い切る（ドロンチ→ドラパルトex進化で失われるため）
        # リーリエが手札にあっても、ドラパルトex進化が控えているならていさつしれいを先に使う
        if can_evolve:
            # ドラパルトex進化が可能か（手札にドラパルトex + 場にドロンチ）
            from .deck_strategies import is_dragapult_deck_for_player as _is_drapa_pre_evo
            _drapa_evo_pending = False
            if _is_drapa_pre_evo(state, state.current_player):
                _has_drapa_hand = any(
                    is_pokemon(c) and (getattr(c, "name", "") or "").strip() == "ドラパルトex"
                    for c in p.hand
                )
                _has_doronchi_field = any(
                    (getattr(bp.card, "name", "") or "").strip() == "ドロンチ"
                    for bp in ([p.active] if p.active else []) + list(p.bench or [])
                )
                _drapa_evo_pending = _has_drapa_hand and _has_doronchi_field
            _has_hr_pre_evo = any(
                is_support(c) and (getattr(c, "id", "") or "") in (
                    "riirienokesshin", "zeiyu", "hakasenokenkyuu",
                )
                for c in p.hand
            ) and not state.support_used_this_turn
            # ドラパルトex進化が控えている → リーリエ延期を無視してていさつしれいを先に使う
            if not _has_hr_pre_evo or _drapa_evo_pending:
                while _try_use_ability_teisatsushirei(state):
                    acted = True
                    p = state.active_player_state()
                    state._record_frame()
        evolve_rounds = 0
        while can_evolve and evolve_rounds < MAX_EVOLVE_ROUNDS_PER_TURN:
            evolve_rounds += 1
            p = state.active_player_state()
            if _try_evolve_once(state):
                acted = True
                state._record_frame()
                # ドロンチ進化後にていさつしれいを即使用
                # ただしリーリエの決心等が手札にある場合はリーリエを先に使う
                _has_hand_refresh_for_teisatsu = any(
                    is_support(c) and (getattr(c, "id", "") or "") in (
                        "riirienokesshin", "zeiyu", "hakasenokenkyuu",
                    )
                    for c in p.hand
                ) and not state.support_used_this_turn
                if not _has_hand_refresh_for_teisatsu:
                    while _try_use_ability_teisatsushirei(state):
                        acted = True
                        p = state.active_player_state()
                        state._record_frame()
            else:
                break

        # 勝ち確チェック（各アクション間）: KOプラン等でエネが付いた後に即攻撃
        if not _is_first_player_first_turn(state) and _can_win_now_check(state):
            break

        if not _is_first_player_first_turn(state) and not state.stadium_played_this_turn:
            for i, c in enumerate(p.hand):
                if is_stadium(c) and play_stadium(state, i):
                    acted = True
                    p = state.active_player_state()
                    state._record_frame()
                    break

        if _try_faitogongu_engine_opener(state):
            acted = True
            p = state.active_player_state()
            state._record_frame()
            _try_put_bench_until_full()
            continue

        if not _is_first_player_first_turn(state) and not state.support_used_this_turn:
            # ボール系ループの前では手札を捨てないサポートのみ使う。
            # ゼイユ等の手札刷新はボール系でグッズを使い切ってから。
            if _try_support_no_discard_only(state):
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

        # ハイパーボールの前にルナサイクルを使う:
        # エンジン揃い＋ルナサイクル未使用＋手札にエネ＋ハイパーボールが手札にある場合、
        # ルナサイクルで3枚引いてからハイパーボールで不要カードを捨てる方が得。
        if (
            getattr(state, "ability_declared_this_turn", None) != "ルナサイクル"
            and _field_has_pokemon(p, _is_lunatone)
            and _field_has_pokemon(p, _is_solrock)
            and any(is_energy(c) and getattr(c, "energy_type", None) == "fighting" for c in p.hand)
            and any(getattr(c, "id", "") == "haipaboru" for c in p.hand)
        ):
            if _try_use_ability_runasaikuru(state):
                acted = True
                p = state.active_player_state()
                state._record_frame()
                _try_put_bench_until_full()
                continue

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
            # 先行1ターンでもHBは使用可（リオル等を早く出す方が重要）
            wants_haipaboru_first = any(
                is_support(c) and (getattr(c, "id", "") or "") in _HAIPABORU_BEFORE_HAND_REFRESH_SUPPORT_IDS
                for c in p.hand
            )

            # ドラパルトexデッキ: なかよしポフィンを最優先（ドラメシヤ・ヨマワルを展開）
            from .deck_strategies import is_dragapult_deck_for_player as _is_drapa_ball
            _is_drapa_deck = _is_drapa_ball(state, state.current_player)
            # ドラパルトデッキ: 1ターン目はHBを使わない（ポフィン/ポケパッドで展開、HBは次ターンにニャースex等に温存）
            # ニャースexが山札にある場合: サポートが手札にない時はHBを温存
            # （次ターンHB→ニャースex→おくのてキャッチ→サポートサーチの連携に使う）
            # ※手札にニャースがいるならHBはアタッカー等に使ってOK
            # ※場・トラッシュにある場合は使用済みなので対象外
            _has_nyarth_in_deck = any(
                (getattr(c, "name", "") or "").strip() == "ニャースex"
                for c in p.deck
            )
            _has_nyarth_in_hand = any(
                (getattr(c, "name", "") or "").strip() == "ニャースex"
                for c in p.hand
            )
            # ドラパルトデッキ: HBスキップ条件
            # 1. ドローサポートが手札にある → リーリエ等で引いてから判断すべき
            # 2. サポートなし+ニャースex確保可能 → HBはニャースex用に温存
            # 3. サポート使用済み → リーリエ等で引いた後の手札でHBを使う必要があるか再判断
            #    ニャースexが手札になければHBで取りに行く価値はあるが、
            #    既にニャースexが場にいるなら不要
            if _is_drapa_deck:
                _has_draw_support_for_hb = any(
                    is_support(c) and (getattr(c, "id", "") or "") in (
                        "riirienokesshin", "zeiyu", "hakasenokenkyuu", "hikari",
                    )
                    for c in p.hand
                )
                _has_support_in_hand = any(is_support(c) for c in p.hand)
                _skip_hb = False
                if not state.support_used_this_turn:
                    # サポート未使用 + ドローサポートが手札にある
                    if _has_draw_support_for_hb:
                        # ニャースexが取れるならHB→ニャースex→おくのてキャッチ連携可能
                        # リーリエ+おくのてキャッチで2つのサポートを活用できる
                        _nyarth_in_deck_for_hb = any(
                            "ニャースex" in (getattr(dc, "name", "") or "")
                            for dc in p.deck
                        )
                        if not _nyarth_in_deck_for_hb:
                            _skip_hb = True  # ニャースex取れない → リーリエで引いてから
                else:
                    # サポート使用済み: ニャースex取っても今ターン活用できない → HBスキップ
                    _skip_hb = True
                if _skip_hb:
                    ball_candidates = [
                        (i, c) for i, c in ball_candidates
                        if getattr(c, "id", "") != "haipaboru"
                    ]

            # ドラパルトexデッキ: ベンチ満員時はHBをスキップ（ニャースex等を取っても出せない）
            # 手札にリーリエの決心等の手札刷新サポートがある場合は先にそちらを使う方が得
            if _is_drapa_deck and len(p.bench) >= BENCH_SIZE:
                _has_hand_refresh = any(
                    is_support(c) and (getattr(c, "id", "") or "") in _SUPPORT_IDS_HAND_REFRESH_FIRST
                    for c in p.hand
                )
                if _has_hand_refresh and not state.support_used_this_turn:
                    ball_candidates = [
                        (i, c) for i, c in ball_candidates
                        if getattr(c, "id", "") != "haipaboru"
                    ]

            def _ball_goods_sort_key(x):
                i, card = x
                cid = getattr(card, "id", "") or ""
                # ドラパルトexデッキ: なかよしポフィン→ポケパッド→HBの順
                # HBは温存してニャースex(→サポートサーチ)に使いたい
                if _is_drapa_deck:
                    if cid == "nakayoshipofuin":
                        return (-2, 0, i)
                    if cid == "pokepaddo":
                        return (-1, 0, i)
                    if cid == "pokemonkixyatchixya":
                        return (0, 0, i)
                    if cid == "haipaboru":
                        # 序盤はHBを温存（ポケパッド/ポフィンで展開できる間は使わない）
                        _has_poffin_or_pad = any(
                            getattr(hc, "id", "") in ("nakayoshipofuin", "pokepaddo")
                            for hc in p.hand
                        )
                        if _has_poffin_or_pad and state.turn_count <= 3:
                            return (2, 0, i)  # 後回し
                        return (1, 0, i)
                    return (1, 1, i)
                # 非ドラパルト: 既存ロジック
                if cid == "pokemonkixyatchixya":
                    return (0, 0, i)
                if cid in ("faitogongu", "pokepaddo"):
                    return (0, 1, i)
                if wants_haipaboru_first and cid == "haipaboru":
                    return (1, 0, i)
                if wants_haipaboru_first:
                    return (1, 1, i)
                return (1, 0, i)

            ball_candidates.sort(key=_ball_goods_sort_key)
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
                    # ファイトゴング/ポケパッドでエンジンが揃ったら、
                    # ルナサイクルを先に使うためにボールループを抜ける。
                    # ドローしてからハイパーボールを使う方が捨てるカードの選択肢が広がる。
                    used_cid = getattr(c, "id", "") or ""
                    # ボール使用後にベンチ出し（種切れ防止＋次のHBで正しい判断ができる）
                    _try_put_bench_until_full()
                    p = state.active_player_state()
                    _engine_break = False
                    if used_cid in ("faitogongu", "pokepaddo"):
                        # エンジンチェック
                        if (
                            _field_has_pokemon(p, _is_lunatone)
                            and _field_has_pokemon(p, _is_solrock)
                            and getattr(state, "ability_declared_this_turn", None) != "ルナサイクル"
                            and any(is_energy(hc) for hc in p.hand)
                        ):
                            _engine_break = True
                            break  # 内側ループを抜ける
                    break
            if not used_ball:
                break
            if _engine_break:
                break  # エンジン揃い→外側ループも抜けてルナサイクルへ
        _try_put_bench_until_full()
        if not _is_first_player_first_turn(state):
            evolve_after_ball = 0
            while evolve_after_ball < MAX_EVOLVE_ROUNDS_PER_TURN and state.turn_count >= 2:
                evolve_after_ball += 1
                p = state.active_player_state()
                if _try_evolve_once(state):
                    acted = True
                    state._record_frame()
                    # ドロンチ進化後にていさつしれい即使用
                    # ただしリーリエの決心等が手札にある場合はリーリエを先に使う
                    _has_hr_teisatsu2 = any(
                        is_support(c) and (getattr(c, "id", "") or "") in (
                            "riirienokesshin", "zeiyu", "hakasenokenkyuu",
                        )
                        for c in p.hand
                    ) and not state.support_used_this_turn
                    if not _has_hr_teisatsu2:
                        while _try_use_ability_teisatsushirei(state):
                            acted = True
                            p = state.active_player_state()
                            state._record_frame()
                else:
                    break

        # ていさつしれい: 毎ターン未使用ドロンチがいれば使う（進化後以外でも）
        # リーリエの決心等が手札にある場合はリーリエを先に使う
        # ていさつしれいでリーリエを引いたら→リーリエ使用→次のていさつしれい
        _HR_IDS_FOR_TEISATSU = ("riirienokesshin", "zeiyu", "hakasenokenkyuu")
        _has_hr_teisatsu3 = any(
            is_support(c) and (getattr(c, "id", "") or "") in _HR_IDS_FOR_TEISATSU
            for c in p.hand
        ) and not state.support_used_this_turn
        if not _has_hr_teisatsu3:
            while _try_use_ability_teisatsushirei(state):
                acted = True
                p = state.active_player_state()
                state._record_frame()
                _try_put_bench_until_full()
                # ていさつしれいでリーリエ等を引いたら先にサポートを使う
                if not state.support_used_this_turn:
                    _drew_hr = any(
                        is_support(c) and (getattr(c, "id", "") or "") in _HR_IDS_FOR_TEISATSU
                        for c in p.hand
                    )
                    if _drew_hr:
                        break  # メインループに戻りサポート使用→再度ていさつしれい

        # 暗号マニアはサポートのため先行 1 ターン目は試さない。ルナサイクルは特性のため先行 1 ターン目でもエネ付与前に試す。
        if not _is_first_player_first_turn(state):
            if _try_angou_before_luna_cycle(state):
                acted = True
                p = state.active_player_state()
                state._record_frame()
                _try_put_bench_until_full()
                continue
        # ルナサイクル: 手札刷新サポート（リーリエの決心等）があるなら後回し。
        # リーリエ→ルナサイクルの順の方が手札が増える（リーリエで6-8枚→ルナサイクルで+3枚）。
        # ルナサイクルを先にすると、引いた3枚がリーリエで山に戻ってしまい無駄。
        # 例外: サポートが既に使用済み、または先行1ターン目（サポート不可）。
        has_hand_refresh_for_luna_delay = (
            not _is_first_player_first_turn(state)
            and not state.support_used_this_turn
            and any(
                is_support(c) and getattr(c, "id", "") in _SUPPORT_IDS_HAND_REFRESH_FIRST
                for c in p.hand
            )
        )
        if not has_hand_refresh_for_luna_delay:
            # ルナサイクルをエネ付与より先に実行する。
            # ルナサイクルで3枚引いた後の方がエネ付与先の判断が良くなる。
            # ただしエネ付けてKOできる確定ケースのみエネ付与を先にする。
            _did_attach_for_ko = False
            if not state.energy_attached_this_turn and not _is_first_player_first_turn(state):
                from .turn_energy import _pick_energy_hand_idx
                from .trainers import attach_energy as _direct_attach
                has_energy = any(is_energy(c) for c in p.hand)
                if has_energy and p.active:
                    opp_lc = state.defending_player_state()
                    our_dmg_now = _our_max_effective_damage(state)
                    # エネ付けてKOできる確定ケースのみ先にエネ付与
                    # ただしルナサイクル可能+夜のタンカがあればルナサイクル先（タンカでエネ回収→KOも後からできる）
                    _luna_can_use = (
                        _field_has_pokemon(p, _is_lunatone)
                        and _field_has_pokemon(p, _is_solrock)
                        and getattr(state, "ability_declared_this_turn", None) != "ルナサイクル"
                    )
                    _has_tanka = any(getattr(c, "id", "") == "yorunotanka" for c in p.hand)
                    _can_luna_then_ko = _luna_can_use and _has_tanka
                    if opp_lc.active and opp_lc.active.hp is not None and opp_lc.active.hp > 0:
                        from .evaluate import _can_ko_with_one_more_energy
                        if (
                            our_dmg_now < opp_lc.active.hp
                            and _can_ko_with_one_more_energy(state, p.active, opp_lc.active, state.current_player)
                            and not _can_luna_then_ko
                        ):
                            # KOに必要なエネタイプを特定し、そのタイプのエネを選ぶ
                            from .damage import _max_effective_damage_if_attach
                            _ko_eidx = None
                            _active_types = list(getattr(p.active, "attached_energy_types", []) or [])
                            for _ei, _ec in enumerate(p.hand):
                                if not is_energy(_ec):
                                    continue
                                _etype = getattr(_ec, "energy_type", None) or "colorless"
                                # ドラパルトexデッキ: 同じタイプのエネを重複して付けない
                                from .deck_strategies import is_dragapult_deck_for_player as _is_drapa_ko
                                if _is_drapa_ko(state, state.current_player):
                                    _an = (getattr(p.active.card, "name", "") or "").strip()
                                    if _an in ("ドラパルトex", "ドロンチ", "ドラメシヤ"):
                                        if _etype == "darkness":
                                            continue
                                        if _etype in ("fire", "psychic") and _etype in _active_types:
                                            continue
                                _dmg_after = _max_effective_damage_if_attach(
                                    state, p.active.card, p.active.attached_energy,
                                    _active_types, _etype, opp_lc.active, state.current_player,
                                )
                                if _dmg_after >= opp_lc.active.hp:
                                    _ko_eidx = _ei
                                    break
                            if _ko_eidx is not None:
                                _direct_attach(state, _ko_eidx)
                                acted = True
                                _did_attach_for_ko = True
                                p = state.active_player_state()
                                state._record_frame()

            # バトル場が攻撃不可+逃げエネ不足+ベンチにアタッカー → 逃げエネ付与
            # ただしルナサイクルが使えるなら3枚ドローの方が価値が高い → ルナサイクル優先
            if not _did_attach_for_ko and not state.energy_attached_this_turn and not _is_first_player_first_turn(state):
                _p_ret = state.active_player_state()
                _luna_can_use_ret = (
                    _field_has_pokemon(_p_ret, _is_lunatone)
                    and _field_has_pokemon(_p_ret, _is_solrock)
                    and getattr(state, "ability_declared_this_turn", None) != "ルナサイクル"
                    and any(
                        is_energy(c) and (getattr(c, "id", "") or "") == "basic-energy-fighting"
                        for c in _p_ret.hand
                    )
                )
                if not _luna_can_use_ret and _p_ret.active and _p_ret.bench:
                    _opp_ret = state.defending_player_state()
                    _legal_active = get_legal_attack_indices(state, _p_ret, _opp_ret) if _opp_ret else []
                    if not _legal_active:
                        _raw_rc = getattr(_p_ret.active.card, "retreat_cost", 1)
                        _tool_ret = getattr(_p_ret.active, "attached_tool", None)
                        _eff_rc = max(0, _raw_rc - (2 if _tool_ret and (getattr(_tool_ret, "id", "") or "") == "fuusen" else 0))
                        if _eff_rc > 0 and _p_ret.active.attached_energy < _eff_rc:
                            _has_energy_hand = any(is_energy(c) for c in _p_ret.hand)
                            if _has_energy_hand:
                                from .turn_energy import _pick_energy_hand_idx
                                from .trainers import attach_energy as _ret_attach
                                _eidx = _pick_energy_hand_idx(_p_ret, state)
                                if _eidx is not None:
                                    _ret_attach(state, _eidx)
                                    acted = True
                                    _did_attach_for_ko = True
                                    p = state.active_player_state()
                                    state._record_frame()

            if not _did_attach_for_ko:
                if _try_use_ability_runasaikuru(state):
                    acted = True
                    p = state.active_player_state()
                    state._record_frame()
                    _try_put_bench_until_full()
                    continue

        # Joint Q: サポート+エネルギーを同時決定（有効なプレイヤーのみ）
        if not state.support_used_this_turn and not state.energy_attached_this_turn:
            if _try_joint_q_decision(state):
                acted = True
                p = state.active_player_state()
                _try_put_bench_until_full()
                if _try_evolve_rounds_after_hand_change(state):
                    acted = True
                continue

        # エネを付けて攻撃すれば今ターン相手を倒せるなら、サポートより先にエネ付与する。
        # 手札刷新サポートを先に使うとエネが山に戻ったり捨てられたりして攻撃チャンスを逃す。
        # バトル場に付けて倒せるケースだけでなく、ベンチに付けてにげる/いれかえで倒せるケースも考慮。
        if not state.energy_attached_this_turn and not _is_first_player_first_turn(state):
            opp_check = state.defending_player_state()
            if opp_check.active and opp_check.active.hp is not None and opp_check.active.hp > 0:
                from .evaluate import _can_ko_with_one_more_energy
                has_energy_in_hand = any(is_energy(c) for c in p.hand)
                if has_energy_in_hand:
                    can_ko_after_attach = False
                    # バトル場にエネ付けて倒せるか
                    if p.active and _can_ko_with_one_more_energy(
                        state, p.active, opp_check.active, state.current_player
                    ):
                        can_ko_after_attach = True
                    # ベンチにエネ付けて、にげる/いれかえで前に出して倒せるか
                    if not can_ko_after_attach:
                        _raw_rc = getattr(p.active.card, "retreat_cost", 1) if p.active else 99
                        _tool = getattr(p.active, "attached_tool", None) if p.active else None
                        _eff_rc = max(0, _raw_rc - (2 if _tool and (getattr(_tool, "id", "") or "") == "fuusen" else 0))
                        _can_retreat_now = (
                            p.active
                            and not getattr(state, "retreat_used_this_turn", False)
                            and getattr(p.active, "special_state", None) not in ("sleep", "paralysis")
                        )
                        can_switch = (
                            # にげるコスト0（ふうせん等）
                            (_can_retreat_now and _eff_rc == 0)
                            # にげるコスト1でエネ付ければにげられる（バトル場にエネ付与→にげる→ベンチから攻撃）
                            or (_can_retreat_now and _eff_rc == 1 and p.active.attached_energy == 0)
                            # いれかえグッズ
                            or any(is_goods(c) and (getattr(c, "effect", None) == "swap_active"
                                   or getattr(c, "id", "") in ("pokemon_irekae", "pokemonirekae"))
                                   for c in p.hand)
                        )
                        if can_switch:
                            for bp in p.bench:
                                # ベンチのポケモンが既に攻撃可能で倒せるか
                                bench_dmg = _max_effective_damage_for_attacker(state, bp, opp_check.active, state.current_player)
                                if bench_dmg >= opp_check.active.hp and get_legal_attack_indices_for_attacker(state, p, opp_check, bp):
                                    can_ko_after_attach = True
                                    break
                                # ベンチにエネ付けて倒せるか（元のロジック）
                                if _can_ko_with_one_more_energy(
                                    state, bp, opp_check.active, state.current_player
                                ):
                                    can_ko_after_attach = True
                                    break
                    if can_ko_after_attach:
                        if _try_attach_energy_with_attack_lookahead(state):
                            acted = True
                            p = state.active_player_state()
                            state._record_frame()

        # 手札刷新サポートの前にエネルギーを付けておく。
        # ゼイユ/博士: トラッシュに送られる → はどうづきで再利用可。
        # ジャッジマン/リーリエ: 山に戻る → 引き直しで戻る保証なし。先に付けた方が得。
        # 将来的にはここの判断も学習で最適化する想定。
        _HAND_REFRESH_SUPPORT_IDS_FOR_ENERGY = (
            "zeiyu", "hakasenokenkyuu", "hakasenokenkyuufutouhakase",
            "jixyajjiman", "riirienokesshin",
        )
        if (
            not state.energy_attached_this_turn
            and not _is_first_player_first_turn(state)
            and not state.support_used_this_turn
            and any(is_support(c) and (getattr(c, "id", "") or "") in _HAND_REFRESH_SUPPORT_IDS_FOR_ENERGY for c in p.hand)
        ):
            if _try_attach_energy_with_attack_lookahead(state):
                acted = True
                p = state.active_player_state()
                state._record_frame()

        # サポートを先に使い、エネQはサポート後の状態を評価する
        has_hand_refresh_support = any(
            is_support(c) and getattr(c, "id", "") in _SUPPORT_IDS_HAND_REFRESH_FIRST for c in p.hand
        )
        if not _is_first_player_first_turn(state) and has_hand_refresh_support and not state.support_used_this_turn:
            if _try_goods_before_hand_refresh(state):
                acted = True
                p = state.active_player_state()
                state._record_frame()
                _try_put_bench_until_full()
                continue

        if not _is_first_player_first_turn(state) and not state.support_used_this_turn:
            has_any_support = any(is_support(c) for c in p.hand)
            if has_any_support and try_support_policy(state):
                acted = True
                p = state.active_player_state()
                state._record_frame()
                _try_put_bench_until_full()
                if _try_evolve_rounds_after_hand_change(state):
                    acted = True
                continue
            _try_put_bench_until_full()

        # エネルギー付与（サポート後の状態を見て判断）
        if not state.energy_attached_this_turn and _try_attach_energy_with_attack_lookahead(state):
            acted = True
            p = state.active_player_state()
            state._record_frame()

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

        if _try_swap_to_attacker(state):
            acted = True
            p = state.active_player_state()
            state._record_frame()

        if _try_attach_one_tool(state):
            acted = True
            p = state.active_player_state()
            state._record_frame()
            # ふうせん装着後ににげコストが変わる → にげる再判断
            if _try_retreat_voluntary(state):
                acted = True
                p = state.active_player_state()
                state._record_frame()
            if _try_swap_to_attacker(state):
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
        ]
        # ドラパルトデッキ: HBを残りグッズでも使わない
        # 1. ドローサポート未使用 → リーリエ等で引いてから判断すべき
        # 2. サポート使用済み → ニャースex取っても今ターン活用できない
        from .deck_strategies import is_dragapult_deck_for_player as _is_drapa_goods
        if _is_drapa_goods(state, state.current_player):
            _skip_hb_goods = False
            if not state.support_used_this_turn:
                _has_draw_support_goods = any(
                    is_support(c) and (getattr(c, "id", "") or "") in (
                        "riirienokesshin", "zeiyu", "hakasenokenkyuu", "hikari",
                    )
                    for c in p.hand
                )
                _has_nyarth_goods = any(
                    (getattr(c, "name", "") or "").strip() == "ニャースex"
                    for c in list(p.hand) + list(p.deck)
                )
                _has_support_goods = any(is_support(c) for c in p.hand)
                if _has_draw_support_goods:
                    _nyarth_in_deck_goods = any(
                        "ニャースex" in (getattr(dc, "name", "") or "")
                        for dc in p.deck
                    )
                    if not _nyarth_in_deck_goods:
                        _skip_hb_goods = True
            else:
                # サポート使用済み → HBでニャースex等を取っても今ターン活用できない
                _skip_hb_goods = True
            if _skip_hb_goods:
                goods_order = [
                    (i, c) for i, c in goods_order
                    if (getattr(c, "id", "") or "") != "haipaboru"
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

    # カースドボム: 攻撃前にサマヨール/ヨノワールで仕留める
    if not _is_first_player_first_turn(state) and state.turn_count > 0:
        if _try_use_ability_cursed_bomb(state):
            acted = True
            p = state.active_player_state()
            opp = state.defending_player_state()
            state._record_frame()

    # メガブレイブ封印解除トリック:
    # メガルカリオexのメガブレイブが封印されているとき、
    # ポケモンいれかえ → ふうせん付きベンチポケモンに交代 → 即にげるでメガルカリオexを戻す
    # → バトル場に戻ると封印がリセットされ、メガブレイブが再び使える。
    if (
        p.active
        and (getattr(p.active.card, "name", "") or "").strip() == "メガルカリオex"
        and getattr(p.active, "disabled_attack_name", None) == "メガブレイブ"
        and not getattr(state, "retreat_used_this_turn", False)
    ):
        # 手札にポケモンいれかえがあるか
        _swap_hi = None
        for _i, _c in enumerate(p.hand):
            if is_goods(_c) and (getattr(_c, "effect", None) == "swap_active" or getattr(_c, "id", "") in ("pokemon_irekae", "pokemonirekae")):
                _swap_hi = _i
                break
        # ベンチにふうせん付き（にげるコスト0）のポケモンがいるか
        _balloon_bi = None
        if _swap_hi is not None:
            for _bi, _bp in enumerate(p.bench):
                _tool = getattr(_bp, "attached_tool", None)
                if _tool and (getattr(_tool, "id", "") or "") == "fuusen":
                    _raw_rc = getattr(_bp.card, "retreat_cost", 1)
                    if max(0, _raw_rc - 2) == 0:
                        _balloon_bi = _bi
                        break
        if _swap_hi is not None and _balloon_bi is not None:
            # いれかえ → ふうせんポケモンに交代
            _mega_lucario_bp = p.active
            if use_pokemon_swap(state, _swap_hi, _balloon_bi):
                p = state.active_player_state()
                state._record_frame()
                # ふうせんポケモンから即にげる → メガルカリオexをベンチから前に戻す
                # メガルカリオexは今ベンチにいるはず
                _mega_bi = None
                for _bi2, _bp2 in enumerate(p.bench):
                    if (getattr(_bp2.card, "name", "") or "").strip() == "メガルカリオex" and _bp2 is _mega_lucario_bp:
                        _mega_bi = _bi2
                        break
                if _mega_bi is not None and retreat(state, _mega_bi):
                    p = state.active_player_state()
                    opp = state.defending_player_state()
                    state.log(
                        f"{state.player_name(state.current_player)}: メガブレイブ封印解除トリック成功 → メガルカリオexが再びメガブレイブを使用可能"
                    )
                    state._record_frame()
                    acted = True

    p = state.active_player_state()
    opp = state.defending_player_state()

    # --- Fallback: エネルギー・スタジアム・どうぐ未消化チェック (Pattern 1/6/7/8) ---
    # メインループが何もせず break した場合でも、手張り・スタジアム・どうぐを試みる。
    if not state.energy_attached_this_turn and not _is_first_player_first_turn(state):
        if any(is_energy(c) for c in p.hand):
            if _try_attach_energy_with_attack_lookahead(state):
                acted = True
                p = state.active_player_state()
                state._record_frame()

    if not state.stadium_played_this_turn and not _is_first_player_first_turn(state):
        for _fi, _fc in enumerate(p.hand):
            if is_stadium(_fc) and play_stadium(state, _fi):
                acted = True
                p = state.active_player_state()
                state._record_frame()
                break

    if _try_attach_one_tool(state):
        acted = True
        p = state.active_player_state()
        state._record_frame()

    # サポート未使用フォールバック: ボスの指令等が手札にあるなら使う
    if not state.support_used_this_turn and not _is_first_player_first_turn(state):
        has_any_support = any(is_support(c) for c in p.hand)
        if has_any_support:
            if try_support_policy(state):
                acted = True
                p = state.active_player_state()
                state._record_frame()
                _try_put_bench_until_full()

    opp = state.defending_player_state()

    # ドラパルトデッキ: 攻撃前にバトル場がexサポートポケモンなら逃げる
    # exがバトル場に残るとサイド2枚献上リスク
    from .deck_strategies import is_dragapult_deck_for_player as _is_drapa_ex_retreat
    if _is_drapa_ex_retreat(state, state.current_player) and p.active and p.bench:
        _ex_active_name = (getattr(p.active.card, "name", "") or "").strip()
        _is_ex_support = _ex_active_name in ("キチキギスex", "ニャースex")
        if _is_ex_support and not getattr(state, "retreat_used_this_turn", False):
            _raw_rc = getattr(p.active.card, "retreat_cost", 1)
            _tool = getattr(p.active, "attached_tool", None)
            _eff_rc = max(0, _raw_rc - (2 if _tool and (getattr(_tool, "id", "") or "") == "fuusen" else 0))
            if p.active.attached_energy >= _eff_rc:
                # 逃げ先: ドラメシヤ/ヨマワル/スボミー等の非exを優先
                _best_retreat_idx = None
                _best_retreat_score = -1
                for _ri, _rbp in enumerate(p.bench):
                    _rn = (getattr(_rbp.card, "name", "") or "").strip()
                    _is_ex_bench = bool(getattr(_rbp.card, "is_ex", False))
                    if _is_ex_bench:
                        continue  # ex同士の入れ替えは意味なし
                    _rscore = 100
                    # ドラメシヤが場に複数いれば1体は壁に使える
                    _drameshiya_count = sum(
                        1 for _bp2 in p.bench
                        if (getattr(_bp2.card, "name", "") or "").strip() == "ドラメシヤ"
                    )
                    if _rn == "ドラメシヤ":
                        if _drameshiya_count >= 2:
                            _rscore = 250  # 余裕あり → 壁として最適
                        else:
                            _rscore = 50  # 1体しかない → 進化の基盤、温存
                    elif _rn == "スボミー":
                        _rscore = 200  # グッズロック壁
                    elif _rn == "ヨマワル":
                        _rscore = 80  # カースドボム用に温存、壁にはしない
                    elif _rn in ("ドロンチ", "ドラパルトex"):
                        _rscore = 300  # アタッカー
                    if _rscore > _best_retreat_score:
                        _best_retreat_score = _rscore
                        _best_retreat_idx = _ri
                if _best_retreat_idx is not None:
                    retreat(state, _best_retreat_idx)
                    acted = True
                    p = state.active_player_state()
                    opp = state.defending_player_state()
                    state._record_frame()

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
