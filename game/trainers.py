"""トレーナー（グッズ・サポート・どうぐ・いれかえ・エネルギー付与）。"""
import random

from card import (
    get_card_by_id,
    get_card_by_name,
    is_basic_pokemon,
    is_energy,
    is_goods,
    is_pokemon,
    is_stage2_pokemon,
    is_support,
)

_BASIC_ENERGY_TYPES = ("grass", "fire", "water", "lightning", "psychic", "fighting", "darkness", "metal", "fairy")

from .damage import _max_effective_damage_for_attacker
from .evolution import _apply_evolution, _can_evolve_onto
from .weights import get_haipaboru_discard_weight
from .state import (
    GameState,
    PlayerState,
    BattlePokemon,
    _card_label,
    _clear_status,
    _flip_coin,
    _is_first_player_first_turn,
    _log_choice,
)


def would_haipaboru_fetch_evolution(p: PlayerState) -> bool:
    """ハイパーボールを使ったときに山札から取ってくる 1 枚が進化ポケモンかどうか。"""
    found = _find_pokemon_for_haipaboru(p)
    if not found:
        return False
    return bool(getattr(found[1], "evolves_from", None))


def _haipaboru_already_have(p: PlayerState, card) -> bool:
    """手札・バトル場・ベンチに同じ id または名前のポケモンがいるか。"""
    cid = getattr(card, "id", None) or ""
    cname = getattr(card, "name", "") or ""
    for h in p.hand:
        if is_pokemon(h) and ((getattr(h, "id", None) or "") == cid or (getattr(h, "name", "") or "") == cname):
            return True
    if p.active:
        if (getattr(p.active.card, "id", None) or "") == cid or (getattr(p.active.card, "name", "") or "") == cname:
            return True
    for bp in p.bench:
        if (getattr(bp.card, "id", None) or "") == cid or (getattr(bp.card, "name", "") or "") == cname:
            return True
    return False


def _haipaboru_strength(card) -> float:
    """強さスコア（HP と ex ボーナス）。"""
    hp = getattr(card, "max_hp", 0) or 0
    ex = getattr(card, "is_ex", False) or ("ex" in (getattr(card, "name", "") or ""))
    return hp + (5000.0 if ex else 0.0)


def _find_pokemon_for_haipaboru(p: PlayerState) -> tuple[int, object] | None:
    """
    ハイパーボール用：山札から 1 枚を選び (deck_index, card) を返す。
    手札・場に既にいるポケモンは避け、優先度は
    (1) 2進化で場にのせられる (2) 1進化で場にのせられる (3) たね (4) 強さ（HP・ex）。
    """
    candidates = [(i, c) for i, c in enumerate(p.deck) if is_pokemon(c)]
    if not candidates:
        return None
    preferred = [(i, c) for i, c in candidates if not _haipaboru_already_have(p, c)]
    pool = preferred if preferred else candidates

    def score(deck_idx: int, c) -> float:
        strength = _haipaboru_strength(c)
        field_cards = ([p.active.card] if p.active else []) + [bp.card for bp in p.bench]
        for field_card in field_cards:
            if _can_evolve_onto(field_card, c):
                if is_stage2_pokemon(c):
                    return 10000.0 + strength
                return 5000.0 + strength
        if is_basic_pokemon(c) or not getattr(c, "evolves_from", None):
            return 1000.0 + strength
        return strength

    best = max(pool, key=lambda x: (score(x[0], x[1]), -x[0]))
    return best


def attach_energy(state: GameState, hand_index: int, bench_index: int | None = None) -> bool:
    """手札の energy_index 番目のエネルギーをバトル場またはベンチのポケモンに付与。bench_index 指定でベンチに付与。"""
    p = state.active_player_state()
    if hand_index < 0 or hand_index >= len(p.hand):
        return False
    if not is_energy(p.hand[hand_index]):
        return False
    energy_card = p.hand[hand_index]
    energy_type = getattr(energy_card, "energy_type", None)
    slot_type = energy_type if energy_type else "colorless"

    if bench_index is not None:
        if bench_index < 0 or bench_index >= len(p.bench):
            return False
        if state.energy_attached_this_turn:
            return False
        target = p.bench[bench_index]
        target.attached_energy += 1
        target.attached_energy_types.append(slot_type)
        p.hand.pop(hand_index)
        state.energy_attached_this_turn = True
        state.log(
            f"{state.player_name(state.current_player)}: エネルギーを 1 つ付与（ベンチの {target.card.name}、エネルギー {target.attached_energy} 個）"
        )
        return True
    if not p.active:
        return False
    if state.energy_attached_this_turn:
        return False
    p.active.attached_energy += 1
    p.active.attached_energy_types.append(slot_type)
    p.hand.pop(hand_index)
    state.energy_attached_this_turn = True
    state.log(f"{state.player_name(state.current_player)}: エネルギーを 1 つ付与（バトル場のエネルギー {p.active.attached_energy} 個）")
    return True


def use_potion(state: GameState, hand_index: int) -> bool:
    """きずぐすり（id=potion）を使用して自分のバトル場のポケモンを回復。手札の hand_index 番目がきずぐすりのときだけ実行。"""
    p = state.active_player_state()
    if not p.active or hand_index < 0 or hand_index >= len(p.hand):
        return False
    card = p.hand[hand_index]
    if not is_goods(card) or getattr(card, "effect", None) != "heal":
        return False
    if getattr(card, "id", None) != "potion":
        return False
    amount = getattr(card, "heal_amount", 20)
    before = p.active.hp
    p.active.hp = min(p.active.hp + amount, p.active.max_hp)
    p.discard.append(p.hand.pop(hand_index))
    when_drawn = "（今ターンのドローで引いた）" if card in state.drawn_this_turn else ""
    state.log(f"{state.player_name(state.current_player)}: きずぐすりを使用 → バトル場のポケモンを {before} → {p.active.hp} に回復{when_drawn}")
    return True


def use_trainer_goods(
    state: GameState,
    hand_index: int,
    *,
    pokemon_catcher_bench_index: int | None = None,
) -> bool:
    """トレーナー（グッズ）の効果を実行。きずぐすり・いれかえ・どうぐ以外のアイテムを id / 名前で判定して処理する。"""
    p = state.active_player_state()
    if hand_index < 0 or hand_index >= len(p.hand):
        return False
    card = p.hand[hand_index]
    if not is_goods(card):
        return False
    cid = getattr(card, "id", "")
    if (getattr(card, "effect", None) in ("heal", "swap_active") or getattr(card, "is_tool", False)) and cid not in ("supaboru", "haipaboru", "fushiginaame"):
        return False
    name_ja = getattr(card, "name", "")

    if cid == "fushiginaame":
        if state.turn_count < 2:
            return False
        stage1_id = None
        stage2_card = None
        stage2_idx = None
        for i, c in enumerate(p.hand):
            if i == hand_index or not is_pokemon(c) or not getattr(c, "evolves_from", None):
                continue
            try:
                stage1_ref = get_card_by_id((c.evolves_from or "").strip())
            except ValueError:
                base = (c.evolves_from or "").strip()
                stage1_ref = get_card_by_name(base)
                if stage1_ref is None:
                    stage1_ref = next(
                        (h for h in p.hand if is_pokemon(h) and (
                            (getattr(h, "id", "") or "").strip() == base
                            or (getattr(h, "id", "") or "").startswith(base + "-")
                            or (getattr(h, "name", "") or "").strip() == base
                        )),
                        None,
                    )
            if not stage1_ref or not is_pokemon(stage1_ref):
                continue
            is_stage2 = is_stage2_pokemon(c) or bool(getattr(stage1_ref, "evolves_from", None))
            if not is_stage2:
                continue
            stage1_id = (stage1_ref.evolves_from or "").strip()
            stage2_card = c
            stage2_idx = i
            break
        if stage2_card is None or stage1_id is None:
            return False
        try:
            stage1_ref = get_card_by_id(stage2_card.evolves_from.strip())
        except ValueError:
            base = (stage2_card.evolves_from or "").strip()
            stage1_ref = get_card_by_name(base)
            if stage1_ref is None:
                stage1_ref = next(
                    (h for h in p.hand if is_pokemon(h) and (
                        (getattr(h, "id", "") or "").strip() == base
                        or (getattr(h, "id", "") or "").startswith(base + "-")
                        or (getattr(h, "name", "") or "").strip() == base
                    )),
                    None,
                )
        if not stage1_ref:
            return False
        target_bp = None
        if p.active and is_basic_pokemon(p.active.card) and not getattr(p.active, "put_on_bench_this_turn", False):
            if _can_evolve_onto(p.active.card, stage1_ref):
                target_bp = p.active
        if target_bp is None:
            for bp in p.bench:
                if not is_basic_pokemon(bp.card) or getattr(bp, "put_on_bench_this_turn", False):
                    continue
                if _can_evolve_onto(bp.card, stage1_ref):
                    target_bp = bp
                    break
        if target_bp is None:
            return False
        _apply_evolution(
            target_bp, stage2_card, state,
            f"{state.player_name(state.current_player)}: ふしぎなアメで ",
        )
        if stage2_idx < hand_index:
            p.hand.pop(hand_index)
            p.hand.pop(stage2_idx)
        else:
            p.hand.pop(stage2_idx)
            p.hand.pop(hand_index)
        p.discard.append(card)
        state.log(f"{state.player_name(state.current_player)}: ふしぎなアメを使用（手札から {stage2_card.name} をバトル場またはベンチのたねにのせて 1 進化をとばして進化）")
        return True

    if cid == "otodokedoron":
        if _flip_coin() and _flip_coin():
            if p.deck:
                idx = random.randint(0, len(p.deck) - 1)
                chosen = p.deck.pop(idx)
                p.hand.append(chosen)
                state.drawn_this_turn.append(chosen)
                random.shuffle(p.deck)
                p.discard.append(p.hand.pop(hand_index))
                state.log(f"{state.player_name(state.current_player)}: おとどけドローンを使用（コイン2回オモテ）→ 山札から {_card_label(chosen)} を手札に加えた")
                return True
        p.discard.append(p.hand.pop(hand_index))
        state.log(f"{state.player_name(state.current_player)}: おとどけドローンを使用（コイン裏）→ 効果なし")
        return True

    _is_enerugikaishixyuu = (cid == "unknown" and name_ja == "エネルギー回収") or cid == "enerugikaishixyuu" or name_ja == "エネルギー回収"
    if _is_enerugikaishixyuu:
        indices = []
        for i, c in enumerate(p.discard):
            if is_energy(c) and getattr(c, "energy_type", None) in _BASIC_ENERGY_TYPES:
                indices.append(i)
                if len(indices) >= 2:
                    break
        taken = [p.discard[i] for i in indices]
        for i in sorted(indices, reverse=True):
            p.discard.pop(i)
        if taken:
            p.hand.extend(taken)
            p.discard.append(p.hand.pop(hand_index))
            state.log(f"{state.player_name(state.current_player)}: エネルギー回収を使用 → トラッシュから基本エネルギー {len(taken)} 枚を手札に加えた")
            return True
        return False

    _is_erekijienereta = cid == "erekijienereta" or name_ja == "エレキジェネレーター"
    if _is_erekijienereta and p.bench and p.deck:
        look = min(5, len(p.deck))
        top = [p.deck.pop(0) for _ in range(look)]
        energies = [c for c in top if is_energy(c) and getattr(c, "energy_type", None) == "lightning"][:2]
        opp = state.defending_player_state()
        best_bi = max(
            range(len(p.bench)),
            key=lambda i: _max_effective_damage_for_attacker(state, p.bench[i], opp.active, state.current_player),
        )
        for c in energies:
            p.bench[best_bi].attached_energy += 1
            p.bench[best_bi].attached_energy_types.append("lightning")
        rest = [c for c in top if c not in energies]
        p.deck.extend(rest)
        random.shuffle(p.deck)
        p.discard.append(p.hand.pop(hand_index))
        if energies:
            state.log(f"{state.player_name(state.current_player)}: エレキジェネレーターを使用 → 山札上から 5 枚のうち基本雷エネルギー {len(energies)} 枚をベンチの {p.bench[best_bi].card.name} につけた")
        else:
            state.log(f"{state.player_name(state.current_player)}: エレキジェネレーターを使用（山札上 5 枚に基本雷エネルギーなし）")
        return True

    if cid == "supaboru" and p.deck:
        look = min(7, len(p.deck))
        top = [p.deck.pop(0) for _ in range(look)]
        pokemon = next((c for c in top if is_pokemon(c)), None)
        if pokemon:
            p.hand.append(pokemon)
            state.drawn_this_turn.append(pokemon)
            top.remove(pokemon)
        random.shuffle(p.deck)
        p.deck.extend(top)
        random.shuffle(p.deck)
        p.discard.append(p.hand.pop(hand_index))
        pokemon_label = _card_label(pokemon) if pokemon else ""
        state.log(
            f"{state.player_name(state.current_player)}: スーパーボールを使用 → 山札上から {look} 枚を見てポケモン 1 枚を手札に加えた → {pokemon_label}"
            if pokemon_label
            else f"{state.player_name(state.current_player)}: スーパーボールを使用 → 山札上から {look} 枚を見たがポケモンなし"
        )
        return True

    if cid == "haipaboru" and len(p.hand) >= 3 and p.deck:
        weights = state.get_weights_for_player(state.current_player)
        hand_without_haipaboru = [(i, p.hand[i]) for i in range(len(p.hand)) if i != hand_index]
        name_counts = {}
        for _i, c in hand_without_haipaboru:
            n = getattr(c, "name", "") or getattr(c, "id", "") or ""
            name_counts[n] = name_counts.get(n, 0) + 1
        support_count = sum(1 for _i, c in hand_without_haipaboru if is_support(c))
        scored = []
        for i, c in hand_without_haipaboru:
            discard_score = get_haipaboru_discard_weight(weights, c)
            if name_counts.get(getattr(c, "name", "") or getattr(c, "id", "") or "", 0) >= 2:
                discard_score += 500
            if is_support(c) and support_count >= 2:
                discard_score += 300
            scored.append((i, discard_score))
        scored.sort(key=lambda x: -x[1])
        to_discard_idx = [scored[0][0], scored[1][0]]
        cards_to_log = [p.hand[i] for i in to_discard_idx]
        for i in sorted(to_discard_idx, reverse=True):
            p.discard.append(p.hand.pop(i))
        for c in cards_to_log:
            _log_choice(state, "haipaboru_discard", card_id=getattr(c, "id", None) or getattr(c, "name", ""))
        new_hi = p.hand.index(card)

        pokemon_found = _find_pokemon_for_haipaboru(p)

        if pokemon_found:
            i, c = pokemon_found
            p.deck.pop(i)
            p.hand.append(c)
            state.drawn_this_turn.append(c)
            random.shuffle(p.deck)
        p.discard.append(p.hand.pop(new_hi))
        add_label = f" → {_card_label(pokemon_found[1])}" if pokemon_found else ""
        state.log(
            f"{state.player_name(state.current_player)}: ハイパーボールを使用（手札 2 枚トラッシュ）→ 山札からポケモン 1 枚を手札に加えた{add_label}"
            if pokemon_found
            else f"{state.player_name(state.current_player)}: ハイパーボールを使用（手札 2 枚トラッシュ、山札にポケモンなし）"
        )
        return True

    if cid == "pokemonkixyatchixya":
        opp = state.defending_player_state()
        if opp.bench and opp.active and _flip_coin():
            idx = (
                pokemon_catcher_bench_index
                if pokemon_catcher_bench_index is not None
                and 0 <= pokemon_catcher_bench_index < len(opp.bench)
                else random.randint(0, len(opp.bench) - 1)
            )
            opp.active, opp.bench[idx] = opp.bench[idx], opp.active
            p.discard.append(p.hand.pop(hand_index))
            state.log(f"{state.player_name(state.current_player)}: ポケモンキャッチャーを使用（コイン表）→ 相手のベンチとバトルポケモンを入れ替えた（{opp.active.card.name} がバトル場に）")
            return True
        p.discard.append(p.hand.pop(hand_index))
        state.log(f"{state.player_name(state.current_player)}: ポケモンキャッチャーを使用（コイン裏）→ 効果なし")
        return True

    return False


def attach_tool(state: GameState, hand_index: int, bench_index: int | None = None) -> bool:
    """ポケモンのどうぐを手札から自分のバトル場またはベンチのポケモンにつける（1 匹に 1 枚まで）。"""
    p = state.active_player_state()
    if hand_index < 0 or hand_index >= len(p.hand):
        return False
    card = p.hand[hand_index]
    if not is_goods(card) or not getattr(card, "is_tool", False):
        return False
    if bench_index is None:
        target = p.active
    else:
        if bench_index < 0 or bench_index >= len(p.bench):
            return False
        target = p.bench[bench_index]
    if not target or getattr(target, "attached_tool", None) is not None:
        return False
    cond = getattr(card, "tool_condition_type", None)
    if cond is not None and getattr(target.card, "pokemon_type", None) != cond:
        return False
    target.attached_tool = card
    p.hand.pop(hand_index)
    where = "バトル場" if bench_index is None else "ベンチ"
    state.log(
        f"{state.player_name(state.current_player)}: {card.name} を {where}の {target.card.name} につけた"
    )
    return True


def use_pokemon_swap(state: GameState, hand_index: int, bench_index: int) -> bool:
    """ポケモンいれかえを使用して自分のバトル場のポケモンとベンチの指定 1 体を入れ替える。"""
    p = state.active_player_state()
    if not p.active or bench_index < 0 or bench_index >= len(p.bench):
        return False
    if hand_index < 0 or hand_index >= len(p.hand):
        return False
    card = p.hand[hand_index]
    if not is_goods(card):
        return False
    if getattr(card, "effect", None) != "swap_active" and getattr(card, "id", "") not in ("pokemon_irekae", "pokemonirekae"):
        return False
    old_active = p.active
    p.active = p.bench[bench_index]
    p.bench[bench_index] = old_active
    _clear_status(old_active)
    p.discard.append(p.hand.pop(hand_index))
    state.log(
        f"{state.player_name(state.current_player)}: ポケモンいれかえを使用 → バトル場に {p.active.card.name}、ベンチに {old_active.card.name} を入れ替え"
    )
    return True


def use_support(state: GameState, hand_index: int) -> bool:
    """
    サポートカードを使用。1 ターンに 1 枚まで。先行の 1 ターン目は使用不可。
    ネモ: デッキから 3 枚引く。
    """
    p = state.active_player_state()
    if hand_index < 0 or hand_index >= len(p.hand):
        return False
    card = p.hand[hand_index]
    if not is_support(card):
        return False
    if state.support_used_this_turn:
        return False
    if _is_first_player_first_turn(state):
        return False
    effect = getattr(card, "effect", "")
    cid = getattr(card, "id", "")
    if cid == "tanpankozou":
        n_hand = len(p.hand)
        used_card = p.hand[hand_index]
        rest = [p.hand[j] for j in range(len(p.hand)) if j != hand_index]
        p.deck.extend(rest)
        p.hand = []
        random.shuffle(p.deck)
        drawn = p.draw(5)
        p.hand.extend(drawn)
        state.drawn_this_turn.extend(drawn)
        p.discard.append(used_card)
        state.support_used_this_turn = True
        drawn_names = ", ".join(_card_label(c) for c in drawn)
        state.log(f"{state.player_name(state.current_player)}: たんぱんこぞうを使用 → 手札 {n_hand} 枚を山札にもどして切り、山札から 5 枚ドロー → [{drawn_names}]（手札 {len(p.hand)} 枚）")
        return True
    if cid in ("hakasenokenkyuu", "hakasenokenkyuufutouhakase"):
        p.discard.extend(p.hand)
        p.hand.clear()
        drawn = p.draw(7)
        p.hand.extend(drawn)
        state.drawn_this_turn.extend(drawn)
        state.support_used_this_turn = True
        drawn_names = ", ".join(_card_label(c) for c in drawn)
        state.log(f"{state.player_name(state.current_player)}: 博士の研究を使用 → 手札をすべてトラッシュし、山札から 7 枚ドロー → [{drawn_names}]（手札 {len(p.hand)} 枚）")
        return True
    if cid == "jixyajjiman":
        opp = state.defending_player_state()
        opp.deck.extend(opp.hand)
        opp.hand = []
        random.shuffle(opp.deck)
        opp_drawn = opp.draw(4)
        opp.hand = opp_drawn
        opp_drawn_names = ", ".join(_card_label(c) for c in opp_drawn)
        p.hand.pop(hand_index)
        p.deck.extend(p.hand)
        p.hand = []
        random.shuffle(p.deck)
        p_drawn = p.draw(4)
        p.hand = p_drawn
        state.drawn_this_turn.extend(p_drawn)
        p.discard.append(card)
        p_drawn_names = ", ".join(_card_label(c) for c in p_drawn)
        state.log(f"{state.player_name(state.current_player)}: ジャッジマン使用")
        state.support_used_this_turn = True
        return True
    if cid == "kihada":
        if len(p.hand) <= 1:
            return False
        if len(p.hand) >= 5:
            return False
        kihada_card = p.hand.pop(hand_index)
        card_to_bottom = p.hand.pop(0)
        p.deck.insert(0, card_to_bottom)
        need = 5 - len(p.hand)
        drawn = p.draw(need)
        p.hand.extend(drawn)
        state.drawn_this_turn.extend(drawn)
        p.discard.append(kihada_card)
        state.support_used_this_turn = True
        drawn_names = ", ".join(_card_label(c) for c in drawn)
        state.log(f"{state.player_name(state.current_player)}: キハダを使用 → 手札の 1 枚を山札の下にもどし、{need} 枚ドロー → [{drawn_names}]（手札 {len(p.hand)} 枚）")
        return True
    if effect == "draw_3" or cid in ("nemo", "nemokako", "nemomirai"):
        n = getattr(card, "draw_count", 3)
        drawn = p.draw(n)
        p.hand.extend(drawn)
        state.drawn_this_turn.extend(drawn)
        p.discard.append(p.hand.pop(hand_index))
        state.support_used_this_turn = True
        drawn_names = ", ".join(_card_label(c) for c in drawn)
        state.log(
            f"{state.player_name(state.current_player)}: {card.name} を使用 → 山札から {len(drawn)} 枚ドロー → [{drawn_names}]（手札 {len(p.hand)} 枚、山札 {len(p.deck)} 枚）"
        )
        return True
    return False
