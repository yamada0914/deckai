"""攻撃実行と最良技選択。"""
import random

from card import is_pokemon, is_energy

from .damage import (
    _attack_damage_for_eval,
    _effective_damage_to_defender,
)
from .state import (
    GameState,
    PlayerState,
    BattlePokemon,
    _can_pay_energy_cost,
    _flip_coin,
    _prizes_for_ko,
    _take_prize,
    _check_game_end,
    _promote_from_bench,
    _apply_status,
    _card_label,
    _put_energy_cards_in_discard,
    _handle_opponent_ko,
    _handle_own_active_ko,
    _log_prize_count_for_ko,
)
from .weights import get_attack_weight

_KO_BONUS_FOR_ATTACK = 10000


def _attack_key(card, atk) -> tuple[str, str, str]:
    """(カード名, レギュレーション, 技名)。同じ技名でもカード・レギュで効果が違う場合の分岐用。"""
    name = getattr(card, "name", "")
    reg = getattr(card, "regulation", None) or ""
    return (name, reg, atk.name)


# 技の特殊効果は (カード名, レギュレーション, 技名) で識別（ダメージ値の差なども区別できる）。
# 新カード追加時: 同じ効果の技なら既存の frozenset にタプルを 1 つ追加。新効果なら新 frozenset を作り attack() 内の分岐と _ATTACK_HAS_MERIT_EFFECT に追加する。
_TSUIGEKI_BARI_BARI = frozenset({("バチンウニ", "G", "ついげきバリバリ")})
_SHIPPEGAESHI_PRIZE_BONUS = frozenset({("ワルビル", "G", "しっぺがえし"), ("ワルビル", "", "しっぺがえし")})
_AVENGE_NAKKLE_KO_BONUS = frozenset({("ルカリオ", "G", "アベンジナックル")})
_NAGUTTE_KAKURERU = frozenset({("ウソッキー", "G", "なぐってかくれる")})
_NAKIGOE_DAMAGE_REDUCTION = frozenset({("ピカチュウ", "G", "なきごえ")})
_KASOKUZUKI_DISABLE = frozenset({("ルカリオ", "G", "かそくづき")})
_FUKIARASU = frozenset({("カイデン", "G", "ふきあらす")})
_MAGNEJECT = frozenset({("ジバコイル", "G", "マグネリジェクト")})
_TOMODACHI_O_SAGASU = frozenset({("ノコッチ", "G", "ともだちをさがす")})
_QUICK_DRAW = frozenset({("ミライドンex", "G", "クイックドロー")})
_EREKI_CHARGE = frozenset({("ライチュウ", "G", "エレキチャージ")})
_10MAN_VOLT = frozenset({("ライチュウ", "G", "10まんボルト")})
_GABUGABU_BITE = frozenset({("ワルビアル", "G", "ガブガブバイト")})
_TECHNO_TURBO = frozenset({("ミライドンex", "G", "テクノターボ")})

# メリット効果がある技のみ 0 ダメージでも選択可能（自傷・コストのみの技は除く）。
# 上記の「効果用 frozenset」のうち、0 ダメージでも打つ価値があるものを | でまとめている。新規効果を追加したらここにも追加する。
_ATTACK_HAS_MERIT_EFFECT = (
    _NAGUTTE_KAKURERU
    | _NAKIGOE_DAMAGE_REDUCTION
    | _FUKIARASU
    | _MAGNEJECT
    | _TOMODACHI_O_SAGASU
    | _QUICK_DRAW
    | _EREKI_CHARGE
    | _GABUGABU_BITE
    | _TECHNO_TURBO
)


def attack_has_merit_effect_at_zero_damage(card, atk) -> bool:
    """0 ダメージでも打つメリットがある技か（生贄を出す代わりに打たない判断に利用）。"""
    atk_key = _attack_key(card, atk)
    if atk_key in _ATTACK_HAS_MERIT_EFFECT:
        return True
    if getattr(atk, "status_effect", None) and getattr(atk, "status_effect_target", "opponent") != "self":
        return True
    if getattr(atk, "bench_damage", 0) > 0 and getattr(atk, "bench_damage_target", "opponent") != "self":
        return True
    return False


def _handle_bench_ko(
    state: GameState,
    owner: PlayerState,
    koed_bench_bp: BattlePokemon,
    bench_target_self: bool,
    p: PlayerState,
    opp: PlayerState,
) -> bool:
    """ベンチ 1 体のきぜつ処理（トラッシュ送り・サイド取得）。勝敗がついたら True。"""
    bp_name = koed_bench_bp.card.name
    owner.discard.append(koed_bench_bp.card)
    tool = getattr(koed_bench_bp, "attached_tool", None)
    if tool:
        owner.discard.append(tool)
    if bench_target_self:
        state.log(f"{state.player_name(state.current_player)} のベンチの {bp_name} がきぜつ！")
        prize_count = _prizes_for_ko(koed_bench_bp)
        _log_prize_count_for_ko(state, state.opponent(), koed_bench_bp, prize_count)
        for _ in range(prize_count):
            if _take_prize(state, state.opponent()):
                return True
    else:
        state.log(f"{state.player_name(state.opponent())} のベンチの {bp_name} がきぜつ！（{opp.knockouts_suffered + 1} 回目）")
        if _handle_opponent_ko(opp, state, koed_bench_bp):
            return True
    return False


def _apply_attack_status_effect(
    state: GameState,
    atk,
    target_bp: BattlePokemon | None,
    skip_opponent_status: bool,
    opp: PlayerState,
) -> None:
    """技の状態異常効果を対象に付与する（コイン指定ありなら判定してから）。"""
    if not target_bp or (skip_opponent_status and target_bp == opp.active):
        return
    status_effect = getattr(atk, "status_effect", None)
    if not status_effect:
        return
    on_coin_heads = getattr(atk, "status_effect_on_coin_heads", False)
    poison_damage = getattr(atk, "poison_damage_if_poison", 10)
    prefix = f"{state.player_name(state.current_player)}: 「{atk.name}」"
    if on_coin_heads:
        if _flip_coin():
            _apply_status(state, target_bp, status_effect, poison_damage=poison_damage, log_prefix=prefix + "コイン表 → ")
        else:
            state.log(prefix + "コイン裏 → 状態異常は付与されなかった")
    else:
        _apply_status(state, target_bp, status_effect, poison_damage=poison_damage, log_prefix=prefix + "→ ")


def attack(state: GameState, attack_index: int) -> bool:
    """
    攻撃を実行。ねむり・マヒ中は攻撃不可。こんらん時はコインで失敗時は自分に30ダメージ。
    相手にダメージ・状態異常、自分に self_damage を適用。
    """
    p = state.active_player_state()
    opp = state.defending_player_state()
    if not p.active or not opp.active:
        return False
    if getattr(p.active, "special_state", None) in ("sleep", "paralysis"):
        state.log(f"{state.player_name(state.current_player)}: {p.active.card.name} は状態異常のためワザが使えない")
        return False
    pokemon_card = p.active.card
    if attack_index < 0 or attack_index >= len(pokemon_card.attacks):
        return False
    atk = pokemon_card.attacks[attack_index]
    atk_key = _attack_key(pokemon_card, atk)
    if getattr(p.active, "disabled_attack_name", None) == atk.name:
        state.log(f"{state.player_name(state.current_player)}: {p.active.card.name} はこのターン「{atk.name}」が使えない")
        return False
    if atk_key in _TSUIGEKI_BARI_BARI:
        last_name = state.last_turn_attack_name[state.current_player]
        last_id = state.last_turn_attack_actor_id[state.current_player]
        actor_id = getattr(p.active.card, "id", getattr(p.active.card, "name", ""))
        if last_name != "しびれはり" or last_id != actor_id:
            state.log(f"{state.player_name(state.current_player)}: 「ついげきバリバリ」は前の自分の番にこのポケモンが「しびれはり」を使っていないと使えない")
            return False
    types = getattr(p.active, "attached_energy_types", [])
    if not _can_pay_energy_cost(
        p.active.attached_energy, types,
        atk.energy_cost, getattr(atk, "energy_cost_typed", None),
    ):
        return False
    if getattr(p.active, "special_state", None) == "confusion":
        if not _flip_coin():
            self_before = p.active.hp
            p.active.hp -= 30
            state.log(f"{state.player_name(state.current_player)}: こんらんでコインが裏 → 自分に 30 ダメージ（HP {self_before} → {max(0, p.active.hp)}）、攻撃失敗")
            if p.active and p.active.hp <= 0:
                koed_conf = p.active
                if _handle_own_active_ko(state, state.current_player, koed_conf, "こんらんの自傷"):
                    return True
                state._record_frame()
            return True
        state.log(f"{state.player_name(state.current_player)}: こんらんコイン：表 → 通常通り攻撃")
    opp_before = opp.active.hp
    coin_flips = getattr(atk, "coin_flips", 0)
    damage_per_coin = getattr(atk, "damage_per_coin", 0)
    if coin_flips > 0 and damage_per_coin > 0:
        heads = sum(1 for _ in range(coin_flips) if _flip_coin())
        damage = heads * damage_per_coin
        state.log(f"{state.player_name(state.current_player)}: 「{atk.name}」コイン {coin_flips} 回 → 表 {heads} 回、ダメージ {damage}")
    else:
        damage = atk.damage
    defender_card = opp.active.card
    attacker_card = pokemon_card
    if (
        getattr(defender_card, "weakness", None)
        and getattr(attacker_card, "pokemon_type", None)
        and defender_card.weakness == attacker_card.pokemon_type
    ):
        damage *= 2
        state.log(f"{state.player_name(state.current_player)}: 弱点一致！ダメージが 2 倍（{damage // 2} → {damage}）")
    if (
        getattr(defender_card, "resistance", None)
        and getattr(attacker_card, "pokemon_type", None)
        and defender_card.resistance == attacker_card.pokemon_type
    ):
        before_r = damage
        damage = max(0, damage - 30)
        state.log(f"{state.player_name(state.current_player)}: 抵抗力一致！ダメージが -30（{before_r} → {damage}）")
    tool = getattr(opp.active, "attached_tool", None)
    if tool and getattr(tool, "is_tool", False) and getattr(tool, "tool_damage_reduce", 0) > 0:
        cond = getattr(tool, "tool_condition_type", None)
        if cond is None or getattr(defender_card, "pokemon_type", None) == cond:
            before_t = damage
            damage = max(0, damage - getattr(tool, "tool_damage_reduce", 0))
            state.log(f"{state.player_name(state.current_player)}: {opp.active.card.name} の {tool.name} でダメージ -{before_t - damage}（{before_t} → {damage}）")
    if atk_key in _SHIPPEGAESHI_PRIZE_BONUS and len(opp.prize_pile) == 1:
        damage += 90
        state.log(f"{state.player_name(state.current_player)}: 「{atk.name}」で相手に {damage} ダメージ（相手サイド残り 1 枚のため 90 ダメージ追加、相手 HP {opp_before} → {max(0, opp_before - damage)}）")
    elif atk_key in _AVENGE_NAKKLE_KO_BONUS and state.our_ko_by_damage_last_turn[state.current_player]:
        damage += 120
        state.log(f"{state.player_name(state.current_player)}: 「{atk.name}」で相手に {damage} ダメージ（前の相手の番に自分の闘ポケモンがきぜつしたため 120 ダメージ追加、相手 HP {opp_before} → {max(0, opp_before - damage)}）")
    else:
        state.log(f"{state.player_name(state.current_player)}: 「{atk.name}」で相手に {damage} ダメージ（相手 HP {opp_before} → {max(0, opp_before - damage)}）")
    if getattr(p.active, "damage_reduction_next_turn", 0) > 0:
        red = p.active.damage_reduction_next_turn
        damage = max(0, damage - red)
        p.active.damage_reduction_next_turn = 0
        if red > 0:
            state.log(f"{state.player_name(state.current_player)}: 前の相手の番に受けた「なきごえ」のためダメージ -{red}（→ {damage}）")
    skip_opponent_status = False
    if getattr(opp.active, "protected_next_opponent_turn", False):
        opp.active.protected_next_opponent_turn = False
        damage = 0
        skip_opponent_status = True
        state.log(f"{state.player_name(state.current_player)}: 相手の {opp.active.card.name} は「なぐってかくれる」のためワザのダメージ・効果を受けない")
    opp.active.hp -= damage
    bench_dmg = getattr(atk, "bench_damage", 0)
    bench_count = getattr(atk, "bench_damage_count", 1)
    bench_target_self = getattr(atk, "bench_damage_target", "opponent") == "self"
    bench_list = p.bench if bench_target_self else opp.bench
    if bench_dmg > 0 and bench_list:
        n = len(bench_list) if bench_count == 0 else min(bench_count, len(bench_list))
        for i in range(n):
            bench_target = bench_list[i]
            bench_before = bench_target.hp
            bench_target.hp -= bench_dmg
            who = "自分のベンチの" if bench_target_self else f"相手のベンチの"
            state.log(f"{state.player_name(state.current_player)}: 「{atk.name}」で{who} {bench_target.card.name} に {bench_dmg} ダメージ（HP {bench_before} → {max(0, bench_target.hp)}）")
        for i in range(n - 1, -1, -1):
            if i < len(bench_list) and bench_list[i].hp <= 0:
                koed_bench_bp = bench_list[i]
                owner = p if bench_target_self else opp
                bench_list.pop(i)
                if _handle_bench_ko(state, owner, koed_bench_bp, bench_target_self, p, opp):
                    return True

    self_dmg = getattr(atk, "self_damage", 0)
    if self_dmg > 0 and p.active:
        self_before = p.active.hp
        p.active.hp -= self_dmg
        state.log(f"{state.player_name(state.current_player)}: 反動で自分に {self_dmg} ダメージ（自分 HP {self_before} → {max(0, p.active.hp)}）")

    target_bp = p.active if getattr(atk, "status_effect_target", "opponent") == "self" else (opp.active if opp.active else None)
    _apply_attack_status_effect(state, atk, target_bp, skip_opponent_status, opp)

    if atk_key in _NAGUTTE_KAKURERU and _flip_coin():
        p.active.protected_next_opponent_turn = True
        state.log(f"{state.player_name(state.current_player)}: 「{atk.name}」コイン表 → 次の相手の番、このポケモンはワザのダメージや効果を受けない")
    elif atk_key in _NAKIGOE_DAMAGE_REDUCTION and opp.active:
        opp.active.damage_reduction_next_turn = 20
        state.log(f"{state.player_name(state.current_player)}: 「{atk.name}」→ 次の相手の番、相手のワザのダメージが -20 される")
    if atk_key in _KASOKUZUKI_DISABLE:
        p.active.disabled_attack_name = "かそくづき"
        state.turn_when_disabled_attack[state.current_player] = state.turn_count
        state.log(f"{state.player_name(state.current_player)}: 次の番、このポケモンは「かそくづき」が使えなくなる")

    state.this_turn_attack_name = atk.name
    state.this_turn_attack_actor_id = getattr(p.active.card, "id", getattr(p.active.card, "name", ""))

    if atk_key in _FUKIARASU:
        n_hand = len(opp.hand)
        opp.deck.extend(opp.hand)
        opp.hand = []
        random.shuffle(opp.deck)
        opp_drawn = opp.draw(4)
        opp.hand = opp_drawn
        opp_drawn_names = ", ".join(_card_label(c) for c in opp_drawn)
        state.log(f"{state.player_name(state.current_player)}: 「{atk.name}」→ 相手は手札 {n_hand} 枚を山札にもどして切り、山札から 4 枚ドロー → [{opp_drawn_names}]（手札 {len(opp.hand)} 枚）")
    elif atk_key in _MAGNEJECT and opp.active and opp.bench:
        idx = random.randint(0, len(opp.bench) - 1)
        opp.active, opp.bench[idx] = opp.bench[idx], opp.active
        state.log(f"{state.player_name(state.current_player)}: 「{atk.name}」→ 相手のバトルポケモンとベンチを入れ替えた（{opp.active.card.name} がバトル場に）")
        # ベンチに移ったポケモンが HP 0 以下ならきぜつ処理
        for i in range(len(opp.bench) - 1, -1, -1):
            if i < len(opp.bench) and opp.bench[i].hp <= 0:
                koed_bench_bp = opp.bench[i]
                opp.bench.pop(i)
                if _handle_bench_ko(state, opp, koed_bench_bp, False, p, opp):
                    return True
    elif atk_key in _TOMODACHI_O_SAGASU:
        for i, c in enumerate(p.deck):
            if is_pokemon(c):
                p.deck.pop(i)
                p.hand.append(c)
                state.drawn_this_turn.append(c)
                random.shuffle(p.deck)
                state.log(f"{state.player_name(state.current_player)}: 「{atk.name}」→ 山札からポケモン 1 枚（{_card_label(c)}）を手札に加え、山札を切った")
                break
    elif atk_key in _QUICK_DRAW:
        drawn = p.draw(2)
        p.hand.extend(drawn)
        state.drawn_this_turn.extend(drawn)
        state.log(f"{state.player_name(state.current_player)}: 「{atk.name}」→ 山札から 2 枚ドロー → [{', '.join(_card_label(c) for c in drawn)}]（手札 {len(p.hand)} 枚）")
    elif atk_key in _EREKI_CHARGE and p.active:
        lightning_in_deck = [
            i for i, c in enumerate(p.deck)
            if is_energy(c) and getattr(c, "energy_type", None) == "lightning"
        ]
        to_attach_count = min(2, len(lightning_in_deck))
        attached = 0
        for idx in sorted(lightning_in_deck[:to_attach_count], reverse=True):
            c = p.deck.pop(idx)
            p.active.attached_energy += 1
            et = getattr(c, "energy_type", None) or "lightning"
            p.active.attached_energy_types.append(et)
            attached += 1
        if attached > 0:
            random.shuffle(p.deck)
            state.log(f"{state.player_name(state.current_player)}: 「{atk.name}」→ 山札から基本雷エネルギー {attached} 枚をこのポケモンにつけた")
    elif atk_key in _10MAN_VOLT and p.active:
        num = p.active.attached_energy
        types_to_discard = list(getattr(p.active, "attached_energy_types", []))
        _put_energy_cards_in_discard(p, types_to_discard, state)
        p.active.attached_energy = 0
        p.active.attached_energy_types = []
        state.log(f"{state.player_name(state.current_player)}: 「{atk.name}」→ このポケモンについているエネルギー {num} 個をすべてトラッシュした")
    elif atk_key in _GABUGABU_BITE and opp.active:
        heads = 0
        while _flip_coin():
            heads += 1
        discard_n = min(heads, opp.active.attached_energy)
        if discard_n > 0:
            types = getattr(opp.active, "attached_energy_types", [])
            discarded_types = types[-discard_n:] if len(types) >= discard_n else list(types)
            _put_energy_cards_in_discard(opp, discarded_types, state)
            opp.active.attached_energy -= discard_n
            if len(types) >= discard_n:
                opp.active.attached_energy_types = types[:-discard_n]
            else:
                opp.active.attached_energy_types = []
            state.log(f"{state.player_name(state.current_player)}: 「{atk.name}」→ コイン表 {heads} 回、相手のバトルポケモンのエネルギー {discard_n} 個をトラッシュした")
    elif atk_key in _TECHNO_TURBO and p.discard and p.bench:
        lightning_idx = None
        for i, c in enumerate(p.discard):
            if is_energy(c) and getattr(c, "energy_type", None) == "lightning":
                lightning_idx = i
                break
        if lightning_idx is not None:
            p.discard.pop(lightning_idx)
            bench_idx = min(range(len(p.bench)), key=lambda i: p.bench[i].attached_energy)
            target = p.bench[bench_idx]
            target.attached_energy += 1
            target.attached_energy_types.append("lightning")
            state.log(f"{state.player_name(state.current_player)}: 「{atk.name}」→ トラッシュから基本雷エネルギー 1 枚をベンチの {target.card.name} につけた")

    if opp.active and opp.active.hp <= 0:
        koed_active = opp.active
        # アベンジナックル用：きぜつしたのが闘ポケモンのときだけフラグを立てる
        if getattr(koed_active.card, "pokemon_type", None) == "fighting":
            state.our_ko_by_damage_last_turn[state.opponent()] = True
        state.log(f"バトル場の {koed_active.card.name} がきぜつ！（{opp.knockouts_suffered + 1} 回目）")
        if _handle_opponent_ko(opp, state, koed_active):
            return True

    if p.active and p.active.hp <= 0:
        koed_recoil = p.active
        if _handle_own_active_ko(state, state.current_player, koed_recoil, "反動"):
            return True
        state._record_frame()
    return True


def get_legal_attack_indices(state: GameState, p: PlayerState, opp: PlayerState) -> list[int]:
    """出せる技のインデックス一覧を返す（minimax 用）。"""
    if not p.active or not p.active.card.attacks:
        return []
    types = getattr(p.active, "attached_energy_types", [])
    legal = []
    for idx, atk in enumerate(p.active.card.attacks):
        if not _can_pay_energy_cost(
            p.active.attached_energy, types,
            atk.energy_cost, getattr(atk, "energy_cost_typed", None),
        ):
            continue
        if getattr(p.active, "disabled_attack_name", None) == atk.name:
            continue
        atk_key = _attack_key(p.active.card, atk)
        if atk_key in _TSUIGEKI_BARI_BARI:
            last_name = state.last_turn_attack_name[state.current_player]
            last_id = state.last_turn_attack_actor_id[state.current_player]
            actor_id = getattr(p.active.card, "id", getattr(p.active.card, "name", ""))
            if last_name != "しびれはり" or last_id != actor_id:
                continue
        base_dmg = _attack_damage_for_eval(atk)
        if atk_key in _SHIPPEGAESHI_PRIZE_BONUS and len(opp.prize_pile) == 1:
            base_dmg += 90
        if atk_key in _AVENGE_NAKKLE_KO_BONUS and state.our_ko_by_damage_last_turn[state.current_player]:
            base_dmg += 120
        effective_dmg = _effective_damage_to_defender(p.active.card, opp.active, base_dmg) if opp.active else base_dmg
        if opp.active and effective_dmg <= 0:
            has_merit = (
                atk_key in _ATTACK_HAS_MERIT_EFFECT
                or (
                    getattr(atk, "status_effect", None)
                    and getattr(atk, "status_effect_target", "opponent") != "self"
                )
                or (
                    getattr(atk, "bench_damage", 0) > 0
                    and getattr(atk, "bench_damage_target", "opponent") != "self"
                )
            )
            if not has_merit:
                continue
        legal.append(idx)
    return legal


def _choose_best_attack_index(state: GameState, p: PlayerState, opp: PlayerState) -> int | None:
    """出せる技のうち、相手をきぜつさせられる技を最優先し、否則有效ダメージが最大のインデックスを返す。"""
    if not p.active or not p.active.card.attacks:
        return None
    opp_hp = opp.active.hp if opp.active else 0
    best_idx = None
    best_score = -1
    types = getattr(p.active, "attached_energy_types", [])
    for idx, atk in enumerate(p.active.card.attacks):
        if not _can_pay_energy_cost(
            p.active.attached_energy, types,
            atk.energy_cost, getattr(atk, "energy_cost_typed", None),
        ):
            continue
        if getattr(p.active, "disabled_attack_name", None) == atk.name:
            continue
        atk_key = _attack_key(p.active.card, atk)
        if atk_key in _TSUIGEKI_BARI_BARI:
            last_name = state.last_turn_attack_name[state.current_player]
            last_id = state.last_turn_attack_actor_id[state.current_player]
            actor_id = getattr(p.active.card, "id", getattr(p.active.card, "name", ""))
            if last_name != "しびれはり" or last_id != actor_id:
                continue
        base_dmg = _attack_damage_for_eval(atk)
        if atk_key in _SHIPPEGAESHI_PRIZE_BONUS and len(opp.prize_pile) == 1:
            base_dmg += 90
        if atk_key in _AVENGE_NAKKLE_KO_BONUS and state.our_ko_by_damage_last_turn[state.current_player]:
            base_dmg += 120
        effective_dmg = _effective_damage_to_defender(p.active.card, opp.active, base_dmg) if opp.active else base_dmg
        if opp.active and effective_dmg <= 0:
            has_merit = (
                atk_key in _ATTACK_HAS_MERIT_EFFECT
                or (
                    getattr(atk, "status_effect", None)
                    and getattr(atk, "status_effect_target", "opponent") != "self"
                )
                or (
                    getattr(atk, "bench_damage", 0) > 0
                    and getattr(atk, "bench_damage_target", "opponent") != "self"
                )
            )
            if not has_merit:
                continue
        ko_bonus = _KO_BONUS_FOR_ATTACK if (opp.active and effective_dmg >= opp_hp) else 0
        score = effective_dmg + ko_bonus + get_attack_weight(state.get_weights_for_player(state.current_player), p.active.card, atk)
        if score > best_score:
            best_score = score
            best_idx = idx
    return best_idx
