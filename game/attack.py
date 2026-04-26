"""攻撃実行と最良技選択。"""
import random

from card import is_pokemon, is_energy

from .damage import (
    _attack_damage_for_eval,
    _bench_has_lunatone,
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
    mark_own_deck_shuffled,
    _log_prize_count_for_ko,
)
from .weights import get_attack_weight

_KO_BONUS_FOR_ATTACK = 10000
_ATTACK_BIAS = 5


def _attack_key(card, atk) -> tuple[str, str, str]:
    """(カード名, レギュレーション, 技名)。同じ技名でもカード・レギュで効果が違う場合の分岐用。"""
    name = getattr(card, "name", "")
    reg = getattr(card, "regulation", None) or ""
    return (name, reg, atk.name)


_TSUIGEKI_BARI_BARI = frozenset({("バチンウニ", "G", "ついげきバリバリ")})
_SHIPPEGAESHI_PRIZE_BONUS = frozenset({("ワルビル", "G", "しっぺがえし"), ("ワルビル", "", "しっぺがえし")})
_AVENGE_NAKKLE_KO_BONUS = frozenset({("ルカリオ", "G", "アベンジナックル")})
_NAGUTTE_KAKURERU = frozenset({("ウソッキー", "G", "なぐってかくれる")})
_NAKIGOE_DAMAGE_REDUCTION = frozenset({("ピカチュウ", "G", "なきごえ")})
_KASOKUZUKI_DISABLE = frozenset({("ルカリオ", "G", "かそくづき")})
_MEGABRAVE_DISABLE = frozenset({
    ("メガルカリオ", "G", "メガブレイブ"),
    ("メガルカリオex", "G", "メガブレイブ"),
})
_FUKIARASU = frozenset({("カイデン", "G", "ふきあらす")})
_MAGNEJECT = frozenset({("ジバコイル", "G", "マグネリジェクト")})
_TOMODACHI_O_SAGASU = frozenset({("ノコッチ", "G", "ともだちをさがす")})
_QUICK_DRAW = frozenset({("ミライドンex", "G", "クイックドロー")})
_EREKI_CHARGE = frozenset({("ライチュウ", "G", "エレキチャージ")})
_10MAN_VOLT = frozenset({("ライチュウ", "G", "10まんボルト")})
_GABUGABU_BITE = frozenset({("ワルビアル", "G", "ガブガブバイト")})
_TECHNO_TURBO = frozenset({("ミライドンex", "G", "テクノターボ")})
_HADOUZUKI = frozenset({("メガルカリオ", "G", "はどうづき"), ("メガルカリオex", "G", "はどうづき")})
_PHANTOM_DIVE = frozenset({("ドラパルトex", "H", "ファントムダイブ")})
_CRUEL_ARROW = frozenset({("キチキギスex", "H", "クルーエルアロー")})
_MUKAENIIKU = frozenset({("ヨマワル", "H", "むかえにいく")})
_MUZUMUZU_KAFUN = frozenset({("スボミー", "H", "むずむずかふん")})
_KAGESHIBARI = frozenset({("ヨノワール", "H", "かげしばり")})

_ATTACK_HAS_MERIT_EFFECT = (
    _NAGUTTE_KAKURERU
    | _NAKIGOE_DAMAGE_REDUCTION
    | _FUKIARASU
    | _MAGNEJECT
    | _TOMODACHI_O_SAGASU
    | _HADOUZUKI
    | _QUICK_DRAW
    | _EREKI_CHARGE
    | _GABUGABU_BITE
    | _TECHNO_TURBO
    | _CRUEL_ARROW
    | _MUKAENIIKU
    | _MUZUMUZU_KAFUN
    | _KAGESHIBARI
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
        # 汎用 KO 追跡（相手の攻撃で自分のベンチがきぜつ）
        if not bench_target_self:
            if not hasattr(state, "any_ko_by_opponent_last_turn"):
                state.any_ko_by_opponent_last_turn = [False, False]
            state.any_ko_by_opponent_last_turn[state.opponent()] = True
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
    if atk.name == "コスモビーム":
        if not _bench_has_lunatone(p):
            state.log(
                f"{state.player_name(state.current_player)}: 「{atk.name}」→ "
                "自分のベンチにルナトーンがいないためワザは失敗"
            )
            return True
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
    if not getattr(atk, "damage_ignores_weakness_resistance", False):
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
    # ジャミングタワー: どうぐ付きポケモンの効果は無効
    _jamming = False
    _stadium = getattr(state, "stadium", None)
    if _stadium is not None:
        _sid = (getattr(_stadium, "id", "") or "").strip()
        _sname = (getattr(_stadium, "name", "") or "").strip()
        _jamming = _sid == "jixyamingutawa" or "ジャミングタワー" in _sname
    if tool and getattr(tool, "is_tool", False) and getattr(tool, "tool_damage_reduce", 0) > 0 and not _jamming:
        cond = getattr(tool, "tool_condition_type", None)
        if cond is None or getattr(defender_card, "pokemon_type", None) == cond:
            before_t = damage
            damage = max(0, damage - getattr(tool, "tool_damage_reduce", 0))
            state.log(f"{state.player_name(state.current_player)}: {opp.active.card.name} の {tool.name} でダメージ -{before_t - damage}（{before_t} → {damage}）")
    # マキシマムベルト: 自分のポケモンにつけていて、相手がexなら+50（ジャミングタワーで無効）
    atk_tool = getattr(p.active, "attached_tool", None)
    if atk_tool and (getattr(atk_tool, "id", "") or "") == "makishimamuberuto" and not _jamming:
        if getattr(defender_card, "is_ex", False) or ("ex" in (getattr(defender_card, "name", "") or "")):
            before_belt = damage
            damage += 50
            state.log(f"{state.player_name(state.current_player)}: マキシマムベルトの効果でダメージ +50（{before_belt} → {damage}）")
    n_plus30 = getattr(state, "fighting_damage_plus_30_count_this_turn", 0)
    if n_plus30 > 0 and getattr(attacker_card, "pokemon_type", None) == "fighting":
        damage += 30 * n_plus30
        state.log(f"{state.player_name(state.current_player)}: パワープロテインの効果でダメージ +{30 * n_plus30}（{damage - 30 * n_plus30} → {damage}）")
    # ブライア: テラスタルポケモンのKOでサイド+1（briar_extra_prize で処理、ここではダメージ加算しない）
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
    if atk_key in _MEGABRAVE_DISABLE:
        p.active.disabled_attack_name = "メガブレイブ"
        state.turn_when_disabled_attack[state.current_player] = state.turn_count
        state.log(f"{state.player_name(state.current_player)}: 次の番、このポケモンは「メガブレイブ」が使えない")

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
                mark_own_deck_shuffled(state)
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
            mark_own_deck_shuffled(state)
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
    elif atk_key in _HADOUZUKI and p.discard and p.bench:
        def _is_fighting_energy(c):
            if not is_energy(c):
                return False
            return getattr(c, "energy_type", None) == "fighting"
        def _best_attack_damage_and_attack(pokemon_card):
            best_dmg = 0
            best_atk = None
            for atk in getattr(pokemon_card, "attacks", []) or []:
                d = _attack_damage_for_eval(atk)
                if d > best_dmg:
                    best_dmg = d
                    best_atk = atk
            return best_dmg, best_atk
        fighting_in_discard = [c for c in p.discard if _is_fighting_energy(c)][:3]
        attached_log = []
        for card in fighting_in_discard:
            p.discard.remove(card)
            best_idx = None
            best_score = -1
            for i, bp in enumerate(p.bench):
                best_dmg, best_atk = _best_attack_damage_and_attack(bp.card)
                if best_atk is None:
                    score = 0
                else:
                    types = list(getattr(bp, "attached_energy_types", []) or [])
                    can_use_best = _can_pay_energy_cost(
                        bp.attached_energy, types,
                        best_atk.energy_cost, getattr(best_atk, "energy_cost_typed", None),
                    )
                    score = best_dmg if not can_use_best else 0
                # マクノシタ/ハリテヤマ育成ボーナス:
                # はどうづきで闘エネ3枚をマクノシタに集中させ、
                # 進化してワイルドプレス（210ダメージ）ですぐ殴れるようにする。
                # マクノシタは手張りで1枚目を付けないルールなので、はどうづきでの加速が重要。
                bp_name = (getattr(bp.card, "name", "") or "").strip()
                if bp_name == "マクノシタ":
                    max_e = max((a.energy_cost for a in bp.card.attacks), default=0)
                    if bp.attached_energy < max_e:
                        score += 300  # ハリテヤマ育成優先
                # リオル/メガルカリオex: 次のメガブレイブ用にエネを貯める
                elif bp_name in ("リオル", "メガルカリオex", "メガルカリオ"):
                    max_e = max((a.energy_cost for a in bp.card.attacks), default=0)
                    if bp.attached_energy < max_e:
                        score += 200  # 2体目のメガルカリオ育成
                # 同名ポケモンが複��いる場合、HP高い方を優先
                # 手負いのポケモンは倒されやすいので無傷の方に投資
                _same_name_higher_hp = any(
                    bp2 is not bp
                    and (getattr(bp2.card, "name", "") or "").strip() == bp_name
                    and (getattr(bp2, "hp", 0) or 0) > (getattr(bp, "hp", 0) or 0)
                    for bp2 in p.bench
                )
                if _same_name_higher_hp:
                    score -= 500  # HPが低い方にはエネを付けない
                if score > best_score:
                    best_score = score
                    best_idx = i
            if best_idx is None:
                best_idx = 0
            bp = p.bench[best_idx]
            bp.attached_energy += 1
            bp.attached_energy_types = getattr(bp, "attached_energy_types", []) or []
            bp.attached_energy_types.append("fighting")
            attached_log.append(bp.card.name)
        if attached_log:
            state.log(
                f"{state.player_name(state.current_player)}: 「{atk.name}」→ トラッシュから基本闘エネルギー {len(attached_log)} 枚をベンチに付けた（{', '.join(attached_log)}）"
            )

    # ドラパルトex: ファントムダイブ（200 ダメージ + ベンチにダメカン 6 個 = 60 ダメージを分配）
    if atk_key in _PHANTOM_DIVE and opp.bench and not skip_opponent_status:
        from .evolution import _distribute_damage_counters
        _distribute_damage_counters(state, opp, 60, pokemon_card.name, "ファントムダイブ", bench_only=True)

    # クルーエルアロー: 相手のポケモン1匹に100ダメージ（バトル場 or ベンチ）。
    # 通常攻撃のダメージはバトル場に行くが、クルーエルアローは任意の1匹を選ぶ。
    # ここではバトル場へのダメージを取り消し、最適なターゲットに100ダメージを与える。
    if atk_key in _CRUEL_ARROW and not skip_opponent_status:
        # バトル場へのダメージを取り消す（すでにopp.active.hpに反映済みなので復元）
        if opp.active:
            opp.active.hp = opp_before
        # ターゲットを選ぶ: KO可能なら最優先（サイド枚数で重み付け）
        cruel_targets = []
        if opp.active and opp.active.hp > 0:
            cruel_targets.append(("active", opp.active))
        for ci, cbp in enumerate(opp.bench):
            if cbp and cbp.hp and cbp.hp > 0:
                cruel_targets.append((f"bench:{ci}", cbp))
        if cruel_targets:
            best_ct_key, best_ct = None, None
            best_ct_score = -1
            for ckey, cbp in cruel_targets:
                cscore = 0
                if cbp.hp <= 100:
                    cscore = 10000 + _prizes_for_ko(cbp) * 1000
                else:
                    cscore = 100  # ダメージを与える意味
                if cscore > best_ct_score:
                    best_ct_score = cscore
                    best_ct_key = ckey
                    best_ct = cbp
            if best_ct:
                ct_before = best_ct.hp
                best_ct.hp -= 100
                ct_loc = "バトル場の" if best_ct_key == "active" else "ベンチの"
                state.log(
                    f"{state.player_name(state.current_player)}: 「クルーエルアロー」→ "
                    f"相手の{ct_loc}{best_ct.card.name} に 100 ダメージ（HP {ct_before} → {max(0, best_ct.hp)}）"
                )
                # ベンチの KO 処理
                if best_ct.hp <= 0 and best_ct_key != "active":
                    for ci2 in range(len(opp.bench) - 1, -1, -1):
                        if ci2 < len(opp.bench) and opp.bench[ci2] is best_ct:
                            opp.bench.pop(ci2)
                            state.log(f"相手のベンチの {best_ct.card.name} がきぜつ！")
                            _handle_opponent_ko(opp, state, best_ct)
                            break

    # むかえにいく: 自分のトラッシュから「ヨマワル」を3枚までベンチに出す
    if atk_key in _MUKAENIIKU:
        from .state import BENCH_SIZE, BattlePokemon as _BP
        yomawaru_in_discard = [c for c in p.discard if is_pokemon(c) and (getattr(c, "name", "") or "").strip() == "ヨマワル"]
        placed = 0
        for yc in yomawaru_in_discard[:3]:
            if len(p.bench) >= BENCH_SIZE:
                break
            p.discard.remove(yc)
            bp = _BP(card=yc.copy())
            bp.put_on_bench_this_turn = True
            p.bench.append(bp)
            placed += 1
        if placed > 0:
            state.log(
                f"{state.player_name(state.current_player)}: 「むかえにいく」→ トラッシュからヨマワル {placed} 枚をベンチに出した"
            )

    # むずむずかふん: 次の相手の番、相手は手札からグッズを出して使えない
    if atk_key in _MUZUMUZU_KAFUN and not skip_opponent_status:
        state.goods_locked_next_turn = state.opponent()
        state.log(
            f"{state.player_name(state.current_player)}: 「むずむずかふん」→ 次の相手の番、相手は手札からグッズを使えない"
        )

    # かげしばり: 次の相手の番、このワザを受けたポケモンは、にげられない
    if atk_key in _KAGESHIBARI and opp.active and not skip_opponent_status:
        opp.active.retreat_locked = True
        state.log(
            f"{state.player_name(state.current_player)}: 「かげしばり」→ 相手の {opp.active.card.name} は次の相手の番にげられない"
        )

    if opp.active and opp.active.hp <= 0:
        koed_active = opp.active
        if getattr(koed_active.card, "pokemon_type", None) == "fighting":
            state.our_ko_by_damage_last_turn[state.opponent()] = True
        # 汎用 KO 追跡（キチキギスex の「さかてにとる」等で使用）
        if not hasattr(state, "any_ko_by_opponent_last_turn"):
            state.any_ko_by_opponent_last_turn = [False, False]
        state.any_ko_by_opponent_last_turn[state.opponent()] = True
        state.log(f"バトル場の {koed_active.card.name} がきぜつ！（{opp.knockouts_suffered + 1} 回目）")
        if _handle_opponent_ko(opp, state, koed_active):
            return True

    if p.active and p.active.hp <= 0:
        koed_recoil = p.active
        if _handle_own_active_ko(state, state.current_player, koed_recoil, "反動"):
            return True
        state._record_frame()
    return True


def get_legal_attack_indices_for_attacker(
    state: GameState, p: PlayerState, opp: PlayerState, attacker: BattlePokemon
) -> list[int]:
    """attacker をバトル場にいるとみなした合法技（いれかえ判断用。p.active は書き換えない）。"""
    if not attacker or not attacker.card or not attacker.card.attacks:
        return []
    types = getattr(attacker, "attached_energy_types", [])
    legal: list[int] = []
    for idx, atk in enumerate(attacker.card.attacks):
        if not _can_pay_energy_cost(
            attacker.attached_energy, types,
            atk.energy_cost, getattr(atk, "energy_cost_typed", None),
        ):
            continue
        if getattr(attacker, "disabled_attack_name", None) == atk.name:
            continue
        atk_key = _attack_key(attacker.card, atk)
        if atk_key in _TSUIGEKI_BARI_BARI:
            last_name = state.last_turn_attack_name[state.current_player]
            last_id = state.last_turn_attack_actor_id[state.current_player]
            actor_id = getattr(attacker.card, "id", getattr(attacker.card, "name", ""))
            if last_name != "しびれはり" or last_id != actor_id:
                continue
        if atk.name == "コスモビーム" and not _bench_has_lunatone(p):
            continue
        base_dmg = _attack_damage_for_eval(atk)
        if atk_key in _SHIPPEGAESHI_PRIZE_BONUS and len(opp.prize_pile) == 1:
            base_dmg += 90
        if atk_key in _AVENGE_NAKKLE_KO_BONUS and state.our_ko_by_damage_last_turn[state.current_player]:
            base_dmg += 120
        ign = getattr(atk, "damage_ignores_weakness_resistance", False)
        effective_dmg = (
            _effective_damage_to_defender(
                attacker.card, opp.active, base_dmg, state=state, attacker_bp=attacker, ignore_weakness_resistance=ign
            )
            if opp.active
            else base_dmg
        )
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


def get_legal_attack_indices(state: GameState, p: PlayerState, opp: PlayerState) -> list[int]:
    """出せる技のインデックス一覧を返す（minimax 用）。"""
    if not p.active:
        return []
    return get_legal_attack_indices_for_attacker(state, p, opp, p.active)


def _choose_best_attack_index(state: GameState, p: PlayerState, opp: PlayerState) -> int | None:
    """出せる技のうち、相手をきぜつさせられる技を最優先し、否則有效ダメージが最大のインデックスを返す。出せる技が 1 本でもあれば必ずどれか選ぶ（0 ダメージでも攻撃する）。"""
    if not p.active or not p.active.card.attacks:
        return None
    opp_hp = opp.active.hp if opp.active else 0
    best_idx = None
    best_score = -1
    first_legal_idx = None
    types = getattr(p.active, "attached_energy_types", [])
    for idx, atk in enumerate(p.active.card.attacks):
        if not _can_pay_energy_cost(
            p.active.attached_energy, types,
            atk.energy_cost, getattr(atk, "energy_cost_typed", None),
        ):
            continue
        if getattr(p.active, "disabled_attack_name", None) == atk.name:
            continue
        if first_legal_idx is None:
            first_legal_idx = idx
        atk_key = _attack_key(p.active.card, atk)
        if atk_key in _TSUIGEKI_BARI_BARI:
            last_name = state.last_turn_attack_name[state.current_player]
            last_id = state.last_turn_attack_actor_id[state.current_player]
            actor_id = getattr(p.active.card, "id", getattr(p.active.card, "name", ""))
            if last_name != "しびれはり" or last_id != actor_id:
                continue
        if atk.name == "コスモビーム" and not _bench_has_lunatone(p):
            continue
        base_dmg = _attack_damage_for_eval(atk)
        if atk_key in _SHIPPEGAESHI_PRIZE_BONUS and len(opp.prize_pile) == 1:
            base_dmg += 90
        if atk_key in _AVENGE_NAKKLE_KO_BONUS and state.our_ko_by_damage_last_turn[state.current_player]:
            base_dmg += 120
        ign = getattr(atk, "damage_ignores_weakness_resistance", False)
        effective_dmg = (
            _effective_damage_to_defender(
                p.active.card, opp.active, base_dmg, state=state, attacker_bp=p.active, ignore_weakness_resistance=ign
            )
            if opp.active
            else base_dmg
        )
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
        # Fix H: 自傷で自分が倒れる技は、相手を倒せない限り大ペナルティ
        self_dmg = getattr(atk, "self_damage", 0)
        self_ko_penalty = 0
        if self_dmg > 0 and p.active.hp <= self_dmg and not (opp.active and effective_dmg >= opp_hp):
            self_ko_penalty = -100000
        _bench_spread_bonus = 0
        score = effective_dmg + ko_bonus + _ATTACK_BIAS + self_ko_penalty + _bench_spread_bonus + get_attack_weight(state.get_weights_for_player(state.current_player), p.active.card, atk)
        if score > best_score:
            best_score = score
            best_idx = idx
    if best_idx is None and first_legal_idx is not None:
        return first_legal_idx
    return best_idx
