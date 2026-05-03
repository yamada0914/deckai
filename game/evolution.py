"""進化（進化適用・進化可否判定・evolve_pokemon）。進化したとき 1 回使える特性のトリガーも扱う。"""
import random

from card import PokemonCard, is_pokemon

from .damage import _max_effective_damage_for_attacker
from .state import (
    GameState,
    PlayerState,
    BattlePokemon,
    _card_label,
    _clear_status,
    _handle_opponent_ko,
    _prizes_for_ko,
    _put_energy_cards_in_discard,
    get_effective_max_hp,
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
    # グラビティーマウンテン等のスタジアム効果を反映した実効max_hp
    eff_max_hp = get_effective_max_hp(state, evolved)
    target.card.max_hp = evolved.max_hp  # カード本来のmax_hpは保持
    target.card.hp = max(0, min(eff_max_hp, eff_max_hp - damage_taken))
    target.hp = target.card.hp
    target.attached_energy = old_energy
    target.attached_energy_types = list(old_energy_types)
    target.evolved_this_turn = True
    _clear_status(target)
    state.log(f"{log_prefix}{old_name} を {evolved.name} に進化（HP {target.hp}/{target.max_hp}）")


def _trigger_ability_when_evolved(state: GameState, target: BattlePokemon) -> None:
    """
    進化したポケモンの「自分の番に手札から出して進化させたとき 1 回使える」特性を発動する。
    例: ハリテヤマのどすこいキャッチャー（相手のベンチ 1 匹をバトルポケモンと入れ替え）。
    """
    if state.ability_used_this_turn:
        return
    ability_name = (getattr(target.card, "ability_name", None) or "").strip()
    if not ability_name:
        return
    if ability_name == "どすこいキャッチャー":
        opp = state.defending_player_state()
        if not opp.bench or not opp.active:
            return
        p = state.active_player_state()
        # 引っ張る対象を選ぶ（メガルカリオexデッキ戦略準拠）:
        # 1. 倒せる相手を最優先（サイド枚数で重み付け: ex=2, メガ=3）
        # 2. 倒せない場合はサイド効率（ex/メガ）＋低HP＋エネ/進化済み（リソース浪費）
        # 3. キチキギスex・ラティアスex等のサポート特性持ちを狙う（妨害にもなる）
        _ABILITY_DENY_NAMES = frozenset({
            "キチキギスex", "ラティアスex", "ニャースex",
        })

        def _dosukoi_score(bp):
            prizes = _prizes_for_ko(bp)
            hp = bp.hp
            can_ko = False
            if p.active and hp is not None and hp > 0:
                dmg = _max_effective_damage_for_attacker(state, p.active, bp, state.current_player)
                can_ko = dmg >= hp
            # KO可能なら大ボーナス（サイド枚数で重み付け）
            ko_bonus = 50000 * prizes if can_ko else 0
            # サイド効率: ex/メガは高価値
            prize_bonus = prizes * 1000 if can_ko else 0
            # 倒しやすさ（HP低いほど高い）
            hp_bonus = 500 - min(hp or 0, 500)
            # エネルギー: KOできるなら浪費ボーナス、KOできないならペナルティ
            # KOできないアタッカー（エネ充実）を引っ張ると次ターン攻撃される
            energy_attached = getattr(bp, "attached_energy", 0) or 0
            is_evolved = bool(getattr(bp.card, "evolves_from", None))
            if can_ko:
                resource_bonus = energy_attached * 200 + (300 if is_evolved else 0)
            else:
                # KO不可でもエネ多い方を優先（同名ポケモンならリソース削り）
                retreat_cost = getattr(bp.card, "retreat_cost", 1) or 1
                resource_bonus = energy_attached * 300 + retreat_cost * 100
            # サポート特性持ちを狙う（特性封じ＋サイド取得で一石二鳥）
            name = (getattr(bp.card, "name", "") or "").strip()
            ability_deny_bonus = 2000 if name in _ABILITY_DENY_NAMES else 0
            return ko_bonus + prize_bonus + hp_bonus + resource_bonus + ability_deny_bonus
        bench_idx = max(range(len(opp.bench)), key=lambda i: _dosukoi_score(opp.bench[i]))
        opp.active, opp.bench[bench_idx] = opp.bench[bench_idx], opp.active
        state.ability_used_this_turn = True
        state.log(
            f"{state.player_name(state.current_player)}: {target.card.name} の特性「どすこいキャッチャー」→ "
            f"相手のベンチとバトルポケモンを入れ替えた（{opp.active.card.name} がバトル場に）"
        )

    # ドロンチ: ていさつしれい は進化時トリガーではない（自分の番に1回使える能動特性）。
    # _try_use_ability_teisatsushirei() で処理するため、ここでは何もしない。

    # サマヨール / ヨノワール: カースドボム は自分の番に1回使える「能動特性」。
    # 進化時に自動発動するのではなく、_try_use_ability_cursed_bomb() で戦略的に使う。
    # ここでは何もしない。


def _place_damage_counters_on_one(
    state: GameState, opp, total_damage: int, source_name: str, ability_name: str,
) -> None:
    """
    ダメカンを相手のポケモン 1 匹にすべてのせる。
    戦略: KO できるポケモンを最優先（サイド枚数で重み付け）。KO できなければ HP が低いポケモンを狙う。
    """
    targets = []
    if opp.active and opp.active.hp and opp.active.hp > 0:
        targets.append(("active", opp.active))
    for i, bp in enumerate(opp.bench):
        if bp and bp.hp and bp.hp > 0:
            targets.append((f"bench:{i}", bp))
    if not targets:
        return
    # KO 可能なターゲットを優先（サイド枚数で重み付け）
    best_key, best_bp = None, None
    best_score = -1
    for key, bp in targets:
        score = 0
        if bp.hp <= total_damage:
            # KO 可能 → 大ボーナス + サイド枚数
            score = 10000 + _prizes_for_ko(bp) * 1000
        else:
            # KO 不可 → HP が低いほど高スコア（次のカースドボムで倒しやすくする）
            score = 500 - min(bp.hp, 500)
        if score > best_score:
            best_score = score
            best_key = key
            best_bp = bp
    if best_bp is None:
        return
    before = best_bp.hp
    best_bp.hp -= total_damage
    loc = "バトル場の" if best_key == "active" else "ベンチの"
    state.log(
        f"{state.player_name(state.current_player)}: {source_name} の特性「{ability_name}」→ "
        f"相手の{loc}{best_bp.card.name} にダメカン {total_damage // 10} 個（{total_damage} ダメージ、HP {before} → {max(0, best_bp.hp)}���"
    )
    # KO 処理
    if best_bp.hp <= 0:
        if best_key == "active":
            state.log(f"相手のバトル場の {best_bp.card.name} がきぜつ！")
            _handle_opponent_ko(opp, state, best_bp)
        else:
            for i in range(len(opp.bench) - 1, -1, -1):
                if i < len(opp.bench) and opp.bench[i] is best_bp:
                    opp.bench.pop(i)
                    state.log(f"相手��ベンチの {best_bp.card.name} がきぜ��！")
                    _handle_opponent_ko(opp, state, best_bp)
                    break


def _distribute_damage_counters(
    state: GameState, opp, total_damage: int, source_name: str, ability_name: str,
    *, bench_only: bool = False,
) -> None:
    """
    ダメカンを相手のポケモンに好きなように分配する。
    戦略: HP が低いポケモンを優先的に倒す（KO 優先）。残りは最大効果になるよう分配。
    bench_only=True の場合ベンチのみ対象（ファントムダイブ用）。
    ファントムダイブ向け: カースドボム（50/130）で倒せるラインにダメージを蓄積する戦略。
    """
    targets = []
    if not bench_only and opp.active and opp.active.hp and opp.active.hp > 0:
        targets.append(("active", opp.active))
    for i, bp in enumerate(opp.bench):
        if bp and bp.hp and bp.hp > 0:
            targets.append((f"bench:{i}", bp))
    if not targets:
        return
    remaining = total_damage
    damage_map: dict[str, int] = {key: 0 for key, _ in targets}

    # ファントムダイブ向け: カースドボム（50/130）で倒せるラインに蓄積する戦略
    # 次のターンにカースドボム50で倒せる = HP <= 50 にする
    # 次のターンにカースドボム130で倒せる = HP <= 130 にする
    # ファントムダイブ自体の60 + カースドボム130 = 190, 60 + 50 = 110
    _is_phantom_dive = ability_name == "ファントムダイブ"

    if _is_phantom_dive and len(targets) > 1:
        # 戦略: まずKO可能なターゲットを倒す。残りは「次のカースドボムで倒せるライン」に蓄積
        # KO可能なターゲット（サイド枚数で重み付け）
        ko_cands = [(key, bp) for key, bp in targets if bp.hp <= remaining]
        ko_cands.sort(key=lambda x: (-_prizes_for_ko(x[1]), x[1].hp))
        for key, bp in ko_cands:
            if remaining <= 0:
                break
            needed = bp.hp
            if needed <= remaining:
                damage_map[key] = needed
                remaining -= needed

        # 残りダメージを「カースドボムで倒せるライン」に蓄積
        if remaining > 0:
            # カースドボム50で倒せる状態にする = HP を 50 以下にする
            # カースドボム130で倒せる状態にする = HP を 130 以下にする
            _CURSED_BOMB_THRESHOLDS = [50, 130]
            non_ko = [(key, bp) for key, bp in targets if damage_map[key] == 0]
            # 最もダメージ効率が良いターゲットをスコアリング
            scored = []
            for key, bp in non_ko:
                hp = bp.hp
                prize = _prizes_for_ko(bp)
                best_alloc = min(remaining, hp)
                score = 0
                # このダメージでカースドボム50圏内に入るか
                if hp - best_alloc <= 50 and hp > 50:
                    score = 3000 + prize * 500
                # カースドボム130圏内に入るか
                elif hp - best_alloc <= 130 and hp > 130:
                    score = 2000 + prize * 500
                # 既にカースドボム圏内ならダメージ不要
                elif hp <= 50:
                    score = -100
                elif hp <= 130:
                    score = -50
                else:
                    # 圏内に入れられなくてもダメージを蓄積
                    score = 1000 + prize * 200
                scored.append((key, bp, score, best_alloc))
            scored.sort(key=lambda x: -x[2])
            for key, bp, _sc, _alloc in scored:
                if remaining <= 0:
                    break
                hp = bp.hp
                # カースドボム圏内にするために必要なダメージ量を計算
                for threshold in _CURSED_BOMB_THRESHOLDS:
                    needed_for_threshold = hp - threshold
                    if needed_for_threshold > 0 and needed_for_threshold <= remaining:
                        damage_map[key] += needed_for_threshold
                        remaining -= needed_for_threshold
                        break
                else:
                    # 閾値に達しなくても残りを分配
                    alloc = min(remaining, hp)
                    if alloc > 0:
                        damage_map[key] += alloc
                        remaining -= alloc

        # まだ残っていれば最初のターゲットに全て
        if remaining > 0 and targets:
            damage_map[targets[0][0]] += remaining
    else:
        # 旧ロジック（カースドボム等の汎用分配）
        # パス 1: KO 可能なポケモンに必要最小限のダメージを割り当て（HP が低い順）
        ko_targets = [(key, bp) for key, bp in targets if bp.hp <= remaining]
        ko_targets.sort(key=lambda x: x[1].hp)
        for key, bp in ko_targets:
            if remaining <= 0:
                break
            needed = bp.hp
            if needed <= remaining:
                damage_map[key] = needed
                remaining -= needed

        # パス 2: 残りをまだダメージを受けていないポケモンに分配（HP が低い順で KO を狙う）
        if remaining > 0:
            non_ko = [(key, bp) for key, bp in targets if damage_map[key] == 0]
            non_ko.sort(key=lambda x: x[1].hp)
            for key, bp in non_ko:
                if remaining <= 0:
                    break
                alloc = min(remaining, bp.hp)
                damage_map[key] += alloc
                remaining -= alloc

        # パス 3: まだ残っていれば最初のターゲットに全て
        if remaining > 0 and targets:
            damage_map[targets[0][0]] += remaining

    # ダメージ適用
    for key, bp in targets:
        dmg = damage_map.get(key, 0)
        if dmg > 0:
            before = bp.hp
            bp.hp -= dmg
            loc = "バトル場の" if key == "active" else "ベンチの"
            state.log(
                f"{state.player_name(state.current_player)}: {source_name} の特性「{ability_name}」→ "
                f"相手の{loc}{bp.card.name} にダメカン {dmg // 10} 個（{dmg} ダメージ、HP {before} → {max(0, bp.hp)}）"
            )

    # KO 処理（ベンチから逆順で処理）
    for i in range(len(opp.bench) - 1, -1, -1):
        if i < len(opp.bench) and opp.bench[i].hp <= 0:
            koed_bp = opp.bench[i]
            opp.bench.pop(i)
            state.log(f"相手のベンチの {koed_bp.card.name} がきぜつ！")
            _handle_opponent_ko(opp, state, koed_bp)
    if opp.active and opp.active.hp <= 0:
        koed_active = opp.active
        state.log(f"相手のバトル場の {koed_active.card.name} がきぜつ！")
        _handle_opponent_ko(opp, state, koed_active)


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

    # ドラパルトex進化前にていさつしれいを使い切るガード
    # 場にていさつしれい未使用のドロンチがいればドラパルトex進化をブロック
    evo_name = (getattr(evolution_card, "name", "") or "").strip()
    if evo_name == "ドラパルトex":
        from .deck_strategies import is_dragapult_deck_for_player
        if is_dragapult_deck_for_player(state, state.current_player):
            _used_ids = getattr(state, "_teisatsushirei_used_ids_this_turn", set())
            _all_field = ([p.active] if p.active else []) + list(p.bench or [])
            _any_doronchi_unused = any(
                (getattr(bp.card, "name", "") or "").strip() == "ドロンチ"
                and id(bp) not in _used_ids
                for bp in _all_field
            )
            # デッキ10枚以下は除外（デッキ切れリスク）
            if _any_doronchi_unused and len(p.deck) > 10:
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
        _trigger_ability_when_evolved(state, p.active)
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
    _trigger_ability_when_evolved(state, bench_pokemon)
    return True
