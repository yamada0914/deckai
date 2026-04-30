"""ターン実行用：エネルギー付与・進化の試行。"""
from card import is_energy, is_goods, is_pokemon, is_support

from .damage import _max_effective_damage_if_attach
from .damage import _max_effective_damage_for_attacker
from .evolution import _can_evolve_onto, evolve_pokemon
from .state import (
    GameState,
    PlayerState,
    MAX_EVOLVE_ROUNDS_PER_TURN,
    _can_pay_energy_cost,
    _is_first_player_first_turn,
    _log_choice,
    rules_only_for_player,
)
from .trainers import attach_energy
from .turn_trainers import _SUPPORT_IDS_TRASH_WHOLE_HAND
from .weights import get_evolve_onto_weight
from .policy_rules_only import pick_energy_attach_candidate
from .energy_policy import pick_energy_attach_by_policy


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

    # ハリテヤマ進化の抑制:
    # - 相手バトル場に「高価値で倒せるカード」がいるなら進化しない
    # - 相手ベンチに「高価値で倒せるカード」がいるときだけ進化する
    opp = state.defending_player_state()

    def _is_high_value_target(bp) -> bool:
        card = getattr(bp, "card", None)
        if not card:
            return False
        name = (getattr(card, "name", "") or "").lower()
        is_ex = bool(getattr(card, "is_ex", False)) or ("ex" in name)
        max_hp = getattr(card, "max_hp", None)
        return is_ex or (max_hp is not None and max_hp >= 120)

    filtered: list[tuple[int, int | None, object]] = []
    for hand_idx, bench_idx, target_card in candidates:
        evo_card = p.hand[hand_idx]
        evo_name = (getattr(evo_card, "name", "") or "").strip()
        if evo_name != "ハリテヤマ":
            filtered.append((hand_idx, bench_idx, target_card))
            continue

        if not opp.active:
            continue
        # 進化先が「バトル場」か「ベンチ」かで、付与されるエネルギー量が変わるため
        # その場の attached_energy を引き継ぐ前提で“仮想攻撃”を見積もる。
        field_bp = p.active if bench_idx is None else p.bench[bench_idx]

        tmp_attacker = type("TmpAttacker", (), {})()
        tmp_attacker.card = evo_card
        tmp_attacker.attached_energy = getattr(field_bp, "attached_energy", 0)
        tmp_attacker.attached_energy_types = getattr(field_bp, "attached_energy_types", []) or []

        # バトル場を倒せるなら、どすこいキャッチャーで引っ張る必要なし（そのまま殴ればいい）
        # 高価値でなくても、倒せるならハリテヤマの進化を温存
        can_ko_active = False
        if opp.active.hp is not None and opp.active.hp > 0:
            # バトル場のメインアタッカーで倒せるか
            if p.active:
                active_dmg = _max_effective_damage_for_attacker(state, p.active, opp.active, state.current_player)
                if active_dmg >= opp.active.hp:
                    can_ko_active = True

        if can_ko_active:
            continue

        can_ko_bench_high = False
        for bp in (opp.bench or []):
            if not bp or bp.hp is None or bp.hp <= 0:
                continue
            if not _is_high_value_target(bp):
                continue
            # ハリテヤマ自身で倒せるか
            dmg_bench = _max_effective_damage_for_attacker(state, tmp_attacker, bp, state.current_player)
            if dmg_bench >= bp.hp:
                can_ko_bench_high = True
                break
            # どすこいキャッチャーで引っ張った後、バトル場のアタッカーで倒せるか
            # （ハリテヤマの真価はワイルドプレスだけでなくどすこいキャッチャー）
            if p.active and p.active.card:
                active_dmg = _max_effective_damage_for_attacker(state, p.active, bp, state.current_player)
                if active_dmg >= bp.hp:
                    can_ko_bench_high = True
                    break

        if can_ko_bench_high:
            filtered.append((hand_idx, bench_idx, target_card))
            continue

        # 高価値KOできなくても、相手を縛る（時間稼ぎ）目的で進化する
        # 逃げコストが高いベンチポケモンを引っ張れば1ターン稼げる
        # 条件: 自分が劣勢（場にアタッカーがいない等）の時のみ
        _has_attacker = p.active and any(
            (getattr(p.active.card, "name", "") or "") in ("メガルカリオex", "メガルカリオ", "ルカリオ", "ハリテヤマ")
            for _ in [1]
        )
        if not _has_attacker:
            # 劣勢 → 時間稼ぎ目的でも進化OK
            filtered.append((hand_idx, bench_idx, target_card))

    if not filtered:
        return False
    candidates = filtered

    def _evolve_score(x):
        hand_idx_, bench_idx_, target_card_ = x
        evo_card_ = p.hand[hand_idx_]
        evo_name_ = (getattr(evo_card_, "name", "") or "").strip()
        w = get_evolve_onto_weight(weights, target_card_)
        on_active = 1 if bench_idx_ is None else 0

        # ベンチのリオルをメガルカリオexに進化させるボーナス
        # （メガブレイブ2連使用のため、ベンチにもメガルカリオを立てておきたい）
        if evo_name_ in ("メガルカリオex", "メガルカリオ") and bench_idx_ is not None:
            # 既にバトル場にメガルカリオがいるなら、ベンチにもう1体立てる価値が高い
            active_name = (getattr(p.active.card, "name", "") or "").strip() if p.active else ""
            if active_name in ("メガルカリオex", "メガルカリオ"):
                w += 500.0  # バトル場にもう1体いる → 2連メガブレイブ可能に

        # ドラパルトexデッキ: ドロンチ進化を最優先（ていさつしれいを先に使い切る）
        # ドラパルトex進化はドロンチ進化が全部終わってから
        from .deck_strategies import is_dragapult_deck_for_player
        if is_dragapult_deck_for_player(state, state.current_player):
            if evo_name_ == "ドロンチ":
                w += 5000.0  # ていさつしれいを先に使うため最優先
            elif evo_name_ == "ドラパルトex":
                # ふしぎなアメが手札にある+ターゲットがドロンチ+ドラメシヤが場にいる
                # → アメでドラメシヤに乗せる方が得（ドロンチをていさつしれい用に残せる）
                _has_ame_evo = any(
                    (getattr(hc, "id", "") or "") == "fushiginaame"
                    for hc in p.hand
                )
                _target_bp_evo = p.active if bench_idx_ is None else p.bench[bench_idx_]
                _target_is_doronchi = (getattr(_target_bp_evo.card, "name", "") or "").strip() == "ドロンチ"
                _has_drameshiya_for_ame = any(
                    (getattr(bp.card, "name", "") or "").strip() == "ドラメシヤ"
                    for bp in ([p.active] if p.active else []) + list(p.bench or [])
                )
                if _has_ame_evo and _target_is_doronchi and _has_drameshiya_for_ame:
                    w -= 20000.0  # アメで乗せるべき → 通常進化を抑制

                # ドロンチ進化候補がまだ残っている場合は後回し
                _has_more_doronchi_evo = any(
                    is_pokemon(hc) and (getattr(hc, "name", "") or "").strip() == "ドロンチ"
                    and hc is not p.hand[hand_idx]
                    for hc in p.hand
                )
                # 手札が少ない（事故中）ならドロンチのていさつしれいを温存
                _hand_size = len(p.hand)
                _doronchi_on_field = sum(
                    1 for bp in ([p.active] if p.active else []) + list(p.bench or [])
                    if (getattr(bp.card, "name", "") or "").strip() == "ドロンチ"
                )
                _drapa_on_field = sum(
                    1 for bp in ([p.active] if p.active else []) + list(p.bench or [])
                    if (getattr(bp.card, "name", "") or "").strip() == "ドラパルトex"
                )
                # 進化対象のドロンチがていさつしれい未使用なら進化を待つ
                _used_ids = getattr(state, "_teisatsushirei_used_ids_this_turn", set())
                _target_bp = p.active if bench_idx_ is None else p.bench[bench_idx_]
                _target_name = (getattr(_target_bp.card, "name", "") or "").strip()
                _teisatsu_unused = (
                    _target_name == "ドロンチ"
                    and id(_target_bp) not in _used_ids
                )
                # 進化後に攻撃できるか（エネルギーが足りるか）
                _target_energy = getattr(_target_bp, "attached_energy", 0) or 0
                _can_attack_after_evo = _target_energy >= 1  # ジェットヘッドはエネ1
                if _has_more_doronchi_evo:
                    w += 1000.0  # ドロンチ進化を先にさせる
                elif _teisatsu_unused:
                    w -= 10000.0  # ていさつしれい未使用 → 絶対に進化しない（先にていさつしれいを使う）
                elif not _can_attack_after_evo:
                    w -= 500.0  # エネ不足で攻撃できない → ていさつしれいを次ターンも使いたいので進化しない
                elif _hand_size <= 4 and _doronchi_on_field <= 1 and _drapa_on_field == 0:
                    w += 500.0  # 手札少なく唯一のドロンチ → ていさつしれい温存、進化抑制
                else:
                    w += 2500.0  # ドロンチ進化候補なし → ドラパルトex進化OK
            elif evo_name_ == "サマヨール":
                w += 1000.0  # カースドボム50
            elif evo_name_ == "ヨノワール":
                w += 1500.0  # カースドボム130

        # ドラパルトexデッキ: ドロンチ進化はベンチ優先（選択肢を残すため）
        # ベンチのドラメシヤ→ドロンチに先に進化 → ていさつしれいでアメを引けば
        # バトル場のドラメシヤをドラパルトexに直接進化できる
        # ドラパルトex進化はバトル場優先（攻撃のため）
        if is_dragapult_deck_for_player(state, state.current_player) and evo_name_ == "ドロンチ":
            on_active = 0 if bench_idx_ is None else 1  # ベンチ優先（逆転）
        return (w, on_active)

    best = max(candidates, key=_evolve_score)
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


def _pick_energy_hand_idx(p: PlayerState, state: GameState | None = None) -> int | None:
    """手札から付与するエネ 1 枚のインデックス。ロック闘エネルギーを基本闘より優先。
    ドラパルトexデッキ時は炎/超エネを悪エネより優先する（ファントムダイブ用）。"""
    indices = [i for i, c in enumerate(p.hand) if is_energy(c)]
    if not indices:
        return None

    _is_drapa = False
    if state is not None:
        from .deck_strategies import is_dragapult_deck_for_player
        _is_drapa = is_dragapult_deck_for_player(state, state.current_player)

    # ドラパルトデッキ: 悪エネはファントムダイブに不要（炎+超の2エネのみ）
    # 悪エネしかない場合、ドラメシヤ逃げ用のみ許可
    if _is_drapa:
        _non_dark = [i for i in indices if getattr(p.hand[i], "energy_type", None) != "darkness"]
        if not _non_dark:
            _active_name = (getattr(p.active.card, "name", "") or "").strip() if p.active else ""
            _active_en = getattr(p.active, "attached_energy", 0) or 0 if p.active else 0
            if _active_name in ("ドラメシヤ", "ヨマワル") and _active_en == 0:
                pass  # 逃げ用にOK
            # ベンチにヨマワル(エネ0)がいれば逃げ用にOK
            elif any(
                (getattr(bp.card, "name", "") or "").strip() == "ヨマワル"
                and getattr(bp, "attached_energy", 0) == 0
                for bp in (p.bench or [])
            ):
                pass  # ベンチヨマワル逃げ用にOK
            else:
                return None

    def _is_lock_fighting_energy(c) -> bool:
        eid = getattr(c, "id", "") or ""
        ename = (getattr(c, "name", "") or "").strip()
        return eid == "rokkutoukenenerugi" or ename == "ロック闘エネルギー"

    def _sort_priority(i: int) -> tuple:
        c = p.hand[i]
        if _is_lock_fighting_energy(c):
            return (0, i)
        eid = getattr(c, "id", "") or ""
        ename = (getattr(c, "name", "") or "").strip()
        etype = getattr(c, "energy_type", None) or ""
        if eid in ("basic-energy-fighting", "kihontouenerugi") or ename == "基本闘エネルギー":
            return (1, i)
        # ドラパルトexデッキ: 場のドラパルトexに必要なタイプのエネを最優先
        if _is_drapa:
            if etype == "darkness":
                return (3, i)  # 悪は低優先（ファントムダイブに使えない）
            # 場のドラパルトexの中で最もファントムダイブに近いポケモンが必要としているタイプを最優先
            if state is not None:
                _best_need_type = None
                _best_priority = -1
                _all_bp = ([state.active_player_state().active] if state.active_player_state().active else []) + list(state.active_player_state().bench or [])
                for _bp in _all_bp:
                    _bpn = (getattr(_bp.card, "name", "") or "").strip()
                    if _bpn not in ("ドラパルトex", "ドロンチ", "ドラメシヤ"):
                        continue
                    _bp_types = list(getattr(_bp, "attached_energy_types", []) or [])
                    _bp_en = getattr(_bp, "attached_energy", 0) or 0
                    _has_fire = "fire" in _bp_types
                    _has_psychic = "psychic" in _bp_types
                    # 優先度: エネが多いほど高い（ファントムダイブに近い）
                    _prio = _bp_en * 10
                    if _bpn == "ドラパルトex":
                        _prio += 5  # ドラパルトexは進化済みなのでボーナス
                    if not _has_fire and _prio > _best_priority:
                        if etype == "fire":
                            _best_need_type = "fire"
                            _best_priority = _prio
                    if not _has_psychic and _prio > _best_priority:
                        if etype == "psychic":
                            _best_need_type = "psychic"
                            _best_priority = _prio
                if _best_need_type == etype:
                    return (0, i)  # 最優先で必要なタイプ
            if etype in ("fire", "psychic"):
                return (1, i)  # 炎/超は高優先
        return (2, i)

    return min(indices, key=_sort_priority)


def _build_energy_attach_input(
    state: GameState,
) -> "tuple[int, object, list] | None":
    """
    エネルギー付与の候補リストを構築して返す（アタッチはしない）。
    ガード条件に引っかかった場合、または候補が 0 件の場合は None を返す。
    進化用即時付与（_should_attach_for_evolution）の判定はしない（呼び出し元で先処理すること）。

    Returns
    -------
    (energy_hand_idx, energy_card, candidates) | None
      candidates = [(bench_index_or_None, dmg), ...]
    """
    from .energy_policy import _is_lunatone, _is_solrock, _is_makunoshita

    p = state.active_player_state()
    energy_hand_idx = _pick_energy_hand_idx(p, state)
    if energy_hand_idx is None or not p.active:
        return None

    energy_card = p.hand[energy_hand_idx]
    new_type = getattr(energy_card, "energy_type", None) or "colorless"
    opp = state.defending_player_state()

    lunatone_in_field = (
        bool(p.active and _is_lunatone(p.active.card))
        or any(_is_lunatone(bp.card) for bp in (p.bench or []))
    )
    solrock_in_field = (
        bool(p.active and _is_solrock(p.active.card))
        or any(_is_solrock(bp.card) for bp in (p.bench or []))
    )
    engine_complete = lunatone_in_field and solrock_in_field

    ability_already_used = getattr(state, "ability_declared_this_turn", None) == "ルナサイクル"
    has_lunatone_fetch_in_hand = any(
        (getattr(c, "id", "") in ("faitogongu", "pokepaddo"))
        or (getattr(c, "name", "") in ("ファイトゴング", "ポケパッド"))
        for c in (p.hand or [])
    )
    if (
        not engine_complete
        and has_lunatone_fetch_in_hand
        and not ability_already_used
        and p.deck
    ):
        return None

    # 手札全捨てサポート（ゼイユ・博士の研究等）がある場合でもエネ付与を許可する。
    # 付けたエネはポケモンに乗り、残りはトラッシュに送られてはどうづきで再利用可能。
    # 将来的にはここの判断も学習で最適化する想定。

    max_active = _max_energy_for_pokemon(p.active.card.attacks)
    candidates: list[tuple] = []
    if p.active.attached_energy < max_active:
        if _is_lunatone(p.active.card):
            # ルナトーンがバトル場 → 基本的にエネを付けない
            # ただし逃げるためにエネが必要（ベンチにアタッカーがいる場合）
            _luna_retreat_cost = getattr(p.active.card, "retreat_cost", 1)
            _luna_tool = getattr(p.active, "attached_tool", None)
            _luna_eff_rc = max(0, _luna_retreat_cost - (2 if _luna_tool and (getattr(_luna_tool, "id", "") or "") == "fuusen" else 0))
            _luna_needs_retreat_energy = p.active.attached_energy < _luna_eff_rc and p.bench
            if not _luna_needs_retreat_energy:
                pass  # 逃げ不要 or ベンチなし → エネ付けない
        elif _is_solrock(p.active.card) and p.active.attached_energy >= 1:
            pass  # ソルロックはコスモビームにエネ1で十分。2枚目以降は不要
        elif not (_is_makunoshita(p.active.card) and p.active.attached_energy == 0):
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
    # ドラパルトexデッキ: サポートポケモンにエネを付ける候補を除外
    from .deck_strategies import is_dragapult_deck_for_player as _is_drapa_build
    _is_drapa_deck = _is_drapa_build(state, state.current_player)
    _drapa_support_names = frozenset({"スボミー", "キチキギスex", "ニャースex", "ヨマワル", "サマヨール", "ヨノワール"})

    # ドラパルトデッキ: バトル場が次ターンKO確定ならエネをベンチに回す（付けたエネが無駄になる）
    if _is_drapa_deck and candidates and p.active and opp.active and p.bench:
        from .damage import _max_effective_damage_for_attacker as _opp_dmg_chk
        _opp_max_dmg = _opp_dmg_chk(state, opp.active, p.active, 1 - state.current_player)
        if _opp_max_dmg >= (p.active.hp or 0):
            candidates = [(bi, d) for bi, d in candidates if bi is not None]  # バトル場候補を除外

    # ドラパルトexデッキ: 悪エネはドラパルトexライン（ドラメシヤ/ドロンチ/ドラパルトex）に付けない
    # ファントムダイブは炎+超が必要。悪エネを付けるとファントムダイブが使えなくなる。
    # また同じタイプのエネを2個付けない（ファントムダイブは火1+超1）
    if _is_drapa_deck:
        _drapa_line_for_energy_filter = frozenset({"ドラパルトex", "ドロンチ", "ドラメシヤ"})
        if candidates:
            active_name = (getattr(p.active.card, "name", "") or "").strip() if p.active else ""
            if active_name in _drapa_line_for_energy_filter:
                if new_type == "darkness":
                    # 悪エネはファントムダイブに貢献しない（炎+超のみ必要）
                    # ドラメシヤの逃げ用のみ許可
                    _active_en = getattr(p.active, "attached_energy", 0) or 0
                    _has_fire_psychic_hand = any(
                        is_energy(c) and getattr(c, "energy_type", None) in ("fire", "psychic")
                        for c in p.hand
                    )
                    if active_name == "ドラメシヤ" and _active_en == 0 and not _has_fire_psychic_hand:
                        pass  # 逃げ用にOK
                    else:
                        candidates.clear()
                elif new_type in ("fire", "psychic"):
                    # 既に同じタイプのエネが付いていたら付けない（火+超が理想）
                    _existing_types = list(getattr(p.active, "attached_energy_types", []) or [])
                    if new_type in _existing_types:
                        candidates.clear()
    # ドラパルトex系が場にいてエネが足りない場合はサポートをフィルタ
    _drapa_line_names = frozenset({"ドラパルトex", "ドロンチ", "ドラメシヤ"})
    _drapa_all_pokemon = ([p.active] if p.active else []) + list(p.bench or [])
    _drapa_needs_energy = _is_drapa_deck and any(
        (getattr(bp.card, "name", "") or "").strip() in _drapa_line_names
        and (getattr(bp, "attached_energy", 0) or 0) < 2
        for bp in _drapa_all_pokemon
    )
    # バトル場がサポートで候補に入っている場合もフィルタ
    # ただしヨマワル+悪エネの場合は逃げ用に付けてOK
    if _is_drapa_deck and candidates:
        active_name = (getattr(p.active.card, "name", "") or "").strip() if p.active else ""
        if active_name == "ヨマワル" and new_type == "darkness" and (getattr(p.active, "attached_energy", 0) or 0) == 0:
            pass  # 悪エネ+ヨマワルエネ0 → 逃げ用OK
        elif active_name in _drapa_support_names or active_name == "マシマシラ":
            candidates.clear()  # サポートがバトル場→候補から除外

    for bi, b in enumerate(p.bench):
        max_b = _max_energy_for_pokemon(b.card.attacks)
        if b.attached_energy >= max_b:
            continue
        if _is_makunoshita(b.card) and b.attached_energy == 0:
            continue
        # ルナトーンにエネを付けない（サポート役、にげるはKOプランナーが処理）
        if _is_lunatone(b.card):
            continue
        # ソルロックはコスモビームにエネ1で十分。2枚目以降は候補から除外。
        if _is_solrock(b.card) and b.attached_energy >= 1:
            continue
        # ドラパルトexデッキ: サポートポケモンをベンチ候補から常に除外
        # ただしヨマワルは悪エネの場合のみ逃げ用として候補に残す（最低優先度）
        if _is_drapa_deck:
            bench_name = (getattr(b.card, "name", "") or "").strip()
            if bench_name == "ヨマワル" and new_type == "darkness" and b.attached_energy == 0:
                pass  # 悪エネ+ヨマワルエネ0 → 逃げ用に付けてOK（最低優先度）
            elif bench_name in _drapa_support_names or bench_name == "マシマシラ":
                continue
        # ドラパルトexデッキ: 悪エネはファントムダイブに不要（炎+超の2エネのみ必要）
        # 同じタイプのエネを2個付けない（火1+超1が必須）
        if _is_drapa_deck:
            bench_name_ene = (getattr(b.card, "name", "") or "").strip()
            if bench_name_ene in ("ドラパルトex", "ドロンチ", "ドラメシヤ"):
                if new_type == "darkness":
                    continue  # 悪エネはドラパルトラインに付けない
                if new_type in ("fire", "psychic"):
                    _bench_existing_types = list(getattr(b, "attached_energy_types", []) or [])
                    if new_type in _bench_existing_types:
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
        return None

    # ドラパルトexデッキ: ファントムダイブ完成に近いドラパルトexにboost（全パス共通）
    if _is_drapa_deck and len(candidates) >= 2:
        _boosted_candidates = []
        for _bi_c, _dmg_c in candidates:
            _bp_c = p.active if _bi_c is None else p.bench[_bi_c]
            _bp_name_c = (getattr(_bp_c.card, "name", "") or "").strip()
            _bonus_c = 0
            if _bp_name_c == "ドラパルトex":
                _types_c = list(getattr(_bp_c, "attached_energy_types", []) or [])
                if new_type in ("fire", "psychic") and new_type not in _types_c:
                    _other_c = "psychic" if new_type == "fire" else "fire"
                    if _other_c in _types_c:
                        _bonus_c = 100000  # このエネでファントムダイブ完成！
            _boosted_candidates.append((_bi_c, _dmg_c + _bonus_c))
        candidates = _boosted_candidates

    return energy_hand_idx, energy_card, candidates


def _try_attach_energy_auto(state: GameState) -> bool:
    """
    自動ターン用：エネルギーを 1 枚付与するなら True を返す。
    手番がルールのみ（rules_only_for_player）のときは付与後の最大有効ダメージが最大の候補（同点はバトル場優先）のみ。
    それ以外は重み (w_energy_attach) とコード内ヒューリスティックを最優先で選び、同点時は有効ダメージで決める。
    手札の進化に必要なときはバトル場に付与する特別扱いあり。
    技で必要な最大までしか付与しない。
    相手バトル場がルカリオ系のときは、ソルロックよりルカリオ（メガ含む）へ基本闘を寄せる。
    エネ 0 のマクノシタは候補に入れない（リオル優先。進化用の active への付与は _should_attach_for_evolution のみ）。
    闘・ロック闘の手張りでは、ルナトーンより既にエネが付いたマクノシタ（ハリテヤマ進化先）を優先。
    """
    p = state.active_player_state()
    can_evolve_this_turn = not _is_first_player_first_turn(state)

    # ドラパルトexデッキ: ファントムダイブ完成即時パス（最優先、他の全パスより先）
    # 手札にfire/psychicがあり、ドラパルトexに不足タイプを付ければファントムダイブが撃てる
    from .deck_strategies import is_dragapult_deck_for_player as _is_drapa_early
    if _is_drapa_early(state, state.current_player) and not state.energy_attached_this_turn:
        _all_bp_pd = ([p.active] if p.active else []) + list(p.bench or [])
        for _bp_pd in _all_bp_pd:
            if (getattr(_bp_pd.card, "name", "") or "").strip() != "ドラパルトex":
                continue
            _types_pd = list(getattr(_bp_pd, "attached_energy_types", []) or [])
            _has_fire_pd = "fire" in _types_pd
            _has_psychic_pd = "psychic" in _types_pd
            _need_type = None
            if _has_fire_pd and not _has_psychic_pd:
                _need_type = "psychic"
            elif _has_psychic_pd and not _has_fire_pd:
                _need_type = "fire"
            if _need_type:
                for _ei_pd, _ec_pd in enumerate(p.hand):
                    if is_energy(_ec_pd) and getattr(_ec_pd, "energy_type", None) == _need_type:
                        # ファントムダイブ完成！即付与
                        _bi_pd = None
                        if _bp_pd is not p.active:
                            for _bi2, _bbp in enumerate(p.bench):
                                if _bbp is _bp_pd:
                                    _bi_pd = _bi2
                                    break
                        if _bi_pd is not None:
                            attach_energy(state, _ei_pd, bench_index=_bi_pd)
                        else:
                            attach_energy(state, _ei_pd)
                        _card_id = getattr(_bp_pd.card, "id", None) or getattr(_bp_pd.card, "name", "")
                        _log_choice(state, "energy_attach", card_id=_card_id)
                        return True

    # ドラパルトexデッキ: 悪エネをドラパルトexラインに付ける早期パスをスキップ
    _drapa_line_early = frozenset({"ドラパルトex", "ドロンチ", "ドラメシヤ"})

    def _drapa_darkness_block(target_bp) -> bool:
        """ドラパルトexデッキで悪エネをドラパルトexラインに付けようとしているかチェック"""
        if not _is_drapa_early(state, state.current_player):
            return False
        eidx = _pick_energy_hand_idx(p, state)
        if eidx is None:
            return False
        etype = getattr(p.hand[eidx], "energy_type", None) or "colorless"
        if etype != "darkness":
            return False
        tname = (getattr(target_bp.card, "name", "") or "").strip()
        return tname in _drapa_line_early

    # 進化用即時付与（特殊ケース：policy を通さずバトル場に付与）
    if can_evolve_this_turn and _should_attach_for_evolution(p):
        energy_hand_idx = _pick_energy_hand_idx(p, state)
        if energy_hand_idx is None:
            return False
        # ドラパルトexデッキ: 悪エネをドラパルトexラインに付けない
        if not _drapa_darkness_block(p.active):
            attach_energy(state, energy_hand_idx)
            return True

    # 次ターン進化→攻撃の準備付与:
    # 次ターン進化→即攻撃の準備付与:
    # 手札に進化先があり、その技がエネ1枚で撃てるなら（はどうづき等）、
    # バトル場またはベンチの進化元にエネを付けておく。
    # バトル場優先、ベンチにリオルがいればそちらにも。
    if p.active and not state.energy_attached_this_turn:
        energy_hand_idx = _pick_energy_hand_idx(p, state)
        if energy_hand_idx is not None:
            # バトル場→ベンチの順で進化元を探す
            targets = [(None, p.active)]  # (bench_index, bp)
            for bi, bp in enumerate(p.bench):
                targets.append((bi, bp))
            for bench_idx, bp in targets:
                for c in p.hand:
                    if not (is_pokemon(c) and _can_evolve_onto(bp.card, c)):
                        continue
                    # 進化先の技でエネ1枚で使えるものがあるか
                    has_cheap_attack = any(a.energy_cost <= 1 for a in (c.attacks or []))
                    if has_cheap_attack and bp.attached_energy == 0:
                        # ドラパルトexデッキ: 悪エネをドラパルトexラインに付けない
                        if _drapa_darkness_block(bp):
                            break
                        if bench_idx is None:
                            attach_energy(state, energy_hand_idx)
                        else:
                            attach_energy(state, energy_hand_idx, bench_index=bench_idx)
                        card_id = getattr(bp.card, "id", None) or getattr(bp.card, "name", "")
                        _log_choice(state, "energy_attach", card_id=card_id)
                        return True
                    break  # 進化先は1体だけ見ればよい

    # バトル場にエネ付けてにげる→ベンチのアタッカーで攻撃してKO:
    # バトル場がにげコスト1でエネ0、ベンチに攻撃可能なアタッカーがいるなら
    # バトル場にエネを付ける（にげるコストを賄うため）。
    # ケース1: ベンチのアタッカーで相手を倒せる
    # ケース2: バトル場が攻撃できない + ベンチに攻撃可能なアタッカーがいる（倒せなくても攻撃する方が得）
    # ドラパルトexデッキ: サポートポケモンにエネを付けてにげるのは非効率。
    # ベンチのドラパルトexに直接エネを付けた方がファントムダイブへの投資になる。
    # ただしKO可能な場合（ケース1）のみ許可。
    from .deck_strategies import is_dragapult_deck_for_player as _is_drapa_retreat
    _skip_retreat_attach = False
    if _is_drapa_retreat(state, state.current_player):
        active_name = (getattr(p.active.card, "name", "") or "").strip() if p.active else ""
        if active_name in ("キチキギスex", "ニャースex", "スボミー"):
            # exサポート/スボミーにエネを付けて逃げるのは無駄（エネはドラパルトexに回す）
            _skip_retreat_attach = True
        elif active_name in ("ヨマワル", "サマヨール", "マシマシラ", "ヨノワール"):
            # 攻撃力がほぼないサポートポケモン → ベンチに攻撃可能なポケモンがいれば逃げる
            # スボミー（むずむずかふん=グッズロック）等がいれば逃げた方が得
            _has_better_bench = any(
                (getattr(bp.card, "name", "") or "").strip() in ("スボミー", "ドラパルトex", "ドロンチ", "ドラメシヤ")
                and bp.card.attacks
                for bp in (p.bench or [])
            )
            if not _has_better_bench:
                _skip_retreat_attach = True
    if not _skip_retreat_attach and (
        p.active
        and not state.energy_attached_this_turn
        and not getattr(state, "retreat_used_this_turn", False)
        and not _is_first_player_first_turn(state)
        and getattr(p.active, "special_state", None) not in ("sleep", "paralysis")
    ):
        from .attack import get_legal_attack_indices, get_legal_attack_indices_for_attacker
        opp = state.defending_player_state()
        raw_rc = getattr(p.active.card, "retreat_cost", 1)
        tool = getattr(p.active, "attached_tool", None)
        eff_rc = max(0, raw_rc - (2 if tool and (getattr(tool, "id", "") or "") == "fuusen" else 0))
        # にげコスト1でエネ0（エネ付ければにげられる）
        if eff_rc >= 1 and p.active.attached_energy < eff_rc and p.active.attached_energy + 1 >= eff_rc:
            if opp.active and opp.active.hp is not None and opp.active.hp > 0:
                our_dmg = _max_effective_damage_for_attacker(state, p.active, opp.active, state.current_player)
                active_can_attack = bool(get_legal_attack_indices(state, p, opp))
                should_retreat_for_bench = False
                for bp in p.bench:
                    if not get_legal_attack_indices_for_attacker(state, p, opp, bp):
                        continue
                    bench_dmg = _max_effective_damage_for_attacker(state, bp, opp.active, state.current_player)
                    # ケース1: ベンチで倒せる
                    if bench_dmg >= opp.active.hp and our_dmg < opp.active.hp:
                        should_retreat_for_bench = True
                        break
                    # ケース2: バトル場が攻撃できない + ベンチで攻撃可能
                    if not active_can_attack and bench_dmg > 0:
                        should_retreat_for_bench = True
                        break
                if should_retreat_for_bench:
                    energy_hand_idx = _pick_energy_hand_idx(p, state)
                    if energy_hand_idx is not None:
                        attach_energy(state, energy_hand_idx)
                        card_id = getattr(p.active.card, "id", None) or getattr(p.active.card, "name", "")
                        _log_choice(state, "energy_attach", card_id=card_id)
                        return True

    result = _build_energy_attach_input(state)
    if result is None:
        return False
    energy_hand_idx, energy_card, candidates = result

    from .deck_strategies import is_dragapult_deck_for_player as _is_drapa_ene_prio
    if _is_drapa_ene_prio(state, state.current_player) and len(candidates) >= 2:
        _active_name_ep = (getattr(p.active.card, "name", "") or "").strip() if p.active else ""
        if _active_name_ep == "ドラメシヤ":
            # ベンチにドロンチ/ドラパルトexがいれば、バトル場のドラメシヤ候補を除外
            _has_better_bench = any(
                bi is not None and (getattr(p.bench[bi].card, "name", "") or "").strip() in ("ドロンチ", "ドラパルトex")
                for bi, _ in candidates
            )
            if _has_better_bench:
                candidates = [(bi, dmg) for bi, dmg in candidates if bi is not None]

    if rules_only_for_player(state):
        best = pick_energy_attach_candidate(candidates)
        target_card = p.active.card if best[0] is None else p.bench[best[0]].card
        card_id = getattr(target_card, "id", None) or getattr(target_card, "name", "")
        if best[0] is None:
            attach_energy(state, energy_hand_idx)
            _log_choice(state, "energy_attach", card_id=card_id)
            return True
        attach_energy(state, energy_hand_idx, bench_index=best[0])
        _log_choice(state, "energy_attach", card_id=card_id)
        return True

    bench_index, log_extras = pick_energy_attach_by_policy(state, candidates, energy_hand_idx, energy_card)
    target_card = p.active.card if bench_index is None else p.bench[bench_index].card
    card_id = getattr(target_card, "id", None) or getattr(target_card, "name", "")
    if bench_index is None:
        attach_energy(state, energy_hand_idx)
    else:
        attach_energy(state, energy_hand_idx, bench_index=bench_index)
    _log_choice(state, "energy_attach", card_id=card_id, **log_extras)
    return True

