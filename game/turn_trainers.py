"""ターン実行用：サポート・グッズ・どうぐの試行順と 1 回試行。"""
from card import is_goods, is_support

from .damage import _max_effective_damage_for_attacker
from .state import BattlePokemon, GameState, PlayerState, _is_first_player_first_turn, _log_choice
from .trainers import attach_tool, use_potion, use_support, use_trainer_goods
from .weights import get_goods_use_weight, get_support_use_weight, get_tool_attach_weight

_HAND_REFRESH_SUPPORT_IDS = ("tanpankozou", "hakasenokenkyuu", "hakasenokenkyuufutouhakase", "jixyajjiman", "kihada")
# 暗号マニアの解読（ルナサイクルより前に使う想定だが、一般サポート枠では山を切った後に回す）
_SUPPORT_ID_ANGOUMANIANOKAIDOKU = "angoumanianokaidoku"
# use_support で自分の山札を shuffle するサポート（暗号より先に使う／1 ターン 1 枚のため使用後は手札にあっても無視）
_SUPPORT_IDS_SHUFFLE_OWN_DECK = frozenset(
    {
        "riirienokesshin",
        "tanpankozou",
        "jixyajjiman",
    }
)
_SUPPORT_IDS_NO_DISCARD = ("nemo", "nemokako", "nemomirai", "kihada", "hikari", "meinohagemashi", "akamatsu")
_SUPPORT_IDS_DISCARD_ALL = ("hakasenokenkyuu", "hakasenokenkyuufutouhakase")
# 手札をすべてトラッシュするサポート（エネを貼らず手札に残してトラッシュへ送り、夜のタンカ等のターゲットにする）
_SUPPORT_IDS_TRASH_WHOLE_HAND = ("zeiyu",) + _SUPPORT_IDS_DISCARD_ALL
_SUPPORT_IDS_HAND_REFRESH_FIRST = ("riirienokesshin", "zeiyu") + _HAND_REFRESH_SUPPORT_IDS

# 手札を「トラッシュする」刷新の直前にハイパーボールを優先。
# リーリエの決心のように山札に戻す系では、マキシマムベルト等「捨てたくない」カードを残したいので対象外。
_HAIPABORU_BEFORE_HAND_REFRESH_SUPPORT_IDS = ("zeiyu",) + _SUPPORT_IDS_DISCARD_ALL


def hand_has_remaining_shuffle_effect_for_angou(state: GameState, p: PlayerState) -> bool:
    """
    暗号マニアより前に処理すべき「自分の山を切る」カードが手札に残っているか。
    サポートは 1 ターン 1 枚のため、すでにサポート使用済みなら手札にシャッフル系サポートがあっても False。
    """
    for _i, c in enumerate(p.hand or []):
        cid = getattr(c, "id", "") or ""
        nm = (getattr(c, "name", "") or "").strip()
        if cid == _SUPPORT_ID_ANGOUMANIANOKAIDOKU:
            continue
        if is_goods(c) and getattr(c, "is_tool", False):
            continue

        if is_support(c):
            if cid not in _SUPPORT_IDS_SHUFFLE_OWN_DECK:
                continue
            if state.support_used_this_turn:
                continue
            return True

        if not is_goods(c):
            continue

        if cid == "otodokedoron" or nm == "おとどけドローン":
            return True
        if cid == "supaboru" or nm == "スーパーボール":
            if p.deck:
                return True
            continue
        if cid == "haipaboru" or nm == "ハイパーボール":
            if len(p.hand) >= 3 and p.deck:
                return True
            continue
        if cid == "faitogongu" or nm == "ファイトゴング":
            if p.deck:
                return True
            continue
        if cid == "pokepaddo" or nm == "ポケパッド":
            if p.deck:
                return True
            continue
        if cid == "erekijienereta" or nm == "エレキジェネレーター":
            if p.bench and p.deck:
                return True
            continue
        if cid == "anfeasutanpu" or nm == "アンフェアスタンプ":
            _any_ko_as = getattr(state, "any_ko_by_opponent_last_turn", [False, False])
            if _any_ko_as[state.current_player] or state.our_ko_by_damage_last_turn[state.current_player]:
                return True
            continue
    return False


def _is_rioru_or_lucario_line(card) -> bool:
    """マキシマムベルトを基本貼りたいライン（リオル・ルカリオ・メガルカリオ系）。"""
    name = (getattr(card, "name", "") or "").strip()
    if name in ("リオル", "ルカリオ", "メガルカリオ", "メガルカリオex"):
        return True
    cid = (getattr(card, "id", "") or "").strip().lower()
    return cid.startswith("rioru") or "rukario" in cid or cid.startswith("mrukario")


def _defender_is_ex(defender_bp: BattlePokemon | None) -> bool:
    if not defender_bp:
        return False
    c = defender_bp.card
    nm = (getattr(c, "name", "") or "").lower()
    return bool(getattr(c, "is_ex", False)) or "ex" in nm


def _belt_enables_ko_on_opp_active_ex(state: GameState, attacker_bp: BattlePokemon, tool_card) -> bool:
    """
    バトル場のこのポケモンにマキシマムベルトを付けたときだけ、
    相手のバトル場の ex をこのターンのワザで倒せるようになるなら True。
    （ベンチに貼る例外には使わない）
    """
    opp = state.defending_player_state()
    oa = opp.active
    if not oa or oa.hp <= 0:
        return False
    if not _defender_is_ex(oa):
        return False
    without = _max_effective_damage_for_attacker(state, attacker_bp, oa, state.current_player)
    if without >= oa.hp:
        return False
    old = attacker_bp.attached_tool
    attacker_bp.attached_tool = tool_card
    try:
        with_belt = _max_effective_damage_for_attacker(state, attacker_bp, oa, state.current_player)
    finally:
        attacker_bp.attached_tool = old
    return with_belt >= oa.hp


def _try_attach_one_tool(state: GameState) -> bool:
    """手札からどうぐを 1 枚、バトル場またはベンチにつける。つけたら True。重みで付ける先を選ぶ。"""
    p = state.active_player_state()
    weights = state.get_weights_for_player(state.current_player)
    candidates: list[tuple[int, int | None, BattlePokemon]] = []
    for i, c in enumerate(p.hand):
        if not is_goods(c) or not getattr(c, "is_tool", False):
            continue
        cond = getattr(c, "tool_condition_type", None)
        if p.active and getattr(p.active, "attached_tool", None) is None:
            if cond is None or getattr(p.active.card, "pokemon_type", None) == cond:
                candidates.append((i, None, p.active))
        for bi, bp in enumerate(p.bench):
            if getattr(bp, "attached_tool", None) is not None:
                continue
            if cond is not None and getattr(bp.card, "pokemon_type", None) != cond:
                continue
            candidates.append((i, bi, bp))
    if not candidates:
        return False
    filtered: list[tuple[int, int | None, BattlePokemon]] = []
    for hand_i, bench_i, bp in candidates:
        tool = p.hand[hand_i]
        if (getattr(tool, "id", "") or "") == "makishimamuberuto":
            ok = _is_rioru_or_lucario_line(bp.card) or (
                bench_i is None and _belt_enables_ko_on_opp_active_ex(state, bp, tool)
            )
            if not ok:
                continue
        filtered.append((hand_i, bench_i, bp))
    if not filtered:
        return False

    def _sort_key(x: tuple[int, int | None, BattlePokemon]) -> tuple:
        hand_i, bench_i, bp = x
        tool = p.hand[hand_i]
        w = get_tool_attach_weight(weights, bp.card)
        on_active = bench_i is None
        tool_id = (getattr(tool, "id", "") or "")
        if tool_id == "makishimamuberuto":
            if _is_rioru_or_lucario_line(bp.card):
                tier = 2_000_000
            else:
                tier = 1_000_000
        elif tool_id == "fuusen":
            bp_name = (getattr(bp.card, "name", "") or "").strip()
            # バトル場が攻撃できない＋にげコスト≥1 → バトル場に付けてにげる（縛られ防止）
            opp = state.defending_player_state()
            from .attack import get_legal_attack_indices
            active_cant_attack = (
                on_active
                and p.active
                and not get_legal_attack_indices(state, p, opp)
                and getattr(p.active.card, "retreat_cost", 0) >= 1
            )
            if active_cant_attack:
                tier = 3_000_000  # バトル場に付けてにげる最優先
            elif bp_name == "ルナトーン" and not on_active:
                tier = 2_500_000  # Fix I: ルナトーンに最優先で付ける（メガブレイブリセットトリック）
            elif bp_name == "ルナトーン" and on_active:
                tier = 2_000_000  # ルナトーンがバトル場でも高優先
            elif bp_name == "マクノシタ" and not on_active:
                tier = 500_000
            elif bp_name == "ハリテヤマ":
                # Pattern 3: ハリテヤマはにげコスト3、ふうせん-2=1でフリーにならない→付けない
                tier = -1_000_000
            elif _is_rioru_or_lucario_line(bp.card):
                tier = 50_000   # Fix I: メインアタッカーには付けない（ルナトーン優先）
            else:
                # にげコストが低いポケモンを優先（ふうせんでフリーになるポケモンのみ高評価）
                rc = getattr(bp.card, "retreat_cost", 1)
                if rc <= 2:
                    tier = 100_000 + (10 - rc) * 1000
                else:
                    # にげコスト3以上はふうせんでフリーにならない → 低優先
                    tier = -500_000
        else:
            tier = 0
        return (tier, w, 1 if on_active else 0)

    best = max(filtered, key=_sort_key)
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
    サポートを試す順序。手札刷新系（リーリエの決心・博士の研究等）をボスの指令より先に試す。
    同系統内は重み (w_support_use + SUPPORT_DRAW_BONUS) の高い順、同点は手札インデックス順。
    """
    support_indices = [(i, p.hand[i]) for i in range(len(p.hand)) if is_support(p.hand[i])]
    if not support_indices:
        return []
    weights = state.get_weights_for_player(state.current_player)

    # ドラパルトexデッキ: アカマツは「このターンにファントムダイブが撃てる」場合のみ高優先
    # ファントムダイブに届かないならアカマツの優先度は低い（エネを付けても次ターンに倒されるリスク）
    from .deck_strategies import is_dragapult_deck_for_player as _is_drapa_supp
    _drapa_akamatsu_enables_phantom = False
    if _is_drapa_supp(state, state.current_player):
        _all_bp = ([p.active] if p.active else []) + list(p.bench or [])
        # ドラパルトexが場にいればアカマツ使用OK（エネを貯め始める）
        _drapa_akamatsu_enables_phantom = any(
            (getattr(bp.card, "name", "") or "").strip() == "ドラパルトex"
            for bp in _all_bp
        )

    def _key(x):
        i, c = x
        cid = getattr(c, "id", "") or ""
        # ドラパルトexデッキ: アカマツ/メイはファントムダイブに届く時のみ最優先
        energy_supp_priority = 1
        if _drapa_akamatsu_enables_phantom and cid in ("akamatsu", "meinohagemashi"):
            energy_supp_priority = -1  # 最優先
        hand_refresh_first = 0 if cid in _SUPPORT_IDS_HAND_REFRESH_FIRST else 1
        # 山札を切るグッズ／サポートを手札に残したまま暗号を切らない
        angou_defer = 0
        if cid == _SUPPORT_ID_ANGOUMANIANOKAIDOKU and hand_has_remaining_shuffle_effect_for_angou(state, p):
            angou_defer = 1
        w = get_support_use_weight(weights, c)
        # ジャッジマン: 相手の手札が多い（6枚以上）時に優先度大幅UP
        # おたがい4枚にリセットするので、相手の手札が多いほど妨害効果が高い
        if cid == "jixyajjiman":
            opp_hand = len(state.defending_player_state().hand)
            if opp_hand >= 6:
                w += 300.0  # 相手の手札を大幅に減らせる → 高優先
        return (energy_supp_priority, hand_refresh_first, angou_defer, -w, i)

    support_indices.sort(key=_key)
    return [i for i, c in support_indices]


def _try_angou_before_luna_cycle(state: GameState) -> bool:
    """
    ルナトーンのルナサイクル（山札 3 枚）の直前に、暗号マニアの解読だけを試す。
    自分の山を切るグッズ・サポートを手札に残したままには使わない（使い切ってから）。
    """
    if state.support_used_this_turn:
        return False
    p = state.active_player_state()
    if hand_has_remaining_shuffle_effect_for_angou(state, p):
        return False
    for i, c in enumerate(p.hand):
        if not is_support(c) or getattr(c, "id", "") != _SUPPORT_ID_ANGOUMANIANOKAIDOKU:
            continue
        if use_support(state, i):
            support_card_id = getattr(c, "id", None) or getattr(c, "name", "")
            if support_card_id:
                _log_choice(state, "support", card_id=support_card_id)
            return True
    return False


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

    # ドラパルトexデッキ: アカマツはファントムダイブに届く時のみ高優先
    from .deck_strategies import is_dragapult_deck_for_player
    _is_drapa_nds = is_dragapult_deck_for_player(state, state.current_player)
    _nds_phantom_reachable = False
    if _is_drapa_nds:
        _all_bp_nds = ([p.active] if p.active else []) + list(p.bench or [])
        # ドラパルトexが場にいればアカマツ高優先（エネを貯め始める）
        _nds_phantom_reachable = any(
            (getattr(_bp_nds.card, "name", "") or "").strip() == "ドラパルトex"
            for _bp_nds in _all_bp_nds
        )

    # ドラパルトデッキ: 1ターン目はヒカリをスキップ（進化ポケモンを取っても使えない）
    # リーリエの決心が手札にあるならリーリエで引き直す方が良い
    if _is_drapa_nds and state.turn_count <= 1:
        _has_lillie_nds = any(
            is_support(c) and (getattr(c, "id", "") or "") == "riirienokesshin"
            for c in p.hand
        )
        if _has_lillie_nds:
            no_discard = [(i, c) for i, c in no_discard if (getattr(c, "id", "") or "") != "hikari"]
            if not no_discard:
                return False

    def _nds_sort_key(x):
        i, c = x
        cid = getattr(c, "id", "") or ""
        w = get_support_use_weight(weights, c)
        if _is_drapa_nds:
            if cid == "meinohagemashi":
                w += 200.0
            elif cid == "akamatsu" and _nds_phantom_reachable:
                w += 150.0
            elif cid == "akamatsu":
                w -= 100.0
        return -w

    no_discard.sort(key=_nds_sort_key)
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
    """
    手札刷新系サポートの前にグッズを試す。使ったら True。
    手札を全捨てするサポート（ゼイユ・博士の研究等）がある場合は、
    使えるグッズを可能な限り使い切る（捨てるのは勿体ない）。
    """
    p = state.active_player_state()
    if _try_attach_one_tool(state):
        return True
    weights = state.get_weights_for_player(state.current_player)
    has_trash_hand_support = any(
        is_support(c) and (getattr(c, "id", "") or "") in _SUPPORT_IDS_TRASH_WHOLE_HAND
        for c in p.hand
    )
    wants_haipaboru_first = any(
        is_support(c) and (getattr(c, "id", "") or "") in _HAIPABORU_BEFORE_HAND_REFRESH_SUPPORT_IDS
        for c in p.hand
    )

    # ドラパルトデッキ: ドローサポートが手札にあるならHBを使わない
    # （リーリエの決心で引いてから判断すべき。HBで貴重なカードを捨てるリスクを避ける）
    from .deck_strategies import is_dragapult_deck_for_player as _is_drapa_gbhr
    _drapa_skip_hb_gbhr = False
    if _is_drapa_gbhr(state, state.current_player):
        _has_draw_supp_gbhr = any(
            is_support(c) and (getattr(c, "id", "") or "") in (
                "riirienokesshin", "zeiyu", "hakasenokenkyuu", "hikari",
            )
            for c in p.hand
        )
        _has_nyarth_gbhr = any(
            (getattr(c, "name", "") or "").strip() == "ニャースex"
            for c in list(p.hand) + list(p.deck)
        )
        _has_support_gbhr = any(is_support(c) for c in p.hand)
        if state.support_used_this_turn:
            _drapa_skip_hb_gbhr = True
        elif _has_draw_supp_gbhr:
            _nyarth_in_deck_gbhr2 = any(
                "ニャースex" in (getattr(dc, "name", "") or "")
                for dc in p.deck
            )
            if not _nyarth_in_deck_gbhr2:
                _drapa_skip_hb_gbhr = True

    def _build_goods_list():
        return [
            (i, c)
            for i, c in enumerate(p.hand)
            if is_goods(c)
            and getattr(c, "effect", None) != "swap_active"
            and not getattr(c, "is_tool", False)
            and not (getattr(c, "id", None) == "haipaboru" and _is_first_player_first_turn(state) and len(p.bench) > 0)
            and not (getattr(c, "id", None) == "haipaboru" and _drapa_skip_hb_gbhr)
        ]

    def _goods_sort_before_refresh(x):
        i, c = x
        w = get_goods_use_weight(weights, c)
        cid = getattr(c, "id", "") or ""
        if wants_haipaboru_first and cid == "haipaboru":
            return (0, -w, i)
        if wants_haipaboru_first:
            return (1, -w, i)
        return (0, -w, i)

    any_used = False
    # 手札全捨て系サポートがある → 使えるグッズを使い切る（最大10回で安全弁）
    max_rounds = 10 if has_trash_hand_support else 1
    for _ in range(max_rounds):
        goods_list = _build_goods_list()
        goods_list.sort(key=_goods_sort_before_refresh)
        used_this_round = False
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
                any_used = True
                used_this_round = True
                break
        if not used_this_round:
            break
        # どうぐも毎ラウンド試す
        if _try_attach_one_tool(state):
            any_used = True
    return any_used
