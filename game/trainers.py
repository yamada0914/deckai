"""トレーナー（グッズ・サポート・どうぐ・いれかえ・エネルギー付与）。"""
import os
import random
from .card_ids import (
    AKAMATSU,
    BOSS_NO_SHIREI,
    BURAIA,
    FIGHT_GONG,
    FUSHIGI_NA_AME,
    FUUSEN,
    HAKASE_NO_KENKYU,
    HIKARI,
    HYPER_BALL,
    JUDGE,
    KIHADA,
    MEI_NO_HAGEMASHI,
    NAKAYOSHI_POFIN,
    POKEMON_IREKAE,
    POKEPAD,
    RIRIE_NO_KESSHIN,
    SPECIAL_RED_CARD,
    SUPER_BALL,
    TANPAN_KOZOU,
    UNFAIR_STAMP,
    YORU_NO_TANKA,
    ZEIYU,
)

from card import (
    get_card_by_id,
    get_card_by_name,
    is_basic_pokemon,
    is_energy,
    is_goods,
    is_pokemon,
    is_stage2_pokemon,
    is_stadium,
    is_support,
)

_BASIC_ENERGY_TYPES = ("grass", "fire", "water", "lightning", "psychic", "fighting", "darkness", "metal", "fairy")

_ONLINE_FINISHER_POS_BONUS = float(os.getenv("DECKAI_ONLINE_FINISHER_POS_BONUS", "150.0"))
_ONLINE_FINISHER_NEG_PENALTY = float(os.getenv("DECKAI_ONLINE_FINISHER_NEG_PENALTY", "60.0"))
_ONLINE_MAIN_ATTACKER_BONUS = float(os.getenv("DECKAI_ONLINE_MAIN_ATTACKER_BONUS", "100.0"))

from .attack import get_legal_attack_indices
from .damage import _max_effective_damage_for_attacker
from .card_roles import is_engine_pair_member, is_finisher, is_main_attacker, is_online, use_online_eval
from .deck_strategies import (
    get_allow_duplicate_bench_ids,
    get_fetch_bonus_for_card,
    is_dragapult_deck_for_player,
    DRAPA_LINE_NAMES,
    DRAPA_SUPPORT_NAMES,
    DRAPA_ENERGY_BANNED_NAMES,
)
from .evolution import _apply_evolution, _can_evolve_onto
from .weights import (
    get_faitogongu_fetch_weight,
    get_haipaboru_discard_weight,
    get_pokepaddo_duplicate_penalty,
    get_pokepaddo_fetch_weight,
)
from .state import (
    GameState,
    PlayerState,
    BattlePokemon,
    _card_label,
    _clear_status,
    card_from_discard_to_hand,
    get_effective_max_hp,
    _flip_coin,
    _is_first_player_first_turn,
    _log_choice,
    _prizes_for_ko,
    mark_own_deck_shuffled,
    mark_deck_searched,
    rules_only_for_player,
)


def _player_has_pokemon_by_name_or_id(p: PlayerState, name: str, ids: set[str]) -> bool:
    """手札・バトル場・ベンチに指定名/IDのポケモンがいるか。
    ファイトゴング・ポケパッドのエンジン判定で共通使用。"""
    for x in (p.hand or []):
        if not is_pokemon(x):
            continue
        if (getattr(x, "name", "") or "") == name or (getattr(x, "id", "") or "") in ids:
            return True
    if getattr(p, "active", None) and getattr(p.active, "card", None):
        a = p.active.card
        if (getattr(a, "name", "") or "") == name or (getattr(a, "id", "") or "") in ids:
            return True
    for bp in (p.bench or []):
        c = getattr(bp, "card", None)
        if not c:
            continue
        if (getattr(c, "name", "") or "") == name or (getattr(c, "id", "") or "") in ids:
            return True
    return False


def would_haipaboru_fetch_evolution(p: PlayerState) -> bool:
    """ハイパーボールを使ったときに山札から取ってくる 1 枚が進化ポケモンかどうか。"""
    found = _find_pokemon_for_haipaboru(p)
    if not found:
        return False
    return bool(getattr(found[1], "evolves_from", None))


def _has_engine_pair_member_in_hand_or_field(p: PlayerState) -> bool:
    """手札・バトル場・ベンチにエンジンペアの片割れが 1 枚でもいるか。盤面に意味があるときだけセットロジックをオンにする。"""
    for c in (p.hand or []):
        if is_engine_pair_member(c):
            return True
    if getattr(p, "active", None) and getattr(p.active, "card", None):
        if is_engine_pair_member(p.active.card):
            return True
    for bp in (p.bench or []):
        c = getattr(bp, "card", None)
        if c and is_engine_pair_member(c):
            return True
    return False


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


def _find_pokemon_for_haipaboru(p: PlayerState, state=None) -> tuple[int, object] | None:
    """
    ハイパーボール用：山札から 1 枚を選び (deck_index, card) を返す。
    手札・場に既にいるポケモンは避け、優先度は
    (1) 2進化で場にのせられる (2) 1進化で場にのせられる (3) たね (4) 強さ（HP・ex）。
    """
    candidates = [(i, c) for i, c in enumerate(p.deck) if is_pokemon(c)]
    if not candidates:
        return None
    # ベンチが空で手札にたねもない（種切れリスク）→ 同名でも取る（リオル2体目など）
    # ただし手札にHBがもう1枚あるなら、このHBでは進化先を取り次のHBでたねを取れる
    _hand_has_basic_for_bench = any(
        is_pokemon(hc) and (is_basic_pokemon(hc) or not getattr(hc, "evolves_from", None))
        for hc in p.hand
    )
    _has_another_hb = any(
        getattr(hc, "id", "") == HYPER_BALL for hc in p.hand
    )
    if len(p.bench) == 0 and not _hand_has_basic_for_bench and not _has_another_hb:
        pool = candidates
    else:
        # ドラパルトデッキ: ドロンチは複数必要（ていさつしれい）なので除外しない
        _drapa_names_pre = {"ドラメシヤ", "ドロンチ", "ドラパルトex"}
        _is_drapa_hb_pre = any(
            is_pokemon(c) and (getattr(c, "name", "") or "").strip() in _drapa_names_pre
            for c in list(p.deck) + list(p.hand)
        )
        _drapa_allow_dup = {"ドロンチ", "ドラメシヤ"} if _is_drapa_hb_pre else set()
        preferred = [
            (i, c) for i, c in candidates
            if not _haipaboru_already_have(p, c)
            or (getattr(c, "name", "") or "").strip() in _drapa_allow_dup
        ]
        if not preferred:
            # 手札・場にいないポケモンが山札にない → ハイパーボールを使う意味がない
            return None
        pool = preferred

    # 場にリオルライン（リオル/メガルカリオ系）がいるか
    _rioru_line_names = {"リオル", "メガルカリオex", "メガルカリオ", "ルカリオ"}
    field_cards = ([p.active.card] if p.active else []) + [bp.card for bp in p.bench]
    has_rioru_line_in_field = any(
        (getattr(fc, "name", "") or "") in _rioru_line_names for fc in field_cards
    )
    # 手札・山札にメガルカリオexがあるか（リオルを取る動機）
    has_mega_lucario_in_hand_or_deck = any(
        is_pokemon(c) and (getattr(c, "name", "") or "") in ("メガルカリオex", "メガルカリオ")
        for c in list(p.hand) + list(p.deck)
    )

    bench_empty = len(p.bench) == 0
    # 手札にたねポケモンがあればベンチに出せる → 実質ベンチ空ではない
    _hand_has_basic = any(
        is_pokemon(hc) and (is_basic_pokemon(hc) or not getattr(hc, "evolves_from", None))
        for hc in p.hand
    )
    bench_truly_empty = bench_empty and not _hand_has_basic

    # ドラパルトexデッキ判定（デッキ内容から推定）
    _drapa_names = {"ドラメシヤ", "ドロンチ", "ドラパルトex"}
    _is_drapa_hb = any(
        is_pokemon(c) and (getattr(c, "name", "") or "").strip() in _drapa_names
        for c in list(p.deck) + list(p.hand)
    )
    # リーリエの決心等の大量ドローサポートが手札にあるか（ドロンチはドローで引けるので優先度下げる）
    _has_draw_support_hb = any(
        is_support(hc) and (getattr(hc, "id", "") or "") in (
            RIRIE_NO_KESSHIN, ZEIYU, HAKASE_NO_KENKYU, HIKARI,
        )
        for hc in p.hand
    )

    def score(deck_idx: int, c) -> float:
        strength = _haipaboru_strength(c)
        c_name = (getattr(c, "name", "") or "").strip()

        # ドラパルトexデッキ固有スコアリング
        if _is_drapa_hb:
            # ベンチ空なら種切れ防止優先
            if bench_truly_empty:
                if is_basic_pokemon(c) or not getattr(c, "evolves_from", None):
                    if c_name == "ドラメシヤ":
                        # ドローサポートが手札にない → ニャースex優先（おくのてキャッチ→リーリエ）
                        # ドローサポートがある → ドラメシヤ優先
                        if _has_draw_support_hb:
                            return 25000.0  # サポートあり → ドラメシヤ最優先
                        return 15000.0  # サポートなし → ニャースexより低い
                    if c_name == "ニャースex":
                        if not _has_draw_support_hb:
                            return 20000.0  # サポートなし → ニャースex最優先
                        return 12000.0  # サポートあり → 種切れ防止用
                    if c_name == "キチキギスex":
                        return 5000.0
                    return 15000.0 + strength
                return strength
            # ベンチにドラメシヤがいない → ドラメシヤ最優先（ドロンチ進化の基盤）
            _has_drameshiya_on_field = any(
                (getattr(fc, "name", "") or "").strip() in ("ドラメシヤ", "ドロンチ", "ドラパルトex")
                for fc in field_cards
            )
            if c_name == "ドラメシヤ" and not _has_drameshiya_on_field:
                return 18000.0 + strength  # ドラメシヤ0体 → 最優先で確保
            # リーリエが手札にある+即進化不可 → 進化ポケモンはリーリエで流される → たねを優先
            if _has_draw_support_hb and is_pokemon(c) and getattr(c, "evolves_from", None):
                _evo_name = (getattr(c, "name", "") or "").strip()
                _can_use_now = False
                # ドロンチ: ドラメシヤが場にいれば即進化可能
                if _evo_name == "ドロンチ" and _has_drameshiya_on_field and state and state.turn_count >= 2:
                    _can_use_now = True
                # ドラパルトex: ドロンチが場にいれば即進化可能
                elif _evo_name == "ドラパルトex":
                    _has_doronchi_for_evo = any(
                        (getattr(fc, "name", "") or "").strip() == "ドロンチ" for fc in field_cards
                    )
                    if _has_doronchi_for_evo and state and state.turn_count >= 2:
                        _can_use_now = True
                if not _can_use_now:
                    return 1000.0  # リーリエで流される+即使えない → 低優先
            # ドラパルトex: 進化できる状態（ドロンチが場にいる or ふしぎなアメ+ドラメシヤ）なら高優先
            # 1ターン目は進化不可なので低優先。それ以外で進化手段なしも低優先。
            if c_name == "ドラパルトex":
                # 1ターン目は進化できない → 手札で腐る
                _is_first_turn = state is not None and getattr(state, "turn_count", 99) <= 2
                if _is_first_turn:
                    return 3000.0 + strength
                _has_ame = any(
                    (getattr(hc, "id", "") or "") == FUSHIGI_NA_AME for hc in p.hand
                )
                _has_doronchi_field = any(
                    (getattr(fc, "name", "") or "") == "ドロンチ" for fc in field_cards
                )
                _has_dorameshiya_field = any(
                    (getattr(fc, "name", "") or "") in ("ドラメシヤ", "ドロンチ") for fc in field_cards
                )
                if _has_doronchi_field:
                    return 14000.0 + strength  # ドロンチから即進化可能
                elif _has_ame and _has_dorameshiya_field:
                    return 13000.0 + strength  # アメ+ドラメシヤで即進化可能
                else:
                    return -5000.0  # 進化手段なし → 持ってこない（次ターンに取る）
            # ドロンチ（ドラメシヤから進化→ていさつしれい、ドラパルトexへの進化元）
            if c_name == "ドロンチ":
                # 手札+場のドロンチが2枚以上あれば十分
                _doronchi_in_hand = sum(
                    1 for hc in p.hand if (getattr(hc, "name", "") or "").strip() == "ドロンチ"
                )
                _doronchi_on_field = sum(
                    1 for fc in field_cards if (getattr(fc, "name", "") or "").strip() == "ドロンチ"
                )
                if _doronchi_in_hand + _doronchi_on_field >= 2:
                    return 2000.0 + strength  # ドロンチ2枚確保済み → 他を優先
                _has_dorameshiya_field = any(
                    (getattr(fc, "name", "") or "") == "ドラメシヤ" for fc in field_cards
                )
                if _has_dorameshiya_field:
                    return 8000.0 + strength
                return 3000.0 + strength
            # ドラメシヤ（ベンチに出して展開）
            if c_name == "ドラメシヤ":
                # 場+手札に2体以上いれば十分 → 他を優先
                _drameshiya_count = sum(
                    1 for fc in field_cards
                    if (getattr(fc, "name", "") or "").strip() in ("ドラメシヤ", "ドロンチ", "ドラパルトex")
                ) + sum(
                    1 for hc in p.hand
                    if is_pokemon(hc) and (getattr(hc, "name", "") or "").strip() == "ドラメシヤ"
                )
                if _drameshiya_count >= 2:
                    return 3000.0  # 2体以上 → スボミー/ヨマワルより低く
                return 6000.0 + strength
            # スボミー（むずむずかふん=グッズロック、壁役）
            if c_name == "スボミー":
                return 5000.0
            # ヨマワル
            if c_name == "ヨマワル":
                return 4500.0
            # サマヨール/ヨノワール（カースドボム）
            if c_name == "サマヨール":
                _has_yomawaru_field = any(
                    (getattr(fc, "name", "") or "").strip() == "ヨマワル" for fc in field_cards
                )
                return 5000.0 if _has_yomawaru_field else 2000.0
            if c_name == "ヨノワール":
                _has_samayoru_field = any(
                    (getattr(fc, "name", "") or "").strip() == "サマヨール" for fc in field_cards
                )
                return 5000.0 if _has_samayoru_field else 1500.0
            # ニャースex: おくのてキャッチ→サポートサーチが強力
            # サポートが手札にあっても、おくのてキャッチで追加サポート(アカマツ等)を取れる価値は高い
            if c_name == "ニャースex":
                if len(p.bench) >= 5:
                    return -5000.0
                if _has_draw_support_hb:
                    return 2000.0  # ドローサポートあり → ドラメシヤ(6000)より低く
                if state is not None and not state.support_used_this_turn:
                    return 14000.0  # サポートなし+未使用 → おくのてキャッチ連携
                return 9000.0
            # キチキギスex: HBで取るバリューは基本的に低い（ベンチ枠を使う割に即効性がない）
            # ニャースex（おくのてキャッチ）やドラメシヤ（ドロンチ進化の基盤）を優先
            if c_name == "キチキギスex":
                if len(p.bench) >= 5:
                    return -5000.0  # ベンチ満杯 → 出せない
                return -1000.0  # HBで積極的に取る必要なし
            if c_name == "マシマシラ":
                return -5000.0  # ベンチに出さない
            return 1000.0 + strength

        # ベンチが空で手札にたねもない（種切れリスク）→ たねポケモンを最優先
        # 進化ポケモンはベンチに出せないので種切れ防止にならない
        if bench_truly_empty:
            if is_basic_pokemon(c) or not getattr(c, "evolves_from", None):
                if c_name == "リオル" and has_mega_lucario_in_hand_or_deck:
                    return 20000.0 + strength  # リオル最優先（次ターン進化→はどうづき）
                return 15000.0 + strength  # それ以外のたねも高優先
            # 進化ポケモンは種切れ防止にならないので低スコア
            return strength
        # エンジンペア（ルナトーン+ソルロック）が揃っているか
        _engine_names = {"ルナトーン", "ソルロック"}
        all_field_names = {(getattr(fc, "name", "") or "") for fc in field_cards}
        # 手札も含めてチェック（これから出せる）
        hand_names = {(getattr(hc, "name", "") or "") for hc in p.hand if is_pokemon(hc)}
        has_lunatone = "ルナトーン" in (all_field_names | hand_names)
        has_solrock = "ソルロック" in (all_field_names | hand_names)
        engine_complete = has_lunatone and has_solrock

        for field_card in field_cards:
            if _can_evolve_onto(field_card, c):
                if is_stage2_pokemon(c):
                    return 10000.0 + strength
                # エンジン未完成なら進化先より低くする（エンジン確保が先）
                if not engine_complete:
                    return 3000.0 + strength
                return 5000.0 + strength
        if is_basic_pokemon(c) or not getattr(c, "evolves_from", None):
            c_name = (getattr(c, "name", "") or "").strip()
            # エンジンペアの片方が欠けているなら最優先で持ってくる
            # ルナサイクルのドローエンジンは序盤の安定性に直結
            if c_name == "ルナトーン" and not has_lunatone:
                return 12000.0 + strength
            if c_name == "ソルロック" and not has_solrock:
                return 11000.0 + strength
            # 場にリオルラインがいない＋メガルカリオexが手札/山札にある → リオル最優先
            # メガルカリオexが手札にある場合はエンジンより優先（すぐ進化できる）
            if c_name == "リオル" and not has_rioru_line_in_field and has_mega_lucario_in_hand_or_deck:
                _has_mega_in_hand = any(
                    is_pokemon(hc) and (getattr(hc, "name", "") or "") in ("メガルカリオex", "メガルカリオ")
                    for hc in p.hand
                )
                if _has_mega_in_hand:
                    return 13000.0 + strength  # 手札にメガルカリオex → エンジンより優先
                return 6000.0 + strength
            if c_name == "リオル" and has_mega_lucario_in_hand_or_deck:
                return 3000.0 + strength
            return 1000.0 + strength
        return strength

    best = max(pool, key=lambda x: (score(x[0], x[1]), -x[0]))
    return best


def _is_lunatone_card(card) -> bool:
    cid = (getattr(card, "id", "") or "").strip()
    name = (getattr(card, "name", "") or "").strip()
    return cid.startswith("runaton") or name == "ルナトーン"


def _haipaboru_lunatone_discard_bonus(p: PlayerState, hand_without_haipaboru: list[tuple[int, object]], c) -> float:
    """
    2 枚目以降のルナトーンは実質使わない前提で捨てやすくする。
    場にルナトーンがいる、または手札に同名が複数あるときにボーナス。
    """
    if not _is_lunatone_card(c):
        return 0.0
    on_field = 0
    if p.active and _is_lunatone_card(p.active.card):
        on_field += 1
    for bp in p.bench:
        if _is_lunatone_card(bp.card):
            on_field += 1
    hand_luna = sum(1 for _i, hc in hand_without_haipaboru if _is_lunatone_card(hc))
    if on_field >= 1:
        return 1500.0
    if hand_luna >= 2:
        return 1200.0
    return 0.0


def _haipaboru_judge_vs_lillie_adjustment(
    state: GameState,
    hand_without_haipaboru: list[tuple[int, object]],
    c,
) -> float:
    """
    相手手札が 5 枚以上のとき、ジャッジマンはリーリエの決心より価値が下がりやすい（お互い 4 枚固定）。
    両方手札にあるならジャッジマンを捨てやすく、リーリエを捨てにくくする。
    """
    has_lillie = any((getattr(hc, "id", "") or "") == RIRIE_NO_KESSHIN for _i, hc in hand_without_haipaboru)
    has_judge = any((getattr(hc, "id", "") or "") == JUDGE for _i, hc in hand_without_haipaboru)
    if not (has_lillie and has_judge):
        return 0.0
    opp = state.defending_player_state()
    if len(opp.hand) < 5:
        return 0.0
    cid = getattr(c, "id", "") or ""
    if cid == JUDGE:
        return 2200.0
    if cid == RIRIE_NO_KESSHIN:
        return -2600.0
    return 0.0


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
        _target_name_bench = (getattr(target.card, "name", "") or "").strip()
        _is_luna_bench = _target_name_bench == "ルナトーン"
        if _is_luna_bench and not getattr(state, "_ko_plan_executing", False):
            return False
        # ドラパルトデッキ: エネ付与禁止ポケモン（スボミー等）
        if _target_name_bench in DRAPA_ENERGY_BANNED_NAMES:
            if is_dragapult_deck_for_player(state, state.current_player):
                return False
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
    _active_name_ae = (getattr(p.active.card, "name", "") or "").strip()
    # ルナトーンにはKOプランナー経由以外でエネを付けない
    if _active_name_ae == "ルナトーン" and not getattr(state, "_ko_plan_executing", False):
        return False
    # ドラパルトデッキ: エネ付与禁止ポケモン（スボミー等）
    if _active_name_ae in DRAPA_ENERGY_BANNED_NAMES:
        if is_dragapult_deck_for_player(state, state.current_player):
            return False
    p.active.attached_energy += 1
    p.active.attached_energy_types.append(slot_type)
    p.hand.pop(hand_index)
    state.energy_attached_this_turn = True
    state.log(f"{state.player_name(state.current_player)}: エネルギーを 1 つ付与（バトル場の {p.active.card.name}、エネルギー {p.active.attached_energy} 個）")
    return True


def use_potion(state: GameState, hand_index: int) -> bool:
    """きずぐすり（id=potion）を使用して自分のバトル場のポケモンを回復。手札の hand_index 番目がきずぐすりのときだけ実行。"""
    if getattr(state, "goods_locked_next_turn", None) == state.current_player:
        return False
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
    cap = get_effective_max_hp(state, p.active.card)
    p.active.hp = min(p.active.hp + amount, cap)
    p.discard.append(p.hand.pop(hand_index))
    when_drawn = "（今ターンのドローで引いた）" if card in state.drawn_this_turn else ""
    state.log(f"{state.player_name(state.current_player)}: きずぐすりを使用 → バトル場のポケモンを {before} → {p.active.hp} に回復{when_drawn}")
    return True





def _use_fight_gong(state: GameState, hand_index: int) -> bool:
    """ファイトゴングのカード効果を実行する。"""
    p = state.active_player_state()
    card = p.hand[hand_index]
    fighting_basic = [c for c in p.deck if is_pokemon(c) and getattr(c, "pokemon_type", None) == "fighting" and getattr(c, "evolution_stage", None) == "basic"]
    # ファイトゴングで取れるのは「基本闘エネルギー」のみ（ロック闘などの特殊エネルギーは対象外）。
    def _is_basic_fighting_energy(x) -> bool:
        if not is_energy(x):
            return False
        cid = getattr(x, "id", "") or ""
        name = getattr(x, "name", "") or ""
        return cid in ("basic-energy-fighting", "kihontouenerugi") or name == "基本闘エネルギー"

    fighting_energy = [c for c in p.deck if _is_basic_fighting_energy(c)]
    candidates = fighting_basic + fighting_energy
    if candidates:
        weights = state.get_weights_for_player(state.current_player)
        deck_index = state.deck_indices[state.current_player] if state.deck_indices else 0
        has_energy_in_hand = any(is_energy(c) for c in p.hand)

        # 盤面に意味があるときだけセットロジック ON（手札 or 場にエンジンペアの片割れがいる）
        if _has_engine_pair_member_in_hand_or_field(p):
            _fg_has_lunatone = _player_has_pokemon_by_name_or_id(p, "ルナトーン", {"runaton"})
            _fg_has_solrock = _player_has_pokemon_by_name_or_id(p, "ソルロック", {"sorurokku-mc-372"})
        else:
            _fg_has_lunatone = False
            _fg_has_solrock = False

        _fg_engine_complete = _fg_has_lunatone and _fg_has_solrock

        # メガルカリオexが手札にあり、リオルが場にも手札にもいない場合のフラグ
        _has_mega_lucario_in_hand = any(
            is_pokemon(c) and "メガルカリオ" in (getattr(c, "name", "") or "")
            for c in (p.hand or [])
        )
        _has_rioru_in_hand_or_field = False
        if _has_mega_lucario_in_hand:
            for c in (p.hand or []):
                if is_pokemon(c) and (getattr(c, "name", "") or "").strip() == "リオル":
                    _has_rioru_in_hand_or_field = True
                    break
            if not _has_rioru_in_hand_or_field:
                if getattr(p, "active", None) and (getattr(p.active.card, "name", "") or "").strip() == "リオル":
                    _has_rioru_in_hand_or_field = True
                if not _has_rioru_in_hand_or_field:
                    for bp in (p.bench or []):
                        if (getattr(bp.card, "name", "") or "").strip() == "リオル":
                            _has_rioru_in_hand_or_field = True
                            break
        _need_rioru_for_mega = _has_mega_lucario_in_hand and not _has_rioru_in_hand_or_field

        # 場にリオルライン（リオル/メガルカリオex等）がいるか
        def _has_rioru_line_in_field() -> bool:
            _rioru_names = {"リオル", "メガルカリオex", "メガルカリオ", "ルカリオ"}
            if getattr(p, "active", None) and getattr(p.active, "card", None):
                if (getattr(p.active.card, "name", "") or "") in _rioru_names:
                    return True
            for bp in (p.bench or []):
                c = getattr(bp, "card", None)
                if c and (getattr(c, "name", "") or "") in _rioru_names:
                    return True
            # 手札にリオルがあれば次に出せる
            for c in (p.hand or []):
                if is_pokemon(c) and (getattr(c, "name", "") or "") in _rioru_names:
                    return True
            return False

        _fg_has_rioru_line = _has_rioru_line_in_field()

        # 展開が十分か判定: リオルライン2体+エンジン揃い+マクノシタがいればたねは不要
        _rioru_names = {"リオル", "メガルカリオex", "メガルカリオ", "ルカリオ"}
        _field_names = []
        if p.active:
            _field_names.append(getattr(p.active.card, "name", ""))
        for bp in (p.bench or []):
            _field_names.append(getattr(bp.card, "name", ""))
        _rioru_line_count = sum(1 for n in _field_names if n in _rioru_names)
        _has_makunoshita_line = any(n in ("マクノシタ", "ハリテヤマ") for n in _field_names)
        _field_sufficient = _fg_engine_complete and _rioru_line_count >= 2 and _has_makunoshita_line

        def _faitogongu_score(c):
            w = get_faitogongu_fetch_weight(weights, c, deck_index)
            if not has_energy_in_hand and is_energy(c):
                w += 1000.0
                # HBでリオルを後から取れる+サポートなし+タンカありでルナサイクル2回分確約
                # → エネ取得でドロー力を確保する方が圧倒的に有利
                _has_hb = any(getattr(hc, "id", "") == HYPER_BALL for hc in p.hand)
                _has_tanka = any(getattr(hc, "id", "") == YORU_NO_TANKA for hc in p.hand)
                _has_draw_support = any(
                    is_support(hc) and getattr(hc, "id", "") in (
                        RIRIE_NO_KESSHIN, ZEIYU, HAKASE_NO_KENKYU,
                        "hakasenokenkyuufutouhakase", JUDGE, "nemo"
                    )
                    for hc in p.hand
                )
                if _has_hb and not _has_draw_support:
                    w += 500.0  # HBでポケモン取れるのでエネ優先
                if _has_tanka and _fg_engine_complete:
                    w += 500.0  # タンカ+エンジン揃い → ルナサイクル2回分確約
            # 展開が十分ならエネルギー優先（ルナサイクルのコスト+手張りに使える）
            if _field_sufficient and is_energy(c):
                w += 800.0
            if not rules_only_for_player(state, state.current_player):
                w += get_fetch_bonus_for_card(deck_index, getattr(c, "id", "") or "")
            # 条件付きカードは online 評価が有効なときだけ加点する（A/B 比較用に無効化可能）。
            if use_online_eval(state):
                if is_pokemon(c) and is_finisher(c):
                    online = is_online(c, state)
                    w += _ONLINE_FINISHER_POS_BONUS * online
                    w -= _ONLINE_FINISHER_NEG_PENALTY * (1.0 - online)
                if is_pokemon(c) and is_main_attacker(c):
                    w += _ONLINE_MAIN_ATTACKER_BONUS * is_online(c, state)
            # エンジンペアのセット優先: 片方だけ場にいるとき強く、両方いないとき少し
            if is_engine_pair_member(c):
                c_name = getattr(c, "name", "") or ""
                if c_name == "ルナトーン":
                    if _fg_has_solrock and (not _fg_has_lunatone):
                        w += 600.0
                    elif (not _fg_has_solrock) and (not _fg_has_lunatone):
                        w += 250.0
                elif c_name == "ソルロック":
                    if _fg_has_lunatone and (not _fg_has_solrock):
                        w += 600.0
                    elif (not _fg_has_lunatone) and (not _fg_has_solrock):
                        w += 250.0
            # メガルカリオexが手札にあるのにリオルが場にも手札にもない → リオル最優先
            if _need_rioru_for_mega and is_pokemon(c):
                c_name = (getattr(c, "name", "") or "").strip()
                if c_name == "リオル":
                    w += 2000.0  # ルナトーンより高い最優先ボーナス
            # エンジンペアが揃っているなら、リオルを優先（メガルカリオex育成）
            # ソルロック/ルナトーンをこれ以上取る必要はない
            if _fg_engine_complete and is_pokemon(c):
                c_name = (getattr(c, "name", "") or "").strip()
                if c_name == "リオル" and not _fg_has_rioru_line:
                    w += 800.0  # エンジン揃い＋リオルなし → リオル最優先
                elif c_name == "リオル" and _rioru_line_count <= 1:
                    w += 400.0  # エンジン揃い＋リオル1体 → 2体目も優先
                elif c_name == "リオル":
                    w -= 500.0  # リオルライン2体以上 → もう不要
                elif c_name == "マクノシタ" and not _has_makunoshita_line:
                    w += 600.0  # マクノシタがいない → ハリテヤマ育成用で優先
                elif c_name == "マクノシタ":
                    w += 200.0
            return (w, -p.deck.index(c))
        chosen = max(candidates, key=_faitogongu_score)
        p.deck.remove(chosen)
        p.hand.append(chosen)
        state.drawn_this_turn.append(chosen)
        random.shuffle(p.deck)
        mark_own_deck_shuffled(state)
        mark_deck_searched(state)
        p.discard.append(p.hand.pop(hand_index))
        state.log(f"{state.player_name(state.current_player)}: ファイトゴングを使用 → 山札から {_card_label(chosen)} を手札に加えた")
        return True
    return False



def _use_pokepad(state: GameState, hand_index: int) -> bool:
    """ポケパッドのカード効果を実行する。"""
    p = state.active_player_state()
    card = p.hand[hand_index]
    no_rule = [c for c in p.deck if is_pokemon(c) and not getattr(c, "has_rule", False)]
    if no_rule:
        weights = state.get_weights_for_player(state.current_player)
        deck_index = state.deck_indices[state.current_player] if state.deck_indices else 0
        allow_duplicate_ids = get_allow_duplicate_bench_ids(deck_index)

        def _card_key(x) -> str:
            return getattr(x, "id", None) or getattr(x, "name", "") or ""

        hand_keys = {_card_key(c) for c in (p.hand or []) if is_pokemon(c)}
        active_key = _card_key(getattr(p, "active", None).card) if getattr(p, "active", None) else ""
        bench_keys = {_card_key(bp.card) for bp in (p.bench or []) if getattr(bp, "card", None) and is_pokemon(bp.card)}

        # 盤面に意味があるときだけセットロジック ON（手札 or 場にエンジンペアの片割れがいる）
        if _has_engine_pair_member_in_hand_or_field(p):
            _pp_has_lunatone = _player_has_pokemon_by_name_or_id(p, "ルナトーン", {"runaton"})
            _pp_has_solrock = _player_has_pokemon_by_name_or_id(p, "ソルロック", {"sorurokku-mc-372"})
        else:
            _pp_has_lunatone = False
            _pp_has_solrock = False
        if len(p.bench) <= 2:
            import sys as _sys

        def _pokepaddo_score(c):
            w = get_pokepaddo_fetch_weight(weights, c, deck_index)
            if not rules_only_for_player(state, state.current_player):
                w += get_fetch_bonus_for_card(deck_index, getattr(c, "id", "") or "")

            ck = _card_key(c)
            c_name = getattr(c, "name", "") or ""

            # すでに「同じ種別」を持っているほど少しだけ優先度を落とす（強い -10000 禁止はしない）。
            owned_cards: list[object] = []
            owned_cards.extend([x for x in (p.hand or []) if is_pokemon(x)])
            if getattr(getattr(p, "active", None), "card", None) and is_pokemon(getattr(p.active, "card")):
                owned_cards.append(p.active.card)
            owned_cards.extend([bp.card for bp in (p.bench or []) if getattr(bp, "card", None) and is_pokemon(bp.card)])

            def _species(x: object) -> str:
                nm = getattr(x, "name", "") or ""
                if nm in ("リオル", "ルナトーン", "ソルロック", "マクノシタ"):
                    return nm
                return _card_key(x)

            species = _species(c)
            existing_count = sum(1 for x in owned_cards if _species(x) == species)

            # 優先度（同じ種別内での相対バイアス）:
            #  リオル1体目 > ルナトーン=ソルロック > リオル2体目 > マクノシタ
            dup_pen = get_pokepaddo_duplicate_penalty(
                weights,
                species=species,
                existing_count=existing_count,
                deck_index=deck_index,
            )

            # exact 同一カードの重なりは少しだけ追加で減点（ただし極端な禁止にはしない）。
            extra_pen = 0.0
            if ck and ck in hand_keys:
                extra_pen -= 50.0
            if ck and ck in bench_keys:
                extra_pen -= 50.0
                if ck not in allow_duplicate_ids:
                    extra_pen -= 80.0
            if ck and ck == active_key:
                extra_pen -= 70.0

            w += dup_pen + extra_pen

            # 条件付きカードは online 評価が有効なときだけ加点する（A/B 比較用に無効化可能）。
            if use_online_eval(state):
                if is_finisher(c):
                    online = is_online(c, state)
                    w += _ONLINE_FINISHER_POS_BONUS * online
                    w -= _ONLINE_FINISHER_NEG_PENALTY * (1.0 - online)
                if is_main_attacker(c):
                    w += _ONLINE_MAIN_ATTACKER_BONUS * is_online(c, state)

            # エンジンペアのセット優先: 片方だけ場にいるとき強く、両方いないとき少し
            # 既に場にいるエンジンは取る意味がない（2体目は不要）
            if is_engine_pair_member(c):
                c_name = getattr(c, "name", "") or ""
                if c_name == "ルナトーン":
                    if _pp_has_lunatone:
                        w -= 3000.0  # 既にいる→取らない
                    elif _pp_has_solrock:
                        w += 600.0
                    else:
                        w += 250.0
                elif c_name == "ソルロック":
                    if _pp_has_solrock:
                        w -= 3000.0  # 既にいる→取らない
                    elif _pp_has_lunatone:
                        w += 600.0
                    else:
                        w += 250.0

            # リオル進化ライン（リオル＋ルカリオ＋メガルカリオex）を同一グループとして数える
            def _is_rioru_line(x) -> bool:
                n = getattr(x, "name", "") or ""
                return n in ("リオル", "メガルカリオex", "メガルカリオ", "ルカリオ")
            rioru_line_count = sum(1 for x in owned_cards if _is_rioru_line(x))
            has_solrock_line = any(_species(x) == "ソルロック" for x in owned_cards)

            # エンジン揃い＋リオルラインが足りない → リオル優先（メガルカリオexで殴るために必須）
            _pp_engine_complete = _pp_has_lunatone and _pp_has_solrock
            if _pp_engine_complete and species == "リオル" and rioru_line_count <= 1:
                w += 1500.0
            elif rioru_line_count >= 1 and has_solrock_line:
                if species == "リオル" and rioru_line_count == 1:
                    w += 520.0   # ライン 2 体目のリオルは優先
                elif species == "リオル" and rioru_line_count >= 2:
                    w -= 800.0   # ライン 3 体目以降は大きく減点
                elif species == "ソルロック" and existing_count >= 1:
                    w -= 520.0
            # リオルライン 2 体以上いるときはマクノシタ（ハリテヤマ育成）を優先
            if rioru_line_count >= 2 and species == "マクノシタ" and existing_count == 0:
                w += 400.0

            # ベンチが満杯でベンチに出せないたねポケモンは大幅減点
            if is_pokemon(c) and not getattr(c, "evolves_from", None) and len(p.bench) >= 5:
                # 進化先がフィールドにいない純粋なたねは出す場所がない
                _can_evolve_onto_field = any(
                    (getattr(fc, "name", "") or "").strip() == (getattr(c, "name", "") or "").strip()
                    for fc in owned_cards
                )
                if not _can_evolve_onto_field:
                    w -= 5000.0

            # ドラパルトexデッキ: 進化ライン優先度
            from .deck_strategies import is_dragapult_deck as _is_drapa_pp
            if _is_drapa_pp(deck_index):
                _pp_field_cards = (
                    ([p.active.card] if getattr(p, "active", None) else [])
                    + [bp.card for bp in (p.bench or [])]
                )
                _pp_field_names = {(getattr(fc, "name", "") or "").strip() for fc in _pp_field_cards}
                _has_drameshiya_field = "ドラメシヤ" in _pp_field_names
                _has_yomawaru_field = "ヨマワル" in _pp_field_names
                _has_samayoru_field = "サマヨール" in _pp_field_names
                _has_yonowaru_field = "ヨノワール" in _pp_field_names
                # 1ターン目は進化ポケモンの優先度を下げる（進化できないので手札で腐る）
                _is_evo = bool(getattr(c, "evolves_from", None))
                if _is_evo and state is not None and getattr(state, "turn_count", 99) <= 1:
                    w -= 3000.0
                # 場に進化元がいない進化カードは低優先（すぐに使えない）
                # 進化元がいれば即進化→ていさつしれいで高優先
                if c_name == "ドロンチ":
                    if _has_drameshiya_field:
                        w += 5000.0  # ドラメシヤ→ドロンチ進化→ていさつしれい（最優先）
                    else:
                        w -= 3000.0
                if c_name == "サマヨール" and not _has_yomawaru_field:
                    w -= 3000.0
                if c_name == "ヨノワール" and not _has_samayoru_field:
                    w -= 3000.0
                # ドラメシヤ: ドラパルトexラインの基盤 → 高優先
                _drapa_line_count = sum(
                    1 for fn in _pp_field_names
                    if fn in ("ドラメシヤ", "ドロンチ", "ドラパルトex")
                )
                if c_name == "ドラメシヤ":
                    if _drapa_line_count >= 2:
                        w += 1000.0  # ラインが2体以上 → スボミー・ヨマワルより低め
                    else:
                        w += 3000.0  # ラインが足りない → 最優先で展開
                # 場にドラパルトラインが2体以上: スボミー > ヨマワル > ドラメシヤの順
                if _drapa_line_count >= 2:
                    _has_subomii_field = "スボミー" in _pp_field_names
                    if c_name == "スボミー" and not _has_subomii_field:
                        w += 2500.0  # むずむずかふん要員
                    elif c_name == "ヨマワル" and not _has_yomawaru_field:
                        w += 2000.0  # カースドボムライン
                if c_name == "サマヨール" and _has_yomawaru_field and not _has_samayoru_field:
                    w += 2000.0  # ヨマワルから進化可能 → サマヨール最優先
                elif c_name == "ヨノワール" and _has_samayoru_field:
                    w += 1500.0  # サマヨールから進化可能 → ヨノワール優先

            return (w, -p.deck.index(c))
        chosen = max(no_rule, key=_pokepaddo_score)
        p.deck.remove(chosen)
        p.hand.append(chosen)
        state.drawn_this_turn.append(chosen)
        random.shuffle(p.deck)
        mark_own_deck_shuffled(state)
        mark_deck_searched(state)
        p.discard.append(p.hand.pop(hand_index))
        state.log(f"{state.player_name(state.current_player)}: ポケパッドを使用 → 山札から {_card_label(chosen)} を手札に加えた")
        return True
    return False

def _use_hyper_ball(state: GameState, hand_index: int) -> bool:
    """ハイパーボールのカード効果を実行する。use_trainer_goodsから呼び出される。"""
    p = state.active_player_state()
    card = p.hand[hand_index]
    # ドラパルトデッキ: ドローサポートが手札にあり未使用 → HBをスキップ（リーリエで引いてから判断）
    # ニャースexが山札にある場合のみ例外（HB→ニャースex→おくのてキャッチ連携）
    if is_dragapult_deck_for_player(state, state.current_player) and not state.support_used_this_turn:
        _has_draw_supp_hb = any(
            is_support(c) and (getattr(c, "id", "") or "") in (
                RIRIE_NO_KESSHIN, ZEIYU, HAKASE_NO_KENKYU, HIKARI,
            )
            for c in p.hand
        )
        _nyarth_in_deck_hb = any(
            "ニャースex" in (getattr(dc, "name", "") or "")
            for dc in p.deck
        )
        if _has_draw_supp_hb and not _nyarth_in_deck_hb:
            return False
    weights = state.get_weights_for_player(state.current_player)
    # 手札が少ないのにハイパーボールを打つと（実質 -2 枚前後になりやすく）ゲームのリソースが枯れやすい。
    # その場合、学習/重みが無いと「サポートを捨てる」確率が上がるため、サポートを強めに減点する。
    small_hand = len(p.hand) <= 4
    hand_without_haipaboru = [(i, p.hand[i]) for i in range(len(p.hand)) if i != hand_index]
    name_counts = {}
    for _i, c in hand_without_haipaboru:
        n = getattr(c, "name", "") or getattr(c, "id", "") or ""
        name_counts[n] = name_counts.get(n, 0) + 1
    support_count = sum(1 for _i, c in hand_without_haipaboru if is_support(c))
    support_indices = [i for i, c in hand_without_haipaboru if is_support(c)]
    only_support_idx = support_indices[0] if len(support_indices) == 1 else None
    # 「最後のサポートを捨てない」ための強めの禁止:
    # ハイパーボールは 2 枚トラッシュが必須。
    # 候補が「サポート1 + 非サポート1」だと、必ずサポートを捨てるため使わない。
    non_support_count = len(hand_without_haipaboru) - support_count
    if support_count == 1 and non_support_count == 1:
        return False
    # ドラパルトデッキ: HBで2枚捨てた結果リーリエの決心を捨てざるを得ない場合はHBを使わない
    # （リーリエがないと立て直せない。手札が少ない時は特に重要）
    if is_dragapult_deck_for_player(state, state.current_player):
        _has_ririe_hb = any(
            is_support(hc) and (getattr(hc, "id", "") or "") == RIRIE_NO_KESSHIN
            for _, hc in hand_without_haipaboru
        )
        if _has_ririe_hb:
            # リーリエ以外で捨てられるカードが2枚以上あるか
            # （保護カード: リーリエ、アンフェアスタンプ、アカマツ）
            _protected_ids = {RIRIE_NO_KESSHIN, UNFAIR_STAMP, AKAMATSU}
            _discardable = sum(
                1 for _, hc in hand_without_haipaboru
                if not (is_support(hc) and (getattr(hc, "id", "") or "") in _protected_ids)
                and not (is_goods(hc) and (getattr(hc, "id", "") or "") == UNFAIR_STAMP)
            )
            if _discardable < 2:
                return False  # リーリエを捨てざるを得ない → HBを使わない
    # 持ってくるカードが手札に既にあるもの(同名)しかない場合、使う意味がない
    found = _find_pokemon_for_haipaboru(p, state)
    if found is None:
        return False
    _fetch_name = (getattr(found[1], "name", "") or "").strip()
    # 取得先と同名カードが手札にある場合、同名カードをトラッシュに選ぶと無意味
    # （捨てて同じものを持ってくるだけ）→ その場合はHBを使わない
    _fetch_in_hand = any(
        is_pokemon(hc) and (getattr(hc, "name", "") or "").strip() == _fetch_name
        for _, hc in hand_without_haipaboru
    )
    if _fetch_in_hand:
        # 手札に同名がある → 2枚目が本当に必要か確認
        # ドロンチは複数必要なので除外
        if _fetch_name != "ドロンチ":
            return False  # 同名を捨てて同名を取る意味なし
    # 場のポケモンの進化先（メガルカリオex等）は捨てたくない
    field_cards = ([p.active.card] if p.active else []) + [bp.card for bp in p.bench]

    def _is_evolution_for_field(c) -> bool:
        if not is_pokemon(c) or not getattr(c, "evolves_from", None):
            return False
        # 場のカード＋手札のたねポケモンも対象（今後ベンチに出して進化できる）
        all_potential = list(field_cards) + [hc for hc in p.hand if is_pokemon(hc) and not getattr(hc, "evolves_from", None)]
        return any(_can_evolve_onto(fc, c) for fc in all_potential)

    # 手札に進化先（メガルカリオex等）があるなら、それを捨てるリスクがあるので
    # 取りたいカードが進化先と同等以上に重要でなければハイパーボールを使わない
    evo_in_hand = [c for _, c in hand_without_haipaboru if _is_evolution_for_field(c)]
    if evo_in_hand:
        found = _find_pokemon_for_haipaboru(p, state)
        if found:
            found_name = getattr(found[1], "name", "")
            # 取ろうとしているのがリオル等のたね → メガルカリオexを捨てるリスクに見合わない
            if not getattr(found[1], "evolves_from", None):
                # 非進化カードが2枚未満なら進化先を捨てることになる → 使わない
                safe = [c for _, c in hand_without_haipaboru if not _is_evolution_for_field(c) and not is_support(c)]
                if len(safe) < 2:
                    return False

    # 進化先を2枚とも捨てなければいけない場合、ハイパーボールを使わない
    non_evo_non_support = [
        (i, c) for i, c in hand_without_haipaboru
        if not _is_evolution_for_field(c) and not is_support(c)
    ]
    evo_cards_in_hand = [(i, c) for i, c in hand_without_haipaboru if _is_evolution_for_field(c)]
    # 捨てられる非進化カードが1枚以下 かつ 進化先がある → 進化先を捨てることになるので使わない
    if evo_cards_in_hand and len(non_evo_non_support) < 2:
        # サポートも含めて2枚捨てられるか確認
        safe_discards = len(non_evo_non_support) + max(0, support_count - 1)  # サポート1枚は残す
        if safe_discards < 2:
            return False

    # 手札の闘エネ枚数をカウント（全部捨てない保護用）
    energy_count_in_hand = sum(
        1 for _, c in hand_without_haipaboru
        if is_energy(c) and getattr(c, "energy_type", None) == "fighting"
    )

    scored = []
    for i, c in hand_without_haipaboru:
        discard_score = get_haipaboru_discard_weight(weights, c)
        if small_hand and is_support(c):
            # 「手札が少ない局面では、サポートを残した方が負けにくい」経験則をソフトに適用。
            # ただし、候補がサポートしかない場合にゼロ除外で詰まないよう "絶対禁止" ではなく強めの減点に留める。
            discard_score -= 2000.0
        # ドラパルトexデッキ: 炎・超エネルギーは絶対に捨てない（ファントムダイブの必須コスト）
        _drapa_deck_check_names = {"ドラメシヤ", "ドロンチ", "ドラパルトex"}
        _is_drapa_hb_discard = any(
            is_pokemon(hc) and (getattr(hc, "name", "") or "").strip() in _drapa_deck_check_names
            for _, hc in hand_without_haipaboru
        ) or any(
            is_pokemon(dc) and (getattr(dc, "name", "") or "").strip() in _drapa_deck_check_names
            for dc in p.deck
        )
        if _is_drapa_hb_discard and is_energy(c):
            etype_hb = getattr(c, "energy_type", None)
            if etype_hb in ("fire", "psychic"):
                # 手札にこのタイプのエネが何枚あるか
                _same_type_count = sum(
                    1 for _, hc in hand_without_haipaboru
                    if is_energy(hc) and getattr(hc, "energy_type", None) == etype_hb
                )
                # 場のドラパルトexラインでこのタイプのエネがまだ付いていないポケモン
                _drapa_needing = 0
                _drapa_line_names_hb = DRAPA_LINE_NAMES
                for _bp in ([p.active] if p.active else []) + list(p.bench):
                    _bp_name_hb = (getattr(_bp.card, "name", "") or "").strip()
                    if _bp_name_hb in _drapa_line_names_hb:
                        _bp_types_hb = list(getattr(_bp, "attached_energy_types", []) or [])
                        if etype_hb not in _bp_types_hb:
                            _drapa_needing += 1
                # このエネを付ければ次ターンFD完成するか
                _completes_fd = False
                for _bp in ([p.active] if p.active else []) + list(p.bench):
                    _bp_name_hb2 = (getattr(_bp.card, "name", "") or "").strip()
                    if _bp_name_hb2 in DRAPA_LINE_NAMES:
                        _bp_types_hb2 = list(getattr(_bp, "attached_energy_types", []) or [])
                        _bp_en_hb2 = getattr(_bp, "attached_energy", 0) or 0
                        _other_type = "psychic" if etype_hb == "fire" else "fire"
                        if _bp_en_hb2 >= 1 and _other_type in _bp_types_hb2 and etype_hb not in _bp_types_hb2:
                            _completes_fd = True
                            break
                # 夜のタンカがデッキ+手札にあるか（エネルギーをトラッシュから回収可能）
                _has_tanka = any(
                    (getattr(hc, "id", "") or "") == YORU_NO_TANKA
                    for _, hc in hand_without_haipaboru
                ) or any(
                    (getattr(dc, "id", "") or "") == YORU_NO_TANKA
                    for dc in p.deck
                )
                if _completes_fd:
                    discard_score -= 9000.0  # このエネでFD完成 → 絶対捨てない
                elif _same_type_count <= _drapa_needing:
                    discard_score -= 9000.0  # 必要数以下 → 捨てると攻撃不能
                elif _same_type_count <= 1 and not _has_tanka:
                    discard_score -= 9000.0  # 最後の1枚+タンカなし → 生命線
                elif _same_type_count <= 1 and _has_tanka:
                    discard_score += 300.0  # 最後の1枚だがタンカあり → 捨ててOK（回収可能）
                elif _same_type_count >= 2:
                    discard_score += 500.0  # 2枚以上 → 1枚は捨ててOK
                else:
                    discard_score -= 4000.0  # その他 → 慎重に
            elif etype_hb == "darkness":
                # ニャースexがバトル場で逃げ用に必要な場合は保護
                _nyarth_active = (
                    p.active and (getattr(p.active.card, "name", "") or "").strip() == "ニャースex"
                    and (getattr(p.active, "attached_energy", 0) or 0) == 0
                )
                if _nyarth_active:
                    discard_score -= 3000.0  # 逃げ用に温存
                else:
                    discard_score += 200.0  # 悪はファントムダイブに不要、むしろ捨てやすい
        # ドラパルトexデッキ: ドラパルトex/ドロンチ/ふしぎなアメは捨てない
        if _is_drapa_hb_discard and is_pokemon(c):
            c_name_hb = (getattr(c, "name", "") or "").strip()
            if c_name_hb == "ドラパルトex":
                discard_score -= 8000.0  # メインアタッカー
            elif c_name_hb == "ドロンチ":
                discard_score -= 7000.0  # 進化の要（ていさつしれい）
        if _is_drapa_hb_discard and is_goods(c) and (getattr(c, "id", "") or "") == FUSHIGI_NA_AME:
            # ドラパルトexが手札にあり場にドラメシヤがいれば即進化可能 → 超重要
            _has_drapa_in_hand = any(
                (getattr(hc, "name", "") or "").strip() == "ドラパルトex"
                for _, hc in hand_without_haipaboru
            )
            _has_drameshiya_field = any(
                (getattr(bp.card, "name", "") or "").strip() == "ドラメシヤ"
                for bp in ([p.active] if p.active else []) + list(p.bench or [])
            )
            if _has_drapa_in_hand and _has_drameshiya_field:
                discard_score -= 9000.0  # 即進化可能 → 絶対捨てない
            else:
                discard_score -= 3000.0
        # ドラパルトexデッキ: メイのはげまし/アカマツはエネ加速の生命線 → 捨てない
        if _is_drapa_hb_discard and is_support(c):
            c_id_hb = (getattr(c, "id", "") or "").strip()
            if c_id_hb == MEI_NO_HAGEMASHI:
                discard_score -= 4000.0  # メイのはげまし: Stage2にエネ2枚付ける唯一の手段
            elif c_id_hb == AKAMATSU:
                discard_score -= 15000.0  # アカマツ: FDの生命線（炎+超を確実に確保）
        # アンフェアスタンプは終盤の逆転カード → 基本的に捨てない（全デッキ共通）
        if is_goods(c) and (getattr(c, "id", "") or "") == UNFAIR_STAMP:
            discard_score -= 15000.0
        # リーリエの決心は最重要ドローサポート → 手札に2枚ある時のみ捨てOK
        if is_support(c) and (getattr(c, "id", "") or "") == RIRIE_NO_KESSHIN:
            _lillie_count = sum(
                1 for _, hc in hand_without_haipaboru
                if is_support(hc) and (getattr(hc, "id", "") or "") == RIRIE_NO_KESSHIN
            )
            if _lillie_count < 2:
                discard_score -= 15000.0  # 1枚しかない → 絶対捨てない
        # 闘エネは捨てにくくする。ルナサイクルのコスト+手張りに必要。
        # ただし手札全捨てサポート（ゼイユ等）があるなら、どうせ捨てるのでペナルティ不要。
        _has_trash_support = any(
            is_support(hc) and (getattr(hc, "id", "") or "") in (ZEIYU, HAKASE_NO_KENKYU, "hakasenokenkyuufutouhakase")
            for _, hc in hand_without_haipaboru
        )
        # エネは常に少し捨てにくい。手札全捨てサポートがある場合のみ免除。
        if is_energy(c) and getattr(c, "energy_type", None) == "fighting" and not _has_trash_support:
            discard_score -= 800.0
            # 手札にエネが2枚しかなく、他にエネ回収手段がない場合は
            # 両方捨てるとエネ枯渇で攻撃不能になる → 超大ペナルティ
            if energy_count_in_hand <= 2:
                _has_tanka = any(
                    (getattr(hc, "id", "") or "") == YORU_NO_TANKA
                    for _, hc in hand_without_haipaboru
                )
                _has_luna_cycle = (
                    _field_has_pokemon(p, "ルナトーン", "runaton")
                    and _field_has_pokemon(p, "ソルロック", "sorurokku-mc-372", "sorurokku")
                )
                if not _has_tanka and not _has_luna_cycle:
                    discard_score -= 3000.0
        # ポケモンいれかえは攻撃可能なアタッカーに交代できる重要カード。捨てにくくする。
        if is_goods(c) and (getattr(c, "effect", None) == "swap_active" or getattr(c, "id", "") in ("pokemon_irekae", POKEMON_IREKAE)):
            discard_score -= 1000.0
            # Fix J: メガルカリオexが場or手札にある場合、メガブレイブリセットトリックに必要なので追加保護
            _mega_present = any(
                "メガルカリオ" in (getattr(hc, "name", "") or "")
                for _, hc in hand_without_haipaboru
            ) or (
                p.active and "メガルカリオ" in (getattr(p.active.card, "name", "") or "")
            ) or any(
                "メガルカリオ" in (getattr(bp.card, "name", "") or "")
                for bp in (p.bench or [])
            )
            if _mega_present:
                discard_score -= 500.0
        # ドローサポート（ゼイユ、リーリエ等）: ルナサイクル不成立時は貴重なドロー源
        _DRAW_SUPPORT_IDS = (ZEIYU, RIRIE_NO_KESSHIN, HAKASE_NO_KENKYU, "hakasenokenkyuufutouhakase", JUDGE, "nemo")
        if is_support(c) and (getattr(c, "id", "") or "") in _DRAW_SUPPORT_IDS:
            _luna_engine_ok = (
                _field_has_pokemon(p, "ルナトーン", "runaton")
                and _field_has_pokemon(p, "ソルロック", "sorurokku-mc-372", "sorurokku")
            )
            if not _luna_engine_ok:
                discard_score -= 1000.0  # エンジン不成立 → ドロサポは唯一のドロー源
            else:
                discard_score -= 300.0  # エンジン成立でも一応保護
        # 夜のタンカはトラッシュからエネ/ポケモンを回収できる重要カード。捨てにくくする。
        if getattr(c, "id", "") == YORU_NO_TANKA:
            discard_score -= 2000.0
        # 場のポケモンに進化できるカードは捨てない(メガルカリオexを捨ててまた取るのは無駄)
        if _is_evolution_for_field(c):
            discard_score -= 5000.0
        # ex/メガポケモンは進化先でなくても重要（将来の切り札）
        elif is_pokemon(c) and (getattr(c, "is_ex", False) or getattr(c, "is_mega", False)):
            discard_score -= 3000.0
        c_name_for_count = getattr(c, "name", "") or getattr(c, "id", "") or ""
        c_count_in_hand = name_counts.get(c_name_for_count, 0)
        if c_count_in_hand >= 2:
            # 同名2枚以上は捨てやすい。ただしメガルカリオex等の重要進化カードは
            # 2枚捨てるとデッキに残り1枚以下になり、サイド落ちで詰むリスクがある。
            if _is_evolution_for_field(c):
                discard_score += 100  # 控えめにしか捨てやすくしない
            else:
                discard_score += 500
        if (not small_hand) and is_support(c) and support_count >= 2:
            discard_score += 300
        # --- データ分析から判明したカード別の調整 ---
        _cid = getattr(c, "id", "") or ""
        # スペシャルレッドカードは妨害手段として捨てにくい（統計: -7.1%）
        if _cid == SPECIAL_RED_CARD:
            discard_score -= 700.0
        # ロケット団の監視塔: 相手に無色ポケモンがいなければ不要 → 捨てやすい
        if _cid == "rokettodannokanshitou":
            opp = state.players[state.opponent()]
            opp_has_colorless = any(
                getattr(getattr(bp, "card", None), "pokemon_type", "") == "colorless"
                for bp in [opp.active] + opp.bench if bp is not None
            ) or any(
                getattr(dc, "pokemon_type", "") == "colorless"
                for dc in opp.deck if is_pokemon(dc)
            )
            if not opp_has_colorless:
                discard_score += 500.0
        # ふうせんは逃げるコスト0にする重要カード（統計: -5.3%）
        if _cid == FUUSEN:
            discard_score -= 500.0
        # Pattern 5: ボスの指令はターン0（先行1ターン目）のみ捨ててOK、それ以降は保護
        if _cid == BOSS_NO_SHIREI:
            if state.turn_count == 0:
                discard_score += 500.0
            else:
                discard_score -= 500.0
        # ミツルの思いやり/ポケパッドは捨てても勝率に影響しない（統計: +4.0%, +4.4%）
        if _cid in ("mitsurunoomoiyari", POKEPAD):
            discard_score += 400.0
        # 余剰ハイパーボール: メガルカリオexが手札/場にない場合はサーチ手段として保護
        if _cid == HYPER_BALL:
            _has_mega_anywhere = any(
                "メガルカリオ" in (getattr(hc, "name", "") or "")
                for hc in p.hand if is_pokemon(hc)
            ) or any(
                "メガルカリオ" in (getattr(bp.card, "name", "") or "")
                for bp in ([p.active] if p.active else []) + list(p.bench)
            )
            if _has_mega_anywhere:
                discard_score += 200.0  # メガルカリオ確保済み → HB捨ててOK
            else:
                discard_score -= 300.0  # メガルカリオ未確保 → HBでサーチが必要
        # Pattern 4: パワープロテイン基本保護（+30は常に貴重、エネルギー並みに保護）
        if _cid == "pawaapurotein":
            discard_score -= 800.0
        # パワープロテイン: 使えば相手を倒せる場合は捨てない
        if _cid == "pawaapurotein" and p.active and getattr(p.active.card, "pokemon_type", None) == "fighting":
            opp_hb = state.defending_player_state()
            if opp_hb.active and opp_hb.active.hp and opp_hb.active.hp > 0:
                # 手札の全パワプロ使用時のダメージを試算
                _pp_count = sum(1 for _, hc in hand_without_haipaboru if getattr(hc, "id", "") == "pawaapurotein")
                _n_now = getattr(state, "fighting_damage_plus_30_count_this_turn", 0)
                state.fighting_damage_plus_30_count_this_turn = _n_now + _pp_count
                try:
                    _dmg_with_pp = _max_effective_damage_for_attacker(state, p.active, opp_hb.active, state.current_player)
                finally:
                    state.fighting_damage_plus_30_count_this_turn = _n_now
                if _dmg_with_pp >= opp_hb.active.hp:
                    discard_score -= 5000.0  # 全パワプロ使えばKO可能 → 捨てない
        discard_score += _haipaboru_lunatone_discard_bonus(p, hand_without_haipaboru, c)
        discard_score += _haipaboru_judge_vs_lillie_adjustment(state, hand_without_haipaboru, c)
        scored.append((i, discard_score))

    scored.sort(key=lambda x: -x[1])
    score_by_index = {i: s for i, s in scored}
    to_discard_idx = [scored[0][0], scored[1][0]]
    i0, i1 = to_discard_idx
    # 同名の重要カード（進化ポケモン等）を2枚とも捨てない。
    # デッキに残り1枚以下になるとサイド落ちで詰むリスクがある。
    c0_name = (getattr(p.hand[i0], "name", "") or "").strip()
    c1_name = (getattr(p.hand[i1], "name", "") or "").strip()
    if c0_name == c1_name and c0_name and is_pokemon(p.hand[i0]) and getattr(p.hand[i0], "evolves_from", None):
        replacement = next(
            (idx for idx, _s in scored if idx != i0 and idx != i1),
            None,
        )
        if replacement is not None:
            to_discard_idx = [i0, replacement]
            i0, i1 = to_discard_idx
    # サポートを 2 枚とも捨てると次ターンのドロー源が枯れやすい → 可能なら 1 枚はサポート以外に差し替え
    if is_support(p.hand[i0]) and is_support(p.hand[i1]):
        ns_scored = [(idx, s) for idx, s in scored if not is_support(p.hand[idx])]
        if ns_scored:
            best_ns_i, _best_ns_s = max(ns_scored, key=lambda x: x[1])
            s0, s1 = score_by_index[i0], score_by_index[i1]
            i_hi = i0 if s0 >= s1 else i1
            to_discard_idx = [i_hi, best_ns_i]
    # 「手札の最後の一枚のサポートカードを捨てない」方針:
    # support_count==1 のとき、唯一のサポートがトラッシュ候補に入っていたら、
    # 代わりに非サポートを 1 枚選び直す。
    _protect_last_support = support_count == 1 and only_support_idx is not None and only_support_idx in to_discard_idx
    # 序盤のボスの指令は使えないので保護不要（捨ててOK��
    if _protect_last_support:
        _last_sup_id = getattr(p.hand[only_support_idx], "id", "") or ""
        if _last_sup_id == BOSS_NO_SHIREI and state.turn_count == 0:
            _protect_last_support = False
    if _protect_last_support:
        replacement = next(
            (idx for idx, _score in scored if idx != only_support_idx and idx not in to_discard_idx),
            None,
        )
        if replacement is not None:
            other_idx = to_discard_idx[0] if to_discard_idx[1] == only_support_idx else to_discard_idx[1]
            to_discard_idx = [other_idx, replacement]
    cards_to_log = [p.hand[i] for i in to_discard_idx]
    discard_names = ", ".join(_card_label(c) for c in cards_to_log)
    for i in sorted(to_discard_idx, reverse=True):
        p.discard.append(p.hand.pop(i))
    for c in cards_to_log:
        _log_choice(state, "haipaboru_discard", card_id=getattr(c, "id", None) or getattr(c, "name", ""))
    new_hi = p.hand.index(card)

    pokemon_found = _find_pokemon_for_haipaboru(p, state)

    if pokemon_found:
        i, c = pokemon_found
        p.deck.pop(i)
        p.hand.append(c)
        state.drawn_this_turn.append(c)
        random.shuffle(p.deck)
        mark_own_deck_shuffled(state)
        mark_deck_searched(state)
    p.discard.append(p.hand.pop(new_hi))
    add_label = f" → {_card_label(pokemon_found[1])}" if pokemon_found else ""
    state.log(
        f"{state.player_name(state.current_player)}: ハイパーボール → {discard_names} 捨て → {_card_label(pokemon_found[1])} を手札に"
        if pokemon_found
        else f"{state.player_name(state.current_player)}: ハイパーボール → {discard_names} 捨て（山札にポケモンなし）"
    )
    return True



def _use_power_protein(state, hand_index: int) -> bool:
    """パワープロテインのカード効果を実行する。"""
    p = state.active_player_state()
    card = p.hand[hand_index]
    if _is_first_player_first_turn(state):
        return False
    if not p.active or getattr(p.active.card, "pokemon_type", None) != "fighting":
        return False
    # 攻撃できない状態（ねむり・マヒ等）なら、ダメージ上振せのためのパワープロテインも打たない
    if getattr(p.active, "special_state", None) in ("sleep", "paralysis"):
        return False
    opp = state.defending_player_state()
    # 手札全捨てサポートの有無を先にチェック
    _TRASH_HAND_SUPPORT_IDS = (ZEIYU, HAKASE_NO_KENKYU, "hakasenokenkyuufutouhakase")
    will_trash_hand = not state.support_used_this_turn and any(
        is_support(c) and (getattr(c, "id", "") or "") in _TRASH_HAND_SUPPORT_IDS
        for c in p.hand
    )
    # 攻撃不可でもゼイユ等で捨てるくらいなら使う（+30バフはターン中有効）
    if not get_legal_attack_indices(state, p, opp) and not will_trash_hand:
        return False
    if True:  # 常に温存チェックを行う（手札全捨てサポートがあっても倒せないなら使わない）
        # 過剰ダメージ抑制:
        # このターンのパワープロテインで相手 active を倒せる場合でも、
        # 倒しきり量が大きい（=無駄になりやすい）なら次ターンに回す。
        hp = getattr(opp.active, "hp", None) if opp and getattr(opp, "active", None) else None
        n_before = getattr(state, "fighting_damage_plus_30_count_this_turn", 0)
        # プロテイン前の最大有効ダメージ
        dmg_before = _max_effective_damage_for_attacker(state, p.active, opp.active, state.current_player) if opp.active else 0
        # プロテイン後の最大有効ダメージ（一時的に count を増やして試算）
        n_after_tmp = n_before + 1
        state.fighting_damage_plus_30_count_this_turn = n_after_tmp
        try:
            dmg_after = _max_effective_damage_for_attacker(state, p.active, opp.active, state.current_player) if opp.active else 0
        finally:
            state.fighting_damage_plus_30_count_this_turn = n_before

        # 既にプロテインなしで倒せるなら、温存する
        # ただし手札全捨てサポートがある場合は使っても損しない
        if hp is not None and hp > 0 and dmg_before >= hp and not will_trash_hand:
            return False

        # プロテイン込みでも倒せないなら、温存する
        # ただし手札全捨てサポート（ゼイユ等）があるなら、どうせ捨てるので使った方が+30ダメ分得
        if hp is not None and hp > 0 and dmg_after < hp and not will_trash_hand:
            # 手札に残っているパワプロ枚数を数えて、全部使えば倒せるか試算
            pp_remaining = sum(
                1 for c in p.hand
                if getattr(c, "id", "") == "pawaapurotein" and c is not card
            )
            if pp_remaining > 0:
                state.fighting_damage_plus_30_count_this_turn = n_before + 1 + pp_remaining
                try:
                    dmg_all_pp = _max_effective_damage_for_attacker(state, p.active, opp.active, state.current_player) if opp.active else 0
                finally:
                    state.fighting_damage_plus_30_count_this_turn = n_before
                if dmg_all_pp >= hp:
                    # 全部使えば倒せる → この1枚は使う（残りも順次使われる）
                    pass
                else:
                    return False
            else:
                return False

        # プロテイン後に倒せてしまう場合のみ、倒し切り量が大きいなら温存する
        # ただし手札全捨てサポートがあるなら温存不要（どうせ捨てられる）
        if hp is not None and hp > 0 and dmg_after >= hp and not will_trash_hand:
            overkill = dmg_after - hp
            ratio = overkill / float(hp)
            threshold = float(os.getenv("DECKAI_POWERPROTEIN_OVERKILL_RATIO", "0.5"))
            if ratio >= threshold:
                return False

    n_before = getattr(state, "fighting_damage_plus_30_count_this_turn", 0)
    state.fighting_damage_plus_30_count_this_turn = n_before + 1
    p.discard.append(p.hand.pop(hand_index))
    n = state.fighting_damage_plus_30_count_this_turn
    state.log(f"{state.player_name(state.current_player)}: パワープロテインを使用 → この番、自分の闘ポケモンのワザダメージ+{30 * n}（{n} 枚目）")
    return True

def use_trainer_goods(
    state: GameState,
    hand_index: int,
    *,
    pokemon_catcher_bench_index: int | None = None,
) -> bool:
    """トレーナー（グッズ）の効果を実行。きずぐすり・いれかえ・どうぐ以外のアイテムを id / 名前で判定して処理する。"""
    # むずむずかふん: グッズ使用ロック中
    if getattr(state, "goods_locked_next_turn", None) == state.current_player:
        state.log(f"{state.player_name(state.current_player)}: 「むずむずかふん」の効果でグッズが使えない")
        return False
    p = state.active_player_state()
    if hand_index < 0 or hand_index >= len(p.hand):
        return False
    card = p.hand[hand_index]
    if not is_goods(card):
        return False
    cid = getattr(card, "id", "")
    if (getattr(card, "effect", None) in ("heal", "swap_active") or getattr(card, "is_tool", False)) and cid not in (
        SUPER_BALL, HYPER_BALL, FUSHIGI_NA_AME,
        UNFAIR_STAMP, "pawaapurotein", FIGHT_GONG, POKEPAD, YORU_NO_TANKA,
    ):
        return False
    name_ja = getattr(card, "name", "")

    if cid == FUSHIGI_NA_AME:
        if state.turn_count < 2:
            return False
        stage1_id = None
        stage2_card = None
        stage2_idx = None
        # 全ての stage2 候補を収集してからスコアリング
        _ame_candidates = []
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
            _s1id = (stage1_ref.evolves_from or "").strip()
            _ame_candidates.append((i, c, _s1id))
        # ドラパルトexデッキ: ドラパルトexを優先
        if _ame_candidates:
            if is_dragapult_deck_for_player(state, state.current_player):
                def _ame_score(tup):
                    _, cc, _ = tup
                    cn = (getattr(cc, "name", "") or "").strip()
                    if cn == "ドラパルトex":
                        return 1000
                    if cn == "ヨノワール":
                        return 500
                    return 0
                _ame_candidates.sort(key=_ame_score, reverse=True)
            stage2_idx, stage2_card, stage1_id = _ame_candidates[0]
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
        _ame_bp_candidates = []
        if p.active and is_basic_pokemon(p.active.card) and not getattr(p.active, "put_on_bench_this_turn", False):
            if _can_evolve_onto(p.active.card, stage1_ref):
                _ame_bp_candidates.append(p.active)
        for bp in p.bench:
            if not is_basic_pokemon(bp.card) or getattr(bp, "put_on_bench_this_turn", False):
                continue
            if _can_evolve_onto(bp.card, stage1_ref):
                _ame_bp_candidates.append(bp)
        if _ame_bp_candidates:
            # エネルギーが付いているたねを優先（進化後すぐ攻撃できる）
            target_bp = max(_ame_bp_candidates, key=lambda b: getattr(b, "attached_energy", 0))
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
                mark_own_deck_shuffled(state)
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
        mark_own_deck_shuffled(state)
        mark_deck_searched(state)
        p.discard.append(p.hand.pop(hand_index))
        if energies:
            state.log(f"{state.player_name(state.current_player)}: エレキジェネレーターを使用 → 山札上から 5 枚のうち基本雷エネルギー {len(energies)} 枚をベンチの {p.bench[best_bi].card.name} につけた")
        else:
            state.log(f"{state.player_name(state.current_player)}: エレキジェネレーターを使用（山札上 5 枚に基本雷エネルギーなし）")
        return True

    if cid == SUPER_BALL and p.deck:
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
        mark_own_deck_shuffled(state)
        mark_deck_searched(state)
        p.discard.append(p.hand.pop(hand_index))
        pokemon_label = _card_label(pokemon) if pokemon else ""
        state.log(
            f"{state.player_name(state.current_player)}: スーパーボールを使用 → 山札上から {look} 枚を見てポケモン 1 枚を手札に加えた → {pokemon_label}"
            if pokemon_label
            else f"{state.player_name(state.current_player)}: スーパーボールを使用 → 山札上から {look} 枚を見たがポケモンなし"
        )
        return True

    if cid == HYPER_BALL and len(p.hand) >= 3 and p.deck:
        return _use_hyper_ball(state, hand_index)


    if cid == UNFAIR_STAMP:
        # 前の相手の番に自分のポケモンがきぜつしていたら使える（タイプ不問）
        _any_ko = getattr(state, "any_ko_by_opponent_last_turn", [False, False])
        _ko_happened = _any_ko[state.current_player] or state.our_ko_by_damage_last_turn[state.current_player]
        if not _ko_happened:
            return False
        opp = state.defending_player_state()
        used_card = p.hand.pop(hand_index)
        p.deck.extend(p.hand)
        p.hand.clear()
        random.shuffle(p.deck)
        mark_own_deck_shuffled(state)
        opp.deck.extend(opp.hand)
        opp.hand.clear()
        random.shuffle(opp.deck)
        p_drawn = p.draw(5)
        opp_drawn = opp.draw(2)
        p.hand.extend(p_drawn)
        opp.hand.extend(opp_drawn)
        state.drawn_this_turn.extend(p_drawn)
        p.discard.append(used_card)
        state.log(f"{state.player_name(state.current_player)}: アンフェアスタンプを使用 → おたがい手札を山札にもどして切り、自分 5 枚・相手 2 枚ドロー")
        return True
    if cid == SPECIAL_RED_CARD:
        opp = state.defending_player_state()
        # カード効果: 相手のサイドの残り枚数が3枚以下のときにしか使えない
        if len(opp.prize_pile) > 3:
            return False
        used_card = p.hand.pop(hand_index)
        # 「ウラにして切り」を、手札順非公開のランダム順にして山札の下へ戻す挙動として扱う
        opp_hidden = list(opp.hand)
        random.shuffle(opp_hidden)
        opp.hand.clear()
        opp.deck[0:0] = opp_hidden
        opp_drawn = opp.draw(3)
        opp.hand.extend(opp_drawn)
        p.discard.append(used_card)
        state.log(
            f"{state.player_name(state.current_player)}: スペシャルレッドカードを使用 → "
            "相手の手札を山札の下にもどし、相手は 3 枚ドロー"
        )
        return True
    if cid == "pawaapurotein":
        return _use_power_protein(state, hand_index)
    if cid == FIGHT_GONG and p.deck:
        return _use_fight_gong(state, hand_index)

    if cid == POKEPAD and p.deck:
        return _use_pokepad(state, hand_index)

    if cid == YORU_NO_TANKA:
        pokemon_or_basic = [c for c in p.discard if is_pokemon(c) or (is_energy(c) and getattr(c, "energy_type", None) in _BASIC_ENERGY_TYPES)]
        if pokemon_or_basic:
            # 状況に応じて最適な回収先を選ぶ（将来は学習で最適化）
            # 今はシンプルなヒューリスティック:
            #   1. 手札にエネがない → 闘エネ優先（手張り+ルナサイクル用）
            #   2. それ以外 → 先頭のカード（ポケモンが先に来ることが多い）
            fighting_energy = [c for c in pokemon_or_basic if is_energy(c) and getattr(c, "energy_type", None) == "fighting"]
            has_energy_in_hand = any(is_energy(c) for c in p.hand)

            # エンジン（ソルロック/ルナトーン）が場に片方しかいない場合、もう片方をトラッシュから回収
            _has_luna_field = any(
                getattr(bp.card, "name", "") == "ルナトーン"
                for bp in ([p.active] if p.active else []) + list(p.bench)
            )
            _has_sol_field = any(
                getattr(bp.card, "name", "") == "ソルロック"
                for bp in ([p.active] if p.active else []) + list(p.bench)
            )
            _sol_in_trash = [c for c in pokemon_or_basic if is_pokemon(c) and getattr(c, "name", "") == "ソルロック"]
            _luna_in_trash = [c for c in pokemon_or_basic if is_pokemon(c) and getattr(c, "name", "") == "ルナトーン"]

            # ドラパルトexデッキ: ドラパルトexがトラッシュ＋ドロンチが場にいる → ドラパルトex最優先回収
            _drapa_in_trash = [c for c in pokemon_or_basic if is_pokemon(c) and (getattr(c, "name", "") or "").strip() == "ドラパルトex"]
            if is_dragapult_deck_for_player(state, state.current_player) and _drapa_in_trash:
                _has_doronchi_to_evolve = any(
                    (getattr(bp.card, "name", "") or "").strip() == "ドロンチ"
                    and not getattr(bp, "evolved_this_turn", False)
                    and not getattr(bp, "put_on_bench_this_turn", False)
                    for bp in ([p.active] if p.active else []) + list(p.bench)
                ) and state.turn_count >= 2
                _drapa_in_hand = any((getattr(c, "name", "") or "").strip() == "ドラパルトex" for c in p.hand)
                if _has_doronchi_to_evolve and not _drapa_in_hand:
                    chosen = _drapa_in_trash[0]
                    p.discard.remove(chosen)
                    p.hand.append(card_from_discard_to_hand(chosen))
                    p.discard.append(p.hand.pop(hand_index))
                    state.log(f"{state.player_name(state.current_player)}: 夜のタンカを使用 → トラッシュから {_card_label(chosen)} を手札に加えた")
                    return True

            # メガルカリオexがトラッシュにあり、場/手札にリオルがいて進化可能 → 最優先回収
            _mega_in_trash = [c for c in pokemon_or_basic if is_pokemon(c) and "メガルカリオ" in (getattr(c, "name", "") or "")]
            _has_rioru_to_evolve = any(
                (getattr(bp.card, "name", "") or "") == "リオル"
                and not getattr(bp, "evolved_this_turn", False)
                and not getattr(bp, "put_on_bench_this_turn", False)
                for bp in ([p.active] if p.active else []) + list(p.bench)
            ) and state.turn_count >= 2
            _mega_in_hand = any("メガルカリオ" in (getattr(c, "name", "") or "") for c in p.hand)
            if _mega_in_trash and _has_rioru_to_evolve and not _mega_in_hand:
                chosen = _mega_in_trash[0]
            elif _has_luna_field and not _has_sol_field and _sol_in_trash:
                # ルナトーンが場にいるがソルロックがいない → ソルロック回収（ルナサイクル完成）
                chosen = _sol_in_trash[0]
            elif _has_sol_field and not _has_luna_field and _luna_in_trash:
                # ソルロックが場にいるがルナトーンがいない → ルナトーン回収
                chosen = _luna_in_trash[0]
            elif fighting_energy and not has_energy_in_hand:
                chosen = fighting_energy[0]
            else:
                # 手札にエネが2枚以上あるならエネ回収は不要 → ポケモンを回収するか温存
                _energy_in_hand_count = sum(1 for c in p.hand if is_energy(c))
                pokemon_in_trash = [c for c in pokemon_or_basic if is_pokemon(c)]
                if pokemon_in_trash:
                    chosen = pokemon_in_trash[0]
                elif _energy_in_hand_count >= 2:
                    # エネ十分 → タンカを温存（後でエンジンやポケモン回収に使える）
                    return False
                else:
                    chosen = pokemon_or_basic[0]
            p.discard.remove(chosen)
            p.hand.append(card_from_discard_to_hand(chosen))
            p.discard.append(p.hand.pop(hand_index))
            state.log(f"{state.player_name(state.current_player)}: 夜のタンカを使用 → トラッシュから {_card_label(chosen)} を手札に加えた")
            return True
        return False
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

    # なかよしポフィン: 山札から HP70 以下のたねポケモンを 2 枚までベンチに出す
    _is_nakayoshipofuin = cid == NAKAYOSHI_POFIN or name_ja == "なかよしポフィン"
    if _is_nakayoshipofuin:
        from .state import BENCH_SIZE, BattlePokemon
        if len(p.bench) >= BENCH_SIZE:
            return False
        basics_in_deck = [
            (i, c) for i, c in enumerate(p.deck)
            if is_pokemon(c)
            and not getattr(c, "evolves_from", None)
            and (getattr(c, "max_hp", 0) or getattr(c, "hp", 0)) <= 70
        ]
        if not basics_in_deck:
            return False
        # 名前が重複しないよ���に、かつ場にい���いものを優先
        existing_names = set()
        if p.active:
            existing_names.add(getattr(p.active.card, "name", ""))
        for bp in p.bench:
            existing_names.add(getattr(bp.card, "name", ""))
        # ドラパルトexデッキ: ドラメシヤ・ヨマワルの重複を許可
        _is_drapa_pofin = is_dragapult_deck_for_player(state, state.current_player)
        _drapa_dup_names = {"ドラメシヤ", "ヨマワル"} if _is_drapa_pofin else set()
        # 場にいないたねを優先（ドラパルトexデッキでは指定名の重複を許可）
        preferred = [
            (i, c) for i, c in basics_in_deck
            if getattr(c, "name", "") not in existing_names
            or getattr(c, "name", "") in _drapa_dup_names
        ]
        if not preferred:
            preferred = basics_in_deck
        # ドラパルトexデッキ: ドラメシヤ→ヨマワルの順に優先（ドロンチ複数がドローエンジン）
        if _is_drapa_pofin:
            _drapafield_names = {getattr(bp.card, "name", "") for bp in p.bench}
            if p.active:
                _drapafield_names.add(getattr(p.active.card, "name", ""))
            _dorameshiya_count = sum(1 for n in _drapafield_names if n == "ドラメシヤ") + sum(
                1 for bp in p.bench if getattr(bp.card, "name", "") == "ドラメシヤ"
            )
            _yoma_line_names_pofin = {"ヨマワル", "サマヨール", "ヨノワール"}
            _yoma_line_on_field = any(
                getattr(bp.card, "name", "") in _yoma_line_names_pofin for bp in p.bench
            ) or (p.active and getattr(p.active.card, "name", "") in _yoma_line_names_pofin)
            def _drapa_pofin_priority(pair):
                _, c = pair
                cn = (getattr(c, "name", "") or "").strip()
                if cn == "マシマシラ":
                    return 99  # マシマシラはベンチに出さない
                if cn == "ドラメシヤ":
                    return 0  # 最優先
                if cn == "ヨマワル" and not _yoma_line_on_field:
                    return 1  # ヨマワルラインが場に0体なら1体だけ
                if cn == "ヨマワル":
                    return 99  # 既に1体いる → 出さない
                return 10
            preferred.sort(key=_drapa_pofin_priority)
            # マシマシラとヨマワル重複を候補から除外
            preferred = [(i, c) for i, c in preferred if _drapa_pofin_priority((i, c)) < 99]
        put_count = 0
        used_indices = []
        for i, c in preferred:
            if len(p.bench) >= BENCH_SIZE or put_count >= 2:
                break
            bp = BattlePokemon(card=c.copy())
            bp.put_on_bench_this_turn = True
            p.bench.append(bp)
            used_indices.append(i)
            put_count += 1
            state.log(
                f"{state.player_name(state.current_player)}: なかよしポフィン → ベンチに {c.name} を出す（ベンチ {len(p.bench)} 体）"
            )
        # 山札からカードを除去（逆順で pop）
        for i in sorted(used_indices, reverse=True):
            p.deck.pop(i)
        if put_count > 0:
            random.shuffle(p.deck)
            mark_own_deck_shuffled(state)
            mark_deck_searched(state)
            p.discard.append(p.hand.pop(hand_index))
            return True
        return False

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
    if getattr(state, "goods_locked_next_turn", None) == state.current_player:
        return False
    p = state.active_player_state()
    if not p.active or bench_index < 0 or bench_index >= len(p.bench):
        return False
    if hand_index < 0 or hand_index >= len(p.hand):
        return False
    card = p.hand[hand_index]
    if not is_goods(card):
        return False
    if getattr(card, "effect", None) != "swap_active" and getattr(card, "id", "") not in ("pokemon_irekae", POKEMON_IREKAE):
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


def _field_has_pokemon(p: PlayerState, name_ja: str, *id_prefixes: str) -> bool:
    """自分のバトル場またはベンチに、名前が name_ja または id が id_prefixes のいずれかで始まるポケモンがいるか。"""
    def match(card) -> bool:
        if not card:
            return False
        n = (getattr(card, "name", "") or getattr(card, "name_ja", "") or "").strip()
        cid = (getattr(card, "id", "") or "").strip()
        if n == name_ja:
            return True
        for prefix in id_prefixes:
            if cid == prefix or cid.startswith(prefix + "-") or cid.startswith(prefix + "_"):
                return True
        return False
    if p.active and match(p.active.card):
        return True
    for bp in p.bench:
        if bp and match(bp.card):
            return True
    return False


def _try_use_ability_runasaikuru(state: GameState) -> bool:
    """
    ルナトーンの特性「ルナサイクル」を宣言して使う。
    条件: 自分の場にルナトーンとソルロックがいる、手札に基本闘エネルギーが 1 枚以上、この番まだルナサイクル未使用。
    効果: 手札から基本闘エネルギー 1 枚をトラッシュし、山札を 3 枚引く。
    宣言して使う特性は先行 1 ターン目でも使える（サポートと異なる）。
    """
    if state.ability_declared_this_turn == "ルナサイクル":
        return False
    p = state.active_player_state()
    # デッキ切れ防止: 山札が少ないときはルナサイクルを控える。
    # ルナサイクルは毎ターン 3 枚引く+エネ 1 枚トラッシュ = 実質 net -2 deck/ターン。
    # 残りサイド枚数に応じて必要なターン数を見積もり、デッキが足りなければスキップ。
    deck_size = len(p.deck)
    remaining_prizes = len(p.prize_pile)
    # 残りサイドから必要ターン推定: 1KO/2ターンとして残り prizes * 2 ターン分のドロー余裕
    # 毎ターン 1 ドロー + ルナサイクル 3 ドロー = 4 枚/ターン消費。安全マージンを加味。
    min_deck_for_luna = max(10, remaining_prizes * 4 + 3)
    if deck_size <= min_deck_for_luna:
        from card import is_support, is_goods
        has_playable = any(is_support(c) or is_goods(c) for c in p.hand)
        if has_playable:
            return False
        # 手札にエネルギーもアタッカーもあるなら、ドロー不要
        has_energy = any(is_energy(c) for c in p.hand)
        has_attacker_or_evo = any(
            is_pokemon(c) and (getattr(c, "name", "") or "").strip() in
            ("メガルカリオex", "ハリテヤマ", "ルカリオ")
            for c in p.hand
        )
        if has_energy and has_attacker_or_evo:
            return False
    if not _field_has_pokemon(p, "ルナトーン", "runaton"):
        return False
    if not _field_has_pokemon(p, "ソルロック", "sorurokku-mc-372", "sorurokku"):
        return False
    energy_idx = None
    for i, c in enumerate(p.hand):
        if not is_energy(c):
            continue
        eid = getattr(c, "id", "") or ""
        if eid != "basic-energy-fighting":
            continue
        energy_idx = i
        break
    if energy_idx is None:
        return False
    trashed = p.hand.pop(energy_idx)
    p.discard.append(trashed)
    drawn = p.draw(3)
    p.hand.extend(drawn)
    state.drawn_this_turn.extend(drawn)
    state.ability_declared_this_turn = "ルナサイクル"
    drawn_names = ", ".join(_card_label(c) for c in drawn)
    state.log(
        f"{state.player_name(state.current_player)}: ルナトーンの特性「ルナサイクル」を使用 → "
        f"手札の基本闘エネルギー 1 枚をトラッシュし、山札から 3 枚ドロー → [{drawn_names}]"
    )
    return True


def _try_use_ability_sakatenitori(state: GameState) -> bool:
    """
    キチキギスex の特性「さかてにとる」を使う。
    条件: 前の相手の番に自分のポケモンがきぜつ、ベンチにキチキギスex がいる、この番まだ「さかてにとる」未使用。
    効果: 山札を 3 枚引く。
    """
    if getattr(state, "_sakatenitori_used_this_turn", False):
        return False
    # 前の相手の番に自分のポケモンがきぜつしていたか（汎用 KO 追跡）
    any_ko = getattr(state, "any_ko_by_opponent_last_turn", [False, False])
    if not any_ko[state.current_player] and not state.our_ko_by_damage_last_turn[state.current_player]:
        return False
    p = state.active_player_state()
    # ジャミングタワーチェック: どうぐ付きなら特性なし（ただしキチキギスexにどうぐを付けるケースは稀）
    if _jamming_tower_active(state):
        for bp in p.bench:
            if getattr(bp, "attached_tool", None):
                card_name = (getattr(bp.card, "name", "") or "").strip()
                if card_name == "キチキギスex":
                    return False  # どうぐ付きで特性無効
    # ベンチにキチキギスex がいるか
    kichikigisu_bp = None
    for bp in p.bench:
        card_name = (getattr(bp.card, "name", "") or "").strip()
        if card_name == "キチキギスex":
            kichikigisu_bp = bp
            break
    if kichikigisu_bp is None:
        return False
    drawn = p.draw(3)
    if not drawn:
        return False
    p.hand.extend(drawn)
    state.drawn_this_turn.extend(drawn)
    state._sakatenitori_used_this_turn = True
    drawn_names = ", ".join(_card_label(c) for c in drawn)
    state.log(
        f"{state.player_name(state.current_player)}: キチキギスex の特性「さかてにとる」→ "
        f"山札から {len(drawn)} 枚ドロー → [{drawn_names}]"
    )
    return True


def _try_use_ability_adrenabrain(state: GameState) -> bool:
    """
    マシマシラの特性「アドレナブレイン」を使う。
    条件: 自分の場にマシマシラがいる（悪エネルギー付き）、自分のポケモンにダメカンがのっている、この番まだ未使用。
    効果: 自分のポケモン 1 匹のダメカンを 3 個まで（30 ダメージ）相手のポケモン 1 匹にのせ替える。
    """
    if getattr(state, "_adrenabrain_used_this_turn", False):
        return False
    p = state.active_player_state()
    opp = state.defending_player_state()
    # 場にマシマシラがいるか
    mashimashira_found = False
    for bp in ([p.active] if p.active else []) + list(p.bench):
        if (getattr(bp.card, "name", "") or "").strip() == "マシマシラ":
            # 悪エネルギーが付いている必要がある（ability_description 参照）
            has_dark = any(t == "darkness" for t in getattr(bp, "attached_energy_types", []))
            if has_dark:
                mashimashira_found = True
                break
    if not mashimashira_found:
        return False
    # ジャミングタワーチェック
    if _jamming_tower_active(state):
        for bp in ([p.active] if p.active else []) + list(p.bench):
            if (getattr(bp.card, "name", "") or "").strip() == "マシマシラ" and getattr(bp, "attached_tool", None):
                return False
    # 自分のポケモンでダメージを受けているものを探す
    source_bp = None
    max_damage_taken = 0
    for bp in ([p.active] if p.active else []) + list(p.bench):
        damage_taken = bp.max_hp - bp.hp if bp.hp < bp.max_hp else 0
        if damage_taken > max_damage_taken:
            max_damage_taken = damage_taken
            source_bp = bp
    if source_bp is None or max_damage_taken < 10:
        return False
    # 相手のポケモンで最もダメージを与える効果が高い対象を選ぶ（KO優先）
    # 最大 3 個（30 ダメージ）まで移動可能
    move_amount = min(30, max_damage_taken)
    # ダメカンは 10 刻み
    move_amount = (move_amount // 10) * 10
    targets = []
    if opp.active and opp.active.hp and opp.active.hp > 0:
        targets.append(opp.active)
    for bp in opp.bench:
        if bp and bp.hp and bp.hp > 0:
            targets.append(bp)
    if not targets:
        return False
    # KO できるターゲットを優先
    best_target = None
    best_score = -1
    for t in targets:
        score = 0
        if t.hp <= move_amount:
            score = 10000 + _prizes_for_ko(t) * 1000
        else:
            score = move_amount  # HP を削る効果
        if score > best_score:
            best_score = score
            best_target = t
    if best_target is None:
        return False
    source_bp.hp += move_amount  # ダメカンを外す（回復）
    source_bp.hp = min(source_bp.hp, source_bp.max_hp)
    before_target = best_target.hp
    best_target.hp -= move_amount
    state._adrenabrain_used_this_turn = True
    target_loc = "バトル場" if best_target is opp.active else "ベンチ"
    state.log(
        f"{state.player_name(state.current_player)}: マシマシラの特性「アドレナブレイン」→ "
        f"{source_bp.card.name} のダメカン {move_amount // 10} 個を相手の{target_loc}の {best_target.card.name} にのせ替え"
        f"（HP {before_target} → {max(0, best_target.hp)}）"
    )
    # KO 処理
    if best_target.hp <= 0:
        if best_target is opp.active:
            state.log(f"相手のバトル場の {best_target.card.name} がきぜつ！")
            from .state import _handle_opponent_ko
            _handle_opponent_ko(opp, state, best_target)
        else:
            for i in range(len(opp.bench) - 1, -1, -1):
                if i < len(opp.bench) and opp.bench[i] is best_target:
                    opp.bench.pop(i)
                    state.log(f"相手のベンチの {best_target.card.name} がきぜつ！")
                    from .state import _handle_opponent_ko as _hok
                    _hok(opp, state, best_target)
                    break
    return True


def _try_use_ability_cursed_bomb(state: GameState) -> bool:
    """
    サマヨール/ヨノワールの特性「カースドボム」を使う。
    効果: このポケモンをきぜつさせ、相手のポケモン1匹にダメカンを5個(サマヨール)or13個(ヨノワール)のせる。
    条件: サイドトレードが有利な場合のみ使う（自分は非ex=サイド1枚渡し、相手をKOできればサイド1-3枚取得）。
    """
    if getattr(state, "ability_used_this_turn", False):
        return False
    p = state.active_player_state()
    opp = state.defending_player_state()
    if not opp:
        return False

    # カースドボム持ちのポケモンを探す（ベンチのみ。バトル場はきぜつするとまずい）
    bomb_candidates = []
    for i, bp in enumerate(p.bench):
        ability = (getattr(bp.card, "ability_name", None) or "").strip()
        if ability != "カースドボム":
            continue
        bp_name = (getattr(bp.card, "name", "") or "").strip()
        # サマヨール: ダメカン5個 = 50ダメージ, ヨノワール: ダメカン13個 = 130ダメ���ジ
        if bp_name == "サマヨール":
            bomb_damage = 50
        elif bp_name == "ヨノワール":
            bomb_damage = 130
        else:
            continue
        bomb_candidates.append((i, bp, bomb_damage))

    if not bomb_candidates:
        return False

    # 相手のKO可能なターゲットを探す
    our_prizes_remaining = len(p.prize_pile)
    opp_prizes_remaining = len(opp.prize_pile)
    _is_drapa = is_dragapult_deck_for_player(state, state.current_player)
    _phantom_bench_dmg = 60  # ファントムダイブのベンチダメカン

    for bomb_i, bomb_bp, bomb_damage in bomb_candidates:
        # ターゲット候補: active + bench
        targets = []
        if opp.active and opp.active.hp and opp.active.hp > 0:
            targets.append(("active", opp.active))
        for ti, tbp in enumerate(opp.bench):
            if tbp and tbp.hp and tbp.hp > 0:
                targets.append((f"bench:{ti}", tbp))

        # --- 1. 勝ち確判定: カースドボム単体 or カースドボム+攻撃でサイド取り切り ---
        can_win_with_bomb = False
        win_target = None
        win_target_key = None
        win_prize_gain = 0

        # 攻撃でactiveを倒せるか事前計算
        _can_ko_active = False
        _atk_prize = 0
        if opp.active and opp.active.hp and opp.active.hp > 0:
            from .damage import _max_effective_damage_for_attacker
            _our_atk_dmg = _max_effective_damage_for_attacker(state, p.active, opp.active, state.current_player)
            _can_ko_active = _our_atk_dmg >= opp.active.hp
            if _can_ko_active:
                from .state import _prizes_for_ko as _pfk2
                _atk_prize = _pfk2(opp.active)

        for tkey, tbp in targets:
            if tbp.hp > bomb_damage:
                continue
            from .state import _prizes_for_ko
            pg = _prizes_for_ko(tbp)
            # カースドボム単体でサイド取り切り
            if our_prizes_remaining <= pg:
                if pg > win_prize_gain:
                    win_target, win_target_key, win_prize_gain = tbp, tkey, pg
                    can_win_with_bomb = True
            # カースドボム(ベンチ) + 攻撃(active)でサイド取り切り
            elif tkey != "active" and _can_ko_active and our_prizes_remaining <= pg + _atk_prize:
                if pg > win_prize_gain:
                    win_target, win_target_key, win_prize_gain = tbp, tkey, pg
                    can_win_with_bomb = True
            # 種切れ勝ち: 全ポケモンをカースドボムで倒せる
            elif tkey == "active" and not opp.bench:
                win_target, win_target_key, win_prize_gain = tbp, tkey, pg
                can_win_with_bomb = True

        # カースドボムで削り + ファントムダイブのベンチ60でKO → サイド取り切り
        # 例: サマヨール50 → HP110のベンチが60に → ファントムダイブ(200+60)で2体KO
        if not can_win_with_bomb and _is_drapa and _can_ko_active:
            _drapa_active = p.active
            _drapa_name = (getattr(_drapa_active.card, "name", "") or "").strip() if _drapa_active else ""
            if _drapa_name == "ドラパルトex":
                _en = getattr(_drapa_active, "attached_energy", 0) or 0
                _types = list(getattr(_drapa_active, "attached_energy_types", []) or [])
                _can_phantom_win = _en >= 2 and "fire" in _types and "psychic" in _types
                if _can_phantom_win:
                    for tkey, tbp in targets:
                        if tkey == "active":
                            continue  # バトル場はファントムダイブ本体で倒す
                        if tbp.hp is None or tbp.hp <= 0:
                            continue
                        # カースドボムで削って、ファントムダイブのベンチ60で倒せるか
                        remaining_hp = tbp.hp - bomb_damage
                        if remaining_hp > 0 and remaining_hp <= _phantom_bench_dmg:
                            from .state import _prizes_for_ko
                            pg_bomb_bench = _prizes_for_ko(tbp)
                            total_prizes = _atk_prize + pg_bomb_bench
                            if our_prizes_remaining <= total_prizes:
                                win_target, win_target_key = tbp, tkey
                                win_prize_gain = pg_bomb_bench
                                can_win_with_bomb = True
                                break

        # --- ブライア解放: 相手サイド3 + ブライア手札 + FD準備完了 → ボムで相手サイドを2にしてブライア有効化 ---
        if not can_win_with_bomb and opp_prizes_remaining == 3 and _is_drapa:
            _has_briar = any((getattr(c, "id", "") or "") == BURAIA for c in p.hand)
            if _has_briar:
                _can_fd_briar = False
                # ドラパルトexがバトル場 or バトル場に出せる状態か
                _drapa_can_attack = False
                _all_our_b = ([p.active] if p.active else []) + list(p.bench or [])
                for _bp_b in _all_our_b:
                    if (getattr(_bp_b.card, "name", "") or "").strip() != "ドラパルトex":
                        continue
                    _is_active = (p.active is _bp_b)
                    # ベンチにいる場合、バトル場に出せるか（逃げコスト0 or いれかえ手札）
                    if not _is_active:
                        _active_rc = getattr(p.active.card, "retreat_cost", 1) if p.active else 99
                        _tool_a = getattr(p.active, "attached_tool", None) if p.active else None
                        _eff_rc = max(0, _active_rc - (2 if _tool_a and (getattr(_tool_a, "id", "") or "") == FUUSEN else 0))
                        _can_retreat = (
                            not getattr(state, "retreat_used_this_turn", False)
                            and getattr(p.active, "special_state", None) not in ("sleep", "paralysis")
                            and (_eff_rc == 0 or (getattr(p.active, "attached_energy", 0) or 0) >= _eff_rc)
                        )
                        _has_switch = any(
                            (getattr(c, "id", "") or "") == "pokemonirekaee" for c in p.hand
                        )
                        if not _can_retreat and not _has_switch:
                            continue
                    # FD準備完了チェック
                    _en_b = getattr(_bp_b, "attached_energy", 0) or 0
                    _types_b = list(getattr(_bp_b, "attached_energy_types", []) or [])
                    if _en_b >= 2 and "fire" in _types_b and "psychic" in _types_b:
                        _can_fd_briar = True
                        break
                    # このターンのエネ付与でFD完成見込み
                    if not getattr(state, "energy_attached_this_turn", False) and _en_b >= 1:
                        _need_type = None
                        if "fire" in _types_b and "psychic" not in _types_b:
                            _need_type = "psychic"
                        elif "psychic" in _types_b and "fire" not in _types_b:
                            _need_type = "fire"
                        if _need_type:
                            from card import is_energy
                            for _hc in p.hand:
                                if is_energy(_hc) and getattr(_hc, "energy_type", None) == _need_type:
                                    _can_fd_briar = True
                                    break
                    if _can_fd_briar:
                        break
                if _can_fd_briar:
                    _briar_best = None
                    _briar_key = None
                    _briar_score = -1
                    for tkey, tbp in targets:
                        if tbp.hp is None or tbp.hp <= 0:
                            continue
                        from .state import _prizes_for_ko
                        _pg = _prizes_for_ko(tbp)
                        _sc = 0
                        if tbp.hp <= bomb_damage:
                            _sc = _pg * 10000  # ボムだけでKO
                        elif tkey == "active" and tbp.hp - bomb_damage <= 200:
                            _sc = _pg * 8000  # ボム+FD200でKO
                        elif tkey != "active" and tbp.hp - bomb_damage <= _phantom_bench_dmg:
                            _sc = _pg * 5000  # ボム+FDベンチ60でKO
                        else:
                            _sc = _pg * 1000 + bomb_damage  # ダメージだけ
                        if _sc > _briar_score:
                            _briar_best = tbp
                            _briar_key = tkey
                            _briar_score = _sc
                    if _briar_best is not None:
                        win_target = _briar_best
                        win_target_key = _briar_key
                        win_prize_gain = _briar_score
                        can_win_with_bomb = True

        # 相手サイド1枚の時は自爆禁止（相手がサイド取って勝ってしまう）
        if opp_prizes_remaining <= 1 and not can_win_with_bomb:
            continue
        # 相手サイド2枚でブライア手札の時も自爆禁止（ブライア条件が崩れる）
        if opp_prizes_remaining == 2 and not can_win_with_bomb:
            if any((getattr(c, "id", "") or "") == BURAIA for c in p.hand):
                continue

        if can_win_with_bomb and win_target is not None:
            best_target = win_target
            best_target_key = win_target_key
            best_prize_gain = win_prize_gain
        else:
            # サマヨール(50ダメ): このターンFDが打てて、ベンチダメカンと合わせてKOできるなら使う
            if bomb_damage <= 50:
                _bomb50_useful = False
                if _is_drapa:
                    # FDが打てるか（バトル場orベンチにFD準備完了のドラパルトex）
                    _can_fd = False
                    _all_our = ([p.active] if p.active else []) + list(p.bench or [])
                    for _bp_fd in _all_our:
                        if (getattr(_bp_fd.card, "name", "") or "").strip() == "ドラパルトex":
                            _bp_en = getattr(_bp_fd, "attached_energy", 0) or 0
                            _bp_types = list(getattr(_bp_fd, "attached_energy_types", []) or [])
                            if _bp_en >= 2 and "fire" in _bp_types and "psychic" in _bp_types:
                                _can_fd = True
                                break
                    if _can_fd:
                        for tkey, tbp in targets:
                            if tkey == "active":
                                continue
                            if tbp.hp and 0 < tbp.hp <= bomb_damage + _phantom_bench_dmg:
                                _bomb50_useful = True
                                break
                if not _bomb50_useful:
                    continue
            # --- 2. 勝てない場合: 130ダメージを有効活用できるターゲットに使う ---
            best_target = None
            best_target_key = None
            best_prize_gain = 0

            for tkey, tbp in targets:
                hp = tbp.hp
                if hp is None or hp <= 0:
                    continue
                from .state import _prizes_for_ko
                _pg = _prizes_for_ko(tbp)

                # カースドボムだけでKOできる場合
                if hp <= bomb_damage:
                    _has_energy = (getattr(tbp, "attached_energy", 0) or 0) >= 1
                    _energy_count = getattr(tbp, "attached_energy", 0) or 0
                    # ファントムダイブのベンチ60で倒せる程度(HP<=60)でエネなしなら無視
                    if _is_drapa and hp <= _phantom_bench_dmg and not _has_energy:
                        continue
                    # exポケモン(サイド2枚以上)なら有利交換
                    if _pg >= 2:
                        _score = _pg * 10000 + _energy_count * 1000
                        if _score > best_prize_gain:
                            best_target, best_target_key, best_prize_gain = tbp, tkey, _score
                    # 非exでもエネ3以上付いた脅威ポケモンは倒す価値あり（相手の主力攻撃手段を潰す）
                    elif _energy_count >= 3:
                        _score = _energy_count * 1000
                        if _score > best_prize_gain:
                            best_target, best_target_key, best_prize_gain = tbp, tkey, _score
                # カースドボム + ファントムダイブベンチ60 でKOできる
                elif _is_drapa and hp <= bomb_damage + _phantom_bench_dmg and tkey != "active":
                    _energy_count_combo = getattr(tbp, "attached_energy", 0) or 0
                    _target_name_combo = (getattr(tbp.card, "name", "") or "").strip()
                    _score = _pg * 5000 + _energy_count_combo * 500 + 1000
                    # マクノシタ/ハリテヤマ進化前はどすこいキャッチャー脅威 → 優先的に排除
                    if _target_name_combo == "マクノシタ":
                        _score += 3000
                    if _score > best_prize_gain:
                        best_target, best_target_key, best_prize_gain = tbp, tkey, _score
                # バトル場: カースドボムで削って攻撃でKOできるか
                elif tkey == "active" and _is_drapa:
                    remaining = hp - bomb_damage
                    _energy_count_active = getattr(tbp, "attached_energy", 0) or 0
                    if remaining > 0 and remaining <= 200 and (_pg >= 2 or _energy_count_active >= 2):
                        _score = _pg * 5000 + _energy_count_active * 500
                        if _score > best_prize_gain:
                            best_target, best_target_key, best_prize_gain = tbp, tkey, _score

            if best_target is None:
                continue

        # 実行
        before_hp = best_target.hp
        best_target.hp -= bomb_damage
        target_loc = "バトル場の" if best_target_key == "active" else "ベンチの"
        state.log(
            f"{state.player_name(state.current_player)}: {bomb_bp.card.name} の特性「カースドボム」→ "
            f"相手の{target_loc}{best_target.card.name} にダメカン {bomb_damage // 10} 個（{bomb_damage} ダメージ、HP {before_hp} → {max(0, best_target.hp)}）"
        )

        # カースドボム使用者をきぜつさせる
        bomb_bp_card = bomb_bp.card
        p.bench.pop(bomb_i)
        from .state import _put_energy_cards_in_discard
        _put_energy_cards_in_discard(p, getattr(bomb_bp, "attached_energy_types", []), state)
        p.discard.append(bomb_bp_card)
        state.log(
            f"{state.player_name(state.current_player)}: {bomb_bp_card.name} がカースドボムの効果できぜつ"
        )
        # 相手にサイド1枚
        opp_player_idx = state.opponent()
        from .state import _take_prize
        _take_prize(state, opp_player_idx)

        # ターゲットのKO処理
        if best_target.hp <= 0:
            if best_target_key == "active":
                state.log(f"相手のバトル場の {best_target.card.name} がきぜつ！")
                from .state import _handle_opponent_ko
                _handle_opponent_ko(opp, state, best_target)
            else:
                for ti2 in range(len(opp.bench) - 1, -1, -1):
                    if ti2 < len(opp.bench) and opp.bench[ti2] is best_target:
                        opp.bench.pop(ti2)
                        state.log(f"相手のベンチの {best_target.card.name} がきぜつ！")
                        from .state import _handle_opponent_ko as _hok2
                        _hok2(opp, state, best_target)
                        break

        state.ability_used_this_turn = True
        return True

    return False


def _try_use_ability_okunote_catch(state: GameState) -> bool:
    """
    ニャースex の特性「���くのてキャッチ」を使う（ベンチに出したとき）。
    この関数はベンチに出した直後に呼ぶ。
    効果: 山札からサポートを1枚選び、手札に加える。そして山札を切る。
    """
    if getattr(state, "_okunote_used_this_turn", False):
        return False
    p = state.active_player_state()
    if len(p.deck) < 1:
        return False
    # 山札からサポートを探す（最も有用なサポートを選ぶ）
    support_candidates = [(i, c) for i, c in enumerate(p.deck) if is_support(c)]
    if not support_candidates:
        support_idx = None
    else:
        # サポートの優先度スコアリング
        _is_drapa_oku = is_dragapult_deck_for_player(state, state.current_player)
        # ドロー系サポートが手札にあるかだけをチェック（ブライア・ボスの指令はドローではない）
        _draw_support_ids = {RIRIE_NO_KESSHIN, ZEIYU, HAKASE_NO_KENKYU, HIKARI, "nemo", AKAMATSU}
        _has_support_in_hand = state.support_used_this_turn or any(
            is_support(hc) and (getattr(hc, "id", "") or "").strip() in _draw_support_ids
            for hc in p.hand
        )
        # ドラパルトexがエネ不足か（ファントムダイブに炎+超が必要）
        _drapa_needs_energy_oku = False
        if _is_drapa_oku:
            _all_bp_oku = ([p.active] if p.active else []) + list(p.bench or [])
            for _bp_oku in _all_bp_oku:
                if (getattr(_bp_oku.card, "name", "") or "").strip() in DRAPA_LINE_NAMES:
                    if (getattr(_bp_oku, "attached_energy", 0) or 0) < 2:
                        _drapa_needs_energy_oku = True
                        break
        _hand_size_oku = len(p.hand)
        # ドラパルトexラインが場にいるか（アカマツの価値判定用）
        _drapa_line_on_field_oku = _is_drapa_oku and any(
            (getattr(bp.card, "name", "") or "").strip() in DRAPA_LINE_NAMES
            for bp in ([p.active] if p.active else []) + list(p.bench or [])
        )
        # ドラパルトexが場にいるか（進化済み）
        _drapa_has_ex_on_field_oku = _is_drapa_oku and any(
            (getattr(bp.card, "name", "") or "").strip() == "ドラパルトex"
            for bp in ([p.active] if p.active else []) + list(p.bench or [])
        )
        def _okunote_support_score(pair):
            _, sc = pair
            sid = (getattr(sc, "id", "") or "").strip()
            # アカマツ: ドラパルトexが場にいてエネ不足なら最優先（FD直結）
            # リーリエでドローしてもエネが来る保証はないが、アカマツなら確実にエネ2枚
            if _is_drapa_oku and sid == AKAMATSU:
                if _drapa_has_ex_on_field_oku and _drapa_needs_energy_oku:
                    return 7000  # FD直結 → 最優先
                elif _drapa_line_on_field_oku and _drapa_needs_energy_oku:
                    return 4000
                elif _drapa_line_on_field_oku:
                    return 1500
                else:
                    return 500  # ラインが場にいない → 低優先
            # 手札が少ない場合はリーリエの決心が最優先（ドローで立て直す）
            if sid == RIRIE_NO_KESSHIN:
                if _hand_size_oku <= 4:
                    return 5000  # 手札少ない → リーリエ最優先
                elif not _has_support_in_hand:
                    return 1800
                else:
                    return 800
            # ボスの指令: ベンチ狙いに必要
            if sid == BOSS_NO_SHIREI:
                return 1500
            # メイのはげまし: きぜつ後のエネ加速（Stage2対象）
            # ドラパルトexがエネ不足でファントムダイブに繋がるなら最優先
            if _is_drapa_oku and sid == MEI_NO_HAGEMASHI:
                if _drapa_needs_energy_oku:
                    # ドラパルトexが場にいてエネ不足 → メイのはげましが最優先
                    _has_drapa_ex_oku = any(
                        (getattr(bp.card, "name", "") or "").strip() == "ドラパルトex"
                        for bp in ([p.active] if p.active else []) + list(p.bench or [])
                    )
                    if _has_drapa_ex_oku:
                        return 6000  # ファントムダイブ直結 → リーリエより上
                    return 4000
                return 1200
            # ブライア: フィニッシュ用
            if sid == BURAIA:
                return 800
            # それ以外のドロー系
            if sid in ("nemo", "nemokako", "nemomirai"):
                return 1000
            return 500
        best_pair = max(support_candidates, key=_okunote_support_score)
        support_idx = best_pair[0]
    if support_idx is None:
        # サポートが山札にない場合でも特性は使ったことにする（空振り）
        state._okunote_used_this_turn = True
        random.shuffle(p.deck)
        mark_own_deck_shuffled(state)
        mark_deck_searched(state)
        state.log(
            f"{state.player_name(state.current_player)}: ニャースex の特性「おくのてキャッチ」→ "
            f"山札にサポートがなかった"
        )
        return True
    found = p.deck.pop(support_idx)
    p.hand.append(found)
    state.drawn_this_turn.append(found)
    random.shuffle(p.deck)
    mark_own_deck_shuffled(state)
    mark_deck_searched(state)
    state._okunote_used_this_turn = True
    state.log(
        f"{state.player_name(state.current_player)}: ニャースex の特性「おくのてキャッチ」→ "
        f"山札からサポート {_card_label(found)} を手札に加えた"
    )
    return True


def _try_use_ability_teisatsushirei(state: GameState) -> bool:
    """
    ドロンチの特性「ていさつしれい」を使う。
    各ドロンチが自分の番に1回ずつ使える（複数ドロンチがいれば複数回使用可）。
    効果: 山札の上から2枚見て、1枚を手札に加え、残りを山札の下に戻す。
    """
    p = state.active_player_state()
    if len(p.deck) < 1:
        return False
    # デッキ残量が少ない＋手札が多い場合はデッキ切れが怖いので使わない
    if len(p.deck) <= 5 and len(p.hand) >= 5:
        return False
    # 使用済みドロンチの id を管理（複数体対応）
    used_set = getattr(state, "_teisatsushirei_used_ids_this_turn", None)
    if used_set is None:
        used_set = set()
        state._teisatsushirei_used_ids_this_turn = used_set
    # 場にまだ未使用のドロンチがいるか
    doronchi_bp = None
    for bp in ([p.active] if p.active else []) + list(p.bench):
        if (getattr(bp.card, "name", "") or "").strip() == "ドロンチ":
            # ジャミングタワーチェック
            if _jamming_tower_active(state) and getattr(bp, "attached_tool", None):
                continue
            if id(bp) not in used_set:
                doronchi_bp = bp
                break
    if doronchi_bp is None:
        return False
    look = min(2, len(p.deck))
    top = [p.deck.pop(0) for _ in range(look)]
    if not top:
        return False
    # ベストカードを選ぶ: 状況に応じた動的スコアリング
    # ドラパルトexデッキ: 進化→ファントムダイブに繋がるカードを最優先
    _all_field_bp = ([p.active] if p.active else []) + list(p.bench or [])
    _has_doronchi_field = any(
        (getattr(bp.card, "name", "") or "").strip() == "ドロンチ" for bp in _all_field_bp
    )
    _has_drameshiya_field = any(
        (getattr(bp.card, "name", "") or "").strip() == "ドラメシヤ" for bp in _all_field_bp
    )
    _has_drapa_in_hand = any(
        is_pokemon(hc) and (getattr(hc, "name", "") or "").strip() == "ドラパルトex"
        for hc in p.hand
    )
    # ドラパルトexが場にいてエネ不足か（アカマツの価値判定用）
    _drapa_ex_needs_energy_tei = any(
        (getattr(bp.card, "name", "") or "").strip() == "ドラパルトex"
        and (getattr(bp, "attached_energy", 0) or 0) < 2
        for bp in _all_field_bp
    )
    # キチキギスex: 前ターンにきぜつ発生+ベンチ空き → さかてにとるで3枚ドロー
    _bench_has_room_tei = len(p.bench) < 5  # ベンチ枠に空きがあるか（activeは別）
    _ko_last_turn_tei = any(getattr(state, "any_ko_by_opponent_last_turn", [False, False]))
    _kichikigisu_valuable_tei = _ko_last_turn_tei and _bench_has_room_tei
    # 手札刷新サポート（リーリエの決心等）が手札にあり、このターン使う見込みがあるか
    # → ていさつしれいで取ったカードも山札に戻されるので、即使えるカード優先
    # ただしアカマツはFD完成に直結するので、ドラパルトexが場にいてエネ不足なら
    # アカマツを使う方がリーリエより優先（アカマツを取るべき）
    _has_hand_refresh = (
        not state.support_used_this_turn
        and any(
            (getattr(hc, "id", "") or "").strip() in (RIRIE_NO_KESSHIN, HAKASE_NO_KENKYU)
            for hc in p.hand
        )
    )
    # アカマツの方がリーリエより優先されるケース:
    # ドラパルトexが場にいてエネ不足 → アカマツでFD完成
    _akamatsu_over_ririe = _drapa_ex_needs_energy_tei

    def _card_value(c):
        if is_support(c):
            cid_v = (getattr(c, "id", "") or "").strip()
            if cid_v == AKAMATSU:
                if _akamatsu_over_ririe:
                    # アカマツを使う方が優先 → 取る
                    return 1800
                if _has_hand_refresh:
                    # リーリエを使う予定でアカマツ不要 → 山札に戻されるので低価値
                    return 200
                return 1100
            # 手札刷新サポートを使う予定なら他のサポートも価値低
            if _has_hand_refresh:
                return 200
            return 1000
        if is_pokemon(c) and getattr(c, "evolves_from", None):
            cname_v = (getattr(c, "name", "") or "").strip()
            if cname_v == "ドラパルトex":
                # ドロンチが場にいて手札にドラパルトexがない → 即進化可能 → 最優先
                if _has_doronchi_field and not _has_drapa_in_hand:
                    return 1500
                return 900
            if cname_v == "ドロンチ":
                # ドラメシヤが場にいる → ドロンチ進化でていさつしれいが増える
                if _has_drameshiya_field:
                    return 1200
                return 900
            return 800
        # たねポケモン（進化なし）
        if is_pokemon(c) and not getattr(c, "evolves_from", None):
            cname_v = (getattr(c, "name", "") or "").strip()
            # キチキギスex: 前ターンきぜつ+ベンチ空き → さかてにとるで3枚ドロー
            if cname_v == "キチキギスex" and _kichikigisu_valuable_tei:
                return 1400  # サポートより高く、進化可能なドラパルトexより低い
            return 200  # たねポケモンはていさつしれいで取る優先度は低い
        if is_energy(c):
            return 500
        if is_goods(c):
            cid_v = (getattr(c, "id", "") or "").strip()
            if cid_v in (FUSHIGI_NA_AME, NAKAYOSHI_POFIN):
                return 600
            if cid_v == HYPER_BALL:
                return 550
            return 400
        return 100
    best = max(top, key=_card_value)
    top.remove(best)
    p.hand.append(best)
    state.drawn_this_turn.append(best)
    # 残りを山札の下に戻す
    for c in top:
        p.deck.append(c)
    used_set.add(id(doronchi_bp))
    # カウントベースの追跡（Usedマーカー用）
    if not hasattr(state, "_teisatsushirei_used_count"):
        state._teisatsushirei_used_count = [0, 0]
    state._teisatsushirei_used_count[state.current_player] += 1
    # 位置ベースの追跡（deepcopy後のスナップショットでも照合可能）
    if not hasattr(state, "_teisatsushirei_used_positions"):
        state._teisatsushirei_used_positions = set()
    p = state.active_player_state()
    if doronchi_bp is p.active:
        state._teisatsushirei_used_positions.add(("active", state.current_player))
    else:
        for bi, bbp in enumerate(p.bench):
            if bbp is doronchi_bp:
                state._teisatsushirei_used_positions.add(("bench", state.current_player, bi))
                break
    # 後方互換: 旧フラグもセット
    state._teisatsushirei_used_this_turn = True
    # ログに2枚とも表示
    if top:
        rest_labels = "、".join(_card_label(c) for c in top)
        state.log(
            f"{state.player_name(state.current_player)}: ドロンチの特性「ていさつしれい」→ "
            f"山札の上 {look} 枚 [{_card_label(best)}、{rest_labels}] を見て "
            f"{_card_label(best)} を手札に加え、{rest_labels} を山札の下にもどした"
        )
    else:
        # 山札1枚のみの場合
        state.log(
            f"{state.player_name(state.current_player)}: ドロンチの特性「ていさつしれい」→ "
            f"山札の上 {look} 枚を見て {_card_label(best)} を手札に加えた"
        )
    return True

def _jamming_tower_active(state: GameState) -> bool:
    """��ャミングタワーが場に出ているか。"""
    stadium = getattr(state, "stadium", None)
    if stadium is None:
        return False
    sid = (getattr(stadium, "id", "") or "").strip()
    sname = (getattr(stadium, "name", "") or "").strip()
    return sid == "jixyamingutawa" or "ジャミングタワー" in sname




def _use_support_akamatsu(state, hand_index: int) -> bool:
    """アカマツのサポート効果を実行する。"""
    p = state.active_player_state()
    # ドラパルトexデッキ: 炎＋超を優先的に取る
    _is_drapa = is_dragapult_deck_for_player(state, state.current_player)
    # ドラパルトデッキ: このターンにファントムダイブに届く場合のみ使う
    # （届かないなら付けたエネが次ターンにKOされて無駄になるリスク）
    if _is_drapa:
        _drapa_line_on_field = any(
            (getattr(bp.card, "name", "") or "").strip() in DRAPA_LINE_NAMES
            for bp in ([p.active] if p.active else []) + list(p.bench or [])
        )
        if not _drapa_line_on_field:
            return False
        # ドラパルトexが場にいればアカマツ使用OK（エネを貯め始める）
        # ドラメシヤ/ドロンチのみ（ドラパルトex未進化）の場合は
        # 付けたエネがKOで無駄になるリスクがあるためスキップ
        _has_drapa_ex = any(
            (getattr(bp.card, "name", "") or "").strip() == "ドラパルトex"
            for bp in ([p.active] if p.active else []) + list(p.bench or [])
        )
        if not _has_drapa_ex:
            return False
    _preferred_types = ["fire", "psychic"] if _is_drapa else []

    # ドラパルトexデッキ: ターゲットに既に付いているタイプは後回し（不足タイプを最優先）
    if _is_drapa and _preferred_types:
        _drapa_attach_targets = [bp for bp in ([p.active] if p.active else []) + list(p.bench)
                                 if (getattr(bp.card, "name", "") or "").strip() in DRAPA_LINE_NAMES]
        if _drapa_attach_targets:
            # HP低いバトル場ポケモンは避ける（KOされてエネ無駄になるリスク）
            opp_bt = state.defending_player_state()
            def _bt_score(bp):
                _is_ex = 1 if (getattr(bp.card, "name", "") or "").strip() == "ドラパルトex" else 0
                _en = getattr(bp, "attached_energy", 0) or 0
                _pen = 0
                if bp is p.active and opp_bt and opp_bt.active:
                    from .damage import _max_effective_damage_for_attacker
                    if _max_effective_damage_for_attacker(state, opp_bt.active, bp, 1 - state.current_player) >= (bp.hp or 0):
                        _pen = -10
                return (_pen, _is_ex, _en)
            _best_target = max(_drapa_attach_targets, key=_bt_score)
            _on_types = list(getattr(_best_target, "attached_energy_types", []) or [])
            # 不足タイプを先にする（例: 超が付いていれば炎を最優先）
            _needed = [t for t in _preferred_types if t not in _on_types]
            _have = [t for t in _preferred_types if t in _on_types]
            _preferred_types = _needed + _have

    energy_by_type: dict[str, int] = {}
    # まず優先タイプを探す
    for ptype in _preferred_types:
        if ptype in energy_by_type:
            continue
        for i, c in enumerate(p.deck):
            if is_energy(c) and getattr(c, "energy_type", None) == ptype and i not in energy_by_type.values():
                energy_by_type[ptype] = i
                break
    # 足りない分は任意のタイプ
    for i, c in enumerate(p.deck):
        if len(energy_by_type) >= 2:
            break
        if is_energy(c):
            etype = getattr(c, "energy_type", None)
            if etype and etype not in energy_by_type:
                energy_by_type[etype] = i
    if not energy_by_type:
        return False
    fetched = []
    indices = sorted(energy_by_type.values(), reverse=True)
    for idx in indices:
        c = p.deck.pop(idx)
        fetched.append(c)
    random.shuffle(p.deck)
    mark_own_deck_shuffled(state)
    mark_deck_searched(state)
    if len(fetched) == 1:
        # 1枚しか見つからなかった場合: 手札に加える
        p.hand.append(fetched[0])
        state.drawn_this_turn.append(fetched[0])
        p.discard.append(p.hand.pop(hand_index))
        state.support_used_this_turn = True
        state.log(
            f"{state.player_name(state.current_player)}: アカマツを使用 → 山札から基本エネルギー 1 枚を手札に加えた → [{_card_label(fetched[0])}]"
        )
    else:
        # 2枚見つかった場合: 1枚を手札に、1枚をポケモンにつける
        # 手札に加えるカードと付けるカードを決める
        attach_card = None
        hand_card = None
        all_pokemon = ([p.active] if p.active else []) + list(p.bench)

        # ドラパルトexデッキ: ドラパルトex/ドロンチ/ドラメシヤに必要なタイプを付ける
        if _is_drapa:
            # ドラパルトex系のポケモンを探す
            _drapa_targets = [bp for bp in all_pokemon
                              if (getattr(bp.card, "name", "") or "").strip() in DRAPA_LINE_NAMES]
            if _drapa_targets:
                # ターゲット選択: ドラパルトex優先、エネ多い方優先
                # ただし次ターンKOされそう（HPが低い）バトル場のポケモンは避ける
                opp_ak = state.defending_player_state()
                def _drapa_target_score(bp):
                    _is_ex = 1 if (getattr(bp.card, "name", "") or "").strip() == "ドラパルトex" else 0
                    _en = getattr(bp, "attached_energy", 0) or 0
                    _is_active = (bp is p.active)
                    # バトル場でHPが低い → 次ターンKOされてエネが無駄になるリスク
                    _hp_penalty = 0
                    if _is_active and opp_ak and opp_ak.active:
                        from .damage import _max_effective_damage_for_attacker
                        _opp_dmg = _max_effective_damage_for_attacker(state, opp_ak.active, bp, 1 - state.current_player)
                        if _opp_dmg >= (bp.hp or 0):
                            _hp_penalty = -10  # 次ターンKO確定 → 大幅減点
                    return (_hp_penalty, _is_ex, _en)
                _best_drapa = max(_drapa_targets, key=_drapa_target_score)
                _types_on = list(getattr(_best_drapa, "attached_energy_types", []) or [])
                # 付いてないタイプのエネを優先的に付ける
                for fc in fetched:
                    et = getattr(fc, "energy_type", None)
                    if et in ("fire", "psychic") and et not in _types_on:
                        attach_card = fc
                        break
                if attach_card is None:
                    # 両方付いている or 両方同じ → 炎/超のうちファントムダイブに有用な方を付ける
                    # ただし既に同じタイプが付いている場合は付けない（無色枠はどのタイプでもOK）
                    for fc in fetched:
                        et = getattr(fc, "energy_type", None)
                        if et in ("fire", "psychic") and et not in _types_on:
                            attach_card = fc
                            break
                    # それでも見つからない場合: 無色枠用に任意のエネを付ける
                    # ただし既に付いているタイプは避ける（重複は無意味）
                    if attach_card is None and len(_types_on) < 3:
                        for fc in fetched:
                            et = getattr(fc, "energy_type", None)
                            if et not in _types_on:
                                attach_card = fc
                                break
            # ドラパルトex系がいない場合も、炎/超を手札に残す（後で手張りで付けるため）
            # attach_card は None のままにして、両方手札に加える
        else:
            # 非ドラパルト: pokemon_type matching（既存ロジック）
            for fc in fetched:
                etype = getattr(fc, "energy_type", None)
                for bp in all_pokemon:
                    ptype = getattr(bp.card, "pokemon_type", None)
                    if ptype and ptype == etype:
                        attach_card = fc
                        break
                if attach_card:
                    break
        if attach_card is None:
            # ドラパルトexデッキで場にドラパルトex系がいない → 両方手札に加える
            if _is_drapa:
                for fc in fetched:
                    p.hand.append(fc)
                    state.drawn_this_turn.append(fc)
                p.discard.append(p.hand.pop(hand_index))
                state.support_used_this_turn = True
                state.log(
                    f"{state.player_name(state.current_player)}: アカマツを使用 → "
                    f"基本エネルギー 2 枚を手札に加えた → [{', '.join(_card_label(fc) for fc in fetched)}]"
                )
                return True
            attach_card = fetched[1]
        hand_card = fetched[0] if attach_card is fetched[1] else fetched[1]
        # 手札に加える
        p.hand.append(hand_card)
        state.drawn_this_turn.append(hand_card)
        # ポケモンにつける: タイプが合うポケモン or エネが少ないポケモン
        attach_target = None
        best_score = -1
        for bp in all_pokemon:
            score = 0
            bp_name = (getattr(bp.card, "name", "") or "").strip()
            ptype = getattr(bp.card, "pokemon_type", None)
            etype = getattr(attach_card, "energy_type", None)
            if ptype and ptype == etype:
                score += 100
            # エネルギーが少ないほど優先
            score -= getattr(bp, "attached_energy", 0) * 10
            # ドラパルトexデッキ: ドラパルトex/ドロンチ/ドラメシヤにエネ付け優先
            if _is_drapa:
                if bp_name == "ドラパルトex":
                    score += 500
                    # 必要な色がまだ付いてなければさらにボーナス
                    types_on = list(getattr(bp, "attached_energy_types", []) or [])
                    if etype == "fire" and "fire" not in types_on:
                        score += 200
                    elif etype == "psychic" and "psychic" not in types_on:
                        score += 200
                    # バトル場で次ターンKOされそう → エネが無駄になる、ベンチ優先
                    if bp is p.active:
                        _opp_ak2 = state.defending_player_state()
                        if _opp_ak2 and _opp_ak2.active:
                            from .damage import _max_effective_damage_for_attacker as _med_aka
                            _opp_dmg2 = _med_aka(state, _opp_ak2.active, bp, 1 - state.current_player)
                            if _opp_dmg2 >= (bp.hp or 0):
                                score -= 2000  # 次ターンKO確定 → ベンチに回す
                elif bp_name in ("ドロンチ", "ドラメシヤ"):
                    score += 300
                elif bp_name in ("スボミー", "キチキギスex", "ニャースex", "ヨマワル", "サマヨール", "ヨノワール", "マシマシラ"):
                    score -= 500  # サポート/カースドボム用にエネは不要
            if score > best_score:
                best_score = score
                attach_target = bp
        if attach_target is None and all_pokemon:
            attach_target = all_pokemon[0]
        if attach_target:
            attach_target.attached_energy += 1
            et = getattr(attach_card, "energy_type", None) or "colorless"
            if not hasattr(attach_target, "attached_energy_types"):
                attach_target.attached_energy_types = []
            attach_target.attached_energy_types.append(et)
            attach_loc = "バトル場の" if attach_target is p.active else "ベンチの"
            state.log(
                f"{state.player_name(state.current_player)}: アカマツを使用 → {_card_label(hand_card)} を手札に加え、"
                f"{_card_label(attach_card)} を{attach_loc}{attach_target.card.name} につけた"
            )
        else:
            # ポケモンがいない場合は両方手札に（フォールバック）
            p.hand.append(attach_card)
            state.drawn_this_turn.append(attach_card)
            state.log(
                f"{state.player_name(state.current_player)}: アカマツを使用 → ポケモンがいないため基本エネルギー 2 枚を手札に加えた"
            )
        p.discard.append(p.hand.pop(hand_index))
        state.support_used_this_turn = True
    return True

    # ブライア: 相手のサイドの残り枚数が2枚のときにしか使えない。
    # テラスタルポケモンのKOでサイド1枚多く取る。



def _use_support_mei(state, hand_index: int) -> bool:
    """メイのはげましのサポート効果を実行する。"""
    p = state.active_player_state()
    opp = state.defending_player_state()
    # 条件: 自分のサイドが相手より多い（=負けている）
    if len(p.prize_pile) <= len(opp.prize_pile):
        return False
    # 2進化ポケモンを探す
    all_pokemon = ([p.active] if p.active else []) + list(p.bench)
    stage2_pokemon = [bp for bp in all_pokemon if getattr(bp.card, "evolution_stage", "") == "stage2"]
    if not stage2_pokemon:
        return False
    # トラッシュから基本エネルギーを探す
    energy_in_discard = [c for c in p.discard if is_energy(c)]
    if not energy_in_discard:
        return False
    # 最もエネルギーが必要な2進化ポケモンを選ぶ
    _is_drapa_mei = is_dragapult_deck_for_player(state, state.current_player)
    # ドラパルトexデッキ: トラッシュにファントムダイブに必要なタイプのエネがなければ使わない
    # また、ターゲットのドラパルトexに既に付いているタイプしかトラッシュにないなら無駄
    if _is_drapa_mei:
        _useful_energy = [e for e in energy_in_discard if getattr(e, "energy_type", None) in ("fire", "psychic")]
        if not _useful_energy:
            return False
        # ドラパルトexが場にいて、そのドラパルトexに不足しているタイプがトラッシュにあるかチェック
        _drapa_targets = [bp for bp in stage2_pokemon if (getattr(bp.card, "name", "") or "").strip() == "ドラパルトex"]
        if _drapa_targets:
            _best_drapa = max(_drapa_targets, key=lambda bp: getattr(bp, "attached_energy", 0) or 0)
            _on_types = list(getattr(_best_drapa, "attached_energy_types", []) or [])
            _needed = [t for t in ("fire", "psychic") if t not in _on_types]
            if not _needed:
                return False  # 炎+超が既に揃っている → メイのはげまし不要
            _trash_has_needed = any(
                getattr(e, "energy_type", None) in _needed for e in _useful_energy
            )
            if not _trash_has_needed:
                return False  # 不足タイプがトラッシュにない → 同タイプ重複になるので使わない

    def _stage2_score(bp):
        max_cost = max((a.energy_cost for a in bp.card.attacks), default=0)
        score = max_cost - bp.attached_energy
        # ドラパルトexデッキ: ドラパルトexを最優先、ヨノワールにはエネ不要
        if _is_drapa_mei:
            bp_name = (getattr(bp.card, "name", "") or "").strip()
            if bp_name == "ドラパルトex":
                score += 100
            elif bp_name == "ヨノワール":
                score -= 200  # カースドボム用→エネ不要。かげしばりにエネを使うのは非効率
        return score
    target_bp = max(stage2_pokemon, key=_stage2_score)
    # ドラパルトexデッキ: ヨノワールしかStage2がいないなら使わない(エネの無駄)
    if _is_drapa_mei:
        target_name = (getattr(target_bp.card, "name", "") or "").strip()
        if target_name == "ヨノワール":
            return False
    # 最大2枚つける
    to_attach = min(2, len(energy_in_discard))
    attached = []

    # ドラパルトex向け: 技のenergy_cost_typedから必要タイプを計算
    _needed_types: list[str] = []
    if _is_drapa_mei and (getattr(target_bp.card, "name", "") or "").strip() == "ドラパルトex":
        types_on = list(getattr(target_bp, "attached_energy_types", []) or [])
        for atk in (target_bp.card.attacks or []):
            typed = getattr(atk, "energy_cost_typed", None)
            if typed and atk.energy_cost >= 2:
                for t in typed:
                    if t != "colorless" and types_on.count(t) < typed.count(t):
                        _needed_types.append(t)
                break

    for _ in range(to_attach):
        best_e = None
        # ドラパルトex: 必要なタイプのエネルギーを優先（火→超の順）
        if _needed_types:
            for need_t in _needed_types:
                for e in energy_in_discard:
                    if e in attached:
                        continue
                    etype = getattr(e, "energy_type", None)
                    if etype == need_t:
                        best_e = e
                        _needed_types.remove(need_t)
                        break
                if best_e:
                    break
        if best_e is None:
            # タイプが合うエネルギーを優先（既存ロジック）
            ptype = getattr(target_bp.card, "pokemon_type", None)
            for e in energy_in_discard:
                if e in attached:
                    continue
                etype = getattr(e, "energy_type", None)
                if etype == ptype:
                    best_e = e
                    break
        if best_e is None:
            # 悪エネルギーはドラパルトexには付けない（ファントムダイブに使えない）
            for e in energy_in_discard:
                if e not in attached:
                    etype = getattr(e, "energy_type", None)
                    if _is_drapa_mei and (getattr(target_bp.card, "name", "") or "").strip() == "ドラパルトex" and etype == "darkness":
                        continue
                    best_e = e
                    break
        if best_e is None:
            # フォールバック: 何でもよい（ただしドラパルトexに悪エネは絶対付けない）
            for e in energy_in_discard:
                if e not in attached:
                    etype = getattr(e, "energy_type", None)
                    if _is_drapa_mei and (getattr(target_bp.card, "name", "") or "").strip() == "ドラパルトex" and etype == "darkness":
                        continue
                    best_e = e
                    break
        if best_e is None:
            break
        attached.append(best_e)
        energy_in_discard = [e for e in energy_in_discard if e is not best_e]
    for e in attached:
        p.discard.remove(e)
        target_bp.attached_energy += 1
        et = getattr(e, "energy_type", None) or "colorless"
        if not hasattr(target_bp, "attached_energy_types"):
            target_bp.attached_energy_types = []
        target_bp.attached_energy_types.append(et)
    p.discard.append(p.hand.pop(hand_index))
    state.support_used_this_turn = True
    attached_names = ", ".join(_card_label(c) for c in attached)
    target_loc = "バトル場の" if target_bp is p.active else "ベンチの"
    state.log(
        f"{state.player_name(state.current_player)}: メイのはげましを使用 → トラッシュから基本エネルギー {len(attached)} 枚を{target_loc}{target_bp.card.name} につけた → [{attached_names}]"
    )
    return True




def _use_support_boss(state, hand_index: int) -> bool:
    """ボスの指令のサポート効果を実行する。"""
    p = state.active_player_state()
    card = p.hand[hand_index]
    opp = state.defending_player_state()
    if not opp.bench or not opp.active:
        return False

    # メガルカリオexデッキ戦略準拠:
    # 1. 相手バトル場を倒せるなら、ボスを温存（そのまま殴ればよい）
    # 2. ベンチの倒せるポケモンをサイド効率順で呼び出す
    #    - ex/メガ（サイド2〜3枚）を最優先
    #    - 低HPのベンチポケモンも対象（はどうづきで倒しつつエネ加速）
    #    - あと1枚で勝てるなら最も倒しやすいポケモンを呼ぶ
    # 3. エネ付き/進化済み/特性サポート持ちを狙えばリソース浪費にもなる
    if not p.active or not getattr(p.active, "card", None):
        return False
    if not getattr(p.active.card, "attacks", None):
        return False

    # 相手バトル場を倒せるなら、ボスを打つ必要はない
    # ただし「なぐってかくれる」等でダメージ無効状態の場合は倒せない
    can_ko_active = False
    opp_protected = getattr(opp.active, "protected_next_opponent_turn", False) if opp.active else False
    if opp.active and opp.active.hp is not None and opp.active.hp > 0 and not opp_protected:
        dmg = _max_effective_damage_for_attacker(state, p.active, opp.active, state.current_player)
        can_ko_active = dmg >= opp.active.hp
    if can_ko_active:
        return False

    # ドラパルトexデッキ: ボスの指令の使用条件
    if is_dragapult_deck_for_player(state, state.current_player):
        _active_name_boss = (getattr(p.active.card, "name", "") or "").strip()
        if _active_name_boss == "ドラパルトex":
            # ファントムダイブが撃てるか（エネ2以上 + 炎+超）
            _en_boss = getattr(p.active, "attached_energy", 0) or 0
            _types_boss = list(getattr(p.active, "attached_energy_types", []) or [])
            _can_phantom = _en_boss >= 2 and "fire" in _types_boss and "psychic" in _types_boss
            if _can_phantom:
                # ファントムダイブ200点はHP高い相手に当てる方が効率的
                # ボスで弱い相手を引っ張るとダメージが無駄になる
                # ボスを使うのは:
                #   1. サイド取り切りできる場合
                #   2. ベンチにエネ付き完成アタッカーがいて脅威を排除したい場合
                my_remaining = len(p.prize_pile)
                _boss_wins = False
                _boss_threat = False
                for i, bp in enumerate(opp.bench):
                    if not bp or bp.hp is None or bp.hp <= 0:
                        continue
                    prizes = _prizes_for_ko(bp)
                    # サイド取り切り
                    if 200 >= bp.hp and my_remaining <= prizes:
                        _boss_wins = True
                        break
                    # ex倒し→残り2以下
                    if 200 >= bp.hp and prizes >= 2 and my_remaining - prizes <= 2:
                        _boss_wins = True
                        break
                    # ベンチのエネ付きアタッカー（脅威排除目的）
                    _bp_energy = getattr(bp, "attached_energy", 0) or 0
                    if _bp_energy >= 2 and 200 >= bp.hp:
                        _boss_threat = True
                # バトル場のHP > ベンチターゲットのHP なら、バトル場に撃つ方が効率的
                _opp_active_hp = opp.active.hp if opp.active else 0
                if not _boss_wins and not _boss_threat:
                    return False  # バトル場にファントムダイブが最善
                if not _boss_wins and _boss_threat and _opp_active_hp >= 200:
                    return False  # バトル場のHP高い → バトル場に200点当てた方が効率的

    my_remaining_prizes = len(p.prize_pile)

    killable_bench: list[int] = []
    killable_bench_values: dict[int, float] = {}
    for i, bp in enumerate(opp.bench):
        if not bp or bp.hp is None or bp.hp <= 0:
            continue
        dmg = _max_effective_damage_for_attacker(state, p.active, bp, state.current_player)
        if dmg < bp.hp:
            continue
        killable_bench.append(i)
        prizes = _prizes_for_ko(bp)
        # サイド効率: ex/メガは高価値
        value = prizes * 10000.0
        # あと1〜2枚で勝てるなら、確実に取れるサイド数が足りるかで超高ボーナス
        if my_remaining_prizes <= prizes:
            value += 100000.0  # これで勝てる！
        # エネ付き/進化済みを狙うとリソース浪費にもなる
        energy_attached = getattr(bp, "attached_energy", 0) or 0
        is_evolved = bool(getattr(bp.card, "evolves_from", None))
        value += energy_attached * 200.0 + (300.0 if is_evolved else 0.0)
        # サポート特性持ち（キチキギスex、ラティアスex等）を倒すと妨害にもなる
        name = (getattr(bp.card, "name", "") or "").strip()
        _ABILITY_DENY_NAMES = frozenset({
            "キチキギスex", "ラティアスex", "ニャースex",
        })
        if name in _ABILITY_DENY_NAMES:
            value += 2000.0
        # HP低いほど倒しやすい（はどうづきで低HPを狩ってエネ加速する戦略）
        value += 500.0 - min(bp.hp or 0, 500)
        killable_bench_values[i] = value

    # 学習モデルのスコアを混合（利用可能な場合）
    try:
        from .decision_models import score_boss_targets
        ml_scores = score_boss_targets(state, state.current_player)
        if ml_scores and killable_bench:
            ml_dict = {idx: sc for idx, sc in ml_scores if idx >= 0}
            for bi in killable_bench:
                if bi in ml_dict:
                    # モデルスコア（0-1程度）をルールベーススケールに変換して加算
                    killable_bench_values[bi] += ml_dict[bi] * 5000.0
    except Exception:
        pass

    if not killable_bench:
        # KO可能ベンチがなくても、時間稼ぎ目的で弱いポケモンを引っ張る
        # 条件: 相手バトル場がダメージを与えてくる + ベンチにエネなし逃げコスト高いポケモンがいる
        from .damage import _opponent_max_effective_damage
        _opp_dmg = _opponent_max_effective_damage(state)
        if _opp_dmg > 0 and opp.bench:
            stall_candidates = []
            for i, bp in enumerate(opp.bench):
                if not bp or bp.hp is None or bp.hp <= 0:
                    continue
                retreat_cost = getattr(bp.card, "retreat_cost", 1) or 1
                energy = getattr(bp, "attached_energy", 0) or 0
                # 逃げにくく攻撃できないポケモンが最適
                stall_score = retreat_cost * 100 - energy * 200 - (bp.hp or 0)
                stall_candidates.append((i, stall_score))
            if stall_candidates:
                idx = max(stall_candidates, key=lambda x: x[1])[0]
                opp.active, opp.bench[idx] = opp.bench[idx], opp.active
                p.discard.append(p.hand.pop(hand_index))
                state.support_used_this_turn = True
                state.log(f"{state.player_name(state.current_player)}: ボスの指令を使用 → 相手のベンチとバトルポケモンを入れ替えた（{opp.active.card.name} がバトル場に）")
                return True
        return False

    idx = max(killable_bench, key=lambda bi: killable_bench_values.get(bi, 0.0))
    opp.active, opp.bench[idx] = opp.bench[idx], opp.active
    p.discard.append(p.hand.pop(hand_index))
    state.support_used_this_turn = True
    state.log(f"{state.player_name(state.current_player)}: ボスの指令を使用 → 相手のベンチとバトルポケモンを入れ替えた（{opp.active.card.name} がバトル場に）")
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
    cid = getattr(card, "id", "")
    if _is_first_player_first_turn(state) and cid != ZEIYU:
        return False

    # デッキ切れ防止: 山札が少ないときにドロー系サポートを使わない。
    # 残りサイド枚数 × 4ターン分のデッキ余裕が必要（毎ターン1ドロー+ルナ3+etc）。
    deck_size = len(p.deck)
    remaining_prizes = len(p.prize_pile)
    # リーリエの決心は手札を山札に戻すので実質デッキ減少は少ない。
    # ゼイユ/博士は手札をトラッシュして山札から引くのでデッキが大きく減る。
    _NET_DRAW_COSTS = {
        RIRIE_NO_KESSHIN: 0,  # 手札→山に戻してから引く: net ≈ 0
        ZEIYU: 5,             # 手札トラッシュ＋山から5枚引く
        HAKASE_NO_KENKYU: 7,
        "hakasenokenkyuufutouhakase": 7,
        TANPAN_KOZOU: 0,       # 手札→山に戻して5枚引く: net ≈ 0 (手札5枚程度)
        "nemo": 3, "nemokako": 3, "nemomirai": 3,
        KIHADA: 2,
        JUDGE: 0,       # ジャッジマンは手札→山に戻して4枚引く
        HIKARI: 4,            # 最大 4 枚ドロー
        MEI_NO_HAGEMASHI: 0,    # トラッシュからエネを付ける（デッキからドローしない）
        AKAMATSU: 0,          # エネを取るだけ（デッキ切れリスクは低い、エネ加速の生命線）
    }
    net_draw = _NET_DRAW_COSTS.get(cid, 0)
    # 固定ガード: draw直後のデッキが残りサイド×4+余裕分を下回るなら使わない
    min_deck_after = max(8, remaining_prizes * 4)
    if net_draw > 0 and (deck_size - net_draw) <= min_deck_after:
        return False
    # リーリエ等のシャッフルドロー系でも、デッキ自体が極端に少ないなら控える
    _DRAW_COUNTS = {
        RIRIE_NO_KESSHIN: 8,
        ZEIYU: 5,
        HAKASE_NO_KENKYU: 7,
        "hakasenokenkyuufutouhakase": 7,
        TANPAN_KOZOU: 5,
        "nemo": 3, "nemokako": 3, "nemomirai": 3,
        KIHADA: 2,
        JUDGE: 4,
        HIKARI: 4,
        MEI_NO_HAGEMASHI: 0,    # トラッシュからエネ付与（ドローなし）
        AKAMATSU: 2,
    }
    draw_count = _DRAW_COUNTS.get(cid, 0)
    if draw_count > 0 and deck_size <= draw_count + 3:
        return False
    effect = getattr(card, "effect", "")
    if cid == ZEIYU:
        used = p.hand.pop(hand_index)
        p.discard.extend(p.hand)
        p.hand.clear()
        p.discard.append(used)
        drawn = p.draw(5)
        p.hand.extend(drawn)
        state.drawn_this_turn.extend(drawn)
        state.support_used_this_turn = True
        drawn_names = ", ".join(_card_label(c) for c in drawn)
        state.log(f"{state.player_name(state.current_player)}: ゼイユを使用 → 手札をすべてトラッシュし、山札から 5 枚ドロー → [{drawn_names}]")
        return True
    if cid == BOSS_NO_SHIREI:
        return _use_support_boss(state, hand_index)
    if cid == RIRIE_NO_KESSHIN:
        used_card = p.hand[hand_index]
        rest = [p.hand[j] for j in range(len(p.hand)) if j != hand_index]
        p.deck.extend(rest)
        p.hand.clear()
        random.shuffle(p.deck)
        mark_own_deck_shuffled(state)
        n_draw = 8 if (len(p.prize_pile) == 6) else 6
        drawn = p.draw(min(n_draw, len(p.deck)))
        p.hand.extend(drawn)
        state.drawn_this_turn.extend(drawn)
        p.discard.append(used_card)
        state.support_used_this_turn = True
        drawn_names = ", ".join(_card_label(c) for c in drawn)
        state.log(f"{state.player_name(state.current_player)}: リーリエの決心を使用 → 手札を山札にもどして切り、山札から {len(drawn)} 枚ドロー → [{drawn_names}]")
        return True
    if cid == "angoumanianokaidoku":
        if len(p.deck) < 2:
            return False
        top2 = [p.deck.pop(0) for _ in range(2)]
        p.deck.extend(reversed(top2))
        p.discard.append(p.hand.pop(hand_index))
        state.support_used_this_turn = True
        new_top = _card_label(top2[1])
        new_second = _card_label(top2[0])
        state.log(f"{state.player_name(state.current_player)}: 暗号マニアの解読を使用 → 山札の上 2 枚の順序を入れ替えた（一番上: {new_top}、2 枚目: {new_second}）")
        return True
    if cid == TANPAN_KOZOU:
        n_hand = len(p.hand)
        used_card = p.hand[hand_index]
        rest = [p.hand[j] for j in range(len(p.hand)) if j != hand_index]
        p.deck.extend(rest)
        p.hand = []
        random.shuffle(p.deck)
        mark_own_deck_shuffled(state)
        drawn = p.draw(5)
        p.hand.extend(drawn)
        state.drawn_this_turn.extend(drawn)
        p.discard.append(used_card)
        state.support_used_this_turn = True
        drawn_names = ", ".join(_card_label(c) for c in drawn)
        state.log(f"{state.player_name(state.current_player)}: たんぱんこぞうを使用 → 手札 {n_hand} 枚を山札にもどして切り、山札から 5 枚ドロー → [{drawn_names}]（手札 {len(p.hand)} 枚）")
        return True
    if cid in (HAKASE_NO_KENKYU, "hakasenokenkyuufutouhakase"):
        p.discard.extend(p.hand)
        p.hand.clear()
        drawn = p.draw(7)
        p.hand.extend(drawn)
        state.drawn_this_turn.extend(drawn)
        state.support_used_this_turn = True
        drawn_names = ", ".join(_card_label(c) for c in drawn)
        state.log(f"{state.player_name(state.current_player)}: 博士の研究を使用 → 手札をすべてトラッシュし、山札から 7 枚ドロー → [{drawn_names}]（手札 {len(p.hand)} 枚）")
        return True
    if cid == JUDGE:
        opp = state.defending_player_state()
        # デッキ切れ防止: 山札が少ないとき、ジャッジマンで手札を山に戻して山札を回復
        deck_low = len(p.deck) <= 10
        if deck_low and len(p.hand) > 6:
            pass  # 山札回復のために使う（手札>相手でもOK）
        elif len(p.hand) > len(opp.hand):
            # 通常時: こちらの手札が多いと損
            return False
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
        mark_own_deck_shuffled(state)
        p_drawn = p.draw(4)
        p.hand = p_drawn
        state.drawn_this_turn.extend(p_drawn)
        p.discard.append(card)
        p_drawn_names = ", ".join(_card_label(c) for c in p_drawn)
        state.log(f"{state.player_name(state.current_player)}: ジャッジマンを使用 → おたがい手札を山札にもどして 4 枚ドロー → [{p_drawn_names}]")
        state.support_used_this_turn = True
        return True
    if cid == KIHADA:
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
    if cid == "mitsurunoomoiyari":
        ex_target = None
        if p.active and (getattr(p.active.card, "is_ex", False) or "ex" in (getattr(p.active.card, "name", "") or "")):
            cap_active = get_effective_max_hp(state, p.active.card)
            if p.active.hp < cap_active:
                ex_target = p.active
        if ex_target is None:
            for bp in p.bench:
                if getattr(bp.card, "is_ex", False) or "ex" in (getattr(bp.card, "name", "") or ""):
                    cap_bp = get_effective_max_hp(state, bp.card)
                    if bp.hp < cap_bp:
                        ex_target = bp
                        break
        if ex_target is None:
            return False
        before_hp = ex_target.hp
        cap = get_effective_max_hp(state, ex_target.card)
        # HPが半分以上残っているなら回復不要（温存）
        if before_hp > cap * 0.5:
            return False
        ex_target.hp = cap  # BattlePokemonのHPを回復（card.hpではない）
        nrg_count = ex_target.attached_energy
        types = getattr(ex_target, "attached_energy_types", [])[:]
        ex_target.attached_energy = 0
        ex_target.attached_energy_types = []
        # 付いていたエネルギーを実際のタイプに基づいてカードとして手札に戻す
        for etype in types:
            energy_id = f"basic-energy-{etype}" if etype != "colorless" else "basic-energy-colorless"
            try:
                c = get_card_by_id(energy_id)
                p.hand.append(c)
            except (ValueError, KeyError):
                # IDで見つからない場合はタイプ名で検索
                from card import EnergyCard
                p.hand.append(EnergyCard(
                    id=energy_id,
                    name=f"基本{etype}エネルギー",
                    energy_type=etype,
                ))
        p.discard.append(p.hand.pop(hand_index))
        state.support_used_this_turn = True
        state.log(f"{state.player_name(state.current_player)}: ミツルの思いやりを使用 → {ex_target.card.name} のHPを全回復（{before_hp} → {cap}）、ついているエネルギー {nrg_count} 個を手札にもどした")
        return True
    # ヒカリ: 山札から「たねポケモン」「1進化ポケモン」「2進化ポケモン」を1枚ずつ手札に加える
    if cid == HIKARI:
        fetched = []
        # 各進化段階の候補を収集
        basic_candidates = []
        stage1_candidates = []
        stage2_candidates = []
        for i, c in enumerate(p.deck):
            if not is_pokemon(c):
                continue
            stage = getattr(c, "evolution_stage", "basic")
            cname = (getattr(c, "name", "") or "").strip()
            if stage == "basic":
                basic_candidates.append((i, c, cname))
            elif stage == "stage1":
                stage1_candidates.append((i, c, cname))
            elif stage == "stage2":
                stage2_candidates.append((i, c, cname))
        # ドラパルトデッキ: メインアタッカーラインを優先
        if is_dragapult_deck_for_player(state, state.current_player):
            _drapa_basic_prio = {"ドラメシヤ": 100, "ヨマワル": 50, "スボミー": 30}
            _drapa_s1_prio = {"ドロンチ": 100, "サマヨール": 50}
            _drapa_s2_prio = {"ドラパルトex": 100, "ヨノワール": 50}
            basic_candidates.sort(key=lambda x: -_drapa_basic_prio.get(x[2], 0))
            stage1_candidates.sort(key=lambda x: -_drapa_s1_prio.get(x[2], 0))
            stage2_candidates.sort(key=lambda x: -_drapa_s2_prio.get(x[2], 0))
        found_basic = basic_candidates[0][0] if basic_candidates else None
        found_stage1 = stage1_candidates[0][0] if stage1_candidates else None
        found_stage2 = stage2_candidates[0][0] if stage2_candidates else None
        indices = [idx for idx in (found_basic, found_stage1, found_stage2) if idx is not None]
        if not indices:
            return False
        for idx in sorted(indices, reverse=True):
            c = p.deck.pop(idx)
            p.hand.append(c)
            state.drawn_this_turn.append(c)
            fetched.append(c)
        random.shuffle(p.deck)
        mark_own_deck_shuffled(state)
        mark_deck_searched(state)
        p.discard.append(p.hand.pop(hand_index))
        state.support_used_this_turn = True
        fetched_names = ", ".join(_card_label(c) for c in fetched)
        state.log(
            f"{state.player_name(state.current_player)}: ヒカリを使用 → 山札からたね・1進化・2進化ポケモンを合計 {len(fetched)} 枚手札に加えた → [{fetched_names}]（手札 {len(p.hand)} 枚）"
        )
        return True

    # アカマツ: 山札からタイプの違う基本エネルギーを2枚まで選び、1枚を手札に加え、残りを自分のポケモンにつける
    if cid == AKAMATSU:
        return _use_support_akamatsu(state, hand_index)

    if cid == BURAIA:
        opp = state.defending_player_state()
        if len(opp.prize_pile) != 2:
            return False
        # ブライアの効果: テラスタルポケモンの攻撃で相手のバトルポケモンをKOしたらサイド+1
        # briar_extra_prize フラグを立てる（攻撃後の KO 処理で参照）
        state.briar_extra_prize = True
        p.discard.append(p.hand.pop(hand_index))
        state.support_used_this_turn = True
        state.log(
            f"{state.player_name(state.current_player)}: ブライアを使用 → この番、テラスタルポケモンのワザでKOしたらサイドを1枚多くとる"
        )
        return True

    # メイのはげまし: 自分のサイドが相手より多いときのみ使用可。トラッシュから基本エネルギーを2枚まで、自分の2進化ポケモン1匹につける。
    if cid == MEI_NO_HAGEMASHI:
        return _use_support_mei(state, hand_index)

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


def play_stadium(state: GameState, hand_index: int) -> bool:
    """
    スタジアムを手札から場に出す。1 ターンに 1 枚まで。
    別名のスタジアムを出すとそれまで出ていたスタジアムはトラッシュ（現在のプレイヤーのトラッシュへ）。
    出ているスタジアムと同じ名前のスタジアムは出せない。
    """
    p = state.active_player_state()
    if hand_index < 0 or hand_index >= len(p.hand):
        return False
    card = p.hand[hand_index]
    if not is_stadium(card):
        return False
    if state.stadium_played_this_turn:
        return False
    current_name = (getattr(card, "name", "") or "").strip()
    if state.stadium is not None:
        in_play_name = (getattr(state.stadium, "name", "") or "").strip()
        if in_play_name == current_name:
            return False
        p.discard.append(state.stadium)
        state.log(f"{state.player_name(state.current_player)}: スタジアム {_card_label(state.stadium)} をトラッシュ")
    p.hand.pop(hand_index)
    state.stadium = card
    state.stadium_played_by = state.current_player
    state.stadium_played_this_turn = True
    state.log(f"{state.player_name(state.current_player)}: スタジアム {_card_label(card)} を場に出す")

    # グラビティーマウンテン: 2進化ポケモンの最大HPを-30 → 既存のポケモンのHPもキャップ
    _sid = (getattr(card, "id", "") or "").strip()
    _sname = (getattr(card, "name", "") or "").strip()
    if _sid == "gurabiteiimaunten" or "グラビティーマウンテン" in _sname:
        for _pi in range(2):
            _ps = state.players[_pi]
            for _bp in ([_ps.active] if _ps.active else []) + list(_ps.bench):
                if not _bp or not _bp.card:
                    continue
                _evo = getattr(_bp.card, "evolution_stage", None)
                if _evo == "stage2" or is_stage2_pokemon(_bp.card):
                    _new_max = max(0, (getattr(_bp.card, "max_hp", 0) or 0) - 30)
                    if _bp.hp is not None and _bp.hp > _new_max:
                        _bp.hp = _new_max

    return True
