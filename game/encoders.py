"""
学習用の state / action のエンコード。

- state: まずは 20〜30 次元程度のコンパクトな特徴量にする（データ不足で学習が壊れにくい）。
- action: まずは固定 action id（例: energy_attach の付与先 = active / bench0..4）にする。
"""

from __future__ import annotations

from dataclasses import dataclass

from .state import BENCH_SIZE, GameState, PRIZE_COUNT


def _bp_hp(bp) -> float:
    return float(getattr(bp, "hp", 0) or 0)


def _bp_energy(bp) -> float:
    return float(getattr(bp, "attached_energy", 0) or 0)


def _max_energy_for_pokemon(attacks: list) -> float:
    """そのポケモンの技で必要な最大エネルギー数（個数）を返す。"""
    return float(max((getattr(a, "energy_cost", 0) for a in attacks), default=0))


def _energy_progress(bp) -> float:
    """
    バトル場ポケモンのエネルギー進捗。

      energy_progress = attached_energy / attack_cost

    attack_cost はそのポケモンの技で必要な最大エネルギー数。
    """
    if not bp:
        return 0.0
    attached = _bp_energy(bp)
    attacks = getattr(getattr(bp, "card", None), "attacks", []) or []
    cost = _max_energy_for_pokemon(attacks)
    if cost <= 0.0:
        return 0.0
    return max(0.0, min(1.0, attached / cost))


def _can_attack_now(bp) -> float:
    """
    現時点で「どれかの技を撃てるか」の近似。
    タイプは無視し、必要エネルギー数だけを見る。
    """
    if not bp:
        return 0.0
    attached = _bp_energy(bp)
    attacks = getattr(getattr(bp, "card", None), "attacks", []) or []
    # 最小コストの技が撃てるかどうかで近似
    min_cost = float(min((getattr(a, "energy_cost", 0) for a in attacks), default=0))
    if min_cost <= 0.0:
        return 0.0
    return 1.0 if attached >= min_cost else 0.0


def _can_attack_next_turn(bp) -> float:
    """
    「もう 1 枚エネルギーを貼ればどれかの技が撃てるか」の近似。
    タイプは無視し、attached_energy+1 と必要エネルギー数だけを見る。
    """
    if not bp:
        return 0.0
    attached_next = _bp_energy(bp) + 1.0
    attacks = getattr(getattr(bp, "card", None), "attacks", []) or []
    min_cost = float(min((getattr(a, "energy_cost", 0) for a in attacks), default=0))
    if min_cost <= 0.0:
        return 0.0
    return 1.0 if attached_next >= min_cost else 0.0


def _prizes_remaining(p) -> float:
    return float(len(getattr(p, "prize_pile", []) or []))


def _prizes_taken(p) -> float:
    return float(PRIZE_COUNT - len(getattr(p, "prize_pile", []) or []))


def encode_state_basic(state: GameState, player_id: int) -> list[float]:
    """
    まずはシンプルな state encoder。

    features（合計 22 次元）:
      自分: prize_taken, active_hp, active_energy, active_energy_progress,
            can_attack_now, can_attack_next_turn,
            bench_count, bench_hp_sum, bench_energy_sum, hand_size
      相手: prize_taken, active_hp, active_energy, active_energy_progress,
            can_attack_now, can_attack_next_turn,
            bench_count, bench_hp_sum, bench_energy_sum
      ゲーム: turn_count, first_player_is_me

    注意:
      - active が無い場合は 0 埋め。
      - どのスケールで学習するかは後段のモデル側で調整する（まずは生値）。
    """
    me = state.players[player_id]
    opp = state.players[1 - player_id]

    me_active = getattr(me, "active", None)
    opp_active = getattr(opp, "active", None)

    me_bench = list(getattr(me, "bench", []) or [])
    opp_bench = list(getattr(opp, "bench", []) or [])

    f: list[float] = []

    # me
    f.append(_prizes_taken(me))
    f.append(_bp_hp(me_active) if me_active else 0.0)
    f.append(_bp_energy(me_active) if me_active else 0.0)
    f.append(_energy_progress(me_active))
    f.append(_can_attack_now(me_active))
    f.append(_can_attack_next_turn(me_active))
    f.append(float(len(me_bench)))
    f.append(float(sum(_bp_hp(bp) for bp in me_bench)))
    f.append(float(sum(_bp_energy(bp) for bp in me_bench)))
    f.append(float(len(getattr(me, "hand", []) or [])))

    # opp
    f.append(_prizes_taken(opp))
    f.append(_bp_hp(opp_active) if opp_active else 0.0)
    f.append(_bp_energy(opp_active) if opp_active else 0.0)
    f.append(_energy_progress(opp_active))
    f.append(_can_attack_now(opp_active))
    f.append(_can_attack_next_turn(opp_active))
    f.append(float(len(opp_bench)))
    f.append(float(sum(_bp_hp(bp) for bp in opp_bench)))
    f.append(float(sum(_bp_energy(bp) for bp in opp_bench)))

    # game
    f.append(float(getattr(state, "turn_count", 0) or 0))
    f.append(1.0 if getattr(state, "first_player", 0) == player_id else 0.0)

    return f


def encode_state_drapa(state: GameState, player_id: int) -> list[float]:
    """
    ドラパルトexデッキ特化エンコーダ（encode_state_basic + ファントムダイブ関連特徴量）。

    追加特徴量（約20次元追加、合計約41次元）:
      - ドラパルトex体数（場）
      - ドロンチ体数（場, ていさつしれいエンジン）
      - ドラメシヤ体数（場, 進化の基盤）
      - ドラパルトexの最大エネルギー数
      - ドラパルトexに炎エネ付いてるか
      - ドラパルトexに超エネ付いてるか
      - ファントムダイブ撃てるドラパルトexがいるか
      - ファントムダイブまでの不足エネ数（最もファントムダイブに近いドラパルトex）
      - 手札にアカマツ/メイのはげましがあるか
      - 手札にふしぎなアメがあるか
      - 手札にドラパルトex/ドロンチがあるか
      - 手札の炎/超エネ枚数
      - ヨノワールライン体数（場）
      - スボミーが場にいるか
      - 相手の最大打点 / 自分active HP比
      - 相手のベンチHP合計（ファントムダイブのダメカン配置ターゲット）
      - サイド差
      - デッキ残り枚数
    """
    # まず基本特徴量
    f = encode_state_basic(state, player_id)

    me = state.players[player_id]
    opp = state.players[1 - player_id]
    me_all_bp = ([me.active] if me.active else []) + list(me.bench or [])
    from card import is_support, is_goods, is_energy, is_pokemon

    def _name(bp):
        return (getattr(getattr(bp, "card", None), "name", "") or "").strip()

    def _types(bp):
        return list(getattr(bp, "attached_energy_types", []) or [])

    # ドラパルトexライン体数
    drapa_count = sum(1 for bp in me_all_bp if _name(bp) == "ドラパルトex")
    doronchi_count = sum(1 for bp in me_all_bp if _name(bp) == "ドロンチ")
    drameshiya_count = sum(1 for bp in me_all_bp if _name(bp) == "ドラメシヤ")
    f.append(float(drapa_count))
    f.append(float(doronchi_count))
    f.append(float(drameshiya_count))

    # ドラパルトexのエネルギー状況
    drapa_bps = [bp for bp in me_all_bp if _name(bp) == "ドラパルトex"]
    best_energy = 0.0
    has_fire = 0.0
    has_psychic = 0.0
    can_phantom_dive = 0.0
    energy_deficit = 3.0  # ファントムダイブまでの不足（最大3）
    for bp in drapa_bps:
        en = float(getattr(bp, "attached_energy", 0) or 0)
        types = _types(bp)
        if en > best_energy:
            best_energy = en
        if "fire" in types:
            has_fire = 1.0
        if "psychic" in types:
            has_psychic = 1.0
        # ファントムダイブ: fire + psychic + colorless = 3エネ
        _has_f = "fire" in types
        _has_p = "psychic" in types
        if _has_f and _has_p and en >= 3:
            can_phantom_dive = 1.0
        deficit = 3.0 - en
        if not _has_f:
            deficit = max(deficit, 1.0)
        if not _has_p:
            deficit = max(deficit, 1.0)
        energy_deficit = min(energy_deficit, max(0.0, deficit))
    f.append(best_energy)
    f.append(has_fire)
    f.append(has_psychic)
    f.append(can_phantom_dive)
    f.append(energy_deficit)

    # 手札情報
    hand = list(getattr(me, "hand", []) or [])
    has_akamatsu = float(any(is_support(c) and (getattr(c, "id", "") or "") == "akamatsu" for c in hand))
    has_mei = float(any(is_support(c) and (getattr(c, "id", "") or "") == "meinohagemashi" for c in hand))
    has_ame = float(any(is_goods(c) and (getattr(c, "id", "") or "") == "fushiginaame" for c in hand))
    has_drapa_hand = float(any(is_pokemon(c) and (getattr(c, "name", "") or "").strip() == "ドラパルトex" for c in hand))
    has_doronchi_hand = float(any(is_pokemon(c) and (getattr(c, "name", "") or "").strip() == "ドロンチ" for c in hand))
    fire_energy_hand = float(sum(1 for c in hand if is_energy(c) and getattr(c, "energy_type", None) == "fire"))
    psychic_energy_hand = float(sum(1 for c in hand if is_energy(c) and getattr(c, "energy_type", None) == "psychic"))
    f.append(has_akamatsu)
    f.append(has_mei)
    f.append(has_ame)
    f.append(has_drapa_hand)
    f.append(has_doronchi_hand)
    f.append(fire_energy_hand)
    f.append(psychic_energy_hand)

    # ヨノワールライン / スボミー
    yono_line = sum(1 for bp in me_all_bp if _name(bp) in ("ヨマワル", "サマヨール", "ヨノワール"))
    has_subomii = float(any(_name(bp) == "スボミー" for bp in me_all_bp))
    f.append(float(yono_line))
    f.append(has_subomii)

    # 相手の脅威度
    from .damage import _max_effective_damage_for_attacker
    opp_max_dmg = 0.0
    if opp.active and me.active:
        opp_max_dmg = float(_max_effective_damage_for_attacker(state, opp.active, me.active, 1 - player_id))
    my_hp = float(getattr(me.active, "hp", 0) or 0) if me.active else 0.0
    threat_ratio = opp_max_dmg / max(1.0, my_hp)  # 1.0以上なら即KO圏内
    f.append(threat_ratio)

    # サイド差・デッキ残り
    my_prizes = float(len(getattr(me, "prize_pile", []) or []))
    opp_prizes = float(len(getattr(opp, "prize_pile", []) or []))
    f.append(opp_prizes - my_prizes)  # 正なら有利
    f.append(float(len(getattr(me, "deck", []) or [])))

    return f


@dataclass(frozen=True)
class EnergyAttachAction:
    """
    energy_attach の固定 action id。

    0: active
    1..BENCH_SIZE: bench index 0..BENCH_SIZE-1
    """

    action_id: int

    @property
    def target_is_active(self) -> bool:
        return self.action_id == 0

    @property
    def bench_index(self) -> int | None:
        if self.action_id == 0:
            return None
        return self.action_id - 1


# ────────────────────────────────────────────────
# encode_state_opening: ポケモン種別 one-hot 付き拡張エンコーダ
# ────────────────────────────────────────────────

# ポケモン種別インデックス（0=empty, 1=rioru, 2=lucario, 3=solrock, 4=lunatone, 5=makunoshita, 6=hariyama, 7=other）
_PTYPE_EMPTY       = 0
_PTYPE_RIORU       = 1  # リオル
_PTYPE_LUCARIO     = 2  # ルカリオ系（リオル除く）
_PTYPE_SOLROCK     = 3  # ソルロック
_PTYPE_LUNATONE    = 4  # ルナトーン
_PTYPE_MAKUNOSHITA = 5  # マクノシタ
_PTYPE_HARIYAMA    = 6  # ハリテヤマ
_PTYPE_OTHER       = 7
_PTYPE_DIM         = 8  # one-hot の次元数


def _pokemon_type_idx(bp) -> int:
    """BattlePokemon → ポケモン種別インデックス。None/空スロットは EMPTY。"""
    if bp is None or getattr(bp, "card", None) is None:
        return _PTYPE_EMPTY
    card = bp.card
    cid = (getattr(card, "id", "") or "").lower()
    name = (getattr(card, "name", "") or "").strip()
    if cid.startswith("rioru") or name == "リオル":
        return _PTYPE_RIORU
    if "rukario" in cid or "mrukario" in cid or name in ("ルカリオ", "メガルカリオ", "メガルカリオex"):
        return _PTYPE_LUCARIO
    if "sorurokku" in cid or name == "ソルロック":
        return _PTYPE_SOLROCK
    if cid.startswith("runaton") or name == "ルナトーン":
        return _PTYPE_LUNATONE
    if "makunoshita" in cid or name == "マクノシタ":
        return _PTYPE_MAKUNOSHITA
    if "hariyama" in cid or name == "ハリテヤマ":
        return _PTYPE_HARIYAMA
    return _PTYPE_OTHER


def _ptype_onehot(bp) -> list[float]:
    """BattlePokemon → 8 次元 one-hot リスト。"""
    v = [0.0] * _PTYPE_DIM
    v[_pokemon_type_idx(bp)] = 1.0
    return v


def _slot_features(bp) -> list[float]:
    """
    1 スロット分の特徴量（12 次元）。
      8: pokemon type one-hot
      1: hp（生値）
      1: attached_energy
      1: can_attack_now
      1: can_attack_next_turn
    """
    f = _ptype_onehot(bp)
    f.append(_bp_hp(bp) if bp else 0.0)
    f.append(_bp_energy(bp) if bp else 0.0)
    f.append(_can_attack_now(bp))
    f.append(_can_attack_next_turn(bp))
    return f  # 12 次元


def encode_state_opening(state: GameState, player_id: int) -> list[float]:
    """
    序盤特化の拡張 state encoder。

    encode_state_basic（21 次元）に加えて、各スロットのポケモン種別 one-hot を付与する。

    追加 features（合計 +84 次元）:
      自分 active（12） + 自分 bench[0-4]（12×5=60）+ 相手 active（12） = 84 次元

    合計: 21 + 84 = 105 次元
    """
    base = encode_state_basic(state, player_id)

    me = state.players[player_id]
    opp = state.players[1 - player_id]
    me_bench = list(getattr(me, "bench", []) or [])

    extended: list[float] = []

    # 自分 active
    extended.extend(_slot_features(getattr(me, "active", None)))

    # 自分 bench[0..BENCH_SIZE-1]（空スロットは empty one-hot + 0 埋め）
    for i in range(BENCH_SIZE):
        bp = me_bench[i] if i < len(me_bench) else None
        extended.extend(_slot_features(bp))

    # 相手 active
    extended.extend(_slot_features(getattr(opp, "active", None)))

    return base + extended  # 21 + 84 = 105 次元


def encode_state_v2(state: GameState, player_id: int) -> list[float]:
    """
    V2 state encoder: openingエンコーダ（105次元）+ 手札/トラッシュ/山札情報。

    追加 features:
      手札: サポート種別フラグ(6), グッズ枚数(1), エネ枚数(1), ポケモン枚数(1),
            進化ポケモン枚数(1), ハイパーボール有無(1), ポケモンいれかえ有無(1),
            パワープロテイン枚数(1), 夜のタンカ有無(1)
      トラッシュ: 闘エネ枚数(1)
      山札: 残り枚数(1)
      相手: ベンチ各体の詳細(12×5=60)

    合計: 105 + 16 + 60 = 181 次元
    """
    from card import is_energy, is_goods, is_support, is_pokemon

    base = encode_state_opening(state, player_id)

    me = state.players[player_id]
    opp = state.players[1 - player_id]
    hand = getattr(me, "hand", []) or []
    discard = getattr(me, "discard", []) or []
    deck = getattr(me, "deck", []) or []

    extra: list[float] = []

    # 手札のサポート種別フラグ
    _SUPPORT_IDS = {
        "riirienokesshin": 0, "zeiyu": 1, "bosunoshirei": 2,
        "jixyajjiman": 3, "nemo": 4, "mitsurunoomoiyari": 5,
    }
    sup_flags = [0.0] * 6
    for c in hand:
        if is_support(c):
            sid = getattr(c, "id", "") or ""
            # nemokako, nemomirai もnemoとして扱う
            if sid in ("nemokako", "nemomirai"):
                sid = "nemo"
            if sid in ("hakasenokenkyuu", "hakasenokenkyuufutouhakase", "tanpankozou", "kihada"):
                sid = "zeiyu"  # ドロー系としてゼイユ枠
            idx = _SUPPORT_IDS.get(sid)
            if idx is not None:
                sup_flags[idx] = 1.0
    extra.extend(sup_flags)

    # 手札のグッズ/エネ/ポケモン枚数
    goods_count = sum(1 for c in hand if is_goods(c))
    energy_count = sum(1 for c in hand if is_energy(c) and getattr(c, "energy_type", None) == "fighting")
    pokemon_count = sum(1 for c in hand if is_pokemon(c))
    evo_count = sum(1 for c in hand if is_pokemon(c) and getattr(c, "evolves_from", None))
    extra.append(float(goods_count))
    extra.append(float(energy_count))
    extra.append(float(pokemon_count))
    extra.append(float(evo_count))

    # 特定グッズの有無
    has_hb = float(any(getattr(c, "id", "") == "haipaboru" for c in hand))
    has_swap = float(any(is_goods(c) and (getattr(c, "effect", None) == "swap_active" or getattr(c, "id", "") in ("pokemon_irekae", "pokemonirekae")) for c in hand))
    pp_count = sum(1 for c in hand if getattr(c, "id", "") == "pawaapurotein")
    has_tanka = float(any(getattr(c, "id", "") == "yorunotanka" for c in hand))
    extra.append(has_hb)
    extra.append(has_swap)
    extra.append(float(pp_count))
    extra.append(has_tanka)

    # トラッシュの闘エネ枚数
    trash_fighting = sum(1 for c in discard if is_energy(c) and getattr(c, "energy_type", None) == "fighting")
    extra.append(float(trash_fighting))

    # 山札残り
    extra.append(float(len(deck)))

    # 相手ベンチ各体の詳細
    opp_bench = list(getattr(opp, "bench", []) or [])
    for i in range(BENCH_SIZE):
        bp = opp_bench[i] if i < len(opp_bench) else None
        extra.extend(_slot_features(bp))

    return base + extra


def energy_attach_action_id_from_target_card_id(state: GameState, player_id: int, target_card_id: str) -> int | None:
    """
    choice_log の card_id（付与先ポケモン）から、固定 action id（active / bench0..）に変換する。
    同名・同 id が複数ある場合は「active を優先し、それ以外は先に見つかった bench」を採用する。
    """
    p = state.players[player_id]
    active = getattr(p, "active", None)
    if active and (getattr(active.card, "id", None) or getattr(active.card, "name", "")) == target_card_id:
        return 0

    bench = list(getattr(p, "bench", []) or [])
    for i, bp in enumerate(bench[:BENCH_SIZE]):
        cid = getattr(bp.card, "id", None) or getattr(bp.card, "name", "")
        if cid == target_card_id:
            return 1 + i
    return None

