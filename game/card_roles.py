"""
カードの役割（ロール）に基づく判定。

「何のカードか」ではなく「何をするカードか」で分岐するための土台。
将来的には card.tags / card.role などのデータ駆動に移行する想定。
"""
import os

from card import is_pokemon
from .damage import _max_effective_damage_for_attacker

# --- 役割タグ（「今の役割」と「将来の役割」を分けると強い）---
# starter: 序盤に絶対欲しい（引けないと負け筋）
# future_attacker: 将来アタッカーになる（進化で main_attacker / finisher に）。序盤は置く・中盤は育てる
# main_attacker: 勝つために殴るやつ（ルカリオ、ドラパルトなど）
# finisher: 条件が揃って後半で強く使うフィニッシャー（例: ハリテヤマ）
# attacker_support: 攻撃を助ける（ソルロック、エネ加速系など）
# support: 汎用サポート（ドロー系、展開系）。1 枚が複数役割 OK（例: ヨマワル = starter + support）
ROLE_STARTER = "starter"
ROLE_FUTURE_ATTACKER = "future_attacker"
ROLE_MAIN_ATTACKER = "main_attacker"
ROLE_FINISHER = "finisher"
ROLE_ATTACKER_SUPPORT = "attacker_support"
ROLE_SUPPORT = "support"

# カード名 -> 役割の集合。1 枚が複数役割を持つ場合あり。
_CARD_ROLES: dict[str, frozenset[str]] = {
    # starter（序盤に欲しい）。ドラメシヤ・リオルは将来アタッカーになるので future_attacker も付与
    "ドラメシヤ": frozenset({ROLE_STARTER, ROLE_FUTURE_ATTACKER}),
    "リオル": frozenset({ROLE_STARTER, ROLE_FUTURE_ATTACKER}),
    "ホーホー": frozenset({ROLE_STARTER}),
    "スボミー": frozenset({ROLE_STARTER}),
    "ヨマワル": frozenset({ROLE_STARTER, ROLE_SUPPORT}),
    # main_attacker / finisher / attacker_support / support
    "ドラパルトex": frozenset({ROLE_MAIN_ATTACKER}),
    "ドロンチ": frozenset({ROLE_SUPPORT}),
    "キチキギスex": frozenset({ROLE_SUPPORT}),
    "ニャースex": frozenset({ROLE_SUPPORT}),
    "サマヨール": frozenset({ROLE_ATTACKER_SUPPORT}),
    "ヨノワール": frozenset({ROLE_ATTACKER_SUPPORT}),
    "マシマシラ": frozenset({ROLE_ATTACKER_SUPPORT}),
    "ルカリオ": frozenset({ROLE_MAIN_ATTACKER}),
    "マクノシタ": frozenset({ROLE_FUTURE_ATTACKER}),
    "ハリテヤマ": frozenset({ROLE_FINISHER, ROLE_ATTACKER_SUPPORT}),
    "ソルロック": frozenset({ROLE_ATTACKER_SUPPORT}),
    "ルナトーン": frozenset({ROLE_SUPPORT}),
}

# エンジンペアの片割れ（揃えると意味が出る組）。現状はルナトーン/ソルロック。将来は役割タグに移行。
_ENGINE_PAIR_NAMES: frozenset[str] = frozenset({
    "ルナトーン",
    "ソルロック",
})
_ENGINE_PAIR_IDS: frozenset[str] = frozenset({
    "runaton",
    "sorurokku-mc-372",
})

_HARIYAMA_RIOLU_BASE = float(os.getenv("DECKAI_HARIYAMA_RIOLU_BASE", "0.6"))
_ONLINE_ENERGY_FLOOR_BASE = float(os.getenv("DECKAI_ONLINE_ENERGY_FLOOR_BASE", "0.85"))
_ONLINE_HP_FLOOR_BASE = float(os.getenv("DECKAI_ONLINE_HP_FLOOR_BASE", "0.85"))
_ONLINE_KILL_BONUS_SCALE = float(os.getenv("DECKAI_ONLINE_KILL_BONUS_SCALE", "0.3"))
_ONLINE_COUNTER_PENALTY_MULT = float(os.getenv("DECKAI_ONLINE_COUNTER_PENALTY_MULT", "0.85"))


def get_roles(card) -> frozenset[str]:
    """カードが持つ役割の集合。該当なしなら空。"""
    if not card or not is_pokemon(card):
        return frozenset()
    name = (getattr(card, "name", "") or "").strip()
    return _CARD_ROLES.get(name, frozenset())


def has_role(card, role: str) -> bool:
    """指定した役割をカードが持つか。"""
    return role in get_roles(card)


def is_future_attacker(card) -> bool:
    """将来アタッカーになるか（進化で main_attacker に）。序盤は置く・中盤は育てる。"""
    return has_role(card, ROLE_FUTURE_ATTACKER)


def is_main_attacker(card) -> bool:
    """メインアタッカーか。"""
    return has_role(card, ROLE_MAIN_ATTACKER)


def is_finisher(card) -> bool:
    """フィニッシャーか（条件が揃って後半で強く使うタイプ）。"""
    return has_role(card, ROLE_FINISHER)


def is_attacker_support(card) -> bool:
    """アタッカー兼サポートか。"""
    return has_role(card, ROLE_ATTACKER_SUPPORT)


def is_support(card) -> bool:
    """サポートか。"""
    return has_role(card, ROLE_SUPPORT)


def is_starter(card) -> bool:
    """序盤に絶対欲しいカードか。初動・展開の起点になる。"""
    return has_role(card, ROLE_STARTER)


def is_engine_pair_member(card) -> bool:
    """エンジンペアの片割れか（例: ルナトーン/ソルロック）。場に片方いるともう片方を取りたくなるカード。"""
    if not card or not is_pokemon(card):
        return False
    name = (getattr(card, "name", "") or "").strip()
    cid = (getattr(card, "id", "") or "").strip()
    return name in _ENGINE_PAIR_NAMES or cid in _ENGINE_PAIR_IDS


def _hariyama_online_score_from_cards(cards) -> float:
    """カード列から、ハリテヤマの online 度合い（0.0〜1.0）を返す。"""
    has_lucario = False
    has_riolu = False
    for c in (cards or []):
        if not c or not is_pokemon(c):
            continue
        name = (getattr(c, "name", "") or "").strip()
        if name == "ルカリオ":
            has_lucario = True
        elif name == "リオル":
            has_riolu = True
    if has_lucario:
        return 1.0
    if has_riolu:
        return _clamp01(_HARIYAMA_RIOLU_BASE)
    return 0.0


def _iter_own_pokemon_from_state(state):
    """
    state から「自分側の可視ポケモン」を取り出す。
    優先順: active_player_state() -> players[current_player]。
    """
    player = None
    if state is None:
        return
    if hasattr(state, "active_player_state"):
        try:
            player = state.active_player_state()
        except Exception:
            player = None
    if player is None and hasattr(state, "players") and hasattr(state, "current_player"):
        try:
            player = state.players[state.current_player]
        except Exception:
            player = None
    if player is None:
        return
    for c in (getattr(player, "hand", None) or []):
        if is_pokemon(c):
            yield c
    active = getattr(player, "active", None)
    if active is not None and getattr(active, "card", None) is not None and is_pokemon(active.card):
        yield active.card
    for bp in (getattr(player, "bench", None) or []):
        c = getattr(bp, "card", None)
        if c is not None and is_pokemon(c):
            yield c


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _required_energy_for_online(card) -> int:
    """online 評価用の必要エネルギー（技コスト最小値）。"""
    attacks = getattr(card, "attacks", None) or []
    if not attacks:
        return 0
    costs: list[int] = []
    for atk in attacks:
        c = int(getattr(atk, "energy_cost", 0) or 0)
        costs.append(c)
    return min(costs) if costs else 0


def _find_own_battle_pokemon_by_name(state, card):
    """state から、自分側の同名 BattlePokemon（active/bench）を 1 体返す。なければ None。"""
    if state is None or not card:
        return None
    name = (getattr(card, "name", "") or "").strip()
    if not name:
        return None
    player = None
    if hasattr(state, "active_player_state"):
        try:
            player = state.active_player_state()
        except Exception:
            player = None
    if player is None:
        return None
    active = getattr(player, "active", None)
    if active is not None and getattr(active, "card", None) is not None:
        if (getattr(active.card, "name", "") or "").strip() == name:
            return active
    for bp in (getattr(player, "bench", None) or []):
        c = getattr(bp, "card", None)
        if c is not None and (getattr(c, "name", "") or "").strip() == name:
            return bp
    return None


def _base_online_score(card, cards) -> float:
    """カード固有の前提条件だけを見た online ベース（0.0〜1.0）。"""
    name = (getattr(card, "name", "") or "").strip()
    if name == "ハリテヤマ":
        return _hariyama_online_score_from_cards(cards)
    return 1.0


def is_online(card, state) -> float:
    """
    カードが現在状態で「機能する（online）」度合い（0.0〜1.0）。
    条件が未定義のカードは 1.0 を返す（安全側）。
    """
    if not card or not is_pokemon(card):
        return 0.0
    if not use_online_eval(state):
        return 1.0
    cards = list(_iter_own_pokemon_from_state(state))
    online = _base_online_score(card, cards)

    # 以降は「いま場にいる同名個体」があるときだけ、連続値で精密化する。
    bp = _find_own_battle_pokemon_by_name(state, card)
    if bp is None:
        return _clamp01(online)

    # 1) energy 要素（0 問題を避けるため、掛け算は floor 付きで安定化）
    required = _required_energy_for_online(bp.card)
    attached = float(getattr(bp, "attached_energy", 0) or 0)
    energy_ratio = 1.0 if required <= 0 else _clamp01(attached / float(required))
    online *= (_ONLINE_ENERGY_FLOOR_BASE + (1.0 - _ONLINE_ENERGY_FLOOR_BASE) * energy_ratio)

    # 2) HP 要素（生存性、同様に floor 付き）
    max_hp = float(getattr(bp.card, "max_hp", 0) or 0)
    cur_hp = float(getattr(bp, "hp", 0) or 0)
    hp_ratio = 1.0 if max_hp <= 0 else _clamp01(cur_hp / max_hp)
    online *= (_ONLINE_HP_FLOOR_BASE + (1.0 - _ONLINE_HP_FLOOR_BASE) * hp_ratio)

    # 3) 倒し切り要素（撃てる度合いに応じて加点）
    try:
        defender = state.defending_player_state().active if state and hasattr(state, "defending_player_state") else None
    except Exception:
        defender = None
    if defender is not None:
        try:
            player_index = int(getattr(state, "current_player", 0))
            dmg = _max_effective_damage_for_attacker(state, bp, defender, player_index)
            if dmg >= (getattr(defender, "hp", 0) or 0):
                online += _ONLINE_KILL_BONUS_SCALE * energy_ratio
        except Exception:
            pass

    # 4) 相手の反撃要素（相手 active がこのポケモンを倒せるなら減衰）
    try:
        opp = state.defending_player_state() if state and hasattr(state, "defending_player_state") else None
    except Exception:
        opp = None
    if opp is not None and getattr(opp, "active", None) is not None:
        try:
            player_index = 1 - int(getattr(state, "current_player", 0))
            opp_dmg = _max_effective_damage_for_attacker(state, opp.active, bp, player_index)
            if opp_dmg >= (getattr(bp, "hp", 0) or 0):
                online *= _ONLINE_COUNTER_PENALTY_MULT
        except Exception:
            pass

    return _clamp01(online)


def use_online_eval(state) -> bool:
    """現在手番プレイヤーで online 評価を使うか。無効化時は is_online を中立値 1.0 として扱う。"""
    if state is None:
        return True
    flags = getattr(state, "online_eval_enabled_by_player", None)
    cp = int(getattr(state, "current_player", 0) or 0)
    if isinstance(flags, list) and 0 <= cp < len(flags):
        return bool(flags[cp])
    return True


def is_hariyama_online(in_play_cards) -> bool:
    """
    ハリテヤマが「実質アタッカー」として機能する状態か。
    互換のため残す。新規コードは is_online(card, state) を使う。
    """
    return _hariyama_online_score_from_cards(in_play_cards) >= 1.0
