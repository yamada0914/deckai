"""
ターン内の確定行動で攻撃可能な手順を全探索する。

ドロー（ルナサイクル、サポート）は不確定なので除外。
手札にあるカード＋場の状態だけで「このターンKOできるか」を判断する。

行動空間（確定要素のみ）:
  - エネルギー付与: バトル場 or ベンチ各体
  - 進化: 手札の進化カード × 場の対象
  - にげる: バトル場 → ベンチ各体（コスト払える場合）
  - 攻撃: 合法技から選択

最大探索深度5（1ターンの行動回数上限）で全通り試す。
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional

from card import is_energy, is_pokemon
from .state import GameState, _can_pay_energy_cost, _is_first_player_first_turn
from .attack import get_legal_attack_indices
from .damage import _max_effective_damage_for_attacker
from .evolution import _can_evolve_onto


@dataclass
class PlanStep:
    action: str  # "energy", "evolve", "retreat", "attack"
    target: str  # 説明用
    detail: dict  # 実行に必要な情報


@dataclass
class AttackPlan:
    """KO可能な攻撃手順"""
    steps: list[PlanStep]
    damage: int
    attack_name: str


def _clone_minimal(state: GameState):
    """探索用の軽量クローン（最小限のコピー）"""
    import copy
    s = copy.copy(state)
    s.players = [copy.copy(p) for p in state.players]
    for i, p in enumerate(s.players):
        if p.active:
            p.active = copy.copy(p.active)
            p.active.card = copy.copy(p.active.card)
            p.active.attached_energy_types = list(p.active.attached_energy_types or [])
        p.bench = [copy.copy(bp) for bp in p.bench]
        for bp in p.bench:
            bp.card = copy.copy(bp.card)
            bp.attached_energy_types = list(bp.attached_energy_types or [])
        p.hand = list(p.hand)
        p.deck = list(p.deck)
        p.discard = list(p.discard)
        p.prize_pile = list(p.prize_pile)
    s.log_fn = None
    s.record_frame_fn = None
    return s


def _try_energy_attach(state: GameState) -> list[tuple[GameState, PlanStep]]:
    """エネ付与の全候補を返す（付与済みstateとステップのペア）"""
    if state.energy_attached_this_turn:
        return []
    p = state.active_player_state()
    results = []
    energy_indices = [i for i, c in enumerate(p.hand) if is_energy(c)]
    if not energy_indices:
        return []
    eidx = energy_indices[0]
    ecard = p.hand[eidx]
    etype = getattr(ecard, "energy_type", None) or "colorless"

    # バトル場
    if p.active:
        _act_name = (getattr(p.active.card, "name", "") or "").strip()
        _skip_active = (
            (_act_name == "ルナトーン")
            or (_act_name == "ソルロック" and p.active.attached_energy >= 1)
            or (_act_name == "マクノシタ" and p.active.attached_energy == 0)
        )
        if not _skip_active:
            s = _clone_minimal(state)
            sp = s.active_player_state()
            sp.active.attached_energy += 1
            sp.active.attached_energy_types.append(etype)
            sp.hand.pop(eidx)
            s.energy_attached_this_turn = True
            results.append((s, PlanStep("energy", f"active:{p.active.card.name}", {})))

    # ベンチ
    for bi, bp in enumerate(p.bench):
        _bp_name = (getattr(bp.card, "name", "") or "").strip()
        _skip_bench = (
            (_bp_name == "ルナトーン")
            or (_bp_name == "ソルロック" and bp.attached_energy >= 1)
            or (_bp_name == "マクノシタ" and bp.attached_energy == 0)
        )
        if _skip_bench:
            continue
        s = _clone_minimal(state)
        sp = s.active_player_state()
        sp.bench[bi].attached_energy += 1
        sp.bench[bi].attached_energy_types.append(etype)
        sp.hand.pop(eidx)
        s.energy_attached_this_turn = True
        results.append((s, PlanStep("energy", f"bench:{bp.card.name}", {})))

    return results


def _try_evolve(state: GameState) -> list[tuple[GameState, PlanStep]]:
    """進化の全候補"""
    if state.turn_count < 2:
        return []
    p = state.active_player_state()
    results = []
    for hi, hc in enumerate(p.hand):
        if not is_pokemon(hc) or not getattr(hc, "evolves_from", None):
            continue
        # バトル場
        if p.active and _can_evolve_onto(p.active.card, hc):
            if not getattr(p.active, "evolved_this_turn", False) and not getattr(p.active, "put_on_bench_this_turn", False):
                s = _clone_minimal(state)
                sp = s.active_player_state()
                old_e = sp.active.attached_energy
                old_et = list(sp.active.attached_energy_types)
                evolved = hc.copy()
                sp.active.card = evolved
                sp.active.attached_energy = old_e
                sp.active.attached_energy_types = old_et
                sp.active.evolved_this_turn = True
                sp.hand.pop(hi)
                results.append((s, PlanStep("evolve", f"active→{hc.name}", {})))
        # ベンチ
        for bi, bp in enumerate(p.bench):
            if _can_evolve_onto(bp.card, hc):
                if not getattr(bp, "evolved_this_turn", False) and not getattr(bp, "put_on_bench_this_turn", False):
                    s = _clone_minimal(state)
                    sp = s.active_player_state()
                    old_e = sp.bench[bi].attached_energy
                    old_et = list(sp.bench[bi].attached_energy_types)
                    evolved = hc.copy()
                    sp.bench[bi].card = evolved
                    sp.bench[bi].attached_energy = old_e
                    sp.bench[bi].attached_energy_types = old_et
                    sp.bench[bi].evolved_this_turn = True
                    sp.hand.pop(hi)
                    results.append((s, PlanStep("evolve", f"bench:{bp.card.name}→{hc.name}", {})))
    return results


def _try_retreat(state: GameState) -> list[tuple[GameState, PlanStep]]:
    """にげるの全候補"""
    if getattr(state, "retreat_used_this_turn", False):
        return []
    p = state.active_player_state()
    if not p.active or not p.bench:
        return []
    if getattr(p.active, "special_state", None) in ("sleep", "paralysis"):
        return []
    raw_rc = getattr(p.active.card, "retreat_cost", 1)
    tool = getattr(p.active, "attached_tool", None)
    eff_rc = max(0, raw_rc - (2 if tool and (getattr(tool, "id", "") or "") == "fuusen" else 0))
    if p.active.attached_energy < eff_rc:
        return []

    results = []
    for bi in range(len(p.bench)):
        s = _clone_minimal(state)
        sp = s.active_player_state()
        # エネルギーを消費
        if eff_rc > 0:
            sp.active.attached_energy -= eff_rc
            types = sp.active.attached_energy_types
            if len(types) >= eff_rc:
                sp.active.attached_energy_types = types[:-eff_rc]
            else:
                sp.active.attached_energy_types = []
        sp.active, sp.bench[bi] = sp.bench[bi], sp.active
        s.retreat_used_this_turn = True
        results.append((s, PlanStep("retreat", f"→{p.bench[bi].card.name}", {})))

    return results


def _try_bench_pokemon(state: GameState) -> list[tuple[GameState, PlanStep]]:
    """手札のたねポケモンをベンチに出す"""
    from .state import BENCH_SIZE, BattlePokemon
    p = state.active_player_state()
    if len(p.bench) >= BENCH_SIZE:
        return []
    results = []
    seen = set()
    for hi, hc in enumerate(p.hand):
        if not is_pokemon(hc) or getattr(hc, "evolves_from", None):
            continue
        hname = getattr(hc, "name", "")
        if hname in seen:
            continue
        seen.add(hname)
        s = _clone_minimal(state)
        sp = s.active_player_state()
        card = sp.hand.pop(hi)
        bp = BattlePokemon(card=card.copy())
        sp.bench.append(bp)
        results.append((s, PlanStep("bench", f"{hname}", {})))
    return results


def _try_attach_tool(state: GameState) -> list[tuple[GameState, PlanStep]]:
    """ふうせん等のどうぐ装着の全候補"""
    from card import is_goods
    p = state.active_player_state()
    results = []
    for hi, hc in enumerate(p.hand):
        if not is_goods(hc) or not getattr(hc, "is_tool", False):
            continue
        tool_id = getattr(hc, "id", "") or ""
        # バトル場
        if p.active and getattr(p.active, "attached_tool", None) is None:
            cond = getattr(hc, "tool_condition_type", None)
            if cond is None or getattr(p.active.card, "pokemon_type", None) == cond:
                s = _clone_minimal(state)
                sp = s.active_player_state()
                sp.active.attached_tool = sp.hand.pop(hi)
                results.append((s, PlanStep("tool", f"active:{p.active.card.name}←{hc.name}", {"hand_idx": hi})))
        # ベンチ
        for bi, bp in enumerate(p.bench):
            if getattr(bp, "attached_tool", None) is not None:
                continue
            cond = getattr(hc, "tool_condition_type", None)
            if cond is not None and getattr(bp.card, "pokemon_type", None) != cond:
                continue
            s = _clone_minimal(state)
            sp = s.active_player_state()
            sp.bench[bi].attached_tool = sp.hand.pop(hi)
            results.append((s, PlanStep("tool", f"bench:{bp.card.name}←{hc.name}", {"hand_idx": hi})))
    return results


def _try_power_protein(state: GameState) -> list[tuple[GameState, PlanStep]]:
    """パワープロテイン使用"""
    from card import is_goods
    p = state.active_player_state()
    results = []
    for hi, hc in enumerate(p.hand):
        if not is_goods(hc) or (getattr(hc, "id", "") or "") != "pawaapurotein":
            continue
        if not p.active or getattr(p.active.card, "pokemon_type", None) != "fighting":
            continue
        s = _clone_minimal(state)
        sp = s.active_player_state()
        sp.hand.pop(hi)
        n = getattr(s, "fighting_damage_plus_30_count_this_turn", 0)
        s.fighting_damage_plus_30_count_this_turn = n + 1
        results.append((s, PlanStep("power_protein", f"+30({n+1}枚目)", {"hand_idx": hi})))
        break  # 1回で十分（探索で複数枚は再帰で見つかる）
    return results


def _try_fight_gong(state: GameState) -> list[tuple[GameState, PlanStep]]:
    """ファイトゴング使用の全候補（闘たねポケモン or 基本闘エネルギー）"""
    from card import is_goods
    p = state.active_player_state()
    results = []
    fg_idx = next((i for i, c in enumerate(p.hand) if getattr(c, "id", "") == "faitogongu"), None)
    if fg_idx is None or not p.deck:
        return results

    def _is_basic_fighting_energy(x):
        if not is_energy(x):
            return False
        return (getattr(x, "id", "") or "") in ("basic-energy-fighting", "kihontouenerugi") or getattr(x, "name", "") == "基本闘エネルギー"

    # KO探索用: アタッカーになれるたね（ソルロック）のみ。エネルギーはKOに直結しない。
    seen = set()
    for di, dc in enumerate(p.deck):
        is_target = False
        if is_pokemon(dc) and getattr(dc, "pokemon_type", None) == "fighting" and getattr(dc, "evolution_stage", None) == "basic":
            is_target = True
        if not is_target:
            continue
        dname = getattr(dc, "name", "")
        if dname in seen:
            continue
        seen.add(dname)

        s = _clone_minimal(state)
        sp = s.active_player_state()
        sp.hand.pop(fg_idx)
        sp.discard.append(dc)  # ファイトゴング自体はトラッシュ
        for si, sc in enumerate(sp.deck):
            if getattr(sc, "name", "") == dname:
                sp.deck.pop(si)
                sp.hand.append(sc)
                break
        results.append((s, PlanStep("fight_gong", f"→{dname}", {"target_name": dname})))

    return results


def _try_hyper_ball(state: GameState) -> list[tuple[GameState, PlanStep]]:
    """ハイパーボール使用の全候補（山札の各ポケモンを取得するパターン）"""
    p = state.active_player_state()
    results = []
    hb_idx = next((i for i, c in enumerate(p.hand) if getattr(c, "id", "") == "haipaboru"), None)
    if hb_idx is None or len(p.hand) < 3 or not p.deck:
        return results

    # KO探索用: 進化ポケモンのみ候補にする（たねはファイトゴングで取れる）
    seen_names = set()
    for di, dc in enumerate(p.deck):
        if not is_pokemon(dc):
            continue
        if not getattr(dc, "evolves_from", None):
            continue  # たねはスキップ
        dname = getattr(dc, "name", "")
        if dname in seen_names:
            continue
        seen_names.add(dname)

        s = _clone_minimal(state)
        sp = s.active_player_state()
        # ハイパーボール消費
        sp.hand.pop(hb_idx)
        # 2枚捨て（探索用なので適当に末尾2枚）
        discarded = 0
        for ri in range(len(sp.hand) - 1, -1, -1):
            if discarded >= 2:
                break
            sp.discard.append(sp.hand.pop(ri))
            discarded += 1
        # ポケモン取得
        for si, sc in enumerate(sp.deck):
            if getattr(sc, "name", "") == dname:
                sp.deck.pop(si)
                sp.hand.append(sc)
                break
        results.append((s, PlanStep("hyper_ball", f"→{dname}", {"target_name": dname})))

    return results


def _check_ko(state: GameState) -> Optional[AttackPlan]:
    """現在の状態で攻撃してKOできるか"""
    p = state.active_player_state()
    opp = state.defending_player_state()
    if not p.active or not opp.active or opp.active.hp is None or opp.active.hp <= 0:
        return None
    if _is_first_player_first_turn(state):
        return None
    if getattr(p.active, "special_state", None) in ("sleep", "paralysis"):
        return None
    # ダメージ無効状態（なぐってかくれる等）の相手には攻撃しても意味がない
    if getattr(opp.active, "protected_next_opponent_turn", False):
        return None

    legal = get_legal_attack_indices(state, p, opp)
    for idx in legal:
        atk = p.active.card.attacks[idx]
        dmg = _max_effective_damage_for_attacker(state, p.active, opp.active, state.current_player)
        if dmg >= opp.active.hp:
            return AttackPlan(steps=[], damage=dmg, attack_name=atk.name)
    return None


def find_ko_plan(state: GameState, max_depth: int = 6, _visited: set | None = None) -> Optional[AttackPlan]:
    """
    確定行動の組み合わせでKO可能な手順を全探索。
    枝刈り: 同じ盤面状態を再訪しない、KOに近い行動を先に試す。
    """
    # まず現状でKOできるか
    plan = _check_ko(state)
    if plan is not None:
        return plan

    if max_depth <= 0:
        return None

    if _visited is None:
        _visited = set()

    # 盤面のハッシュで重複排除
    p = state.active_player_state()
    opp = state.defending_player_state()
    hand_sig = tuple(sorted(getattr(c, "name", "") for c in p.hand))
    sig = (
        getattr(p.active.card, "name", "") if p.active else "",
        p.active.attached_energy if p.active else 0,
        getattr(p.active, "attached_tool", None) is not None if p.active else False,
        tuple(sorted((bp.card.name, bp.attached_energy) for bp in p.bench)),
        hand_sig,
        getattr(state, "energy_attached_this_turn", False),
        getattr(state, "retreat_used_this_turn", False),
        getattr(state, "fighting_damage_plus_30_count_this_turn", 0),
    )
    if sig in _visited:
        return None
    _visited.add(sig)

    # 各行動を試す（にげるを先に試して無駄なエネ付与を避ける）
    candidates = []
    candidates.extend(_try_retreat(state))
    candidates.extend(_try_evolve(state))
    candidates.extend(_try_power_protein(state))
    candidates.extend(_try_energy_attach(state))
    candidates.extend(_try_attach_tool(state))
    candidates.extend(_try_fight_gong(state))
    candidates.extend(_try_bench_pokemon(state))
    candidates.extend(_try_hyper_ball(state))

    for new_state, step in candidates:
        sub_plan = find_ko_plan(new_state, max_depth - 1, _visited)
        if sub_plan is not None:
            sub_plan.steps.insert(0, step)
            return sub_plan

    return None
