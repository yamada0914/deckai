"""
負け試合から判断ポイント（ミス）を自動抽出し、学習データを収集する。

使い方:
  python scripts/collect_mistakes.py                          # デフォルト（3セット）
  python scripts/collect_mistakes.py --n 50                   # 各50試合
  python scripts/collect_mistakes.py --out datasets/mistakes.jsonl
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


def _analyze_boss_decision(state, player_idx: int) -> dict | None:
    """ボスの指令の判断ポイントを分析。"""
    from game.damage import _max_effective_damage_for_attacker
    from game.state import _prizes_for_ko
    from card import is_support

    p = state.players[player_idx]
    opp = state.players[1 - player_idx]

    # 手札にボスの指令があるか
    boss_in_hand = any(
        is_support(c) and (getattr(c, "id", "") or "") == "bosunoshirei"
        for c in p.hand
    )
    if not boss_in_hand or not p.active or not opp.bench:
        return None
    if state.support_used_this_turn:
        return None

    # 各ベンチポケモンをターゲットにした場合のスコア
    targets = []
    for i, bp in enumerate(opp.bench):
        if not bp or not bp.hp or bp.hp <= 0:
            continue
        dmg = _max_effective_damage_for_attacker(state, p.active, bp, player_idx)
        can_ko = dmg >= bp.hp
        prizes = _prizes_for_ko(bp)
        retreat_cost = getattr(bp.card, "retreat_cost", 1) or 1
        energy = getattr(bp, "attached_energy", 0) or 0
        targets.append({
            "bench_idx": i,
            "name": getattr(bp.card, "name", ""),
            "hp": bp.hp,
            "can_ko": can_ko,
            "prizes": prizes,
            "retreat_cost": retreat_cost,
            "energy": energy,
            "dmg": dmg,
        })

    # バトル場のスコア（ボス使わない場合）
    active_dmg = 0
    active_can_ko = False
    if opp.active and opp.active.hp and opp.active.hp > 0:
        active_dmg = _max_effective_damage_for_attacker(state, p.active, opp.active, player_idx)
        active_can_ko = active_dmg >= opp.active.hp

    if not targets:
        return None

    return {
        "type": "boss_target",
        "turn": state.turn_count,
        "player": player_idx,
        "our_prizes_remaining": len(p.prize_pile),
        "opp_prizes_remaining": len(opp.prize_pile),
        "active_target": {
            "name": getattr(opp.active.card, "name", "") if opp.active else "",
            "hp": opp.active.hp if opp.active else 0,
            "can_ko": active_can_ko,
            "dmg": active_dmg,
        },
        "bench_targets": targets,
    }


def _analyze_energy_decision(state, player_idx: int) -> dict | None:
    """エネルギー配置の判断ポイントを分析。"""
    from game.damage import _max_effective_damage_if_attach
    from card import is_energy

    p = state.players[player_idx]
    opp = state.players[1 - player_idx]

    if state.energy_attached_this_turn:
        return None
    energy_idx = next((i for i, c in enumerate(p.hand) if is_energy(c)), None)
    if energy_idx is None or not p.active:
        return None

    ecard = p.hand[energy_idx]
    new_type = getattr(ecard, "energy_type", None) or "colorless"

    candidates = []
    # バトル場
    if p.active:
        dmg = _max_effective_damage_if_attach(
            state, p.active.card, p.active.attached_energy,
            getattr(p.active, "attached_energy_types", []),
            new_type, opp.active, player_idx,
        )
        candidates.append({
            "target": "active",
            "name": getattr(p.active.card, "name", ""),
            "current_energy": p.active.attached_energy,
            "dmg_after": dmg,
            "hp": p.active.hp,
        })
    # ベンチ
    for bi, bp in enumerate(p.bench):
        dmg = _max_effective_damage_if_attach(
            state, bp.card, bp.attached_energy,
            getattr(bp, "attached_energy_types", []),
            new_type, opp.active, player_idx,
        )
        candidates.append({
            "target": f"bench:{bi}",
            "name": getattr(bp.card, "name", ""),
            "current_energy": bp.attached_energy,
            "dmg_after": dmg,
            "hp": bp.hp,
        })

    if len(candidates) <= 1:
        return None

    return {
        "type": "energy_attach",
        "turn": state.turn_count,
        "player": player_idx,
        "candidates": candidates,
    }


def _collect_one_game(deck0: int, deck1: int, seed: int, both_sides: bool = False) -> tuple[list[dict], int]:
    """1試合から判断ポイントを収集。"""
    from game import setup_game, run_game_auto, start_turn, end_turn, run_turn_auto
    from game.state import _check_game_end
    from game.encoders import encode_state_v2

    random.seed(seed)
    records = []

    # フックで各ターンの判断ポイントを記録
    import game.turn as gt
    _orig_run = gt.run_turn_auto

    def _hooked_run(state):
        player = state.current_player
        # 両サイド or 負けた側のみ記録（ゲーム終了後に勝敗でフィルタ）
        if both_sides or True:  # 一旦全部記録、後でフィルタ
            boss = _analyze_boss_decision(state, player)
            if boss:
                boss["state"] = encode_state_v2(state, player)
                boss["seed"] = seed
                records.append(boss)

            energy = _analyze_energy_decision(state, player)
            if energy:
                energy["state"] = encode_state_v2(state, player)
                energy["seed"] = seed
                records.append(energy)

        return _orig_run(state)

    gt.run_turn_auto = _hooked_run

    try:
        state = setup_game(seed=seed, deck0=deck0, deck1=deck1, record_frame_fn=lambda s: None)
        run_game_auto(state)
        winner = state.winner

        # 勝敗を記録に付与
        for r in records:
            r["won"] = int(winner == r["player"])
            if both_sides:
                pass  # 両方記録
            # ミラーでない場合はplayer 0のみ
    finally:
        gt.run_turn_auto = _orig_run

    return records, winner


def main():
    parser = argparse.ArgumentParser(description="負け試合から判断ポイントを収集")
    parser.add_argument("--n", type=int, default=100, help="各セットの試合数")
    parser.add_argument("--out", type=Path, default=_REPO_ROOT / "datasets" / "mistakes.jsonl")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    all_records = []

    matchups = [
        (5, 7, "vs_koraidon", False),
        (5, 8, "vs_miraidon", False),
        (5, 6, "mirror_lucario", True),
    ]

    for deck0, deck1, name, both_sides in matchups:
        wins = 0
        match_records = []
        for seed in range(args.n):
            records, winner = _collect_one_game(deck0, deck1, seed, both_sides=both_sides)
            if winner == 0:
                wins += 1

            for r in records:
                r["matchup"] = name
                r["deck0"] = deck0
                r["deck1"] = deck1
                match_records.append(r)

            if (seed + 1) % 25 == 0:
                elapsed = time.time() - t0
                print(f"  {name} [{seed+1}/{args.n}] wins={wins} records={len(match_records)} ({elapsed:.0f}s)")

        # フィルタ: 負けた試合の判断ポイントのみ（ミラーは両方）
        if both_sides:
            filtered = match_records  # ミラーは全記録
        else:
            filtered = [r for r in match_records if r["won"] == 0]  # 負け側のみ... いや勝ちも比較用に必要
            filtered = match_records  # 全部記録

        all_records.extend(filtered)
        print(f"  {name}: {wins}/{args.n} wins, {len(filtered)} decision points")

    # 保存
    with open(args.out, "w") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    elapsed = time.time() - t0
    total_loss = sum(1 for r in all_records if r["won"] == 0)
    total_win = sum(1 for r in all_records if r["won"] == 1)
    print(f"\n収集完了: {len(all_records)} decision points (win={total_win}, loss={total_loss}) ({elapsed:.0f}s)")
    print(f"出力: {args.out}")

    # 統計
    by_type = {}
    for r in all_records:
        t = r["type"]
        by_type.setdefault(t, {"total": 0, "win": 0, "loss": 0})
        by_type[t]["total"] += 1
        if r["won"]:
            by_type[t]["win"] += 1
        else:
            by_type[t]["loss"] += 1
    print("\n判断タイプ別:")
    for t, counts in by_type.items():
        print(f"  {t}: {counts['total']} (win={counts['win']}, loss={counts['loss']})")


if __name__ == "__main__":
    main()
