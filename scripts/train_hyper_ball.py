"""
ハイパーボールのトラッシュ選択を学習する（V2: ゲーム結果ベース）。

方式:
1. ゲーム実行中、HB使用時に一定確率でランダムなトラッシュを選択
2. 各HB使用の (盤面, カード特徴, 選択, ゲーム勝敗) を記録
3. 各カードの「捨てやすさ」をゲーム勝率から学習

出力: datasets/hyper_ball_data.jsonl, models/hyper_ball/
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# 1. データ収集: ランダム化したトラッシュ選択でゲーム結果を記録
# ---------------------------------------------------------------------------

def _collect_games(n_games: int, deck0: int, deck1: int, explore_rate: float = 0.3) -> list[dict]:
    """複数試合からHBトラッシュデータを収集。

    explore_rate の確率でランダムなトラッシュ選択を行い、
    ゲーム結果と紐づけて記録する。
    """
    from game import setup_game, run_game_auto
    from game.encoders import encode_state_v2
    from card import is_goods, is_pokemon, is_energy, is_support
    import game.trainers as gt
    import game.turn as g_turn

    _orig = gt.use_trainer_goods
    all_records = []

    for seed in range(n_games):
        random.seed(seed)
        game_records = []  # このゲーム中のHB使用記録

        def _hooked(state, hand_index, **kw):
            p = state.active_player_state()
            card = p.hand[hand_index] if hand_index < len(p.hand) else None
            cid = getattr(card, "id", "") if card else ""

            if cid == "haipaboru" and len(p.hand) >= 4 and p.deck:
                hand_without_hb = [(i, p.hand[i]) for i in range(len(p.hand)) if i != hand_index]

                if len(hand_without_hb) >= 3 and random.random() < explore_rate:
                    # ランダムなトラッシュ選択を強制
                    state_vec = encode_state_v2(state, state.current_player)
                    player_idx = state.current_player

                    # カード特徴
                    card_features = []
                    for i, c in hand_without_hb:
                        evolves_from = getattr(c, "evolves_from", None)
                        has_evo_target = False
                        if evolves_from:
                            for bp in [p.active] + p.bench:
                                if bp and getattr(bp.card, "name", "") == evolves_from:
                                    has_evo_target = True
                                    break

                        cname = getattr(c, "name", "")
                        on_field = any(
                            bp and getattr(bp.card, "name", "") == cname
                            for bp in [p.active] + p.bench
                        )

                        cf = {
                            "idx": i,
                            "name": cname,
                            "card_id": getattr(c, "id", ""),
                            "is_energy": int(is_energy(c)),
                            "is_support": int(is_support(c)),
                            "is_pokemon": int(is_pokemon(c)),
                            "is_basic": int(is_pokemon(c) and not bool(getattr(c, "evolves_from", None))),
                            "is_evolution": int(bool(getattr(c, "evolves_from", None))),
                            "is_ex": int(bool(getattr(c, "is_ex", False))),
                            "is_goods": int(is_goods(c)),
                            "has_evo_target": int(has_evo_target),
                            "on_field": int(on_field),
                            "hp": getattr(c, "hp", 0),
                        }
                        card_features.append(cf)

                    # ランダムに2枚選んでトラッシュ
                    combo = random.sample(range(len(hand_without_hb)), 2)
                    chosen_indices = [hand_without_hb[c][0] for c in combo]

                    # 選択されたカードのマスク（1=捨てた、0=残した）
                    discard_mask = [0] * len(hand_without_hb)
                    for c in combo:
                        discard_mask[c] = 1

                    game_records.append({
                        "state": state_vec,
                        "card_features": card_features,
                        "discard_mask": discard_mask,
                        "turn": state.turn_count,
                        "player": player_idx,
                        "seed": seed,
                    })

                    # ランダムなトラッシュを実行（通常のuse_trainer_goodsの代わり）
                    from game.trainers import _find_pokemon_for_haipaboru, mark_own_deck_shuffled
                    for di in sorted(chosen_indices, reverse=True):
                        p.discard.append(p.hand.pop(di))

                    new_hi = p.hand.index(card)
                    found = _find_pokemon_for_haipaboru(p)
                    if found:
                        fi, fc = found
                        p.deck.pop(fi)
                        p.hand.append(fc)
                        state.drawn_this_turn.append(fc)
                        random.shuffle(p.deck)
                        mark_own_deck_shuffled(state)
                    p.discard.append(p.hand.pop(new_hi))
                    return True

            return _orig(state, hand_index, **kw)

        gt.use_trainer_goods = _hooked
        g_turn.use_trainer_goods = _hooked

        try:
            state = setup_game(deck0=deck0, deck1=deck1, record_frame_fn=lambda s: None)
            run_game_auto(state)

            # ゲーム結果を各記録に付与
            winner = state.winner
            for rec in game_records:
                rec["won"] = int(winner == rec["player"])
        except Exception:
            pass

        all_records.extend(game_records)

    gt.use_trainer_goods = _orig
    g_turn.use_trainer_goods = _orig
    return all_records


# ---------------------------------------------------------------------------
# 2. 通常プレイデータも収集（ルールベースの選択 + 勝敗）
# ---------------------------------------------------------------------------

def _collect_normal_games(n_games: int, deck0: int, deck1: int) -> list[dict]:
    """ルールベースの通常選択 + 勝敗を記録。"""
    from game import setup_game, run_game_auto
    from game.encoders import encode_state_v2
    from card import is_goods, is_pokemon, is_energy, is_support
    import game.trainers as gt
    import game.turn as g_turn

    _orig = gt.use_trainer_goods
    all_records = []

    for seed in range(n_games):
        random.seed(seed + 100000)  # 探索データとシードを分離
        game_records = []

        def _hooked(state, hand_index, **kw):
            p = state.active_player_state()
            card = p.hand[hand_index] if hand_index < len(p.hand) else None
            cid = getattr(card, "id", "") if card else ""

            if cid == "haipaboru" and len(p.hand) >= 4 and p.deck:
                hand_without_hb = [(i, p.hand[i]) for i in range(len(p.hand)) if i != hand_index]
                if len(hand_without_hb) >= 3:
                    state_vec = encode_state_v2(state, state.current_player)
                    player_idx = state.current_player

                    card_features = []
                    for i, c in hand_without_hb:
                        evolves_from = getattr(c, "evolves_from", None)
                        has_evo_target = False
                        if evolves_from:
                            for bp in [p.active] + p.bench:
                                if bp and getattr(bp.card, "name", "") == evolves_from:
                                    has_evo_target = True
                                    break
                        cname = getattr(c, "name", "")
                        on_field = any(
                            bp and getattr(bp.card, "name", "") == cname
                            for bp in [p.active] + p.bench
                        )
                        cf = {
                            "idx": i, "name": cname, "card_id": getattr(c, "id", ""),
                            "is_energy": int(is_energy(c)), "is_support": int(is_support(c)),
                            "is_pokemon": int(is_pokemon(c)),
                            "is_basic": int(is_pokemon(c) and not bool(getattr(c, "evolves_from", None))),
                            "is_evolution": int(bool(getattr(c, "evolves_from", None))),
                            "is_ex": int(bool(getattr(c, "is_ex", False))),
                            "is_goods": int(is_goods(c)),
                            "has_evo_target": int(has_evo_target), "on_field": int(on_field),
                            "hp": getattr(c, "hp", 0),
                        }
                        card_features.append(cf)

                    game_records.append({
                        "state": state_vec,
                        "card_features": card_features,
                        "discard_mask": None,  # 後で結果から埋める
                        "turn": state.turn_count,
                        "player": player_idx,
                        "seed": seed + 100000,
                        "_hand_before": [(i, getattr(c, "name", "")) for i, c in hand_without_hb],
                    })

            # 通常のトラッシュ選択を実行
            result = _orig(state, hand_index, **kw)

            # 選択後の手札を確認してdiscard_maskを推測
            if game_records and game_records[-1]["discard_mask"] is None:
                rec = game_records[-1]
                remaining_names = {getattr(c, "name", "") for c in p.hand}
                mask = []
                for _, name in rec["_hand_before"]:
                    mask.append(1 if name not in remaining_names else 0)
                # 正確さのため: 捨てた枚数が2ならOK
                if sum(mask) == 2:
                    rec["discard_mask"] = mask
                else:
                    # 名前重複で判定不能 → 推測（上位2枚）
                    mask = [0] * len(rec["_hand_before"])
                    # use heuristic: just mark first 2 as discarded (imperfect)
                    game_records.pop()  # 不正確なデータは除外
                rec.pop("_hand_before", None)

            return result

        gt.use_trainer_goods = _hooked
        g_turn.use_trainer_goods = _hooked

        try:
            state = setup_game(deck0=deck0, deck1=deck1, record_frame_fn=lambda s: None)
            run_game_auto(state)
            winner = state.winner
            for rec in game_records:
                if "_hand_before" in rec:
                    del rec["_hand_before"]
                if rec.get("discard_mask") is not None:
                    rec["won"] = int(winner == rec["player"])
                    all_records.append(rec)
        except Exception:
            pass

    gt.use_trainer_goods = _orig
    g_turn.use_trainer_goods = _orig
    return all_records


# ---------------------------------------------------------------------------
# 3. データ保存
# ---------------------------------------------------------------------------

def collect_data(args):
    """データ収集フェーズ。"""
    args.out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # 探索データ（ランダムトラッシュ）
    print("探索データ収集中...")
    explore_records = _collect_games(args.n, args.deck0, args.deck1, explore_rate=args.explore_rate)
    print(f"  探索: {len(explore_records)}件")

    # 通常データ（ルールベース選択）
    print("通常データ収集中...")
    normal_records = _collect_normal_games(args.n // 2, args.deck0, args.deck1)
    print(f"  通常: {len(normal_records)}件")

    all_records = explore_records + normal_records
    total = len(all_records)

    with open(args.out, "w") as f:
        for r in all_records:
            row = {
                "state": r["state"],
                "n_cards": len(r["card_features"]),
                "card_features": r["card_features"],
                "discard_mask": r["discard_mask"],
                "won": r["won"],
                "turn": r["turn"],
                "seed": r["seed"],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    elapsed = time.time() - t0
    wins = sum(r["won"] for r in all_records)
    print(f"収集完了: {total}件 (勝率: {wins/total*100:.1f}%)  ({elapsed:.1f}秒)")
    print(f"出力: {args.out}")
    return total


# ---------------------------------------------------------------------------
# 4. モデル定義（各カードのトラッシュスコア → 勝率予測）
# ---------------------------------------------------------------------------

CARD_FEAT_DIM = 11
STATE_DIM = 181


def _card_feat_to_vec(cf: dict, turn: int) -> list[float]:
    return [
        cf["is_energy"], cf["is_support"], cf["is_pokemon"],
        cf.get("is_basic", 0), cf["is_evolution"], cf["is_ex"],
        cf["is_goods"], cf.get("has_evo_target", 0), cf.get("on_field", 0),
        cf.get("hp", 0) / 300.0, turn / 20.0,
    ]


def train_model(args):
    """モデル訓練フェーズ。

    方式: 各カードの捨てやすさスコアを学習。
    - 入力: 盤面 + 各カードの特徴
    - 出力: 各カードのスコア
    - 教師: トラッシュ選択(mask) × ゲーム結果(won) → 良い捨て方のパターンを学習

    損失関数:
    - 勝ったゲームでの選択: discard_mask の方向にスコアを上げる
    - 負けたゲームでの選択: discard_mask の逆方向にスコアを上げる
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np

    data_path = args.out
    if not data_path.exists():
        print(f"データファイルが見つかりません: {data_path}")
        return

    # データ読み込み
    samples = []
    with open(data_path) as f:
        for line in f:
            row = json.loads(line)
            if row.get("discard_mask") is not None:
                samples.append(row)

    print(f"データ: {len(samples)}件")
    if len(samples) < 50:
        print("データが少なすぎます。")
        return

    # シャッフル & 分割
    random.shuffle(samples)
    split = max(10, int(len(samples) * 0.1))
    val_samples = samples[:split]
    train_samples = samples[split:]

    # テンソル化
    max_cards = max(s["n_cards"] for s in samples)

    def _to_tensors(slist):
        states, feats, masks, outcomes = [], [], [], []
        for s in slist:
            states.append(s["state"])
            n = s["n_cards"]
            turn = s.get("turn", 5)
            f = [_card_feat_to_vec(cf, turn) for cf in s["card_features"]]
            m = list(s["discard_mask"])
            while len(f) < max_cards:
                f.append([0.0] * CARD_FEAT_DIM)
                m.append(-1)  # padding
            feats.append(f)
            masks.append(m)
            outcomes.append(s["won"])

        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(feats, dtype=torch.float32),
            torch.tensor(masks, dtype=torch.float32),
            torch.tensor(outcomes, dtype=torch.float32),
        )

    s_t, f_t, m_t, o_t = _to_tensors(train_samples)
    s_v, f_v, m_v, o_v = _to_tensors(val_samples)

    # モデル
    class HBModel(nn.Module):
        def __init__(self, state_dim, card_feat_dim, hidden=64):
            super().__init__()
            self.state_enc = nn.Sequential(
                nn.Linear(state_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
            )
            self.card_scorer = nn.Sequential(
                nn.Linear(hidden + card_feat_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, 1),
            )

        def forward(self, state_vec, card_feats):
            bs, mc, _ = card_feats.shape
            se = self.state_enc(state_vec)
            se = se.unsqueeze(1).expand(-1, mc, -1)
            combined = torch.cat([se, card_feats], dim=-1)
            return self.card_scorer(combined).squeeze(-1)

    model = HBModel(state_dim=len(samples[0]["state"]), card_feat_dim=CARD_FEAT_DIM)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_val = float("inf")
    patience = 0

    for epoch in range(args.epochs):
        model.train()
        indices = list(range(len(train_samples)))
        random.shuffle(indices)
        total_loss = 0.0
        n_batches = 0

        for i in range(0, len(indices), args.batch_size):
            batch = indices[i:i + args.batch_size]
            s = s_t[batch]
            f = f_t[batch]
            m = m_t[batch]
            o = o_t[batch]

            scores = model(s, f)  # (batch, max_cards)

            # 有効マスク（パディング除外）
            valid = (m >= 0).float()

            # 目標: 勝ったゲームでは捨てたカードのスコアを高く、
            #       負けたゲームでは捨てたカードのスコアを低く
            # → target = mask * (2*won - 1) を正規化して使う
            # won=1 → target = mask (捨てたカード=1)
            # won=0 → target = -mask → 1-mask (残したカード=1)
            target = torch.where(
                o.unsqueeze(1) > 0.5,
                m.clamp(0, 1),           # 勝ち: 捨てたカード = 1
                (1 - m.clamp(0, 1)),     # 負け: 残したカード = 1（反転）
            )

            # sigmoid + BCE loss（有効部分のみ）
            loss = nn.functional.binary_cross_entropy_with_logits(
                scores, target, weight=valid, reduction="sum"
            ) / valid.sum().clamp(min=1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        # 検証
        model.eval()
        with torch.no_grad():
            v_scores = model(s_v, f_v)
            v_valid = (m_v >= 0).float()
            v_target = torch.where(
                o_v.unsqueeze(1) > 0.5,
                m_v.clamp(0, 1),
                (1 - m_v.clamp(0, 1)),
            )
            val_loss = nn.functional.binary_cross_entropy_with_logits(
                v_scores, v_target, weight=v_valid, reduction="sum"
            ) / v_valid.sum().clamp(min=1)

        if (epoch + 1) % 10 == 0:
            train_l = total_loss / max(n_batches, 1)
            print(f"  Epoch {epoch+1}: train={train_l:.4f}  val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            patience = 0
            model_dir = _REPO_ROOT / "models" / "hyper_ball"
            model_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), model_dir / "model.pt")
            with open(model_dir / "meta.json", "w") as mf:
                json.dump({
                    "state_dim": len(samples[0]["state"]),
                    "card_feat_dim": CARD_FEAT_DIM,
                    "hidden": 64,
                    "max_cards": max_cards,
                    "n_train": len(train_samples),
                    "n_val": len(val_samples),
                    "best_val_loss": float(best_val),
                    "version": 2,
                    "loss_type": "bce",
                }, mf, indent=2)
        else:
            patience += 1
            if patience >= 30:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    print(f"学習完了: best_val_loss={best_val:.4f}")
    print(f"モデル: models/hyper_ball/")


# ---------------------------------------------------------------------------
# 5. メイン
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ハイパーボールのトラッシュ選択学習 V2")
    sub = parser.add_subparsers(dest="cmd")

    p_col = sub.add_parser("collect")
    p_col.add_argument("--n", type=int, default=2000)
    p_col.add_argument("--deck0", type=int, default=5)
    p_col.add_argument("--deck1", type=int, default=6)
    p_col.add_argument("--explore-rate", type=float, default=0.3)
    p_col.add_argument("--out", type=Path, default=_REPO_ROOT / "datasets" / "hyper_ball_data.jsonl")

    p_tr = sub.add_parser("train")
    p_tr.add_argument("--out", type=Path, default=_REPO_ROOT / "datasets" / "hyper_ball_data.jsonl")
    p_tr.add_argument("--epochs", type=int, default=200)
    p_tr.add_argument("--batch-size", type=int, default=64)

    p_all = sub.add_parser("all")
    p_all.add_argument("--n", type=int, default=2000)
    p_all.add_argument("--deck0", type=int, default=5)
    p_all.add_argument("--deck1", type=int, default=6)
    p_all.add_argument("--explore-rate", type=float, default=0.3)
    p_all.add_argument("--out", type=Path, default=_REPO_ROOT / "datasets" / "hyper_ball_data.jsonl")
    p_all.add_argument("--epochs", type=int, default=200)
    p_all.add_argument("--batch-size", type=int, default=64)

    args = parser.parse_args()
    if args.cmd == "collect":
        collect_data(args)
    elif args.cmd == "train":
        train_model(args)
    elif args.cmd == "all":
        n = collect_data(args)
        if n > 0:
            train_model(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
