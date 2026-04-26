"""ハイパーボールのトラッシュ選択学習モデル。

学習済みモデルを読み込み、手札の各カードの「捨てやすさスコア」を返す。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

_MODEL = None
_META = None
_REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_model():
    """学習済みモデルをロード（初回のみ）。"""
    global _MODEL, _META
    if _MODEL is not None:
        return True

    model_dir = _REPO_ROOT / "models" / "hyper_ball"
    model_path = model_dir / "model.pt"
    meta_path = model_dir / "meta.json"

    if not model_path.exists() or not meta_path.exists():
        return False

    try:
        import torch
        import torch.nn as nn

        with open(meta_path) as f:
            _META = json.load(f)

        class HyperBallDiscardModel(nn.Module):
            def __init__(self, state_dim, card_feat_dim, hidden):
                super().__init__()
                self.state_enc = nn.Sequential(
                    nn.Linear(state_dim, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, hidden),
                    nn.ReLU(),
                )
                self.card_scorer = nn.Sequential(
                    nn.Linear(hidden + card_feat_dim, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, 1),
                )

            def forward(self, state_vec, card_feats):
                batch_size, max_cards, _ = card_feats.shape
                state_encoded = self.state_enc(state_vec)
                state_expanded = state_encoded.unsqueeze(1).expand(-1, max_cards, -1)
                combined = torch.cat([state_expanded, card_feats], dim=-1)
                scores = self.card_scorer(combined).squeeze(-1)
                return scores

        model = HyperBallDiscardModel(
            state_dim=_META["state_dim"],
            card_feat_dim=_META["card_feat_dim"],
            hidden=_META["hidden"],
        )
        model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        model.eval()
        _MODEL = model
        return True
    except Exception as e:
        print(f"[hyper_ball_model] load error: {e}")
        return False


def score_discard_candidates(
    state,
    hand_index: int,
    hand_without_hb: list[tuple[int, object]],
) -> Optional[list[tuple[int, float]]]:
    """学習モデルで各カードの捨てやすさスコアを返す。

    Returns:
        [(hand_index, score), ...] スコアが高いほど捨てるべき。
        モデル未ロードの場合はNone。
    """
    if not _load_model():
        return None

    import torch
    from game.encoders import encode_state_v2
    from card import is_goods, is_pokemon, is_energy, is_support

    state_vec = encode_state_v2(state, state.current_player)
    p = state.active_player_state()
    turn = state.turn_count

    card_feats = []
    for i, c in hand_without_hb:
        evolves_from = getattr(c, "evolves_from", None)
        has_evo_target = False
        if evolves_from:
            for bp in [p.active] + p.bench:
                if bp and getattr(bp.card, "name", "") == evolves_from:
                    has_evo_target = True
                    break

        cname = getattr(c, "name", "")
        on_field = False
        for bp in [p.active] + p.bench:
            if bp and getattr(bp.card, "name", "") == cname:
                on_field = True
                break

        cf = [
            int(is_energy(c)),
            int(is_support(c)),
            int(is_pokemon(c)),
            int(is_pokemon(c) and not bool(getattr(c, "evolves_from", None))),
            int(bool(getattr(c, "evolves_from", None))),
            int(bool(getattr(c, "is_ex", False))),
            int(is_goods(c)),
            int(has_evo_target),
            int(on_field),
            getattr(c, "hp", 0) / 300.0,
            turn / 20.0,
        ]
        card_feats.append(cf)

    # パディング
    max_cards = _META.get("max_cards", len(card_feats))
    while len(card_feats) < max_cards:
        card_feats.append([0.0] * _META["card_feat_dim"])

    with torch.no_grad():
        s_t = torch.tensor([state_vec], dtype=torch.float32)
        f_t = torch.tensor([card_feats], dtype=torch.float32)
        scores = _MODEL(s_t, f_t)[0].tolist()

    result = []
    for j, (hand_idx, _) in enumerate(hand_without_hb):
        result.append((hand_idx, scores[j]))

    return result
