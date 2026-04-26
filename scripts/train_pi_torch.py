"""
JSONL dataset から policy π(a | s) を PyTorch で学習する。

入力（1 行 JSON）:
  {"state":[...], "action_id":2, "win":1, "turn": 12, ...}

学習:
  - 既定は勝ち負け両方の行を使用し、方策勾配（REINFORCE 風）で更新する。
  - advantage = +1（勝ち）/ -1（負け）、clamp(-1,1)、dtype float32
  - loss = -(log π(a|s) × advantage × turn_weight × sample_weight).sum() / sum(weight)（ sample_weight は top2_gap 等）
  - --win-only で従来どおり勝ち行だけ＋交差エントロピー模倣に切り替え可能
  - --entropy-schedule linear | exp でエントロピー係数をエポック進行に応じて減衰（--entropy-coeff が基準値）
  - --pg-sample-weight で top2_gap 由来の重み（難しい局面ほど学習強く）
  - --entropy-coeff-sync-schedule-progress でバッチ平均の retreat_policy_schedule_progress に合わせて entropy 係数を掛ける

出力:
  - モデル重み: models/pi_energy_attach_mlp.pt
  - メタ情報:   models/pi_energy_attach_mlp.meta.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _default_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _turn_weight(turn: int, *, mode: str, max_turn: int = 200) -> float:
    if mode == "none":
        return 1.0
    t = max(0.0, min(1.0, float(turn) / float(max_turn)))
    if mode == "linear":
        v = t
    else:
        v = math.sqrt(t)
    return 0.2 + 0.8 * v


class PiNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 64):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)  # logits [B, action_dim]


def _gap_sample_weight(r: dict, *, mode: str, gap_cap: float) -> float:
    if mode == "none":
        return 1.0
    gap = r.get("top2_gap")
    if gap is None:
        gap = r.get("retreat_policy_top2_gap")
    if gap is None:
        return 1.0
    try:
        gf = float(gap)
    except (TypeError, ValueError):
        return 1.0
    if math.isinf(gf) or math.isnan(gf):
        gf = gap_cap
    gf = max(0.0, min(gf, gap_cap))
    if mode == "exp_neg_top2_gap":
        return math.exp(-gf)
    if mode == "inv_one_plus_gap":
        return 1.0 / (1.0 + gf)
    return 1.0


def _entropy_low_factor_for_row(r: dict, *, threshold: float, factor: float) -> float:
    if threshold <= 0.0:
        return 1.0
    ent = r.get("retreat_policy_entropy")
    if ent is None:
        return 1.0
    try:
        e = float(ent)
    except (TypeError, ValueError):
        return 1.0
    if e < threshold:
        return float(factor)
    return 1.0


class JsonlPiDataset(Dataset):
    def __init__(
        self,
        path: Path,
        *,
        action_dim: int,
        win_only: bool = False,
        loss_mode: str = "pg_adv",
        pg_sample_weight_mode: str = "none",
        top2_gap_cap: float = 80.0,
        pg_entropy_low_threshold: float = 0.0,
        pg_entropy_low_factor: float = 0.5,
        advantage_source: str = "win",
        prize_delta_scale: float = 3.0,
        tail_rows: int = 0,
        ap_scale: float = 0.0,
    ):
        self.path = path
        self.action_dim = action_dim
        self.loss_mode = loss_mode
        self.pg_sample_weight_mode = pg_sample_weight_mode
        self.top2_gap_cap = float(top2_gap_cap)
        self.pg_entropy_low_threshold = float(pg_entropy_low_threshold)
        self.pg_entropy_low_factor = float(pg_entropy_low_factor)
        self.advantage_source = advantage_source
        self.prize_delta_scale = max(1e-6, float(prize_delta_scale))
        self.ap_scale = float(ap_scale)
        self.rows: list[dict] = []
        raw_rows: list[dict] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                raw_rows.append(json.loads(line))
        # 最新データ重視: 末尾 tail_rows 件だけ残す
        if tail_rows > 0 and len(raw_rows) > tail_rows:
            raw_rows = raw_rows[-tail_rows:]
        for r in raw_rows:
            if win_only and int(r.get("win", 0)) != 1:
                continue
            # prize_delta モードではゼロ advantage をスキップ（崩壊防止）
            # ap_scale > 0 なら action_potential も考慮してスキップ判定
            if advantage_source == "prize_delta":
                pd = float(r.get("prize_delta", 0))
                ap = float(r.get("action_potential", 0)) if ap_scale != 0.0 else 0.0
                if (pd + ap_scale * ap) == 0.0:
                    continue
            self.rows.append(r)
        if not self.rows:
            raise ValueError(
                f"dataset が空です: {path}\n"
                "win_only=True の場合は win==1 の行だけ。False の場合は JSONL 全体を確認してください。"
                "export_policy_dataset.py で十分な行数が出ているか確認してください。"
            )
        self.state_dim = len(self.rows[0]["state"])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        state = np.asarray(r["state"], dtype=np.float32)
        action_id = int(r["action_id"])
        turn = int(r.get("turn", 0))
        if not (0 <= action_id < self.action_dim):
            action_id = 0
        win = int(r.get("win", r.get("success", 0)))
        # advantage の計算
        if self.advantage_source == "prize_delta":
            # prize_delta + ap_scale * action_potential を合成して [-1, 1] に正規化
            pd = float(r.get("prize_delta", 0))
            ap = float(r.get("action_potential", 0)) if self.ap_scale != 0.0 else 0.0
            advantage = max(-1.0, min(1.0, (pd + self.ap_scale * ap) / self.prize_delta_scale))
        elif self.loss_mode == "pg_success":
            advantage = float(win)  # 0.0 or 1.0
        else:
            advantage = 1.0 if win == 1 else -1.0
        sw = _gap_sample_weight(r, mode=self.pg_sample_weight_mode, gap_cap=self.top2_gap_cap)
        sw *= _entropy_low_factor_for_row(
            r,
            threshold=self.pg_entropy_low_threshold,
            factor=self.pg_entropy_low_factor,
        )
        sp = r.get("retreat_policy_schedule_progress")
        sp_f = float(sp) if sp is not None else float("nan")
        return state, np.int64(action_id), turn, np.float32(advantage), np.float32(sw), np.float32(sp_f)


@dataclass
class TrainMeta:
    data: str
    state_dim: int
    action_dim: int
    hidden: int
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    seed: int
    device: str
    turn_weight: str
    val_split: float
    win_only: bool
    loss: str
    advantage_clamp: float
    entropy_coeff: float
    entropy_schedule: str
    entropy_decay_k: float
    pg_sample_weight_mode: str
    top2_gap_cap: float
    pg_entropy_low_threshold: float
    pg_entropy_low_factor: float
    entropy_coeff_sync_schedule_progress: bool
    advantage_source: str
    prize_delta_scale: float


def entropy_coeff_for_epoch(
    base: float,
    ep: int,
    epochs: int,
    schedule: str,
    k: float,
) -> float:
    """エポック進行に応じたエントロピー係数（PG の探索正則）。base<=0 または constant は base のまま。"""
    if base <= 0.0 or str(schedule).lower() == "constant":
        return base
    if epochs <= 1:
        p = 1.0
    else:
        p = float(ep - 1) / float(epochs - 1)
    sched = str(schedule).lower()
    if sched == "linear":
        return base * (1.0 - p)
    if sched == "exp":
        return base * math.exp(-float(k) * p)
    return base


def main() -> None:
    parser = argparse.ArgumentParser(description="π(a|s) を PyTorch MLP で学習（既定: 方策勾配 + advantage ±1）")
    parser.add_argument("--data", type=Path, required=True, help="入力 JSONL（例: datasets/policy_energy_attach.jsonl）")
    parser.add_argument("--out", type=Path, default=_REPO_ROOT / "models" / "pi_energy_attach_mlp.pt", help="出力 .pt")
    parser.add_argument("--meta-out", type=Path, default=_REPO_ROOT / "models" / "pi_energy_attach_mlp.meta.json", help="出力 meta JSON")
    parser.add_argument("--encoder", type=str, default="opening", choices=["basic", "opening"], help="状態エンコーダー名（モデルに埋め込まれて推論時に使われる）")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--action-dim", type=int, default=6, help="energy_attach は 6=active+bench5")
    parser.add_argument("--turn-weight", type=str, default="none", choices=["none", "linear", "sqrt"])
    parser.add_argument("--val-split", type=float, default=0.1, help="検証用に取り分ける割合")
    parser.add_argument("--no-mps", action="store_true", help="MPS を使わない（cpu で実行）")
    parser.add_argument(
        "--win-only",
        action="store_true",
        help="win==1 の行だけ使い、交差エントロピーで行動模倣（従来モード）",
    )
    parser.add_argument(
        "--loss-mode",
        type=str,
        default="pg_adv",
        choices=["pg_adv", "pg_success"],
        help=(
            "pg_adv（既定）: advantage=±1 で勝ちを強化・負けを抑制 / "
            "pg_success: advantage=0/1 で勝ちのみ強化・負けはゼロ勾配（ε探索データ向け）"
        ),
    )
    parser.add_argument(
        "--advantage-clamp",
        type=float,
        default=1.0,
        help="方策勾配時の advantage を [-v, v] に clamp（既定 1.0 で ±1）",
    )
    parser.add_argument(
        "--entropy-coeff",
        type=float,
        default=0.0,
        help="方策勾配時: loss -= coeff * mean(entropy(π)) の基準値（--entropy-schedule でエポック減衰可）",
    )
    parser.add_argument(
        "--entropy-schedule",
        type=str,
        default="constant",
        choices=["constant", "linear", "exp"],
        help="エントロピー係数のエポック減衰: constant | linear(base*(1-p)) | exp(base*exp(-k*p))",
    )
    parser.add_argument(
        "--entropy-decay-k",
        type=float,
        default=3.0,
        help="--entropy-schedule exp 時の k（大きいほど早く減衰）",
    )
    parser.add_argument(
        "--pg-sample-weight",
        type=str,
        default="none",
        choices=["none", "exp_neg_top2_gap", "inv_one_plus_gap"],
        help="方策勾配時: top2_gap に基づくサンプル重み（JSONL に top2_gap がある行で有効）",
    )
    parser.add_argument(
        "--top2-gap-cap",
        type=float,
        default=80.0,
        help="top2_gap の上限（inf 近似・ exp 安定化）",
    )
    parser.add_argument(
        "--pg-entropy-low-threshold",
        type=float,
        default=0.0,
        metavar="NAT",
        help="ログの retreat_policy_entropy がこれ未満なら重みに --pg-entropy-low-factor を掛ける（0 で無効）",
    )
    parser.add_argument(
        "--pg-entropy-low-factor",
        type=float,
        default=0.5,
        help="低エントロピー行の重み倍率（確信しすぎ抑制）",
    )
    parser.add_argument(
        "--entropy-coeff-sync-schedule-progress",
        action="store_true",
        help="バッチ内の retreat_policy_schedule_progress 平均 p に対し entropy 係数を ×(1-p)（ゲーム側カリキュラムと同期）",
    )
    parser.add_argument(
        "--advantage-source",
        type=str,
        default="win",
        choices=["win", "prize_delta"],
        help="方策勾配の advantage 源泉: win（±1）/ prize_delta（サイドレース報酬、--prize-delta-scale で正規化）",
    )
    parser.add_argument(
        "--prize-delta-scale",
        type=float,
        default=3.0,
        help="prize_delta を advantage に変換するスケール（prize_delta / scale で [-1,1] に clamp）",
    )
    parser.add_argument(
        "--ap-scale",
        type=float,
        default=0.0,
        help="action_potential への重み（reward = prize_delta + ap_scale * action_potential、既定 0.0=無効）",
    )
    parser.add_argument(
        "--tail-rows",
        type=int,
        default=0,
        help="末尾 N 行だけ学習（最新データ優先。0 で全件使用）",
    )
    args = parser.parse_args()

    data_path = args.data if args.data.is_absolute() else (_REPO_ROOT / args.data)
    if not data_path.is_file():
        print(f"エラー: データセットが見つかりません: {data_path}", file=sys.stderr)
        print("先に以下で JSONL を作成してください:", file=sys.stderr)
        print("  python scripts/export_policy_dataset.py --battles-dir battles/train_belt --out datasets/policy_energy_attach.jsonl", file=sys.stderr)
        sys.exit(1)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cpu") if args.no_mps else _default_device()
    win_only = bool(args.win_only)
    loss_mode = "cross_entropy" if win_only else str(args.loss_mode)
    adv_clamp = max(1e-6, float(args.advantage_clamp))
    entropy_coeff_base = float(args.entropy_coeff)
    entropy_sched = str(args.entropy_schedule)
    entropy_decay_k = float(args.entropy_decay_k)

    try:
        ds = JsonlPiDataset(
            data_path,
            action_dim=int(args.action_dim),
            win_only=win_only,
            loss_mode=loss_mode,
            pg_sample_weight_mode=str(args.pg_sample_weight),
            top2_gap_cap=float(args.top2_gap_cap),
            pg_entropy_low_threshold=float(args.pg_entropy_low_threshold),
            pg_entropy_low_factor=float(args.pg_entropy_low_factor),
            advantage_source=str(args.advantage_source),
            prize_delta_scale=float(args.prize_delta_scale),
            tail_rows=int(args.tail_rows),
            ap_scale=float(args.ap_scale),
        )
    except ValueError as e:
        if "dataset が空です" in str(e):
            print(str(e), file=sys.stderr)
            sys.exit(1)
        raise

    n = len(ds)
    idxs = np.arange(n)
    np.random.shuffle(idxs)
    n_val = int(n * float(args.val_split))
    val_idxs = set(map(int, idxs[:n_val].tolist()))

    class _Subset(Dataset):
        def __init__(self, base: Dataset, keep: set[int]):
            self.base = base
            self.keep = sorted(list(keep))

        def __len__(self):
            return len(self.keep)

        def __getitem__(self, i):
            return self.base[self.keep[i]]

    train_keep = set(range(n)) - val_idxs
    train_ds = _Subset(ds, train_keep)
    val_ds = _Subset(ds, val_idxs) if n_val > 0 else None

    def _collate(batch):
        states, action_ids, turns, advantages, s_weights, sched_p = zip(*batch)
        s = torch.from_numpy(np.stack(states)).to(torch.float32)
        a = torch.from_numpy(np.asarray(action_ids, dtype=np.int64))
        tw = np.asarray([_turn_weight(int(t), mode=args.turn_weight) for t in turns], dtype=np.float32)
        sw = torch.from_numpy(np.asarray(s_weights, dtype=np.float32))
        w = torch.from_numpy(tw).to(torch.float32) * sw.to(torch.float32)
        adv = torch.tensor(advantages, dtype=torch.float32)
        sp = torch.tensor(sched_p, dtype=torch.float32)
        return s, a, w, adv, sp

    train_loader = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=True, num_workers=0, collate_fn=_collate)
    val_loader = DataLoader(val_ds, batch_size=int(args.batch_size), shuffle=False, num_workers=0, collate_fn=_collate) if val_ds else None

    model = PiNet(state_dim=ds.state_dim, action_dim=int(args.action_dim), hidden=int(args.hidden)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    ce = nn.CrossEntropyLoss(reduction="none")

    def _run(loader: DataLoader, train: bool, *, ent_coeff: float) -> tuple[float, float]:
        model.train() if train else model.eval()
        loss_sum = 0.0
        correct_sum = 0.0
        count = 0
        entropy_sync = bool(args.entropy_coeff_sync_schedule_progress)
        for s, a, w, advantages, sp in loader:
            s = s.to(device)
            a = a.to(device)
            w = w.to(device)
            sp = sp.to(device)
            advantages = advantages.to(device).clamp(-adv_clamp, adv_clamp)
            logits = model(s)
            w_sum = w.sum().clamp_min(1e-8)
            if win_only:
                loss_vec = ce(logits, a)
                loss = (loss_vec * w).sum() / w_sum
            else:
                log_p = F.log_softmax(logits, dim=-1)
                selected_log_prob = log_p.gather(1, a.unsqueeze(1)).squeeze(1)
                term = -(selected_log_prob * advantages * w)
                loss = term.sum() / w_sum
                ec = float(ent_coeff)
                if ec > 0.0 and entropy_sync:
                    valid = torch.isfinite(sp) & (sp >= 0.0) & (sp <= 1.0)
                    if valid.any():
                        mean_p = sp[valid].mean()
                        ec = ec * (1.0 - mean_p.clamp(0.0, 1.0))
                if ec > 0.0:
                    probs = log_p.exp()
                    pol_entropy = -(probs * log_p).sum(dim=-1)
                    loss = loss - ec * (pol_entropy * w).sum() / w_sum
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            with torch.no_grad():
                pred = torch.argmax(logits, dim=-1)
                correct_sum += (pred == a).to(torch.float32).sum().item()
                loss_sum += loss.item() * a.shape[0]
                count += a.shape[0]
        return (loss_sum / max(1, count), correct_sum / max(1, count))

    print(
        f"device={device}  n_train={len(train_ds)}  n_val={len(val_ds) if val_ds else 0}  "
        f"state_dim={ds.state_dim}  action_dim={args.action_dim}  win_only={win_only}  loss={loss_mode}  "
        f"adv_clamp=±{adv_clamp}  entropy_coeff_base={entropy_coeff_base}  entropy_schedule={entropy_sched}  "
        f"pg_sample_weight={args.pg_sample_weight}  entropy_sync_progress={bool(args.entropy_coeff_sync_schedule_progress)}"
    )
    for ep in range(1, int(args.epochs) + 1):
        ec = entropy_coeff_for_epoch(
            entropy_coeff_base,
            ep,
            int(args.epochs),
            entropy_sched,
            entropy_decay_k,
        )
        tr_loss, tr_acc = _run(train_loader, train=True, ent_coeff=ec)
        if val_loader:
            va_loss, va_acc = _run(val_loader, train=False, ent_coeff=ec)
            print(
                f"epoch {ep}/{args.epochs}  entropy_coeff={ec:.6f}  "
                f"train loss={tr_loss:.6f} acc={tr_acc:.3f}  val loss={va_loss:.6f} acc={va_acc:.3f}"
            )
        else:
            print(
                f"epoch {ep}/{args.epochs}  entropy_coeff={ec:.6f}  train loss={tr_loss:.6f} acc={tr_acc:.3f}"
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "state_dim": ds.state_dim,
            "action_dim": int(args.action_dim),
            "hidden": int(args.hidden),
            "encoder_name": str(args.encoder),
        },
        args.out,
    )
    print(f"saved: {args.out}")

    meta = TrainMeta(
        data=str(data_path),
        state_dim=ds.state_dim,
        action_dim=int(args.action_dim),
        hidden=int(args.hidden),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        seed=int(args.seed),
        device=str(device),
        turn_weight=str(args.turn_weight),
        val_split=float(args.val_split),
        win_only=bool(win_only),
        loss="cross_entropy" if win_only else "policy_gradient",
        advantage_clamp=float(adv_clamp),
        entropy_coeff=float(entropy_coeff_base),
        entropy_schedule=str(entropy_sched),
        entropy_decay_k=float(entropy_decay_k),
        pg_sample_weight_mode=str(args.pg_sample_weight),
        top2_gap_cap=float(args.top2_gap_cap),
        pg_entropy_low_threshold=float(args.pg_entropy_low_threshold),
        pg_entropy_low_factor=float(args.pg_entropy_low_factor),
        entropy_coeff_sync_schedule_progress=bool(args.entropy_coeff_sync_schedule_progress),
        advantage_source=str(args.advantage_source),
        prize_delta_scale=float(args.prize_delta_scale),
    )
    args.meta_out.parent.mkdir(parents=True, exist_ok=True)
    args.meta_out.write_text(json.dumps(asdict(meta), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {args.meta_out}")


if __name__ == "__main__":
    main()

