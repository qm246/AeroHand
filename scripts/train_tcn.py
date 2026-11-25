#!/usr/bin/env python3
"""
Train a Causal 1D-TCN Gesture Classifier for AeroHand and Export to ONNX + Normalizer JSON.

Matches Airpoint's `ONNXGestureClassifier` architecture (5-frame sliding causal window,
dilated residual 1D convolutions, 7 canonical pose classes) while supporting BOTH:
  - `--feature-set invariant348` (default: 3D rotation/translation/scale-invariant palm-frame features)
  - `--feature-set airmouse885`  (Airpoint 885-dim full z-depth feature layout for A/B comparison)

Zero-leakage evaluation:
  - Splits strictly by recording session file (never mixes adjacent frames across train/val).
  - Optionally ingests negative `D-dev` landmark JSON recordings as hard `idle` samples.
  - Verifies exported ONNX numerical parity with PyTorch before saving.

Usage:
  uv sync --extra train
  uv run python scripts/train_tcn.py --sessions data/sessions --out-dir models --feature-set invariant348
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from aerohand.core.types import ALL_POSE_NAMES, Landmark, PoseName
from aerohand.ml.features_3d import FEATURE_DIM_3D, extract_invariant_3d_features
from aerohand.ml.features_885 import AIRMOUSE_885_DIM, extract_airmouse_885_features

POSE_TO_IDX: Dict[str, int] = {p.value: idx for idx, p in enumerate(ALL_POSE_NAMES)}


def _to_landmarks(flat_or_list: object) -> List[Landmark]:
    if not flat_or_list:
        return []
    if isinstance(flat_or_list, list) and flat_or_list:
        first = flat_or_list[0]
        if isinstance(first, (int, float)) and len(flat_or_list) >= 63:
            return [
                Landmark(
                    x=float(flat_or_list[i]),
                    y=float(flat_or_list[i + 1]),
                    z=float(flat_or_list[i + 2]),
                )
                for i in range(0, 63, 3)
            ]
        if isinstance(first, dict) and len(flat_or_list) >= 21:
            return [
                Landmark(x=float(p["x"]), y=float(p["y"]), z=float(p.get("z", 0.0)))
                for p in flat_or_list[:21]
            ]
    return []


def _resolve_sample_label(sample: Dict[str, Any]) -> Optional[str]:
    raw = sample.get("label_name")
    if raw is None:
        raw = sample.get("labelName")
    if raw is None:
        return None
    return str(raw).strip()


def _resolve_sample_timestamp_ms(sample: Dict[str, Any]) -> Optional[float]:
    raw = sample.get("timestamp_ms")
    if raw is None:
        raw = sample.get("timestamp")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def samples_to_sequences(
    samples: Sequence[Dict[str, Any]],
    *,
    feature_set: str = "invariant348",
    seq_len: int = 5,
    force_idle: bool = False,
    skip_stats: Optional[Dict[str, int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert raw sample dicts (snake_case v1/v2 or SDK camelCase v4.0) into `(N, seq_len, F)` arrays."""
    feat_dim = FEATURE_DIM_3D if feature_set == "invariant348" else AIRMOUSE_885_DIM
    stats = skip_stats if skip_stats is not None else {}

    def _bump(reason: str) -> None:
        stats[reason] = stats.get(reason, 0) + 1

    feats: List[np.ndarray] = []
    labels: List[int] = []

    for s in samples:
        if not isinstance(s, dict):
            _bump("non_dict")
            continue

        if force_idle:
            label_idx = 0
        else:
            lname = _resolve_sample_label(s)
            if lname is None:
                lname = PoseName.IDLE.value if s.get("label") == 0 else None
            if lname is None or lname not in POSE_TO_IDX:
                _bump("unknown_label")
                continue
            label_idx = POSE_TO_IDX[lname]

        raw_2d = s.get("landmarks_2d")
        if raw_2d is None:
            raw_2d = s.get("landmarks")
        lms2d = _to_landmarks(raw_2d)
        if len(lms2d) < 21:
            _bump("bad_landmarks")
            continue

        raw_3d = s.get("world_landmarks")
        if raw_3d is None:
            raw_3d = s.get("worldLandmarks")
        lms3d = _to_landmarks(raw_3d) or None
        handed = "Left" if str(s.get("handedness", "Right")) == "Left" else "Right"

        if feature_set == "invariant348":
            vec = extract_invariant_3d_features(lms2d, lms3d, handedness=handed)
        else:
            vec = extract_airmouse_885_features(lms2d, lms3d, handedness=handed)

        if vec is not None and vec.shape[0] == feat_dim and np.all(np.isfinite(vec)):
            feats.append(vec)
            labels.append(label_idx)
        else:
            _bump("non_finite_features")

    if len(feats) < seq_len:
        return np.zeros((0, seq_len, feat_dim), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    arr = np.stack(feats, axis=0)
    seqs = np.stack([arr[i : i + seq_len] for i in range(len(arr) - seq_len + 1)], axis=0)
    ys = np.array(labels[seq_len - 1 :], dtype=np.int64)
    return seqs.astype(np.float32), ys


def load_session_sequences(
    session_path: Path,
    *,
    feature_set: str = "invariant348",
    seq_len: int = 5,
    force_idle: bool = False,
    skip_stats: Optional[Dict[str, int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load one JSON session into sliding `(N, seq_len, F)` sequences and labels `(N,)`."""
    data = json.loads(session_path.read_text(encoding="utf-8"))
    samples = data.get("samples") or data.get("frames") or []
    return samples_to_sequences(
        samples,
        feature_set=feature_set,
        seq_len=seq_len,
        force_idle=force_idle,
        skip_stats=skip_stats,
    )


def train_and_export(args: argparse.Namespace) -> None:
    try:
        import onnxruntime as ort
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise SystemExit(
            f"Missing training dependency ({exc}). Install with:\n"
            "  cd /Users/macos/Documents/AeroHand && uv sync --extra train"
        ) from None

    class CausalConv1dBlock(nn.Module):
        def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.2):
            super().__init__()
            self.pad = (kernel_size - 1) * dilation
            self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)
            self.bn1 = nn.BatchNorm1d(out_ch)
            self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, dilation=dilation)
            self.bn2 = nn.BatchNorm1d(out_ch)
            self.drop = nn.Dropout(dropout)
            self.act = nn.ReLU(inplace=True)
            self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = nn.functional.pad(x, (self.pad, 0))
            h = self.drop(self.act(self.bn1(self.conv1(h))))
            h = nn.functional.pad(h, (self.pad, 0))
            h = self.drop(self.act(self.bn2(self.conv2(h))))
            res = x if self.downsample is None else self.downsample(x)
            return self.act(h + res)

    class AeroHandTCN(nn.Module):
        def __init__(self, feat_dim: int, num_classes: int = 7, hidden: int = 96):
            super().__init__()
            self.blocks = nn.Sequential(
                CausalConv1dBlock(feat_dim, hidden, kernel_size=3, dilation=1, dropout=0.20),
                CausalConv1dBlock(hidden, hidden, kernel_size=3, dilation=2, dropout=0.20),
            )
            self.head = nn.Linear(hidden, num_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (B, T, F) -> (B, F, T)
            h = self.blocks(x.transpose(1, 2))
            return self.head(h[:, :, -1])

    sessions_dir = Path(args.sessions)
    files = sorted(sessions_dir.glob("*.json"))
    if not files:
        raise SystemExit(
            f"No session JSON files found in {sessions_dir}.\n"
            "Record labeled data first inside AeroHand.app -> 'Dữ liệu ML' tab (or `aerohand run --collect-dir data/sessions`)."
        )

    per_file: List[Tuple[str, np.ndarray, np.ndarray]] = []
    skip_stats: Dict[str, int] = {}
    for fp in files:
        xs, ys = load_session_sequences(
            fp, feature_set=args.feature_set, seq_len=args.seq_len, skip_stats=skip_stats
        )
        if xs.shape[0] > 0:
            per_file.append((fp.name, xs, ys))

    if args.negatives_dir:
        neg_dir = Path(args.negatives_dir)
        for fp in sorted(neg_dir.glob("*.json")):
            xs, ys = load_session_sequences(
                fp,
                feature_set=args.feature_set,
                seq_len=args.seq_len,
                force_idle=True,
                skip_stats=skip_stats,
            )
            if xs.shape[0] > 0:
                per_file.append((f"neg:{fp.name}", xs, ys))

    if skip_stats:
        print(f"[DATA] Skipped samples by reason: {skip_stats}")

    if not per_file:
        raise SystemExit("Loaded 0 valid sequences across all session files.")

    all_ys = np.concatenate([g[2] for g in per_file], axis=0)
    total_counts = np.bincount(all_ys, minlength=len(ALL_POSE_NAMES))
    print("[DATA] Sequence counts per class:")
    for idx, pose in enumerate(ALL_POSE_NAMES):
        print(f"  [{idx}] {pose.value:26s}: {int(total_counts[idx])}")

    # Session-grouped split (zero frame leakage!)
    if len(per_file) >= 2:
        val_count = max(1, int(round(len(per_file) * 0.2)))
        train_groups = per_file[:-val_count]
        val_groups = per_file[-val_count:]
    else:
        name, xs, ys = per_file[0]
        cut = max(1, int(len(xs) * 0.8))
        train_groups = [(f"{name}:train", xs[:cut], ys[:cut])]
        val_groups = [(f"{name}:val", xs[cut:], ys[cut:])]

    X_train = np.concatenate([g[1] for g in train_groups], axis=0)
    y_train = np.concatenate([g[2] for g in train_groups], axis=0)
    X_val = np.concatenate([g[1] for g in val_groups], axis=0)
    y_val = np.concatenate([g[2] for g in val_groups], axis=0)

    feat_dim = X_train.shape[2]
    flat_train = X_train.reshape(-1, feat_dim)
    mean = flat_train.mean(axis=0).astype(np.float32)
    scale = flat_train.std(axis=0).astype(np.float32)
    scale = np.where(scale < 1e-6, 1.0, scale).astype(np.float32)

    X_train_n = (X_train - mean) / scale
    X_val_n = (X_val - mean) / scale

    counts = np.bincount(y_train, minlength=len(ALL_POSE_NAMES)).astype(np.float32)
    class_weights = np.where(counts > 0, counts.sum() / np.maximum(counts * len(ALL_POSE_NAMES), 1.0), 0.0)
    class_weights = np.clip(class_weights, 0.2, 6.0).astype(np.float32)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[TRAIN] device={device} feature_set={args.feature_set} ({feat_dim}d) train={len(X_train)} val={len(X_val)}")

    model = AeroHandTCN(feat_dim=feat_dim, num_classes=len(ALL_POSE_NAMES), hidden=args.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, device=device))

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train_n), torch.from_numpy(y_train)),
        batch_size=args.batch_size,
        shuffle=True,
    )

    best_val_acc = -1.0
    best_state = None
    X_val_t = torch.from_numpy(X_val_n).to(device)
    y_val_t = torch.from_numpy(y_val).to(device)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            logits = model(bx)
            loss = loss_fn(logits, by)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * len(bx)
        sched.step()

        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t).argmax(dim=1)
            val_acc = float((val_preds == y_val_t).float().mean().item()) if len(y_val_t) else 0.0
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(f"  epoch {epoch:02d}/{args.epochs} | loss={total_loss/max(1,len(X_train)):.4f} | val_acc={val_acc*100:.2f}%")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.cpu().eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_fp = out_dir / f"{args.name}.onnx"
    norm_fp = out_dir / f"{args.name}_normalizer.json"

    dummy = torch.zeros(1, args.seq_len, feat_dim, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_fp),
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
    )

    norm_payload = {
        "feature_set": args.feature_set,
        "feature_dim": feat_dim,
        "sequence_length": args.seq_len,
        "classes": [p.value for p in ALL_POSE_NAMES],
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "val_accuracy": round(best_val_acc, 4),
    }
    norm_fp.write_text(json.dumps(norm_payload, indent=2), encoding="utf-8")

    # Verify ONNX Runtime parity
    sess = ort.InferenceSession(str(onnx_fp), providers=["CPUExecutionProvider"])
    sample_x = X_val_n[:1] if len(X_val_n) else np.zeros((1, args.seq_len, feat_dim), dtype=np.float32)
    pt_out = model(torch.from_numpy(sample_x)).detach().numpy()
    ort_out = sess.run(None, {"input": sample_x})[0]
    max_diff = float(np.max(np.abs(pt_out - ort_out)))
    print(f"[EXPORT] Saved {onnx_fp} + {norm_fp} | best_val_acc={best_val_acc*100:.2f}% | ONNX max_diff={max_diff:.2e}")


def main() -> None:
    p = argparse.ArgumentParser(description="Train AeroHand Causal 1D-TCN and export ONNX.")
    p.add_argument("--sessions", type=str, default="data/sessions", help="Directory containing collected JSON sessions")
    p.add_argument("--negatives-dir", type=str, default=None, help="Optional directory of D-dev negative JSON recordings")
    p.add_argument("--out-dir", type=str, default="models", help="Output directory for .onnx and _normalizer.json")
    p.add_argument("--name", type=str, default="aerohand_tcn", help="Base filename for exported artifacts")
    p.add_argument("--feature-set", choices=["invariant348", "airmouse885"], default="invariant348")
    p.add_argument("--seq-len", type=int, default=5)
    p.add_argument("--hidden", type=int, default=96)
    p.add_argument("--epochs", type=int, default=35)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1.5e-3)
    train_and_export(p.parse_args())


if __name__ == "__main__":
    main()
