#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare per-window prediction errors between B2 and B3 TFB result archives.

Works with TFB result .csv.tar.gz files created using:
    --save-true-pred True

Outputs:
  - window_comparison.csv
  - worst_b3_windows.csv
  - best_b3_windows.csv
  - step_comparison.csv
  - summary.txt

Usage:
python diagnose_b2_b3_errors.py \
  --b2 result/.../DAG.xxx.csv.tar.gz \
  --b3 result/.../DAG.xxx.csv.tar.gz \
  --out-dir diagnostics/b2_vs_b3_seed42_h96 \
  --top 30
"""

import argparse
import base64
import io
import pickle
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd


def read_result_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    name = p.name.lower()
    if name.endswith(".tar.gz") or name.endswith(".tgz"):
        with tarfile.open(p, "r:gz") as tf:
            members = [m for m in tf.getmembers() if m.isfile() and m.name.lower().endswith(".csv")]
            if not members:
                raise RuntimeError(f"No CSV found inside archive: {p}")
            # TFB result archive normally contains one CSV.
            member = members[0]
            f = tf.extractfile(member)
            if f is None:
                raise RuntimeError(f"Failed to extract {member.name}")
            raw = f.read()
            return pd.read_csv(io.BytesIO(raw))
    return pd.read_csv(p)


def decode_cell(cell):
    if pd.isna(cell):
        raise ValueError(
            "inference_data/actual_data is NaN. "
            "Rerun benchmark with --save-true-pred True."
        )
    raw = base64.b64decode(cell)
    return pickle.loads(raw)


def to_3d(x, name: str) -> np.ndarray:
    # Batch rolling evaluation stores ndarray [num_rollings, horizon, dim].
    if isinstance(x, pd.DataFrame):
        arr = x.to_numpy()[None, ...]
    elif isinstance(x, list):
        # Sample evaluation may store list[DataFrame].
        elems = []
        for item in x:
            if isinstance(item, pd.DataFrame):
                elems.append(item.to_numpy())
            else:
                elems.append(np.asarray(item))
        arr = np.stack(elems, axis=0)
    else:
        arr = np.asarray(x)

    if arr.ndim == 1:
        arr = arr[None, :, None]
    elif arr.ndim == 2:
        arr = arr[None, :, :]
    elif arr.ndim != 3:
        raise ValueError(f"{name} must be 3D [N,H,D] after decoding; got {arr.shape}")

    return arr.astype(np.float64, copy=False)


def load_true_pred(path: str):
    df = read_result_csv(path)
    required = {"actual_data", "inference_data"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"{path} missing columns: {sorted(missing)}")

    valid = df["actual_data"].notna() & df["inference_data"].notna()
    if not valid.any():
        raise ValueError(
            f"{path}: actual_data/inference_data are empty. "
            "Rerun with --save-true-pred True."
        )

    row = df.loc[valid].iloc[0]
    actual = to_3d(decode_cell(row["actual_data"]), "actual_data")
    pred = to_3d(decode_cell(row["inference_data"]), "inference_data")

    if actual.shape != pred.shape:
        raise ValueError(f"Shape mismatch in {path}: actual={actual.shape}, pred={pred.shape}")

    return actual, pred


def per_window_metrics(actual, pred):
    err = pred - actual
    mse = np.mean(err ** 2, axis=(1, 2))
    mae = np.mean(np.abs(err), axis=(1, 2))
    max_abs = np.max(np.abs(err), axis=(1, 2))
    # Horizon step of maximum absolute error, aggregated over channels.
    step_abs = np.mean(np.abs(err), axis=2)
    peak_step = np.argmax(step_abs, axis=1)
    return mse, mae, max_abs, peak_step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b2", required=True, help="B2 TFB result .csv.tar.gz or extracted CSV")
    ap.add_argument("--b3", required=True, help="B3 TFB result .csv.tar.gz or extracted CSV")
    ap.add_argument("--out-dir", default="diagnostics/b2_vs_b3")
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    y2, p2 = load_true_pred(args.b2)
    y3, p3 = load_true_pred(args.b3)

    if y2.shape != y3.shape:
        raise ValueError(f"B2/B3 target shape differs: {y2.shape} vs {y3.shape}")

    if not np.allclose(y2, y3, rtol=1e-7, atol=1e-9, equal_nan=True):
        max_diff = np.nanmax(np.abs(y2 - y3))
        raise ValueError(
            f"B2 and B3 actual targets are not identical (max diff={max_diff}). "
            "Make sure dataset/seed/horizon/rolling config are identical."
        )

    actual = y2
    n, h, d = actual.shape

    mse2, mae2, max2, peak2 = per_window_metrics(actual, p2)
    mse3, mae3, max3, peak3 = per_window_metrics(actual, p3)

    delta_mse = mse3 - mse2
    delta_mae = mae3 - mae2

    table = pd.DataFrame({
        "sample_id": np.arange(n, dtype=int),
        "mse_b2": mse2,
        "mse_b3": mse3,
        "delta_mse_b3_minus_b2": delta_mse,
        "mae_b2": mae2,
        "mae_b3": mae3,
        "delta_mae_b3_minus_b2": delta_mae,
        "max_abs_err_b2": max2,
        "max_abs_err_b3": max3,
        "peak_error_step_b2_0based": peak2,
        "peak_error_step_b3_0based": peak3,
        "b3_better_mse": delta_mse < 0,
        "b3_better_mae": delta_mae < 0,
    })

    # Rank: positive means B3 got worse.
    table["rank_worst_delta_mse"] = (
        table["delta_mse_b3_minus_b2"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    table.to_csv(out_dir / "window_comparison.csv", index=False)

    worst = table.sort_values(
        "delta_mse_b3_minus_b2", ascending=False
    ).head(args.top)
    best = table.sort_values(
        "delta_mse_b3_minus_b2", ascending=True
    ).head(args.top)

    worst.to_csv(out_dir / "worst_b3_windows.csv", index=False)
    best.to_csv(out_dir / "best_b3_windows.csv", index=False)

    # Per-horizon-step diagnostics.
    e2 = p2 - actual
    e3 = p3 - actual
    step_mse2 = np.mean(e2 ** 2, axis=(0, 2))
    step_mse3 = np.mean(e3 ** 2, axis=(0, 2))
    step_mae2 = np.mean(np.abs(e2), axis=(0, 2))
    step_mae3 = np.mean(np.abs(e3), axis=(0, 2))

    step_df = pd.DataFrame({
        "horizon_step_0based": np.arange(h, dtype=int),
        "mse_b2": step_mse2,
        "mse_b3": step_mse3,
        "delta_mse_b3_minus_b2": step_mse3 - step_mse2,
        "mae_b2": step_mae2,
        "mae_b3": step_mae3,
        "delta_mae_b3_minus_b2": step_mae3 - step_mae2,
    })
    step_df.to_csv(out_dir / "step_comparison.csv", index=False)

    # How concentrated is the MSE deterioration?
    positive = np.maximum(delta_mse, 0.0)
    total_pos = positive.sum()
    order = np.argsort(-positive)
    cumulative = np.cumsum(positive[order])

    def contribution(frac):
        k = max(1, int(np.ceil(n * frac)))
        if total_pos <= 0:
            return 0.0
        return float(cumulative[k - 1] / total_pos)

    # Error-tail counts.
    summary_lines = [
        f"shape [num_windows, horizon, dim] = {actual.shape}",
        f"B2 raw overall MSE = {np.mean((p2-actual)**2):.10f}",
        f"B3 raw overall MSE = {np.mean((p3-actual)**2):.10f}",
        f"B2 raw overall MAE = {np.mean(np.abs(p2-actual)):.10f}",
        f"B3 raw overall MAE = {np.mean(np.abs(p3-actual)):.10f}",
        "",
        f"B3 better MSE windows = {(delta_mse < 0).sum()} / {n} ({np.mean(delta_mse < 0):.2%})",
        f"B3 worse  MSE windows = {(delta_mse > 0).sum()} / {n} ({np.mean(delta_mse > 0):.2%})",
        f"B3 better MAE windows = {(delta_mae < 0).sum()} / {n} ({np.mean(delta_mae < 0):.2%})",
        f"B3 worse  MAE windows = {(delta_mae > 0).sum()} / {n} ({np.mean(delta_mae > 0):.2%})",
        "",
        f"median delta MSE (B3-B2) = {np.median(delta_mse):.10f}",
        f"90th pct delta MSE       = {np.quantile(delta_mse, 0.90):.10f}",
        f"95th pct delta MSE       = {np.quantile(delta_mse, 0.95):.10f}",
        f"99th pct delta MSE       = {np.quantile(delta_mse, 0.99):.10f}",
        "",
        f"Top 1% windows share of all positive MSE deterioration  = {contribution(0.01):.2%}",
        f"Top 5% windows share of all positive MSE deterioration  = {contribution(0.05):.2%}",
        f"Top 10% windows share of all positive MSE deterioration = {contribution(0.10):.2%}",
        "",
        "Interpretation:",
        "- If B3 MAE improves on many windows but a small fraction contributes most positive MSE delta,",
        "  then B3 is helping typical cases but creating a heavy error tail.",
        "- Open worst_b3_windows.csv first; those sample_id values are the exact windows to inspect.",
        "- step_comparison.csv tells whether the deterioration clusters at early/middle/late forecast steps.",
    ]

    summary = "\n".join(summary_lines)
    (out_dir / "summary.txt").write_text(summary, encoding="utf-8")

    print(summary)
    print()
    print(f"Saved: {out_dir / 'window_comparison.csv'}")
    print(f"Saved: {out_dir / 'worst_b3_windows.csv'}")
    print(f"Saved: {out_dir / 'best_b3_windows.csv'}")
    print(f"Saved: {out_dir / 'step_comparison.csv'}")
    print(f"Saved: {out_dir / 'summary.txt'}")


if __name__ == "__main__":
    main()
