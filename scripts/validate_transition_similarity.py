#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Validate whether historical transitions become more transferable when
exogenous context (especially known-future exogenous variables) is included.

Core question
-------------
For an historical transition A -> B and a current transition C -> D:

Historical context:
    Q_AB = (A_endo, A_exo, B_exo)

Current context:
    Q_CD = (C_endo, C_exo, D_exo)

D_endo is NEVER used for retrieval.  It is used only afterwards, offline,
to measure the true transition similarity / forecast error.

We compare three retrieval contexts:
    M0: current endogenous state only
        A_endo  <->  C_endo

    M1: endogenous + current exogenous
        (A_endo, A_exo)  <->  (C_endo, C_exo)

    M2: endogenous + current exogenous + known-future exogenous
        (A_endo, A_exo, B_exo)  <->  (C_endo, C_exo, D_exo)

If M2 is consistently better than M1/M0 at finding historical transitions
whose future endogenous evolution resembles C -> D, that supports the
hypothesis that exogenous context helps identify transferable dynamics.

Transition representation
-------------------------
For a source block S and its next endogenous block Y:

    T = (Y - S_endo[-1]) / std(S_endo)

This represents the future endogenous trajectory relative to the current
endpoint and scale.  It avoids the artificial shared-middle-block problem
of comparing (B-A) with (C-B).

A simple analog forecast is also evaluated:
    D_hat = C_endo[-1] + std(C_endo) * weighted_average(T_historical)

No recursive rollout is used.

Example (ETTh1):
python scripts/validate_context_transition.py \
    --csv dataset/forecasting/ETTh1.csv \
    --target OT \
    --history 96 \
    --horizon 96 \
    --stride 12 \
    --topk 5 \
    --pairs_per_query 50 \
    --output_dir results_context_transition_etth1

Optional explicit exogenous columns:
    --exo_cols HUFL HULL MUFL MULL LUFL LULL
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

EPS = 1e-12


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def zscore_1d(x):
    x = np.asarray(x, dtype=float)
    s = np.std(x)
    if s < EPS:
        return x - np.mean(x)
    return (x - np.mean(x)) / s


def zscore_channels(x):
    """
    x: [time, channels]
    Each channel is standardized within the block.
    This focuses the first-stage validation on temporal pattern/context shape.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]

    out = np.zeros_like(x, dtype=float)
    for j in range(x.shape[1]):
        out[:, j] = zscore_1d(x[:, j])
    return out


def unit_vector(x):
    x = np.asarray(x, dtype=float).ravel()
    n = np.linalg.norm(x)
    if n < EPS:
        return np.zeros_like(x)
    return x / n


def cosine_from_unit(a, b):
    if len(a) != len(b):
        raise ValueError("Vector lengths differ.")
    if np.linalg.norm(a) < EPS or np.linalg.norm(b) < EPS:
        return np.nan
    return float(np.dot(a, b))


def safe_mean(values):
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    return float(np.mean(arr)) if len(arr) else np.nan


def spearman_corr(a, b):
    df = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(df) < 3:
        return np.nan
    ra = df["a"].rank(method="average")
    rb = df["b"].rank(method="average")
    if ra.std() < EPS or rb.std() < EPS:
        return np.nan
    return float(ra.corr(rb))


def pearson_corr(a, b):
    df = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(df) < 3:
        return np.nan
    if df["a"].std() < EPS or df["b"].std() < EPS:
        return np.nan
    return float(df["a"].corr(df["b"]))


def parse_exo_cols(values):
    if not values:
        return None

    cols = []
    for item in values:
        cols.extend([x.strip() for x in str(item).split(",") if x.strip()])
    return cols


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def load_multivariate_series(csv_path, target, exo_cols=None):
    """
    Supports:

    1) Wide CSV
       date, HUFL, HULL, MUFL, ..., OT

    2) Project long CSV
       date, data, cols

    Returns:
        dates        : optional array
        y_endo       : [T]
        x_exo        : [T, D]
        exo_names    : list[str]
    """
    path = Path(csv_path)

    if not path.exists():
        try:
            project_root = Path(__file__).resolve().parents[1]
            fallback = project_root / "dataset" / "forecasting" / path.name
        except NameError:
            fallback = Path("dataset") / "forecasting" / path.name

        if fallback.exists():
            path = fallback
        else:
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

    raw = pd.read_csv(path)
    target = str(target)
    exo_cols = parse_exo_cols(exo_cols)

    # ---------------------------------------------------------
    # Wide format
    # ---------------------------------------------------------
    if target in raw.columns and target not in {"data", "cols"}:
        dates = None
        if "date" in raw.columns:
            dates = pd.to_datetime(raw["date"], errors="coerce")
            if dates.isna().any():
                dates = raw["date"].astype(str).to_numpy()
            else:
                dates = dates.to_numpy()

        if exo_cols is None:
            candidate_cols = [
                c for c in raw.columns
                if c not in {target, "date"}
            ]
            exo_cols = [
                c for c in candidate_cols
                if pd.api.types.is_numeric_dtype(raw[c])
            ]

        missing = [c for c in exo_cols if c not in raw.columns]
        if missing:
            raise ValueError(f"Missing exogenous columns: {missing}")

        if not exo_cols:
            raise ValueError(
                "No exogenous variables found. Use --exo_cols explicitly."
            )

        y = pd.to_numeric(raw[target], errors="coerce").to_numpy(dtype=float)
        x = raw[exo_cols].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=float)

    # ---------------------------------------------------------
    # Long project format: date,data,cols
    # ---------------------------------------------------------
    elif {"data", "cols"}.issubset(raw.columns):
        raw = raw.copy()
        raw["cols"] = raw["cols"].astype(str)
        raw["data"] = pd.to_numeric(raw["data"], errors="coerce")

        if "date" in raw.columns:
            raw["_time"] = pd.to_datetime(raw["date"], errors="coerce")
            if raw["_time"].isna().any():
                raise ValueError("Invalid timestamps in long-format CSV.")

            wide = raw.pivot_table(
                index="_time",
                columns="cols",
                values="data",
                aggfunc="first"
            ).sort_index()

            dates = wide.index.to_numpy()

        else:
            raw["_seq"] = raw.groupby("cols").cumcount()
            wide = raw.pivot(
                index="_seq",
                columns="cols",
                values="data"
            ).sort_index()
            dates = None

        if target not in wide.columns:
            raise ValueError(
                f"Target {target!r} not found. "
                f"Available channels: {list(wide.columns)}"
            )

        if exo_cols is None:
            exo_cols = [c for c in wide.columns if c != target]

        missing = [c for c in exo_cols if c not in wide.columns]
        if missing:
            raise ValueError(f"Missing exogenous channels: {missing}")

        if not exo_cols:
            raise ValueError(
                "No exogenous variables found. Use --exo_cols explicitly."
            )

        selected = wide[[target] + exo_cols]

        if selected.isna().any().any():
            bad = selected.isna().sum()
            bad = bad[bad > 0].to_dict()
            raise ValueError(
                f"Missing values after aligning channels: {bad}"
            )

        y = selected[target].to_numpy(dtype=float)
        x = selected[exo_cols].to_numpy(dtype=float)

    else:
        raise ValueError(
            "Unsupported CSV format. Expected a wide table or "
            "long format with columns {'data','cols'}."
        )

    if not np.isfinite(y).all():
        raise ValueError("Target contains missing/non-finite values.")
    if not np.isfinite(x).all():
        raise ValueError("Exogenous variables contain missing/non-finite values.")

    if len(y) != len(x):
        raise ValueError("Endogenous/exogenous lengths are inconsistent.")

    return dates, y, x, list(exo_cols)


# ---------------------------------------------------------------------
# Transition records
# ---------------------------------------------------------------------

def build_transition_records(y, x_exo, history, horizon, stride):
    """
    Each record is one forecasting situation:

        source endogenous : S_endo = y[s : s+HIST]
        source exogenous  : S_exo  = x[s : s+HIST]
        future exogenous  : F_exo  = x[s+HIST : s+HIST+HOR]
        future endogenous : Y      = y[s+HIST : s+HIST+HOR]

    For the first clean validation, history == horizon is required so
    future trajectory shapes are directly comparable.
    """
    if history != horizon:
        raise ValueError(
            "For the first clean validation, use history == horizon. "
            f"Got history={history}, horizon={horizon}."
        )

    records = []
    total = history + horizon

    for idx, s in enumerate(range(0, len(y) - total + 1, stride)):
        src_y = y[s:s + history]
        src_x = x_exo[s:s + history]
        fut_x = x_exo[s + history:s + total]
        fut_y = y[s + history:s + total]

        scale = np.std(src_y)
        if scale < EPS:
            scale = 1.0

        # Context embeddings.
        endo_ctx = unit_vector(zscore_1d(src_y))
        past_exo_ctx = unit_vector(
            zscore_channels(src_x).ravel()
        )
        future_exo_ctx = unit_vector(
            zscore_channels(fut_x).ravel()
        )

        # True transition trajectory relative to source endpoint.
        transition_profile = (
            fut_y - src_y[-1]
        ) / scale

        transition_unit = unit_vector(transition_profile)

        records.append({
            "idx": idx,
            "start": s,
            "source_start": s,
            "source_end": s + history,
            "future_end": s + total,

            "src_y": src_y,
            "src_x": src_x,
            "fut_x": fut_x,
            "fut_y": fut_y,

            "source_scale": scale,

            "endo_ctx": endo_ctx,
            "past_exo_ctx": past_exo_ctx,
            "future_exo_ctx": future_exo_ctx,

            "transition_profile": transition_profile,
            "transition_unit": transition_unit,
        })

    return records


def component_similarities(hist, query):
    """
    Similarity of information that is available at prediction time.

    No query future endogenous values are used here.
    """
    s_endo = cosine_from_unit(
        hist["endo_ctx"], query["endo_ctx"]
    )
    s_past_exo = cosine_from_unit(
        hist["past_exo_ctx"], query["past_exo_ctx"]
    )
    s_future_exo = cosine_from_unit(
        hist["future_exo_ctx"], query["future_exo_ctx"]
    )

    # Equal component weighting prevents exogenous dimensionality from
    # automatically dominating simply because it has more channels.
    m0 = s_endo
    m1 = safe_mean([s_endo, s_past_exo])
    m2 = safe_mean([s_endo, s_past_exo, s_future_exo])

    return s_endo, s_past_exo, s_future_exo, m0, m1, m2


def true_transition_similarity(hist, query):
    """
    Offline-only truth:
    compares the historical future endogenous evolution with the
    query's actual future endogenous evolution.

    query future endogenous is used ONLY here for evaluation.
    """
    return cosine_from_unit(
        hist["transition_unit"],
        query["transition_unit"]
    )


# ---------------------------------------------------------------------
# Pair-level hypothesis test
# ---------------------------------------------------------------------

def build_pair_table(records, pairs_per_query=50, seed=42):
    """
    Sample genuinely historical transitions for each query.

    Historical transition h is eligible only if its future block has ended
    before the query source block begins:
        hist.future_end <= query.source_start

    For every sampled pair we compute:
        M0/M1/M2 context similarity
        true transition similarity

    Then we can test whether M2 similarity is more predictive of transition
    similarity than M0/M1.
    """
    rng = np.random.default_rng(seed)
    rows = []

    for q_idx, query in enumerate(records):
        eligible = [
            h for h in records[:q_idx]
            if h["future_end"] <= query["source_start"]
        ]

        if not eligible:
            continue

        if pairs_per_query > 0 and len(eligible) > pairs_per_query:
            chosen_idx = rng.choice(
                len(eligible),
                size=pairs_per_query,
                replace=False
            )
            candidates = [eligible[i] for i in chosen_idx]
        else:
            candidates = eligible

        for hist in candidates:
            (
                s_endo,
                s_past_exo,
                s_future_exo,
                m0,
                m1,
                m2
            ) = component_similarities(hist, query)

            t_sim = true_transition_similarity(hist, query)

            rows.append({
                "query_idx": query["idx"],
                "hist_idx": hist["idx"],
                "query_start": query["start"],
                "hist_start": hist["start"],

                "sim_endo": s_endo,
                "sim_past_exo": s_past_exo,
                "sim_future_exo": s_future_exo,

                "sim_M0_endo": m0,
                "sim_M1_endo_past_exo": m1,
                "sim_M2_endo_past_future_exo": m2,

                "true_transition_similarity": t_sim,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Retrieval + analog forecast
# ---------------------------------------------------------------------

def similarity_by_mode(hist, query, mode):
    _, _, _, m0, m1, m2 = component_similarities(hist, query)

    if mode == "M0":
        return m0
    if mode == "M1":
        return m1
    if mode == "M2":
        return m2

    raise ValueError(mode)


def analog_forecast(query, selected, similarities):
    """
    Transfer historical normalized future trajectories to the current source.

    Historical:
        T_h = (B_endo - A_endo[-1]) / std(A_endo)

    Current:
        D_hat = C_endo[-1] + std(C_endo) * weighted_average(T_h)

    No recursive prediction.
    """
    sims = np.asarray(similarities, dtype=float)

    # Convert cosine similarities into nonnegative weights.
    # Negative matches should not dominate the analog.
    weights = np.clip(sims, 0.0, None)

    if np.sum(weights) < EPS:
        weights = np.ones(len(selected), dtype=float)

    weights = weights / np.sum(weights)

    profiles = np.stack(
        [r["transition_profile"] for r in selected],
        axis=0
    )

    avg_profile = np.sum(
        profiles * weights[:, None],
        axis=0
    )

    pred = (
        query["src_y"][-1]
        + query["source_scale"] * avg_profile
    )

    return pred


def evaluate_retrieval(records, topk=5, seed=42):
    """
    Compare M0/M1/M2 retrieval.

    Retrieval NEVER uses query future endogenous D_endo.
    D_endo is used only after retrieval to score prediction / true transition
    similarity.
    """
    rng = np.random.default_rng(seed)
    rows = []

    for q_idx, query in enumerate(records):
        eligible = [
            h for h in records[:q_idx]
            if h["future_end"] <= query["source_start"]
        ]

        if len(eligible) < max(1, topk):
            continue

        persistence = np.repeat(
            query["src_y"][-1],
            len(query["fut_y"])
        )
        persistence_mse = mean_squared_error(
            query["fut_y"], persistence
        )
        persistence_mae = mean_absolute_error(
            query["fut_y"], persistence
        )

        row = {
            "query_idx": query["idx"],
            "query_start": query["start"],
            "persistence_mse": persistence_mse,
            "persistence_mae": persistence_mae,
        }

        # Random baseline
        random_ids = rng.choice(
            len(eligible),
            size=topk,
            replace=False
        )
        random_selected = [eligible[i] for i in random_ids]
        random_sims = np.ones(topk, dtype=float)
        random_pred = analog_forecast(
            query, random_selected, random_sims
        )
        row["random_mse"] = mean_squared_error(
            query["fut_y"], random_pred
        )
        row["random_mae"] = mean_absolute_error(
            query["fut_y"], random_pred
        )
        row["random_true_transition_similarity"] = safe_mean([
            true_transition_similarity(h, query)
            for h in random_selected
        ])

        for mode in ("M0", "M1", "M2"):
            scored = [
                (similarity_by_mode(h, query, mode), h)
                for h in eligible
            ]
            scored = [
                (s, h) for s, h in scored if np.isfinite(s)
            ]
            scored.sort(key=lambda x: x[0], reverse=True)

            selected = [h for _, h in scored[:topk]]
            sims = [s for s, _ in scored[:topk]]

            if not selected:
                continue

            pred = analog_forecast(
                query, selected, sims
            )

            mse = mean_squared_error(
                query["fut_y"], pred
            )
            mae = mean_absolute_error(
                query["fut_y"], pred
            )

            row[f"{mode}_mse"] = mse
            row[f"{mode}_mae"] = mae
            row[f"{mode}_mean_context_similarity"] = safe_mean(sims)
            row[f"{mode}_mean_true_transition_similarity"] = safe_mean([
                true_transition_similarity(h, query)
                for h in selected
            ])
            row[f"{mode}_beats_persistence"] = float(
                mse < persistence_mse
            )
            row[f"{mode}_skill_vs_persistence"] = (
                1.0 - mse / persistence_mse
                if persistence_mse > EPS else np.nan
            )

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------

def summarize_pairs(pair_df):
    if pair_df.empty:
        return {}

    y = pair_df["true_transition_similarity"]

    out = {
        "n_pair_samples": int(len(pair_df)),
    }

    for mode, col in [
        ("M0", "sim_M0_endo"),
        ("M1", "sim_M1_endo_past_exo"),
        ("M2", "sim_M2_endo_past_future_exo"),
    ]:
        out[f"{mode}_pearson_context_vs_transition"] = pearson_corr(
            pair_df[col], y
        )
        out[f"{mode}_spearman_context_vs_transition"] = spearman_corr(
            pair_df[col], y
        )

    out["M2_minus_M0_spearman_gain"] = (
        out["M2_spearman_context_vs_transition"]
        - out["M0_spearman_context_vs_transition"]
    )
    out["M2_minus_M1_spearman_gain"] = (
        out["M2_spearman_context_vs_transition"]
        - out["M1_spearman_context_vs_transition"]
    )

    return out


def summarize_retrieval(ret_df):
    if ret_df.empty:
        return {}

    out = {
        "n_retrieval_queries": int(len(ret_df)),
        "mean_persistence_mse": float(
            ret_df["persistence_mse"].mean()
        ),
        "median_persistence_mse": float(
            ret_df["persistence_mse"].median()
        ),
        "mean_random_mse": float(
            ret_df["random_mse"].mean()
        ),
        "median_random_mse": float(
            ret_df["random_mse"].median()
        ),
        "mean_random_true_transition_similarity": float(
            ret_df["random_true_transition_similarity"].mean()
        ),
    }

    for mode in ("M0", "M1", "M2"):
        if f"{mode}_mse" not in ret_df.columns:
            continue

        valid = ret_df.dropna(subset=[f"{mode}_mse"])

        out[f"{mode}_mean_mse"] = float(
            valid[f"{mode}_mse"].mean()
        )
        out[f"{mode}_median_mse"] = float(
            valid[f"{mode}_mse"].median()
        )
        out[f"{mode}_mean_true_transition_similarity"] = float(
            valid[f"{mode}_mean_true_transition_similarity"].mean()
        )
        out[f"{mode}_beats_persistence_ratio"] = float(
            valid[f"{mode}_beats_persistence"].mean()
        )
        out[f"{mode}_mean_skill_vs_persistence"] = float(
            valid[f"{mode}_skill_vs_persistence"].mean()
        )
        out[f"{mode}_median_skill_vs_persistence"] = float(
            valid[f"{mode}_skill_vs_persistence"].median()
        )

    if "M2_mean_mse" in out and "M0_mean_mse" in out:
        out["M2_minus_M0_mean_mse"] = (
            out["M2_mean_mse"] - out["M0_mean_mse"]
        )

    if (
        "M2_mean_true_transition_similarity" in out
        and "M0_mean_true_transition_similarity" in out
    ):
        out["M2_minus_M0_transition_similarity_gain"] = (
            out["M2_mean_true_transition_similarity"]
            - out["M0_mean_true_transition_similarity"]
        )

    return out


def json_safe(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, (float, np.floating)) and not np.isfinite(v):
            out[k] = None
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------

def make_plots(pair_df, ret_df, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pair-level scatter: M0
    if not pair_df.empty:
        for mode, col, title in [
            (
                "M0",
                "sim_M0_endo",
                "M0: Endogenous context vs transition similarity"
            ),
            (
                "M1",
                "sim_M1_endo_past_exo",
                "M1: + Current exogenous context"
            ),
            (
                "M2",
                "sim_M2_endo_past_future_exo",
                "M2: + Known-future exogenous context"
            ),
        ]:
            tmp = pair_df[[col, "true_transition_similarity"]].dropna()

            plt.figure(figsize=(7, 4.8))
            plt.scatter(
                tmp[col],
                tmp["true_transition_similarity"],
                alpha=0.25,
                s=12
            )
            plt.xlabel("Context similarity")
            plt.ylabel("True transition similarity")
            plt.title(title)
            plt.tight_layout()
            plt.savefig(
                out_dir / f"pair_{mode}_context_vs_transition.png",
                dpi=200
            )
            plt.close()

    # Retrieval transition similarity comparison
    if not ret_df.empty:
        cols = []
        labels = []

        for mode in ("M0", "M1", "M2"):
            col = f"{mode}_mean_true_transition_similarity"
            if col in ret_df.columns:
                vals = ret_df[col].dropna().values
                if len(vals):
                    cols.append(vals)
                    labels.append(mode)

        if cols:
            plt.figure(figsize=(7, 4.8))
            plt.boxplot(cols, labels=labels, showfliers=False)
            plt.ylabel("True transition similarity")
            plt.title(
                "Do exogenous-aware contexts retrieve more transferable history?"
            )
            plt.tight_layout()
            plt.savefig(
                out_dir / "retrieval_transition_similarity.png",
                dpi=200
            )
            plt.close()

        # Retrieval forecast MSE comparison
        mse_cols = []
        mse_labels = []

        for col, label in [
            ("M0_mse", "M0"),
            ("M1_mse", "M1"),
            ("M2_mse", "M2"),
            ("random_mse", "Random"),
            ("persistence_mse", "Persistence"),
        ]:
            if col in ret_df.columns:
                vals = ret_df[col].dropna().values
                if len(vals):
                    mse_cols.append(vals)
                    mse_labels.append(label)

        if mse_cols:
            plt.figure(figsize=(8, 4.8))
            plt.boxplot(
                mse_cols,
                labels=mse_labels,
                showfliers=False
            )
            plt.ylabel("MSE")
            plt.title("Analog forecast error")
            plt.tight_layout()
            plt.savefig(
                out_dir / "retrieval_forecast_mse.png",
                dpi=200
            )
            plt.close()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv", required=True)
    parser.add_argument("--target", default="OT")
    parser.add_argument(
        "--exo_cols",
        nargs="*",
        default=None,
        help=(
            "Exogenous columns/channels. "
            "Default: all channels except target/date. "
            "Accepts space-separated or comma-separated names."
        )
    )

    parser.add_argument("--history", type=int, default=96)
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--stride", type=int, default=12)

    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="Number of historical analog transitions used for retrieval."
    )
    parser.add_argument(
        "--pairs_per_query",
        type=int,
        default=50,
        help=(
            "Historical pairs sampled per query for the pair-level "
            "context->transition correlation test. "
            "Use 0 to use all eligible pairs."
        )
    )
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--output_dir",
        default="context_transition_validation"
    )

    args = parser.parse_args()

    if args.history <= 0 or args.horizon <= 0:
        raise ValueError("history/horizon must be positive.")
    if args.history != args.horizon:
        raise ValueError(
            "First validation should use history == horizon."
        )
    if args.stride <= 0:
        raise ValueError("stride must be positive.")
    if args.topk <= 0:
        raise ValueError("topk must be positive.")
    if args.pairs_per_query < 0:
        raise ValueError("pairs_per_query must be >= 0.")

    _, y, x_exo, exo_names = load_multivariate_series(
        args.csv,
        args.target,
        args.exo_cols
    )

    records = build_transition_records(
        y=y,
        x_exo=x_exo,
        history=args.history,
        horizon=args.horizon,
        stride=args.stride
    )

    if len(records) < 4:
        raise ValueError("Too few transition records.")

    pair_df = build_pair_table(
        records,
        pairs_per_query=args.pairs_per_query,
        seed=args.seed
    )

    retrieval_df = evaluate_retrieval(
        records,
        topk=args.topk,
        seed=args.seed
    )

    pair_summary = summarize_pairs(pair_df)
    retrieval_summary = summarize_retrieval(retrieval_df)

    summary = {
        "dataset": str(args.csv),
        "target": args.target,
        "exogenous_variables": exo_names,
        "history": args.history,
        "horizon": args.horizon,
        "stride": args.stride,
        "topk": args.topk,
        "pairs_per_query": args.pairs_per_query,
        **pair_summary,
        **retrieval_summary,
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pair_df.to_csv(
        out_dir / "pair_level_results.csv",
        index=False
    )
    retrieval_df.to_csv(
        out_dir / "retrieval_results.csv",
        index=False
    )

    with open(
        out_dir / "summary.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            json_safe(summary),
            f,
            ensure_ascii=False,
            indent=2,
            allow_nan=False
        )

    make_plots(pair_df, retrieval_df, out_dir)

    print("\n========== CONTEXT -> TRANSITION VALIDATION ==========")
    print(f"Target: {args.target}")
    print(f"Exogenous variables ({len(exo_names)}): {exo_names}")
    print()

    print("---- Pair-level hypothesis test ----")
    for key in [
        "n_pair_samples",
        "M0_pearson_context_vs_transition",
        "M0_spearman_context_vs_transition",
        "M1_pearson_context_vs_transition",
        "M1_spearman_context_vs_transition",
        "M2_pearson_context_vs_transition",
        "M2_spearman_context_vs_transition",
        "M2_minus_M0_spearman_gain",
        "M2_minus_M1_spearman_gain",
    ]:
        if key in summary:
            print(f"{key}: {summary[key]}")

    print("\n---- Retrieval / analog forecast ----")
    for key in [
        "n_retrieval_queries",
        "mean_persistence_mse",
        "mean_random_mse",

        "M0_mean_true_transition_similarity",
        "M1_mean_true_transition_similarity",
        "M2_mean_true_transition_similarity",
        "M2_minus_M0_transition_similarity_gain",

        "M0_mean_mse",
        "M1_mean_mse",
        "M2_mean_mse",
        "M2_minus_M0_mean_mse",

        "M0_beats_persistence_ratio",
        "M1_beats_persistence_ratio",
        "M2_beats_persistence_ratio",

        "M0_median_skill_vs_persistence",
        "M1_median_skill_vs_persistence",
        "M2_median_skill_vs_persistence",
    ]:
        if key in summary:
            print(f"{key}: {summary[key]}")

    print("\nHow to judge the hypothesis:")
    print("1) If M2 context->transition correlation > M1 > M0,")
    print("   exogenous context helps identify transferable transitions.")
    print("2) If M2 retrieved true-transition similarity > M1/M0,")
    print("   future-known exogenous information improves historical matching.")
    print("3) If M2 analog forecast MSE < M1/M0 and beats persistence more often,")
    print("   the improved matching also has predictive value.")
    print("4) If pair-level correlation improves but forecast MSE does not,")
    print("   the structural hypothesis may hold while the simple analog-transfer")
    print("   mechanism is still insufficient.")


if __name__ == "__main__":
    main()
