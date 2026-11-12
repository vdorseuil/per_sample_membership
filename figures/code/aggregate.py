"""Aggregate per-model top-k recall and Spearman correlations for paper figures.

For each dataset and each model in ``MODEL_IDS`` we load:
    - per-sample loss / gradient norm (from ``metrics.pkl``, ``split == 'train'``)
    - leverage_newton, leverage_influence, optionally leverage_h_ii (pickled dicts)
    - the reference-model attack scores per sample (LiRA, RMIA, Shokri) from
      ``experiments/reference_models/<dataset>/asr_per_sample.csv``.

We compute, per (dataset, model, attack):
    1. Top-k recall arrays for surrogates {loss, grad, newton, influence, exact*, random}
       on a shared grid of k percentiles, where ``exact`` is only available for
       regression tasks.
    2. Spearman rank correlation between each surrogate and each attack target.

Outputs (under ``figures/aggregated/``):
    - topk_recall.npz : arrays indexed by (dataset, attack, surrogate, model, k)
    - spearman_long.csv : tidy table with one row per (dataset, attack, surrogate, model)
    - meta.json : k_pct grid and bookkeeping
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from style import DATASETS, MODEL_IDS, REGRESSION_DATASETS

HERE = Path(__file__).resolve().parent
FIG = HERE.parent
REPO = FIG.parent

TGT = REPO / "experiments" / "target_models"
REF = REPO / "experiments" / "reference_models"
OUT = FIG / "aggregated"
OUT.mkdir(parents=True, exist_ok=True)


# k-percentile grid for top-k recall: log-spaced from 0.01% to 30%.
K_PCT = np.geomspace(0.0001, 0.30, 36)

ATTACKS = ["lira", "rmia", "shokri"]

# Surrogates we always evaluate (random baseline is plotted analytically).
BASE_SURROGATES = ["loss", "grad", "newton", "influence"]
# Plus "exact" for regression tasks.


def latest_run(model_dir: Path) -> Path | None:
    base = model_dir / "leverage_score"
    if not base.exists():
        return None
    candidates = []
    for r in base.iterdir():
        if not r.is_dir():
            continue
        if not (r / "leverage_newton.pkl").exists():
            continue
        if not (r / "leverage_influence.pkl").exists():
            continue
        candidates.append(r)
    if not candidates:
        return None
    return sorted(candidates)[-1]


def topk_recall(score: np.ndarray, target: np.ndarray, k_grid: np.ndarray,
                descending: bool = True) -> np.ndarray:
    n = len(score)
    if descending:
        order_s = np.argsort(-score, kind="stable")
        order_t = np.argsort(-target, kind="stable")
    else:
        order_s = np.argsort(score, kind="stable")
        order_t = np.argsort(target, kind="stable")
    out = np.empty(len(k_grid))
    for j, k in enumerate(k_grid):
        m = max(1, int(round(k * n)))
        out[j] = np.intersect1d(order_s[:m], order_t[:m], assume_unique=True).size / m
    return out


def safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    rho, _ = spearmanr(a, b)
    if rho is None or np.isnan(rho):
        return float("nan")
    return float(rho)


def collect_dataset(dataset: str) -> dict | None:
    asr_path = REF / dataset / "asr_per_sample.csv"
    if not asr_path.exists():
        print(f"  [{dataset}] missing asr_per_sample.csv")
        return None
    asr_full = pd.read_csv(asr_path).set_index("idx")
    if not all(c in asr_full.columns for c in ATTACKS):
        print(f"  [{dataset}] asr columns missing: {asr_full.columns.tolist()}")
        return None

    base = TGT / dataset / "outputs"
    if not base.exists():
        print(f"  [{dataset}] no outputs dir")
        return None

    is_regression = dataset in REGRESSION_DATASETS
    surrogates = list(BASE_SURROGATES)
    if is_regression:
        surrogates = ["loss", "grad", "exact", "newton", "influence"]
    else:
        # Predictive entropy is only meaningful for classification (needs a
        # categorical log_proba vector).
        surrogates = ["loss", "grad", "entropy", "newton", "influence"]

    # Output array: [n_models, n_attacks, n_surrogates, n_k]
    n_models = len(MODEL_IDS)
    recall = np.full((n_models, len(ATTACKS), len(surrogates), len(K_PCT)), np.nan)
    spearman_rows = []
    used_models = []

    for mi, mname in enumerate(MODEL_IDS):
        model_dir = base / mname
        if not model_dir.is_dir():
            continue
        run = latest_run(model_dir)
        if run is None:
            continue
        try:
            with open(run / "leverage_newton.pkl", "rb") as fh:
                newton = pickle.load(fh)
            with open(run / "leverage_influence.pkl", "rb") as fh:
                influence = pickle.load(fh)
        except Exception as exc:
            print(f"  [{dataset}/{mname}] load fail: {exc}")
            continue
        h_ii = None
        if (run / "leverage_h_ii.pkl").exists():
            with open(run / "leverage_h_ii.pkl", "rb") as fh:
                h_ii = pickle.load(fh)

        try:
            mdf = pickle.load(open(model_dir / "metrics.pkl", "rb"))
        except Exception as exc:
            print(f"  [{dataset}/{mname}] metrics.pkl fail: {exc}")
            continue
        train = mdf[mdf["split"] == "train"].set_index("idx")
        if "loss" not in train.columns or "local_grad_norm" not in train.columns:
            continue
        # Predictive entropy: H = -sum_c p_c log p_c, computed from per-sample
        # log-probabilities. Only available for classification metrics.pkl.
        entropy_series = None
        if not is_regression and "log_proba" in train.columns:
            def _entropy(lp):
                lp = np.asarray(lp, dtype=float)
                if lp.ndim != 1 or lp.size < 2:
                    return float("nan")
                return float(-(np.exp(lp) * lp).sum())
            entropy_series = train["log_proba"].map(_entropy)

        common = sorted(set(newton.keys())
                        & set(influence.keys())
                        & set(asr_full.index)
                        & set(train.index))
        if len(common) < 50:
            continue

        idx = np.asarray(common)
        loss_arr = train.loc[idx, "loss"].to_numpy(dtype=float)
        grad_arr = train.loc[idx, "local_grad_norm"].to_numpy(dtype=float)
        newton_arr = np.asarray([newton[i] for i in idx], dtype=float)
        # Score samples are vulnerable when |newton| / |influence| are large
        # (sign depends on label encoding). Use magnitude.
        newton_arr = np.abs(newton_arr)
        infl_arr = np.abs(np.asarray([influence[i] for i in idx], dtype=float))
        h_arr = None
        if h_ii is not None and is_regression:
            h_arr = np.asarray([h_ii.get(i, np.nan) for i in idx], dtype=float)
            # Clip to keep formula well-defined
            h_arr = np.clip(h_arr, 1e-6, 1 - 1e-6)
            exact_arr = loss_arr * (2 * h_arr - h_arr ** 2) / (1 - h_arr) ** 2
        else:
            exact_arr = None

        attack_arr = {att: asr_full.loc[idx, att].to_numpy(dtype=float) for att in ATTACKS}

        score_map: dict[str, np.ndarray] = {
            "loss": loss_arr,
            "grad": grad_arr,
            "newton": newton_arr,
            "influence": infl_arr,
        }
        if exact_arr is not None:
            score_map["exact"] = exact_arr
        if entropy_series is not None:
            score_map["entropy"] = entropy_series.loc[idx].to_numpy(dtype=float)

        for ai, att in enumerate(ATTACKS):
            tgt = attack_arr[att]
            for si, sname in enumerate(surrogates):
                if sname not in score_map:
                    continue
                s = score_map[sname]
                # NaNs (from exact when h is ill-defined) -> skip those rows.
                m = np.isfinite(s) & np.isfinite(tgt)
                if m.sum() < 50:
                    continue
                recall[mi, ai, si] = topk_recall(s[m], tgt[m], K_PCT, descending=True)
                rho = safe_spearman(s[m], tgt[m])
                spearman_rows.append({
                    "dataset": dataset,
                    "model": mname,
                    "attack": att,
                    "surrogate": sname,
                    "rho": rho,
                    "n": int(m.sum()),
                })
        used_models.append(mname)

    if not used_models:
        print(f"  [{dataset}] no usable models")
        return None
    print(f"  [{dataset}] {len(used_models)} models: {used_models[0]}..{used_models[-1]}")

    return {
        "dataset": dataset,
        "surrogates": surrogates,
        "recall": recall,
        "models": MODEL_IDS,
        "used_models": used_models,
        "spearman_rows": spearman_rows,
    }


def main() -> None:
    bundles = {}
    spear_long = []
    for dataset, _label, _ in DATASETS:
        print(f"[aggregate] {dataset}")
        b = collect_dataset(dataset)
        if b is None:
            continue
        bundles[dataset] = b
        spear_long.extend(b["spearman_rows"])

    # Save per-dataset arrays into a single npz keyed by f"{dataset}/recall" etc.
    flat = {}
    meta = {"k_pct": K_PCT.tolist(), "attacks": ATTACKS, "datasets": {}}
    for ds, b in bundles.items():
        flat[f"{ds}/recall"] = b["recall"]
        meta["datasets"][ds] = {
            "surrogates": b["surrogates"],
            "models": b["models"],
            "used_models": b["used_models"],
        }
    np.savez(OUT / "topk_recall.npz", **flat)
    with open(OUT / "meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)

    pd.DataFrame(spear_long).to_csv(OUT / "spearman_long.csv", index=False)
    print(f"\n[aggregate] wrote {OUT / 'topk_recall.npz'}")
    print(f"[aggregate] wrote {OUT / 'meta.json'}")
    print(f"[aggregate] wrote {OUT / 'spearman_long.csv'}")


if __name__ == "__main__":
    main()
