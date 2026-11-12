r"""Print a wide-format LaTeX summary table for LiRA ASR (Recall @ 1% only).

Separates the Dataset and Model into two distinct columns.
Uses \multirow to group datasets that share multiple model architectures.
Best results are in bold, second-best are underlined.
"""
from __future__ import annotations

import json
import collections
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FIG = HERE.parent

AGG = FIG / "aggregated"

DATASET_FAMILIES = {
    "BERT": [
        ("Bert", "full ft."),
        ("Bert_freeze", "linear probe"),
    ],
    "CalHousing": [
        ("CaliforniaHousing_linear", "ridge"),
        ("CaliforniaHousing_mlp", "MLP"),
    ],
    "CIFAR-10": [("Cifar10", "CNN")],
    "CIFAR-100": [("Cifar100", "ResNet-20")],
    "BloodMNIST": [("BloodMnist", "DINOv2")],
    "UCI Energy": [("UCI_energy", r"tab.\ MLP")],
}

SURROGATE_LABEL = {
    "loss":      r"Loss $\ell_i$",
    "newton":    r"Newton",
    "influence": r"Influence",
}
SURROGATE_ORDER = ["loss", "newton", "influence"]

K_TARGET = 0.01  # Only 1%
ATTACK = "lira"


def fmt(mean: float, std: float, *, scale: float, digits: int, is_best: bool = False, is_second: bool = False) -> str:
    if not np.isfinite(mean):
        return "--"
    
    m_str = f"{mean * scale:.{digits}f}"
    s_str = f"{std * scale:.{digits}f}" if np.isfinite(std) else ""

    if is_best:
        m_str = f"\\mathbf{{{m_str}}}"
        if s_str:
            s_str = f"\\mathbf{{{s_str}}}"
    elif is_second:
        m_str = f"\\underline{{{m_str}}}"
        if s_str:
            s_str = f"\\underline{{{s_str}}}"

    if not np.isfinite(std):
        return f"${m_str}$"
    return f"${m_str} \\pm {s_str}$"


def loginterp(k_grid: np.ndarray, recall_curve: np.ndarray, k: float) -> float:
    valid = ~np.isnan(recall_curve)
    if valid.sum() == 0:
        return np.nan
    return float(np.interp(np.log(k), np.log(k_grid[valid]), recall_curve[valid]))


def main() -> None:
    meta = json.load(open(AGG / "meta.json"))
    npz = np.load(AGG / "topk_recall.npz")
    k_pct = np.asarray(meta["k_pct"])
    attacks = meta["attacks"]
    ai = attacks.index(ATTACK)

    spearman = pd.read_csv(AGG / "spearman_long.csv")
    spearman = spearman[spearman["attack"] == ATTACK]

    table_data = collections.defaultdict(dict)
    rho_data = collections.defaultdict(dict)
    best_scores = {}
    second_best_scores = {}
    best_rho = {}
    second_best_rho = {}

    all_keys = [key for variants in DATASET_FAMILIES.values() for key, _ in variants]

    for ds_key in all_keys:
        ds_meta = meta["datasets"][ds_key]
        surrogates = ds_meta["surrogates"]
        models = ds_meta["models"]
        used = set(ds_meta["used_models"])
        keep = np.array([m in used for m in models])

        recall = npz[f"{ds_key}/recall"]
        for s in SURROGATE_ORDER:
            if s not in surrogates:
                table_data[ds_key][s] = (np.nan, np.nan)
                rho_data[ds_key][s] = (np.nan, np.nan)
                continue

            si = surrogates.index(s)
            curves = recall[keep, ai, si, :]

            vals = np.array([loginterp(k_pct, c, K_TARGET) for c in curves])
            vals = vals[~np.isnan(vals)]

            if vals.size == 0:
                table_data[ds_key][s] = (np.nan, np.nan)
            else:
                table_data[ds_key][s] = (
                    float(vals.mean()),
                    float(vals.std(ddof=1)) if vals.size > 1 else 0.0
                )

            sp = spearman[(spearman["dataset"] == ds_key) & (spearman["surrogate"] == s)]
            sp = sp[sp["model"].isin(used)]
            rho = sp["rho"].dropna().to_numpy()
            if rho.size == 0:
                rho_data[ds_key][s] = (np.nan, np.nan)
            else:
                rho_data[ds_key][s] = (
                    float(rho.mean()),
                    float(rho.std(ddof=1)) if rho.size > 1 else 0.0,
                )

        # Calculate best and second best scores for the row (recall)
        means = [table_data[ds_key][s][0] for s in SURROGATE_ORDER if np.isfinite(table_data[ds_key][s][0])]
        unique_means = sorted(list(set(means)), reverse=True)
        best_scores[ds_key] = unique_means[0] if len(unique_means) > 0 else np.nan
        second_best_scores[ds_key] = unique_means[1] if len(unique_means) > 1 else np.nan

        # Best / second-best for Spearman rho
        rhos = [rho_data[ds_key][s][0] for s in SURROGATE_ORDER if np.isfinite(rho_data[ds_key][s][0])]
        unique_rhos = sorted(list(set(rhos)), reverse=True)
        best_rho[ds_key] = unique_rhos[0] if len(unique_rhos) > 0 else np.nan
        second_best_rho[ds_key] = unique_rhos[1] if len(unique_rhos) > 1 else np.nan

    # Build LaTeX
    out: list[str] = []
    out.append(r"\begin{table}[t]")
    out.append(r"\centering")
    out.append(r"\small")
    out.append(r"\setlength{\tabcolsep}{6pt}") 
    out.append(
        r"\caption{LiRA membership-inference attack: top-$1\%$ recall and "
        r"Spearman rank correlation with the LiRA score. "
        r"Best results per model are in \textbf{bold}, second best are \underline{underlined}.}"
    )
    out.append(r"\label{tab:lira_summary_separated}")

    out.append(r"\begin{tabular}{ll ccc ccc}")
    out.append(r"\toprule")
    out.append(
        r"& & \multicolumn{3}{c}{Recall @ 1\% (\%)} "
        r"& \multicolumn{3}{c}{Spearman $\rho$} \\"
    )
    out.append(r"\cmidrule(lr){3-5} \cmidrule(lr){6-8}")

    surr_headers = [SURROGATE_LABEL[s] for s in SURROGATE_ORDER]
    headers = ["Dataset", "Model"] + surr_headers + surr_headers
    out.append(" & ".join(headers) + r" \\")
    out.append(r"\midrule")

    for family, variants in DATASET_FAMILIES.items():
        num_variants = len(variants)
        
        for i, (ds_key, model_name) in enumerate(variants):
            row_str = []
            
            if i == 0:
                if num_variants > 1:
                    row_str.append(rf"\multirow{{{num_variants}}}{{*}}{{{family}}}")
                else:
                    row_str.append(family)
            else:
                row_str.append("")
                
            row_str.append(model_name)
            
            for s in SURROGATE_ORDER:
                mean, std = table_data[ds_key][s]

                is_best = np.isfinite(mean) and abs(mean - best_scores.get(ds_key, np.nan)) < 1e-9
                is_second = np.isfinite(mean) and not is_best and abs(mean - second_best_scores.get(ds_key, np.nan)) < 1e-9

                row_str.append(fmt(mean, std, scale=100.0, digits=2, is_best=is_best, is_second=False))

            for s in SURROGATE_ORDER:
                mean, std = rho_data[ds_key][s]

                is_best = np.isfinite(mean) and abs(mean - best_rho.get(ds_key, np.nan)) < 1e-9
                is_second = np.isfinite(mean) and not is_best and abs(mean - second_best_rho.get(ds_key, np.nan)) < 1e-9

                row_str.append(fmt(mean, std, scale=1.0, digits=3, is_best=is_best, is_second=False))

            out.append(" & ".join(row_str) + r" \\")
            
        if family != list(DATASET_FAMILIES.keys())[-1]:
            out.append(r"\addlinespace")

    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    out.append(r"\end{table}")

    print("\n".join(out))

if __name__ == "__main__":
    main()