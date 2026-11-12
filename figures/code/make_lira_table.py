r"""Print a LaTeX (booktabs) summary table for LiRA ASR.

Per (dataset, surrogate) row: mean +- 1 std of top-k recall at k in
{0.1%, 1%, 5%} and Spearman rho. Recall is log-interpolated on the saved
k_pct grid; the average is taken across each dataset's ``used_models``.
Best results per dataset are highlighted in bold.
Dataset names are split across two lines using \makecell to save space.
"""
from __future__ import annotations

import json
from pathlib import Path
import collections

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FIG = HERE.parent

AGG = FIG / "aggregated"

# Added \\ to split the dataset name and the architecture
DATASET_LABEL = {
    "Bert":                     r"SST2 \\ (BERT, full ft.)",
    "Bert_freeze":              r"SST2 \\ (BERT, linear probe)",
    "CaliforniaHousing_linear": r"CalHousing \\ (ridge)",
    "CaliforniaHousing_mlp":    r"CalHousing \\ (MLP)",
    "Cifar10":                  r"CIFAR-10 \\ (CNN)",
    "Cifar100":                 r"CIFAR-100 \\ (ResNet-20)",
    "BloodMnist":               r"BloodMNIST \\ (DINOv2)",
    "UCI_energy":               r"UCI Energy \\ (tab.\ MLP)",
}
DATASET_ORDER = list(DATASET_LABEL.keys())

SURROGATE_LABEL = {
    "loss":      r"Loss $\ell_i$",
    "grad":      r"$\|\nabla\ell_i\|$",
    # "exact":     r"$\ell_i\,(2 h_{ii}-h_{ii}^2)/(1-h_{ii})^2$",
    "entropy":   r"Entropy $H_i$",
    "newton":    r"Newton",
    "influence": r"Influence",
}
SURROGATE_ORDER = ["loss", "grad", "entropy", "newton", "influence"]

K_TARGETS = [0.001, 0.01, 0.05]  # 0.1 %, 1 %, 5 %
ATTACK = "lira"


def fmt(mean: float, std: float, *, scale: float, digits: int, is_best: bool = False) -> str:
    if not np.isfinite(mean):
        return "--"
    
    m_str = f"{mean * scale:.{digits}f}"
    s_str = f"{std * scale:.{digits}f}" if np.isfinite(std) else ""

    if is_best:
        m_str = f"\\mathbf{{{m_str}}}"
        if s_str:
            s_str = f"\\mathbf{{{s_str}}}"

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

    rows: list[tuple[str, str, dict[str, tuple[float, float]]]] = []
    for ds in DATASET_ORDER:
        ds_meta = meta["datasets"][ds]
        surrogates = ds_meta["surrogates"]
        models = ds_meta["models"]
        used = set(ds_meta["used_models"])
        keep = np.array([m in used for m in models])

        recall = npz[f"{ds}/recall"] 
        for s in SURROGATE_ORDER:
            if s not in surrogates:
                continue
            si = surrogates.index(s)
            curves = recall[keep, ai, si, :]
            stats: dict[str, tuple[float, float]] = {}
            for k in K_TARGETS:
                vals = np.array([loginterp(k_pct, c, k) for c in curves])
                vals = vals[~np.isnan(vals)]
                if vals.size == 0:
                    stats[f"r{k}"] = (np.nan, np.nan)
                else:
                    stats[f"r{k}"] = (
                        float(vals.mean()),
                        float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
                    )

            sp = spearman[(spearman["dataset"] == ds) & (spearman["surrogate"] == s)]
            sp = sp[sp["model"].isin(used)]
            rho = sp["rho"].dropna().to_numpy()
            if rho.size == 0:
                stats["rho"] = (np.nan, np.nan)
            else:
                stats["rho"] = (
                    float(rho.mean()),
                    float(rho.std(ddof=1)) if rho.size > 1 else 0.0,
                )
            rows.append((ds, s, stats))

    best_scores = {ds: {} for ds in DATASET_ORDER}
    keys_to_check = [f"r{k}" for k in K_TARGETS] + ["rho"]
    
    for ds in DATASET_ORDER:
        ds_rows = [r for r in rows if r[0] == ds]
        for key in keys_to_check:
            means = [r[2][key][0] for r in ds_rows if np.isfinite(r[2][key][0])]
            best_scores[ds][key] = max(means) if means else np.nan

    # Count how many surrogates (rows) belong to each dataset for \multirow
    ds_row_counts = collections.Counter([r[0] for r in rows])

    # Build LaTeX.
    out: list[str] = []
    out.append(r"\begin{table}[t]")
    out.append(r"\centering")
    out.append(r"\small")
    out.append(r"\renewcommand{\arraystretch}{1.2}") # Slightly larger stretch helps multi-line cells breathe
    out.append(r"\setlength{\tabcolsep}{4pt}")
    out.append(
        r"\caption{LiRA membership-inference attack: top-$k$ recall (\%) and "
        r"Spearman rank correlation. Best results per dataset are in \textbf{bold}.}"
    )
    out.append(r"\label{tab:lira_summary}")
    # Reverted to 'll' for standard left-aligned columns
    out.append(r"\begin{tabular}{ll ccc c}") 
    out.append(r"\toprule")
    out.append(
        r"& & \multicolumn{3}{c}{Recall (\%)} & \\"
    )
    out.append(r"\cmidrule(lr){3-5}")
    out.append(
        r"Dataset & Surrogate "
        r"& @0.1\% & @1\% & @5\% "
        r"& Spearman $\rho$ \\"
    )
    out.append(r"\midrule")

    last_ds = None
    for ds, s, stats in rows:
        if ds != last_ds:
            if last_ds is not None:
                out.append(r"\midrule")
            
            # Use \multirow and \makecell[l] to allow line breaks while vertically centering
            num_rows = ds_row_counts[ds]
            ds_name = DATASET_LABEL[ds]
            ds_cell = rf"\multirow{{{num_rows}}}{{*}}{{\makecell[l]{{{ds_name}}}}}"
            last_ds = ds
        else:
            ds_cell = ""
            
        def is_best(key: str) -> bool:
            mean = stats[key][0]
            best = best_scores[ds][key]
            return np.isfinite(mean) and abs(mean - best) < 1e-9

        r01 = fmt(*stats["r0.001"], scale=100.0, digits=2, is_best=is_best("r0.001"))
        r1  = fmt(*stats["r0.01"], scale=100.0, digits=2, is_best=is_best("r0.01"))
        r5  = fmt(*stats["r0.05"], scale=100.0, digits=2, is_best=is_best("r0.05"))
        rho = fmt(*stats["rho"], scale=1.0, digits=3, is_best=is_best("rho"))
        
        out.append(
            f"{ds_cell} & {SURROGATE_LABEL[s]} & {r01} & {r1} & {r5} & {rho} \\\\"
        )

    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    out.append(r"\end{table}")

    print("\n".join(out))

if __name__ == "__main__":
    main()