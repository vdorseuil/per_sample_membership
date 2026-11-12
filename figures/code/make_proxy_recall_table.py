r"""Top-1\,\% recovery tables for cheap proxies vs reference attacks.

For each (dataset, target model, attack, surrogate) we compute, on the
training set:

    Q1. ``k95``  -- smallest top-$k\,\%$ by surrogate that contains 95\,\% of
        the true top-1\,\% by attack score.  Lower is better.
    Q2. ``r5``   -- recall of the true top-1\,\% by attack when keeping the
        top-5\,\% by surrogate.  Higher is better.

Per-sample data is loaded the same way as
``figures/code/aggregate.py`` (so the surrogate set matches the LiRA recall
table, ``make_lira_table.py``):

    * loss, ``\|\nabla\ell\|``  from ``metrics.pkl`` (split == 'train')
    * Newton, Influence         from the latest ``leverage_score/<run>/`` dir
    * attack score              from ``asr_per_sample.csv``

We then average mean $\pm$ std across each dataset's ``used_models`` (same
pool the LiRA table uses).  Outputs four LaTeX ``table`` environments saved
as ``.txt``:

    proxy_k_for_95recall_lira.txt
    proxy_recall_at_5pct_lira.txt
    proxy_k_for_95recall_all_attacks.txt
    proxy_recall_at_5pct_all_attacks.txt
"""
from __future__ import annotations

import json
import pickle
import collections
from pathlib import Path

import numpy as np
import pandas as pd

from style import DATASETS, MODEL_IDS, REGRESSION_DATASETS

HERE = Path(__file__).resolve().parent
FIG = HERE.parent
REPO = FIG.parent

TGT = REPO / "experiments" / "target_models"
REF = REPO / "experiments" / "reference_models"
OUT_DIR = FIG / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ATTACKS = ["lira", "rmia", "shokri"]
ATTACK_LABEL = {
    "lira":   r"LiRA",
    "rmia":   r"RMIA",
    "shokri": r"Shokri",
}

# Same surrogate set & order as make_lira_table.py.
SURROGATE_ORDER = ["loss", "grad", "newton", "influence"]
SURROGATE_LABEL = {
    "loss":      r"Loss $\ell_i$",
    "grad":      r"$\|\nabla\ell_i\|$",
    "newton":    r"Newton",
    "influence": r"Influence",
}

# Dataset family -> list of (dataset_key, architecture label). Mirrors the
# split in make_small_lira_table.py so the rendered tables match the existing
# LiRA-only table style.
DATASET_FAMILIES = {
    "SST-2": [
        ("Bert",        r"BERT, full ft."),
        ("Bert_freeze", r"BERT, linear probe"),
    ],
    "CalHousing": [
        ("CaliforniaHousing_linear", r"Ridge regression"),
        ("CaliforniaHousing_mlp",    r"MLP"),
    ],
    "CIFAR-10":   [("Cifar10",    r"CNN")],
    "CIFAR-100":  [("Cifar100",   r"ResNet-20")],
    "BloodMNIST": [("BloodMnist", r"DINOv2")],
    "UCI Energy": [("UCI_energy", r"tab.\ MLP")],
}
DATASET_ORDER = [k for variants in DATASET_FAMILIES.values() for k, _ in variants]

TARGET_TOP_FRAC = 0.01      # "top 1 %"
RECALL_TARGET   = 0.95      # "recover 95 %"
FIXED_K_FRAC    = 0.05      # "top 5 %"


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


def k_to_recover(scores: np.ndarray, attack: np.ndarray,
                 top_frac: float, recall_target: float) -> float:
    n = len(scores)
    n_top = max(1, int(round(top_frac * n)))
    top_idx = np.argsort(-attack, kind="stable")[:n_top]
    is_top = np.zeros(n, dtype=bool)
    is_top[top_idx] = True

    proxy_order = np.argsort(-scores, kind="stable")
    cum_top_hits = np.cumsum(is_top[proxy_order])
    needed = int(np.ceil(recall_target * n_top))
    pos = np.searchsorted(cum_top_hits, needed, side="left")
    if pos >= n:
        return 1.0
    return (pos + 1) / n


def recall_at_fixed_k(scores: np.ndarray, attack: np.ndarray,
                      top_frac: float, k_frac: float) -> float:
    n = len(scores)
    n_top = max(1, int(round(top_frac * n)))
    n_k = max(1, int(round(k_frac * n)))
    top_set = set(np.argsort(-attack, kind="stable")[:n_top].tolist())
    k_set   = set(np.argsort(-scores, kind="stable")[:n_k].tolist())
    return len(top_set & k_set) / n_top


def collect() -> pd.DataFrame:
    """One row per (dataset, model, attack, surrogate) with k95 and r5."""
    rows = []
    for dataset, _, _ in DATASETS:
        asr_path = REF / dataset / "asr_per_sample.csv"
        if not asr_path.exists():
            print(f"  [{dataset}] no asr_per_sample.csv -- skipping")
            continue
        asr = pd.read_csv(asr_path).set_index("idx")
        if not all(c in asr.columns for c in ATTACKS):
            print(f"  [{dataset}] asr columns missing")
            continue

        base = TGT / dataset / "outputs"
        if not base.exists():
            print(f"  [{dataset}] no target outputs -- skipping")
            continue

        for mname in MODEL_IDS:
            mdir = base / mname
            if not mdir.is_dir():
                continue
            run = latest_run(mdir)
            if run is None:
                continue
            try:
                with open(run / "leverage_newton.pkl", "rb") as fh:
                    newton = pickle.load(fh)
                with open(run / "leverage_influence.pkl", "rb") as fh:
                    influence = pickle.load(fh)
                metrics_df = pickle.load(open(mdir / "metrics.pkl", "rb"))
            except Exception as exc:
                print(f"  [{dataset}/{mname}] load fail: {exc}")
                continue

            train = metrics_df[metrics_df["split"] == "train"].set_index("idx")
            if "loss" not in train.columns or "local_grad_norm" not in train.columns:
                continue

            common = sorted(set(newton.keys())
                            & set(influence.keys())
                            & set(asr.index)
                            & set(train.index))
            if len(common) < 50:
                continue
            idx = np.asarray(common)

            score_map = {
                "loss":      train.loc[idx, "loss"].to_numpy(dtype=float),
                "grad":      train.loc[idx, "local_grad_norm"].to_numpy(dtype=float),
                "newton":    np.abs(np.asarray([newton[i]    for i in idx], dtype=float)),
                "influence": np.abs(np.asarray([influence[i] for i in idx], dtype=float)),
            }
            attack_map = {a: asr.loc[idx, a].to_numpy(dtype=float) for a in ATTACKS}

            for attack in ATTACKS:
                t = attack_map[attack]
                for s in SURROGATE_ORDER:
                    sc = score_map[s]
                    m = np.isfinite(sc) & np.isfinite(t)
                    if m.sum() < 50:
                        continue
                    k95 = k_to_recover(sc[m], t[m], TARGET_TOP_FRAC, RECALL_TARGET)
                    r5  = recall_at_fixed_k(sc[m], t[m], TARGET_TOP_FRAC, FIXED_K_FRAC)
                    rows.append({
                        "dataset":   dataset,
                        "model":     mname,
                        "attack":    attack,
                        "surrogate": s,
                        "k95_pct":   k95 * 100,
                        "r5_pct":    r5 * 100,
                    })
        used = sorted(set(r["model"] for r in rows if r["dataset"] == dataset))
        print(f"  [{dataset}] used models: {len(used)}")

    return pd.DataFrame(rows)


def fmt(mean: float, std: float, *, digits: int = 1, is_best: bool = False) -> str:
    if not np.isfinite(mean):
        return "--"
    m_str = f"{mean:.{digits}f}"
    if is_best:
        m_str = f"\\mathbf{{{m_str}}}"
    if not np.isfinite(std):
        return f"${m_str}$"
    s_str = f"{std:.{digits}f}"
    if is_best:
        s_str = f"\\mathbf{{{s_str}}}"
    return f"${m_str} \\pm {s_str}$\\,\\%"


def aggregate(table: pd.DataFrame, value_col: str
              ) -> dict[tuple[str, str, str], tuple[float, float]]:
    """Mean +- std across models for each (dataset, attack, surrogate)."""
    stats = {}
    for (ds, attack, s), grp in table.groupby(["dataset", "attack", "surrogate"]):
        v = grp[value_col].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            stats[(ds, attack, s)] = (np.nan, np.nan)
        elif v.size == 1:
            stats[(ds, attack, s)] = (float(v.mean()), 0.0)
        else:
            stats[(ds, attack, s)] = (float(v.mean()),
                                      float(v.std(ddof=1)))
    return stats


def _wrap_table(body: list[str], colspec: str, caption: str, label: str) -> str:
    out = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\renewcommand{\arraystretch}{1.2}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{" + caption + r"}",
        r"\label{" + label + r"}",
        r"\begin{tabular}{" + colspec + r"}",
        r"\toprule",
    ]
    out.extend(body)
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    out.append(r"\end{table}")
    return "\n".join(out) + "\n"


def _best_per_dataset(stats, datasets, attacks, surrogates, *, lower_is_better):
    """Map (ds, attack) -> the surrogate's mean that wins (min or max)."""
    best = {}
    for ds in datasets:
        for a in attacks:
            means = []
            for s in surrogates:
                m, _ = stats.get((ds, a, s), (np.nan, np.nan))
                if np.isfinite(m):
                    means.append(m)
            if not means:
                best[(ds, a)] = np.nan
            else:
                best[(ds, a)] = min(means) if lower_is_better else max(means)
    return best


def write_lira_only(stats, value_col_name: str, caption: str, label: str,
                    out: Path, *, lower_is_better: bool,
                    top_header: str | None = None) -> None:
    """One row per (dataset, model) pair, columns = surrogates. LiRA only.

    Layout follows make_small_lira_table.py: ``Dataset`` and ``Model`` are
    separate columns, with \\multirow grouping multi-architecture families.
    """
    a = "lira"
    best = _best_per_dataset(stats, DATASET_ORDER, [a], SURROGATE_ORDER,
                             lower_is_better=lower_is_better)
    body = []
    n_surr = len(SURROGATE_ORDER)
    if top_header is not None:
        body.append(r"& & \multicolumn{" + str(n_surr) + r"}{c}{"
                    + top_header + r"} \\")
        body.append(r"\cmidrule(lr){3-" + str(2 + n_surr) + r"}")
    body.append(r"Dataset & Model & "
                + " & ".join(SURROGATE_LABEL[s] for s in SURROGATE_ORDER)
                + r" \\")
    body.append(r"\midrule")
    families = list(DATASET_FAMILIES.items())
    for fi, (family, variants) in enumerate(families):
        for vi, (ds, arch) in enumerate(variants):
            cells = []
            for s in SURROGATE_ORDER:
                m, sd = stats.get((ds, a, s), (np.nan, np.nan))
                is_best = (np.isfinite(m) and np.isfinite(best[(ds, a)])
                           and abs(m - best[(ds, a)]) < 1e-9)
                cells.append(fmt(m, sd, is_best=is_best))
            if vi == 0:
                fam_cell = (rf"\multirow{{{len(variants)}}}{{*}}{{{family}}}"
                            if len(variants) > 1 else family)
            else:
                fam_cell = ""
            body.append(fam_cell + " & " + arch + " & "
                        + " & ".join(cells) + r" \\")
        if fi < len(families) - 1:
            body.append(r"\addlinespace")

    colspec = "ll " + " ".join(["c"] * len(SURROGATE_ORDER))
    out.write_text(_wrap_table(body, colspec, caption, label))
    print(f"wrote {out}")


def write_all_attacks(stats, value_col_name: str, caption: str, label: str,
                      out: Path, *, lower_is_better: bool,
                      top_header: str | None = None) -> None:
    """One row per (dataset, model, surrogate), columns = attacks.

    Dataset family is grouped via \\multirow spanning every surrogate-row of
    every architecture in the family. Within a family, each architecture's
    block is preceded by a small horizontal rule.
    """
    best = _best_per_dataset(stats, DATASET_ORDER, ATTACKS, SURROGATE_ORDER,
                             lower_is_better=lower_is_better)
    n_surr = len(SURROGATE_ORDER)
    n_atk = len(ATTACKS)
    body = []
    if top_header is not None:
        body.append(r"& & & \multicolumn{" + str(n_atk) + r"}{c}{"
                    + top_header + r"} \\")
        body.append(r"\cmidrule(lr){4-" + str(3 + n_atk) + r"}")
    body.append(r"Dataset & Model & Surrogate & "
                + " & ".join(ATTACK_LABEL[a] for a in ATTACKS) + r" \\")
    body.append(r"\midrule")
    families = list(DATASET_FAMILIES.items())
    for fi, (family, variants) in enumerate(families):
        family_rows = len(variants) * n_surr
        for vi, (ds, arch) in enumerate(variants):
            for si, s in enumerate(SURROGATE_ORDER):
                cells = []
                for a in ATTACKS:
                    m, sd = stats.get((ds, a, s), (np.nan, np.nan))
                    is_best = (np.isfinite(m) and np.isfinite(best[(ds, a)])
                               and abs(m - best[(ds, a)]) < 1e-9)
                    cells.append(fmt(m, sd, is_best=is_best))
                if vi == 0 and si == 0:
                    fam_cell = (rf"\multirow{{{family_rows}}}{{*}}{{{family}}}"
                                if family_rows > 1 else family)
                else:
                    fam_cell = ""
                arch_cell = (rf"\multirow{{{n_surr}}}{{*}}{{{arch}}}"
                             if si == 0 else "")
                body.append(fam_cell + " & " + arch_cell + " & "
                            + SURROGATE_LABEL[s] + " & "
                            + " & ".join(cells) + r" \\")
            if vi < len(variants) - 1:
                # Separate architectures within the same family with a partial
                # rule that keeps the family multirow intact.
                body.append(r"\cmidrule(lr){2-" + str(3 + len(ATTACKS)) + r"}")
        if fi < len(families) - 1:
            body.append(r"\midrule")

    colspec = "lll " + " ".join(["c"] * len(ATTACKS))
    out.write_text(_wrap_table(body, colspec, caption, label))
    print(f"wrote {out}")


def main() -> None:
    table = collect()
    raw_csv = OUT_DIR / "proxy_recall_raw.csv"
    table.to_csv(raw_csv, index=False)
    print(f"wrote {raw_csv}")

    stats_k95 = aggregate(table, "k95_pct")
    stats_r5  = aggregate(table, "r5_pct")

    cap_k95_lira = (
        r"LiRA: smallest top-$k\,\%$ of training points (ranked by each "
        r"surrogate) that recovers 95\,\% of the true LiRA top-1\,\%. "
        r"Mean $\pm$ std across target models. Lower is better; "
        r"\textbf{bold} = best per dataset."
    )
    cap_r5_lira = (
        r"LiRA: recall (\%) of the true LiRA top-1\,\% when keeping the "
        r"top-5\,\% by each surrogate. Mean $\pm$ std across target models. "
        r"Higher is better; \textbf{bold} = best per dataset."
    )
    cap_k95_all = (
        r"Smallest top-$k\,\%$ of training points (ranked by each surrogate) "
        r"that recovers 95\,\% of the true top-1\,\% by each attack. "
        r"Mean $\pm$ std across target models. Lower is better; "
        r"\textbf{bold} = best per (dataset, attack)."
    )
    cap_r5_all = (
        r"Recall (\%) of the true top-1\,\% by each attack when keeping the "
        r"top-5\,\% by each surrogate. Mean $\pm$ std across target models. "
        r"Higher is better; \textbf{bold} = best per (dataset, attack)."
    )

    write_lira_only(stats_k95, "k95_pct", cap_k95_lira,
                    label="tab:proxy-k-for-95recall-lira",
                    out=OUT_DIR / "proxy_k_for_95recall_lira.txt",
                    lower_is_better=True)
    write_lira_only(stats_r5, "r5_pct", cap_r5_lira,
                    label="tab:proxy-recall-at-5pct-lira",
                    out=OUT_DIR / "proxy_recall_at_5pct_lira.txt",
                    lower_is_better=False,
                    top_header=r"Recall of Top 1\% @ 5\% (in \%)")
    write_all_attacks(stats_k95, "k95_pct", cap_k95_all,
                      label="tab:proxy-k-for-95recall-all",
                      out=OUT_DIR / "proxy_k_for_95recall_all_attacks.txt",
                      lower_is_better=True)
    write_all_attacks(stats_r5, "r5_pct", cap_r5_all,
                      label="tab:proxy-recall-at-5pct-all",
                      out=OUT_DIR / "proxy_recall_at_5pct_all_attacks.txt",
                      lower_is_better=False,
                      top_header=r"Recall of Top 1\% @ 5\% (in \%)")


if __name__ == "__main__":
    main()
