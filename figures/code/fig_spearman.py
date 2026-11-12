"""Spearman rank correlation between surrogate scores and attack ASR.

For every dataset and every attack (LiRA / RMIA / Shokri), we plot the absolute
rank correlation $|\\rho_S(\\textit{surrogate}, \\textit{attack ASR})|$ as a
grouped bar chart. Heights are means over the 16 target models, error bars are
$\\pm 1$ std across models. We also overlay a thin marker showing the value
obtained from a single model (model_200) for context.

One figure per attack, with all 8 datasets on a single 2x4 grid.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from style import (
    ATTACK_DISPLAY,
    DATASETS,
    PALETTE,
    REGRESSION_DATASETS,
    SURROGATE_LABELS,
    apply_style,
)

HERE = Path(__file__).resolve().parent
FIG = HERE.parent

AGG = FIG / "aggregated"
OUT = FIG / "pdf" / "spearman"
OUT.mkdir(parents=True, exist_ok=True)


SURROGATE_ORDER = [
    ("loss",      PALETTE["loss"]),
    ("grad",      PALETTE["grad"]),
    ("exact",     PALETTE["exact"]),
    ("newton",    PALETTE["newton"]),
    ("influence", PALETTE["influence"]),
]

SINGLE_MODEL = "model_200"


def main() -> None:
    apply_style()
    df = pd.read_csv(AGG / "spearman_long.csv")
    df["abs_rho"] = df["rho"].abs()

    attacks = ["lira", "rmia", "shokri"]
    for attack in attacks:
        sub_all = df[df.attack == attack]

        n_ds = len(DATASETS)
        ncols = 4
        nrows = int(np.ceil(n_ds / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(7.2, 1.95 * nrows),
                                 sharey=True)
        axes = np.atleast_2d(axes)

        seen_handles: dict[str, plt.Rectangle] = {}

        for di, (dataset, label, _) in enumerate(DATASETS):
            ax = axes[di // ncols, di % ncols]
            sub = sub_all[sub_all.dataset == dataset]
            if sub.empty:
                ax.set_title(label + r"\\(no data)")
                ax.set_axis_off()
                continue

            # Restrict surrogates: skip exact for non-regression.
            surrogates_here = [(s, c) for s, c in SURROGATE_ORDER
                               if s != "exact" or dataset in REGRESSION_DATASETS]
            n_s = len(surrogates_here)
            xpos = np.arange(n_s, dtype=float)

            means, stds, singles = [], [], []
            for sname, _ in surrogates_here:
                vals = sub[sub.surrogate == sname]["abs_rho"].dropna().values
                if vals.size == 0:
                    means.append(np.nan); stds.append(0.0); singles.append(np.nan)
                    continue
                means.append(float(np.mean(vals)))
                stds.append(float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0)
                row = sub[(sub.surrogate == sname) & (sub.model == SINGLE_MODEL)]
                singles.append(float(row["abs_rho"].iloc[0]) if not row.empty else np.nan)

            for i, (sname, colour) in enumerate(surrogates_here):
                bar = ax.bar(xpos[i], means[i], width=0.7, color=colour,
                             edgecolor="white", linewidth=0.5,
                             label=SURROGATE_LABELS[sname])
                seen_handles.setdefault(sname, bar[0])
                ax.errorbar(xpos[i], means[i], yerr=stds[i],
                            fmt="none", ecolor="0.2", elinewidth=0.7,
                            capsize=1.8, capthick=0.5)
                if not np.isnan(singles[i]):
                    ax.plot(xpos[i], singles[i], marker="D", markersize=3.0,
                            markerfacecolor="white",
                            markeredgecolor="0.15",
                            markeredgewidth=0.6, linestyle="None", zorder=5)

            ax.set_xticks(xpos)
            ax.set_xticklabels([], visible=False)
            ax.set_title(label)
            ax.set_ylim(0, 1.0)
            ax.tick_params(axis="x", which="both", length=0)

            if di % ncols == 0:
                ax.set_ylabel(rf"$|\rho_S|$ vs.\ {ATTACK_DISPLAY[attack]} ASR")

        # Hide unused axes
        for k in range(n_ds, nrows * ncols):
            axes[k // ncols, k % ncols].set_axis_off()

        # Legend: bars in PLOT order + marker for single model
        from matplotlib.lines import Line2D

        bar_handles = [seen_handles[s] for s, _ in SURROGATE_ORDER if s in seen_handles]
        bar_labels = [SURROGATE_LABELS[s] for s, _ in SURROGATE_ORDER if s in seen_handles]
        proxy = Line2D([], [], marker="D", color="white", markerfacecolor="white",
                       markeredgecolor="0.15", markeredgewidth=0.6, markersize=4,
                       linestyle="None",
                       label=r"single model (\texttt{model\_200})")
        leg = fig.legend(
            handles=bar_handles + [proxy],
            labels=bar_labels + [r"single model (\texttt{model\_200})"],
            ncol=min(7, len(bar_handles) + 1),
            loc="lower center",
            bbox_to_anchor=(0.5, 1.0),
            frameon=True,
            fancybox=False,
            edgecolor="0.4",
            handletextpad=0.5,
            columnspacing=1.0,
        )
        leg.get_frame().set_linewidth(0.5)

        fig.suptitle(rf"Spearman $|\rho_S|$ vs.\ {ATTACK_DISPLAY[attack]} ASR "
                     r"(mean $\pm 1$ std over 16 models)",
                     y=1.10, fontsize=10)

        out_path = OUT / f"spearman_{attack}.pdf"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
