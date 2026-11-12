"""Top-k recall of attack-vulnerable points by model-only surrogates.

NeurIPS-text-width 2x4 panel (one panel per dataset, one figure per attack
target). Per panel we plot:
    * solid line / shaded ribbon = mean / +-1 std across the 16 target models,
    * dashed line                = single model (model_200),
    * dotted grey diagonal       = analytical random baseline (recall=k%).

The legend sits on the right of the figure as a single column split into
three blocks: surrogates (coloured), mean vs single-model (line styles),
and the random baseline.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from style import (
    ATTACK_DISPLAY,
    DATASETS,
    PALETTE,
    SURROGATE_LABELS,
    apply_style,
)

HERE = Path(__file__).resolve().parent
FIG = HERE.parent

AGG = FIG / "aggregated"
OUT = FIG / "pdf" / "topk_recall"
OUT.mkdir(parents=True, exist_ok=True)


# Loss is recoloured (a saturated blue) so it does not collide with the grey
# diagonal random baseline. Other surrogate colours come from the shared
# PALETTE in style.py.
LOSS_COLOUR = "#1f77b4"
PLOT_SURROGATES = [
    ("loss",      LOSS_COLOUR),
    ("grad",      PALETTE["grad"]),
    ("newton",    PALETTE["newton"]),
    ("influence", PALETTE["influence"]),
]

SINGLE_MODEL = "model_205"
RANDOM_COLOUR = "#888888"


# Two-line panel titles so they fit inside narrow NeurIPS-textwidth subplots.
PANEL_TITLES = {
    "Bert":                     "SST-2\n(BERT full ft.)",
    "Bert_freeze":              "SST-2\n(BERT linear probe)",
    "CaliforniaHousing_linear": "CalHousing\n(Ridge)",
    "CaliforniaHousing_mlp":    "CalHousing\n(MLP)",
    "Cifar10":                  "CIFAR-10\n(CNN)",
    "Cifar100":                 "CIFAR-100\n(ResNet-20)",
    "BloodMnist":               "BloodMNIST\n(DINOv2)",
    "UCI_energy":               "UCI Energy\n(MLP)",
}

# NeurIPS \textwidth (single column) is 5.5 in. We keep the legend on the
# right with a small font; tight left/right margins give the 2x4 grid as much
# horizontal real estate as possible.
FIG_WIDTH = 5.5
FIG_HEIGHT = 2.8
RIGHT_MARGIN = 0.82  # fraction of figure width occupied by the axes grid
LEFT_MARGIN = 0.075

NCOLS = 4
NROWS = 2

# Two figure variants: ``zoom`` controls the bottom-left axis limit. The wider
# 1e-2 variant also extends the y-axis down to 0.01% so the small-k tail is
# visible.
VARIANTS = [
    # (suffix, x_min_pct, y_min_pct)
    ("1e-1", 0.1,  0.1),
    ("1e-2", 0.01, 0.01),
]


def _draw_panel(ax, k_pct, recall, surrogates, ai, m_single):
    """Draw all curves for one (dataset, attack) panel."""
    x = k_pct * 100.0
    for sname, colour in PLOT_SURROGATES:
        if sname not in surrogates:
            continue
        si = surrogates.index(sname)
        arr = recall[:, ai, si, :] * 100.0
        valid = ~np.all(np.isnan(arr), axis=1)
        if valid.sum() == 0:
            continue
        arr_v = arr[valid]
        mean = np.nanmean(arr_v, axis=0)
        std = (np.nanstd(arr_v, axis=0, ddof=1)
               if arr_v.shape[0] > 1 else np.zeros_like(mean))
        ax.fill_between(x, np.clip(mean - std, 1e-3, None), mean + std,
                        color=colour, alpha=0.15, linewidth=0)
        ax.plot(x, mean, "-", color=colour, linewidth=1.1)
        if not np.all(np.isnan(recall[m_single, ai, si, :])):
            single = recall[m_single, ai, si, :] * 100.0
            ax.plot(x, single, "--", color=colour, linewidth=0.7, alpha=0.85)

    xx = np.geomspace(x.min(), x.max(), 200)
    ax.plot(xx, xx, ":", color=RANDOM_COLOUR, linewidth=0.7, zorder=0)


def _build_legends(fig):
    """Two stacked single-column legends on the right edge of the figure.

    Block 1 -- coloured techniques (loss / grad / Newton / Influence) and the
    random y=x baseline; Block 2 -- aggregation conventions (mean, single
    model, +-1 std band).
    """
    surro_handles = [
        Line2D([], [], color=col, linestyle="-", linewidth=1.3,
               label=SURROGATE_LABELS[name])
        for name, col in PLOT_SURROGATES
    ] + [
        Line2D([], [], color=RANDOM_COLOUR, linestyle=":", linewidth=0.9,
               label=r"random $y\!=\!x$"),
    ]
    style_handles = [
        Line2D([], [], color="0.25", linestyle="-",  linewidth=1.2,
               label=r"\begin{tabular}{l} 16-model \\ mean \end{tabular}"),
        Line2D([], [], color="0.25", linestyle="--", linewidth=0.9,
               label=r"single model"),
        Patch(facecolor="0.45", edgecolor="none", alpha=0.25,
              label=r"$\pm 1$ std band"),
    ]

    # Stacked anchor points on the right edge (figure coords). fig.legend
    # itself appends to fig.legends, so we do NOT call fig.add_artist.
    # Block 2 sits just below block 1 (small visual gap between them).
    x0 = RIGHT_MARGIN + 0.01
    leg_kwargs = dict(
        loc="upper left",
        frameon=False,
        handlelength=1.3,
        handletextpad=0.45,
        labelspacing=0.40,
        borderaxespad=0.1,
        fontsize=8,
    )
    fig.legend(handles=surro_handles, ncol=1,
               bbox_to_anchor=(x0, 0.85), **leg_kwargs)
    fig.legend(handles=style_handles, ncol=1,
               bbox_to_anchor=(x0, 0.5), **leg_kwargs)


def _build_legend_top(fig):
    """Single legend laid out across the top of the figure (2 rows x 4 cols).

    Row layout (matplotlib fills row-major):
        row 1 -- coloured techniques: Loss, grad, Newton, Influence;
        row 2 -- baseline + aggregation: random, mean, single, +-1 std.
    """
    surro_handles = [
        Line2D([], [], color=col, linestyle="-", linewidth=1.4,
               label=SURROGATE_LABELS[name])
        for name, col in PLOT_SURROGATES
    ]
    bottom_row = [
        Line2D([], [], color=RANDOM_COLOUR, linestyle=":", linewidth=1.0,
               label=r"random $y\!=\!x$"),
        Line2D([], [], color="0.25", linestyle="-",  linewidth=1.3,
               label=r"16-model mean"),
        Line2D([], [], color="0.25", linestyle="--", linewidth=1.0,
               label=r"single model"),
        Patch(facecolor="0.45", edgecolor="none", alpha=0.25,
              label=r"$\pm 1$ std band"),
    ]
    handles = surro_handles + bottom_row

    fig.legend(
        handles=handles,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
        handlelength=1.6,
        handletextpad=0.45,
        columnspacing=1.8,
        labelspacing=0.45,
        fontsize=8,
        
    )


def main() -> None:
    apply_style()
    # Constrained layout fights with externally-anchored legends; turn it off
    # for this figure (subplots_adjust handles spacing instead).
    plt.rcParams["figure.constrained_layout.use"] = False

    npz = np.load(AGG / "topk_recall.npz")
    meta = json.load(open(AGG / "meta.json"))
    k_pct = np.asarray(meta["k_pct"])
    attacks = meta["attacks"]

    for ai, attack in enumerate(attacks):
        for suffix, x_min, y_min in VARIANTS:
            # Existing layout: legend stacked on the right.
            _make_figure(npz, meta, k_pct, ai, attack,
                         suffix=suffix, x_min=x_min, y_min=y_min,
                         legend_pos="right")
            # Comparison layout: legend laid out across the top.
            _make_figure(npz, meta, k_pct, ai, attack,
                         suffix=f"{suffix}_topleg", x_min=x_min, y_min=y_min,
                         legend_pos="top")


def _make_figure(npz, meta, k_pct, ai, attack, *, suffix, x_min, y_min,
                 legend_pos="right"):
    if legend_pos == "right":
        figsize = (FIG_WIDTH, FIG_HEIGHT)
        adjust = dict(left=LEFT_MARGIN, right=RIGHT_MARGIN,
                      top=0.9, bottom=0.12, wspace=0.18, hspace=0.40)
    else:  # top
        # Wider effective grid (no right legend column) and a taller figure to
        # fit a 2-row legend above the panels.
        figsize = (FIG_WIDTH, 3.20)
        adjust = dict(left=LEFT_MARGIN, right=0.99,
                      top=0.8, bottom=0.1, wspace=0.18, hspace=0.40)

    fig, axes = plt.subplots(NROWS, NCOLS, figsize=figsize,
                             sharex=True, sharey=True,
                             constrained_layout=False)
    fig.subplots_adjust(**adjust)

    for di, (dataset, label, _) in enumerate(DATASETS):
        ax = axes[di // NCOLS, di % NCOLS]
        key = f"{dataset}/recall"
        if key not in npz.files:
            ax.set_title(label + r"\\(no data)", fontsize=8)
            ax.set_axis_off()
            continue
        recall = npz[key]
        ds_meta = meta["datasets"][dataset]
        surrogates = ds_meta["surrogates"]
        try:
            m_single = ds_meta["models"].index(SINGLE_MODEL)
        except ValueError:
            m_single = 0

        _draw_panel(ax, k_pct, recall, surrogates, ai, m_single)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(PANEL_TITLES.get(dataset, label),
                     fontsize=8, pad=1.5)
        ax.set_xlim(x_min, k_pct.max() * 100.0)
        ax.set_ylim(y_min, 100)

        ax.minorticks_on()
        ax.tick_params(which="major", length=2.8, width=0.5,
                       direction="in", pad=2.5)
        ax.tick_params(which="minor", length=1.6, width=0.35,
                       direction="in")

    for k in range(len(DATASETS), NROWS * NCOLS):
        axes[k // NCOLS, k % NCOLS].set_axis_off()

    # Single shared x/y label for the whole grid.
    fig.supxlabel(r"top-$k$ percentile (\%)", fontsize=9, y=0.005)
    fig.supylabel(r"recall (\%)",             fontsize=9, x=0.001)

    if legend_pos == "right":
        _build_legends(fig)
    else:
        _build_legend_top(fig)

    out_path = OUT / f"topk_recall_{attack}_{suffix}.pdf"
    fig.savefig(out_path)  # no bbox_inches='tight' -> exact 5.5in width
    plt.close(fig)
    print(f"wrote {out_path}")
    try:
        import subprocess
        r = subprocess.run(["pdfinfo", str(out_path)],
                           capture_output=True, text=True)
        for line in r.stdout.splitlines():
            if line.startswith("Page size:"):
                print(f"          {line.strip()}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
