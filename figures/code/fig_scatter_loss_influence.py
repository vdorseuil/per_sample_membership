"""Scatter of (local loss, influence) per training sample, coloured by LiRA ASR.

For each of the 8 (dataset, model arch) entries in ``style.DATASETS`` we plot
one scatter: x = local loss $\\ell_i$, y = $|\\text{Influence}_i|$, colour =
per-sample LiRA ASR. Two outputs:

    1. A 2x4 grid figure (one panel per dataset), matching the layout of
       ``fig_topk_recall.py``.
    2. One standalone figure per dataset under
       ``figures/pdf/scatter_loss_influence/``.

The colormap is a custom sequential ramp from light grey to the LiRA red used
in the rest of the paper (``style.PALETTE['lira']``), so that the scatters read
consistently with the top-k recall and Spearman figures.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from style import DATASETS, apply_style

HERE = Path(__file__).resolve().parent
FIG = HERE.parent
REPO = FIG.parent

TGT = REPO / "experiments" / "target_models"
REF = REPO / "experiments" / "reference_models"
OUT = FIG / "pdf" / "scatter_loss_influence"
OUT.mkdir(parents=True, exist_ok=True)


SINGLE_MODEL = "model_205"
ATTACK = "lira"

# Colormap: ``viridis`` — perceptually uniform, runs purple -> blue -> green ->
# yellow. Picks up the same green / yellow / cyan family as the surrogate line
# colours in fig_topk_recall (loss=blue, newton=cyan, influence=green,
# grad=yellow), so the LiRA-ASR encoding sits naturally next to those plots
# without the red clash of the previous custom ramp.
LIRA_CMAP = plt.get_cmap("viridis")

# Two-line panel titles (kept identical to fig_topk_recall.py for visual
# consistency in the paper).
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

NCOLS = 4
NROWS = 2


def _latest_run(model_dir: Path) -> Path | None:
    base = model_dir / "leverage_score"
    if not base.exists():
        return None
    candidates = [
        r for r in base.iterdir()
        if r.is_dir()
        and (r / "leverage_influence.pkl").exists()
    ]
    return sorted(candidates)[-1] if candidates else None


def _load_dataset(dataset: str):
    """Return arrays (loss, influence, lira_asr) for SINGLE_MODEL on ``dataset``.

    Returns None if any required file is missing or the intersection of indices
    is too small to be informative.
    """
    asr_path = REF / dataset / "asr_per_sample.csv"
    if not asr_path.exists():
        return None
    asr_full = pd.read_csv(asr_path).set_index("idx")
    if ATTACK not in asr_full.columns:
        return None

    model_dir = TGT / dataset / "outputs" / SINGLE_MODEL
    if not model_dir.is_dir():
        return None
    run = _latest_run(model_dir)
    if run is None:
        return None

    try:
        with open(run / "leverage_influence.pkl", "rb") as fh:
            influence = pickle.load(fh)
        mdf = pickle.load(open(model_dir / "metrics.pkl", "rb"))
    except Exception as exc:
        print(f"  [{dataset}] load fail: {exc}")
        return None

    train = mdf[mdf["split"] == "train"].set_index("idx")
    if "loss" not in train.columns:
        return None

    common = sorted(set(influence.keys())
                    & set(asr_full.index)
                    & set(train.index))
    if len(common) < 50:
        return None
    idx = np.asarray(common)
    loss_arr = train.loc[idx, "loss"].to_numpy(dtype=float)
    infl_arr = np.abs(np.asarray([influence[i] for i in idx], dtype=float))
    asr_arr = asr_full.loc[idx, ATTACK].to_numpy(dtype=float)
    return loss_arr, infl_arr, asr_arr


def _draw_scatter(ax, loss, infl, asr, *, vmin, vmax, marker_size=3.5):
    """Scatter a single panel; high-ASR points drawn last (on top)."""
    order = np.argsort(asr)  # low ASR first, high last -> drawn on top
    sc = ax.scatter(
        loss[order], infl[order], c=asr[order],
        s=marker_size, cmap=LIRA_CMAP,
        norm=Normalize(vmin=vmin, vmax=vmax),
        linewidths=0, alpha=0.5,
        rasterized=True,
    )
    # Heavy-tailed scores: log axes with a small floor on positives.
    pos_loss = loss[loss > 0]
    pos_infl = infl[infl > 0]
    if pos_loss.size and pos_infl.size:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(pos_loss.min() * 0.5, loss.max() * 1.5)
        ax.set_ylim(pos_infl.min() * 0.5, infl.max() * 1.5)
    return sc


def _global_asr_range(bundles):
    """Shared colour limits across panels so colours are comparable.

    LiRA ASR is in [0, 1] in principle; we still compute robust limits in case
    the per-sample CSV stores something narrower or rescaled.
    """
    vals = np.concatenate([b[2] for b in bundles.values()])
    return float(np.nanmin(vals)), float(np.nanmax(vals))


def _make_grid(bundles, vmin, vmax) -> None:
    """2x4 grid figure -- mirrors fig_topk_recall.py's layout.

    Width is set to NeurIPS \\textwidth (5.5 in) so the figure can be dropped
    into the paper with ``\\includegraphics{...}`` (no scaling).
    """
    FIG_WIDTH, FIG_HEIGHT = 5.5, 2.9
    fig, axes = plt.subplots(NROWS, NCOLS, figsize=(FIG_WIDTH, FIG_HEIGHT),
                             constrained_layout=False)
    fig.subplots_adjust(left=0.085, right=0.88, top=0.9, bottom=0.13,
                        wspace=0.35, hspace=0.6)

    last_sc = None
    for di, (dataset, _label, _) in enumerate(DATASETS):
        ax = axes[di // NCOLS, di % NCOLS]
        if dataset not in bundles:
            ax.set_title(PANEL_TITLES.get(dataset, dataset) +
                         r"\\(no data)", fontsize=8)
            ax.set_axis_off()
            continue
        loss, infl, asr = bundles[dataset]
        last_sc = _draw_scatter(ax, loss, infl, asr,
                                vmin=vmin, vmax=vmax, marker_size=2.0)
        ax.set_title(PANEL_TITLES.get(dataset, dataset), fontsize=8, pad=1.5)
        ax.tick_params(which="major", length=2.5, width=0.5,
                       direction="in", pad=2.0)
        ax.tick_params(which="minor", length=1.4, width=0.35,
                       direction="in")
        # Per-panel grid is fine to read the heavy-tailed log axes.
        ax.grid(True, which="major", axis="y", linewidth=0.3, alpha=0.4,
                linestyle="--")

    for k in range(len(DATASETS), NROWS * NCOLS):
        axes[k // NCOLS, k % NCOLS].set_axis_off()

    fig.supxlabel(r"local loss $\ell_i$", fontsize=9, y=0.005)
    fig.supylabel(r"$\text{Influence}_i$", fontsize=9, x=0.001)

    if last_sc is not None:
        cax = fig.add_axes([0.91, 0.18, 0.018, 0.66])
        # Build a fresh mappable so the colorbar shows the cmap at full
        # opacity, independent of the alpha=0.5 used on the scatter points.
        mappable = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax),
                                  cmap=LIRA_CMAP)
        mappable.set_array([])
        cbar = fig.colorbar(mappable, cax=cax)
        cbar.set_label(r"LiRA ASR", fontsize=8)
        cbar.ax.tick_params(labelsize=7)
        cbar.outline.set_linewidth(0.4)

    out_path = OUT / f"scatter_loss_influence_{ATTACK}.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


def _make_individual(dataset, loss, infl, asr, vmin, vmax) -> None:
    # Half-textwidth (2.7 in) so a pair of these fits side-by-side on a
    # NeurIPS page; full width is 5.5 in.
    fig, ax = plt.subplots(figsize=(2.7, 2.4), constrained_layout=False)
    fig.subplots_adjust(left=0.22, right=0.80, top=0.86, bottom=0.18)

    _draw_scatter(ax, loss, infl, asr,
                  vmin=vmin, vmax=vmax, marker_size=6.0)
    ax.set_title(PANEL_TITLES.get(dataset, dataset).replace("\n", r"\\"),
                 fontsize=9, pad=4.0)
    ax.set_xlabel(r"local loss $\ell_i$", fontsize=9)
    ax.set_ylabel(r"$|\text{Influence}_i|$", fontsize=9)
    ax.grid(True, which="major", axis="both", linewidth=0.3, alpha=0.4,
            linestyle="--")
    ax.tick_params(which="major", length=2.8, width=0.5, direction="in", pad=2.5)
    ax.tick_params(which="minor", length=1.6, width=0.35, direction="in")

    mappable = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax),
                              cmap=LIRA_CMAP)
    mappable.set_array([])
    cbar = fig.colorbar(mappable, ax=ax, fraction=0.05, pad=0.03)
    cbar.set_label(r"LiRA ASR", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    cbar.outline.set_linewidth(0.4)

    out_path = OUT / f"scatter_{dataset}_{ATTACK}.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    apply_style()
    plt.rcParams["figure.constrained_layout.use"] = False

    bundles: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for dataset, _label, _ in DATASETS:
        b = _load_dataset(dataset)
        if b is None:
            print(f"  [{dataset}] skipped (missing files / too few samples)")
            continue
        bundles[dataset] = b
        print(f"  [{dataset}] {len(b[0])} samples")

    if not bundles:
        print("no datasets loaded; aborting")
        return

    vmin, vmax = _global_asr_range(bundles)
    print(f"shared LiRA ASR range: [{vmin:.3f}, {vmax:.3f}]")

    _make_grid(bundles, vmin, vmax)
    for dataset, (loss, infl, asr) in bundles.items():
        _make_individual(dataset, loss, infl, asr, vmin, vmax)


if __name__ == "__main__":
    main()
