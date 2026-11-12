"""Shared matplotlib style for paper figures.

Lean, latexified figures: only horizontal grid, legend boxed, Computer Modern
serif rendered through real LaTeX (text.usetex=True). All figure scripts call
``apply_style()`` first.
"""
from __future__ import annotations

import matplotlib as mpl


RC = {
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{bm}",
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "lines.linewidth": 1.0,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
    # Y-axis grid only (no vertical grid).
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.linewidth": 0.35,
    "grid.alpha": 0.4,
    "grid.linestyle": "--",
    "figure.constrained_layout.use": True,
}


def apply_style() -> None:
    mpl.rcParams.update(RC)


# Colour-blind-friendly palette, kept consistent with v1 where possible.
PALETTE = {
    "loss":          "#7f7f7f",
    "grad":          "#e8cf14",
    "exact":         "#000000",
    "newton":        "#17becf",
    "influence":     "#2ca02c",
    "h_ii":          "#1f77b4",
    "random":        "#cccccc",
    "lira":          "#d62728",
    "rmia":          "#ff7f0e",
    "shokri":        "#9467bd",
}


# Order, display name, model id, has_h_ii flag.
DATASETS = [
    ("Bert",                     r"SST2 (BERT, full ft.)",        True),
    ("Bert_freeze",              r"SST2 (BERT, linear probe)",    True),
    ("CaliforniaHousing_linear", r"CalHousing (ridge)",     True),
    ("CaliforniaHousing_mlp",    r"CalHousing (MLP)",       True),
    ("Cifar10",                  r"CIFAR-10 (CNN)",         True),
    ("Cifar100",                 r"CIFAR-100 (ResNet-20)",  True),
    ("BloodMnist",               r"BloodMNIST (DINOv2)",    True),
    ("UCI_energy",               r"UCI Energy (tab.\ MLP)", True),
]

REGRESSION_DATASETS = {"CaliforniaHousing_linear", "CaliforniaHousing_mlp", "UCI_energy"}


# Models considered in the average.
MODEL_IDS = [f"model_{i}" for i in range(200, 216)]

# Style reused across surrogates: (key, label, colour).
SURROGATE_LABELS = {
    "loss":      r"Loss $\ell_i$",
    "grad":      r"$\|\nabla\ell_i\|$",
    "exact":     r"$\ell_i\,(2 h_{ii}-h_{ii}^2)/(1-h_{ii})^2$",
    "newton":    r"Newton",
    "influence": r"Influence",
    "random":    r"Random",
}


ATTACK_DISPLAY = {
    "lira":   "LiRA",
    "rmia":   "RMIA",
    "shokri": "Shokri",
}
