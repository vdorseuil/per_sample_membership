"""Top-10 most-leveraged training samples per image dataset (model_210).

Produces a single NeurIPS-column-width PDF with two stacked sections
(Newton on top, Influence on bottom), each section being
``len(IMAGE_DATASETS) x TOP_K`` cells:

    figures/pdf/top_samples_images.pdf

BERT samples (sentences) are emitted as a separate LaTeX table since they
do not share the visual layout — see ``make_bert_top_table.py``.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets as tv_datasets

from style import apply_style

HERE = Path(__file__).resolve().parent
FIG = HERE.parent
REPO = FIG.parent

TGT = REPO / "experiments" / "target_models"
OUT = FIG / "pdf"
OUT.mkdir(parents=True, exist_ok=True)

DEFAULT_MODEL = "model_210"
TOP_K = 10

IMAGE_DATASETS = [
    ("BloodMnist", r"BloodMNIST", None),
    ("Cifar10",    r"CIFAR-10",   None),
    ("Cifar100",   r"CIFAR-100",  None),
]

SURROGATES = [("newton", "Newton"), ("influence", "Influence")]


def latest_run(model_dir: Path) -> Path:
    base = model_dir / "leverage_score"
    cands = [r for r in base.iterdir()
             if r.is_dir() and (r / "leverage_newton.pkl").exists()
             and (r / "leverage_influence.pkl").exists()]
    return sorted(cands)[-1]


def top_indices(scores: dict[int, float], k: int) -> list[int]:
    items = [(i, abs(v)) for i, v in scores.items() if np.isfinite(v)]
    items.sort(key=lambda t: -t[1])
    return [i for i, _ in items[:k]]


_CACHE: dict[str, np.ndarray] = {}


def get_images(key: str) -> np.ndarray:
    if key in _CACHE:
        return _CACHE[key]
    if key == "BloodMnist":
        # 224x224 RGB microscopy images stored as a single MedMNIST npz.
        npz = np.load(str(REPO / "data" / "BloodMNIST" / "bloodmnist_224.npz"))
        arr = npz["train_images"]
    elif key == "Cifar10":
        ds = tv_datasets.CIFAR10(root=str(REPO / "data" / "Cifar10"), train=True, download=False)
        arr = ds.data
    elif key == "Cifar100":
        ds = tv_datasets.CIFAR100(root=str(REPO / "data" / "Cifar100"), train=True, download=False)
        arr = ds.data
    else:
        raise KeyError(key)
    _CACHE[key] = arr
    return arr


def main() -> None:
    apply_style()
    plt.rcParams.update({"axes.grid": False, "figure.constrained_layout.use": False})

    n_ds = len(IMAGE_DATASETS)
    n_surr = len(SURROGATES)
    n_rows = n_ds * n_surr  # 6
    n_cols = TOP_K           # 10

    cell = 0.50  # inches
    fig_w = 5.5
    fig_h = cell * n_rows + 0.30  # add room for top section title
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(fig_w, fig_h),
    )
    fig.subplots_adjust(left=0.135, right=0.995, top=0.93, bottom=0.005,
                        wspace=0.05, hspace=0.10)

    for si, (sname, slabel) in enumerate(SURROGATES):
        for di, (key, dlabel, model_override) in enumerate(IMAGE_DATASETS):
            row = si * n_ds + di
            model = model_override or DEFAULT_MODEL
            run = latest_run(TGT / key / "outputs" / model)
            with open(run / f"leverage_{sname}.pkl", "rb") as fh:
                scores = pickle.load(fh)
            top = top_indices(scores, TOP_K)
            imgs = get_images(key)
            cmap = None
            for c in range(n_cols):
                ax = axes[row, c]
                ax.imshow(imgs[top[c]], cmap=cmap, interpolation="nearest")
                ax.set_xticks([]); ax.set_yticks([])
                if c == 0:
                    ax.annotate(dlabel,
                                xy=(-0.10, 0.5), xycoords="axes fraction",
                                ha="right", va="center",
                                fontsize=6.5, family="serif")

        # Section label on the left margin spanning the surrogate's rows.
        # Compute y-center of the section in figure coordinates.
        top_row_ax = axes[si * n_ds, 0]
        bot_row_ax = axes[si * n_ds + n_ds - 1, 0]
        bbox_top = top_row_ax.get_position()
        bbox_bot = bot_row_ax.get_position()
        y_center = 0.5 * (bbox_top.y1 + bbox_bot.y0)
        fig.text(0.012, y_center, slabel,
                 ha="left", va="center", rotation=90,
                 fontsize=8, family="serif", weight="bold")

    # Section divider line between Newton and Influence groups.
    div_ax_top = axes[n_ds - 1, 0].get_position()
    div_ax_bot = axes[n_ds, 0].get_position()
    y_div = 0.5 * (div_ax_top.y0 + div_ax_bot.y1)
    fig.lines.append(plt.Line2D([0.135, 0.995], [y_div, y_div],
                                transform=fig.transFigure,
                                color="#888888", linewidth=0.4))

    out = OUT / "top_samples_images.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
