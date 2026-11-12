"""Run the full paper-figure pipeline.

Order:
    1. aggregate.py           -> writes figures/aggregated/{topk_recall.npz,
                                 spearman_long.csv, meta.json}.
    2. fig_topk_recall.py     -> figures/pdf/topk_recall/topk_recall_<attack>.pdf
    3. fig_spearman.py        -> figures/pdf/spearman/spearman_<attack>.pdf
    4. fig_summary_heatmap.py -> figures/pdf/summary/spearman_heatmap.pdf
    5. fig_linear_toy.py      -> figures/pdf/toy/linear_toy.pdf
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = [
    "aggregate.py",
    "fig_topk_recall.py",
    "fig_spearman.py",
    "fig_summary_heatmap.py",
    "fig_th_plot.py"
]


def main() -> None:
    for s in SCRIPTS:
        print(f"\n========== {s} ==========")
        rc = subprocess.call([sys.executable, str(HERE / s)])
        if rc != 0:
            sys.exit(rc)


if __name__ == "__main__":
    main()
