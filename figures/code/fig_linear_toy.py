"""Linear-regression toy: exact LOO gap vs. Newton/ALO and Influence.

Two-panel figure:

    (a) Single noise realisation. We sweep the target leverage $h_{ii}$ on a
        grid, build a (n-1)-design with the chosen $h_{ii}$, draw labels with
        Gaussian noise of fixed magnitude, and report
        $\\ell_{-i} - \\ell_i$ for every grid point. This shows that on a
        single seed the empirical gap is noisy.

    (b) Many noise realisations (n_seeds = 200). Solid lines / shaded bands are
        mean and $\\pm 1$ std of the simulated gap and the two approximations
        across seeds. Across realisations, the empirical gap concentrates on
        the closed-form curve.

Theoretical curves overlay:
    Exact:      $\\ell_i\\,(2h_{ii}-h_{ii}^2)/(1-h_{ii})^2$
    Newton/ALO: $\\ell_i\\,h_{ii}/(1-h_{ii})$
    Influence:  $\\ell_i\\,h_{ii}$

with $\\ell_i$ replaced by its expected value $(1-h_{ii})^2$ under the OLS
geometry used in the sim.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from style import PALETTE, apply_style

HERE = Path(__file__).resolve().parent
FIG = HERE.parent

OUT = FIG / "pdf" / "toy"
OUT.mkdir(parents=True, exist_ok=True)


def simulate(n_seeds: int, h_grid: np.ndarray, *, n_minus_i: int = 199, p: int = 10,
             sigma: float = 0.5, seed0: int = 2026) -> dict[str, np.ndarray]:
    """Return arrays of shape (n_seeds, n_h) for {gap, newton, influence}."""
    rng = np.random.default_rng(seed0)

    gap = np.empty((n_seeds, h_grid.size))
    new = np.empty((n_seeds, h_grid.size))
    inf = np.empty((n_seeds, h_grid.size))

    for s in range(n_seeds):
        # Random orthonormal columns for X_{-i}.
        A = rng.standard_normal((n_minus_i, p))
        Q, _ = np.linalg.qr(A)
        X_minus_i = Q
        theta = rng.standard_normal(p)
        eps = rng.standard_normal(n_minus_i) * sigma
        y_minus_i = X_minus_i @ theta + eps

        for j, h_target in enumerate(h_grid):
            a = np.sqrt(h_target / (1.0 - h_target))
            x_i = np.zeros(p)
            x_i[0] = a
            y_i = float(x_i @ theta) + 1.0  # injected residual = 1

            X_full = np.vstack([X_minus_i, x_i])
            y_full = np.concatenate([y_minus_i, [y_i]])

            beta_full = np.linalg.solve(X_full.T @ X_full, X_full.T @ y_full)
            beta_loo = np.linalg.solve(X_minus_i.T @ X_minus_i, X_minus_i.T @ y_minus_i)

            r_i = y_i - x_i @ beta_full
            ell_i = r_i ** 2
            ell_minus = (y_i - x_i @ beta_loo) ** 2

            H_inv = np.linalg.inv(X_full.T @ X_full)
            h_ii_emp = float(x_i @ H_inv @ x_i)

            gap[s, j] = ell_minus - ell_i
            new[s, j] = ell_i * h_ii_emp / (1.0 - h_ii_emp)
            inf[s, j] = ell_i * h_ii_emp

    return {"gap": gap, "newton": new, "influence": inf}


def panel(ax, h_grid, sim, h_smooth, *, mode: str) -> None:
    """Plot the single-seed (mode='single') or multi-seed (mode='mean') panel."""
    ell_i_expected = (1 - h_smooth) ** 2
    exact_curve  = ell_i_expected * (2 * h_smooth - h_smooth ** 2) / (1 - h_smooth) ** 2
    newton_curve = ell_i_expected * h_smooth / (1.0 - h_smooth)
    inf_curve    = ell_i_expected * h_smooth

    # Theoretical curves (always)
    ax.plot(h_smooth, exact_curve,  "-",  color=PALETTE["exact"],     linewidth=1.3,
            label=r"Exact $\ell_i(2h_{ii}-h_{ii}^2)/(1-h_{ii})^2$")
    ax.plot(h_smooth, newton_curve, "-",  color=PALETTE["newton"],    linewidth=1.3,
            label=r"Newton/ALO $\ell_i h_{ii}/(1-h_{ii})$")
    ax.plot(h_smooth, inf_curve,    "--", color=PALETTE["influence"], linewidth=1.3,
            label=r"Influence $\ell_i h_{ii}$")

    if mode == "single":
        # Single seed: show open-circle markers
        ax.plot(h_grid, sim["gap"][0],       "o", color=PALETTE["exact"],
                markersize=2.7, markerfacecolor="white",
                markeredgecolor=PALETTE["exact"], markeredgewidth=0.7,
                label=r"Sim. $\ell_{-i}-\ell_i$")
        ax.plot(h_grid, sim["newton"][0],    "o", color=PALETTE["newton"],
                markersize=2.7, markerfacecolor="white",
                markeredgecolor=PALETTE["newton"], markeredgewidth=0.7)
        ax.plot(h_grid, sim["influence"][0], "o", color=PALETTE["influence"],
                markersize=2.7, markerfacecolor="white",
                markeredgecolor=PALETTE["influence"], markeredgewidth=0.7)
        ax.set_title(r"(a) Single noise realisation")
    else:
        # Many seeds: mean ± std bands, no markers (means already match curves).
        for key, colour in [("gap",       PALETTE["exact"]),
                            ("newton",    PALETTE["newton"]),
                            ("influence", PALETTE["influence"])]:
            mean = sim[key].mean(axis=0)
            std  = sim[key].std(axis=0, ddof=1)
            ax.fill_between(h_grid, mean - std, mean + std,
                            color=colour, alpha=0.15, linewidth=0)
            ax.plot(h_grid, mean, "o", color=colour, markersize=2.4,
                    markerfacecolor="white", markeredgecolor=colour,
                    markeredgewidth=0.7)
        ax.set_title(rf"(b) Mean $\pm 1$ std over {sim['gap'].shape[0]} realisations")

    ax.set_xlabel(r"Leverage score $h_{ii}$")
    ax.set_xlim(0, 1)
    ax.set_ylim(-2, 8)


def main() -> None:
    apply_style()
    h_grid   = np.linspace(0.05, 0.92, 30)
    h_smooth = np.linspace(0.005, 0.95, 400)

    sim_single = simulate(n_seeds=1,   h_grid=h_grid)
    sim_many   = simulate(n_seeds=200, h_grid=h_grid)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4), sharey=True)

    panel(axes[0], h_grid, sim_single, h_smooth, mode="single")
    panel(axes[1], h_grid, sim_many,   h_smooth, mode="mean")

    axes[0].set_ylabel(r"$\ell_{-i}-\ell_i$")

    # Legend in box, only on first axis (curves are shared).
    handles, labels = axes[0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, ncol=4,
                     loc="lower center", bbox_to_anchor=(0.5, 1.0),
                     frameon=True, fancybox=False, edgecolor="0.4",
                     handletextpad=0.5, columnspacing=1.0)
    leg.get_frame().set_linewidth(0.5)

    out_path = OUT / "linear_toy.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
