from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ncx2
from sklearn.metrics import roc_auc_score, roc_curve

from style import PALETTE, apply_style  # noqa: E402

OUT = Path("figures/pdf/mia_gaussian")
OUT.mkdir(parents=True, exist_ok=True)

# NeurIPS textwidth = 5.5 inches.
TEXTWIDTH = 5.5
HALF      = TEXTWIDTH / 2.0      # 2.75 in
THIRD     = TEXTWIDTH / 3.0      # 1.833 in

C_MEMBER    = PALETTE["lira"]    # red  (#d62728)
C_NONMEMBER = PALETTE["h_ii"]    # blue (#1f77b4)

N_DF = 50
N_SAMPLES = 100_000

def sample_losses(tau, mu, sigma, n_minus_p, n_samples, rng):
    """Joint draw of (ell_member, ell_nonmember) per the corollary.
    Returns arrays of shape (n_samples,)."""
    V_in = rng.chisquare(df=n_minus_p, size=n_samples)
    Z_in = rng.standard_normal(n_samples)
    
    V_out = rng.chisquare(df=n_minus_p, size=n_samples)
    Z_out = rng.standard_normal(n_samples)
    l_in   = (V_in / (V_in + tau)) ** 2 * (mu + sigma * tau / np.sqrt(V_in) * Z_in) ** 2          #
    l_out = (mu + sigma * tau / np.sqrt(V_out) * Z_out)**2
    return l_in, l_out


def auc_from_losses(ell_member, ell_nonmember):
    """AUC of the optimal threshold attack on -loss (members have smaller loss)."""
    y = np.concatenate([np.ones_like(ell_member), np.zeros_like(ell_nonmember)])
    s = np.concatenate([-ell_member, -ell_nonmember])
    return roc_auc_score(y, s)


def asr_from_losses(ell_member, ell_nonmember):
    """Balanced accuracy at the optimal threshold of the loss attack.

    Equivalent to 1 - 0.5 * min_t (alpha(t) + beta(t)).
    """
    y = np.concatenate([np.ones_like(ell_member), np.zeros_like(ell_nonmember)])
    s = np.concatenate([-ell_member, -ell_nonmember])
    fpr, tpr, _ = roc_curve(y, s)
    return float(((tpr - fpr + 1.0) / 2.0).max())

def gen_gap_from_losses(ell_member, ell_nonmember):
    return np.mean(ell_nonmember) - np.mean(ell_member)



def th_figure(rng):

    fig, axes = plt.subplots(figsize=(TEXTWIDTH, THIRD*0.9), layout="constrained", ncols=3)
    fig.get_layout_engine().set(w_pad=0.03, h_pad=0, 
                            wspace=0.1, hspace=0, rect=(0, 0, 1, 1))
    tau_grid = np.linspace(0, 100, 20)
    snr_grid = np.linspace(0, 15, 20)**2

    gen_grid = np.empty((snr_grid.size, tau_grid.size))
    for i, snr in enumerate(snr_grid):
        for j, tau in enumerate(tau_grid):
            ell_m, ell_n = sample_losses(tau, np.sqrt(snr), 1.0, 100, 10_000, rng)
            gen_grid[i, j] = gen_gap_from_losses(ell_m, ell_n)
    ax = axes[0]
    im = ax.pcolormesh(tau_grid, snr_grid, gen_grid,
                        shading="auto", cmap="viridis", rasterized=True)

    ax.grid(False)
    ax.set_xlim(0)
    ax.set_ylim(0)
    ax.set_xlabel(r"Population leverage $\bar{h}_{ii}$", fontsize=8)
    ax.set_ylabel(r"Squared error $\varepsilon_i^2$", fontsize=8)
    ax.set_box_aspect(1)                         
    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0)

    cbar.ax.set_title(r"$\mathbb{E}[\Delta \ell_i]$", pad=5, fontsize=8)
    cbar.outline.set_linewidth(0.5)
    cs = ax.contour(tau_grid, snr_grid, gen_grid,
                    levels=[10, 50, 100],
                    colors="white", linewidths=0.3)

    ax.clabel(cs, levels=[10, 50, 100], manual=[(0.5, 50), (50, 100), (70, 150)], fontsize=6)

    tau, sigma = 1.0, 1.0
    mus = [0.0, 0.5, 1.0, 5.0, 10.0]
    ax = axes[1]
    cmap = plt.get_cmap("viridis")
    for k, mu in enumerate(mus):
        ell_m, ell_n = sample_losses(tau, mu, sigma, N_DF, N_SAMPLES, rng)
        y = np.concatenate([np.ones_like(ell_m), np.zeros_like(ell_n)])
        s = np.concatenate([-ell_m, -ell_n])
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, 1 - tpr, color=cmap(k / max(1, len(mus) - 1)),
                label=rf"$\varepsilon_i={mu:g}$")
    ax.plot([1, 0], [0, 1], color="0.6", lw=0.7, ls="--")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel(r"FPR $(\alpha_i)$", fontsize=8)
    ax.set_ylabel(r"FNR $(\beta_i)$", fontsize=8)
    leg = ax.legend(loc="upper right", frameon=False, fancybox=False,
                    edgecolor="0.4", fontsize=8, bbox_to_anchor=(1.05, 1.05), borderaxespad=0, handlelength=1.2, labelspacing=0.3)

    ax.text(0.62, 1.08, r"($\bar{h}_{ii}$ fixed)", fontsize=8)

    

    ax = axes[2]
    mu, sigma = 1.0, 1.0
    taus = [5, 10, 50, 100, 500]

    cmap = plt.get_cmap("viridis")
    for k, tau in enumerate(taus):
        ell_m, ell_n = sample_losses(tau, mu, sigma, N_DF, N_SAMPLES, rng)
        y = np.concatenate([np.ones_like(ell_m), np.zeros_like(ell_n)])
        s = np.concatenate([-ell_m, -ell_n])
        fpr, tpr, _ = roc_curve(y, s)
        auc = roc_auc_score(y, s)
        ax.plot(fpr, 1 -  tpr, color=cmap(k / max(1, len(taus) - 1)), 
            label=rf"$\bar{{h}}_{{ii}}$={tau:g}")
    ax.plot([1, 0], [0, 1], color="0.6", lw=0.7, ls="--")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_xlabel(r"FPR $(\alpha_i)$", fontsize=8)
    ax.set_ylabel(r"FNR $(\beta_i)$", fontsize=8)

    leg = ax.legend(loc="upper right", frameon=False, fancybox=False,
                    edgecolor="0.4", fontsize=8, bbox_to_anchor=(1.05, 1.05), borderaxespad=0, handlelength=1.2, labelspacing=0.3)

    ax.text(0.6, 1.08, r"($\varepsilon_i$ fixed)", fontsize=8)
    out = OUT / "toy_model.pdf"
    fig.savefig(out)

    plt.close(fig)
    print(f"wrote {out}")

def main():
    rng = np.random.default_rng(0)
    apply_style()
    th_figure(rng)


if __name__ == "__main__":
    main()
