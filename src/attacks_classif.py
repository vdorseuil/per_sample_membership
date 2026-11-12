import sys
sys.modules['xformers'] = None

import os
import time
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from joblib import Parallel, delayed
import pickle
import itertools
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as mcm
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from helpers import log_likelihood_gaussian


ATTACKS = {
    "loss":           lambda df, **kw: loss_attack(df),
    "self_entropy":   lambda df, **kw: self_entropy_attack(df),
    "gradient_norm":  lambda df, **kw: gradient_norm_attack(df),
    "lira":           lambda df, **kw: lira_attack(df,    kw["lira_stats"]),
    "rmia":           lambda df, **kw: rmia_attack(df,    kw["rmia_stats"]),
    "shokri":         lambda df, **kw: shokri_attack(df,  kw["shokri_clfs"]),
}

_FPR_GRID = np.concatenate([
    np.linspace(0.0,   0.001,  50,  endpoint=False),
    np.linspace(0.001, 0.01,  100,  endpoint=False),
    np.linspace(0.01,  0.1,   200,  endpoint=False),
    np.linspace(0.1,   1.0,   200,  endpoint=True),
])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir",   type=str, required=True,
                        help="Directory where per-model results are written.")
    parser.add_argument("--shadow_path",  type=str, required=True,
                        help="Directory that contains pre-computed shadow-model statistics "
                             "(lira_stats.csv, rmia_stats.csv, shokri_classifiers.pkl).")
    parser.add_argument("--target_path",  type=str, required=True,
                        help="Directory that contains the target model sub-directories.")
    parser.add_argument("--no_asr", action="store_true", default=False, help="Disable the computation of the ASR and top-k")
    parser.add_argument("--num_classes",  type=int, default=10)
    # parser.add_argument("--no_shokri",  type=bool, default=True,action="store_false", help="Disable the computing of the Shokri attack")
    return parser.parse_args()




def compute_metrics(scores: np.ndarray, labels: np.ndarray) -> dict:
    """
    Compute ROC-based metrics.

    Parameters
    ----------
    scores : float array, higher = more likely to be a member.
    labels : int array, 1 = member (train), 0 = non-member (test).

    Returns
    -------
    dict with keys: auc, tpr_at_1fpr, tpr_at_5fpr, advantage, fpr, tpr, thresholds
    """
    fpr, tpr, thresholds = roc_curve(labels, scores)
    auc = roc_auc_score(labels, scores)

    tpr_at_1fpr  = float(np.interp(0.01, fpr, tpr))   # TPR @ FPR = 1 %
    tpr_at_5fpr  = float(np.interp(0.05, fpr, tpr))   # TPR @ FPR = 5 %
    advantage    = float(np.max(tpr - fpr))            # max_{threshold}(TPR - FPR)

    return dict(
        auc=auc,
        tpr_at_1fpr=tpr_at_1fpr,
        tpr_at_5fpr=tpr_at_5fpr,
        advantage=advantage,
        fpr=fpr,
        tpr=tpr,
        thresholds=thresholds,
    )


def save_results(metrics: dict, attack_name: str, model_name: str,
                 output_dir: Path) -> None:
    """
    Persist the alpha-beta (ROC) plot and scalar metrics for one
    (attack, target-model) pair.
    """
    model_out = output_dir / model_name / attack_name
    model_out.mkdir(parents=True, exist_ok=True)

    # -- ROC / alpha-beta plot ------------------------------------------------
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(metrics["fpr"], metrics["tpr"],
            lw=2, color="steelblue",
            label=f"AUC = {metrics['auc']:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")
    ax.set_xlabel("FPR  (α – false positive rate)")
    ax.set_ylabel("TPR  (β – true positive rate)")
    ax.set_title(f"{attack_name}  ·  {model_name}", fontsize=10)
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(model_out / "roc_curve.pdf", dpi=150)
    plt.close(fig)

    # -- Scalar metrics -------------------------------------------------------
    pd.DataFrame([{
        "auc":          metrics["auc"],
        "tpr_at_1fpr":  metrics["tpr_at_1fpr"],
        "tpr_at_5fpr":  metrics["tpr_at_5fpr"],
        "advantage":    metrics["advantage"],
    }]).to_csv(model_out / "metrics.csv", index=False)

    print(
        f"  [{attack_name:16s}]  "
        f"AUC={metrics['auc']:.4f}  "
        f"TPR@1%FPR={metrics['tpr_at_1fpr']:.4f}  "
        f"TPR@5%FPR={metrics['tpr_at_5fpr']:.4f}  "
        f"Adv={metrics['advantage']:.4f}"
    )


def loss_attack(df: pd.DataFrame):
    """
    Baseline: members tend to have lower cross-entropy loss.
    Score = –loss  (higher ↔ more likely member).
    """
    scores = -df["loss"].values
    labels = (df["split"] == "train").astype(int).values
    return scores, labels


def self_entropy_attack(df: pd.DataFrame):
    """
    Prediction entropy: members → confident predictions → low entropy.
    Score = –H(p)  (higher ↔ more likely member).
    """
    log_proba = np.stack(df["log_proba"].values)            # (N, C)
    proba     = np.exp(np.clip(log_proba, -100, 0))
    proba    /= proba.sum(axis=1, keepdims=True)            # re-normalise for safety
    entropy   = -(proba * np.log(proba + 1e-12)).sum(axis=1)
    scores    = -entropy
    labels    = (df["split"] == "train").astype(int).values
    return scores, labels


def gradient_norm_attack(df: pd.DataFrame):
    """
    Members → small gradient norms (model has already fit them).
    Score = –grad_norm  (higher ↔ more likely member).
    """
    scores = -df["local_grad_norm"].values
    labels = (df["split"] == "train").astype(int).values
    return scores, labels


def lira_attack(df: pd.DataFrame, lira_stats: pd.DataFrame):
    """
    LiRA – "Membership Inference Attacks From First Principles" (Carlini et al., 2022).

    For each sample i we compute the log-likelihood ratio:
        score_i = log p(phi_i | IN ; mu_in, sigma_in)
                – log p(phi_i | OUT; mu_out, sigma_out)

    where phi = logit(p_y / (1 – p_y)) is the pre-computed confidence
    transformation stored in the metrics dataframe.

    The Gaussian parameters (mu_in/out, sigma_in/out) were estimated from
    shadow models by compute_attacks_statistics.py::compute_lira_stats().
    """
    phi       = df["phi"].values
    indices   = df["idx"].values.astype(int)

    stats_idx = lira_stats.set_index("idx")

    scores = np.empty(len(df))
    for i, idx in enumerate(indices):
        row     = stats_idx.loc[idx]
        scores[i] = log_likelihood_gaussian(phi[i], row["mu_out"],  row["sigma_out"], row["mu_in"], row["sigma_in"])

    labels = (df["split"] == "train").astype(int).values
    return scores, labels


def rmia_attack(df: pd.DataFrame, rmia_stats: pd.DataFrame, gamma: float = 1.0):
    """
    RMIA membership inference attack (Zarifzadeh et al., ICML 2024).

    For each sample x, the score is the fraction of population points z
    (non-members, i.e. the target's test split) for which the pairwise
    likelihood ratio exceeds gamma:

        Score(x) = mean_z 1{ [Pr(x|θ)/Pr̂(x)] / [Pr(z|θ)/Pr̂(z)] >= gamma }

    where Pr(.|θ) = exp(-loss) from the target model and Pr̂(.) is the
    reference-model average probability (avg_prob in rmia_stats).

    Parameters
    ----------
    df         : target-model metrics (idx, split, loss, ...)
    rmia_stats : reference-model statistics (idx, avg_prob)
    gamma      : likelihood-ratio threshold (> 1)

    Returns
    -------
    scores : float array in [0, 1], higher = more likely member
    labels : int array, 1 = member (train), 0 = non-member (test)
    """
   
    # Step 1: Compute Pr(x|theta) / Pr(x)
    prob_target_x = np.exp(-df["loss"].values) #Pr(x|theta), shape (N,)
    indices     = df["idx"].values.astype(int)
    splits      = df["split"].values

    ref_map  = rmia_stats.set_index("idx")["avg_prob"].to_dict()
    prob_ref_x = np.array([ref_map.get(idx, 1e-10) for idx in indices])

    ratio_x = prob_target_x / (prob_ref_x + 1e-12) # Ratio_x = Pr(x|theta) / Pr(x), shape (N,)

    # Step2: Compute Pr(z|theta) / Pr(z) for multiple z\sim \pi : out of the training set.
    pop_mask = splits == "test"
    if pop_mask.sum() < 1: # fallback: use all samples
        pop_mask = np.ones(len(df), dtype=bool)
        
    ratio_z = np.sort(ratio_x[pop_mask])  # Ratio_z = Pr(z|theta) / Pr(z), shape (|Z|,)
    
    
    inv_ratio_z = 1.0 / (ratio_z + 1e-12)
    attack_scores = np.searchsorted(ratio_z, ratio_x / gamma, side="left") / len(ratio_z)
    labels = (splits == "train").astype(int)
    return attack_scores, labels



def shokri_attack(df, shokri_clfs):
    log_proba = np.vstack(df["log_proba"].values)
    proba = np.exp(np.clip(log_proba, -100, 0))
    class_labels = df["label"].values.astype(int)
    
    scores = np.full(len(df), 0.5)
    unique_classes = [c for c in np.unique(class_labels) if c in shokri_clfs]

    tasks = []
    for c in unique_classes:
        mask = (class_labels == c)
        tasks.append((c, mask, shokri_clfs[c], proba[mask]))

    def worker(c, mask, clf, proba_sub):
        return mask, clf.predict_proba(proba_sub)[:, 1]
    
    n_cores = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
    results = Parallel(n_jobs=n_cores, backend="threading")(
        delayed(worker)(*t) for t in tasks
    )

    for mask, preds in results:
        scores[mask] = preds
        
    labels = (df["split"] == "train").astype(int).values

    return scores, labels
# ---------------------------------------------------------------------------
# Averaged alpha–beta (ROC) curve across all models, one curve per attack
# ---------------------------------------------------------------------------

# Common FPR grid onto which every per-model TPR curve is interpolated before
# averaging.  Piecewise spacing gives dense resolution at low FPR (where
# TPR@1%FPR / TPR@5%FPR are most sensitive) and coarser elsewhere.



def plot_averaged_roc(
    target_path: Path,
    output_dir: Path,
    shared_kwargs: dict,
) -> pd.DataFrame:
    """
    Collect per-model ROC curves for every attack, interpolate onto a shared
    FPR grid, and plot mean ± 1 std TPR as a filled band for each attack on a
    single figure.

    Two plots are produced:
      • linear x-axis  – classic ROC view
      • log    x-axis  – magnifies the low-FPR region with reference lines at
                         FPR = 1 % and FPR = 5 %

    Outputs (written to *output_dir / "averaged_roc/"*)
    -------
    averaged_roc_linear.pdf      – linear-scale plot
    averaged_roc_logx.pdf        – log-scale plot
    averaged_roc_stats.csv       – mean_tpr, std_tpr, n_models per (attack, fpr)
    per_model_tpr.csv            – raw interpolated TPR at 50 coarse FPR points
                                   for every (model, attack) pair

    Parameters
    ----------
    target_path   : directory containing model_* sub-directories
    output_dir    : root results directory  (plots go to output_dir/averaged_roc/)
    shared_kwargs : dict with lira_stats, rmia_stats, shokri_clfs

    Returns
    -------
    pd.DataFrame  – the averaged_roc_stats table (attack × fpr × stats)
    """
    roc_dir = output_dir / "averaged_roc"
    roc_dir.mkdir(parents=True, exist_ok=True)

    attack_names = list(ATTACKS.keys())
    model_dirs   = sorted(
        d for d in target_path.iterdir()
        if d.is_dir() and d.name.startswith("model_")
    )

    # tpr_curves[attack] → list of 1-D arrays (one per model), each length = len(_FPR_GRID)
    tpr_curves : dict[str, list[np.ndarray]] = {a: [] for a in attack_names}
    auc_values : dict[str, list[float]]      = {a: [] for a in attack_names}

    # coarse index for the per-model CSV (keeps file size manageable)
    coarse_idx = np.linspace(0, len(_FPR_GRID) - 1, 50, dtype=int)
    raw_records: list[dict] = []

    print("\n── Collecting ROC curves for averaged plot ──")
    for m_dir in tqdm(model_dirs, desc="Averaged ROC – models"):
        metrics_path = m_dir / "metrics.pkl"
        if not metrics_path.exists():
            continue

        df = pd.read_pickle(metrics_path)

        for attack_name, attack_fn in ATTACKS.items():
            try:
                scores, labels = attack_fn(df, **shared_kwargs)
                if len(np.unique(labels)) < 2:
                    continue

                fpr_raw, tpr_raw, _ = roc_curve(labels, scores)

                # Guarantee boundary points (0,0) and (1,1) for clean interpolation
                fpr_raw = np.concatenate([[0.0], fpr_raw, [1.0]])
                tpr_raw = np.concatenate([[0.0], tpr_raw, [1.0]])

                tpr_interp = np.interp(_FPR_GRID, fpr_raw, tpr_raw)
                tpr_curves[attack_name].append(tpr_interp)
                auc_values[attack_name].append(roc_auc_score(labels, scores))

                for ci in coarse_idx:
                    raw_records.append({
                        "model":  m_dir.name,
                        "attack": attack_name,
                        "fpr":    float(_FPR_GRID[ci]),
                        "tpr":    float(tpr_interp[ci]),
                    })

            except Exception as exc:
                print(f"    [{attack_name}/{m_dir.name}] ERROR: {exc}")

    # ── Aggregate: mean ± std TPR per attack ─────────────────────────────────
    cmap   = plt.get_cmap("tab10", len(attack_names))
    colors = {a: cmap(i) for i, a in enumerate(attack_names)}

    stat_records: list[dict] = []
    summaries:    dict[str, dict] = {}

    for attack_name in attack_names:
        curves = tpr_curves[attack_name]
        if not curves:
            continue
        mat      = np.stack(curves, axis=0)   # (n_models, n_fpr_grid)
        mean_tpr = mat.mean(axis=0)
        std_tpr  = mat.std(axis=0)
        n        = mat.shape[0]
        aucs     = auc_values[attack_name]

        summaries[attack_name] = dict(
            mean_tpr=mean_tpr,
            std_tpr=std_tpr,
            n=n,
            mean_auc=float(np.mean(aucs)),
            std_auc=float(np.std(aucs)),
        )

        for gi, fpr_val in enumerate(_FPR_GRID):
            stat_records.append({
                "attack":   attack_name,
                "fpr":      float(fpr_val),
                "mean_tpr": float(mean_tpr[gi]),
                "std_tpr":  float(std_tpr[gi]),
                "n_models": n,
            })

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    stats_df = pd.DataFrame(stat_records)
    stats_df.to_csv(roc_dir / "averaged_roc_stats.csv", index=False)
    pd.DataFrame(raw_records).to_csv(roc_dir / "per_model_tpr.csv", index=False)
    print(f"  ✓ Stats CSV     → {roc_dir / 'averaged_roc_stats.csv'}")
    print(f"  ✓ Raw TPR CSV   → {roc_dir / 'per_model_tpr.csv'}")

    # ── Shared drawing helper ─────────────────────────────────────────────────
    def _draw(ax: plt.Axes, xscale: str) -> None:
        for attack_name, s in summaries.items():
            mean_tpr = s["mean_tpr"]
            std_tpr  = s["std_tpr"]
            color    = colors[attack_name]
            label    = (
                f"{attack_name}  "
                f"AUC={s['mean_auc']:.3f}±{s['std_auc']:.3f}  "
                f"(n={s['n']})"
            )
            ax.plot(_FPR_GRID, mean_tpr, color=color, lw=2.0, label=label)
            ax.fill_between(
                _FPR_GRID,
                np.clip(mean_tpr - std_tpr, 0, 1),
                np.clip(mean_tpr + std_tpr, 0, 1),
                color=color, alpha=0.15,
            )

        if xscale == "linear":
            ax.plot([0, 1], [0, 1], "k--", lw=1.0, alpha=0.45, label="Random")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        else:
            log_grid = _FPR_GRID[_FPR_GRID > 0]
            ax.plot(log_grid, log_grid, "k--", lw=1.0, alpha=0.45, label="Random")
            ax.set_xscale("log")
            ax.set_xlim(1e-3, 1)
            ax.set_ylim(0, 1)
            for ref_fpr, ref_label in [(0.01, "FPR=1%"), (0.05, "FPR=5%")]:
                ax.axvline(ref_fpr, color="grey", lw=0.8, linestyle=":", alpha=0.7)
                ax.text(ref_fpr * 1.08, 0.03, ref_label,
                        fontsize=7, color="grey", va="bottom")

        ax.set_xlabel("FPR  (α – false positive rate)", fontsize=11)
        ax.set_ylabel("TPR  (β – true positive rate)", fontsize=11)
        ax.grid(True, which="both", alpha=0.20, linestyle="--")
        ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)

    # ── Produce and save both plots ───────────────────────────────────────────
    n_max = max((s["n"] for s in summaries.values()), default=0)
    for xscale, fname in [("linear", "averaged_roc_linear.pdf"),
                          ("log",    "averaged_roc_logx.pdf")]:
        fig, ax = plt.subplots(figsize=(8, 6.5))
        _draw(ax, xscale)
        ax.set_title(
            f"α–β ROC Curve – All Attacks\n"
            f"(mean ± 1 std,  up to {n_max} models)",
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(roc_dir / fname, dpi=150)
        plt.close(fig)
        print(f"  ✓ Plot ({xscale:6s} x) → {roc_dir / fname}")

    return stats_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _optimal_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    Return the decision threshold that maximises the Youden index (TPR – FPR).
    This is the threshold that achieves the highest attack advantage for a
    given (model, attack) pair.
    """
    fpr, tpr, thresholds = roc_curve(labels, scores)
    best_idx = int(np.argmax(tpr - fpr))
    return float(thresholds[best_idx])


def compute_and_save_asr(
    target_path: Path,
    output_dir: Path,
    shared_kwargs: dict,
) -> pd.DataFrame:
    """
    Compute the per-sample Average Success Rate (ASR) for every attack,
    averaged over all target models that contain each sample.

    Definition
    ----------
    For each (sample i, model m, attack a):
        success(i, m, a) = 1  if  predict(score(i), τ*_ma) == label(i, m)
                           0  otherwise
    where τ*_ma is the optimal Youden threshold for attack a on model m.

    ASR(i, a) = mean_{m : sample i appears in model m} success(i, m, a)

    Interpretation
    --------------
    • ASR ≈ 1   → the sample is consistently and correctly classified as
                  member / non-member by the attack across all models.
    • ASR ≈ 0.5 → the attack performs at chance level for this sample.

    Because the optimal threshold is chosen *after* seeing the scores, ASR
    measures raw discriminability rather than calibration, making it a fair
    cross-attack comparison metric.

    Outputs
    -------
    output_dir / "asr_per_sample.csv"   — index=idx, one column per attack.

    Returns
    -------
    pd.DataFrame  (index=idx, columns=attack_names)
    """
    attack_names = list(ATTACKS.keys())

    model_dirs = sorted(
        d for d in target_path.iterdir()
        if d.is_dir() and d.name.startswith("model_")
    )

    # success_sum[idx][attack]  – cumulative correct predictions
    # attack_appear[idx][attack] – number of models where this (idx, attack) was valid
    success_sum   : dict[int, dict[str, float]] = {}
    attack_appear : dict[int, dict[str, int]]   = {}

    print("\n── Computing per-sample ASR across target models ──")
    for m_dir in tqdm(model_dirs, desc="ASR – target models"):
        metrics_path = m_dir / "metrics.pkl"
        if not metrics_path.exists():
            continue

        df      = pd.read_pickle(metrics_path)
        indices = df["idx"].values.astype(int)

        # Initialise accumulators for any new sample indices
        for idx in indices:
            if idx not in success_sum:
                success_sum[idx]   = {a: 0.0 for a in attack_names}
                attack_appear[idx] = {a: 0   for a in attack_names}

        for attack_name, attack_fn in ATTACKS.items():
            try:
                scores, labels = attack_fn(df, **shared_kwargs)
                if len(np.unique(labels)) < 2:
                    continue   # degenerate model – skip, don't increment counter

                tau     = _optimal_threshold(scores, labels)
                preds   = (scores >= tau).astype(int)
                correct = (preds == labels).astype(float)

                for i, idx in enumerate(indices):
                    success_sum[idx][attack_name]   += correct[i]
                    attack_appear[idx][attack_name] += 1

            except Exception as exc:
                print(f"    [ASR/{attack_name}/{m_dir.name}] ERROR: {exc}")

    # Build the ASR dataframe
    rows = []
    for idx in sorted(success_sum.keys()):
        row = {"idx": idx}
        for a in attack_names:
            n = attack_appear[idx][a]
            row[a] = success_sum[idx][a] / n if n > 0 else np.nan
        rows.append(row)

    asr_df = pd.DataFrame(rows).set_index("idx")
    asr_df.to_csv(output_dir / "asr_per_sample.csv")
    print(
        f"  ✓ ASR saved → {output_dir / 'asr_per_sample.csv'} "
        f"({len(asr_df)} samples × {len(attack_names)} attacks)"
    )
    return asr_df


def _spearman_heatmap(asr_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """
    Compute and plot the Spearman rank-correlation matrix of per-sample ASR
    vectors.  A high ρ between attack A and attack B means that the samples
    A finds easy are also easy for B (correlated vulnerability).

    Outputs
    -------
    output_dir / "spearman_correlation.csv"   – full ρ matrix
    output_dir / "spearman_heatmap.pdf"       – annotated heatmap
    """
    attacks  = asr_df.columns.tolist()
    n_atk    = len(attacks)
    rho_mat  = np.eye(n_atk)
    pval_mat = np.zeros((n_atk, n_atk))

    for i, a1 in enumerate(attacks):
        for j, a2 in enumerate(attacks):
            if i >= j:
                continue
            mask = asr_df[a1].notna() & asr_df[a2].notna()
            rho, pval = spearmanr(asr_df.loc[mask, a1], asr_df.loc[mask, a2])
            rho_mat[i, j] = rho_mat[j, i] = rho
            pval_mat[i, j] = pval_mat[j, i] = pval

    corr_df = pd.DataFrame(rho_mat, index=attacks, columns=attacks)
    corr_df.to_csv(output_dir / "spearman_correlation.csv")

    # --- heatmap plot --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(rho_mat, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    ax.set_xticks(range(n_atk))
    ax.set_xticklabels(attacks, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(n_atk))
    ax.set_yticklabels(attacks, fontsize=9)

    for i in range(n_atk):
        for j in range(n_atk):
            stars = ""
            if i != j:
                p = pval_mat[i, j]
                stars = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
            cell_text = f"{rho_mat[i, j]:.2f}\n{stars}"
            text_color = "white" if abs(rho_mat[i, j]) > 0.6 else "black"
            ax.text(j, i, cell_text, ha="center", va="center",
                    fontsize=7.5, color=text_color)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Spearman ρ", fontsize=9)
    ax.set_title("Spearman Rank Correlation of Per-Sample ASR\n"
                 "(* p<.05  ** p<.01  *** p<.001)", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_dir / "spearman_heatmap.pdf", dpi=150)
    plt.close(fig)
    print(f"  ✓ Spearman heatmap → {output_dir / 'spearman_heatmap.pdf'}")
    return corr_df


def _topk_intersection(asr_df: pd.DataFrame, output_dir: Path, comparison_column="lira_asr") -> pd.DataFrame:
    """
    For every ordered pair (A, B) of attacks and every k ∈ {1, 2, …, 100}%,
    compute what fraction of A's top-k% most vulnerable samples also appear
    in B's top-k% most vulnerable samples:

        overlap_k(A, B) = |top_k(A) ∩ top_k(B)| / |top_k(A)|

    Because both sets have the same cardinality the metric is symmetric and
    equals the Jaccard index for equal-size sets.

    The random baseline is the identity line: if A and B are independent,
    expected overlap at k% is k%.

    Outputs
    -------
    output_dir / "topk_intersection.csv"   – one column per pair, rows = k
    output_dir / "topk_intersection.pdf"   – one curve per pair
    """
    attacks = asr_df.columns.tolist()
    k_grid = np.geomspace(1e-3, 1e0, 500)   # k = 1e-4 % … 100 %
    # k = 1 % … 100 %

    # Pre-sort indices by descending ASR (most vulnerable first) per attack
    sorted_idx: dict[str, list] = {}
    for a in attacks:
        valid = asr_df[a].dropna()
        sorted_idx[a] = valid.sort_values(ascending=False).index.tolist()

    # pairs = list(itertools.combinations(attacks, 2))
    pairs = [(comparison_column, att) for att in attacks if att != comparison_column]
    print(pairs)
    records = []
    for k in k_grid:
        row: dict = {"k_pct": k}
        for a1, a2 in pairs:
            n1 = max(1, int(len(sorted_idx[a1]) * k))
            n2 = max(1, int(len(sorted_idx[a2]) * k))
            set1 = set(sorted_idx[a1][:n1])
            set2 = set(sorted_idx[a2][:n2])
            row[f"{a1}_vs_{a2}"] = len(set1 & set2) / len(set1)
        records.append(row)

    topk_df = pd.DataFrame(records)
    topk_df.to_csv(output_dir / "topk_intersection.csv", index=False)

    # --- plot ----------------------------------------------------------------
    pair_cols = [f"{a1}_vs_{a2}" for a1, a2 in pairs]
    cmap      = plt.get_cmap("tab10", len(pairs))

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, col in enumerate(pair_cols):
        label = col.replace("_vs_", " vs ")
        ax.plot(topk_df["k_pct"], topk_df[col],
                label=label, color=cmap(i), lw=2)

    # Random baseline: at k%, two independent uniform draws over N share k% on average
    ax.plot([0, 100], [0, 100], "k--", lw=1.2, alpha=0.5, label="Random baseline")

    ax.set_xlabel("k  (top-k% most vulnerable samples)", fontsize=11)
    ax.set_ylabel("Overlap  (% of top-k(A) also in top-k(B))", fontsize=11)
    ax.set_title("Shared Vulnerable Samples Between Attack Pairs vs k", fontsize=12)
    ax.set_aspect("equal", adjustable="box")    
    ax.set_ylim(k_grid.min(), k_grid.max())
    ax.set_xlim(k_grid.min(), k_grid.max())
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8.5,
              framealpha=0.9)
    fig.tight_layout()
    fig.savefig(output_dir / f"topk_intersection_{comparison_column}.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Top-k intersection plot → {output_dir}/topk_intersection_{comparison_column}.pdf")
    return topk_df


def analyze_attack_vulnerability(asr_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Run both cross-attack analyses on the per-sample ASR matrix:
      1. Spearman rank correlation + heatmap
      2. Top-k% overlap curves across all attack pairs

    All artefacts are written to *output_dir / "cross_attack_analysis/"*.
    """
    analysis_dir = output_dir / "cross_attack_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    print("\n── Cross-attack vulnerability analysis ──")
    corr_df  = _spearman_heatmap(asr_df, analysis_dir)
    topk_df  = _topk_intersection(asr_df, analysis_dir)

    # Concise summary printed to console
    print("\n  Spearman ρ summary (lower triangle):")
    attacks = asr_df.columns.tolist()
    for i, a1 in enumerate(attacks):
        for j, a2 in enumerate(attacks):
            if i > j:
                print(f"    {a1:16s} vs {a2:16s}: ρ = {corr_df.loc[a1, a2]:+.3f}")


def main():
    args = parse_args()

    shadow_path = Path(args.shadow_path)
    target_path = Path(args.target_path)
    output_dir  = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -- Load pre-computed shadow statistics ----------------------------------
    print("Loading shadow-model statistics …")
    lira_stats  = pd.read_csv(shadow_path / "lira_stats.csv")
    rmia_stats  = pd.read_csv(shadow_path / "rmia_stats.csv")
    # Note: compute_attacks_statistics.py saves "shokri_classifiers.pkl"
    with open(shadow_path / "shokri_classifiers.pkl", "rb") as f:
        shokri_clfs = pickle.load(f)

    shared_kwargs = dict(
        lira_stats=lira_stats,
        rmia_stats=rmia_stats,
        shokri_clfs=shokri_clfs,
    )

    # -- Iterate over target models (per-model ROC + metrics) -----------------
    model_dirs = sorted(
        d for d in target_path.iterdir()
        if d.is_dir() and d.name.startswith("model_")
    )
    print(f"Found {len(model_dirs)} target model(s).\n")

    timing = {}
    for m_dir in tqdm(model_dirs, desc="Target models – per-model metrics"):
        metrics_path = m_dir / "metrics.pkl"
        if not metrics_path.exists():
            print(f"  [skip] {m_dir.name}: metrics.pkl not found.")
            continue

        df         = pd.read_pickle(metrics_path)
        model_name = m_dir.name
        print(f"\n── {model_name} ──")

        for attack_name, attack_fn in ATTACKS.items():
            t0 = time.perf_counter()
            try:
                scores, labels = attack_fn(df, **shared_kwargs)

                if len(np.unique(labels)) < 2:
                    print(f"  [{attack_name}] Skipped: only one membership class present.")
                    continue

                metrics = compute_metrics(scores, labels)
                save_results(metrics, attack_name, model_name, output_dir)
                timing[f"{model_name}/{attack_name}"] = time.perf_counter() - t0

            except Exception as exc:
                print(f"  [{attack_name}] ERROR: {exc}")

    # -- Write timing summary -------------------------------------------------
    with open(output_dir / "execution_time_attacks.txt", "w") as f:
        for key, elapsed in timing.items():
            f.write(f"{key}: {elapsed:.4f} s\n")

    # -- Per-sample ASR averaged over all target models -----------------------
    t_asr_start = time.perf_counter()
    if not args.no_asr:
        asr_df = compute_and_save_asr(target_path, output_dir.parent, shared_kwargs)
    t_asr_end = time.perf_counter()

        # -- Cross-attack vulnerability analysis ----------------------------------
    if not args.no_asr:
        analyze_attack_vulnerability(asr_df, output_dir.parent)
    t_analysis_end = time.perf_counter()

    # -- Averaged ROC curve across all models, all attacks on one plot --------
    plot_averaged_roc(target_path, output_dir.parent, shared_kwargs)
    t_roc_end = time.perf_counter()

    # Append ASR + analysis + ROC timings

    with open(output_dir.parent / "execution_time_attacks.txt", "a") as f:
        f.write(f"\nASR computation:        {t_asr_end    - t_asr_start:.4f} s\n")
        f.write(f"Cross-attack analysis:  {t_analysis_end - t_asr_end:.4f} s\n")
        f.write(f"Averaged ROC plot:      {t_roc_end    - t_analysis_end:.4f} s\n")

    print(f"\n✓ All done. Results written to {output_dir}")


if __name__ == "__main__":
    main()