import time
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import pickle
from sklearn.ensemble import RandomForestClassifier


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True, help="Path to target model output dir")
    parser.add_argument("--shadow_root", type=str, required=True, help="Path where shadow models were saved")
    return parser.parse_args()


def compute_lira_stats(shadow_root, output_dir):
    """
    LiRA for regression (Carlini et al., "Membership Inference Attacks From First Principles").

    In classification, phi = log(p / (1 - p)) is the logit of the confidence score.
    Under a Gaussian noise assumption for regression, the log-likelihood of observing
    target y given prediction f(x) is:
        log p(y | x) = -loss / (2 * sigma^2) + const
    where loss = (y - f(x))^2.

    Since sigma^2 and the constant are shared across samples, we use:
        phi = -loss   (negative MSE)
    as a sufficient statistic for the log-likelihood ratio test that LiRA performs.

    Expected columns in metrics.pkl: idx, split, loss
        - loss: scalar MSE for that sample = (prediction - target)^2
    """
    model_dirs = [d for d in shadow_root.iterdir() if d.is_dir() and d.name.startswith("model_")]
    num_samples = len(pd.read_pickle(model_dirs[0] / "metrics.pkl"))

    vals_train = [[] for _ in range(num_samples)]  # IN  (member)
    vals_test  = [[] for _ in range(num_samples)]  # OUT (non-member)

    for m_dir in tqdm(model_dirs, desc="Processing LiRA Stats"):
        metrics_path = m_dir / "metrics.pkl"
        if not metrics_path.exists():
            continue

        df = pd.read_pickle(metrics_path)  # columns: idx, split, loss
        indices = df['idx'].values
        splits  = df['split'].values

        # phi = -loss: higher phi <=> lower loss <=> model fits this sample well.
        # This mirrors the logit-of-confidence intuition: members tend to have higher phi.
        phi_values = -df['loss'].values

        for i in range(len(df)):
            idx = int(indices[i])
            if splits[i] == 'train':
                vals_train[idx].append(phi_values[i])
            else:
                vals_test[idx].append(phi_values[i])

    lira_df = pd.DataFrame({
        'idx':       np.arange(num_samples),
        'mu_in':     [np.mean(v) if v else 0.0  for v in vals_train],
        'sigma_in':  [np.std(v)  if len(v) > 1 else 1e-8 for v in vals_train],
        'mu_out':    [np.mean(v) if v else 0.0  for v in vals_test],
        'sigma_out': [np.std(v)  if len(v) > 1 else 1e-8 for v in vals_test],
    })

    out_path = output_dir / "lira_stats.csv"
    lira_df.to_csv(out_path, index=False)
    print(f"LiRA stats saved to {out_path}")


def compute_rmia_stats(shadow_root, output_dir):
    """
    RMIA for regression (Zarifzadeh et al., "Low-Cost High-Power Membership Inference Attacks").

    In classification, the membership score is the average confidence (= probability of the
    correct class) across reference models.

    For regression, under a Gaussian noise model, the likelihood of observing target y is:
        p(y | x) ∝ exp(-loss / (2 * sigma^2))

    We use sigma^2 = 1 (or any fixed constant, since it cancels in the ratio used by RMIA),
    giving:
        likelihood_i = exp(-loss_i)

    avg_likelihood then replaces avg_prob from the classification version.

    Expected columns in metrics.pkl: idx, loss
        - loss: scalar MSE for that sample
    """
    model_dirs = [d for d in shadow_root.iterdir() if d.is_dir() and d.name.startswith("model_")]
    num_samples = len(pd.read_pickle(model_dirs[0] / "metrics.pkl"))

    full_likelihoods = [[] for _ in range(num_samples)]

    for m_dir in tqdm(model_dirs, desc="Processing RMIA Stats"):
        metrics_path = m_dir / "metrics.pkl"
        if not metrics_path.exists():
            continue

        df = pd.read_pickle(metrics_path)  # columns: idx, loss
        indices = df['idx'].values

        # Convert MSE loss to a Gaussian likelihood proxy.
        # Clipping the loss avoids underflow in exp() for very large errors.
        losses = np.clip(df['loss'].values, 0, 500)
        likelihoods = np.exp(-losses)

        for i in range(len(df)):
            idx = int(indices[i])
            full_likelihoods[idx].append(likelihoods[i])

    avg_likelihoods = np.array([
        np.mean(v) if v else 1e-10 for v in full_likelihoods
    ])

    rmia_df = pd.DataFrame({
        'idx':             np.arange(num_samples),
        'avg_likelihood':  avg_likelihoods,   # replaces avg_prob from classification
    })

    out_path = output_dir / "rmia_stats.csv"
    rmia_df.to_csv(out_path, index=False)
    print(f"RMIA stats saved to {out_path}")


def compute_shokri_stats(shadow_root, output_dir):
    """
    Shokri-style attack for regression (Shokri et al., "Membership Inference Attacks
    Against Machine Learning Models").

    In classification, one meta-classifier per class is trained on the full posterior
    probability vector (softmax output) to distinguish members from non-members.

    For regression, there are no class probabilities. Instead, we train a SINGLE
    meta-classifier on scalar features that carry membership signal:
        - prediction:  the model's scalar output f(x)
        - target:      the true label y
        - loss:        (prediction - target)^2  — the most informative single feature
        - abs_error:   |prediction - target|    — provides complementary signal

    A single classifier suffices because regression has no class axis to stratify on.

    Expected columns in metrics.pkl: idx, split, prediction, target, loss
    """
    model_dirs = [d for d in shadow_root.iterdir() if d.is_dir() and d.name.startswith("model_")]
    all_data = []

    for m_dir in tqdm(model_dirs, desc="Collecting Shokri Data"):
        metrics_path = m_dir / "metrics.pkl"
        if metrics_path.exists():
            df = pd.read_pickle(metrics_path)
            df['target'] = df['prediction'] - df['error']
            # Keep only the columns needed for feature construction
            all_data.append(df[['split', 'prediction', 'target', 'loss']])

    full_df = pd.concat(all_data, ignore_index=True)

    if len(full_df) < 50:
        print("Insufficient data to train Shokri classifier.")
        return

    # Feature matrix: scalar regression-specific signals
    full_df = full_df.copy()
    full_df['abs_error'] = np.abs(full_df['prediction'] - full_df['target'])

    X = full_df[['prediction', 'target', 'loss', 'abs_error']].values
    y = (full_df['split'] == 'train').astype(int).values   # 1 = member, 0 = non-member

    # A single classifier replaces the per-class dict from classification
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1)
    clf.fit(X, y)

    save_path = output_dir / "shokri_classifier.pkl"   # note: singular, not a dict
    with open(save_path, "wb") as f:
        pickle.dump(clf, f)
    print(f"Shokri classifier saved to {save_path}")


def main():
    args = parse_args()

    shadow_root = Path(args.shadow_root)
    output_dir  = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()

    compute_lira_stats(shadow_root, output_dir)
    end_time_lira = time.perf_counter()

    compute_rmia_stats(shadow_root, output_dir)
    end_time_rmia = time.perf_counter()

    compute_shokri_stats(shadow_root, output_dir)   # no num_classes argument
    end_time_shokri = time.perf_counter()

    with open(output_dir / "execution_time_statistics.txt", "w") as f:
        f.write(f"LiRA time:              {end_time_lira   - start_time:.4f} seconds\n")
        f.write(f"RMIA time:              {end_time_rmia   - end_time_lira:.4f} seconds\n")
        f.write(f"Shokri classifier time: {end_time_shokri - end_time_rmia:.4f} seconds\n")


if __name__ == "__main__":
    main()