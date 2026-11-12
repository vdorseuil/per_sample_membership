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
    parser.add_argument("--shadow_root", type=str, required=True, help="the path where the shadow models were saved.")
    parser.add_argument("--num_classes", type=int, default=10)
    return parser.parse_args()


def compute_lira_stats(shadow_root, output_dir): # Implementation of the Lira attack, more info in the paper "Membership Inference Attacks From First Principles"
    """
    Returns: the mean, std for the phi stats = log(p/1-p) for in training set and out training set
    """
    model_dirs = [d for d in shadow_root.iterdir() if d.is_dir() and d.name.startswith("model_")]
    num_samples = len(pd.read_pickle(model_dirs[0] / "metrics.pkl"))
    array_samples = (pd.read_pickle(model_dirs[0] / "metrics.pkl"))["idx"].values
    
    vals_train = [[] for _ in range(num_samples)] # IN
    vals_test = [[] for _ in range(num_samples)]  # OUT
    

    for m_dir in tqdm(model_dirs, desc="Processing LiRA Stats"):
        metrics_path = m_dir / "metrics.pkl"
        if not metrics_path.exists():
            continue
            
        df = pd.read_pickle(metrics_path) # Contains idx, split, phi
        scores = df['phi'].values
        indices = df['idx'].values
        splits = df['split'].values
        
        for i in array_samples:
            idx = int(indices[i])
            val = scores[i]
            if splits[i] == 'train':
                vals_train[idx].append(val)
            else:
                vals_test[idx].append(val)
                
    lira_df = pd.DataFrame({
        'idx': np.arange(num_samples),
        'mu_in': [np.mean(v) if v else 0 for v in vals_train],
        'sigma_in': [np.std(v) if len(v) > 1 else 1e-8 for v in vals_train],
        'mu_out': [np.mean(v) if v else 0 for v in vals_test],
        'sigma_out': [np.std(v) if len(v) > 1 else 1e-8 for v in vals_test]
    })
    
    lira_df.to_csv(output_dir / "lira_stats.csv", index=False)
    print(f"LiRA stats saved to {output_dir / 'lira_stats.csv'}")
    


def compute_rmia_stats(shadow_root, output_dir, is_npz=False): #Attacks from the paper "Low-Cost High-Power Membership Inference Attacks"
    """
    Compute containing the mean confidence (=proba of the correct label) across reference models.
    """
    model_dirs = [d for d in shadow_root.iterdir() if d.is_dir() and d.name.startswith("model_")]
    if not is_npz:
        num_samples = len(pd.read_pickle(model_dirs[0] / "metrics.pkl"))
    else: #npz file
        array_samples = (np.load(model_dirs[0] / "metrics.npz")["idx"])
        num_samples = len(array_samples)
        
    full_probs = [[] for _ in range(num_samples)]
    
    for m_dir in tqdm(model_dirs, desc="Processing RMIA Stats"):
        if not is_npz:
            metrics_path = m_dir / "metrics.pkl"
            if not metrics_path.exists():
                continue
                
            df = pd.read_pickle(metrics_path)        

            # The 'loss' column is the cross-entropy, i.e. MINUS the log-probability
            # of the label, so p = exp(-loss).  exp(+loss) would yield 1/p, which is
            # the reciprocal of what rmia_attack() divides by.
            losses = np.stack(df['loss'].values)
            probs = np.exp(-losses)
            indices = df['idx'].values
        else:
            metrics_path = m_dir / "metrics.npz"
            if not metrics_path.exists():
                continue
                
            df = np.load(metrics_path)        

            losses = np.stack(df['loss'])          # cross-entropy = -log p (see above)
            probs = np.exp(-losses)
            indices = df['idx']
        
        # len(indices), NOT len(df): in the npz branch `df` is an NpzFile, whose
        # len() is the number of stored arrays (7), so only idx 0..6 ever got a
        # real avg_prob and every other sample fell back to the 1e-10 default.
        for i in range(len(indices)):
            idx = int(indices[i])
            val = probs[i]
            full_probs[idx].append(val)
        
    avg_probs = np.array([np.mean(v) if v else 1e-10 for v in full_probs])
    rmia_df = pd.DataFrame({
        'idx': np.arange(num_samples),
        'avg_prob': avg_probs
    })    
    rmia_df.to_csv(output_dir / "rmia_stats.csv", index=False)
    print(f"RMIA stats saved to {output_dir / 'rmia_stats.csv'}")
   
    

def compute_shokri_stats(shadow_root, output_dir, num_classes): #Following the paper "Membership Inference Attacks Against  Machine Learning Models"
    """
    Trains one classifier per class using posterior probability vectors 
    from all shadow models to distinguish members from non-members.
    """
    model_dirs = [d for d in shadow_root.iterdir() if d.is_dir() and d.name.startswith("model_")]
    all_data = []


    for m_dir in tqdm(model_dirs, desc="Collecting Shokri Data"):
        metrics_path = m_dir / "metrics.pkl"
        if metrics_path.exists():
            df = pd.read_pickle(metrics_path)
            essential_df = df[['label', 'split', 'log_proba', 'loss']] 
            all_data.append(essential_df)
    
    full_df = pd.concat(all_data, ignore_index=True)
    shokri_clfs = {}
    
    for c in tqdm(range(num_classes), desc="Training Shokri Classifiers"):
        class_df = full_df[full_df['label'] == c]
        if len(class_df) < 50: 
            print(f"Skipping class {c}: insufficient data.")
            continue 
        
        # Features: The full vector of posterior probabilities
        X = np.exp(np.stack(class_df['log_proba'].values))
        # Target: 1 if member (train), 0 if non-member (test)
        y = (class_df['split'] == 'train').astype(int).values
        
        clf = RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1)
        clf.fit(X, y)
        shokri_clfs[c] = clf
    
    save_path = output_dir / "shokri_classifiers.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(shokri_clfs, f)
    print(f"Shokri classifiers saved to {save_path}")

def main():
    args = parse_args()

    shadow_root = Path(args.shadow_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = time.perf_counter()
    
    compute_lira_stats(shadow_root, output_dir)
    end_time_lira = time.perf_counter()

    compute_rmia_stats(shadow_root, output_dir)
    end_time_rmia = time.perf_counter()

    compute_shokri_stats(shadow_root, output_dir, args.num_classes)
    end_time_shokri = time.perf_counter()
    
    with open(output_dir / "execution_time_statistics.txt", "w") as file:
        file.write(f"Lira time: {end_time_lira-start_time:.4f} seconds\n")
        file.write(f"RMIA time: {end_time_rmia-end_time_lira:.4f} seconds\n")
        file.write(f"Shokri classifier time: {end_time_shokri-end_time_rmia:.4f} seconds\n")

if __name__ == "__main__":
    main()