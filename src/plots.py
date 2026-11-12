import sys
sys.modules['xformers'] = None

import pandas as pd
import numpy as np
import argparse
import json
import matplotlib.pyplot as plt
from pathlib import Path
import itertools
from scipy.stats import spearmanr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from model import get_model
from dataset import get_dataset
from attacks_classif import _spearman_heatmap, _topk_intersection

def parse_args():
    parser = argparse.ArgumentParser(description="Compute leverage scores and ASR intersections.")
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to target model output dir (e.g., experiments/target_models/Cifar10/outputs/model_200)")
    parser.add_argument("--ref_path", type=str, required=True, 
                        help="Path to the reference models directory (e.g., experiments/reference_models/Cifar10)")
    return parser.parse_args()

args = parse_args()

# Setup paths
model_path = Path(args.model_path)
ref_path = Path(args.ref_path)
leverage_base_dir = model_path / "leverage_score"

# Device configuration (Unified)
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"Using device: {device}")

# Load Config
config_path = model_path / "config.json"
with open(config_path, "r") as f:
    config = json.load(f)

# Load Model & Dataset
model = get_model(config["model"], **config)
dataset_name = config.get("dataset", "cifar10")
dataset = get_dataset(dataset_name)


# Load Weights
weights_path = model_path / "model_weights.pth"
model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True), strict=False)
model.eval()
model.to(device)

def _to_device(x, device):
    if isinstance(x, dict):
        return {k: v.to(device) for k, v in x.items()}
    return x.to(device)



# Load Data Splits
splits_path = model_path / "data_splits.npz"
with np.load(splits_path) as f:
    train_indices = f["train"]
    test_indices = f["test"]

train_dataset = Subset(dataset, indices=train_indices)
train_loader = DataLoader(train_dataset, batch_size=256)

# --- 1. Compute Features and Proxy Leverage ONCE ---
print("Computing model features and proxy leverage scores...")
if config["model"] in  ["resnet20", "logistic_regression", "cnn", "bert", "dino"]:

    all_features, all_probs, all_targets, all_idx, all_x, all_labels = [], [], [], [], [], []

    for x, y, idx_list in train_loader:
        x, y = _to_device(x, device), y.to(device)
        with torch.no_grad():
            feat = model.get_features(x) 
            logits = model(x) 
            probs = F.softmax(logits, dim=1) 
            
        all_x.append(x)    
        all_labels.append(y.float().cpu())
        all_features.append(feat.cpu())
        all_probs.append(probs.cpu())
        all_targets.append(F.one_hot(y, num_classes=logits.shape[1]).float().cpu())
        all_idx.append(torch.Tensor(idx_list).cpu())

    X = torch.cat(all_features, dim=0) 
    P = torch.cat(all_probs, dim=0)    
    Y = torch.cat(all_targets, dim=0)  
    labels = torch.cat(all_labels, dim=0).int()
    indices = torch.cat(all_idx, dim=0)

    p_true = P[torch.arange(len(labels)), labels]  
    w = p_true * (1.0 - p_true)  

    XTWX = (X * w.unsqueeze(1)).T @ X  
    eye = torch.eye(XTWX.size(0), device="cpu")
    XTWX += 1e-4 * eye

    inv = torch.linalg.inv(XTWX)  
    quad = torch.einsum("ij,jk,ik->i", X, inv, X)  
        
    h = w * quad
    asr_proxy = quad * np.linalg.norm(P - Y, axis=1)**2 / (1 - h)

    df_proxy_lvg = pd.DataFrame({
        "idx": indices, 
        "lvg_proxy": asr_proxy,
        "h_ii": -h
    })

# --- 2. Pre-load common evaluation metrics ---
target_values = pd.read_pickle(model_path / "metrics.pkl")
df_asr = pd.read_csv(ref_path / "asr_per_sample.csv")
df_asr = df_asr.rename(columns=lambda col: col + "_asr" if col != 'idx' else col)

# --- 3. Loop over all leverage_score subdirectories ---
if not leverage_base_dir.exists():
    raise FileNotFoundError(f"Leverage directory not found at {leverage_base_dir}")

for target_dir in leverage_base_dir.iterdir():
    if not target_dir.is_dir():
        continue
        
    # Find all pickle files starting with 'leverage'
    pkl_files = list(target_dir.glob("leverage*.pkl"))
    
    # Skip empty folders or folders without matching pickle files
    if not pkl_files:
        print(f"Skipping empty or invalid directory: {target_dir.name}")
        continue
        
    print(f"\nProcessing leverage data in: {target_dir.name}")
    
    full_df = None
    dynamic_lvg_cols = []
    first_file = True

    for pkl_file in pkl_files:
        # Dynamically determine the column name based on the filename
        stem = pkl_file.stem  # e.g., 'leverage_newton', 'leverage', etc.
        
        if "_" in stem:
            # Extracts 'newton' from 'leverage_newton'
            suffix = stem.split("_", 1)[1]
            col_name = f"lvg_{suffix}"
        else:
            # Handles generic 'leverage.pkl'
            col_name = "lvg_base"
            
        dynamic_lvg_cols.append(col_name)
        
        # Load the specific leverage data
        lvg_data = pd.read_pickle(pkl_file)
        df_lvg = pd.DataFrame(lvg_data.items(), columns=["idx", col_name])
        
        if first_file:
            # Use keys from the first valid leverage file to slice target_values and df_asr
            keys_list = list(lvg_data.keys())
            
            full_df = pd.merge(left=target_values.iloc[keys_list], right=df_lvg, on="idx")
            full_df = pd.merge(left=full_df, right=df_asr.iloc[keys_list], on="idx")
            if config["model"] in  ["resnet20", "logistic_regression", "cnn", "bert", "dino"]:
                full_df = pd.merge(left=full_df, right=df_proxy_lvg, on="idx")
            
            first_file = False
        else:
            # Merge subsequent dynamic leverage files onto the existing full_df
            full_df = pd.merge(left=full_df, right=df_lvg, on="idx")


    # Generate Heatmap
    columns_base = ["loss", "local_grad_norm", "lvg_proxy", "h_ii"] + dynamic_lvg_cols 
    columns_base_reg = ["loss", "local_grad_norm"] + dynamic_lvg_cols 
    if config["model"] in  ["resnet20", "logistic_regression", "cnn", "bert", "dino"]:
        _spearman_heatmap(asr_df=full_df[columns_base + ["lira_asr"]], output_dir=target_dir)
    
    if config["model"] in ["linear_regression", "mlp"]:
        _spearman_heatmap(asr_df=full_df[columns_base_reg + ["lira_asr"]], output_dir=target_dir)

    
    # Process Top-K Intersections safely in a loop
    for asr_col in ["lira_asr", "rmia_asr", "shokri_asr"]:
        if asr_col in full_df.columns:
            if config["model"] in  ["resnet20", "logistic_regression", "cnn", "bert", "dino"]:
                _topk_intersection(
                    asr_df=full_df[columns_base + [asr_col]], 
                    output_dir=target_dir, 
                    comparison_column=asr_col
                )
            if config["model"] in ["linear_regression", "mlp", "tabular_mlp"]:
                _topk_intersection(
                    asr_df=full_df[columns_base_reg + [asr_col]], 
                    output_dir=target_dir, 
                    comparison_column=asr_col
                )
        else:
            print(f"  -> Warning: '{asr_col}' not found in merged dataframe. Skipping intersection.")
            
print("\nAll leverage directories processed successfully!")