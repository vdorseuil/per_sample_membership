import sys
sys.modules['xformers'] = None

import argparse
import datetime
import time
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.func import functional_call, grad, jacrev, jvp, vmap
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset import get_dataset
from helpers import compute_metrics_classif, get_criterion, get_optimizer
from helpers_leverage import compute_multivariate_leverage_logistic, compute_leverage_regression
from model import get_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_dir", type=str, required=True, help="Path to target model output dir")
    parser.add_argument("--query_bs", type=int, default=256, help="The query batch size.")
    parser.add_argument("--train_bs", type=int, default=256, help="The train batch size for the Hessian.")
    parser.add_argument("--tol", type=float, default=1e-3, help="The tolerance for the result of the conjugate gradient.")
    parser.add_argument("--iter", type=int, default=50, help="Number of iterations in the conjugate gradient.")
    parser.add_argument("--damping", type=float, default=1e-5, help="Damping parameter in the iHVP")
    parser.add_argument("--keep_layers",nargs="+",default=None,help="The layers to keep when computing the leverage score. If None, keep all the layers.",)
    return parser.parse_args()


args = parse_args()
args_dict = vars(args)

target_dir = Path(args.target_dir)
leverage_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = target_dir  / "leverage_score" / leverage_id
output_dir.mkdir(parents=True, exist_ok=True)


with open(target_dir / "config.json", "r") as f:
    config =  json.load(f)


TRAIN_BS = args.train_bs
QUERY_BS = args.query_bs
DAMPING = args.damping
ITER = args.iter
TOL = args.tol

N = config['num_points']
SEED = config['seed']
EPOCHS = config['epochs']
LR = config['learning_rate']
WD = config['weight_decay']


DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
OUT_DIR = Path(config['out_dir']) / f"model_{SEED}"
NUM_CLASSES = config.get('num_classes', 0)
LABEL_SMOOTHING = config.get('label_smoothing', 0.1)
MOMENTUM = config.get('momentum', 0.9)
FEAT_DIM = 256

np.random.seed(SEED)
torch.manual_seed(SEED)

dataset = get_dataset(config['dataset'])


with np.load(target_dir / "data_splits.npz") as f:
    train_indices = f["train"]
    test_indices = f["test"]


train_dataset = Subset(dataset, indices=train_indices)
test_dataset = Subset(dataset, indices=test_indices)

train_loader = DataLoader(train_dataset, batch_size=TRAIN_BS, shuffle=False) #We don't need to shuffle here as we don't train.
query_loader = DataLoader(train_dataset, batch_size=QUERY_BS, shuffle=False) #We don't need to shuffle here as we don't train.
test_loader = DataLoader(test_dataset, batch_size=TRAIN_BS, shuffle=False)
      
model = get_model(config['model'],  **config).to(DEVICE)
optimizer = get_optimizer(config['optimizer'], model, LR, WD, MOMENTUM)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

model.load_state_dict(torch.load(target_dir / "model_weights.pth", map_location=DEVICE, weights_only=True), strict=False)

model.eval()

N_TRAIN = len(train_loader.dataset)
start_time = time.perf_counter()
if config["model"] in ["resnet20", "logistic_regression", "cnn", "dino", "bert"]:
    leverages_newton, leverages_influence, levearges_not_converged = compute_multivariate_leverage_logistic(model, train_loader, device=DEVICE, damping=DAMPING)
    h_ii = None
elif config["model"] in ["linear_regression", "mlp", "tabular_mlp"]:
    leverages_newton, leverages_influence, levearges_not_converged, h_ii = compute_leverage_regression(model, train_loader, DEVICE, DAMPING)
else:
    print(f"Leverage computation not supported for model '{config['model']}', skipping.")
    sys.exit(0)

end_time = time.perf_counter()


with open(output_dir / 'leverage_newton.pkl', 'wb') as f:
    pickle.dump(leverages_newton, f)

with open(output_dir / 'leverage_newton_not_cv.pkl', 'wb') as f:
    pickle.dump(levearges_not_converged, f)

with open(output_dir / 'leverage_influence.pkl', 'wb') as f:
    pickle.dump(leverages_influence, f)

if h_ii is not None:
    with open(output_dir / 'leverage_h_ii.pkl', 'wb') as f:
        pickle.dump(h_ii, f)
    
with open(output_dir / "config.json", 'w') as f:
    json.dump(args_dict, f, indent=4)

with open(output_dir / "time_leverage.txt", 'w') as f:
    f.write(f"Time to compute all leverage scores : {(end_time-start_time):.4f} seconds")