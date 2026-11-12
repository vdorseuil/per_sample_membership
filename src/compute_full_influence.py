"""Compute the *full* influence-function ASR proxy for a single target model.

For each training point i:

    infl_i = nabla l_i(theta)^T  H_theta^{-1}  nabla l_i(theta)

with the gradient and Hessian taken over **all** trainable parameters of the
network (no last-layer / feature shortcut). Uses CG with a dataloader to
estimate H^{-1} J_i for chunks of query points, then contracts with the loss
output-gradient to get a scalar score per sample. Output is dropped under

    <target_dir>/leverage_score/<timestamp>/leverage_full_influence.pkl

so the existing aggregation pipeline can pick it up alongside the cheap
last-layer leverage scores.
"""
import sys
sys.modules['xformers'] = None

import argparse
import datetime
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from dataset import get_dataset
from helpers import get_criterion
from helpers_leverage import compute_full_influence
from model import get_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_dir", type=str, required=True,
                        help="Path to target model output dir")
    parser.add_argument("--query_bs", type=int, default=32,
                        help="Query chunk size. Smaller = less GPU memory.")
    parser.add_argument("--train_bs", type=int, default=128,
                        help="Train batch size used to estimate the Hessian.")
    parser.add_argument("--iter", type=int, default=20,
                        help="Maximum number of CG iterations.")
    parser.add_argument("--tol", type=float, default=1e-3,
                        help="CG tolerance.")
    parser.add_argument("--damping", type=float, default=1e-2,
                        help="Tikhonov damping added to the Hessian.")
    parser.add_argument("--keep_layers", nargs="+", default=None,
                        help="Subset of parameter names to keep. Defaults to "
                             "ALL trainable parameters (full influence).")
    parser.add_argument("--hessian_subsample", type=int, default=None,
                        help="If set, use this many random training points "
                             "(seeded) to estimate H instead of the full set.")
    parser.add_argument("--n_query_subsample", type=int, default=None,
                        help="If set, evaluate the influence proxy on a "
                             "random subsample of this many query points "
                             "(seeded) instead of the full train set.")
    parser.add_argument("--method", type=str, default="cg",
                        choices=["cg", "lissa"],
                        help="iHVP solver. 'cg' streams the full hessian "
                             "subsample per iter; 'lissa' uses a Neumann "
                             "series with one mini-batch per step.")
    parser.add_argument("--lissa_recursions", type=int, default=1,
                        help="LiSSA: number of independent series averaged.")
    parser.add_argument("--lissa_depth", type=int, default=200,
                        help="LiSSA: length of each Neumann series.")
    parser.add_argument("--lissa_scale", type=float, default=300,
                        help="LiSSA: scale s. Must exceed lambda_max(H_batch).")
    return parser.parse_args()


args = parse_args()
args_dict = vars(args)

target_dir = Path(args.target_dir)
leverage_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = target_dir / "leverage_score" / leverage_id
output_dir.mkdir(parents=True, exist_ok=True)

with open(target_dir / "config.json", "r") as f:
    config = json.load(f)

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)
SEED = config["seed"]
np.random.seed(SEED)
torch.manual_seed(SEED)

# ─── data ─────────────────────────────────────────────────────────────────
dataset = get_dataset(config["dataset"])
with np.load(target_dir / "data_splits.npz") as f:
    train_indices = f["train"]

train_dataset = Subset(dataset, indices=train_indices)

if args.n_query_subsample is not None and args.n_query_subsample < len(train_dataset):
    rng2 = np.random.default_rng(SEED + 1)
    qsub = rng2.choice(len(train_dataset), size=args.n_query_subsample, replace=False)
    query_dataset = Subset(train_dataset, indices=qsub.tolist())
else:
    query_dataset = train_dataset

if args.hessian_subsample is not None and args.hessian_subsample < len(train_dataset):
    rng = np.random.default_rng(SEED)
    sub = rng.choice(len(train_dataset), size=args.hessian_subsample, replace=False)
    hessian_dataset = Subset(train_dataset, indices=sub.tolist())
else:
    hessian_dataset = train_dataset

train_loader = DataLoader(hessian_dataset, batch_size=args.train_bs, shuffle=False)
query_loader = DataLoader(query_dataset, batch_size=args.query_bs, shuffle=False)

# ─── model ────────────────────────────────────────────────────────────────
model = get_model(config["model"], **config).to(DEVICE)
model.load_state_dict(
    torch.load(target_dir / "model_weights.pth", map_location=DEVICE, weights_only=True),
    strict=False,
)
model.eval()
criterion = get_criterion(config["criterion"])

# ─── compute ──────────────────────────────────────────────────────────────
start = time.perf_counter()
scores = compute_full_influence(
    model=model,
    criterion=criterion,
    query_loader=query_loader,
    train_loader=train_loader,
    device=DEVICE,
    keep_layers=args.keep_layers,
    iters=args.iter,
    tol=args.tol,
    damping=args.damping,
    method=args.method,
    lissa_recursions=args.lissa_recursions,
    lissa_depth=args.lissa_depth,
    lissa_scale=args.lissa_scale,
)
elapsed = time.perf_counter() - start

# ─── persist ──────────────────────────────────────────────────────────────
with open(output_dir / f"leverage_full_influence_{args.method}.pkl", "wb") as f:
    pickle.dump(scores, f)

with open(output_dir / "config.json", "w") as f:
    json.dump(args_dict, f, indent=4)

with open(output_dir / "time_full_influence.txt", "w") as f:
    f.write(f"Time to compute full influence: {elapsed:.4f} seconds\n"
            f"n_query={len(scores)}  n_hessian={len(hessian_dataset)}\n")

print(f"[full_influence] wrote {len(scores)} scores to {output_dir}")
print(f"[full_influence] elapsed = {elapsed:.1f}s")
