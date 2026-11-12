import time

start_time = time.perf_counter()

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset import get_dataset
from helpers import compute_metrics_regress, get_criterion, get_optimizer
from model import get_model

import warnings; warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="")
    
    # Model and Dataset
    parser.add_argument("--dataset", type=str, required=True, choices=["mnist", "cifar10", "cifar100", "sst2","california_housing", "uci", "purchase100", "texas100", "medical", "blood_mnist"], help="The dataset.")
    parser.add_argument("--sub_dataset", type=str, choices=["energy", "diabetes"])
    parser.add_argument("--model",type=str,required=True,choices=["logistic_regression", "cnn", "resnet20", "mlp", "linear_regression", "tabular_mlp"] ,help="The model.",)
    parser.add_argument("-n", "--num_points", type=int, default=50000)
    parser.add_argument("--output_dim", type=int, default=10)
    parser.add_argument("--input_dim", type=int, default=1)
    parser.add_argument("--hidden_dim", type=int, default=784)
    parser.add_argument('--hidden_layers', nargs='+', type=int, default=[64, 32],
                    help='Provide a space-separated list of hidden layer sizes.')
    
    # Training
    parser.add_argument("--criterion",type=str,default="MSELoss",choices=["CrossEntropyLoss", "MSELoss", "BCELoss", "BCEWithLogitsLoss"],help="The loss function used for the training.")
    parser.add_argument("--optimizer",type=str,default="Adam",choices=["Adam", "SGD", "RMSProp", "AdamW"],help="The optimizer used for the training.",)
    parser.add_argument("--num_workers", type=int, default=4, help="The number of workers in the Dataloader.")

    
    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=25, help="Number of epochs")
    parser.add_argument("-bs", "--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-3, help="The learning rate.")
    parser.add_argument("-wd", "--weight_decay", type=float, default=5e-4, help="The weight decay in the optimizer.")
    parser.add_argument("--momentum", type=float, default=0.9, help="The momentum in the SGD (only) optimizer.")
    parser.add_argument("--dropout", type=float, default=0.3, help="The dropout rate to apply to models.")
    
    # Other
    parser.add_argument("--seed", type=int, default=0, help="The seed_number.")
    parser.add_argument("--no_checkpoint",dest="checkpoint",action="store_false",default=True,help="Disable the saving of the checkpoint",)
    parser.add_argument("--out_dir", type=str, default="experiments/reference_models/Cifar10/outputs", help="The output directory.")
    parser.add_argument("--no_hii", dest="hii", action="store_false", default=True, help="Disable the computation of the hii")
    

    return parser.parse_args()

args = parse_args()
args_dict = vars(args)

N = args.num_points
N_TRAIN = N // 2
SEED = args.seed
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
LR = args.learning_rate
WD = args.weight_decay  
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
OUT_DIR = Path(args.out_dir) / f"model_{SEED}"

np.random.seed(SEED)
torch.manual_seed(SEED)
os.makedirs(OUT_DIR, exist_ok=True)

dataset = get_dataset(args.dataset)
y = np.array(dataset.y)


rng_s = np.random.RandomState(SEED)
train_indices = rng_s.choice(N, N_TRAIN, replace=False)
test_indices = np.array([i for i in range(N) if i not in train_indices])

train_mask = np.zeros(N, dtype=bool)
train_mask[train_indices] = True
test_mask = ~train_mask
    
train_dataset =  Subset(dataset, indices=train_indices)
test_dataset = Subset(dataset, indices=test_indices)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


model = get_model(args.model, **args_dict).to(DEVICE)
optimizer = get_optimizer(args.optimizer, model, args.learning_rate, args.weight_decay, args.momentum)
criterion = get_criterion(args.criterion)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

def train():
    model.train()
    train_losses = []
    test_losses = []
    for _ in tqdm(range(EPOCHS)):
        train_losses_loader = []
        for xb, yb, idx_list in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb.view(-1, 1))
            loss.backward()
            optimizer.step()
            train_losses_loader.append(loss.detach().float().item())
        train_losses.append(float(torch.mean(torch.Tensor(train_losses_loader))))
        with torch.no_grad():
            for xb, yb, idx_list in test_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb.view(-1, 1))
                test_losses.append(loss.detach().float().item())
                break
        scheduler.step()
    return train_losses, test_losses

train_losses, test_losses = train()


model.eval()
with torch.no_grad():
    train_mse = 0
    test_mse = 0
    for xb, yb, idx_list in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        preds = model(xb)
        train_mse += F.mse_loss(preds, yb.view(-1, 1), reduction='sum').item()

    for xb, yb, idx_list in test_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        preds = model(xb)
        test_mse += F.mse_loss(preds, yb.view(-1, 1), reduction='sum').item()



test_mse /= len(test_dataset)
train_mse /= len(train_dataset)

df_metrics = compute_metrics_regress(model, train_loader, test_loader, DEVICE)


df_losses = pd.DataFrame({
    'epoch': range(1, EPOCHS + 1),
    'train_loss': train_losses,
    'test_loss': test_losses
})
df_losses.to_csv(OUT_DIR / "losses.csv", index=False)

df_acc = pd.DataFrame({
    'train_mse': [train_mse],
    'test_mse': [test_mse],
})
df_acc.to_csv(OUT_DIR / "mse.csv", index=False)

df_metrics.to_pickle(OUT_DIR / "metrics.pkl")

torch.save(model.state_dict(), OUT_DIR / "model_weights.pth")
np.savez(OUT_DIR / "data_splits.npz", train=train_indices, test=test_indices)

with open(OUT_DIR / "config.json", "w") as f:
    json.dump(vars(args), f, indent=4)
    
print("Finish")
end_time = time.perf_counter()
execution_time = end_time - start_time

with open(OUT_DIR / "execution_time.txt", "w") as file:
    file.write(f"Execution time: {execution_time:.4f} seconds\n")