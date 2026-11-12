import time
import sys
# This completely hides the broken xformers package from DINOv2
sys.modules['xformers'] = None

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
from transformers import get_cosine_schedule_with_warmup

from dataset import get_dataset
from helpers import compute_metrics_classif, get_criterion, get_optimizer
from model import get_model

import warnings; warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="")
    
    # Model and Dataset
    parser.add_argument("--dataset", type=str, required=True, choices=["cifar10", "cifar100", "sst2", "blood_mnist"], help="The dataset.")
    parser.add_argument("--model",type=str,required=True,choices=["logistic_regression", "cnn", "resnet20", "bert", "dino"] ,help="The model.",)
    parser.add_argument("-n", "--num_points", type=int, default=50000)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--bert_model_name", type=str, default="google/bert_uncased_L-2_H-128_A-2", help="The type of bert model to load.")
    
    # Training
    parser.add_argument("--criterion",type=str,default="CrossEntropyLoss",choices=["CrossEntropyLoss", "MSELoss", "BCELoss", "BCEWithLogitsLoss"],help="The loss function used for the training.")
    parser.add_argument("--optimizer",type=str,default="Adam",choices=["Adam", "SGD", "RMSProp", "AdamW"],help="The optimizer used for the training.",)
    parser.add_argument("--num_workers", type=int, default=4, help="The number of workers in the Dataloader.")
    parser.add_argument("--freeze", action="store_true", default=False, help="to freeze the model except the last layer for training (works only for bert and for dino models)")
    
    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=25, help="Number of epochs")
    parser.add_argument("-bs", "--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-3, help="The learning rate.")
    parser.add_argument("-wd", "--weight_decay", type=float, default=5e-4, help="The weight decay in the optimizer.")
    parser.add_argument("--momentum", type=float, default=0.9, help="The momentum in the SGD (only) optimizer.")
    parser.add_argument("--dropout", type=float, default=0.3, help="The dropout rate to apply to models.")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="The label smoothing (applied in the criterion)")
    
    # Other
    parser.add_argument("--seed", type=int, default=0, help="The seed_number.")
    parser.add_argument("--no_checkpoint",dest="checkpoint",action="store_false",default=True,help="Disable the saving of the checkpoint",)
    parser.add_argument("--out_dir", type=str, default="experiments/reference_models/Cifar10/outputs", help="The output directory.")
    parser.add_argument("--no_hii", dest="hii", action="store_false", default=True, help="Disable the computation of the hii")

    return parser.parse_args()

args = parse_args()
args_dict = vars(args)
print(args_dict)


N = args.num_points
N_TRAIN = N // 2
SEED = args.seed
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
LR = args.learning_rate
WD = args.weight_decay
FEAT_DIM = 256      
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
OUT_DIR = Path(args.out_dir) / f"model_{SEED}"
NUM_CLASSES = args.num_classes

np.random.seed(SEED)
torch.manual_seed(SEED)
os.makedirs(OUT_DIR, exist_ok=True)




dataset = get_dataset(args.dataset)
y = dataset.labels
N = len(dataset)
N_TRAIN = N // 2

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


def _to_device(x, device):
    if isinstance(x, dict):
        return {k: v.to(device) for k, v in x.items()}
    return x.to(device)


model = get_model(args.model, **args_dict).to(DEVICE)
optimizer = get_optimizer(args.optimizer, model, args.learning_rate, args.weight_decay, args.momentum)
criterion = get_criterion(args.criterion, label_smoothing=args.label_smoothing)
if args.model == "bert":
    num_training_steps = len(train_loader) * EPOCHS
    num_warmup_steps = int(0.1 * num_training_steps) # 10% warmup

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=num_warmup_steps, 
        num_training_steps=num_training_steps
    )
else:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)



def train():
    model.train()
    train_losses = []
    test_losses = []
    for _ in tqdm(range(EPOCHS)):
        train_losses_loader = []
        for xb, yb, idx_list in train_loader:
            xb, yb = _to_device(xb, DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            if args.model == "bert":
                scheduler.step()
            train_losses_loader.append(loss.detach().float().item())
        train_losses.append(float(torch.mean(torch.Tensor(train_losses_loader))))
        with torch.no_grad():
            for xb, yb, idx_list in test_loader:
                xb, yb = _to_device(xb, DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                test_losses.append(loss.detach().float().item())
                break
        if args.model != "bert":
            scheduler.step()
    return train_losses, test_losses

train_losses, test_losses = train()


model.eval()
with torch.no_grad():
    train_acc_list = []
    test_acc_list = []
    for xb, yb, idx_list in train_loader:
        xb, yb = _to_device(xb, DEVICE), yb.to(DEVICE)
        train_acc_list += (model(xb).argmax(1) == yb).float().tolist()

    for xb, yb, idx_list in test_loader:
        xb, yb = _to_device(xb, DEVICE), yb.to(DEVICE)
        test_acc_list += (model(xb).argmax(1) == yb).float().tolist()

test_acc = np.mean(test_acc_list)
train_acc = np.mean(train_acc_list)

df_metrics = compute_metrics_classif(model, train_loader, test_loader, DEVICE)


df_losses = pd.DataFrame({
    'epoch': range(1, EPOCHS + 1),
    'train_loss': train_losses,
    'test_loss': test_losses
})
df_losses.to_csv(OUT_DIR / "losses.csv", index=False)

df_acc = pd.DataFrame({
    'train_acc': [train_acc],
    'test_acc': [test_acc]
})
df_acc.to_csv(OUT_DIR / "accuracies.csv", index=False)

df_metrics.to_pickle(OUT_DIR / "metrics.pkl")

# For BERT/DINO with frozen backbone, save only the trainable head to save disk space
# (200 shadow models would otherwise duplicate the heavy backbone weights).
if args.model in ["bert", "dino"]:
    trainable_names = {name for name, p in model.named_parameters() if p.requires_grad}
    state_to_save = {k: v for k, v in model.state_dict().items() if k in trainable_names}
else:
    state_to_save = model.state_dict()
torch.save(state_to_save, OUT_DIR / "model_weights.pth")
np.savez(OUT_DIR / "data_splits.npz", train=train_indices, test=test_indices)

with open(OUT_DIR / "config.json", "w") as f:
    json.dump(vars(args), f, indent=4)
    
print("Finish")
end_time = time.perf_counter()
execution_time = end_time - start_time

with open(OUT_DIR / "execution_time.txt", "w") as file:
    file.write(f"Execution time: {execution_time:.4f} seconds\n")