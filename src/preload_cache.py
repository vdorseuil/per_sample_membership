"""Pre-download all backbones and datasets used by BERT/DINO experiments.

Run this LOCALLY on a machine with internet access. Then rsync/scp the resulting
two directories to  any offline server and point LEVERAGE_CACHE_DIR
at the cache root before running training/leverage jobs.

Local usage:
    python src/preload_cache.py

This will populate (relative to the project root):
    ./cache/huggingface/   <- BERT model, tokenizer, SST2 dataset
    ./cache/torch_hub/     <- DINOv2 backbone source + checkpoint
    ./data/BloodMNIST/     <- BloodMNIST 224x224 npz files




"""
import os
from pathlib import Path

CACHE_DIR = Path("./cache")
HF_CACHE = CACHE_DIR / "huggingface"
TORCH_HUB_CACHE = CACHE_DIR / "torch_hub"
DATA_DIR = Path("./data")

HF_CACHE.mkdir(parents=True, exist_ok=True)
TORCH_HUB_CACHE.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

print(f"Populating cache at: {CACHE_DIR}")

# 1. BERT backbone + tokenizer (TinyBERT)
print("\n Tiny BERT")
from transformers import AutoTokenizer, AutoModelForSequenceClassification
BERT_NAME = "google/bert_uncased_L-2_H-128_A-2"
AutoModelForSequenceClassification.from_pretrained(BERT_NAME, num_labels=2, cache_dir=str(HF_CACHE))
AutoTokenizer.from_pretrained(BERT_NAME, cache_dir=str(HF_CACHE))

print("\n Base BERT")
from transformers import AutoTokenizer, AutoModelForSequenceClassification
BERT_NAME = "google-bert/bert-base-uncased"
AutoModelForSequenceClassification.from_pretrained(BERT_NAME, num_labels=2, cache_dir=str(HF_CACHE))
AutoTokenizer.from_pretrained(BERT_NAME, cache_dir=str(HF_CACHE))

# 2. SST2 dataset
print("\n SST2 dataset")
from datasets import load_dataset
load_dataset("glue", "sst2", cache_dir=str(HF_CACHE / "datasets"))

# 3. DINOv2 backbone (source + checkpoint)
print("\n DINOv2 ViT-S/14")
import torch
torch.hub.set_dir(str(TORCH_HUB_CACHE))
torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")

# 4. BloodMNIST (224x224 for DINOv2)
print("\n BloodMNIST dataset")
from medmnist import BloodMNIST
os.makedirs(str(DATA_DIR / "BloodMNIST"), exist_ok=True)
for split in ("train", "val", "test"):
    BloodMNIST(split=split, download=True, root=str(DATA_DIR / "BloodMNIST"), size=224)

print("\nDone.")
print(f"  Cache:  {CACHE_DIR}")
print(f"  Data:   {DATA_DIR / 'BloodMNIST'}")

