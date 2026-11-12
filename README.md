# Predicting Per-Sample Membership Inference Vulnerability without Retraining

Code for the NeurIPS 2026 paper **"Predicting Per-Sample Membership Inference Vulnerability without Retraining"**.

![Main experimental result](figures/images/topk_recall_lira_1e-1_topleg.pdf)
*Figure: Comparison of our methods $\widehat{\nabla \ell_i}^{\text{IF, NS}}$ versus retraining-free proxies across different datasets. Higher curves are better.*

## 1. Setup

We tested our code with Python 3.11 and PyTorch 2.11 on a single NVIDIA V100 (40 GB). CPU-only runs work for the small datasets (CaliforniaHousing, UCI Energy, Diabetes). Start by cloning the repository : `git clone https://github.com/vdorseuil/per_sample_membership.git)`. 

Then install the dependencies using `uv`:

```bash
   # Install uv: https://docs.astral.sh/uv/getting-started/installation/
   uv sync --frozen --locked
   uv run python src/preload_cache.py   # BERT/DINO backbones + MedMNIST → ./cache
```

or if you cannot use `uv` you can reproduce the environment with the original `requirements.txt`:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

For BERT and DINO experiments the script [`src/preload_cache.py`](src/preload_cache.py) downloads the pretrained backbones into `./cache` once, as well as the MedMNIST dataset:

```bash
uv run python src/preload_cache.py
```

## 2. Pipeline overview

The reproduction process follows a four-step pipeline:

1. *Model Training*: Taining of the reference and target models.
2. *Attack Analysis*: Calculation of attack statistics.
3. *Surrogate Scoring*: Computation of surrogate scores and leverages, or full-network influence function.
4. *Visualization*: Generation of all plots and summary tables.

In the  codebase **shadow** and **reference** refer to the same set of models (the ones used to calibrate the attack), while **target** is the model whose membership predictions we evaluate. All datasets are wrapped in [`src/dataset.py`](src/dataset.py) via `get_dataset(name, ...)`. They download automatically the first time they are used and are then stored in the folder `data`.

## 3. Quickstart (CPU, ~5 min)

The fastest end-to-end run is California Housing with a linear model.

```bash
# 1. Train few (50) reference models (vary --seed) and one target model
for s in $(seq 0 50); do
  python src/train_regress.py \
      --seed $s \
      --dataset "california_housing" \
      --model "linear_regression" \
      --out_dir "experiments/reference_models/CaliforniaHousing_linear/outputs" \
      --input_dim 8 \
      --hidden_dim 128 \
      --output_dim 1 \
      --epochs 100 \
      --batch_size 256 \
      --weight_decay 1e-3 \
      --learning_rate 1e-2 \
      --num_points 16512 \
      --learning_rate 1e-2 \
      --optimizer "Adam" \
      --criterion "MSELoss"
done
python src/train_regress.py \
    --seed 51 \
    --dataset "california_housing" \
    --model "linear_regression" \
    --out_dir "experiments/target_models/CaliforniaHousing_linear/outputs" \
    --input_dim 8 \
    --hidden_dim 128 \
    --output_dim 1 \
    --epochs 100 \
    --batch_size 256 \
    --weight_decay 1e-3 \
    --learning_rate 1e-2 \
    --num_points 16512 \
    --optimizer "Adam" \
    --criterion "MSELoss" \

# 2. Attack statistics + attacks (reference + target)
python src/compute_statistics_regress.py \
    --output_dir "experiments/reference_models/CaliforniaHousing_linear" \
    --shadow_root "experiments/reference_models/CaliforniaHousing_linear/outputs"

python src/attacks_regress.py \
    --output_dir "experiments/reference_models/CaliforniaHousing_linear/outputs" \
    --target_path "experiments/reference_models/CaliforniaHousing_linear/outputs" \
    --shadow_path "experiments/reference_models/CaliforniaHousing_linear/"

python src/attacks_regress.py \
    --output_dir "experiments/target_models/CaliforniaHousing_linear/outputs" \
    --target_path "experiments/target_models/CaliforniaHousing_linear/outputs" \
    --shadow_path "experiments/reference_models/CaliforniaHousing_linear/" \
    --no_asr

# 3. Leverage / influence / Newton scores for the target
python src/compute_leverage.py \
    --target_dir "experiments/target_models/CaliforniaHousing_linear/outputs/model_51"

# 4. Per-model plots
python src/plots.py \
    --model_path "experiments/target_models/CaliforniaHousing_linear/outputs/model_51" \
    --ref_path "experiments/reference_models/CaliforniaHousing_linear"
```

## 4. Full reproduction

### 4.1 Training reference and target models

[`src/train_classif.py`](src/train_classif.py) and [`src/train_regress.py`](src/train_regress.py) train **one** model on a random $N/2$ split of the data and write artifacts (weights, predictions, member mask) to `out_dir/model_{seed}/`. Vary `--seed` to obtain a population of shadow/reference models; pick a different `--seed` and `--out_dir` for the target.

**CIFAR-10 with a CNN (reference model):**

```bash
python src/train_classif.py \
    --seed 0 \
    --epochs 25 \
    --num_classes 10 \
    --out_dir "experiments/reference_models/Cifar10/outputs" \
    --batch_size 256 \
    --weight_decay 5e-4 \
    --learning_rate 1e-3 \
    --model "cnn" \
    --dataset "cifar10" \
    --optimizer "AdamW" \
    --criterion "CrossEntropyLoss"
```

**BloodMNIST with a frozen DINO backbone (target model):**

```bash
python src/preload_cache.py        # downloads DINO/BERT weights into ./cache
python src/train_classif.py \
    --seed 200 \
    --epochs 100 \
    --num_classes 8 \
    --num_points 8000 \
    --out_dir "experiments/target_models/BloodMnist/outputs" \
    --batch_size 512 \
    --weight_decay 1e-6 \
    --learning_rate 1e-2 \
    --model "dino" \
    --dataset "blood_mnist" \
    --momentum 0.9 \
    --optimizer "SGD" \
    --cache_dir "./cache" \
    --criterion "CrossEntropyLoss" \
    --freeze
```

### 4.2 Attack statistics and attacks

[`src/compute_statistics_{classif,regress}.py`](src/) aggregate per-sample statistics across the shadow population; [`src/attacks_{classif,regress}.py`](src/) then run LiRA / RMIA / loss attacks and emit the $(\alpha,\beta)$ trade-off curve and the average ROC for each target.

We pass `--no_asr` when attacking target models because the Average Success Rate is only meaningful when membership ground-truth varies across seeds, which is the case for the shadow population but not for a single target.

**Regression:**

```bash
python src/compute_statistics_regress.py \
    --output_dir "experiments/reference_models/CaliforniaHousing_linear" \
    --shadow_root "experiments/reference_models/CaliforniaHousing_linear/outputs"

python src/attacks_regress.py \
    --output_dir "experiments/reference_models/CaliforniaHousing_linear/outputs" \
    --target_path "experiments/reference_models/CaliforniaHousing_linear/outputs" \
    --shadow_path "experiments/reference_models/CaliforniaHousing_linear/"

python src/attacks_regress.py \
    --output_dir "experiments/target_models/CaliforniaHousing_linear/outputs" \
    --target_path "experiments/target_models/CaliforniaHousing_linear/outputs" \
    --shadow_path "experiments/reference_models/CaliforniaHousing_linear/" \
    --no_asr
```

**Classification:**

```bash
python src/compute_statistics_classif.py \
    --output_dir "experiments/reference_models/BloodMnist" \
    --shadow_root "experiments/reference_models/BloodMnist/outputs"

python src/attacks_classif.py \
    --output_dir "experiments/reference_models/BloodMnist/outputs" \
    --target_path "experiments/reference_models/BloodMnist/outputs" \
    --shadow_path "experiments/reference_models/BloodMnist/" \
    --num_classes 8

python src/attacks_classif.py \
    --output_dir "experiments/target_models/BloodMnist/outputs" \
    --target_path "experiments/target_models/BloodMnist/outputs" \
    --shadow_path "experiments/reference_models/BloodMnist/" \
    --num_classes 8 \
    --no_asr
```

### 4.3 Leverage, influence functions and Newton-step scores

[`src/compute_leverage.py`](src/compute_leverage.py) (using helpers in [`src/helpers_leverage.py`](src/helpers_leverage.py)) computes the **last-layer** leverage scores, influence-function approximation, and Newton-step approximation for one trained model. The script auto-detects classification vs. regression from the saved model. Outputs are written into `target_dir` as `.npy` arrays.

```bash
python src/compute_leverage.py \
    --target_dir "experiments/target_models/CaliforniaHousing_linear/outputs/model_200"
```

### 4.4 Per-model plots

```bash
python src/plots.py \
    --model_path "experiments/target_models/BloodMnist/outputs/model_200" \
    --ref_path "experiments/reference_models/BloodMnist"
```

### 4.5 Compute full-network influence functions.

```bash
python src/compute_full_influence.py \
    --target_dir "experiments/target_models/Cifar10/outputs/model_200" \
    --method cg \
    --query_bs 32 \
    --train_bs 128 \
    --damping 1e1 \
    --hessian_subsample 5000 \
```

## 5. Reproducing paper figures and tables

After running stages 5.1–5.3 for every dataset, regenerate every figure and table in the paper with:

```bash
python figures/code/run_all.py
```

This calls the scripts below in order and writes results to `figures/pdf/` and `figures/tables/`.

## 6. License

Released under the MIT License. See `LICENSE.txt`.

## 7. Citation

Please cite this work as follows:

```bibtex
@inproceedings{dorseuil2026AssessingPerSample,
  title = {Assessing Per-Sample Membership Inference Vulnerability without Retraining},
  author = {Dorseuil, Valentin and Atif, Jamal and Cappé, Olivier},
  year = {2026},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  url = {https://arxiv.org/abs/2602.15919},
}
```
