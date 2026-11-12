import torch
import numpy as np
import pandas as pd
from scipy.special import logsumexp
import torch.nn as nn
from torch.optim import SGD, Adam, Optimizer, RMSprop, AdamW, lr_scheduler



def get_optimizer(
    optimizer_name: str, model: nn.Module, lr: float = 0.001, weight_decay=5e-4, momentum=0.9,
) -> Optimizer:
    """Get optimizer by name."""

    if optimizer_name == "Adam":
        return Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "AdamW":
        return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "SGD":
        return SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    elif optimizer_name == "RMSProp":
        return RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

def get_criterion(criterion_name: str, **kwargs) -> nn.Module:
    """Get loss criterion by name."""
    if criterion_name == "CrossEntropyLoss":
        return nn.CrossEntropyLoss(reduction="mean", **kwargs)
    elif criterion_name == "MSELoss":
        return nn.MSELoss(reduction="mean", **kwargs)
    elif criterion_name == "BCELoss":
        return nn.BCELoss(reduction="mean", **kwargs)
    elif criterion_name == "BCEWithLogitsLoss":
        return nn.BCEWithLogitsLoss(reduction="mean", **kwargs)
    else:
        raise ValueError(f"Unknown criterion: {criterion_name}")

def _to_device(x, device):
    if isinstance(x, dict):
        return {k: v.to(device) for k, v in x.items()}
    return x.to(device)


def compute_metrics_classif(model, train_loader, test_loader, device):
    model.eval()

    all_log_proba = []
    all_indices = []
    all_labels = []
    all_grads = []
    all_splits = []

    loaders = {'train': train_loader, 'test': test_loader}

    for split_name, loader in loaders.items():
        for (xb, yb, idx_list) in loader:
            xb = _to_device(xb, device)
            yb = yb.to(device)

            is_tensor = isinstance(xb, torch.Tensor)
            if is_tensor:
                xb.requires_grad_(True)

            logits = model(xb)
            log_prob = torch.log_softmax(logits, dim=1)

            loss = torch.nn.functional.cross_entropy(logits, yb, reduction='none')

            if is_tensor:
                grad_x = torch.autograd.grad(outputs=loss.sum(), inputs=xb)[0]
                grad_norm = grad_x.view(xb.size(0), -1).norm(p=2, dim=1)
            else:
                grad_norm = torch.zeros(logits.size(0))

            all_log_proba.append(log_prob.detach().cpu().numpy())
            all_indices.append(idx_list.cpu().numpy())
            all_labels.append(yb.cpu().numpy())
            all_grads.append(grad_norm.detach().cpu().numpy())

            all_splits.extend([split_name] * len(yb))
                

    indices_arr = np.concatenate(all_indices, axis=0)       # (N,)
    log_proba_arr = np.concatenate(all_log_proba, axis=0)   # (N, K)
    y_a = np.concatenate(all_labels, axis=0)                # (N,)
    grads_arr = np.concatenate(all_grads, axis=0)           # (N,)
    

    log_p_y = log_proba_arr[np.arange(len(y_a)), y_a]       # (N,)
    losses_arr = -log_p_y                                   # (N,)
    
    mask = np.ones_like(log_proba_arr, dtype=bool)
    mask[np.arange(len(y_a)), y_a] = False
    
    K = log_proba_arr.shape[1]
    others = log_proba_arr[mask].reshape(-1, K - 1)
    lse_others = logsumexp(others, axis=1)
    
    phi_arr = log_p_y - lse_others #Used for the Lira Attack (see the paper "Membership Inference Attacks From First Principles").
    

    df = pd.DataFrame({
        'idx': indices_arr,
        'split': all_splits,
        'label': y_a,
        'loss': losses_arr,
        'phi': phi_arr,
        'local_grad_norm': grads_arr
    })
    
    df['log_proba'] = list(log_proba_arr)
    df = df.sort_values(by='idx').reset_index(drop=True)
    
    return df


def compute_metrics_regress(model, train_loader, test_loader, device):
    model.eval()
    
    all_preds = []
    all_indices = []
    all_labels = []
    all_grads = []
    all_splits = [] 
    all_losses = []

    loaders = {'train': train_loader, 'test': test_loader}
    
    for split_name, loader in loaders.items():
        for (xb, yb, idx_list) in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            
            xb.requires_grad_(True)
            
            # Forward pass
            preds = model(xb)
            
            # Ensure yb matches preds shape (e.g., if preds is [N, 1] and yb is [N])
            if yb.dim() == 1 and preds.dim() == 2:
                yb_reshaped = yb.view(-1, 1)
            else:
                yb_reshaped = yb
            
            # Loss (Per-sample Squared Error)
            # Using MSE as the regression equivalent to Cross Entropy
            loss = torch.nn.functional.mse_loss(preds, yb_reshaped, reduction='none')
            
            # If multi-target regression, average loss across the target dimension
            if loss.dim() > 1:
                loss = loss.mean(dim=1)
            
            # Gradients w.r.t input
            grad_x = torch.autograd.grad(outputs=loss.sum(), inputs=xb)[0]
            grad_norm = grad_x.view(xb.size(0), -1).norm(p=2, dim=1)
            
            # Store metrics
            all_preds.append(preds.detach().cpu().numpy())
            all_losses.append(loss.detach().cpu().numpy())
            all_indices.append(idx_list.cpu().numpy())
            all_labels.append(yb.cpu().numpy())
            all_grads.append(grad_norm.detach().cpu().numpy())
            
            # Track whether these points came from train or test
            all_splits.extend([split_name] * len(xb))
                
    # Concatenate arrays
    indices_arr = np.concatenate(all_indices, axis=0)       # (N,)
    preds_arr = np.concatenate(all_preds, axis=0)           # (N, 1) or (N, D)
    labels_arr = np.concatenate(all_labels, axis=0)         # (N,) or (N, D)
    losses_arr = np.concatenate(all_losses, axis=0)         # (N,)
    grads_arr = np.concatenate(all_grads, axis=0)           # (N,)
    
    # Calculate raw error (residual)
    # Flattens single-target regression arrays for a cleaner DataFrame
    if preds_arr.ndim == 2 and preds_arr.shape[1] == 1:
        preds_flat = preds_arr.flatten()
        labels_flat = labels_arr.flatten()
        errors_arr = preds_flat - labels_flat
        
        preds_out = preds_flat
        labels_out = labels_flat
    else:
        # Handles multi-dimensional output targets gracefully
        errors_arr = [p - y for p, y in zip(preds_arr, labels_arr)]
        preds_out = list(preds_arr)
        labels_out = list(labels_arr)

    df = pd.DataFrame({
        'idx': indices_arr,
        'split': all_splits,
        'label': labels_out,
        'prediction': preds_out,
        'error': errors_arr,     # The primary signal for regression MIA (y_hat - y)
        'loss': losses_arr,      # The squared error metric
        'local_grad_norm': grads_arr
    })
    
    df = df.sort_values(by='idx').reset_index(drop=True)
    
    return df


def log_likelihood_gaussian(x,  mu_0, sigma_0, mu_1, sigma_1):
    """Calculates the log-likelihood ratio of x under test vs. train Gaussian models.

    The function computes log( P(x | H_test) / P(x | H_train) ), where:
    - x = phi(p) is the transformed input score. x = max(logits) - LogSumExp(logits[without the max])
    - P(x | H_test) is the PDF of a Gaussian with mean_test and std_test.
    - P(x | H_train) is the PDF of a Gaussian with mean_train and std_train.
    A higher return value suggests x is more likely under the "test" model's parameters.

    Args:
        x (float): input to test
        mu_0(float): mean of phi(p) for a given point in the test set
        sigma_0 (float): standard deviation of phi(p) for a given point in the test set
        mu_1 (float):  mean of phi(p) for a given point in the train set
        sigma_1 (float):  standard deviation of phi(p) for a given point in the train set

    Returns:
        The log-likelihood ratio between two scalar Gaussian
    """
    return (
        np.log(sigma_0 / sigma_1)
        - 0.5 * ((x - mu_1) ** 2) / (sigma_1**2)
        + 0.5 * ((x - mu_0) ** 2) / (sigma_0**2)
    )