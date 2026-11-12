import sys
sys.modules['xformers'] = None

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, grad, jvp, vmap, jacrev
from tqdm import tqdm

from typing import Dict, Tuple, Callable


def _to_device(x, device):
    if isinstance(x, dict):
        return {k: v.to(device) for k, v in x.items()}
    return x.to(device)



def compute_multivariate_leverage_logistic(model, train_loader, device, damping=1e-5):
    model.eval()
    all_features = []
    all_probs = []
    all_targets = []
    all_idx = []
    
    for x, y, idx_list in train_loader:
        x, y = _to_device(x, device), _to_device(y, device)
        with torch.no_grad():
            feat = model.get_features(x) # (Batch, d)
            logits = model(x) 
            probs = F.softmax(logits, dim=1) # (Batch, m)
            
        all_features.append(feat)
        all_probs.append(probs)
        all_targets.append(F.one_hot(y, num_classes=logits.shape[1]).float())
        all_idx.append(torch.Tensor(idx_list))

    X = torch.cat(all_features, dim=0) # (N, d)
    P = torch.cat(all_probs, dim=0)    # (N, m)
    Y = torch.cat(all_targets, dim=0)  # (N, m)
    idx_tensor = torch.cat(all_idx, dim=0)
    
    N, d = X.shape
    m = P.shape[1]


    hessian = torch.zeros((m * d, m * d), device=device)
    g_full = torch.zeros((m * d, 1), device=device)
    
    for j in range(N):
        xj = X[j].unsqueeze(1) # (d, 1)
        pj = P[j].unsqueeze(1) # (m, 1)
        yj = Y[j].unsqueeze(1) # (m, 1)
        
        # Hessian components
        Wj = torch.diag(P[j]) - (pj @ pj.T) # (m, m)
        XXj = xj @ xj.T # (d, d)
        hessian += torch.kron(XXj, Wj)
        
        # Gradient components: nabla l_j = X_tilde_j @ g_j
        gj = pj - yj # (m, 1)
        Xj_tilde = torch.kron(xj, torch.eye(m, device=device)) # (d*m, m)
        g_full += Xj_tilde @ gj

    # Add damping for numerical stability
    hessian += damping * torch.eye(m * d, device=device)
    hessian_inv = torch.linalg.inv(hessian)

    # 3. Compute leverage scores
    leverages_newton = {}
    leverages_influence = {}
    leverages_unconverged = {}
    
    for i, idx in zip(range(N), idx_tensor):
        xi = X[i].unsqueeze(1) # (d, 1)
        pi = P[i].unsqueeze(1) # (m, 1)
        yi = Y[i].unsqueeze(1) # (m, 1)
        
        gi = pi - yi # (m, 1)
        Wi = torch.diag(P[i]) - (pi @ pi.T) # (m, m)
        
        Xi_tilde = torch.kron(xi, torch.eye(m, device=device)) # (d*m, m)
        Hii = Xi_tilde.T @ hessian_inv @ Xi_tilde # (m, m)
        
        term_to_inv = torch.eye(m, device=device) - (Wi @ Hii)
        term_inv = torch.linalg.inv(term_to_inv)
        
        # 1. Converged Newton Step (ALO / Leave-One-Out)
        score_newton = gi.T @ Hii @ term_inv @ gi
        
        # 2. Standard Influence Function (First-order)
        score_influence = gi.T @ Hii @ gi
        
        # 3. Unconverged General Form (g_{-i} version)
        g_minus_i = g_full - (Xi_tilde @ gi) # (d*m, 1)
    
        score_unconverged = gi.T @ term_inv.T @ Xi_tilde.T @ hessian_inv @ g_minus_i
        
        idx_key = int(idx.item())
        leverages_newton[idx_key] = score_newton.item()
        leverages_influence[idx_key] = score_influence.item()
        leverages_unconverged[idx_key] = score_unconverged.item()
    
    return leverages_newton, leverages_influence, leverages_unconverged


def compute_leverage_regression(model, train_loader, device, damping=1e-5):
    """
    Compute three leverage-based membership scores for a regression model
    trained with MSE (quadratic) loss.

    For MSE loss l_i = (1/2)(y_i - f(x_i))^2:

        Hessian  : H      = Σ_j x_j x_j^T = X^T X          (d × d, closed form)
        Residual : r_i    = y_i - ŷ_i                        (scalar)
        Gradient : g_i    = ŷ_i - y_i = -r_i                 (scalar, dL_i/dŷ)
        g_full   :        = Σ_j g_j x_j = -X^T R             (d-vector, closed form)
        Leverage : h_ii   = x_i^T H^{-1} x_i                 (scalar)

    Three scores (in increasing order of approximation quality):

        1. Influence (first-order)  :  r_i^2 * h_ii
        2. Newton / ALO (LOO-exact) :  r_i^2 * h_ii / (1 - h_ii)
        3. Unconverged (g_{-i})     :  (-r_i) * (x_i^T H^{-1} g_{-i}) / (1 - h_ii)
                                       where g_{-i} = g_full + r_i * x_i

    Score 3 reduces to score 2 when the model is fully converged (g_full ≈ 0),
    and is useful when the target model may not be at a stationary point.

    Required model interface:
        model.get_features(x)  →  (B, d) feature tensor, last layer.
        model(x).squeeze(-1)   →  (B,)   scalar prediction

    Parameters
    ----------
    model        : regression model with get_features()
    train_loader : yields (x, y, idx_list) batches
    device       : torch device
    damping      : ridge term added to H for numerical stability

    Returns
    -------
    leverages_influence   : dict  idx → r_i^2 * h_ii
    leverages_newton      : dict  idx → r_i^2 * h_ii / (1 - h_ii)
    leverages_unconverged : dict  idx → unconverged g_{-i} score
    """
    model.eval()

    all_features, all_preds, all_targets, all_idx = [], [], [], []


    for x, y, idx_list in train_loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            feat  = model.get_features(x)    # (B, d) (last layer features.)
            preds = model(x).squeeze(-1)     # (B,)  scalar output
        all_features.append(feat)
        all_preds.append(preds)
        all_targets.append(y.float())
        all_idx.append(torch.tensor(idx_list, device=device))

    X = torch.cat(all_features, dim=0)
    P = torch.cat(all_preds, dim=0)  # (N,)
    Y = torch.cat(all_targets, dim=0).squeeze() # (N,)
    idx_tensor = torch.cat(all_idx, dim=0) # (N,)
    
    N, d = X.shape
    R = Y - P 
    
    hessian = X.T @ X + damping * torch.eye(d, device=device) #(d, d)
    hessian_inv = torch.linalg.inv(hessian)  #(d, d)

    g_full = -(X * R.unsqueeze(1)).sum(dim=0)
    H_inv_X = X @ hessian_inv

    leverages_influence = {}
    leverages_newton = {}
    leverages_unconverged = {}
    h_ii_mat = {}

    for i, idx in enumerate(idx_tensor):
        xi = X[i]
        ri = R[i].item() 
        gi = -ri

        # Scalar leverage: h_ii = x_i^T H^{-1} x_i
        h_ii = (H_inv_X[i] * xi).sum().item()
        idx_key = int(idx.item())
        leverages_influence[idx_key] = ri**2 * h_ii


        leverages_newton[idx_key] = ri**2 * h_ii / (1.0 - h_ii)


        g_minus_i = g_full + ri * xi # (d,)
        score_unc = gi * (H_inv_X[i] @ g_minus_i) / (1.0 - h_ii)
        leverages_unconverged[idx_key] = score_unc.item()
        h_ii_mat[idx_key] = h_ii

    return leverages_influence, leverages_newton, leverages_unconverged, h_ii_mat


def compute_full_influence(
    model: nn.Module,
    criterion: nn.Module,
    query_loader: torch.utils.data.DataLoader,
    train_loader: torch.utils.data.DataLoader,
    device: torch.device = "cuda",
    keep_layers: list[str] = None,
    iters: int = 20,
    tol: float = 1e-3,
    damping: float = 1e-2,
    method: str = "cg",
    lissa_recursions: int = 1,
    lissa_depth: int = 200,
    lissa_scale: float = 25.0,
) -> dict[int, float]:
    r"""Per-sample full-model influence-function score:

        infl_i = nabla l_i(theta)^T  H_theta^{-1}  nabla l_i(theta)

    where the Hessian H_theta is averaged over the entire ``train_loader`` and
    the gradient is the per-sample loss gradient w.r.t. all trainable
    parameters (or the subset selected via ``keep_layers``).

    Implementation notes
    --------------------
    We compute per-sample loss gradients directly via ``vmap(grad(...))`` and
    solve ``H v = g_i`` with one of two methods (``method='cg'`` or
    ``'lissa'``). CG streams the *full* ``train_loader`` per iteration; LiSSA
    (Agarwal-Bullins-Hazan / Koh-Liang) uses one mini-batch per recursion
    step and is the standard choice for influence functions on CNNs because
    it is robust to indefiniteness and has a fixed compute budget.

    To stay compatible with models that contain ``BatchNorm`` layers (which
    reject 1-D inputs), each ``vmap``ed forward pass adds a singleton batch
    axis before calling the model. The model is expected to be in ``eval``
    mode so that BN uses its running stats.

    Parameters specific to LiSSA
    ----------------------------
    lissa_recursions : int   number of independent Neumann series averaged.
    lissa_depth      : int   length of each Neumann series.
    lissa_scale      : float must exceed ``lambda_max(H_batch)`` for the
                             recurrence to contract; bigger = safer but
                             slower convergence.

    Returns
    -------
    dict mapping the dataset index of every query point to its scalar score.
    """
    assert method in ("cg", "lissa"), f"unknown method {method!r}"
    model.eval()
    is_regression = isinstance(criterion, torch.nn.MSELoss)
    n_train = len(train_loader.dataset)

    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())

    if keep_layers:
        p_train = {k: p for k, p in params.items() if any(layer in k for layer in keep_layers)}
        p_fixed = {k: p for k, p in params.items() if k not in p_train}
    else:
        p_train = {k: p for k, p in params.items() if p.requires_grad}
        p_fixed = {k: p for k, p in params.items() if not p.requires_grad}

    p_train_keys = tuple(p_train.keys())
    p_train_values = tuple(p_train.values())

    # ── per-sample loss gradient (handles BatchNorm via unsqueeze(0)) ─────
    def _single_sample_loss(p_tv, xi, yi):
        p_train_dict = dict(zip(p_train_keys, p_tv))
        all_p = {**p_train_dict, **p_fixed, **buffers}
        xi_b = xi.unsqueeze(0)
        yi_b = yi.unsqueeze(0)
        out = functional_call(model, all_p, (xi_b,))
        return criterion(out, yi_b)

    _per_sample_grad_fn = vmap(
        grad(_single_sample_loss, argnums=0), in_dims=(None, 0, 0)
    )

    # ── per-sample HVP against a Hessian computed on a (mini-batch) of train data ──
    def _single_sample_hvp(v_sample, X_train, y_train):
        def loss_fn(p_tv):
            p_train_dict = dict(zip(p_train_keys, p_tv))
            all_p = {**p_train_dict, **p_fixed, **buffers}
            out = functional_call(model, all_p, (X_train,))
            return criterion(out, y_train)

        def directional_derivative(p_tv):
            _, tangent = jvp(loss_fn, (p_tv,), (v_sample,))
            return tangent

        return grad(directional_derivative)(p_train_values)

    _per_sample_hvp_fn = vmap(_single_sample_hvp, in_dims=(0, None, None))

    # ── helpers operating on tuple-of-batched-tensor "vectors" ────────────
    def _batched_dot(v1, v2):
        out = None
        for t1, t2 in zip(v1, v2):
            prod = t1 * t2
            sum_dims = tuple(range(1, prod.ndim))
            d = torch.sum(prod, dim=sum_dims) if sum_dims else prod
            out = d if out is None else out + d
        return out

    def _update(vec, direction, step, sign=1.0):
        new = []
        for v, d in zip(vec, direction):
            view_shape = [-1] + [1] * (d.ndim - 1)
            new.append(v + sign * step.view(view_shape) * d)
        return tuple(new)

    def _hessian_vp_avg(direction):
        """H · direction averaged over the entire train_loader.

        direction is a tuple of [chunk, *params] tensors -- one direction per
        query sample. Output has the same shape.
        """
        accum = tuple(torch.zeros_like(t) for t in direction)
        n_batches = 0
        for X, y, _ in train_loader:
            X = _to_device(X, device)
            y = y.to(device)
            batch_Hp = _per_sample_hvp_fn(direction, X, y)
            with torch.no_grad():
                accum = tuple(a + h.detach() for a, h in zip(accum, batch_Hp))
            n_batches += 1
            del X, y, batch_Hp
        return tuple(a / max(n_batches, 1) for a in accum)

    def _cg_solve(b):
        """Solve H x = b (per-sample), b: tuple of [chunk, *params] tensors."""
        x = tuple(torch.zeros_like(t) for t in b)
        r = tuple(t.clone() for t in b)
        p = tuple(t.clone() for t in r)
        r_norm_sq = _batched_dot(r, r)

        for it in range(iters):
            if torch.max(r_norm_sq) < tol:
                break

            Hp = _hessian_vp_avg(p)
            Hp = tuple(h + damping * d for h, d in zip(Hp, p))

            p_Hp = _batched_dot(p, Hp)
            alpha = r_norm_sq / (p_Hp + 1e-10)

            x = _update(x, p, alpha, sign=1.0)
            r = _update(r, Hp, alpha, sign=-1.0)

            r_new_norm_sq = _batched_dot(r, r)
            beta = r_new_norm_sq / (r_norm_sq + 1e-10)
            p = _update(r, p, beta, sign=1.0)
            r_norm_sq = r_new_norm_sq
            if it % 5 == 0:
                print(f"    [cg iter {it:2d}] max r^2 = "
                      f"{r_norm_sq.max().item():.3e}  "
                      f"mean r^2 = {r_norm_sq.mean().item():.3e}")

        return x, r_norm_sq

    def _estimate_lambda_max(n_iters: int = 20, n_batches: int = 5) -> float:
        """Power iteration on a single non-batched direction to estimate
        ``lambda_max(H_batch)``. We average over a few mini-batches so the
        estimate is not pinned to a single noisy sample.
        """
        v = tuple(torch.randn_like(t) for t in p_train_values)
        # Normalise.
        norm = sum((vi * vi).sum() for vi in v).sqrt() + 1e-12
        v = tuple(vi / norm for vi in v)

        train_iter = iter(train_loader)
        lam = 0.0
        for it in range(n_iters):
            Hv_accum = tuple(torch.zeros_like(vi) for vi in v)
            for _ in range(n_batches):
                try:
                    X, y, _ = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    X, y, _ = next(train_iter)
                X = _to_device(X, device)
                y = y.to(device)

                def loss_fn(p_tv):
                    p_train_dict = dict(zip(p_train_keys, p_tv))
                    all_p = {**p_train_dict, **p_fixed, **buffers}
                    out = functional_call(model, all_p, (X,))
                    return criterion(out, y)

                def dd(p_tv):
                    _, tan = jvp(loss_fn, (p_tv,), (v,))
                    return tan

                Hv = grad(dd)(p_train_values)
                with torch.no_grad():
                    Hv_accum = tuple(a + h.detach() for a, h in zip(Hv_accum, Hv))
                del X, y, Hv

            Hv_mean = tuple(a / n_batches for a in Hv_accum)
            lam = sum((vi * hv).sum() for vi, hv in zip(v, Hv_mean)).item()
            norm = sum((hv * hv).sum() for hv in Hv_mean).sqrt() + 1e-12
            v = tuple(hv / norm for hv in Hv_mean)
        return float(abs(lam))

    def _lissa_solve(b):
        """Approximate H^{-1} b via Neumann series (Agarwal-Bullins-Hazan).

        Recurrence (one mini-batch per step):
            h_{j+1} = b + (1 - damping) * h_j - H_batch h_j / scale

        Final estimate of H^{-1} b is ``mean_r(h_T) / scale``. Provided
        ``scale > lambda_max(H_batch)`` the operator
        ``(1-damping) I - H/scale`` is contractive.

        If the effective scale is non-positive we calibrate it via power
        iteration: ``scale = max(2 * lambda_max, lissa_scale)``. This avoids
        the silent-NaN failure mode where the user's hand-picked scale is
        below the true spectral radius.
        """
        scale = lissa_scale
        if scale <= 0:
            lam = _estimate_lambda_max()
            scale = 2.0 * lam
            print(f"    [lissa] estimated lambda_max ~ {lam:.3e}, "
                  f"using scale = {scale:.3e}")

        finals = tuple(torch.zeros_like(t) for t in b)
        train_iter = iter(train_loader)

        def _next_batch():
            nonlocal train_iter
            try:
                X, y, _ = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                X, y, _ = next(train_iter)
            return _to_device(X, device), y.to(device)

        for r in range(lissa_recursions):
            h = tuple(t.clone() for t in b)
            diverged = False
            for j in tqdm(range(lissa_depth),
                          desc=f"LiSSA recursion {r+1}/{lissa_recursions}",
                          leave=False):
                X, y = _next_batch()
                Hp = _per_sample_hvp_fn(h, X, y)
                with torch.no_grad():
                    h = tuple(
                        bi + (1.0 - damping) * hi - hp / scale
                        for bi, hi, hp in zip(b, h, Hp)
                    )
                del X, y, Hp

                if (j + 1) % 25 == 0:
                    norm = _batched_dot(h, h).mean().item()
                    if not np.isfinite(norm):
                        print(f"    [lissa r{r+1} step {j+1}] DIVERGED "
                              f"(non-finite ||h||^2). Increase --lissa_scale "
                              f"or set it to -1 for auto-calibration.")
                        diverged = True
                        break
                    print(f"    [lissa r{r+1} step {j+1}] "
                          f"mean ||h||^2 = {norm:.3e}")

            if diverged:
                # Zero-out this recursion so it does not poison the average.
                continue
            with torch.no_grad():
                finals = tuple(f + hi for f, hi in zip(finals, h))

        denom = lissa_recursions * scale
        return tuple(f / denom for f in finals), None

    # ── main loop over query chunks ───────────────────────────────────────
    scores: dict[int, float] = {}

    for x_batch, y_batch, idx_batch in tqdm(query_loader, desc="Full Influence"):
        x_batch = _to_device(x_batch, device)
        y_batch = y_batch.to(device)

        # Per-sample loss gradients: tuple of [chunk, *params].
        grads = _per_sample_grad_fn(p_train_values, x_batch, y_batch)

        # H^{-1} grads via CG or LiSSA.
        if method == "cg":
            v, _ = _cg_solve(grads)
        else:
            v, _ = _lissa_solve(grads)

        # Influence scalar per sample: <g_i, v_i>.
        chunk_scores = None
        for g, vi in zip(grads, v):
            prod = g * vi
            sum_dims = tuple(range(1, prod.ndim))
            d = torch.sum(prod, dim=sum_dims) if sum_dims else prod
            chunk_scores = d if chunk_scores is None else chunk_scores + d

        # Optional 1/N rescale (drops out of any ranking but keeps the
        # magnitude in line with the last-layer scores).
        chunk_scores = (chunk_scores / n_train).detach().cpu()

        for i_idx, sc in zip(idx_batch, chunk_scores):
            scores[int(i_idx)] = float(sc)

        del grads, v, chunk_scores
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    return scores
