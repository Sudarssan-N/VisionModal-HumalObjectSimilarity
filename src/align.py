"""Core alignment machinery shared by Phase 3 (per-model) and Phase 4 (transfer).

The odd-one-out objective follows the SPoSE/VICE formulation: for a triplet with
the chosen-similar pair (a, b) and odd-one-out o, define pairwise similarities as
dot products in the transformed space and apply a 3-way softmax over the pairs
(a,b), (a,o), (b,o). The target is the (a,b) pair, so the model is trained to make
the human-chosen pair the most similar. Odd-one-out accuracy == how often the
(a,b) pair has the highest similarity (equivalently: o is the predicted odd one).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Feature preprocessing
# --------------------------------------------------------------------------- #
def standardize(X: np.ndarray, mean=None, std=None):
    """Per-dimension z-score. Returns (Xs, mean, std)."""
    if mean is None:
        mean = X.mean(axis=0, keepdims=True)
    if std is None:
        std = X.std(axis=0, keepdims=True) + 1e-6
    return (X - mean) / std, mean, std


def pca_reduce(X: np.ndarray, n_components: int, seed: int = 0) -> np.ndarray:
    """PCA-reduce standardized features to a shared dimensionality."""
    from sklearn.decomposition import PCA

    n_components = min(n_components, X.shape[1], X.shape[0])
    Xs, _, _ = standardize(X)
    return PCA(n_components=n_components, random_state=seed).fit_transform(Xs)


# --------------------------------------------------------------------------- #
# Linear transform
# --------------------------------------------------------------------------- #
def _normalize(Z: torch.Tensor) -> torch.Tensor:
    return Z / (Z.norm(dim=-1, keepdim=True) + 1e-8)


class LinearAligner(torch.nn.Module):
    def __init__(self, d_in: int, d_out: int | None = None, init_identity: bool = True):
        super().__init__()
        d_out = d_out or d_in
        W = torch.empty(d_in, d_out)
        if init_identity and d_in == d_out:
            torch.nn.init.eye_(W)
        else:
            torch.nn.init.orthogonal_(W)
        self.W = torch.nn.Parameter(W)
        # Learnable temperature on the cosine logits (CLIP-style), clamped.
        self.logit_scale = torch.nn.Parameter(torch.log(torch.tensor(10.0)))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return X @ self.W

    def scale(self) -> torch.Tensor:
        return self.logit_scale.clamp(max=np.log(100.0)).exp()


def _triplet_logits(Zn: torch.Tensor, t: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Zn: (N, d) L2-normalized embeddings. t: (B, 3) [a, b, odd]. Returns (B, 3)
    cosine logits ordered [s_ab, s_ao, s_bo]; column 0 is the correct pair."""
    a, b, o = t[:, 0], t[:, 1], t[:, 2]
    za, zb, zo = Zn[a], Zn[b], Zn[o]
    s_ab = (za * zb).sum(-1)
    s_ao = (za * zo).sum(-1)
    s_bo = (zb * zo).sum(-1)
    return scale * torch.stack([s_ab, s_ao, s_bo], dim=1)


@torch.no_grad()
def accuracy(W: torch.Tensor, Xs: torch.Tensor, triplets: torch.Tensor,
             chunk: int = 200_000) -> float:
    """Odd-one-out accuracy of transform W on a triplet set (cosine similarity)."""
    Zn = _normalize(Xs @ W)
    correct = 0
    n = len(triplets)
    for s in range(0, n, chunk):
        logits = _triplet_logits(Zn, triplets[s:s + chunk])
        correct += int((logits.argmax(1) == 0).sum().item())
    return correct / n


def train_aligner(
    X: np.ndarray,
    train_t: np.ndarray,
    val_t: np.ndarray,
    d_out: int | None = None,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    epochs: int = 30,
    batch_size: int = 8192,
    patience: int = 4,
    max_train: int | None = None,
    device: str = "cpu",
    seed: int = 0,
    verbose: bool = True,
):
    """Learn a linear transform W maximizing odd-one-out agreement.

    Returns dict with W (np.ndarray), best_val, baseline_val (W=identity),
    and the standardization stats.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    Xs_np, mean, std = standardize(X)
    Xs = torch.tensor(Xs_np, dtype=torch.float32, device=device)
    d_in = Xs.shape[1]

    tr = torch.tensor(train_t, dtype=torch.long, device=device)
    va = torch.tensor(val_t, dtype=torch.long, device=device)
    if max_train is not None and len(tr) > max_train:
        idx = torch.randperm(len(tr), device=device)[:max_train]
        tr = tr[idx]

    model = LinearAligner(d_in, d_out, init_identity=(d_out in (None, d_in))).to(device)
    opt = torch.optim.Adam([
        {"params": [model.W], "weight_decay": weight_decay},
        {"params": [model.logit_scale], "weight_decay": 0.0},
    ], lr=lr)

    baseline_val = accuracy(model.W.detach(), Xs, va)
    if verbose:
        print(f"    baseline (init) val acc: {baseline_val:.4f}")

    best_val, best_W, bad = baseline_val, model.W.detach().clone(), 0
    n = len(tr)
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        running = 0.0
        for s in range(0, n, batch_size):
            b = tr[perm[s:s + batch_size]]
            Zn = _normalize(model(Xs))
            logits = _triplet_logits(Zn, b, scale=model.scale())
            loss = F.cross_entropy(logits, torch.zeros(len(b), dtype=torch.long,
                                                       device=device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item() * len(b)
        val_acc = accuracy(model.W.detach(), Xs, va)
        if verbose:
            print(f"    epoch {epoch + 1:2d}  loss {running / n:.4f}  "
                  f"val {val_acc:.4f}")
        if val_acc > best_val + 1e-4:
            best_val, best_W, bad = val_acc, model.W.detach().clone(), 0
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"    early stop @ epoch {epoch + 1}")
                break

    return {
        "W": best_W.cpu().numpy(),
        "best_val": float(best_val),
        "baseline_val": float(baseline_val),
        "mean": mean,
        "std": std,
    }
