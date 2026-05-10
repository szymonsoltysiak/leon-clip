"""GeometryAnalyzer: stateless math utilities for embedding analysis."""
from typing import Dict, List, Optional, Tuple
import numpy as np
import math
import warnings

try:
    import torch
    _TORCH = True
except Exception:
    torch = None
    _TORCH = False

def _to_numpy(x):
    if _TORCH and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def covariance_eigstats(X: np.ndarray) -> Dict:
    """Compute covariance, eigendecomposition and anisotropy/PR stats.

    Returns dict with keys: eigenvalues (desc), isotropy, participation_ratio,
    explained_variance_ratio, effective_dim
    """
    X = _to_numpy(X)
    if X.ndim != 2:
        raise ValueError("X must be 2D (N, D)")
    # center
    Xc = X - X.mean(axis=0, keepdims=True)
    cov = np.cov(Xc, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    # sort descending
    idx = np.argsort(vals)[::-1]
    vals = vals[idx]
    explained = vals / (vals.sum() + 1e-12)
    isotropy = float(vals[-1] / (vals[0] + 1e-12))
    pr = float((vals.sum() ** 2) / (np.sum(vals ** 2) + 1e-12))
    # effective dim (participation ratio)
    return {
        "eigenvalues": vals,
        "explained_variance_ratio": explained,
        "isotropy": isotropy,
        "participation_ratio": pr,
        "effective_dim": pr,
    }

def _pairwise_cosine_similarity(X: np.ndarray, chunk_size: int = 4096):
    X = _to_numpy(X)
    # normalize
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    Xn = X / norms
    N = Xn.shape[0]
    # compute pairwise cosine (returns upper triangular vectorizable?)
    # We'll return full matrix; caller can reduce memory using chunks.
    S = np.zeros((N, N), dtype=np.float32)
    for i0 in range(0, N, chunk_size):
        i1 = min(N, i0 + chunk_size)
        S[i0:i1] = Xn[i0:i1] @ Xn.T
    return S

def uniformity_gaussian(X: np.ndarray, temperatures: List[float] = [0.5, 1.0, 2.0, 5.0, 10.0], chunk_size: int = 4096) -> Dict:
    """Compute Wang & Isola uniformity values for given temperatures.

    The returned values are signed like the paper's objective:
    log E[exp(-t*||x - y||^2)]
    so they are typically non-positive.
    """
    X = _to_numpy(X)
    # L2-normalize
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    Xn = X / norms
    N = Xn.shape[0]
    results = {}
    # compute pairwise squared distances in chunks
    for t in temperatures:
        Ksum = 0.0
        pair_count = 0
        for i0 in range(0, N, chunk_size):
            i1 = min(N, i0 + chunk_size)
            block = Xn[i0:i1]
            # cosine similarities
            sim = block @ Xn.T
            # squared Euclidean on unit sphere: 2 - 2*cos
            sqdist = np.clip(2.0 - 2.0 * sim, 0.0, None)
            K = np.exp(-float(t) * sqdist)
            Ksum += float(K.sum())
            pair_count += int(K.size)
        mean_kernel = max(Ksum / max(pair_count, 1), 1e-12)
        results[float(t)] = float(np.log(mean_kernel))
    return results

def haversine_distance_matrix(coords: np.ndarray, chunk_size: int = 4096) -> np.ndarray:
    """Compute full pairwise haversine distances (kilometers) for coords (N,2).

    Uses chunking to limit memory.
    """
    coords = _to_numpy(coords)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must be (N,2) array of lat,lon in degrees")
    R = 6371.0
    lat = np.radians(coords[:, 0])
    lon = np.radians(coords[:, 1])
    N = coords.shape[0]
    D = np.zeros((N, N), dtype=np.float32)
    for i0 in range(0, N, chunk_size):
        i1 = min(N, i0 + chunk_size)
        dlat = lat[i0:i1, None] - lat[None, :]
        dlon = lon[i0:i1, None] - lon[None, :]
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat[i0:i1, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2.0) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(0.0, 1.0 - a)))
        D[i0:i1] = (R * c).astype(np.float32)
    return D

def geographic_correlation(embeddings: np.ndarray, coords: np.ndarray, chunk_size: int = 4096) -> Dict:
    """Compute correlation between physical distance and semantic similarity.

    Returns dict with pearson and spearman coefficients.
    """
    from scipy import stats

    embeddings = _to_numpy(embeddings)
    coords = _to_numpy(coords)
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be (N, D)")
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must be (N, 2)")
    if embeddings.shape[0] != coords.shape[0]:
        raise ValueError("embeddings and coords must have the same number of rows")

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
    Xn = embeddings / norms
    sim_mat = np.empty((Xn.shape[0], Xn.shape[0]), dtype=np.float32)
    for i0 in range(0, Xn.shape[0], chunk_size):
        i1 = min(Xn.shape[0], i0 + chunk_size)
        sim_mat[i0:i1] = Xn[i0:i1] @ Xn.T

    dist_mat = haversine_distance_matrix(coords, chunk_size=chunk_size)
    iu = np.triu_indices(Xn.shape[0], k=1)
    sims = sim_mat[iu]
    dists = dist_mat[iu]
    # remove NaNs
    mask = np.isfinite(sims) & np.isfinite(dists)
    if mask.sum() == 0:
        return {"pearson": None, "spearman": None}
    sims = sims[mask]
    dists = dists[mask]
    pearson = float(stats.pearsonr(dists, sims)[0])
    spearman = float(stats.spearmanr(dists, sims)[0])
    return {"pearson": pearson, "spearman": spearman}
