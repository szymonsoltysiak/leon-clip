"""Visualization helpers for alignment analysis."""
from typing import Dict, Optional
import os
import numpy as np
import math

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
except Exception:
    plt = None
    LogNorm = None

try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
except Exception:
    TSNE = None
    PCA = None

def save_figure(fig, outpath: str):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath, bbox_inches="tight", dpi=150)
    plt.close(fig)

def plot_spectrum(eigenvalues: np.ndarray, stats: Dict, outpath: str):
    if plt is None:
        raise RuntimeError("matplotlib required for plotting")
    vals = np.asarray(eigenvalues)
    explained = vals / (vals.sum() + 1e-12)
    x = np.arange(1, len(vals) + 1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(x, explained, marker=".")
    ax.set_xlabel("Eigenvalue index")
    ax.set_ylabel("Explained variance ratio")
    ax.grid(True, which="both", ls="--", alpha=0.3)
    info = (
        f"EffectiveDim={stats.get('effective_dim'):.2f}\n"
        f"Isotropy={stats.get('isotropy'):.2e}\n"
        f"Top1={(explained[0]*100):.2f}%"
    )
    ax.text(0.95, 0.95, info, transform=ax.transAxes, ha="right", va="top",
            bbox=dict(facecolor="white", alpha=0.7))
    save_figure(fig, outpath)

def tsne_multimodal(modality_dict: Dict[str, np.ndarray], outpath: str, sample_limit: int = 5000):
    if plt is None:
        raise RuntimeError("matplotlib required for plotting")
    # collect and label
    arrays = []
    labels = []
    for m, X in modality_dict.items():
        if X is None:
            continue
        arrays.append(X)
        labels.extend([m] * X.shape[0])
    if not arrays:
        raise RuntimeError("No modalities to plot")
    X = np.vstack(arrays)
    N = X.shape[0]
    if N > sample_limit:
        idx = np.random.choice(N, sample_limit, replace=False)
        Xs = X[idx]
        labs = [labels[i] for i in idx]
    else:
        Xs = X
        labs = labels
    # normalize
    norms = np.linalg.norm(Xs, axis=1, keepdims=True) + 1e-12
    Xs = Xs / norms
    if TSNE is not None:
        emb = TSNE(n_components=2, metric="cosine", init="pca", random_state=0).fit_transform(Xs)
    elif PCA is not None:
        emb = PCA(n_components=2).fit_transform(Xs)
    else:
        raise RuntimeError("sklearn required for t-SNE or PCA fallback")
    fig, ax = plt.subplots(figsize=(6, 6))
    modalities = sorted(list(set(labs)))
    cmap = plt.get_cmap("tab10")
    for i, m in enumerate(modalities):
        mask = [l == m for l in labs]
        ax.scatter(emb[mask, 0], emb[mask, 1], s=6, alpha=0.7, label=m, color=cmap(i))
    ax.legend()
    ax.set_title("Multimodal embedding visualization")
    save_figure(fig, outpath)

def geographic_density_and_trend(physical_dists: np.ndarray, similarities: np.ndarray, outpath: str, distance_unit: str = "deg"):
    if plt is None:
        raise RuntimeError("matplotlib required for plotting")
    
    x = physical_dists
    y = similarities
    
    # 1. FIX: Changed figsize from (12, 4) to (12, 5) to match the taller aspect ratio
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. Hexbin
    hb = axs[0].hexbin(x, y, gridsize=50, cmap='inferno', mincnt=1, bins='log')
    
    # 2. FIX: Matched capitalization and labels exactly
    axs[0].set_xlabel(f"Geographic Distance ({distance_unit})")
    axs[0].set_ylabel("Cosine Similarity")
    axs[0].set_title("Density: Distance vs Similarity")
    fig.colorbar(hb, ax=axs[0], label='Log Count')

    # 2. Trend
    bins = np.linspace(min(x), max(x), 20)
    bin_indices = np.digitize(x, bins)
    bin_means = []
    bin_centers = []
    
    for i in range(1, len(bins)):
        mask = bin_indices == i
        if np.any(mask):
            bin_means.append(np.mean(y[mask]))
            bin_centers.append(0.5 * (bins[i] + bins[i-1]))
            
    axs[1].plot(bin_centers, bin_means, marker='o', color='blue')
    
    # 3. FIX: Fixed typo "similarit" and matched exact capitalization
    axs[1].set_title("Trend: Avg Similarity vs Distance")
    axs[1].set_xlabel(f"Geographic Distance ({distance_unit})")
    axs[1].set_ylabel("Average Cosine Similarity")
    axs[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_figure(fig, outpath)
