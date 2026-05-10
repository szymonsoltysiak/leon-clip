import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Reuse code from alignment/geometry.py
from alignment.geometry import covariance_eigstats, uniformity_gaussian

# Reuse exact loading components from eval_alignment.py / triple_encoder
from triple_encoder import EmbeddingStore, H3TripleDataset
from triple_encoder.benchmarks.runner import (
    _build_model_from_checkpoint,
    _collect_pre_projection_features,
)

def load_raw_embeddings(dataset: H3TripleDataset, h3_values: list, max_samples: int) -> dict:
    """
    Module 1: Data Loading
    Loads raw (unprojected) embeddings exactly like eval_alignment.py pre-stage.
    """
    h3_values = h3_values[:max_samples]
    raw_data = {}
    for modality in ["text", "image", "graph"]:
        collected, _ = _collect_pre_projection_features(
            dataset,
            h3_values,
            [modality],
            progress_desc=f"Loading raw embeddings ({modality})",
        )
        raw_data[modality] = torch.from_numpy(collected)
    return raw_data

def apply_projection_heads(model, raw_embeddings: dict, device: torch.device) -> dict:
    """
    Module 2: The Projection Pipeline
    Applies learned projection heads to map raw embeddings into a shared space.
    Uses the TriModalCLIP model loaded from the checkpoint.
    """
    model.eval()
    projected = {}
    with torch.no_grad():
        if "text" in raw_embeddings:
            proj = model.text_head(raw_embeddings["text"].to(device))
            projected["text"] = torch.nn.functional.normalize(proj, p=2, dim=1).cpu()
        if "image" in raw_embeddings:
            proj = model.image_head(raw_embeddings["image"].to(device))
            projected["image"] = torch.nn.functional.normalize(proj, p=2, dim=1).cpu()
        if "graph" in raw_embeddings:
            proj = model.graph_head(raw_embeddings["graph"].to(device))
            projected["graph"] = torch.nn.functional.normalize(proj, p=2, dim=1).cpu()
    return projected

def calculate_metrics(projected_embeddings: dict, output_file: str = "evaluation_metrics.json"):
    """
    Module 3: Metrics Calculation
    Computes pairwise alignment (cosine similarity of matches) and modality gap (l2 distance of centroids).
    Also reuses alignment/geometry.py to compute Wang & Isola uniformity and covariance_eigstats.
    """
    text_emb = projected_embeddings['text'].numpy()
    image_emb = projected_embeddings['image'].numpy()
    graph_emb = projected_embeddings['graph'].numpy()
    
    # 1. Pairwise Alignment (mean cosine similarity of matched pairs i.e. diagonal)
    align_ti = float(np.mean(np.sum(text_emb * image_emb, axis=1)))
    align_tg = float(np.mean(np.sum(text_emb * graph_emb, axis=1)))
    align_ig = float(np.mean(np.sum(image_emb * graph_emb, axis=1)))
    
    # 2. Modality Gap (Centroid Distance)
    centroid_text = np.mean(text_emb, axis=0)
    centroid_image = np.mean(image_emb, axis=0)
    centroid_graph = np.mean(graph_emb, axis=0)
    
    gap_ti = float(np.linalg.norm(centroid_text - centroid_image))
    gap_tg = float(np.linalg.norm(centroid_text - centroid_graph))
    gap_ig = float(np.linalg.norm(centroid_image - centroid_graph))

    metrics = {
        "alignment": {
            "text_image": align_ti,
            "text_graph": align_tg,
            "image_graph": align_ig
        },
        "modality_gap": {
            "text_image": gap_ti,
            "text_graph": gap_tg,
            "image_graph": gap_ig
        },
        "geometry": {}
    }

    # Reusing code from alignment folder:
    for mod, emb in [("text", text_emb), ("image", image_emb), ("graph", graph_emb)]:
        stats = covariance_eigstats(emb)
        unif = uniformity_gaussian(emb)
        metrics["geometry"][mod] = {
            "isotropy": float(stats["isotropy"]),
            "effective_dim": float(stats["effective_dim"]),
            "uniformity": {str(k): float(v) for k, v in unif.items()}
        }
    
    # Dump metrics using proper encoding
    with open(output_file, 'w', encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
        
    print(f"Metrics saved to {output_file}")
    return metrics

def visualize_embeddings(projected_embeddings: dict, output_file: str = "modality_gap_visualization.png"):
    """
    Module 4: Visualization
    Reduces dimensionality via t-SNE and plots the embedding space along with centroids.
    """
    text_emb = projected_embeddings['text'].numpy()
    image_emb = projected_embeddings['image'].numpy()
    graph_emb = projected_embeddings['graph'].numpy()
    
    max_samples = min(30000, text_emb.shape[0])
    text_sub = text_emb[:max_samples]
    image_sub = image_emb[:max_samples]
    graph_sub = graph_emb[:max_samples]
    
    all_emb = np.concatenate([text_sub, image_sub, graph_sub], axis=0)
    
    print("Running t-SNE... this might take a moment.")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    all_2d = tsne.fit_transform(all_emb)
    
    text_2d = all_2d[:max_samples]
    image_2d = all_2d[max_samples:2*max_samples]
    graph_2d = all_2d[2*max_samples:]
    
    plt.figure(figsize=(10, 8))
    
    plt.scatter(text_2d[:, 0], text_2d[:, 1], alpha=0.3, label='Text Points', color='blue', s=10)
    plt.scatter(image_2d[:, 0], image_2d[:, 1], alpha=0.3, label='Image Points', color='red', s=10)
    plt.scatter(graph_2d[:, 0], graph_2d[:, 1], alpha=0.3, label='Graph Points', color='green', s=10)
    
    centroid_text_2d = np.mean(text_2d, axis=0)
    centroid_image_2d = np.mean(image_2d, axis=0)
    centroid_graph_2d = np.mean(graph_2d, axis=0)
    
    plt.scatter(*centroid_text_2d, marker='X', s=200, color='darkblue', edgecolors='white', label='Text Centroid')
    plt.scatter(*centroid_image_2d, marker='X', s=200, color='darkred', edgecolors='white', label='Image Centroid')
    plt.scatter(*centroid_graph_2d, marker='X', s=200, color='darkgreen', edgecolors='white', label='Graph Centroid')
    
    plt.title('Modality Gap Visualization (t-SNE)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Visualization saved to {output_file}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True, help="Path to TriModalCLIP checkpoint")
    p.add_argument("--embeddings-root", default="data")
    p.add_argument("--store-db", default="data/embeddings/embeddings.sqlite")
    p.add_argument("--max-samples", type=int, default=30000)
    p.add_argument("--output-json", type=str, default="evaluation_metrics.json")
    p.add_argument("--output-plot", type=str, default="modality_gap_visualization.png")
    args = p.parse_args()

    print("Starting Modality Gap Evaluation Pipeline...")
    store_path = Path(args.store_db)

    # Initialize EmbeddingStore
    with EmbeddingStore(store_path, read_only=True, initialize_schema=False) as store:
        dataset = H3TripleDataset(store=store, active_modalities=("text", "image", "graph"))
        if len(dataset) == 0:
            raise RuntimeError(f"No H3 samples found under {args.embeddings_root}")

        modalities = ("text", "image", "graph")
        presence = dataset._exact_presence_map
        h3_values = [h for h in dataset.h3_ids if all(presence[m][h] for m in modalities)]
        if not h3_values:
            raise RuntimeError("No H3 cells have all modalities present; cannot compute pairwise alignment")
        print(f"Filtered to {len(h3_values)} cells with text+image+graph present (of {len(dataset.h3_ids)})")
        random.Random(42).shuffle(h3_values)

        print("\nLoading checkpoint...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = _build_model_from_checkpoint(Path(args.checkpoint), dataset, device=device, use_ema=True)

        print("\n1. Loading raw embeddings")
        raw_data = load_raw_embeddings(dataset, h3_values, max_samples=args.max_samples)

    print("\n2. Applying projection heads from checkpoint")
    projected_data = apply_projection_heads(model, raw_data, device=device)

    print("\n3. Calculating metrics (incorporating alignment code)")
    metrics = calculate_metrics(projected_data, output_file=args.output_json)
    print(json.dumps(metrics, indent=2))

    print("\n4. Generating visualization")
    visualize_embeddings(projected_data, output_file=args.output_plot)

    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    main()
