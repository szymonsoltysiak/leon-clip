import os
import numpy as np
import torch

class ModalityRegistry:
    """Registry mapping modality names to lazy loader callables.

    Expected storage layout (convention):
      data/<modality>_embeddings/ (e.g. data/image_embeddings)
        - pre_projection.npy or post_projection.npy or embeddings.npy
      graph coordinates: data/graph_embeddings/coords.npy (N,2)

    Loaders only read files when called.
    """

    def __init__(self, root="data"):
        self.root = root
        self._map = {
            "image": self._make_loader("image"),
            "text": self._make_loader("text"),
            "graph": self._make_loader("graph"),
            "all": self._load_all,
        }

    def _make_loader(self, modality):
        def loader(stage="post"):
            # Check common file names
            base = os.path.join(self.root, f"{modality}_embeddings")
            candidates = [
                os.path.join(base, f"{stage}_projection.npy"),
                os.path.join(base, f"{stage}.npy"),
                os.path.join(base, "embeddings.npy"),
                os.path.join(base, f"{stage}_embeddings.npy"),
            ]
            for p in candidates:
                if os.path.exists(p):
                    return np.load(p)
            # try torch file
            for p in candidates:
                pt = p.replace(".npy", ".pt")
                if os.path.exists(pt):
                    return torch.load(pt)
            raise FileNotFoundError(
                f"No embeddings found for modality={modality} in {base}. Tried: {candidates}"
            )

        return loader

    def _load_all(self, stage="post"):
        out = {}
        for m in ("image", "text", "graph"):
            try:
                out[m] = self._make_loader(m)(stage=stage)
            except FileNotFoundError:
                out[m] = None
        return out

    def get(self, modality, stage="post"):
        if modality not in self._map:
            raise KeyError(f"Unknown modality: {modality}")
        return self._map[modality](stage=stage)

    def load_coords(self):
        p = os.path.join(self.root, "graph_embeddings", "coords.npy")
        if os.path.exists(p):
            return np.load(p)
        return None
