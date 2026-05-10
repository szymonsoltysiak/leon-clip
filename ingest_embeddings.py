"""Ingest embeddings from `data/{text,image,graph}_embeddings/` into the SQLite store.

The store uses `INSERT OR IGNORE` with a UNIQUE INDEX on `(modality, h3, vec_hash)`,
so re-running this script is safe and only adds new vectors. Pass `--rebuild` to
clear the store first.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from triple_encoder import EmbeddingStore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings-root", type=str, default="data")
    parser.add_argument("--store-db", type=str, default="data/embeddings/embeddings.sqlite")
    parser.add_argument("--rebuild", action="store_true", help="Drop existing rows before ingesting")
    args = parser.parse_args(argv)

    root = Path(args.embeddings_root)
    if not root.exists():
        print(f"embeddings root not found: {root}", file=sys.stderr)
        return 1

    with EmbeddingStore(args.store_db) as store:
        dims = store.build_from_embeddings_root(args.embeddings_root, rebuild=args.rebuild)
        for modality in ("text", "image", "graph"):
            count = store.count_embeddings(modality)
            cells = len(store.get_cells_for_modality(modality))
            dim = dims.get(modality)
            print(f"{modality}: rows={count} cells={cells} dim={dim}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
