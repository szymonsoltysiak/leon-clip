from __future__ import annotations

import pickle
import random
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional, Tuple

import h3
import numpy as np
import pandas as pd


def _normalize_h3_key(raw_key: Any) -> Optional[str]:
    if raw_key is None:
        return None
    key = str(raw_key).strip()
    if not key:
        return None
    return key.split("_", 1)[0]


def _to_numpy_1d(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 0:
        return None
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    return arr


def _first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    lower_to_original = {c.lower(): c for c in df.columns}
    for name in candidates:
        hit = lower_to_original.get(name.lower())
        if hit is not None:
            return hit
    return None


def _normalize_target_resolutions(
    target_res: Optional[int] = None,
    target_resolutions: Optional[Iterable[int]] = None,
) -> tuple[int, ...]:
    if target_resolutions is not None:
        vals = sorted({int(r) for r in target_resolutions})
    elif target_res is not None:
        vals = [int(target_res)]
    else:
        vals = [8]

    if not vals:
        raise ValueError("At least one target resolution must be provided")
    for r in vals:
        if r < 0 or r > 15:
            raise ValueError(f"Invalid H3 resolution: {r}")
    return tuple(vals)


class EmbeddingStore:
    """SQLite-backed storage for text, image, and graph embeddings."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self._create_schema()

    def close(self) -> None:
        self.conn.close()

    def _create_schema(self) -> None:
        if self._needs_embeddings_migration():
            self.conn.executescript(
                """
                ALTER TABLE embeddings RENAME TO embeddings_old;
                CREATE TABLE embeddings (
                    sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    modality TEXT NOT NULL,
                    h3 TEXT NOT NULL,
                    dim INTEGER NOT NULL,
                    vec BLOB NOT NULL
                );
                INSERT INTO embeddings(modality, h3, dim, vec)
                SELECT modality, h3, dim, vec FROM embeddings_old;
                DROP TABLE embeddings_old;
                """
            )

        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
                modality TEXT NOT NULL,
                h3 TEXT NOT NULL,
                dim INTEGER NOT NULL,
                vec BLOB NOT NULL
            );
            CREATE TABLE IF NOT EXISTS dims (
                modality TEXT PRIMARY KEY,
                dim INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_embeddings_h3 ON embeddings(h3);
            CREATE INDEX IF NOT EXISTS idx_embeddings_modality_h3 ON embeddings(modality, h3);
            """
        )
        self.conn.commit()

    def _needs_embeddings_migration(self) -> bool:
        table = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'embeddings'"
        ).fetchone()
        if table is None:
            return False

        cols = self.conn.execute("PRAGMA table_info(embeddings)").fetchall()
        col_names = [str(c[1]) for c in cols]
        return "sample_id" not in col_names

    def clear(self) -> None:
        self.conn.executescript(
            """
            DELETE FROM embeddings;
            DELETE FROM dims;
            """
        )
        self.conn.commit()

    def upsert_embedding(self, modality: str, h3_id: str, vec: np.ndarray) -> None:
        arr = np.asarray(vec, dtype=np.float32)
        self.conn.execute(
            """
            INSERT INTO embeddings(modality, h3, dim, vec)
            VALUES (?, ?, ?, ?)
            """,
            (modality, h3_id, int(arr.shape[0]), sqlite3.Binary(arr.tobytes())),
        )

    def set_dim(self, modality: str, dim: int) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO dims(modality, dim) VALUES (?, ?)",
            (modality, int(dim)),
        )

    def get_dim(self, modality: str) -> Optional[int]:
        row = self.conn.execute(
            "SELECT dim FROM dims WHERE modality = ?",
            (modality,),
        ).fetchone()
        if row is None:
            return None
        return int(row[0])

    def get_embeddings(self, modality: str, h3_id: str) -> list[np.ndarray]:
        rows = self.conn.execute(
            "SELECT dim, vec FROM embeddings WHERE modality = ? AND h3 = ?",
            (modality, h3_id),
        ).fetchall()
        out: list[np.ndarray] = []
        for dim, raw in rows:
            vec = np.frombuffer(raw, dtype=np.float32)
            if vec.shape[0] != int(dim):
                continue
            out.append(vec)
        return out

    def get_embedding(self, modality: str, h3_id: str, strategy: str = "random") -> Optional[np.ndarray]:
        vecs = self.get_embeddings(modality, h3_id)
        if not vecs:
            return None

        if strategy == "first":
            return vecs[0]
        if strategy == "mean":
            return np.mean(vecs, axis=0, dtype=np.float32)
        if strategy == "random":
            return random.choice(vecs)
        raise ValueError(f"Unknown embedding selection strategy: {strategy}")

    def _anchor_for_resolutions(self, h3_id: str, target_resolutions: tuple[int, ...]) -> Optional[str]:
        res = h3.get_resolution(h3_id)
        if res in target_resolutions:
            return h3_id

        lower = [r for r in target_resolutions if r <= res]
        if not lower:
            return None
        target = max(lower)
        if target == res:
            return h3_id
        return h3.cell_to_parent(h3_id, target)

    def get_h3_ids(
        self,
        target_res: Optional[int] = None,
        target_resolutions: Optional[Iterable[int]] = None,
    ) -> list[str]:
        targets = _normalize_target_resolutions(target_res=target_res, target_resolutions=target_resolutions)
        rows = self.conn.execute(
            "SELECT DISTINCT h3 FROM embeddings"
        ).fetchall()
        output: set[str] = set()
        for (h3_id,) in rows:
            try:
                anchor = self._anchor_for_resolutions(h3_id, targets)
                if anchor is not None:
                    output.add(anchor)
            except Exception:
                continue
        return sorted(output)

    def get_h3_ids_for_modality(
        self,
        modality: str,
        target_res: Optional[int] = None,
        target_resolutions: Optional[Iterable[int]] = None,
    ) -> list[str]:
        targets = _normalize_target_resolutions(target_res=target_res, target_resolutions=target_resolutions)
        rows = self.conn.execute(
            "SELECT h3 FROM embeddings WHERE modality = ?",
            (modality,),
        ).fetchall()
        output: set[str] = set()
        for (h3_id,) in rows:
            try:
                anchor = self._anchor_for_resolutions(h3_id, targets)
                if anchor is not None:
                    output.add(anchor)
            except Exception:
                continue
        return sorted(output)

    def has_embedding(self, modality: str, h3_id: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM embeddings WHERE modality = ? AND h3 = ? LIMIT 1",
            (modality, h3_id),
        ).fetchone()
        return row is not None

    def commit(self) -> None:
        self.conn.commit()

    def _iter_text(self, folder: Path) -> Iterator[Tuple[str, np.ndarray]]:
        for parquet_path in sorted(folder.glob("*.parquet")):
            try:
                df = pd.read_parquet(parquet_path)
            except Exception:
                continue
            h3_col = _first_existing_column(df, ["h3", "h3_index", "hex_id", "h3_id"])
            emb_col = _first_existing_column(df, ["embedding", "emb", "vector"])
            if h3_col is None or emb_col is None:
                continue
            for raw_h3, raw_vec in zip(df[h3_col], df[emb_col], strict=False):
                h3_id = _normalize_h3_key(raw_h3)
                vec = _to_numpy_1d(raw_vec)
                if h3_id is None or vec is None:
                    continue
                yield h3_id, vec

    def _iter_pickle(self, folder: Path) -> Iterator[Tuple[str, np.ndarray]]:
        for pkl_path in sorted(folder.glob("*.pkl")):
            try:
                with pkl_path.open("rb") as handle:
                    obj = pickle.load(handle)
            except Exception:
                continue

            if isinstance(obj, dict):
                iterable = obj.items()
            elif isinstance(obj, pd.DataFrame):
                h3_col = _first_existing_column(obj, ["h3", "h3_index", "hex_id", "h3_id"])
                emb_col = _first_existing_column(obj, ["embedding", "emb", "vector"])
                if h3_col is None or emb_col is None:
                    continue
                iterable = zip(obj[h3_col], obj[emb_col], strict=False)
            else:
                continue

            for raw_h3, raw_vec in iterable:
                h3_id = _normalize_h3_key(raw_h3)
                vec = _to_numpy_1d(raw_vec)
                if h3_id is None or vec is None:
                    continue
                yield h3_id, vec

    def build_from_embeddings_root(self, embeddings_root: str | Path, rebuild: bool = False) -> Dict[str, int]:
        root = Path(embeddings_root)
        modality_dirs = ("text_embeddings", "image_embeddings", "graph_embeddings")
        if not any((root / name).exists() for name in modality_dirs):
            parent = root.parent
            if any((parent / name).exists() for name in modality_dirs):
                root = parent
        if rebuild:
            self.clear()

        modal_to_iter = {
            "text": self._iter_text(root / "text_embeddings"),
            "image": self._iter_pickle(root / "image_embeddings"),
            "graph": self._iter_pickle(root / "graph_embeddings"),
        }

        dims: Dict[str, int] = {}
        for modality, iterator in modal_to_iter.items():
            first_dim: Optional[int] = None
            for h3_id, vec in iterator:
                self.upsert_embedding(modality, h3_id, vec)
                if first_dim is None:
                    first_dim = int(vec.shape[0])
            if first_dim is not None:
                self.set_dim(modality, first_dim)
                dims[modality] = first_dim

        self.commit()
        if not dims:
            raise ValueError(f"No embeddings found under {root}")
        return dims
