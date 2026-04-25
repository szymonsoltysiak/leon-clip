from __future__ import annotations

import hashlib
import pickle
import random
import sqlite3
import warnings
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


def _iter_numpy_vectors(value: Any) -> Iterator[np.ndarray]:
    if value is None:
        return
    if isinstance(value, np.ndarray):
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim == 1:
            yield arr.reshape(-1)
            return
        if arr.ndim == 2:
            for row in arr:
                yield np.asarray(row, dtype=np.float32).reshape(-1)
            return

    if isinstance(value, (list, tuple)):
        if not value:
            return
        first = value[0]
        if isinstance(first, (list, tuple, np.ndarray)):
            for row in value:
                vec = _to_numpy_1d(row)
                if vec is not None:
                    yield vec
            return

    vec = _to_numpy_1d(value)
    if vec is not None:
        yield vec


def _first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    lower_to_original = {column.lower(): column for column in df.columns}
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
        values = sorted({int(resolution) for resolution in target_resolutions})
    elif target_res is not None:
        values = [int(target_res)]
    else:
        values = [8]

    if not values:
        raise ValueError("At least one target resolution must be provided")
    for resolution in values:
        if resolution < 0 or resolution > 15:
            raise ValueError(f"Invalid H3 resolution: {resolution}")
    return tuple(values)


def _is_valid_h3_cell(h3_id: str) -> bool:
    try:
        return bool(h3.is_valid_cell(h3_id))
    except Exception:
        return False


class EmbeddingStore:
    """SQLite-backed storage for text, image, and graph embeddings."""

    def __init__(self, db_path: str | Path, *, read_only: bool = False, initialize_schema: bool = True) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._read_only = bool(read_only)

        if self._read_only:
            if not self.db_path.exists():
                raise FileNotFoundError(f"Read-only embedding store does not exist: {self.db_path}")
            uri = f"{self.db_path.resolve().as_uri()}?mode=ro"
            self._conn = sqlite3.connect(uri, uri=True)
            self._conn.execute("PRAGMA query_only = ON;")
        else:
            self._conn = sqlite3.connect(str(self.db_path), timeout=30.0)
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA synchronous=NORMAL;")
            self._conn.execute("PRAGMA busy_timeout=30000;")
            if initialize_schema:
                self._create_schema()

    def __enter__(self) -> EmbeddingStore:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @property
    def conn(self) -> sqlite3.Connection:
        return self._conn

    def close(self) -> None:
        if self._conn is None:
            return
        try:
            self._conn.commit()
        finally:
            self._conn.close()
            self._conn = None  # type: ignore[assignment]

    def commit(self) -> None:
        self.conn.commit()

    def _create_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
                modality TEXT NOT NULL,
                h3 TEXT NOT NULL,
                dim INTEGER NOT NULL,
                vec_hash TEXT NOT NULL,
                vec BLOB NOT NULL
            );
            CREATE TABLE IF NOT EXISTS dims (
                modality TEXT PRIMARY KEY,
                dim INTEGER NOT NULL
            );
            """
        )
        self._migrate_embeddings_schema_if_needed()
        self._deduplicate_embeddings()
        self.conn.execute("DROP INDEX IF EXISTS idx_embeddings_unique_modality_h3")
        self.conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_embeddings_unique_modality_h3_hash ON embeddings(modality, h3, vec_hash)"
        )
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_modality_h3 ON embeddings(modality, h3)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_h3 ON embeddings(h3)")
        self.conn.commit()

    def _migrate_embeddings_schema_if_needed(self) -> None:
        columns = {
            row[1]: row
            for row in self.conn.execute("PRAGMA table_info(embeddings)").fetchall()
        }
        if not columns:
            return
        if "vec_hash" in columns:
            return

        self.conn.executescript(
            """
            ALTER TABLE embeddings RENAME TO embeddings_legacy;
            CREATE TABLE embeddings (
                sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
                modality TEXT NOT NULL,
                h3 TEXT NOT NULL,
                dim INTEGER NOT NULL,
                vec_hash TEXT NOT NULL,
                vec BLOB NOT NULL
            );
            """
        )

        rows = self.conn.execute(
            "SELECT modality, h3, dim, vec FROM embeddings_legacy ORDER BY sample_id"
        ).fetchall()
        for modality, h3_id, dim, raw_vec in rows:
            vec = np.frombuffer(raw_vec, dtype=np.float32)
            if vec.shape[0] != int(dim):
                continue
            vec = np.asarray(vec, dtype=np.float32).reshape(-1)
            vec_hash = self._vector_hash(vec)
            self.conn.execute(
                """
                INSERT OR IGNORE INTO embeddings(modality, h3, dim, vec_hash, vec)
                VALUES (?, ?, ?, ?, ?)
                """,
                (modality, h3_id, int(vec.shape[0]), vec_hash, sqlite3.Binary(vec.tobytes())),
            )

        self.conn.executescript(
            """
            DROP TABLE embeddings_legacy;
            DROP INDEX IF EXISTS idx_embeddings_unique_modality_h3;
            """
        )

    @staticmethod
    def _vector_hash(vec: np.ndarray) -> str:
        normalized = np.asarray(vec, dtype=np.float32).reshape(-1)
        return hashlib.sha1(normalized.tobytes()).hexdigest()

    def _deduplicate_embeddings(self) -> None:
        self.conn.executescript(
            """
            DELETE FROM embeddings
            WHERE sample_id NOT IN (
                SELECT MIN(sample_id)
                FROM embeddings
                GROUP BY modality, h3, vec_hash
            );
            """
        )
        self.conn.commit()

    def clear(self) -> None:
        self.conn.executescript(
            """
            DELETE FROM embeddings;
            DELETE FROM dims;
            """
        )
        self.conn.commit()

    def insert_embedding(self, modality: str, h3_id: str, vec: np.ndarray) -> bool:
        arr = np.asarray(vec, dtype=np.float32).reshape(-1)
        vec_hash = self._vector_hash(arr)
        cursor = self.conn.execute(
            """
            INSERT OR IGNORE INTO embeddings(modality, h3, dim, vec_hash, vec)
            VALUES (?, ?, ?, ?, ?)
            """,
            (modality, h3_id, int(arr.shape[0]), vec_hash, sqlite3.Binary(arr.tobytes())),
        )
        return cursor.rowcount > 0

    def set_dim(self, modality: str, dim: int) -> None:
        self.conn.execute(
            "INSERT INTO dims(modality, dim) VALUES (?, ?) ON CONFLICT(modality) DO UPDATE SET dim = excluded.dim",
            (modality, int(dim)),
        )

    def get_dim(self, modality: str) -> Optional[int]:
        row = self.conn.execute("SELECT dim FROM dims WHERE modality = ?", (modality,)).fetchone()
        if row is None:
            return None
        return int(row[0])

    def count_embeddings(self, modality: Optional[str] = None) -> int:
        if modality is None:
            row = self.conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()
        else:
            row = self.conn.execute("SELECT COUNT(*) FROM embeddings WHERE modality = ?", (modality,)).fetchone()
        return int(row[0] if row is not None else 0)

    def get_embeddings(self, modality: str, h3_id: str) -> list[np.ndarray]:
        rows = self.conn.execute(
            "SELECT dim, vec FROM embeddings WHERE modality = ? AND h3 = ? ORDER BY sample_id",
            (modality, h3_id),
        ).fetchall()
        embeddings: list[np.ndarray] = []
        for dim, raw in rows:
            vector = np.frombuffer(raw, dtype=np.float32)
            if vector.shape[0] != int(dim):
                continue
            embeddings.append(vector)
        return embeddings

    def get_embedding(self, modality: str, h3_id: str, strategy: str = "random") -> Optional[np.ndarray]:
        vectors = self.get_embeddings(modality, h3_id)
        if not vectors:
            return None

        if strategy == "first":
            return vectors[0]
        if strategy == "mean":
            return np.mean(vectors, axis=0, dtype=np.float32)
        if strategy == "random":
            return random.choice(vectors)
        raise ValueError(f"Unknown embedding selection strategy: {strategy}")

    def _anchor_for_resolutions(self, h3_id: str, target_resolutions: tuple[int, ...]) -> Optional[str]:
        if not _is_valid_h3_cell(h3_id):
            return None

        resolution = h3.get_resolution(h3_id)
        if resolution in target_resolutions:
            return h3_id

        lower_resolutions = [candidate for candidate in target_resolutions if candidate <= resolution]
        if not lower_resolutions:
            return None

        target = max(lower_resolutions)
        if target == resolution:
            return h3_id
        return h3.cell_to_parent(h3_id, target)

    def get_h3_ids(
        self,
        target_res: Optional[int] = None,
        target_resolutions: Optional[Iterable[int]] = None,
    ) -> list[str]:
        targets = _normalize_target_resolutions(target_res=target_res, target_resolutions=target_resolutions)
        rows = self.conn.execute("SELECT DISTINCT h3 FROM embeddings").fetchall()
        output: set[str] = set()
        for (h3_id,) in rows:
            anchor = self._anchor_for_resolutions(h3_id, targets)
            if anchor is not None:
                output.add(anchor)
        return sorted(output)

    def get_h3_ids_for_modality(
        self,
        modality: str,
        target_res: Optional[int] = None,
        target_resolutions: Optional[Iterable[int]] = None,
    ) -> list[str]:
        targets = _normalize_target_resolutions(target_res=target_res, target_resolutions=target_resolutions)
        rows = self.conn.execute("SELECT h3 FROM embeddings WHERE modality = ?", (modality,)).fetchall()
        output: set[str] = set()
        for (h3_id,) in rows:
            anchor = self._anchor_for_resolutions(h3_id, targets)
            if anchor is not None:
                output.add(anchor)
        return sorted(output)

    def get_cells_for_modality(self, modality: str) -> set[str]:
        rows = self.conn.execute("SELECT DISTINCT h3 FROM embeddings WHERE modality = ?", (modality,)).fetchall()
        return {str(h3_id) for (h3_id,) in rows}

    def has_embedding(self, modality: str, h3_id: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM embeddings WHERE modality = ? AND h3 = ? LIMIT 1",
            (modality, h3_id),
        ).fetchone()
        return row is not None

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
                if h3_id is None:
                    continue
                for vector in _iter_numpy_vectors(raw_vec):
                    yield h3_id, vector

    def _iter_pickle(self, folder: Path) -> Iterator[Tuple[str, np.ndarray]]:
        for pickle_path in sorted(folder.glob("*.pkl")):
            try:
                with pickle_path.open("rb") as handle:
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
                if h3_id is None:
                    continue
                for vector in _iter_numpy_vectors(raw_vec):
                    yield h3_id, vector

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
            expected_dim: Optional[int] = None
            inserted = 0
            skipped_invalid_h3 = 0
            skipped_dim_mismatch = 0
            skipped_duplicate = 0

            for h3_id, vector in iterator:
                if not _is_valid_h3_cell(h3_id):
                    skipped_invalid_h3 += 1
                    continue

                vector = np.asarray(vector, dtype=np.float32).reshape(-1)
                if expected_dim is None:
                    expected_dim = int(vector.shape[0])
                elif int(vector.shape[0]) != expected_dim:
                    skipped_dim_mismatch += 1
                    continue

                if not self.insert_embedding(modality, h3_id, vector):
                    skipped_duplicate += 1
                    continue
                inserted += 1

            if expected_dim is not None:
                self.set_dim(modality, expected_dim)
                dims[modality] = expected_dim

            if skipped_invalid_h3 or skipped_dim_mismatch or skipped_duplicate:
                warnings.warn(
                    (
                        f"{modality} ingestion summary: inserted={inserted}, "
                        f"invalid_h3={skipped_invalid_h3}, dim_mismatch={skipped_dim_mismatch}, "
                        f"duplicates={skipped_duplicate}"
                    ),
                    stacklevel=2,
                )

        self.commit()
        if not dims:
            raise ValueError(f"No embeddings found under {root}")
        return dims
