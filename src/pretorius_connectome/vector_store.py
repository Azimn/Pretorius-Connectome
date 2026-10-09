"""Persistent, exact sparse vector retrieval over canonical Pretorius memories.

Stores source-linked training-split TF-IDF vectors as an indexed SQLite
inverted index. Never describes lexical features as semantic embeddings.
This is a database/retrieval layer, NOT a neural FlyWire connectome or engram.
"""
from __future__ import annotations

from hashlib import sha256
import json
import math
import os
from pathlib import Path
import sqlite3
import tempfile

import numpy as np

from pretorius_connectome.shared_memory_l2 import (
    SharedCache, ENCODER, SCHEMA, SOURCE_BLOB,
    canon_json,
)

STORE_SCHEMA = "vector-fly-sqlite-exact-tfidf/1"
SCHEMA_SQL = """
CREATE TABLE metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
) WITHOUT ROWID;
CREATE TABLE memories (
    doc_id INTEGER PRIMARY KEY,
    event_id TEXT UNIQUE NOT NULL,
    episode_id TEXT NOT NULL,
    provenance TEXT NOT NULL,
    record_json TEXT NOT NULL
);
CREATE INDEX memory_episode ON memories(episode_id, event_id);
CREATE INDEX memory_provenance ON memories(provenance, event_id);
CREATE TABLE postings (
    feature_id INTEGER NOT NULL,
    doc_id INTEGER NOT NULL,
    weight REAL NOT NULL,
    PRIMARY KEY(feature_id, doc_id),
    FOREIGN KEY(doc_id) REFERENCES memories(doc_id)
) WITHOUT ROWID;
CREATE INDEX postings_by_doc ON postings(doc_id, feature_id);
"""


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _selected_rows(cache: SharedCache, scope: str) -> list[int]:
    if scope == "all":
        return list(range(len(cache.ids)))
    if scope == "train":
        train = set(cache.manifest["fit_event_ids"])
        return [i for i, event_id in enumerate(cache.ids) if event_id in train]
    raise ValueError("scope must be 'train' or 'all'")


def build_store(db_path: str | Path, cache_dir: str | Path,
                *, scope: str = "train") -> dict:
    """Atomically create an immutable SQLite index from an already verified L2.

    scope='train' is appropriate for episode holdout experiments; 'all' is
    convenient for direct 450-memory lookup but is NOT a train/test benchmark.
    Existing database paths are never silently replaced.
    """
    path = Path(db_path)
    if path.exists():
        raise FileExistsError("Existing vector index is immutable; use a new output path")
    path.parent.mkdir(parents=True, exist_ok=True)
    cache = SharedCache(cache_dir)
    cache.assert_original()
    positions = _selected_rows(cache, scope)
    if not positions:
        raise ValueError("No documents selected for the vector index")
    parent = Path(cache_dir) / "manifest.json"
    meta = {
        "schema": STORE_SCHEMA,
        "vector_kind": "lexical-tfidf-not-semantic",
        "vector_metric": "exact-dot-product-of-unit-normalized-nonnegative-vectors",
        "source_git_blob": SOURCE_BLOB,
        "l2_schema": SCHEMA,
        "l2_encoder": ENCODER,
        "l2_encoder_code_hash": cache.manifest["encoder_code_hash"],
        "l2_manifest_sha256": _digest(parent),
        "l2_documents_sha256": cache.manifest["shard_sha256"]["docs.npz"],
        "random_seed": cache.manifest["random_seed"],
        "vector_dim": cache.manifest["vector_dim"],
        "scope": scope,
        "indexed_event_ids": [cache.ids[i] for i in positions],
        "fit_event_ids": cache.manifest["fit_event_ids"],
        "all_source_records": len(cache.ids),
        "indexed_documents": len(positions),
        "not_a_truth_checker": True,
    }

    # The SQLite file is built in the destination filesystem and published
    # by atomic rename after a successful commit and integrity verification.
    tmp_handle = tempfile.NamedTemporaryFile(
        prefix=".vector-fly-", suffix=".sqlite", dir=path.parent, delete=False
    )
    tmp = Path(tmp_handle.name)
    tmp_handle.close()
    connection = None
    try:
        connection = sqlite3.connect(str(tmp))
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=FULL")
        connection.executescript(SCHEMA_SQL)
        connection.executemany(
            "INSERT INTO metadata(key, value) VALUES (?,?)",
            [(k, canon_json(v)) for k, v in sorted(meta.items())],
        )
        records = [
            (i, cache.ids[i], cache.records[i]["episode_id"],
             cache.records[i]["provenance"], canon_json(cache.records[i]))
            for i in positions
        ]
        connection.executemany(
            "INSERT INTO memories(doc_id,event_id,episode_id,provenance,record_json) "
            "VALUES (?,?,?,?,?)", records
        )

        def posting_rows():
            for doc_id in positions:
                vector = cache.docs.getrow(doc_id)
                for col, val in zip(vector.indices, vector.data):
                    if val == 0:
                        continue
                    yield (int(col), int(doc_id), float(val))

        connection.executemany(
            "INSERT INTO postings(feature_id,doc_id,weight) VALUES (?,?,?)",
            posting_rows(),
        )
        connection.commit()
        check = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if check != "ok":
            raise ValueError("SQLite index integrity check failed: " + str(check))
        n_post = connection.execute("SELECT COUNT(*) FROM postings").fetchone()[0]
        connection.close()
        connection = None
        # Reopen and compare each stored vector and original source record
        # before publishing. A generated .sqlite is not authoritative.
        with VectorFlyStore(tmp, cache_dir, check_vectors=True) as test:
            assert test.info()["indexed_documents"] == len(positions)
        # Same-filesystem hard-link creation is an atomic no-replace
        # publish operation: competing builders cannot overwrite one another.
        # os.replace() would silently replace a rival's newly published file
        # after both passed the initial path.exists() precheck.
        os.link(tmp, path)
        return {
            **{k: meta[k] for k in (
                "schema", "vector_kind", "l2_encoder", "random_seed",
                "scope", "indexed_documents", "vector_dim", "source_git_blob",
                "l2_manifest_sha256"
            )},
            "nonzero_postings": n_post,
            "database_bytes": path.stat().st_size,
            "database_sha256": _digest(path),
            "database": str(path),
        }
    finally:
        if connection is not None:
            connection.close()
        tmp.unlink(missing_ok=True)


class VectorFlyStore:
    """Read-only indexed nearest-neighbor search, without a paid API.

    The provider's trained vocabulary and query transform are reused as-is.
    Scores derive from indexed sparse postings, not a full matrix scan.
    """

    def __init__(self, db_path: str | Path, cache_dir: str | Path,
                 *, check_vectors: bool = True):
        self.cache = SharedCache(cache_dir)
        self.cache.assert_original()
        p = Path(db_path).resolve()
        if not p.is_file():
            raise FileNotFoundError(str(p))
        # The local read-only HTTPServer may be served on a separate thread
        # from its startup/verification thread (including integration tests).
        # Allow that handoff, while the API intentionally handles requests
        # serially; no shared-connection concurrent writes are permitted.
        self.conn = sqlite3.connect(
            p.as_uri() + "?mode=ro", uri=True, check_same_thread=False
        )
        self.conn.execute("PRAGMA query_only=ON")
        try:
            self.meta = {
                key: json.loads(value) for key, value in
                self.conn.execute("SELECT key,value FROM metadata")
            }
            expected = {
                "schema": STORE_SCHEMA,
                "vector_kind": "lexical-tfidf-not-semantic",
                "l2_schema": SCHEMA,
                "l2_encoder": ENCODER,
                "l2_encoder_code_hash": self.cache.manifest["encoder_code_hash"],
                "l2_manifest_sha256": _digest(Path(cache_dir) / "manifest.json"),
                "l2_documents_sha256":
                    self.cache.manifest["shard_sha256"]["docs.npz"],
                "source_git_blob": SOURCE_BLOB,
                "random_seed": self.cache.manifest["random_seed"],
                "vector_dim": self.cache.manifest["vector_dim"],
                "fit_event_ids": self.cache.manifest["fit_event_ids"],
                "all_source_records": len(self.cache.ids),
            }
            for key, val in expected.items():
                if self.meta.get(key) != val:
                    raise ValueError("Database/cache provenance mismatch: " + key)
            self.positions = _selected_rows(self.cache, self.meta.get("scope"))
            ids = [self.cache.ids[i] for i in self.positions]
            if (ids != self.meta.get("indexed_event_ids")
                    or len(ids) != self.meta.get("indexed_documents")
                    or not self.meta.get("not_a_truth_checker")):
                raise ValueError("Database membership/limitations mismatch")
            self._verify_records()
            if check_vectors:
                self.verify_vectors()
        except Exception:
            self.close()
            raise

    def close(self):
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def _verify_records(self):
        actual = self.conn.execute(
            "SELECT doc_id,event_id,episode_id,provenance,record_json "
            "FROM memories ORDER BY doc_id"
        ).fetchall()
        if len(actual) != len(self.positions):
            raise ValueError("Database row count mismatch")
        for (doc_id, event_id, episode, provenance, record), i in zip(
            actual, self.positions
        ):
            source = self.cache.records[i]
            if (doc_id != i or event_id != self.cache.ids[i]
                    or episode != source["episode_id"]
                    or provenance != source["provenance"]
                    or json.loads(record) != source):
                raise ValueError("Indexed memory differs from canonical L1")

    def verify_vectors(self) -> dict:
        """Compare complete database vector postings with pinned L2 rows."""
        result = self.conn.execute("PRAGMA integrity_check").fetchone()[0]
        if result != "ok":
            raise ValueError("Invalid SQLite index: " + str(result))
        n = 0
        for i in self.positions:
            expected = self.cache.docs.getrow(i)
            actual = self.conn.execute(
                "SELECT feature_id,weight FROM postings WHERE doc_id=? ORDER BY feature_id",
                (i,),
            ).fetchall()
            if (len(actual) != expected.nnz
                    or any(j != int(col) or not math.isfinite(weight)
                           or weight != float(val)
                           for (j, weight), col, val in
                           zip(actual, expected.indices, expected.data))):
                raise ValueError("Vector postings differ from pinned L2")
            n += len(actual)
        total = self.conn.execute("SELECT COUNT(*) FROM postings").fetchone()[0]
        if total != n:
            raise ValueError("Unknown vector postings detected")
        return {"verified_documents": len(self.positions),
                "verified_postings": total, "source_git_blob": SOURCE_BLOB}

    def info(self) -> dict:
        return {key: self.meta[key] for key in (
            "schema", "vector_kind", "l2_encoder", "scope", "random_seed",
            "indexed_documents", "vector_dim", "source_git_blob"
        )}

    def get(self, event_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT record_json FROM memories WHERE event_id=?", (event_id,)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def search(self, query: str, *, top_k: int = 5,
               episode_id: str | None = None,
               provenance: str | None = None) -> list[dict]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be nonempty")
        if top_k < 1:
            raise ValueError("top_k must be >=1")
        vector = self.cache.query(query)
        vector.sort_indices()
        if vector.nnz == 0:
            return []  # no invented zero-similarity memories
        where, params = [], []
        if episode_id is not None:
            where.append("episode_id=?")
            params.append(episode_id)
        if provenance is not None:
            where.append("provenance=?")
            params.append(provenance)
        predicate = " WHERE " + " AND ".join(where) if where else ""
        permitted = {
            row[0] for row in self.conn.execute(
                "SELECT doc_id FROM memories" + predicate, params
            )
        }
        if not permitted:
            return []
        scores: dict[int, float] = {}
        for col, val in zip(vector.indices, vector.data):
            for i, stored in self.conn.execute(
                "SELECT doc_id,weight FROM postings WHERE feature_id=?",
                (int(col),),
            ):
                if i in permitted:
                    scores[i] = scores.get(i, 0.0) + float(val) * stored
        winners = sorted(
            ((i, s) for i, s in scores.items() if s > 0),
            key=lambda item: (-item[1], self.cache.ids[item[0]]),
        )[:top_k]
        output = []
        for doc_id, score in winners:
            original = self.cache.records[doc_id]
            output.append({
                "event_id": original["event_id"],
                "episode_id": original["episode_id"],
                "title": original.get("title"),
                "approximate_date": original.get("approximate_date"),
                "provenance": original["provenance"],
                "similarity": round(float(score), 12),
                "source_excerpt": original["memory_text"][:360],
                "evidence_status": "source text similarity, not claim entailment",
            })
        return output
