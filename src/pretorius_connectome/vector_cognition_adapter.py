"""One-way adapter: existing pinned VectorFlyStore -> cognitive evidence.

All 450 Pretorius fictional prehistory records keep their original source
identity. This module never imports them as lived experiences or writes to DB.
"""
from __future__ import annotations

from pretorius_connectome.cognition_port import (
    MemoryCandidate, MemorySourceMismatch, SourceDescriptor,
)
from pretorius_connectome.vector_store import VectorFlyStore
from pretorius_connectome.shared_memory import SOURCE_BLOB


class VectorFlyCognitionSource:
    def __init__(self, store: VectorFlyStore):
        if not isinstance(store, VectorFlyStore):
            raise TypeError("A verified VectorFlyStore instance is required")
        metadata = store.info()
        if (metadata.get("source_git_blob") != SOURCE_BLOB
                or metadata.get("vector_kind") != "lexical-tfidf-not-semantic"
                or metadata.get("scope") not in ("all", "train")):
            raise MemorySourceMismatch("Vector Fly source/encoder/scope mismatch")
        self._store = store
        self.descriptor = SourceDescriptor(
            subject_id="pretorius", namespace="pretorius.reconstructed.v12",
            source_version="v12-450", source_hash=SOURCE_BLOB,
            encoder_id=metadata["l2_encoder"], index_scope=metadata["scope"],
        )

    def _candidate(self, record: dict, score: float | None, excerpt: str) -> MemoryCandidate:
        if record.get("provenance") != "reconstructed":
            raise MemorySourceMismatch("Unexpected Pretorius provenance class")
        if not isinstance(excerpt, str) or not record["memory_text"].startswith(excerpt):
            raise MemorySourceMismatch("Unverified excerpt against original memory")
        return MemoryCandidate(
            subject_id="pretorius", namespace=self.descriptor.namespace,
            record_id=record["event_id"], source_hash=SOURCE_BLOB,
            origin_class="reconstructed_prehistory",
            source_excerpt=excerpt, score=score,
        )

    def search(self, query: str, *, top_k: int) -> list[MemoryCandidate]:
        rows = self._store.search(query, top_k=top_k)
        output = []
        for row in rows:
            record = self._store.get(row["event_id"])
            if (record is None or record["event_id"] != row["event_id"]
                    or row["provenance"] != record["provenance"]
                    or row["episode_id"] != record["episode_id"]):
                raise MemorySourceMismatch("Search result cannot be resolved to source")
            output.append(self._candidate(record, row["similarity"], row["source_excerpt"]))
        return output

    def get(self, record_id: str) -> MemoryCandidate | None:
        record = self._store.get(record_id)
        return None if record is None else self._candidate(
            record, None, record["memory_text"]
        )
