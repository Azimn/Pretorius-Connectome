"""Memory-to-cognition port v0.1: verified retrieval, never identity transfer.

Standard-library-only protocol for subject-scoped read-only evidence. Source
owners retain their canonical ledgers. This module has no event-write path.
An issuing application MUST be trusted: in-process capabilities are not user
authentication and must not be exposed to untrusted renderer/model prompts.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import re
import secrets
from typing import Protocol

SCHEMA = "memory-cognition-port/0.1"
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_HASH = re.compile(r"^(?:[a-f0-9]{40}|[a-f0-9]{64})$")
ORIGINS = frozenset({
    "reconstructed_prehistory", "canonical_prehistory", "observed_runtime",
    "synthesized_prehistory", "authored_fiction", "synthetic_fixture",
    "external_document", "uncertain",
})


class MemoryPortError(ValueError):
    """Invalid, unauthorized or source-inconsistent request."""


class MemoryAccessDenied(MemoryPortError):
    """No usable capability for the selected source and subject."""


class MemorySourceMismatch(MemoryPortError):
    """A source returned an item inconsistent with its trusted binding."""


@dataclass(frozen=True)
class SourceDescriptor:
    subject_id: str
    namespace: str
    source_version: str
    source_hash: str
    encoder_id: str
    index_scope: str

    def __post_init__(self):
        for field in ("subject_id", "namespace", "source_version", "encoder_id", "index_scope"):
            if not isinstance(getattr(self, field), str) or not _ID.fullmatch(getattr(self, field)):
                raise MemoryPortError("Invalid source descriptor: " + field)
        if not isinstance(self.source_hash, str) or not _HASH.fullmatch(self.source_hash):
            raise MemoryPortError("Source requires pinned SHA-1 Git blob or SHA-256 digest")


@dataclass(frozen=True)
class MemoryCandidate:
    subject_id: str
    namespace: str
    record_id: str
    source_hash: str
    origin_class: str
    source_excerpt: str
    score: float | None = None


@dataclass(frozen=True)
class CognitionEvidence:
    """Transport-neutral external evidence, NOT admitted belief or lived memory."""
    schema: str
    subject_id: str
    namespace: str
    source_version: str
    source_hash: str
    encoder_id: str
    index_scope: str
    record_id: str
    origin_class: str
    text: str
    relevance_score: float | None
    channel: str = "external_memory_evidence"
    epistemic_status: str = "candidate_not_entailment"
    eligible_for_lived_write: bool = False


class ReadOnlyMemorySource(Protocol):
    @property
    def descriptor(self) -> SourceDescriptor: ...
    def search(self, query: str, *, top_k: int) -> list[MemoryCandidate]: ...
    def get(self, record_id: str) -> MemoryCandidate | None: ...


@dataclass(frozen=True)
class _Grant:
    subject_id: str
    namespace: str
    descriptor: SourceDescriptor


class MemoryCognitionPort:
    """Subject-bound, fail-closed read-only capability router.

    The trusted host registers each source and mints capabilities *outside*
    the renderer. Each capability works for one subject and one fixed source.
    No cross-subject capability issuance is supported in v0.1. The source
    adapter must itself validate source data against an independently pinned
    digest and enforce any per-record visibility restrictions.
    """

    def __init__(self):
        self._sources: dict[str, ReadOnlyMemorySource] = {}
        self._grants: dict[str, _Grant] = {}

    def register(self, source: ReadOnlyMemorySource) -> None:
        descriptor = source.descriptor
        if not isinstance(descriptor, SourceDescriptor):
            raise MemoryPortError("Source must provide a version-pinned descriptor")
        if descriptor.namespace in self._sources:
            raise MemoryPortError("Namespace already registered")
        self._sources[descriptor.namespace] = source

    def grant_self(self, *, subject_id: str, namespace: str) -> str:
        """Trusted-host operation. Never show the token to a renderer."""
        source = self._sources.get(namespace)
        if source is None or subject_id != source.descriptor.subject_id:
            raise MemoryAccessDenied("Only the source subject can receive this capability")
        token = secrets.token_urlsafe(32)
        self._grants[sha256(token.encode("utf-8")).hexdigest()] = _Grant(
            subject_id, namespace, source.descriptor
        )
        return token

    def revoke(self, token: str) -> None:
        if isinstance(token, str):
            self._grants.pop(sha256(token.encode("utf-8")).hexdigest(), None)

    def _source(self, token: str, subject_id: str, namespace: str) -> ReadOnlyMemorySource:
        if not isinstance(token, str) or not token:
            raise MemoryAccessDenied("Missing memory capability")
        grant = self._grants.get(sha256(token.encode("utf-8")).hexdigest())
        source = self._sources.get(namespace)
        if (grant is None or source is None or grant.subject_id != subject_id
                or grant.namespace != namespace or source.descriptor != grant.descriptor):
            raise MemoryAccessDenied("Memory capability invalid, stale or revoked")
        return source

    @staticmethod
    def _evidence(item: MemoryCandidate, descriptor: SourceDescriptor) -> CognitionEvidence:
        if not isinstance(item, MemoryCandidate):
            raise MemorySourceMismatch("Source must return typed memory candidates")
        if (item.subject_id != descriptor.subject_id
                or item.namespace != descriptor.namespace
                or item.source_hash != descriptor.source_hash
                or not isinstance(item.record_id, str)
                or not _ID.fullmatch(item.record_id)
                or item.origin_class not in ORIGINS
                or not isinstance(item.source_excerpt, str)
                or not item.source_excerpt.strip()):
            raise MemorySourceMismatch("Wrong owner, provenance, hash or record content")
        if item.score is not None:
            if (type(item.score) not in (int, float) or item.score < 0
                    or not float(item.score) < float("inf")):
                raise MemorySourceMismatch("Invalid or non-finite retrieval score")
        return CognitionEvidence(
            schema=SCHEMA, subject_id=descriptor.subject_id,
            namespace=descriptor.namespace, source_version=descriptor.source_version,
            source_hash=descriptor.source_hash, encoder_id=descriptor.encoder_id,
            index_scope=descriptor.index_scope, record_id=item.record_id,
            origin_class=item.origin_class, text=item.source_excerpt,
            relevance_score=float(item.score) if item.score is not None else None,
        )

    def search(self, *, token: str, subject_id: str, namespace: str,
               query: str, top_k: int = 5) -> tuple[CognitionEvidence, ...]:
        source = self._source(token, subject_id, namespace)  # authorize BEFORE search
        if (not isinstance(query, str) or not query.strip() or len(query) > 2000
                or type(top_k) is not int or not 1 <= top_k <= 20):
            raise MemoryPortError("Invalid query or top_k")
        rows = source.search(query, top_k=top_k)
        if not isinstance(rows, (list, tuple)) or len(rows) > top_k:
            raise MemorySourceMismatch("Source returned unbounded candidates")
        results = tuple(self._evidence(row, source.descriptor) for row in rows)
        if len({row.record_id for row in results}) != len(results):
            raise MemorySourceMismatch("Duplicate record identifiers")
        return results

    def get(self, *, token: str, subject_id: str, namespace: str,
            record_id: str) -> CognitionEvidence | None:
        source = self._source(token, subject_id, namespace)  # authorize BEFORE get
        if not isinstance(record_id, str) or not _ID.fullmatch(record_id):
            raise MemoryPortError("Invalid record identifier")
        row = source.get(record_id)
        if row is None:
            return None
        item = self._evidence(row, source.descriptor)
        if item.record_id != record_id:
            raise MemorySourceMismatch("Source returned a different record")
        return item
