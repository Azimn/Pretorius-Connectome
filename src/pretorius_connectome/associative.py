"""Connectome-constrained autobiographical retrieval.

Purely computational association study. Anatomical synapse counts are not
learned weights, activation, neurotransmitter signs, or decoded experiences.
The original narrative texts remain the source of truth.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b
from pathlib import Path
import re

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


@dataclass(frozen=True)
class Topology:
    root_ids: np.ndarray
    indptr: np.ndarray
    indices: np.ndarray
    synapse_counts: np.ndarray
    provenance: str

    def __post_init__(self):
        n = len(self.root_ids)
        ptr = self.indptr
        idx = self.indices
        c = self.synapse_counts
        if (n < 2 or len(ptr) != n + 1 or not len(idx) or len(c) != len(idx)
                or int(ptr[0]) != 0 or int(ptr[-1]) != len(idx)
                or np.any(np.diff(ptr) < 0)
                or np.any(idx < 0) or np.any(idx >= n)
                or np.any(c <= 0)
                or len(np.unique(self.root_ids)) != n):
            raise ValueError("invalid directed connectivity arrays")

    @classmethod
    def read(cls, path: str | Path) -> "Topology":
        with np.load(path, allow_pickle=False) as data:
            required = {"root_ids", "indptr", "indices", "synapse_counts"}
            if not required.issubset(data.files):
                raise ValueError("CSR NPZ missing required biological arrays")
            arrays = {key: data[key].copy() for key in required}
        return cls(**arrays, provenance="FlyWire-derived user-supplied CSR: " + str(path))

    @classmethod
    def synthetic(cls, n: int = 512, degree: int = 8, seed: int = 7) -> "Topology":
        """Explicit nonbiological test fixture; not a substitute for FlyWire."""
        if n < 16 or degree < 1 or degree >= n:
            raise ValueError("invalid synthetic graph size")
        rng = np.random.default_rng(seed)
        sources = np.repeat(np.arange(n), degree)
        targets = rng.integers(0, n, size=len(sources), dtype=np.int32)
        order = np.lexsort((targets, sources))
        targets = targets[order]
        counts = rng.integers(1, 8, size=len(sources), dtype=np.int64)[order]
        return cls(np.arange(n, dtype=np.uint64),
                   np.arange(n + 1, dtype=np.int64) * degree,
                   targets, counts, "synthetic random directed test graph")

    def permuted_null(self, seed: int = 7) -> "Topology":
        """Permute destination stubs, retaining raw in/out stub counts.

        Allows multiedges and self-loops; not a simple-graph edge-swap null.
        Source-row synapse-count multisets are fixed; target weighted degrees
        are not fixed. Report these limitations in any comparison.
        """
        rng = np.random.default_rng(seed)
        return Topology(
            self.root_ids.copy(), self.indptr.copy(),
            self.indices[rng.permutation(len(self.indices))].copy(),
            self.synapse_counts.copy(),
            "permuted-target-stub null of " + self.provenance,
        )

    def transition(self) -> sparse.csr_matrix:
        """Directed source-to-target row stochastic operator.

        log1p anatomical contact count is a modeling assumption, not
        a physical synaptic efficacy. Signs and neurotransmitters are unknown.
        """
        n = len(self.root_ids)
        weights = np.log1p(self.synapse_counts.astype(np.float64))
        operator = sparse.csr_matrix(
            (weights, self.indices, self.indptr), shape=(n, n),
        )
        operator.sum_duplicates()
        row_sum = np.asarray(operator.sum(axis=1)).ravel()
        scale = 1.0 / np.maximum(row_sum, 1e-12)
        return (sparse.diags(scale) @ operator).tocsr()


def _prune_rows(matrix: sparse.spmatrix, cap: int) -> sparse.csr_matrix:
    """Deterministic bounded-memory top-k activity, row by row."""
    matrix = matrix.tocsr()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    ptr, cols, vals = [0], [], []
    for row in range(matrix.shape[0]):
        start, end = matrix.indptr[row:row + 2]
        ix = matrix.indices[start:end]
        vv = matrix.data[start:end]
        if len(vv) > cap:
            selected = np.argpartition(-np.abs(vv), cap - 1)[:cap]
            selected = selected[np.lexsort((ix[selected], -np.abs(vv[selected])))]
            ix, vv = ix[selected], vv[selected]
        order = np.argsort(ix)
        cols.extend(ix[order])
        vals.extend(vv[order])
        ptr.append(len(cols))
    return sparse.csr_matrix(
        (np.asarray(vals, dtype=np.float64),
         np.asarray(cols, dtype=np.int32),
         np.asarray(ptr, dtype=np.int64)),
        shape=matrix.shape,
    )


class AssociativeMemory:
    """Evidence-preserving sparse vector retrieval and optional graph diffusion.

    Every method is fit only on supplied indexed memories. No query labels,
    challenge anchors, or target event IDs enter vectorization or mapping.
    """

    def __init__(self, memories, topology: Topology | None = None, *,
                 steps: int = 2, activity_cap: int = 256,
                 diffusion: float = 0.4, hybrid_fraction: float = 0.25):
        if not memories or len({m.event_id for m in memories}) != len(memories):
            raise ValueError("nonempty memories with unique IDs required")
        if steps < 0 or activity_cap < 1 or not (0 <= diffusion <= 1):
            raise ValueError("invalid spreading parameters")
        if not (0 <= hybrid_fraction <= 1):
            raise ValueError("invalid hybrid mixing factor")
        self.memories = list(memories)
        self.topology = topology
        self.steps = steps
        self.activity_cap = activity_cap
        self.diffusion = diffusion
        self.hybrid_fraction = hybrid_fraction
        self.encoder = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2),
            max_features=8192, sublinear_tf=True, norm="l2",
            dtype=np.float64,
        )
        self.docs = self.encoder.fit_transform(
            [m.memory_text for m in self.memories]
        )
        self.ids = [m.event_id for m in self.memories]
        self.graph_docs = None
        self.projection = None
        self.propagator = None
        if topology is not None:
            n = len(topology.root_ids)
            features = self.encoder.get_feature_names_out()
            anchors = np.asarray([
                int.from_bytes(blake2b(f.encode("utf-8"), digest_size=8,
                                      person=b"pt-graph-v1").digest(), "little") % n
                for f in features
            ], dtype=np.int64)
            self.projection = sparse.csr_matrix(
                (np.ones(len(features), dtype=np.float64),
                 (np.arange(len(features)), anchors)),
                shape=(len(features), n),
            )
            self.propagator = topology.transition()
            self.graph_docs = self._graph(self.docs)

    def _graph(self, documents: sparse.spmatrix) -> sparse.csr_matrix:
        if self.projection is None or self.propagator is None:
            raise ValueError("no graph configured")
        anchored = normalize(_prune_rows(
            documents @ self.projection, self.activity_cap
        ))
        state = anchored
        for _ in range(self.steps):
            state = normalize(_prune_rows(
                (1.0 - self.diffusion) * anchored +
                self.diffusion * (state @ self.propagator),
                self.activity_cap,
            ))
        return state.tocsr()

    def score(self, query: str, mode: str = "hybrid") -> np.ndarray:
        if mode not in {"lexical", "graph", "hybrid"}:
            raise ValueError("unknown retrieval mode")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a nonempty string")
        query_vec = self.encoder.transform([query])
        lexical = (self.docs @ query_vec.T).toarray().ravel()
        if mode == "lexical":
            return lexical
        if self.graph_docs is None:
            raise ValueError("graph or hybrid scoring requires a topology")
        graph_query = self._graph(query_vec)
        graph = (self.graph_docs @ graph_query.T).toarray().ravel()
        if mode == "graph":
            return graph
        return (1.0 - self.hybrid_fraction) * lexical + self.hybrid_fraction * graph

    def rank(self, query: str, mode: str = "hybrid", top_k: int = 5) -> list[dict]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        scores = self.score(query, mode)
        winners = sorted(range(len(scores)), key=lambda i: (-scores[i], self.ids[i]))
        output = []
        terms = set(re.findall(r"\w+", query.casefold()))
        for i in winners[:top_k]:
            narrative = self.memories[i].memory_text
            sentences = re.split(r"(?<=[.!?])\s+", narrative)
            evidence = max(sentences, key=lambda s: len(
                terms.intersection(re.findall(r"\w+", s.casefold()))
            )) if sentences else narrative
            output.append({
                "event_id": self.ids[i],
                "score": round(float(scores[i]), 6),
                "retrieved_excerpt": evidence[:300],
                "evidence_status": "retrieved text only; factual entailment not checked",
            })
        return output


def summarize_challenge(engine: AssociativeMemory, cases, train_ids: set[str],
                        absent_ids: set[str], threshold: float, mode: str):
    """Three-way post-hoc test on the existing, nonblind 68-case challenge."""
    groups = {"positive": [], "contradiction": [], "absent": []}
    for case in cases:
        if case.event_id in train_ids:
            group = "positive" if case.kind == "paraphrase" else "contradiction"
        elif case.event_id in absent_ids and case.kind == "paraphrase":
            group = "absent"
        else:
            continue
        scores = engine.score(case.query, mode)
        top = int(np.argmax(scores))
        guess = engine.ids[top]
        accepted = bool(scores[top] > threshold)
        groups[group].append({
            "case_id": case.case_id, "target": case.event_id,
            "predicted": guess, "accepted": accepted,
            "score": round(float(scores[top]), 6),
        })
    if any(not records for records in groups.values()):
        raise ValueError("no valid cases in one or more evaluation groups")
    p, c, a = (groups[key] for key in ("positive", "contradiction", "absent"))
    rate = lambda items: round(sum(bool(x) for x in items) / len(items), 6)
    return {
        "mode": mode, "threshold": round(float(threshold), 6),
        "positive_n": len(p),
        "positive_top1": rate(r["predicted"] == r["target"] for r in p),
        "positive_correct_and_accepted": rate(
            r["accepted"] and r["predicted"] == r["target"] for r in p
        ),
        "contradiction_n": len(c),
        "contradiction_false_acceptance": rate(r["accepted"] for r in c),
        "absent_n": len(a),
        "absent_false_acceptance": rate(r["accepted"] for r in a),
        "case_results": groups,
    }
