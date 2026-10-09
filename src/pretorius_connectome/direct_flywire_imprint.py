"""Direct content-conditioned synaptic overlay on existing FlyWire directed edges.

This is a computational heteroassociative readout, not fly physiology, STDP,
semantic language recall, or an autobiographical person. The ORIGINAL graph is
immutable. Model inference has no narrative texts, event-ID table, memory
database, nearest-neighbor retriever, or evaluator target codebook.

Training maps (1) stateless BC01 lexical cue features and (2) independently
encoded content features to fixed, disjoint neuron populations. Only EXISTING
directed synapses crossing the two populations may receive signed numerical
weight deltas. The output of a cue is a distributed 256D lexical-feature signal.
Any event identification needs a separate, explicitly oracle-assisted scorer.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import numpy as np

from pretorius_connectome.associative import Topology
from pretorius_connectome.shared_features_bc01 import DIM, encode_bc_sensory


def fingerprint(topology: Topology) -> dict[str, str]:
    """Hash each anatomical array, including raw original integer contacts."""
    result = {}
    for field in ("root_ids", "indptr", "indices", "synapse_counts"):
        value = np.ascontiguousarray(getattr(topology, field))
        result[field] = sha256(memoryview(value).cast("B")).hexdigest()
    return result


def select_features(vector: np.ndarray, *, top_k: int) -> np.ndarray:
    """Deterministic signed sparse features with fixed lexicographic tie break."""
    v = np.asarray(vector, dtype=np.float32)
    if v.shape != (DIM,) or not np.isfinite(v).all() or top_k < 1 or top_k > DIM:
        raise ValueError("Expected a finite signed BC01 256-dimensional vector")
    chosen = np.lexsort((np.arange(DIM), -np.abs(v)))[:top_k]
    x = np.zeros(DIM, dtype=np.float32)
    x[chosen] = v[chosen]
    norm = float(np.linalg.norm(x))
    if norm > 0:
        x /= norm
    return x


class DirectFlywireOverlay:
    """Only numerical weights, fixed feature-to-neuron maps and original graph.

    Critically there is no API accepting event IDs, source texts, trained
    narrative matrix or per-event codebook at inference. Calling infer(text)
    performs a fixed lexical hash, a directed sparse synaptic readout, and a
    fixed neuron-to-feature readout. The model cannot return a narrative.
    """

    def __init__(self, topology: Topology, *, seed: int = 31,
                 cells_per_feature: int = 32, cue_features: int = 8,
                 content_features: int = 32, rate: float = 0.7,
                 weight_cap: float = 4.0):
        if (cells_per_feature < 1 or len(topology.root_ids) < 2 * DIM * cells_per_feature
                or cue_features < 1 or cue_features > DIM
                or content_features < 1 or content_features > DIM
                or not (0 < rate <= 5) or not np.isfinite(rate)
                or weight_cap <= 0 or not np.isfinite(weight_cap)):
            raise ValueError("Invalid topology, feature population or learning rule")
        self.topology = topology
        self.seed = int(seed)
        self.cells_per_feature = int(cells_per_feature)
        self.cue_features = int(cue_features)
        self.content_features = int(content_features)
        self.rate = float(rate)
        self.weight_cap = float(weight_cap)
        self.original_sha256 = fingerprint(topology)
        rng = np.random.default_rng(seed)
        selected = rng.permutation(len(topology.root_ids))[:2 * DIM * cells_per_feature]
        self.pre_cells = selected[:DIM * cells_per_feature].reshape(
            DIM, cells_per_feature
        ).astype(np.int32)
        self.post_cells = selected[DIM * cells_per_feature:].reshape(
            DIM, cells_per_feature
        ).astype(np.int32)
        self.post_feature = np.full(len(topology.root_ids), -1, dtype=np.int16)
        self.post_feature[self.post_cells.ravel()] = np.repeat(
            np.arange(DIM, dtype=np.int16), cells_per_feature
        )
        self.delta = np.zeros(len(topology.indices), dtype=np.float32)
        self.imprints = 0
        self.edge_update_events = 0

    def _input(self, cue: str) -> np.ndarray:
        if not isinstance(cue, str) or not cue.strip():
            raise ValueError("Inference accepts only a nonempty literal cue")
        return select_features(
            encode_bc_sensory(cue), top_k=self.cue_features
        )

    def imprint(self, cue: str, encoded_content: np.ndarray) -> int:
        """One supervised coactivation presentation, no event ID involved."""
        x = self._input(cue)
        y = select_features(encoded_content, top_k=self.content_features)
        source_ptr = self.topology.indptr
        targets = self.topology.indices
        counts = self.topology.synapse_counts
        updated = 0
        for i in np.flatnonzero(x):
            for pre in self.pre_cells[int(i)]:
                start, stop = int(source_ptr[pre]), int(source_ptr[pre + 1])
                if start == stop:
                    continue
                feats = self.post_feature[targets[start:stop]]
                valid = feats >= 0
                if not np.any(valid):
                    continue
                relative = np.flatnonzero(valid)
                feat_ids = feats[relative].astype(np.intp)
                nonzero = y[feat_ids] != 0
                relative = relative[nonzero]
                if not len(relative):
                    continue
                coactive = feat_ids[nonzero]
                edge_ids = relative + start
                increments = (
                    self.rate * x[int(i)] * y[coactive]
                    * np.log1p(counts[edge_ids].astype(np.float32))
                )
                self.delta[edge_ids] = np.clip(
                    self.delta[edge_ids] + increments, -self.weight_cap, self.weight_cap
                )
                updated += len(edge_ids)
        self.imprints += 1
        self.edge_update_events += updated
        return updated

    def infer(self, cue: str) -> np.ndarray:
        """Model-only numeric readout: no training corpus or candidate texts."""
        x = self._input(cue)
        decoded = np.zeros(DIM, dtype=np.float64)
        ptr = self.topology.indptr
        targets = self.topology.indices
        for i in np.flatnonzero(x):
            for pre in self.pre_cells[int(i)]:
                start, stop = int(ptr[pre]), int(ptr[pre + 1])
                if start == stop:
                    continue
                feat = self.post_feature[targets[start:stop]]
                allowed = (feat >= 0) & (self.delta[start:stop] != 0)
                if not np.any(allowed):
                    continue
                decoded += float(x[int(i)]) * np.bincount(
                    feat[allowed].astype(np.intp),
                    weights=self.delta[start:stop][allowed].astype(np.float64),
                    minlength=DIM,
                )
        decoded /= self.cells_per_feature
        norm = np.linalg.norm(decoded)
        if norm > 0:
            decoded /= norm
        return decoded.astype(np.float32)

    @property
    def modified_edges(self) -> int:
        return int(np.count_nonzero(self.delta))

    def assert_original_unchanged(self) -> None:
        if fingerprint(self.topology) != self.original_sha256:
            raise AssertionError("Original biological neurons/edges/contact counts mutated")

    def save(self, filename: Path) -> str:
        """Persist the learned synaptic state, no target codebook or corpus."""
        self.assert_original_unchanged()
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)
        active = np.flatnonzero(self.delta).astype(np.int32)
        np.savez_compressed(
            filename, edge_positions=active, learned_deltas=self.delta[active],
            anatomical_array_sha256=np.asarray(
                [self.original_sha256[k] for k in
                 ("root_ids", "indptr", "indices", "synapse_counts")],
                dtype="U64",
            ),
            settings=np.asarray([
                self.seed, self.cells_per_feature, self.cue_features,
                self.content_features, self.imprints, self.edge_update_events
            ], dtype=np.int64),
            rate=np.float64(self.rate), weight_cap=np.float64(self.weight_cap),
        )
        return sha256(filename.read_bytes()).hexdigest()

    @classmethod
    def load(cls, topology: Topology, filename: Path) -> "DirectFlywireOverlay":
        """Source-bound checkpoint replay without autobiographical source."""
        with np.load(filename, allow_pickle=False) as archive:
            config = archive["settings"].astype(np.int64)
            if config.shape != (6,):
                raise ValueError("Malformed overlay checkpoint settings")
            seed, group, cue, content, n, nupdates = map(int, config)
            model = cls(
                topology, seed=seed, cells_per_feature=group, cue_features=cue,
                content_features=content, rate=float(archive["rate"]),
                weight_cap=float(archive["weight_cap"]),
            )
            old = [str(k) for k in archive["anatomical_array_sha256"]]
            now = [model.original_sha256[k] for k in
                   ("root_ids", "indptr", "indices", "synapse_counts")]
            if old != now:
                raise ValueError("Learned overlay bound to a different original CSR")
            positions = archive["edge_positions"]
            learned = archive["learned_deltas"]
            if (positions.ndim != 1 or learned.shape != positions.shape
                or positions.dtype.kind not in "iu" or learned.dtype != np.float32
                or np.any(np.diff(positions.astype(np.int64)) <= 0)
                or np.any(positions < 0) or np.any(positions >= len(model.delta))
                or not np.all(np.isfinite(learned))
                or np.any(np.abs(learned) > model.weight_cap)
                or n < 0 or nupdates < len(positions)):
                raise ValueError("Invalid overlay weights or edge identifiers")
            model.delta[positions] = learned
            model.imprints = n
            model.edge_update_events = nupdates
        model.assert_original_unchanged()
        return model
