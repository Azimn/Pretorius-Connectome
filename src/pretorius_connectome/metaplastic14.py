"""Pilot14: usage-dependent synaptic metaplasticity and matched linear null.

Differentiate an interference-control mechanism from merely increasing raw
storage. No event IDs, source narratives, labels or retrieval candidates live
inside any of these models. The original FlyWire topology is read-only.

UsageProtectedOverlay is a numerical toy metaplasticity rule, NOT a model of
actual fly learning. Each existing directed synapse progressively reduces its
future update amplitude in proportion to the number of previous activations.
At protection_beta=0 the state and forward pass are byte-for-byte equivalent
to the original frozen Pilot10 DirectFlywireOverlay under identical conditions.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import numpy as np

from pretorius_connectome.direct_flywire_imprint import (
    DirectFlywireOverlay, fingerprint, select_features,
)
from pretorius_connectome.shared_features_bc01 import DIM, encode_bc_sensory


class UsageProtectedOverlay(DirectFlywireOverlay):
    """A fixed-size, cue-only existing-edge signed synaptic numerical system."""

    def __init__(self, topology, *, protection_beta: float = 1.0, **kwargs):
        super().__init__(topology, **kwargs)
        if not np.isfinite(protection_beta) or protection_beta < 0:
            raise ValueError("Nonnegative finite protection beta required")
        self.protection_beta = float(protection_beta)
        # Fixed state indexed by original biological directed synapse positions.
        # Stores exposure count ONLY, not per-event text or a learned event key.
        self.edge_exposures = np.zeros(len(topology.indices), dtype=np.uint16)

    def imprint(self, cue: str, encoded_content: np.ndarray) -> int:
        x = self._input(cue)
        y = select_features(encoded_content, top_k=self.content_features)
        ptr = self.topology.indptr
        targets = self.topology.indices
        counts = self.topology.synapse_counts
        updated = 0
        for i in np.flatnonzero(x):
            for pre in self.pre_cells[int(i)]:
                start, stop = int(ptr[pre]), int(ptr[pre + 1])
                if stop == start:
                    continue
                feats = self.post_feature[targets[start:stop]]
                relative = np.flatnonzero(feats >= 0)
                if not len(relative):
                    continue
                out_features = feats[relative].astype(np.intp)
                nonzero = y[out_features] != 0
                relative = relative[nonzero]
                if not len(relative):
                    continue
                feat = out_features[nonzero]
                positions = start + relative
                if np.any(self.edge_exposures[positions] == np.iinfo(np.uint16).max):
                    raise OverflowError("Synaptic exposure counter exhausted")
                baseline_increment = (
                    self.rate * x[int(i)] * y[feat] *
                    np.log1p(counts[positions].astype(np.float32))
                )
                # No per-record or source ID used. The first write has full
                # original strength; an existing often-used edge stabilizes.
                if self.protection_beta:
                    scale = 1.0 / (
                        1.0 + self.protection_beta *
                        self.edge_exposures[positions].astype(np.float32)
                    )
                    increment = baseline_increment * scale
                else:
                    increment = baseline_increment
                self.delta[positions] = np.clip(
                    self.delta[positions] + increment,
                    -self.weight_cap, self.weight_cap
                )
                self.edge_exposures[positions] += 1
                updated += len(positions)
        self.imprints += 1
        self.edge_update_events += updated
        return updated

    @property
    def exposure_diagnostics(self) -> dict:
        active = self.edge_exposures[self.edge_exposures > 0]
        return {
            "ever_touched_directed_edges": int(len(active)),
            "multiply_touched_directed_edges": int(np.count_nonzero(active > 1)),
            "max_exposures_of_one_directed_edge": int(active.max()) if len(active) else 0,
            "nonzero_learned_edge_positions": self.modified_edges,
            "edge_update_operations": self.edge_update_events,
        }

    def save_protected(self, path: Path) -> str:
        self.assert_original_unchanged()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        active = np.flatnonzero(self.delta != 0).astype(np.int32)
        exposure_active = np.flatnonzero(self.edge_exposures != 0).astype(np.int32)
        np.savez_compressed(
            path,
            format_version=np.asarray([1], dtype=np.int64),
            original_csr_sha256=np.asarray(
                [self.original_sha256[k] for k in (
                    "root_ids", "indptr", "indices", "synapse_counts"
                )], dtype="U64",
            ),
            settings=np.asarray([
                self.seed, self.cells_per_feature, self.cue_features,
                self.content_features, self.imprints, self.edge_update_events
            ], dtype=np.int64),
            protection_beta=np.asarray([self.protection_beta], dtype=np.float64),
            rate=np.asarray([self.rate], dtype=np.float64),
            weight_cap=np.asarray([self.weight_cap], dtype=np.float64),
            delta_positions=active,
            deltas=self.delta[active],
            exposure_positions=exposure_active,
            exposure_counts=self.edge_exposures[exposure_active],
        )
        return sha256(path.read_bytes()).hexdigest()

    @classmethod
    def load_protected(cls, topology, path):
        with np.load(path, allow_pickle=False) as saved:
            settings = saved["settings"]
            if settings.shape != (6,) or saved["format_version"].tolist() != [1]:
                raise ValueError("Wrong protected source synaptic checkpoint version")
            model = cls(
                topology, seed=int(settings[0]),
                cells_per_feature=int(settings[1]),
                cue_features=int(settings[2]), content_features=int(settings[3]),
                protection_beta=float(saved["protection_beta"][0]),
                rate=float(saved["rate"][0]),
                weight_cap=float(saved["weight_cap"][0]),
            )
            expected = [model.original_sha256[k] for k in (
                "root_ids", "indptr", "indices", "synapse_counts")]
            if saved["original_csr_sha256"].tolist() != expected:
                raise ValueError("Checkpoint belongs to a different original biological graph")
            a = saved["delta_positions"]
            b = saved["exposure_positions"]
            if (a.ndim != 1 or b.ndim != 1 or saved["deltas"].shape != a.shape
                or saved["exposure_counts"].shape != b.shape
                or a.dtype.kind not in "iu" or b.dtype.kind not in "iu"
                or np.any(np.diff(a) <= 0) or np.any(np.diff(b) <= 0)
                or np.any(a < 0) or np.any(b < 0)
                or np.any(a >= len(model.delta)) or np.any(b >= len(model.delta))
                or not np.all(np.isfinite(saved["deltas"]))
                or np.any(np.abs(saved["deltas"]) > model.weight_cap)
                or int(np.sum(saved["exposure_counts"].astype(np.int64))) !=
                    int(settings[5])):
                raise ValueError("Malformed sparse learned synaptic state")
            model.delta[a] = saved["deltas"]
            model.edge_exposures[b] = saved["exposure_counts"]
            model.imprints = int(settings[4])
            model.edge_update_events = int(settings[5])
        model.assert_original_unchanged()
        return model


class MatchedSlotLinear:
    """Non-biological 256x256 sparse signed associative matrix.

    Exactly slot_count random trainable scalar pairs (<=65,536). This matches
    the COUNT of FlyWire eligible original directed edges, not their graph
    structure, receptor strength, contact multiplicity or update norms.

    Model-only inference accepts a cue string and returns a vector.
    It does not store event IDs, candidate targets or narrative source.
    """

    def __init__(self, *, slot_count: int, seed: int = 73,
                 cue_features: int = 8, content_features: int = 32,
                 rate: float = 0.7 / 3):
        if not 1 <= slot_count <= DIM * DIM:
            raise ValueError("Linear comparator requires 1..65536 parameter slots")
        self.slot_count = int(slot_count)
        self.seed = seed
        self.cue_features = cue_features
        self.content_features = content_features
        self.rate = rate
        rng = np.random.default_rng(seed)
        self.mask = np.zeros(DIM * DIM, dtype=np.bool_)
        self.mask[rng.choice(DIM*DIM, slot_count, replace=False)] = True
        self.mask = self.mask.reshape(DIM, DIM)
        self.weights = np.zeros((DIM, DIM), dtype=np.float32)
        self.imprints = 0
        self.edge_update_events = 0

    def _input(self, cue):
        return select_features(encode_bc_sensory(cue), top_k=self.cue_features)

    def imprint(self, cue, content):
        x = self._input(cue)
        y = select_features(content, top_k=self.content_features)
        a = np.flatnonzero(x)
        b = np.flatnonzero(y)
        ix = np.ix_(a, b)
        allowed = self.mask[ix]
        updates = (self.rate * x[a, None] * y[None, b])
        window = self.weights[ix]
        window += np.where(allowed, updates, 0)
        np.clip(window, -4.0, 4.0, out=window)
        self.weights[ix] = window
        self.imprints += 1
        self.edge_update_events += int(np.count_nonzero(allowed))
        return int(np.count_nonzero(allowed))

    def infer(self, cue):
        result = self._input(cue) @ self.weights
        norm = float(np.linalg.norm(result))
        if norm:
            result /= norm
        return result.astype(np.float32)

    @property
    def modified_edges(self):
        return int(np.count_nonzero(self.weights))

    def save_linear(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, format_version=np.asarray([1], dtype=np.int64),
            weights=self.weights, mask=self.mask,
            settings=np.asarray([
                self.slot_count, self.seed, self.cue_features,
                self.content_features, self.imprints, self.edge_update_events
            ], dtype=np.int64),
            rate=np.asarray([self.rate], dtype=np.float64),
        )
        return sha256(path.read_bytes()).hexdigest()
