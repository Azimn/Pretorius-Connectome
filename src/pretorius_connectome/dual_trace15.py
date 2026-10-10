"""Pilot15: two timescale source-cue associative memory without event lookup.

A slower usage-protected original-edge channel and a faster exponentially
decaying original-edge channel learn the same source cue->signed BC01 content
under a partitioned total rate, and their individually normalized cue-only
responses are blended. No source IDs, source corpus, archive, event keys or
oracle candidate codebook are accepted by infer(cue). This is engineered
numerical metaplasticity, not a model of fly physiology.

Important cost disclaimer: two numerical trace arrays DOUBLE eligible
synaptic scalar capacity and model memory, compared with Pilot14's original
single channel; a two-trace matched-slot NONNEURAL comparator is mandatory.
The fast channel decays after each memory's exactly three presentations,
not after every cue. Stage 0 is initialized and tested.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import numpy as np

from pretorius_connectome.direct_flywire_imprint import (
    DirectFlywireOverlay, fingerprint
)
from pretorius_connectome.metaplastic14 import (
    UsageProtectedOverlay, MatchedSlotLinear
)


def eligible_slots(model: DirectFlywireOverlay) -> np.ndarray:
    """Unique eligible biological CSR edge positions, never source story IDs."""
    ptr = model.topology.indptr
    targets = model.topology.indices
    result = []
    for pre in model.pre_cells.ravel():
        start, stop = int(ptr[pre]), int(ptr[pre + 1])
        valid = model.post_feature[targets[start:stop]] >= 0
        result.extend((start + np.flatnonzero(valid)).tolist())
    arr = np.asarray(result, dtype=np.int32)
    if len(arr) != len(np.unique(arr)):
        raise ValueError("Original graph eligible directed edge support duplicated")
    return np.sort(arr)


class DualTraceSynapses:
    """Two independent source-edge numeric trace arrays on identical anatomy."""

    def __init__(self, graph, *, beta: float = 1.0, fast_decay: float = 0.97,
                 slow_share: float = 0.6, rate: float = 0.7/3,
                 seed: int = 31, cells_per_feature: int = 32,
                 cue_features: int = 8, content_features: int = 32):
        if (beta < 0 or not np.isfinite(beta)
            or not 0 < fast_decay < 1
            or not 0 < slow_share < 1):
            raise ValueError("Invalid fixed dual timescale parameters")
        self.topology = graph
        self.beta = float(beta)
        self.fast_decay = float(fast_decay)
        self.slow_share = float(slow_share)
        self.original_sha256 = fingerprint(graph)
        kw = dict(seed=seed, cells_per_feature=cells_per_feature,
                  cue_features=cue_features,
                  content_features=content_features)
        self.slow = UsageProtectedOverlay(
            graph, protection_beta=beta, rate=rate*slow_share, **kw
        )
        self.fast = DirectFlywireOverlay(
            graph, rate=rate*(1-slow_share), **kw
        )
        if (not np.array_equal(self.slow.pre_cells,self.fast.pre_cells)
            or not np.array_equal(self.slow.post_cells,self.fast.post_cells)):
            raise AssertionError("Dual traces must share the same source and readout neurons")
        self.eligible_positions = eligible_slots(self.slow)
        self.imprints = 0
        self.completed_memories = 0
        self.edge_update_events = 0

    @property
    def modified_edges(self):
        return int(np.count_nonzero(
            (self.slow.delta != 0) | (self.fast.delta != 0)
        ))

    def imprint(self, cue, encoded_content):
        a = self.slow.imprint(cue, encoded_content)
        b = self.fast.imprint(cue, encoded_content)
        if a != b:
            raise AssertionError("Dual channels did not update exact same eligible synaptic slots")
        self.imprints += 1
        self.edge_update_events += a + b
        return a + b

    def finish_memory(self):
        """For each fully trained three-cue memory: old fast trace attenuates."""
        if self.imprints != (self.completed_memories + 1)*3:
            raise AssertionError("Finish only after exactly 3 source-authored cues")
        # Decay only learnable slots, not all 15M biological edges.
        active=self.eligible_positions
        self.fast.delta[active] *= np.float32(self.fast_decay)
        self.completed_memories += 1

    def infer(self, cue):
        slow = self.slow.infer(cue)
        fast = self.fast.infer(cue)
        combined = (self.slow_share * slow.astype(np.float64)
                    + (1-self.slow_share) * fast.astype(np.float64))
        norm=float(np.linalg.norm(combined))
        if norm:
            combined /= norm
        return combined.astype(np.float32)

    def assert_original_unchanged(self):
        self.slow.assert_original_unchanged()
        self.fast.assert_original_unchanged()
        if fingerprint(self.topology)!=self.original_sha256:
            raise AssertionError("Original FlyWire source biology changed")

    @property
    def exposure_diagnostics(self):
        return {
            "slow":self.slow.exposure_diagnostics,
            "fast_nonzero_synapses":self.fast.modified_edges,
            "union_nonzero_synapses":self.modified_edges,
            "completed_memories":self.completed_memories,
            "independent_numeric_trace_slots":len(self.eligible_positions)*2,
            "total_source_edge_update_operations_both_traces":self.edge_update_events,
        }

    def save_dual(self, path):
        self.assert_original_unchanged()
        path=Path(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        pos=self.eligible_positions
        counts=self.slow.edge_exposures[pos]
        if (self.completed_memories * 3 != self.imprints
            or self.slow.imprints != self.imprints
            or self.fast.imprints != self.imprints):
            raise AssertionError("Incomplete independent channel exposure contract")
        np.savez_compressed(
            path, format_version=np.asarray([1],dtype=np.int64),
            original_array_sha256=np.asarray([
                self.original_sha256[k] for k in (
                    "root_ids","indptr","indices","synapse_counts")
            ], dtype="U64"),
            eligible_original_directed_edge_positions=pos,
            slow_signed_deltas=self.slow.delta[pos],
            fast_signed_deltas=self.fast.delta[pos],
            prior_synaptic_exposures=counts,
            settings=np.asarray([
                self.slow.seed, self.slow.cells_per_feature,
                self.slow.cue_features,self.slow.content_features,
                self.imprints,self.completed_memories,self.edge_update_events,
            ],dtype=np.int64),
            beta=np.asarray([self.beta],dtype=np.float64),
            fast_decay=np.asarray([self.fast_decay],dtype=np.float64),
            slow_share=np.asarray([self.slow_share],dtype=np.float64),
            total_rate=np.asarray([self.slow.rate + self.fast.rate],dtype=np.float64),
        )
        return sha256(path.read_bytes()).hexdigest()

    @classmethod
    def load_dual(cls,graph,path):
        with np.load(path,allow_pickle=False) as z:
            if z["format_version"].tolist()!=[1]:
                raise ValueError("Unknown dual source synaptic state version")
            opts=z["settings"]
            if opts.shape!=(7,):
                raise ValueError("Malformed dual-state training counters")
            model=cls(
                graph, seed=int(opts[0]),cells_per_feature=int(opts[1]),
                cue_features=int(opts[2]),content_features=int(opts[3]),
                beta=float(z["beta"][0]),fast_decay=float(z["fast_decay"][0]),
                slow_share=float(z["slow_share"][0]),
                rate=float(z["total_rate"][0])
            )
            if z["original_array_sha256"].tolist()!=[
                model.original_sha256[k] for k in
                ("root_ids","indptr","indices","synapse_counts")
            ]:
                raise ValueError("Original anatomical graph hash differs")
            pos=z["eligible_original_directed_edge_positions"]
            if not np.array_equal(pos,model.eligible_positions):
                raise ValueError("Trained directed synapse pool differs from exact source")
            for name in ("slow_signed_deltas","fast_signed_deltas",
                         "prior_synaptic_exposures"):
                if z[name].shape!=pos.shape:
                    raise ValueError("Incomplete source-bound synaptic weights or counts")
            if (not np.all(np.isfinite(z["slow_signed_deltas"]))
                or not np.all(np.isfinite(z["fast_signed_deltas"]))
                or np.any(np.abs(z["slow_signed_deltas"])>4)
                or np.any(np.abs(z["fast_signed_deltas"])>4)):
                raise ValueError("Nonfinite or out-of-bound signed learned state")
            model.slow.delta[pos]=z["slow_signed_deltas"]
            model.fast.delta[pos]=z["fast_signed_deltas"]
            model.slow.edge_exposures[pos]=z["prior_synaptic_exposures"]
            model.imprints=int(opts[4])
            model.completed_memories=int(opts[5])
            model.edge_update_events=int(opts[6])
            model.slow.imprints=model.fast.imprints=model.imprints
            model.slow.edge_update_events=int(np.sum(
                model.slow.edge_exposures,dtype=np.int64))
            model.fast.edge_update_events=model.slow.edge_update_events
            if (model.slow.edge_update_events * 2 != model.edge_update_events
                or model.imprints != model.completed_memories*3):
                raise ValueError("Stored per-synapse event count contradicts source training schedule")
        model.assert_original_unchanged()
        return model


class DualTraceLinear:
    """Two-trace nonneural numerical comparator matched to DOUBLE slot count.

    Same trainable source-feature mask for both traces (2 x slot_count scalar
    states), and identical presentation schedule, per-trace allocated rate,
    linear fast-trace decay and post-normalization mixing. It does NOT match
    biological contact strengths or precise edge-update numerics.
    """

    def __init__(self,*,slot_count,seed=73,fast_decay=.97,
                 slow_share=.6,rate=.7/3):
        self.fast_decay=fast_decay
        self.slow_share=slow_share
        self.slow=MatchedSlotLinear(slot_count=slot_count,seed=seed,
                                    rate=rate*slow_share)
        self.fast=MatchedSlotLinear(slot_count=slot_count,seed=seed,
                                    rate=rate*(1-slow_share))
        self.slot_count=slot_count
        self.imprints=0
        self.completed_memories=0
        self.edge_update_events=0

    @property
    def modified_edges(self):
        return int(np.count_nonzero(
            (self.slow.weights != 0) | (self.fast.weights != 0)
        ))

    def imprint(self,cue,content):
        n=self.slow.imprint(cue,content)
        m=self.fast.imprint(cue,content)
        if n!=m:
            raise AssertionError("Two-trace nonneural masks not matched")
        self.imprints+=1
        self.edge_update_events+=n+m
        return n+m

    def finish_memory(self):
        if self.imprints!=(self.completed_memories+1)*3:
            raise AssertionError("Linear fast decay only after three cues")
        self.fast.weights *= np.float32(self.fast_decay)
        self.completed_memories+=1

    def infer(self,cue):
        s=self.slow.infer(cue)
        f=self.fast.infer(cue)
        out=self.slow_share*s.astype(np.float64)+(1-self.slow_share)*f.astype(np.float64)
        norm=float(np.linalg.norm(out))
        if norm:
            out/=norm
        return out.astype(np.float32)

    @property
    def exposure_diagnostics(self):
        return {
            "two_separate_nonbiological_weight_matrices":True,
            "nonzero_union_mask_slots":self.modified_edges,
            "independent_numeric_trainable_slots":2*self.slot_count,
            "completed_memories":self.completed_memories,
        }

    def save_dual_linear(self,path):
        path=Path(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(
            path,format_version=np.asarray([1],dtype=np.int64),
            mask=self.slow.mask,
            slow_weights=self.slow.weights,fast_weights=self.fast.weights,
            settings=np.asarray([
                self.slot_count,self.imprints,self.completed_memories,
                self.edge_update_events],dtype=np.int64),
            fast_decay=np.asarray([self.fast_decay],dtype=np.float64),
            slow_share=np.asarray([self.slow_share],dtype=np.float64),
        )
        return sha256(path.read_bytes()).hexdigest()
