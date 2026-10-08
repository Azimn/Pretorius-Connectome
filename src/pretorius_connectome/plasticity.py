"""Train-only activity-trace plasticity on an existing directed sparse topology.

This is an experimental *weight overlay*, not a change to FlyWire synapse
counts, transmitter signs, anatomical edges, or a validated biological STDP
model. It measures whether a simple Hebbian-like graph reweighting offers any
benefit over equally trained rewired graphs and unchanged text retrieval.
"""
from __future__ import annotations

from hashlib import sha256

import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize

from pretorius_connectome.associative import AssociativeMemory, _prune_rows


def fit_trace_overlay(engine: AssociativeMemory, *, gain: float = 2.0,
                      activity_cap: int = 256) -> tuple[sparse.csr_matrix, dict]:
    """Return a *new* row-stochastic operator and an auditable learning report.

    Each *training narrative only* supplies presynaptic lexical activity.
    One frozen-graph step supplies postsynaptic activity. Existing directed
    edges accumulate pre*post coactivity across all training narratives.
    Edge multipliers range from 1 to (1+gain), then each row is normalized.
    No labels, query texts, held-out episodes or oracle memory IDs are used.
    Anatomical counts and raw CSR arrays are never mutated.
    """
    if engine.projection is None or engine.propagator is None:
        raise ValueError("plasticity requires an indexed graph")
    if not np.isfinite(gain) or gain < 0:
        raise ValueError("gain must be finite and nonnegative")
    if activity_cap < 1:
        raise ValueError("activity_cap must be positive")

    frozen = engine.propagator.tocsr(copy=True)
    frozen.sum_duplicates()
    frozen.sort_indices()
    raw_counts = engine.topology.synapse_counts.copy()
    raw_indices = engine.topology.indices.copy()
    raw_ptr = engine.topology.indptr.copy()
    input_state = normalize(
        _prune_rows(engine.docs @ engine.projection, activity_cap)
    ).tocsr()
    output_state = normalize(
        _prune_rows(input_state @ frozen, activity_cap)
    ).tocsr()
    trace = np.zeros(len(frozen.data), dtype=np.float64)

    # Work only at active presynaptic nodes and their existing outgoing edges.
    # This avoids a dense N*N weight matrix for real FlyWire connectivity.
    for row in range(input_state.shape[0]):
        ins = input_state.indices[input_state.indptr[row]:input_state.indptr[row + 1]]
        inv = input_state.data[input_state.indptr[row]:input_state.indptr[row + 1]]
        outs = output_state.indices[output_state.indptr[row]:output_state.indptr[row + 1]]
        outv = output_state.data[output_state.indptr[row]:output_state.indptr[row + 1]]
        if not len(outs):
            continue
        for source, pre_value in zip(ins, inv):
            lo, hi = int(frozen.indptr[source]), int(frozen.indptr[source + 1])
            targets = frozen.indices[lo:hi]
            if not len(targets):
                continue
            positions = np.searchsorted(outs, targets)
            inside = positions < len(outs)
            matches = np.flatnonzero(inside)
            matches = matches[outs[positions[matches]] == targets[matches]]
            if len(matches):
                trace[lo + matches] += float(pre_value) * outv[positions[matches]]

    peak = float(trace.max(initial=0.0))
    learned = frozen.copy()
    if gain > 0 and peak > 0:
        learned.data *= 1.0 + (gain / peak) * trace
        row_totals = np.asarray(learned.sum(axis=1)).ravel()
        learned = (sparse.diags(1.0 / np.maximum(row_totals, 1e-12))
                   @ learned).tocsr()
        learned.sort_indices()

    # Structural protection is checked even if the overlay is disabled.
    if (not np.array_equal(raw_counts, engine.topology.synapse_counts)
            or not np.array_equal(raw_indices, engine.topology.indices)
            or not np.array_equal(raw_ptr, engine.topology.indptr)
            or learned.shape != frozen.shape
            or not np.array_equal(learned.indptr, frozen.indptr)
            or not np.array_equal(learned.indices, frozen.indices)):
        raise AssertionError("training modified published connectivity structure")
    if not np.isfinite(learned.data).all():
        raise AssertionError("nonfinite learned overlay")

    audit = {
        "rule": "presynaptic x frozen-one-hop-postsynaptic coactivity",
        "train_narrative_count": int(engine.docs.shape[0]),
        "feature_count": int(engine.docs.shape[1]),
        "activity_cap": int(activity_cap),
        "gain": float(gain),
        "edge_parameters": int(len(frozen.data)),
        "edges_with_positive_trace": int(np.count_nonzero(trace)),
        "maximum_trace": peak,
        "baseline_operator_sha256": sha256(frozen.data.tobytes()).hexdigest(),
        "learned_operator_sha256": sha256(learned.data.tobytes()).hexdigest(),
        "anatomical_counts_unchanged": True,
        "original_edge_targets_unchanged": True,
        "source": "indexed train narratives only; no challenge prompts or labels",
    }
    return learned, audit


def apply_trace_overlay(engine: AssociativeMemory, *, gain: float = 2.0,
                        activity_cap: int = 256) -> dict:
    """Install the learned operator on an isolated engine, then re-embed docs."""
    trained, audit = fit_trace_overlay(
        engine, gain=gain, activity_cap=activity_cap
    )
    engine.propagator = trained
    engine.graph_docs = engine._graph(engine.docs)
    return audit
