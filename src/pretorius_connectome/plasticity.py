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


def support_matched_null(topology, *, seed: int, attempts_per_edge: int = 3):
    """Rewire existing directed edges without changing effective graph capacity.

    Two distinct-source target stubs are exchanged only when neither new edge
    already exists, the old endpoints have multiplicity one, and neither swap
    creates a new self-loop. Existing parallel edges, if any, stay untouched.
    This preserves exactly: per-source raw outdegree and unique target count,
    global target indegree stub counts, CSR edge count after deduplication,
    row-wise synapse-count multisets and the original root-ID inventory.

    Weighted target indegrees, motifs and neurotransmitter classes are NOT
    preserved. This is a support/degree-matched computational null.
    """
    from collections import Counter

    if attempts_per_edge < 1:
        raise ValueError("attempts_per_edge must be positive")
    original = topology.indices
    pointers = topology.indptr
    m = len(original)
    n = len(topology.root_ids)
    edges = original.copy()
    sources = np.repeat(np.arange(n, dtype=np.int32), np.diff(pointers))
    rows = [
        Counter(map(int, edges[int(pointers[i]):int(pointers[i + 1])]))
        for i in range(n)
    ]
    rng = np.random.default_rng(seed)
    attempts = int(m * attempts_per_edge)
    candidates = rng.integers(0, m, size=(attempts, 2), dtype=np.int32)
    accepted = 0
    for e0, e1 in candidates:
        u, v = int(e0), int(e1)
        if u == v:
            continue
        source_u, source_v = int(sources[u]), int(sources[v])
        if source_u == source_v:
            continue
        target_u, target_v = int(edges[u]), int(edges[v])
        if (target_u == target_v or target_v == source_u
                or target_u == source_v):
            continue
        row_u, row_v = rows[source_u], rows[source_v]
        if (row_u[target_u] != 1 or row_v[target_v] != 1
                or target_v in row_u or target_u in row_v):
            continue
        del row_u[target_u]
        del row_v[target_v]
        row_u[target_v] = 1
        row_v[target_u] = 1
        edges[u], edges[v] = target_v, target_u
        accepted += 1

    # The old permutation stub null collapsed many CSR duplicates during
    # transition construction. Prove the replacement never does so.
    original_support = sum(
        len(set(map(int, original[int(pointers[i]):int(pointers[i + 1])])))
        for i in range(n)
    )
    changed_support = sum(len(row) for row in rows)
    if (changed_support != original_support
            or not np.array_equal(np.sort(original), np.sort(edges))
            or not np.array_equal(topology.indices, original)
            or accepted < max(1, m // 10)):
        raise AssertionError("degree/capacity-matched rewiring invariant failed")

    from pretorius_connectome.associative import Topology
    rewired = Topology(
        topology.root_ids.copy(), topology.indptr.copy(), edges,
        topology.synapse_counts.copy(),
        "unique-support and in/out-stub-preserving edge-swap null of "
        + topology.provenance,
    )
    return rewired, {
        "null_method": "existing-edge target swaps without new duplicate support",
        "attempted_swaps": attempts,
        "accepted_swaps": accepted,
        "original_aggregate_edges": m,
        "original_unique_edges": original_support,
        "rewired_aggregate_edges": len(edges),
        "rewired_unique_edges": changed_support,
        "target_stub_degree_preserved": True,
        "source_degree_and_support_preserved": True,
        "synapse_counts_preserved": True,
        "weight_in_degree_preserved": False,
        "synapse_type_preserved": False,
        "seed": int(seed),
    }
