"""Pilot12 reproducible degree/support-matched wiring null for fly imprinting.

Swaps ONLY directed edges from the fixed Pilot08/10 feature-encoding
presynaptic neuron pool to the fixed output-readout neuron pool. The original
source graph is immutable. Every source neuron retains the same number of
eligible outgoing edges; every readout neuron retains the same number of
eligible incoming edges. Original per-edge integer synapse_counts stay at
their source-row edge position. No new duplicate directed pairs are allowed.

Preserves trainable synapse SLOT count and binary pre/post degree sequence,
but NOT contact-weighted in-degree of each postsynaptic neuron, nor the exact
set of nonzero weights learned after training. Those are reported separately.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha256

import numpy as np

from pretorius_connectome.associative import Topology
from pretorius_connectome.direct_flywire_imprint import fingerprint


@dataclass(frozen=True)
class RewireReport:
    algorithm: str
    seed: int
    selected_pre_neurons: int
    selected_post_neurons: int
    originally_eligible_edges: int
    rewired_eligible_edges: int
    accepted_double_edge_swaps: int
    changed_edge_destinations: int
    attempts: int
    binary_outdegrees_equal: bool
    binary_indegrees_equal: bool
    same_number_of_synaptic_slots: bool
    same_original_source_row_contact_strength: bool
    same_original_total_integer_synapses: bool
    source_graph_unchanged: bool
    original_csr_array_sha256: dict[str, str]
    rewired_csr_indices_sha256: str


def rewire_effective_edges(
    source: Topology, pre_cells: np.ndarray, post_cells: np.ndarray,
    *, seed: int = 73, swaps_per_edge: int = 2
) -> tuple[Topology, dict]:
    """Degree-preserving double-edge switch on exactly existing learnable slots.

    Each accepted (source_a->target_x, source_b->target_y) swap becomes
    (source_a->target_y, source_b->target_x), skipping self-row, repeated
    destination and duplicate-pair candidates. No contacts are added/deleted.
    """
    if swaps_per_edge < 1:
        raise ValueError("swaps_per_edge must be positive")
    source_sha = fingerprint(source)
    pre = np.asarray(pre_cells, dtype=np.int64).ravel()
    post = np.asarray(post_cells, dtype=np.int64).ravel()
    node_count = len(source.root_ids)
    if (not len(pre) or not len(post)
        or np.any(pre < 0) or np.any(pre >= node_count)
        or np.any(post < 0) or np.any(post >= node_count)
        or len(np.unique(pre)) != len(pre)
        or len(np.unique(post)) != len(post)
        or np.intersect1d(pre, post).size):
        raise ValueError("Pre/post feature neuron populations must be disjoint unique IDs")
    is_post = np.zeros(node_count, dtype=np.bool_)
    is_post[post] = True
    src_ids, slots, targets = [], [], []
    neighbors = {}
    before_source_count = {}
    before_target_count = Counter()
    for source_id in pre.tolist():
        start, end = map(int, source.indptr[source_id:source_id + 2])
        # Original CSR has one directed entry per source-target neuron pair.
        local_edges = np.arange(start, end, dtype=np.int64)
        active = local_edges[is_post[source.indices[start:end]]]
        chosen_targets = [int(x) for x in source.indices[active]]
        if len(chosen_targets) != len(set(chosen_targets)):
            raise ValueError("Publisher source graph has duplicate directed neuron pairs")
        before_source_count[source_id] = len(active)
        neighbors[source_id] = set(chosen_targets)
        for edge, target in zip(active.tolist(), chosen_targets):
            slots.append(int(edge))
            src_ids.append(int(source_id))
            targets.append(int(target))
            before_target_count[target] += 1
    if not slots:
        raise ValueError("No eligible original source->readout anatomical edges")
    slots = np.asarray(slots, dtype=np.int64)
    src_ids = np.asarray(src_ids, dtype=np.int32)
    destinations = np.asarray(targets, dtype=np.int32)
    original_destinations = destinations.copy()
    rng = np.random.default_rng(seed)
    n = len(slots)
    target_successes = swaps_per_edge * n
    attempts_limit = max(40 * n, 5000)
    success = attempts = 0
    while success < target_successes and attempts < attempts_limit:
        attempts += 1
        i, j = map(int, rng.integers(0, n, size=2))
        a, c = int(src_ids[i]), int(src_ids[j])
        b, d = int(destinations[i]), int(destinations[j])
        if a == c or b == d:
            continue
        if d in neighbors[a] or b in neighbors[c]:
            continue
        neighbors[a].remove(b)
        neighbors[c].remove(d)
        neighbors[a].add(d)
        neighbors[c].add(b)
        destinations[i] = d
        destinations[j] = b
        success += 1
    if success < n:
        raise AssertionError("Failed degree-preserving rewiring: insufficient legal swaps")

    # Source receives NO writes. Only the null graph owns its copied edge index.
    new_indices = source.indices.copy()
    new_indices[slots] = destinations
    post_after = Counter(map(int, destinations))
    same_out = (all(len(neighbors[src]) == count
                    for src, count in before_source_count.items()))
    same_in = (post_after == before_target_count)
    if not same_out or not same_in:
        raise AssertionError("Degree-preserving swap changed source/output degrees")
    if any(len(s) != before_source_count[k] for k, s in neighbors.items()):
        raise AssertionError("Null graph introduced duplicate target for one source")
    if fingerprint(source) != source_sha:
        raise AssertionError("Original biological graph mutated during control construction")

    control = Topology(
        source.root_ids, source.indptr, new_indices, source.synapse_counts,
        "original FlyWire directed-edge-degree-preserved double-switch control",
    )
    changed = int(np.count_nonzero(destinations != original_destinations))
    if changed < n // 2:
        raise AssertionError("Rewiring had insufficient actual target displacement")
    original_contacts = int(source.synapse_counts.sum(dtype=np.int64))
    new_contacts = int(control.synapse_counts.sum(dtype=np.int64))
    report = RewireReport(
        algorithm="unique-directed-target-2switch-on-feature-to-readout-support-v1",
        seed=int(seed),
        selected_pre_neurons=len(pre),
        selected_post_neurons=len(post),
        originally_eligible_edges=n,
        rewired_eligible_edges=n,
        accepted_double_edge_swaps=success,
        changed_edge_destinations=changed,
        attempts=attempts,
        binary_outdegrees_equal=same_out,
        binary_indegrees_equal=same_in,
        same_number_of_synaptic_slots=len(control.indices) == len(source.indices),
        same_original_source_row_contact_strength=True,
        same_original_total_integer_synapses=original_contacts == new_contacts,
        source_graph_unchanged=True,
        original_csr_array_sha256=source_sha,
        rewired_csr_indices_sha256=sha256(
            np.ascontiguousarray(control.indices).tobytes()
        ).hexdigest(),
    )
    return control, asdict(report)


def synthetic_aggregated_fixture(*, n: int = 8192, degree: int = 16,
                                 seed: int = 15) -> Topology:
    """Convert synthetic edge MULTIgraph to unique-pair aggregated CSR.

    Real FlyWire's published CSR contains one aggregated source/destination
    entry with integer synaptic contact count. This synthetic-only fixture
    removes synthetic parallel pairs by summing their original integer
    contact counts; otherwise it would violate real-connectome assumptions.
    Nothing from this fixture can substantiate a biological result.
    """
    source = Topology.synthetic(n=n, degree=degree, seed=seed)
    indptr = [0]
    indices = []
    counts = []
    for neuron in range(len(source.root_ids)):
        start, end = map(int, source.indptr[neuron:neuron+2])
        neighbors = source.indices[start:end]
        contacts = source.synapse_counts[start:end]
        if len(neighbors):
            unique, inverse = np.unique(neighbors, return_inverse=True)
            sums = np.bincount(
                inverse, weights=contacts.astype(np.float64),
                minlength=len(unique)
            ).astype(source.synapse_counts.dtype)
            indices.extend(unique.tolist())
            counts.extend(sums.tolist())
        indptr.append(len(indices))
    synthetic = Topology(
        source.root_ids,
        np.asarray(indptr, dtype=source.indptr.dtype),
        np.asarray(indices, dtype=source.indices.dtype),
        np.asarray(counts, dtype=source.synapse_counts.dtype),
        "synthetic-only unique directed-pair aggregated neural fixture",
    )
    if int(synthetic.synapse_counts.sum(dtype=np.int64)) != int(
            source.synapse_counts.sum(dtype=np.int64)):
        raise AssertionError("Synthetic multigraph contact totals not preserved")
    return synthetic
