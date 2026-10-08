"""Deterministic safeguards for train-only FlyWire trace-plasticity Pilot 01."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pretorius_connectome.associative import AssociativeMemory, Topology
from pretorius_connectome.imprinting import Cue, Memory
from pretorius_connectome.plasticity import fit_trace_overlay, apply_trace_overlay, support_matched_null


def sample():
    return [
        Memory("A", "E1",
               "I kept brass instruments by the laboratory window and a lamp.",
               (Cue("ca", "brass"),)),
        Memory("B", "E2",
               "The church bells rang as Clara arrived in the cathedral.",
               (Cue("cb", "Clara"),)),
        Memory("C", "E3",
               "In the garden I examined the rain soaked violet flowers.",
               (Cue("cc", "violet"),)),
    ]


class PlasticityTests(unittest.TestCase):

    def test_rewired_control_preserves_exact_directed_support_and_degrees(self):
        topology = Topology.synthetic(n=128, degree=7, seed=14)
        raw_indices = topology.indices.copy()
        raw_counts = topology.synapse_counts.copy()
        rewired, audit = support_matched_null(topology, seed=941)
        repeat, other_audit = support_matched_null(topology, seed=941)
        self.assertEqual(audit, other_audit)
        np.testing.assert_array_equal(rewired.indices, repeat.indices)
        np.testing.assert_array_equal(topology.indices, raw_indices)
        np.testing.assert_array_equal(topology.synapse_counts, raw_counts)
        np.testing.assert_array_equal(topology.indptr, rewired.indptr)
        np.testing.assert_array_equal(topology.synapse_counts, rewired.synapse_counts)
        np.testing.assert_array_equal(np.sort(topology.indices), np.sort(rewired.indices))
        self.assertEqual(topology.transition().nnz, rewired.transition().nnz)
        self.assertEqual(audit["original_unique_edges"], audit["rewired_unique_edges"])
        self.assertGreater(audit["accepted_swaps"], 0)
        self.assertFalse(np.array_equal(raw_indices, rewired.indices))
        for row in range(len(topology.root_ids)):
            first, last = int(topology.indptr[row]), int(topology.indptr[row + 1])
            self.assertEqual(
                len(set(raw_indices[first:last])),
                len(set(rewired.indices[first:last]))
            )
        with self.assertRaises(ValueError):
            support_matched_null(topology, seed=941, attempts_per_edge=0)

    def test_train_only_overlay_is_deterministic_and_structurally_safe(self):
        graph = Topology.synthetic(n=64, degree=8, seed=5)
        snapshots = [x.copy() for x in
                     (graph.root_ids, graph.indptr, graph.indices, graph.synapse_counts)]
        first = AssociativeMemory(sample(), graph)
        second = AssociativeMemory(sample(), graph)
        frozen = first.propagator.copy()
        trained, evidence = fit_trace_overlay(first)
        repeat, repeated_evidence = fit_trace_overlay(second)

        self.assertEqual(evidence, repeated_evidence)
        self.assertGreater(evidence["edges_with_positive_trace"], 0)
        self.assertEqual(evidence["train_narrative_count"], 3)
        self.assertEqual(evidence["edge_parameters"], frozen.nnz)
        self.assertFalse(np.array_equal(trained.data, frozen.data))
        np.testing.assert_array_equal(trained.data, repeat.data)
        expected_structure = frozen.copy()
        expected_structure.sort_indices()
        np.testing.assert_array_equal(trained.indices, expected_structure.indices)
        np.testing.assert_array_equal(trained.indptr, expected_structure.indptr)
        for actual, old in zip((graph.root_ids, graph.indptr, graph.indices,
                                graph.synapse_counts), snapshots):
            np.testing.assert_array_equal(actual, old)
        np.testing.assert_allclose(np.asarray(trained.sum(axis=1)).ravel(), 1.0)
        self.assertEqual(first.propagator.shape, trained.shape)
        np.testing.assert_array_equal(first.propagator.data, frozen.data)

    def test_zero_gain_keeps_frozen_exactly_and_rejects_bad_inputs(self):
        model = AssociativeMemory(sample(), Topology.synthetic(n=64, degree=5))
        frozen = model.propagator.copy()
        unchanged, report = fit_trace_overlay(model, gain=0)
        self.assertEqual((unchanged - frozen).nnz, 0)
        self.assertEqual(report["gain"], 0)
        for gain in (-1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                fit_trace_overlay(model, gain=gain)
        with self.assertRaises(ValueError):
            fit_trace_overlay(model, activity_cap=0)
        with self.assertRaises(ValueError):
            fit_trace_overlay(AssociativeMemory(sample()))

    def test_learning_does_not_change_lexical_fallback(self):
        model = AssociativeMemory(sample(), Topology.synthetic(n=64, degree=5))
        original = model.score("brass instrument lamp", "lexical")
        report = apply_trace_overlay(model)
        self.assertTrue(report["anatomical_counts_unchanged"])
        np.testing.assert_array_equal(original, model.score("brass instrument lamp", "lexical"))
        self.assertTrue(np.isfinite(model.score("brass instrument lamp", "graph")).all())
        self.assertIn("retrieved_excerpt", model.rank("cathedral Clara", "graph")[0])

    def test_unchanged_training_vectorizer_is_fitted_only_on_indexed_events(self):
        a = sample()[:2]
        b = sample()[:2]
        model_a = AssociativeMemory(a, Topology.synthetic(n=64, degree=4, seed=8))
        model_b = AssociativeMemory(b, Topology.synthetic(n=64, degree=4, seed=8))
        learned_a, audit_a = fit_trace_overlay(model_a)
        learned_b, audit_b = fit_trace_overlay(model_b)
        np.testing.assert_array_equal(learned_a.data, learned_b.data)
        self.assertEqual(audit_a, audit_b)
        self.assertNotIn("violet", model_a.encoder.vocabulary_)


if __name__ == "__main__":
    unittest.main()
