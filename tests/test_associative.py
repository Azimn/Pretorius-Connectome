"""Tests for topology import, graph controls, retrieval and invalid inputs."""
import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pretorius_connectome.associative import (
    AssociativeMemory, Topology, summarize_challenge, paired_case_diagnostics
)
from pretorius_connectome.imprinting import Memory, Cue
from pretorius_connectome.pilot04 import ChallengeCase


def sample():
    return [
        Memory("A", "E1",
               "I placed the brass instrument by the laboratory window. "
               "The lamp broke during the evening experiment.", (Cue("a", "brass"),)),
        Memory("B", "E2",
               "I spoke with Clara about our old agreement by the cathedral door. "
               "She did not forgive the forgotten appointment.", (Cue("b", "Clara"),)),
        Memory("C", "E3",
               "I repaired the wooden cabinet and arranged the violet flowers. "
               "The garden was wet after rain.", (Cue("c", "garden"),)),
    ]


class AssociativeTests(unittest.TestCase):
    def test_explicit_synthetic_control_and_determinism(self):
        a = Topology.synthetic(n=128, degree=4, seed=4)
        b = Topology.synthetic(n=128, degree=4, seed=4)
        self.assertIn("synthetic", a.provenance)
        np.testing.assert_array_equal(a.indices, b.indices)
        saved_indices, saved_ptr = a.indices.copy(), a.indptr.copy()
        self.assertEqual(a.transition().shape, (128, 128))
        np.testing.assert_array_equal(a.indices, saved_indices)
        np.testing.assert_array_equal(a.indptr, saved_ptr)
        total = np.asarray(a.transition().sum(axis=1)).ravel()
        self.assertTrue(np.allclose(total, 1.0))

    def test_permutation_preserves_stubs_not_measured_connections(self):
        a = Topology.synthetic(n=128, degree=5, seed=3)
        b = a.permuted_null(seed=11)
        np.testing.assert_array_equal(np.sort(a.indices), np.sort(b.indices))
        np.testing.assert_array_equal(a.indptr, b.indptr)
        np.testing.assert_array_equal(a.synapse_counts, b.synapse_counts)
        self.assertFalse(np.array_equal(a.indices, b.indices))
        self.assertIn("null", b.provenance)

    def test_load_roundtrip_and_corrupt_input(self):
        a = Topology.synthetic(n=64, degree=3)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "graph.npz"
            np.savez(p, root_ids=a.root_ids, indptr=a.indptr,
                     indices=a.indices, synapse_counts=a.synapse_counts)
            b = Topology.read(p)
            np.testing.assert_array_equal(a.indices, b.indices)
            np.savez(p, root_ids=np.arange(2), indptr=[0, 1, 2],
                     indices=[0, 99], synapse_counts=[1, 2])
            with self.assertRaises(ValueError):
                Topology.read(p)

    def test_lexical_graph_and_hybrid_return_source_evidence(self):
        train = sample()
        topo = Topology.synthetic(n=256, degree=4)
        first = AssociativeMemory(train, topo, steps=2, activity_cap=64)
        second = AssociativeMemory(train, topo, steps=2, activity_cap=64)
        for mode in ("lexical", "graph", "hybrid"):
            ranked = first.rank("brass instrument laboratory", mode)
            self.assertEqual(ranked[0]["event_id"], "A")
            self.assertIn(ranked[0]["retrieved_excerpt"], train[0].memory_text)
            self.assertEqual(first.rank("brass instrument laboratory", mode),
                             second.rank("brass instrument laboratory", mode))
            self.assertIn("not checked", ranked[0]["evidence_status"])

    def test_fallback_is_explicit_and_rejects_unavailable_graph(self):
        baseline = AssociativeMemory(sample())
        self.assertEqual(baseline.rank("cathedral appointment", "lexical")[0]["event_id"], "B")
        with self.assertRaises(ValueError):
            baseline.score("cathedral", "graph")
        with self.assertRaises(ValueError):
            baseline.rank("", "lexical")
        with self.assertRaises(ValueError):
            baseline.rank("x", "wrong")

    def test_challenge_cannot_change_training_or_answer_truth(self):
        train = sample()
        network = AssociativeMemory(train, Topology.synthetic(n=128, degree=3))
        cases = [
            ChallengeCase("p", "A", "paraphrase",
                          "brass laboratory instrument", ""),
            ChallengeCase("c", "A", "contradiction",
                          "brass instrument never existed", ""),
            ChallengeCase("u", "X", "paraphrase",
                          "an unfamiliar seaport and telescope", ""),
        ]
        result = summarize_challenge(
            network, cases, {"A", "B", "C"}, {"X"}, 0.1, "hybrid"
        )
        self.assertEqual(result["positive_n"], 1)
        self.assertEqual(result["contradiction_n"], 1)
        self.assertEqual(result["absent_n"], 1)
        self.assertEqual(result["mode"], "hybrid")


    def test_paired_diagnostics_identity_and_label_integrity(self):
        def group(case_id, target, predicted, accepted):
            return {"case_id": case_id, "target": target,
                    "predicted": predicted, "accepted": accepted, "score": 0.4}
        reference = {"case_results": {
            "positive": [
                group("p1", "A", "A", True),
                group("p2", "B", "A", True),
            ],
            "contradiction": [group("c1", "A", "A", True)],
            "absent": [group("u1", "X", "B", False)],
        }}
        challenger = {"case_results": {
            "positive": [
                group("p1", "A", "B", True),
                group("p2", "B", "B", True),
            ],
            "contradiction": [group("c1", "A", "A", False)],
            "absent": [group("u1", "X", "C", False)],
        }}
        result = paired_case_diagnostics(reference, challenger)
        self.assertEqual(result["positive"]["prediction_changes"], 2)
        self.assertEqual(result["positive"]["correct_and_accepted_gains"], 1)
        self.assertEqual(result["positive"]["correct_and_accepted_losses"], 1)
        self.assertEqual(result["contradiction"]["acceptance_changes"], 1)
        self.assertEqual(result["absent"]["prediction_changes"], 1)
        altered = {"case_results": {
            **challenger["case_results"],
            "positive": [group("p1", "X", "B", True),
                         group("p2", "B", "B", True)],
        }}
        with self.assertRaises(ValueError):
            paired_case_diagnostics(reference, altered)


if __name__ == "__main__":
    unittest.main()
