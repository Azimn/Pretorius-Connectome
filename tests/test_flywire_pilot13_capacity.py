"""Independent guards for Pilot13 staged FlyWire memory-interference claims."""
from __future__ import annotations

import unittest
import numpy as np

from pretorius_connectome.associative import Topology
from pretorius_connectome.direct_flywire_imprint import (
    DirectFlywireOverlay, fingerprint, select_features,
)
from pretorius_connectome.imprinting import Cue, Memory
from pretorius_connectome.rewire12 import (
    synthetic_aggregated_fixture, rewire_effective_edges,
)
from pretorius_connectome.shared_features_bc01 import encode_bc_sensory
from scripts.run_flywire_pilot13_capacity import (
    CUTS, ARM_NAMES, ANCHORS, rank_measured, describe, rank_auc,
    compare_original_pilot10_rows,
)


def example(event_id, words):
    return Memory(
        event_id, event_id.split("-")[0], words,
        (Cue("first", "red glass"), Cue("middle", "wooden room"),
         Cue("last", words)),
    )


class Pilot13CapacityTests(unittest.TestCase):
    def test_stage_count_and_arm_frozen_contract(self):
        self.assertEqual(CUTS, (0, 16, 32, 64, 128, 256, 317))
        self.assertEqual(ANCHORS, 16)
        self.assertEqual(len(ARM_NAMES), 4)

    def test_oracle_is_outside_neural_inference(self):
        topo = Topology.synthetic(n=2048, degree=32, seed=41)
        model = DirectFlywireOverlay(topo, cells_per_feature=4, seed=31)
        a = example("E01-001", "blue ribbon and stone tower")
        b = example("E01-002", "iron locomotive through fog")
        vec = np.stack([
            select_features(encode_bc_sensory(x.memory_text), top_k=32)
            for x in (a, b)
        ])
        originals = fingerprint(topo)
        zero = rank_measured(model, [a, b], [a.event_id, b.event_id],
                             vec, learned_ids=set())
        self.assertEqual(len(zero), 2)
        self.assertEqual(describe(zero)["correct_top1"], 0)
        self.assertFalse(zero[0]["was_imprinted_by_this_stage"])
        self.assertIsNone(zero[0]["predicted_event_id"])
        self.assertEqual(rank_auc(zero, [
            {"best_cosine": 0.0}
        ]), .5)
        for literal in ("red glass", "wooden room", a.memory_text):
            model.imprint(literal, encode_bc_sensory(a.memory_text))
        learned = rank_measured(model, [a, b], [a.event_id, b.event_id],
                                vec, learned_ids={a.event_id})
        self.assertTrue(learned[0]["was_imprinted_by_this_stage"])
        self.assertFalse(learned[1]["was_imprinted_by_this_stage"])
        self.assertIsInstance(learned[0]["true_content_margin"], float)
        self.assertEqual(fingerprint(topo), originals)
        self.assertNotIn("event_ids", model.__dict__)
        self.assertNotIn("target_vectors", model.__dict__)

    def test_heldout_absent_and_external_candidate_universe(self):
        topo = Topology.synthetic(n=2048, degree=32, seed=17)
        model = DirectFlywireOverlay(topo, cells_per_feature=4)
        a = example("E01-001", "green glass")
        b = example("E02-001", "silver telescope")
        c = example("E99-001", "an absent corridor")
        refs = np.stack([
            select_features(encode_bc_sensory(m.memory_text), top_k=32)
            for m in (a, b)
        ])
        unknown = rank_measured(model, [c], [a.event_id, b.event_id],
                                refs, learned_ids={a.event_id})
        self.assertFalse(unknown[0]["has_oracle_target"])
        self.assertIsNone(unknown[0]["true_content_margin"])
        self.assertEqual(describe(unknown)["absent_false_acceptances"], 0)
        with self.assertRaises(AssertionError):
            describe(rank_measured(model, [a], [a.event_id, b.event_id],
                                   refs, learned_ids=set()) + unknown)

    def test_replay_guard_refuses_wrong_event_even_if_scores_match(self):
        a = [{
            "event_id": "E01-001", "predicted_event_id": "E01-002",
            "correct_top1": False, "neural_response_nonzero": True,
            "best_cosine": 0.23456781,
        }]
        b = [{
            "event_id": "E01-001", "predicted": "E01-002",
            "correct": False, "response_norm": 1.0,
            "best_cosine": 0.2345678,
        }]
        compare_original_pilot10_rows(a, b, "fixture")
        b[0]["predicted"] = "E01-001"
        with self.assertRaises(AssertionError):
            compare_original_pilot10_rows(a, b, "fixture")

    def test_rewiring_never_modifies_biological_arrays(self):
        graph = synthetic_aggregated_fixture(n=4096, degree=24, seed=61)
        before = fingerprint(graph)
        model = DirectFlywireOverlay(graph, seed=31, cells_per_feature=8)
        switched, proof = rewire_effective_edges(
            graph, model.pre_cells, model.post_cells, seed=73
        )
        self.assertTrue(proof["binary_outdegrees_equal"])
        self.assertTrue(proof["binary_indegrees_equal"])
        self.assertEqual(proof["originally_eligible_edges"],
                         proof["rewired_eligible_edges"])
        self.assertGreater(proof["changed_edge_destinations"], 0)
        self.assertEqual(fingerprint(graph), before)
        self.assertNotEqual(fingerprint(switched)["indices"], before["indices"])


if __name__ == "__main__":
    unittest.main()
