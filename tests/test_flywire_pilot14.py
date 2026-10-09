"""Pilot14 source-bound metaplasticity and slot-matched linear controls."""
from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import numpy as np

from pretorius_connectome.associative import Topology
from pretorius_connectome.direct_flywire_imprint import (
    DirectFlywireOverlay, fingerprint, select_features,
)
from pretorius_connectome.metaplastic14 import (
    UsageProtectedOverlay, MatchedSlotLinear,
)
from pretorius_connectome.shared_features_bc01 import encode_bc_sensory
from scripts.run_flywire_pilot14 import (
    predeclared_models, CONDITIONS, MAX_BASELINE_NUMERIC_TOLERANCE,
)


class Pilot14Controls(unittest.TestCase):
    def test_protection_zero_is_exact_original_rule(self):
        graph = Topology.synthetic(n=4096, degree=32, seed=19)
        base = DirectFlywireOverlay(
            graph, seed=31, cells_per_feature=4,
            rate=.7/3,
        )
        variant = UsageProtectedOverlay(
            graph, protection_beta=0., seed=31, cells_per_feature=4,
            rate=.7/3,
        )
        original_sha = fingerprint(graph)
        cues = ["glass machinery", "iron town", "old garden", "silver mirror"] * 3
        for literal in cues:
            content = encode_bc_sensory("transmitter " + literal)
            self.assertEqual(base.imprint(literal, content),
                             variant.imprint(literal, content))
        self.assertTrue(np.array_equal(base.delta, variant.delta))
        self.assertEqual(base.edge_update_events, variant.edge_update_events)
        for cue in cues[:4]:
            self.assertTrue(np.array_equal(base.infer(cue), variant.infer(cue)))
        self.assertEqual(fingerprint(graph), original_sha)

    def test_protection_keeps_first_write_then_reduces_subsequent_gain(self):
        graph = Topology.synthetic(n=4096, degree=128, seed=12)
        base = UsageProtectedOverlay(
            graph, protection_beta=0., seed=31, cells_per_feature=4,
            rate=.7/3,
        )
        protected = UsageProtectedOverlay(
            graph, protection_beta=4., seed=31, cells_per_feature=4,
            rate=.7/3,
        )
        cue = "bronze physician instruments"
        content = encode_bc_sensory("a cold hospital and moonlit corridor")
        self.assertGreater(base.imprint(cue, content), 0)
        protected.imprint(cue, content)
        self.assertTrue(np.array_equal(base.delta, protected.delta))
        base.imprint(cue, content)
        protected.imprint(cue, content)
        self.assertGreater(float(np.linalg.norm(base.delta)),
                           float(np.linalg.norm(protected.delta)))
        self.assertEqual(protected.edge_update_events, base.edge_update_events)
        self.assertEqual(protected.imprints, 2)
        self.assertGreater(
            protected.exposure_diagnostics["multiply_touched_directed_edges"], 0
        )
        self.assertNotIn("event_ids", protected.__dict__)
        self.assertNotIn("narrative_targets", protected.__dict__)
        with self.assertRaises(ValueError):
            UsageProtectedOverlay(graph, protection_beta=-1)

    def test_learned_protected_checkpoint_is_original_graph_bound(self):
        graph = Topology.synthetic(n=4096, degree=64, seed=6)
        model = UsageProtectedOverlay(
            graph, protection_beta=1.0, cells_per_feature=4,
            rate=.7/3,
        )
        for cue in ["iron", "glass", "glass"]:
            model.imprint(cue, encode_bc_sensory("fictional laboratory memory"))
        with tempfile.TemporaryDirectory() as temp:
            file = Path(temp) / "protected.npz"
            digest = model.save_protected(file)
            self.assertEqual(len(digest), 64)
            loaded = UsageProtectedOverlay.load_protected(graph, file)
            self.assertTrue(np.array_equal(loaded.delta, model.delta))
            self.assertTrue(np.array_equal(loaded.edge_exposures,
                                           model.edge_exposures))
            self.assertEqual(loaded.protection_beta, 1.0)
            self.assertEqual(loaded.edge_update_events, model.edge_update_events)
            different = Topology.synthetic(n=4096, degree=64, seed=7)
            with self.assertRaises(ValueError):
                UsageProtectedOverlay.load_protected(different, file)

    def test_non_neural_mask_is_exact_parameter_count_not_an_event_retriever(self):
        linear = MatchedSlotLinear(slot_count=30000, seed=73)
        self.assertEqual(int(np.count_nonzero(linear.mask)), 30000)
        same = MatchedSlotLinear(slot_count=30000, seed=73)
        self.assertTrue(np.array_equal(linear.mask, same.mask))
        self.assertNotIn("event_ids", linear.__dict__)
        self.assertNotIn("memory_text", linear.__dict__)
        self.assertTrue(np.all(linear.weights[~linear.mask] == 0))
        c = "rattling engines inside a laboratory"
        content = encode_bc_sensory("light blue notebook")
        self.assertGreater(linear.imprint(c, content), 0)
        self.assertTrue(np.all(linear.weights[~linear.mask] == 0))
        out = linear.infer(c)
        self.assertEqual(out.shape, (256,))
        self.assertTrue(np.all(np.isfinite(out)))
        with self.assertRaises(ValueError):
            MatchedSlotLinear(slot_count=65537)

    def test_predeclared_controls_and_parity_tol(self):
        graph = Topology.synthetic(n=4096, degree=64, seed=12)
        models = predeclared_models(graph, graph, group=4, n_eligible=4096)
        self.assertEqual(tuple(models), CONDITIONS)
        self.assertEqual(len(models), 7)
        self.assertEqual(models["original_beta1"].protection_beta, 1.0)
        self.assertEqual(models["original_beta4"].protection_beta, 4.0)
        self.assertEqual(models["matched_slot_linear"].slot_count, 4096)
        self.assertEqual(MAX_BASELINE_NUMERIC_TOLERANCE, 5e-7)


if __name__ == "__main__":
    unittest.main()
