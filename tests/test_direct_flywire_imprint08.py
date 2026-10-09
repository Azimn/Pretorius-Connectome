"""Pilot 08: real-edge constrained direct imprint has no external retriever."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from pretorius_connectome.associative import Topology
from pretorius_connectome.direct_flywire_imprint import (
    DirectFlywireOverlay, fingerprint, select_features,
)
from pretorius_connectome.shared_features_bc01 import encode_bc_sensory


class DirectFlywireSynapticImprintTests(unittest.TestCase):
    def setUp(self):
        self.topology = Topology.synthetic(n=2048, degree=16, seed=21)
        self.model = DirectFlywireOverlay(
            self.topology, seed=31, cells_per_feature=4
        )

    def test_signed_imprint_uses_real_existing_directed_edges_only(self):
        before = fingerprint(self.topology)
        self.assertEqual(self.model.modified_edges, 0)
        self.assertFalse(np.any(self.model.infer("green ribbon laboratory")))
        count = self.model.imprint(
            "green ribbon laboratory",
            encode_bc_sensory("glass instruments and saltwater"),
        )
        self.assertGreater(count, 0)
        self.assertGreater(self.model.modified_edges, 0)
        self.assertGreater(float(np.linalg.norm(
            self.model.infer("green ribbon laboratory")
        )), 0)
        self.assertEqual(before, fingerprint(self.topology))
        self.model.assert_original_unchanged()
        self.assertEqual(self.model.imprints, 1)

    def test_no_source_texts_ids_or_oracle_codebook_at_inference(self):
        for name in ("memories", "records", "docs", "event_ids", "content_targets",
                     "codebook", "retriever", "texts", "narratives"):
            self.assertNotIn(name, self.model.__dict__)
        with self.assertRaises(ValueError):
            self.model.infer("")
        with self.assertRaises(ValueError):
            self.model.imprint("legitimate cue", np.ones(12))
        self.assertEqual(self.model.infer("untrained cue").shape, (256,))

    def test_train_and_reload_only_synaptic_state(self):
        self.model.imprint(
            "the blue room", encode_bc_sensory("bridge and moonlight")
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "actual_synaptic_overlay.npz"
            digest = self.model.save(path)
            restored = DirectFlywireOverlay.load(self.topology, path)
            self.assertEqual(len(digest), 64)
            self.assertTrue(np.array_equal(restored.delta, self.model.delta))
            self.assertTrue(np.array_equal(
                restored.infer("the blue room"), self.model.infer("the blue room")
            ))
            self.assertEqual(restored.imprints, 1)
            with np.load(path, allow_pickle=False) as ckpt:
                self.assertEqual(
                    set(ckpt.files),
                    {"edge_positions", "learned_deltas", "anatomical_array_sha256",
                     "settings", "rate", "weight_cap"},
                )

    def test_checkpoint_rejects_changed_biological_anatomy(self):
        self.model.imprint("silver ribbon", encode_bc_sensory("old machine"))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "weights.npz"
            self.model.save(path)
            new = Topology(
                self.topology.root_ids.copy(), self.topology.indptr.copy(),
                self.topology.indices.copy(), self.topology.synapse_counts.copy(),
                "synthetic modified contact count",
            )
            new.synapse_counts[0] += 1
            with self.assertRaisesRegex(ValueError, "different original CSR"):
                DirectFlywireOverlay.load(new, path)

    def test_deterministic_mapping_and_feature_sparsification(self):
        twin = DirectFlywireOverlay(self.topology, seed=31, cells_per_feature=4)
        self.assertTrue(np.array_equal(self.model.pre_cells, twin.pre_cells))
        self.assertTrue(np.array_equal(self.model.post_cells, twin.post_cells))
        self.assertFalse(np.intersect1d(
            self.model.pre_cells.ravel(), self.model.post_cells.ravel()
        ).size)
        vec = select_features(encode_bc_sensory("sample glass paper"), top_k=8)
        self.assertLessEqual(np.count_nonzero(vec), 8)
        self.assertAlmostEqual(float(np.linalg.norm(vec)), 1.0, places=5)

    def test_corrupt_original_arrays_detected_after_learning(self):
        self.model.imprint("blue room", encode_bc_sensory("hollow bells"))
        self.topology.indices[0] = (self.topology.indices[0] + 1) % len(
            self.topology.root_ids
        )
        with self.assertRaisesRegex(AssertionError, "mutated"):
            self.model.assert_original_unchanged()

    def test_zero_overlay_and_shuffled_content_are_distinct_conditions(self):
        cue = "millstream and a window"
        a = DirectFlywireOverlay(self.topology, seed=3, cells_per_feature=4)
        b = DirectFlywireOverlay(self.topology, seed=3, cells_per_feature=4)
        baseline = DirectFlywireOverlay(self.topology, seed=3, cells_per_feature=4)
        a.imprint(cue, encode_bc_sensory("gothic chapel"))
        b.imprint(cue, encode_bc_sensory("brass observatory"))
        self.assertEqual(a.modified_edges > 0, True)
        self.assertFalse(np.array_equal(a.delta, b.delta))
        self.assertEqual(baseline.modified_edges, 0)
        self.assertFalse(np.any(baseline.infer(cue)))


if __name__ == "__main__":
    unittest.main()
