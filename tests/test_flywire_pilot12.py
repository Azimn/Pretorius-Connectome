"""Pilot12 deterministic semantic projection, real-edge/null rewiring safeguards."""
from __future__ import annotations

from collections import Counter
import unittest

import numpy as np

from pretorius_connectome.associative import Topology
from pretorius_connectome.direct_flywire_imprint import (
    fingerprint, select_features,
)
from pretorius_connectome.rewire12 import rewire_effective_edges
from pretorius_connectome.semantic_cue12 import (
    CueOnlyOverlay, project_sentence_vectors, projection_sha256,
)
from pretorius_connectome.shared_features_bc01 import encode_bc_sensory
from scripts.run_flywire_pilot12 import (
    cue_for, measured_rows, summary, MODEL_IDS,
)
from pretorius_connectome.imprinting import Cue, Memory


def event(ident="E01-001"):
    return Memory(
        ident, "E01", "laboratory glass and luminous blue ribbon",
        tuple(Cue("cue" + str(i), surface) for i, surface in enumerate(
            ("blue ribbon", "a pale laboratory", "moonlit equipment", "glass tools")
        )),
    )


class Pilot12Tests(unittest.TestCase):
    def test_projection_is_statically_pinned_and_does_not_use_event_labels(self):
        sample = np.random.default_rng(2).normal(size=(3, 384)).astype(np.float32)
        first = project_sentence_vectors(sample)
        second = project_sentence_vectors(sample.copy())
        self.assertEqual(first.shape, (3, 256))
        self.assertTrue(np.array_equal(first, second))
        self.assertTrue(np.allclose(np.linalg.norm(first, axis=1), 1.0))
        self.assertEqual(len(projection_sha256()), 64)
        with self.assertRaises(ValueError):
            project_sentence_vectors(np.zeros((2, 256)))

    def test_rewire_preserves_every_eligible_source_and_target_binary_degree(self):
        topo = Topology.synthetic(n=4096, degree=32, seed=2)
        original_sha = fingerprint(topo)
        model = CueOnlyOverlay(topo, seed=31, cells_per_feature=8)
        changed, report = rewire_effective_edges(
            topo, model.pre_cells, model.post_cells,
            seed=73, swaps_per_edge=2,
        )
        self.assertTrue(report["binary_outdegrees_equal"])
        self.assertTrue(report["binary_indegrees_equal"])
        self.assertTrue(report["source_graph_unchanged"])
        self.assertEqual(report["originally_eligible_edges"],
                         report["rewired_eligible_edges"])
        self.assertGreater(report["accepted_double_edge_swaps"], 0)
        self.assertGreater(report["changed_edge_destinations"], 0)
        self.assertEqual(report["same_original_total_integer_synapses"], True)
        self.assertEqual(fingerprint(topo), original_sha)
        self.assertEqual(len(changed.indices), len(topo.indices))
        self.assertTrue(np.array_equal(changed.synapse_counts, topo.synapse_counts))
        self.assertFalse(np.array_equal(changed.indices, topo.indices))
        # Everything outside eligible directed learnable connections is fixed.
        active = np.zeros(len(topo.root_ids), dtype=bool)
        active[model.post_cells.ravel()] = True
        pre_ids = model.pre_cells.ravel()
        for pre in pre_ids[:30]:
            start, end = map(int, topo.indptr[pre:pre+2])
            permitted = active[topo.indices[start:end]]
            self.assertTrue(np.array_equal(
                changed.indices[start:end][~permitted],
                topo.indices[start:end][~permitted],
            ))
        twin, info = rewire_effective_edges(
            topo, model.pre_cells, model.post_cells, seed=73,
        )
        self.assertTrue(np.array_equal(changed.indices, twin.indices))
        self.assertEqual(info["rewired_csr_indices_sha256"],
                         report["rewired_csr_indices_sha256"])

    def test_swapped_edge_never_creates_duplicate_aggregated_directed_pair(self):
        topo = Topology.synthetic(n=2048, degree=64, seed=9)
        model = CueOnlyOverlay(topo, seed=31, cells_per_feature=4)
        rewired, report = rewire_effective_edges(
            topo, model.pre_cells, model.post_cells)
        for pre in model.pre_cells.ravel():
            start, end = map(int, rewired.indptr[pre:pre+2])
            neighbors = rewired.indices[start:end]
            self.assertEqual(len(neighbors), len(np.unique(neighbors)))
        self.assertTrue(report["same_number_of_synaptic_slots"])

    def test_external_cue_encoder_does_not_hold_event_ids(self):
        topo = Topology.synthetic(n=2048, degree=24, seed=11)
        class FakeFrozenSemantic:
            def __init__(self):
                self.calls = 0
            def __call__(self, phrase):
                self.calls += 1
                return encode_bc_sensory("shift " + phrase)
        encoder = FakeFrozenSemantic()
        m = CueOnlyOverlay(topo, seed=31, cells_per_feature=4,
                           cue_encoder=encoder)
        ref = CueOnlyOverlay(topo, seed=31, cells_per_feature=4)
        source = event()
        content = encode_bc_sensory(source.memory_text)
        for cue in ("blue ribbon", "a pale laboratory", "glass tools"):
            m.imprint(cue, content)
            ref.imprint(cue, content)
        self.assertEqual(m.imprints, ref.imprints)
        self.assertTrue(m.modified_edges > 0)
        self.assertTrue(encoder.calls >= 3)
        self.assertNotIn("event_ids", m.__dict__)
        self.assertNotIn("target_vectors", m.__dict__)
        self.assertEqual(m.infer("glass tools").shape, (256,))
        self.assertNotIn("event_ids", encoder.__dict__)

    def test_unseen_source_is_not_a_trained_cue(self):
        sample = event()
        self.assertEqual(
            cue_for("genuinely_untrained_fourth_source_cue", sample),
            "a pale laboratory",
        )
        self.assertEqual(
            cue_for("familiar_trained_last", sample),
            "glass tools",
        )
        self.assertNotEqual(
            cue_for("controlled_last_cue_token_deletion", sample),
            "glass tools",
        )
        self.assertEqual(len(MODEL_IDS), 6)

    def test_oracle_scorer_does_not_affect_neural_inference(self):
        topo = Topology.synthetic(n=2048, degree=32, seed=13)
        model = CueOnlyOverlay(topo, seed=31, cells_per_feature=4)
        sample = event()
        reference = select_features(
            encode_bc_sensory(sample.memory_text), top_k=32
        )[np.newaxis, :]
        evidence = measured_rows(model, [sample], [sample.event_id],
                                 reference, "genuinely_untrained_fourth_source_cue")
        self.assertEqual(evidence[0]["event_id"], sample.event_id)
        self.assertIsNone(evidence[0]["predicted"])
        self.assertEqual(evidence[0]["same_encoder_trained_input_overlap"] >= 0, True)
        self.assertEqual(summary(evidence, .5)["correct_top1"], 0)


if __name__ == "__main__":
    unittest.main()
