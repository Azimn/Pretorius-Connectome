"""Pilot10 enforces content-linked multi-cue presentations and clean controls."""
from __future__ import annotations

import unittest

import numpy as np

from pretorius_connectome.associative import Topology
from pretorius_connectome.direct_flywire_imprint import DirectFlywireOverlay, fingerprint
from pretorius_connectome.imprinting import Cue, Memory
from scripts.run_direct_flywire_imprint10 import (
    binding_cues, cue_drop_last_token, score, summarize, PROBES, CONDITIONS,
)


def memory(cues=("green tape", "small box", "blue light"), ident="E01-001"):
    return Memory(ident, "E01", "blue light and green tape by the window",
                  tuple(Cue("candidate-" + str(i), cue)
                        for i, cue in enumerate(cues)))


class RealEdgeMultiCueTests(unittest.TestCase):
    def test_source_only_cue_selection(self):
        self.assertEqual(binding_cues(memory()),
                         ("green tape", "small box", "blue light"))
        self.assertEqual(binding_cues(memory(("yellow tape", "blue light"))),
                         ("yellow tape", "blue light", "blue light"))
        self.assertEqual(binding_cues(memory(("a", "b", "c", "d"))),
                         ("a", "c", "d"))
        with self.assertRaises(ValueError):
            binding_cues(memory(("one",)))

    def test_new_lexical_probe_is_declared_not_semantic(self):
        self.assertEqual(cue_drop_last_token("blue light"), "blue")
        self.assertEqual(cue_drop_last_token("glass"), "glas")
        self.assertEqual(cue_drop_last_token("a"), "ax")

    def test_each_training_cue_updates_only_original_directed_connections(self):
        graph = Topology.synthetic(n=8192, degree=24, seed=19)
        orig = fingerprint(graph)
        model = DirectFlywireOverlay(
            graph, seed=31, cells_per_feature=8, rate=0.7 / 3
        )
        source = memory()
        from pretorius_connectome.shared_features_bc01 import encode_bc_sensory
        content = encode_bc_sensory(source.memory_text)
        before = model.modified_edges
        for cue in binding_cues(source):
            model.imprint(cue, content)
        self.assertEqual(model.imprints, 3)
        self.assertGreater(model.modified_edges, before)
        self.assertEqual(fingerprint(graph), orig)
        self.assertEqual(model.infer("blue light").shape, (256,))
        self.assertFalse(any(key in model.__dict__
                             for key in ("event_ids", "memory_texts",
                                         "oracle_targets", "retrieval_index")))

    def test_offline_identity_scoring_is_separate_from_neural_readout(self):
        from pretorius_connectome.shared_features_bc01 import encode_bc_sensory
        from pretorius_connectome.direct_flywire_imprint import select_features
        graph = Topology.synthetic(n=8192, degree=24, seed=20)
        model = DirectFlywireOverlay(graph, seed=31, cells_per_feature=8)
        source = memory()
        value = select_features(encode_bc_sensory(source.memory_text), top_k=32)
        for condition in PROBES[:3]:
            results = score(model, [source], [source.event_id],
                            value[np.newaxis, :], scenario=condition)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["event_id"], source.event_id)
            self.assertTrue(results[0]["known"])
            self.assertIsNone(results[0]["predicted"])
            self.assertEqual(summarize(results)["correct_top1"], 0)
        absent = memory(ident="E99-001")
        values = score(model, [absent], [source.event_id],
                       value[np.newaxis, :], scenario=PROBES[-1])
        self.assertFalse(values[0]["known"])
        self.assertIsNone(values[0]["correct"])
        self.assertEqual(summarize(values)["absent_false_acceptance"], 0)
        self.assertEqual(len(CONDITIONS), 4)


if __name__ == "__main__":
    unittest.main()
