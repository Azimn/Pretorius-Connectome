"""Pilot09 diagnostic must distinguish trained input and independent literal cue."""
from __future__ import annotations

import unittest

import numpy as np

from pretorius_connectome.associative import Topology
from pretorius_connectome.direct_flywire_imprint import DirectFlywireOverlay, select_features
from pretorius_connectome.imprinting import Cue, Memory
from pretorius_connectome.shared_features_bc01 import encode_bc_sensory
from scripts.diagnose_flywire_imprint09 import case_vectors, summarize, select_probe_ids


def item(event_id="E1-001"):
    return Memory(
        event_id, "E1", "the glass observatory and my impossible machine",
        (Cue("c1", "glass observatory"), Cue("c2", "midnight machine")),
    )


class CueTransferDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.topology = Topology.synthetic(n=8192, degree=64, seed=23)

    def test_same_model_distinct_train_and_last_cue(self):
        model = DirectFlywireOverlay(self.topology, seed=31, cells_per_feature=8)
        event = item()
        model.imprint("glass observatory", encode_bc_sensory(event.memory_text))
        oracle = select_features(encode_bc_sensory(event.memory_text), top_k=32)
        expected = oracle[np.newaxis, :]
        trained = case_vectors(model, [event], [event.event_id], expected, "trained_cue")
        withheld = case_vectors(model, [event], [event.event_id], expected, "withheld_last_cue")
        self.assertEqual(len(trained), len(withheld))
        self.assertEqual(trained[0]["event_id"], withheld[0]["event_id"])
        self.assertIn("cue_feature_overlap_count", trained[0])
        self.assertIn("cue_feature_cosine", withheld[0])
        self.assertNotEqual(model._input("glass observatory").tolist(),
                            model._input("midnight machine").tolist())
        self.assertEqual(summarize(trained, 0.5)["count"], 1)
        self.assertEqual(summarize(withheld, 0.5)["count"], 1)

    def test_absent_cases_are_not_scored_as_known(self):
        model = DirectFlywireOverlay(self.topology, seed=31, cells_per_feature=8)
        target = item()
        new = item("E2-005")
        new = Memory(new.event_id, "E2", new.memory_text, new.cues)
        oracle = select_features(encode_bc_sensory(target.memory_text), top_k=32)
        scores = case_vectors(model, [new], [target.event_id],
                              oracle[np.newaxis, :], "absent_episode_last_cue")
        self.assertIsNone(scores[0]["correct"])
        self.assertIsNone(scores[0]["target_cosine"])
        self.assertEqual(summarize(scores, 0.0)["fraction_accepted"], 0.0)

    def test_reused_probe_selection_independent_of_oracle(self):
        # This test passes no narrative content into ID-based seed selection.
        memories = [Memory("X" + str(i), "E1", "narrative " + str(i),
                           (Cue("a", "first cue"), Cue("b", "last cue")))
                    for i in range(11)]
        a, b = select_probe_ids(memories, 31)
        x, y = select_probe_ids(memories, 31)
        self.assertEqual([m.event_id for m in a], [m.event_id for m in x])
        self.assertEqual([m.event_id for m in b], [m.event_id for m in y])


if __name__ == "__main__":
    unittest.main()
