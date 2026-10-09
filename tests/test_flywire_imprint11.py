"""Source-locked Pilot11 unseen-cue and calibration leakage safeguards."""
from __future__ import annotations

import unittest
import numpy as np

from pretorius_connectome.associative import Topology
from pretorius_connectome.direct_flywire_imprint import DirectFlywireOverlay
from pretorius_connectome.imprinting import Cue, Memory
from scripts.diagnose_flywire_imprint11 import (
    unseen_literal_source_cue, calibrated_gate, summarize,
    compare_source_case_rows, score_rows,
)


def item(words, eid="E01-001"):
    return Memory(eid, "E01", "I remembered a glass observatory.",
                  tuple(Cue("c" + str(j), word) for j, word in enumerate(words)))


def case(score, correct=True):
    return {"best_cosine": score, "correct": correct, "nonzero_model_response": True}


class Pilot11FrozenRecallTests(unittest.TestCase):
    def test_unseen_source_cue_never_was_trained(self):
        event = item(["green tape", "untrained brass", "blue box", "violet glass"])
        self.assertEqual(unseen_literal_source_cue(event), "untrained brass")
        self.assertIsNone(unseen_literal_source_cue(
            item(["first", "middle", "last"])))
        self.assertIsNone(unseen_literal_source_cue(
            item(["first", "first", "middle", "last"])))
        six = item(["a", "b", "c", "d", "e", "f"])
        self.assertEqual(unseen_literal_source_cue(six), "b")

    def test_gate_calibrates_only_on_validation_and_enforces_fpr_cap(self):
        p = [case(.70, True), case(.61, True), case(.33, True),
             case(.11, False)]
        n = [case(.15, False), case(.20, False), case(.40, False),
             case(.45, False), case(.52, False), case(.55, False),
             case(.58, False), case(.63, False), case(.67, False),
             case(.81, False)]
        gate = calibrated_gate(p, n, max_validation_fpr=0.10)
        self.assertLessEqual(gate["calibration_false_acceptances"], 1)
        self.assertEqual(gate["calibration_known_events"], 4)
        self.assertEqual(gate["calibration_absent_events"], 10)
        self.assertGreater(gate["threshold"], .50)
        with self.assertRaises(ValueError):
            calibrated_gate([], n)
        with self.assertRaises(ValueError):
            calibrated_gate(p, n, max_validation_fpr=1.0)

    def test_gate_tie_and_nonzero_calibration(self):
        gate = calibrated_gate([case(.1, False)], [case(.1, False)])
        self.assertEqual(gate["calibration_false_acceptances"], 0)
        self.assertGreater(gate["threshold"], .1)

    def test_summary_unknown_versus_known(self):
        known = [{
            "known_train_event": True, "nonzero_model_response": True,
            "best_cosine": .4, "correct": True, "target_cosine": .3,
            "unseen_shared_feature_count": 0,
        }]
        unknown = [{
            "known_train_event": False, "nonzero_model_response": True,
            "best_cosine": .4, "correct": None, "target_cosine": None,
        }]
        self.assertEqual(summarize(known, .3)["correct_top1"], 1)
        self.assertEqual(summarize(known, .3)["no_shared_source_feature_count"], 1)
        self.assertEqual(summarize(unknown, .3)["absent_false_acceptance"], 1)
        with self.assertRaises(ValueError):
            summarize(known + unknown, .3)

    def test_historical_replay_allows_roundoff_but_not_changed_identity(self):
        baseline = [{"event_id": "E01-002", "predicted": "E01-004",
                     "correct": False, "response_norm": 1.0,
                     "best_cosine": .4444444, "target_cosine": -.04}]
        now = [{"event_id": "E01-002", "predicted": "E01-004",
                "correct": False, "nonzero_model_response": True,
                "best_cosine": .4444445, "target_cosine": -.0400001}]
        compare_source_case_rows(now, baseline, condition="synthetic")
        now[0]["predicted"] = "E01-003"
        with self.assertRaises(AssertionError):
            compare_source_case_rows(now, baseline, condition="synthetic")

    def test_external_target_index_is_not_part_of_neural_substrate(self):
        topology = Topology.synthetic(n=2048, degree=32, seed=21)
        model = DirectFlywireOverlay(topology, seed=31, cells_per_feature=4)
        memory = item(["green tape", "untrained brass", "blue box", "violet glass"])
        targets = np.ones((1, 256), dtype=np.float32)
        out = score_rows(model, [memory], [memory.event_id], targets,
                         unseen_literal_source_cue, "unseen_fourth_cue")
        self.assertEqual(out[0]["event_id"], memory.event_id)
        self.assertIn("unseen_shared_feature_count", out[0])
        self.assertIsNone(out[0]["predicted"])
        self.assertNotIn("event_ids", model.__dict__)
        self.assertNotIn("oracle_targets", model.__dict__)


if __name__ == "__main__":
    unittest.main()
