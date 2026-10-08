"""Pilot 02 regression and leakage tests on frozen v12 source."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np

from pretorius_connectome.imprinting import Cue, Memory, SynapticOverlay, load_v12
from pretorius_connectome.pilot02 import (
    Probe, _oracle_scores, _positive_summary, _negative_summary,
    calibrate, episode_split, generalization_assay, interference_assay,
    lexical_probe, lexical_retrieval, train_conditions,
)
from scripts.run_imprinting_pilot import EVENTS, SIDECARS, verify_sources
from scripts.run_imprinting_pilot02 import run


def corpus():
    verify_sources(EVENTS, SIDECARS)
    return load_v12(EVENTS, SIDECARS)


class Pilot02Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.memories = corpus()

    def test_episode_splits_cover_archive_without_overlap(self):
        train, val, test = episode_split(self.memories, 0)
        self.assertEqual(len(train) + len(val) + len(test), 450)
        self.assertEqual(len(set(x.event_id for x in train + val + test)), 450)
        sets = [set(m.episode_id for m in group) for group in (train, val, test)]
        self.assertFalse(sets[0] & sets[1])
        self.assertFalse(sets[1] & sets[2])
        self.assertFalse(sets[0] & sets[2])
        self.assertEqual(len(sets[0] | sets[1] | sets[2]), 27)
        self.assertEqual(episode_split(self.memories, 0), episode_split(self.memories, 0))

    def test_perturbed_cues_are_not_trained_literal_ids(self):
        multi = next(m for m in self.memories if lexical_probe(m) is not None)
        for mode in ("swap", "drop"):
            p = lexical_probe(multi, mode)
            self.assertIsNotNone(p)
            original = {c.cue_id for c in multi.cues}
            self.assertNotIn(p.cue.cue_id, original)
            self.assertNotIn(multi.event_id, p.cue.cue_id)
            self.assertNotEqual(p.cue.surface, next(
                c.surface for c in multi.cues if c.surface and len(c.surface.split()) > 1
            ).casefold())
            self.assertEqual(p, lexical_probe(multi, mode))
        with self.assertRaises(ValueError):
            lexical_probe(multi, "undefined")

    def test_no_event_codebook_stored_in_synapses(self):
        m = Memory("item1", "e1", "I repaired the copper bell.",
                   (Cue("surface:copper bell", "copper bell"), Cue("cue:bell", "")))
        n = Memory("item2", "e2", "I replaced the glass.",
                   (Cue("surface:glass jar", "glass jar"), Cue("cue:glass", "")))
        trained = train_conditions([m, n], 9, 128, 64, 0.8)
        blank = trained["unmodified"]
        self.assertFalse(np.any(blank.weights))
        self.assertTrue(np.any(trained["full"].weights))
        self.assertTrue(np.all(trained["full"].weights[~trained["full"].mask] == 0))
        self.assertTrue(np.array_equal(trained["full"].mask, trained["ids_only"].mask))
        p = lexical_probe(m)
        self.assertIsNotNone(p)
        scores = _oracle_scores(trained["full"], [m, n], [p])
        self.assertEqual(len(scores), 1)
        self.assertTrue(scores[0]["known"])
        self.assertIsInstance(scores[0]["top_cosine"], float)
        self.assertNotIn("target_cosine", trained["full"].__dict__)

    def test_calibration_never_uses_test_episode_and_rejects_untrained(self):
        known = [
            {"known": True, "top_cosine": 0.9},
            {"known": True, "top_cosine": 0.7},
        ]
        unknown = [
            {"known": False, "top_cosine": 0.2},
            {"known": False, "top_cosine": 0.3},
        ]
        threshold = calibrate(known, unknown)
        self.assertGreater(threshold, 0.3)
        self.assertLess(threshold, 0.7)
        with self.assertRaises(ValueError):
            calibrate([], unknown)
        with self.assertRaises(ValueError):
            calibrate(unknown, known)

    def test_fixed_candidate_margins_and_abstention(self):
        a = Memory("A", "E1", "I wrote an entry about the wet glove.",
                   (Cue("surface:wet glove", "wet glove"),))
        b = Memory("B", "E1", "I poured a bottle of iron solution.",
                   (Cue("surface:iron solution", "iron solution"),))
        unseen = Memory("C", "E2", "I moved a chair across the room.",
                        (Cue("surface:wood chair", "wood chair"),))
        model = SynapticOverlay(128, 64, 1, 0)
        model.imprint(a.cues, a.memory_text)
        model.imprint(b.cues, b.memory_text)
        positive = _oracle_scores(model, [a, b], [lexical_probe(a)])
        negative = _oracle_scores(model, [a, b], [lexical_probe(unseen)])
        self.assertTrue(positive[0]["known"])
        self.assertFalse(negative[0]["known"])
        self.assertIsNone(negative[0]["target_cosine"])
        self.assertIsNone(negative[0]["correct"])
        self.assertEqual(_positive_summary(positive)["n"], 1)
        self.assertEqual(_negative_summary(negative, 0.1)["n"], 1)
        self.assertEqual(lexical_retrieval([a, b], [lexical_probe(a)])["n"], 1)

    def test_real_archive_smoke_end_to_end(self):
        record = run(seeds=(7,), cue_units=64, memory_units=64,
                     density=0.7, include_interference=False)
        self.assertEqual(record["events"], 450)
        self.assertEqual(len(record["generalization"]), 1)
        row = record["generalization"][0]
        self.assertTrue(row["counts"]["train"] > 100)
        self.assertTrue(row["counts"]["test_unknown_probe"] > 0)
        self.assertEqual(set(row["conditions"]),
                         {"full", "surface_only", "ids_only", "shuffled", "unmodified"})
        for result in row["conditions"].values():
            self.assertTrue(0 <= result["swap_balanced_accuracy"] <= 1)
            self.assertTrue(0 <= result["swap_withheld_test"]["false_acceptance_rate"] <= 1)
        self.assertEqual(
            row["conditions"]["unmodified"]["swap_known_test"]["top1_no_abstention"], 0.0
        )

    def test_short_interference_curve_reports_cosine_and_margin(self):
        scores = interference_assay(self.memories, seed=7,
                                    loads=(50, 100), cue_units=64, memory_units=64)
        self.assertEqual([x["load"] for x in scores], [50, 100])
        self.assertTrue(all(x["n"] == 50 for x in scores))
        self.assertTrue(all(x["mean_margin"] is not None for x in scores))


if __name__ == "__main__":
    unittest.main()
