"""Tests distinguish a working weight update from database lookup."""
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np

from pretorius_connectome.imprinting import (
    Cue, Memory, SynapticOverlay, content_fingerprint, cue_vector, decode,
    evaluate, load_v12, order_by_episode, retrieval_only, shuffled_targets,
)
from scripts.run_imprinting_pilot import EVENTS, SIDECARS, git_blob_sha, run, verify_sources


def fixture():
    return [
        Memory(f"X{i}", "E1" if i < 2 else "E2",
               f"Unique long narrative number {i}, with different decisions.",
               (Cue(f"cue:{i}", name),))
        for i, name in enumerate(("amber prism", "brass dial", "violet flask", "copper bell"))
    ]


class ImprintingPilotTests(unittest.TestCase):
    def test_v12_lock_alignment_and_total(self):
        verify_sources(EVENTS, SIDECARS)
        items = load_v12(EVENTS, SIDECARS)
        self.assertEqual(len(items), 450)
        self.assertEqual(len({m.event_id for m in items}), 450)
        self.assertEqual(len({m.episode_id for m in items}), 27)
        self.assertTrue(all(m.cues for m in items))

    def test_encoder_determinism_and_event_id_exclusion(self):
        one = (Cue("cue:one", "copper lamp"),)
        self.assertTrue(np.array_equal(cue_vector(one, 128), cue_vector(one, 128)))
        self.assertFalse(np.array_equal(
            content_fingerprint("I withdrew the apparatus.", 128),
            content_fingerprint("I repaired the apparatus.", 128),
        ))
        self.assertAlmostEqual(float(np.linalg.norm(cue_vector(one, 128))), 1.0, places=5)
        with self.assertRaises(ValueError):
            cue_vector((), 128)

    def test_fixed_edges_only_and_replay(self):
        a, b = SynapticOverlay(64, 32, 0.3, seed=7), SynapticOverlay(64, 32, 0.3, seed=7)
        self.assertTrue(np.array_equal(a.mask, b.mask))
        original = a.mask.copy()
        for machine in (a, b):
            machine.imprint((Cue("cue:lamp", "brass lamp"),), "the candle went out")
        self.assertTrue(np.array_equal(a.weights, b.weights))
        self.assertTrue(np.array_equal(a.mask, original))
        self.assertTrue(np.all(a.weights[~a.mask] == 0))
        self.assertTrue(np.any(a.weights))
        with self.assertRaises(ValueError):
            a.mask[0, 0] = not bool(a.mask[0, 0])
        a.reset_overlay()
        self.assertTrue(np.all(a.weights == 0))
        self.assertEqual(a.updates, 0)

    def test_imprinting_identifies_training_patterns(self):
        items = fixture()
        model = SynapticOverlay(512, 256, density=1, seed=3)
        blank = SynapticOverlay(512, 256, density=1, seed=3)
        self.assertIsNone(decode(blank.readout(items[0].cues), items, 256))
        self.assertEqual(evaluate(blank, items)["top1"], 0)
        for item in items:
            model.imprint(item.cues, item.memory_text)
        self.assertGreaterEqual(evaluate(model, items)["top1"], 0.75)
        self.assertEqual(retrieval_only(items, items), 1.0)
        self.assertTrue(np.array_equal(model.weights, model.weights.copy()))

    def test_order_and_mismatched_controls(self):
        items = fixture()
        self.assertEqual(order_by_episode(items, 14), order_by_episode(items, 14))
        self.assertEqual(len(order_by_episode(items, 14)), 4)
        wrong = shuffled_targets(items, 14)
        self.assertEqual(sorted(wrong), sorted(m.memory_text for m in items))
        self.assertNotEqual(wrong, [m.memory_text for m in items])

    def test_source_fingerprint_is_blob_sha(self):
        self.assertEqual(len(git_blob_sha(EVENTS)), 40)
        self.assertEqual(len(git_blob_sha(SIDECARS)), 40)

    def test_small_real_corpus_run(self):
        result = run(seeds=(23,), loads=(50,), cue_units=64, memory_units=64, density=0.7)
        self.assertEqual(result["corpus_events"], 450)
        self.assertEqual(len(result["results"]), 1)
        line = result["results"][0]
        self.assertEqual(line["load"], 50)
        self.assertEqual(line["unmodified"]["top1"], 0.0)
        self.assertEqual(line["learned"]["n_probes"], 50)
        self.assertGreater(line["episode_count"], 1)
        self.assertIn("fixed50_learned", line)


if __name__ == "__main__":
    unittest.main()
