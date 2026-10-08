"""Pilot 03 leakage and regression tests on frozen 450-memory v12."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from pretorius_connectome.imprinting import Cue, Memory, load_v12
from pretorius_connectome.pilot02 import episode_split
from pretorius_connectome.pilot03 import (
    CueEncoder, EncoderOverlay, _features, _fit_conditions, _auc,
    evaluate_seed, probes, scores, typo_probe,
)
from scripts.run_imprinting_pilot import EVENTS, SIDECARS, verify_sources
from scripts.run_imprinting_pilot03 import run


def tiny():
    return [
        Memory("A", "E1", "My glass vial cracked beneath the shutter.",
               (Cue("surface:glass vial", "glass vial"), Cue("cue:vial", ""))),
        Memory("B", "E2", "The brass lever made a scraping sound.",
               (Cue("surface:brass lever", "brass lever"), Cue("cue:lever", ""))),
        Memory("C", "E3", "I studied the bent copper pipe.",
               (Cue("surface:copper pipe", "copper pipe"), Cue("cue:pipe", ""))),
    ]


class Pilot03Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        verify_sources(EVENTS, SIDECARS)
        cls.archive = load_v12(EVENTS, SIDECARS)

    def test_token_encoder_ignores_ids_and_word_order(self):
        fit = tiny()
        model = CueEncoder("token", 128).fit(fit)
        x = model.encode((Cue("surface:brass lever", "brass lever"),))
        y = model.encode((Cue("probe:unknown", "lever brass"),))
        self.assertTrue(np.array_equal(x, y))
        x2 = model.encode((Cue("totally-different-id", "brass lever"),))
        self.assertTrue(np.array_equal(x, x2))
        legacy = CueEncoder("legacy_surface", 128).fit(fit)
        self.assertFalse(np.array_equal(
            legacy.encode((Cue("surface:brass lever", "brass lever"),)),
            legacy.encode((Cue("probe:unknown", "lever brass"),))
        ))

    def test_char_grams_tolerate_single_character_change(self):
        fitted = tiny()
        model = CueEncoder("char3", 512).fit(fitted)
        original = model.encode((Cue("a", "copper pipe"),))
        typo = model.encode((Cue("x", "copxer pipe"),))
        unrelated = model.encode((Cue("z", "violet cradle"),))
        self.assertGreater(float(original @ typo), float(original @ unrelated))
        self.assertFalse(np.array_equal(original, typo))
        self.assertGreater(len(_features("copper pipe", "char3")), 2)

    def test_idf_uses_only_training_events(self):
        train, validation, test = episode_split(self.archive, 31)
        a = CueEncoder("idf_char3", 96).fit(train)
        modified_holdout = [
            Memory(m.event_id, m.episode_id,
                   "SENTINEL NEVER FIT INTO TRAINING",
                   (Cue("new:sentinel", "zzzzsentineljqxx"),))
            for m in validation + test
        ]
        b = CueEncoder("idf_char3", 96).fit(list(train))
        self.assertEqual(a.idf, b.idf)
        self.assertEqual(a.fitted_events, len(train))
        self.assertTrue(modified_holdout)
        self.assertFalse(any("sentinel" in key for key in a.idf))
        with self.assertRaises(ValueError):
            CueEncoder("not-real", 64)

    def test_mask_and_parameter_budget_identical(self):
        models = _fit_conditions(tiny(), 7, 128, 64, 0.6)
        reference = models["token"].mask
        for model in models.values():
            self.assertTrue(np.array_equal(model.mask, reference))
            self.assertEqual(model.weights.shape, (64, 128))
            self.assertTrue(np.all(model.weights[~model.mask] == 0))
            self.assertEqual(model.encoder.fitted_events, 3)
        self.assertFalse(np.any(models["token_unmodified"].weights))
        self.assertTrue(np.any(models["char3"].weights))
        with self.assertRaises(ValueError):
            CueEncoder("char3", 128).encode((Cue("unknown", "brass"),))

    def test_typos_are_new_and_label_free(self):
        m = tiny()[1]
        probe = typo_probe(m)
        self.assertIsNotNone(probe)
        self.assertNotEqual(probe.cue.surface, "brass lever")
        self.assertFalse(any(probe.cue.cue_id == c.cue_id for c in m.cues))
        self.assertNotIn("B", probe.cue.cue_id)
        self.assertEqual(typo_probe(m), probe)
        self.assertEqual(len(probes(tiny(), "typo")), 3)
        with self.assertRaises(ValueError):
            probes(tiny(), "semantic")

    def test_oracle_does_not_store_labels_inside_model(self):
        subset = tiny()
        model = EncoderOverlay(CueEncoder("token", 128).fit(subset), 64, 1, 7)
        model.imprint(subset[0])
        p = probes(subset[:1], "swap")[0]
        scored = scores(model, subset, [p])
        self.assertEqual(len(scored), 1)
        self.assertTrue(scored[0]["known"])
        self.assertNotIn("event_id", model.__dict__)
        unknown = scores(model, subset[:1], probes(subset[1:2], "swap"))
        self.assertFalse(unknown[0]["known"])
        self.assertIsNone(unknown[0]["target_cosine"])
        self.assertAlmostEqual(_auc(
            [{"top_cosine": 0.5}, {"top_cosine": 0.5}],
            [{"top_cosine": 0.5}]
        ), 0.5)

    def test_full_archive_seeded_smoke(self):
        result = run(seeds=(31,), cue_units=96, memory_units=64, density=0.7)
        self.assertEqual(result["source_record_count"], 450)
        self.assertEqual(result["source_episode_count"], 27)
        self.assertEqual(len(result["trials"]), 1)
        trial = result["trials"][0]
        self.assertEqual(sum(trial["count"][key] for key in (
            "train_events", "calibration_unknown_events", "test_unknown_events"
        )), 450)
        self.assertEqual(len(trial["methods"]), 7)
        for model in trial["methods"].values():
            self.assertEqual(model["topology_edges"], trial["methods"]["token"]["topology_edges"])
            self.assertEqual(model["train_events_used_for_encoder_fit"], trial["count"]["train_events"])
            for variant in ("swap", "drop", "typo"):
                score = model["tests"][variant]
                self.assertTrue(0 <= score["discrimination_auc"] <= 1)
                self.assertTrue(0 <= score["known"]["top1_no_abstention"] <= 1)
        self.assertEqual(trial["methods"]["token_unmodified"]["updates"], 0)


if __name__ == "__main__":
    unittest.main()
