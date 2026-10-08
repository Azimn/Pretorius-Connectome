"""Pilot 05A regression: train-only narrative index and post-hoc accountability."""
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
from pretorius_connectome.pilot04 import load_challenge
from pretorius_connectome.pilot05 import NarrativeModels, _summary, evaluate_seed, METHODS
from scripts.run_imprinting_pilot import EVENTS, SIDECARS, verify_sources, git_blob_sha
from scripts.run_imprinting_pilot04 import CHALLENGE, CHALLENGE_BLOB
from scripts.run_imprinting_pilot05 import _decisions, run


def small_memories():
    return [
        Memory("A", "E01",
               "I opened the brass box and removed the glass flask. "
               "My partner arranged the specimen by the window.",
               (Cue("surface:brass box", "brass box"),)),
        Memory("B", "E02",
               "She closed the wooden door and returned to the laboratory. "
               "I decided to keep the machine running.",
               (Cue("surface:wood door", "wood door"),)),
        Memory("C", "E03",
               "A copper bell sounded beside the second chamber. "
               "The doctor studied a violet shadow.",
               (Cue("surface:copper bell", "copper bell"),)),
    ]


class Pilot05Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        verify_sources(EVENTS, SIDECARS)
        cls.events = load_v12(EVENTS, SIDECARS)
        cls.cases = load_challenge(CHALLENGE, cls.events)
        cls.decisions = _decisions()

    def test_source_and_prior_diagnostic_set_locked(self):
        self.assertEqual(len(self.events), 450)
        self.assertEqual(len(self.decisions), 450)
        self.assertEqual(len(self.cases), 68)
        self.assertEqual(git_blob_sha(CHALLENGE), CHALLENGE_BLOB)

    def test_models_use_full_narrative_and_same_mask(self):
        items = small_memories()
        m = NarrativeModels(items, seed=37, cue_units=128,
                            memory_units=64, density=0.65)
        self.assertEqual(len(m.identifiers), len(items))
        self.assertEqual(set(m.overlays), {
            "hebb_word_narrative", "hebb_char_narrative",
            "hebb_word_shuffled", "hebb_word_unmodified"
        })
        for model in m.overlays.values():
            self.assertEqual(model.weights.shape, (64, 128))
            self.assertTrue(np.array_equal(model.mask, m.overlays["hebb_word_narrative"].mask))
            self.assertTrue(np.all(model.weights[~model.mask] == 0))
        self.assertFalse(np.any(m.overlays["hebb_word_unmodified"].weights))
        self.assertTrue(np.any(m.overlays["hebb_word_narrative"].weights))
        self.assertTrue(all(i != p for i, p in enumerate(m.shuffled_mapping)))

    def test_neural_identifies_training_narrative_from_itself(self):
        train = small_memories()
        model = NarrativeModels(train, seed=4, cue_units=128, memory_units=96, density=1)
        for item in train:
            scores = model.score_batch([item.memory_text], "hebb_word_narrative")
            self.assertEqual(len(scores), 1)
            self.assertIsNotNone(scores[0]["predicted"])
        blank = model.score_batch([train[0].memory_text], "hebb_word_unmodified")
        self.assertIsNone(blank[0]["predicted"])

    def test_retriever_can_cite_only_source_text(self):
        train = small_memories()
        model = NarrativeModels(train, seed=4, cue_units=128, memory_units=64,
                                density=0.7)
        for method in ("bm25_narrative", "tfidf_word_narrative",
                       "tfidf_char_narrative"):
            ranked = model.score_batch(["brass glass flask"], method)[0]
            self.assertEqual(ranked["predicted"], "A")
            quote = model.evidence(ranked["predicted"], "brass glass flask")
            self.assertIn(quote, train[0].memory_text)
        self.assertIsNone(model.evidence("missing", "glass"))

    def test_query_inputs_cannot_access_labels(self):
        items = small_memories()
        model = NarrativeModels(items, seed=7, cue_units=128, memory_units=64,
                                density=0.7)
        queries = ["I opened a brass box.", "The doctor walked outdoors."]
        for method in METHODS:
            s1 = model.score_batch(queries, method)
            s2 = model.score_batch(queries, method)
            self.assertEqual(s1, s2)
            self.assertTrue(all("target" not in r for r in s1))
        with self.assertRaises(ValueError):
            model.score_batch(["something"], "unsupported")

    def test_unknown_episode_grouping_and_counts(self):
        for seed in (31, 37, 43):
            train, validation, test = episode_split(self.events, seed)
            self.assertFalse({m.episode_id for m in train} &
                             {m.episode_id for m in test})
            self.assertTrue(all(e.event_id in self.decisions
                                for e in train + validation + test))

    def test_no_claim_of_truth_from_high_similarity(self):
        examples = [
            {"predicted": "A", "target": "A", "accepted": True},
            {"predicted": "B", "target": "A", "accepted": False},
        ]
        self.assertEqual(_summary(examples, "positive")["correct_and_accepted"], 0.5)
        self.assertEqual(_summary(examples, "contradiction")["false_target_confirmation"], 0.5)
        self.assertEqual(_summary(examples, "absent")["false_acceptance"], 0.5)

    def test_archive_smoke_on_one_seed(self):
        r = run(seeds=(31,), cue_units=128, memory_units=64, density=0.6)
        self.assertEqual(r["corpus_events"], 450)
        self.assertEqual(r["challenge_cases"], 68)
        self.assertIn("REUSED", r["challenge_status"])
        trial = r["trials"][0]
        self.assertEqual(len(trial["method_results"]), len(METHODS))
        self.assertEqual(trial["trained_events"] +
                         trial["validation_unknown_events"] +
                         trial["test_unknown_events"], 450)
        for result in trial["method_results"].values():
            self.assertTrue(0 <= result["summary"]["positive"]["top1"] <= 1)
            self.assertTrue(0 <= result["summary"]["contradiction"]["false_acceptance"] <= 1)
            self.assertTrue(0 <= result["summary"]["absent"]["false_acceptance"] <= 1)
        self.assertEqual(
            trial["method_results"]["hebb_word_unmodified"]["summary"]["positive"]["top1"], 0
        )


if __name__ == "__main__":
    unittest.main()
