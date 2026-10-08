"""Pilot 06A regression for source-only evidence gate, no challenge label leaks."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np

from pretorius_connectome.imprinting import Cue, Memory, load_v12
from pretorius_connectome.pilot04 import load_challenge
from pretorius_connectome.pilot06 import (
    METHODS, EvidenceIndex, EvidenceMatch, _verification, _make_rows,
    polarity, significant_tokens, evaluate_seed,
)
from pretorius_connectome.pilot05 import NarrativeModels
from scripts.run_imprinting_pilot import EVENTS, SIDECARS, git_blob_sha, verify_sources
from scripts.run_imprinting_pilot04 import CHALLENGE, CHALLENGE_BLOB
from scripts.run_imprinting_pilot05 import _decisions
from scripts.run_imprinting_pilot06 import run


def three():
    return [
        Memory("A", "E01",
               "I preserved the blue glass vessel in the cabinet. "
               "The laboratory keeper inspected the intact sample.",
               (Cue("a", "blue vessel"),)),
        Memory("B", "E02",
               "Marta returned the wooden box and locked the workshop door. "
               "The young clerk checked the delivery note.",
               (Cue("b", "wooden box"),)),
        Memory("C", "E03",
               "Clara washed the silver instrument beside the window. "
               "I documented the copper railing.",
               (Cue("c", "silver instrument"),)),
    ]


class Pilot06Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        verify_sources(EVENTS, SIDECARS)
        cls.memories = load_v12(EVENTS, SIDECARS)
        cls.cases = load_challenge(CHALLENGE, cls.memories)

    def test_provenance_is_frozen(self):
        self.assertEqual(len(self.memories), 450)
        self.assertEqual(len(self.cases), 68)
        self.assertEqual(git_blob_sha(CHALLENGE), CHALLENGE_BLOB)

    def test_evidence_quotes_are_exact_source_spans(self):
        memories = three()
        idx = EvidenceIndex(memories)
        self.assertEqual(len(idx.flat_sentences), 6)
        match = idx.match("preserved the blue glass vessel", "A")
        self.assertEqual(match.event_id, "A")
        self.assertIn(match.quote, memories[0].memory_text)
        self.assertIn(match.sentence_index, (0, 1))
        self.assertGreater(match.score, 0)
        self.assertGreaterEqual(match.lexical_anchors, 2)
        self.assertIsNone(idx.match("not found", "not-existing").quote)
        self.assertEqual(idx.match("not found", None).score, 0)

    def test_only_narrow_explicit_negation_is_detected(self):
        self.assertFalse(polarity("I preserved the blue glass vessel"))
        self.assertTrue(polarity("I never preserved the blue glass vessel"))
        self.assertTrue(polarity("The witness did not move"))
        self.assertFalse(polarity("The witness remained still"))
        self.assertNotIn("not", significant_tokens("not preserved the blue glass vessel"))

    def test_negative_flip_blocked_after_evidence_is_present(self):
        idx = EvidenceIndex(three())
        actual = idx.match("I never preserved the blue glass vessel", "A")
        self.assertTrue(actual.negation_mismatch)
        self.assertGreaterEqual(actual.lexical_anchors, 2)
        self.assertEqual(_verification("bm25_sentence_polarity", actual, -0.1),
                         "possible_contradiction")
        self.assertEqual(_verification("bm25_sentence_only", actual, -0.1),
                         "lexically_supported")
        self.assertEqual(_verification("bm25_sentence_polarity", actual, 0.99),
                         "insufficient_evidence")
        self.assertEqual(_verification("reject_all", actual, -0.1),
                         "insufficient_evidence")

    def test_verdict_does_not_depend_on_case_label_or_event_target(self):
        memories = three()
        model = NarrativeModels(memories, seed=19, cue_units=96,
                                memory_units=64, density=0.7)
        evidence = EvidenceIndex(memories)
        from pretorius_connectome.pilot04 import ChallengeCase
        query = "I preserved the blue glass vessel"
        true = ChallengeCase("CASEP", "A", "paraphrase", query, "some source")
        false = ChallengeCase("CASEN", "C", "contradiction", query, "other source")
        r1 = _make_rows(model, evidence, "bm25_sentence_polarity", [true], -1, -1)[0]
        r2 = _make_rows(model, evidence, "bm25_sentence_polarity", [false], -1, -1)[0]
        for field in ("predicted", "accepted", "verdict", "evidence_quote",
                      "evidence_score", "lexical_anchors"):
            self.assertEqual(r1[field], r2[field], field)
        self.assertEqual(r1["target"], "A")
        self.assertEqual(r2["target"], "C")

    def test_full_archive_one_seed_smoke(self):
        output = run(seeds=(31,), cue_units=96, memory_units=64, density=0.65)
        self.assertEqual(output["record_count"], 450)
        self.assertEqual(output["challenge_count"], 68)
        self.assertIn("NOT independent validation", output["challenge_status"])
        trial = output["trials"][0]
        self.assertEqual(len(trial["conditions"]), len(METHODS))
        self.assertEqual(trial["train_events"] +
                         trial["calibration_unknown_events"] +
                         trial["test_unknown_events"], 450)
        for method, result in trial["conditions"].items():
            self.assertEqual(result["summary"]["positive"]["n"],
                             trial["trained_positive_cases"])
            self.assertEqual(result["summary"]["contradiction"]["n"],
                             trial["trained_counterfactual_cases"])
            self.assertTrue(0 <= result["summary"]["positive"]["accepted"] <= 1)
            self.assertTrue(0 <= result["summary"]["absent"]["false_acceptance"] <= 1)
            for section in ("positive", "contradiction", "absent"):
                for row in result["rows"][section]:
                    if row["evidence_quote"] is not None:
                        self.assertIn(row["evidence_event_id"], {m.event_id for m in self.memories})
                        self.assertLessEqual(len(row["evidence_quote"]), 300)
        self.assertEqual(
            trial["conditions"]["reject_all"]["summary"]["positive"]["correct_and_accepted"],
            0.0
        )


if __name__ == "__main__":
    unittest.main()
