"""Pilot07 regression tests: mocked NLI ONLY, never reported as model inference."""
from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from pretorius_connectome.imprinting import Cue, Memory, load_v12
from pretorius_connectome.pilot04 import ChallengeCase, load_challenge
from pretorius_connectome.pilot07 import (
    _policy, retrieve_top_k, triage_pairs, evaluate_seed, TOP_K,
    NLI_MODEL_ID, NLI_MODEL_REVISION,
)
from pretorius_connectome.pilot07_review import (
    build_packet, write_packet, validate_packet_submission, choose_events,
)
from pretorius_connectome.pilot05 import NarrativeModels
from scripts.run_imprinting_pilot import EVENTS, SIDECARS, verify_sources
from scripts.run_imprinting_pilot04 import CHALLENGE
from scripts.run_imprinting_pilot05 import _decisions


class FakeScorer:
    """SYNTHETIC TEST DOUBLE; NEVER a trained NLI model."""
    def score_pairs(self, premises, hypotheses):
        assert len(premises) == len(hypotheses)
        return np.array([
            [0.75, 0.10, 0.15] if "never " in claim.lower()
            else [0.05, 0.75, 0.20] if "glass" in claim.lower()
            else [0.15, 0.15, 0.70]
            for claim in hypotheses
        ], dtype=np.float32)


def demo_memories():
    return [
        Memory("A", "E01",
               "I inspected a glass vessel and preserved the old specimen. "
               "Clara arranged the silver instruments on the table.",
               (Cue("A", "glass vessel"),)),
        Memory("B", "E02",
               "I carried the paper box back to the old library. "
               "The professor spoke with the porter by the door.",
               (Cue("B", "paper box"),)),
        Memory("C", "E03",
               "Marta examined the wooden window and repaired a hinge. "
               "I recorded the effects of sunlight on copper.",
               (Cue("C", "wooden window"),)),
    ]


class Pilot07Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        verify_sources(EVENTS, SIDECARS)
        cls.memories = load_v12(EVENTS, SIDECARS)
        cls.cases = load_challenge(CHALLENGE, cls.memories)

    def test_pinned_checkpoint_is_full_revision(self):
        self.assertEqual(len(NLI_MODEL_REVISION), 40)
        self.assertEqual(NLI_MODEL_ID, "cross-encoder/nli-MiniLM2-L6-H768")

    def test_retrieval_only_train_narratives(self):
        model = NarrativeModels(demo_memories(), seed=13,
                                cue_units=128, memory_units=64)
        found = retrieve_top_k(model, "glass vessel preserved", k=2)
        self.assertEqual(found[0].event_id, "A")
        self.assertEqual(len(found), 2)
        self.assertTrue(all(c.story in [m.memory_text for m in demo_memories()]
                            for c in found))

    def test_score_pairing_not_using_gold_id_or_kind(self):
        model = NarrativeModels(demo_memories(), seed=13,
                                cue_units=128, memory_units=64)
        q = "I inspected a glass vessel"
        possibilities = [retrieve_top_k(model, q, 2)]
        first = triage_pairs(possibilities, [q], FakeScorer())[0]
        second = triage_pairs(possibilities, [q], FakeScorer())[0]
        self.assertEqual(first, second)
        self.assertEqual(first[0]["entailment"], 0.75)
        self.assertNotIn("target", str(first))
        with self.assertRaises(ValueError):
            triage_pairs(possibilities, [], FakeScorer())

    def test_policy_needs_entailment_not_just_confidence(self):
        entailed = [{"event_id": "A", "rank": 1,
                     "entailment": 0.76, "contradiction": 0.08,
                     "neutral": 0.16}]
        refuted = [{"event_id": "A", "rank": 1,
                    "entailment": 0.15, "contradiction": 0.76,
                    "neutral": 0.09}]
        neutral = [{"event_id": "A", "rank": 1,
                    "entailment": 0.19, "contradiction": 0.07,
                    "neutral": 0.74}]
        self.assertEqual(_policy(entailed, False),
                         ("A", True, "model_entailment"))
        self.assertEqual(_policy(refuted, False)[2],
                         "model_possible_contradiction")
        self.assertEqual(_policy(neutral, False)[2],
                         "insufficient_evidence")

    def test_review_packet_excludes_all_previous_target_ids(self):
        packet = build_packet(self.memories, self.cases, max_events=16)
        self.assertEqual(packet["selected_events"], 16)
        self.assertEqual(packet["draft_cases"], 64)
        prev = {c.event_id for c in self.cases}
        self.assertFalse(prev & {x["source_event_id"]
                                 for x in packet["source_events"]})
        self.assertEqual(packet, build_packet(self.memories, self.cases, 16))

    def test_review_template_is_not_a_valid_submission(self):
        packet = build_packet(self.memories, self.cases, max_events=4)
        with tempfile.TemporaryDirectory() as td:
            destination = Path(td)
            wrote = write_packet(packet, destination)
            self.assertEqual(wrote["cases"], 16)
            self.assertEqual(len(wrote["paths"]), 5)
            with self.assertRaises(ValueError):
                validate_packet_submission(destination)

    def test_archive_posthoc_smoke_fake_backend(self):
        # Fake scorer proves orchestration, NOT model performance.
        result = evaluate_seed(
            self.memories, _decisions(), self.cases, 31,
            FakeScorer(), top_k=2, cue_units=128,
        )
        self.assertEqual(result["train_events"] +
                         result["validation_unknown_events"] +
                         result["test_unknown_events"], 450)
        self.assertEqual(set(result["summaries"]), {
            "bm25_calibrated", "nli_top1", "nli_rerank_top3", "reject_all"
        })
        for key, scored in result["summaries"].items():
            self.assertTrue(0 <= scored["positive"]["correct_and_accepted"] <= 1)
            self.assertTrue(0 <= scored["contradiction"]["false_acceptance"] <= 1)
        self.assertEqual(result["summaries"]["reject_all"]["positive"]
                         ["correct_and_accepted"], 0.0)
        self.assertGreater(result["nli_candidate_count"], 0)


if __name__ == "__main__":
    unittest.main()
