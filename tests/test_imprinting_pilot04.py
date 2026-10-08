"""Regression tests for Pilot 04 prompt quarantine and evaluation."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from pretorius_connectome.imprinting import Cue, load_v12
from pretorius_connectome.pilot02 import episode_split
from pretorius_connectome.pilot04 import _probes, load_challenge
from scripts.run_imprinting_pilot import EVENTS, SIDECARS, git_blob_sha, verify_sources
from scripts.run_imprinting_pilot04 import CHALLENGE, CHALLENGE_BLOB, run


class Pilot04Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        verify_sources(EVENTS, SIDECARS)
        cls.memories = load_v12(EVENTS, SIDECARS)
        cls.cases = load_challenge(CHALLENGE, cls.memories)

    def test_challenge_is_immutable_34_balanced_pairs(self):
        self.assertEqual(git_blob_sha(CHALLENGE), CHALLENGE_BLOB)
        self.assertEqual(len(self.cases), 68)
        self.assertEqual(len(set(c.event_id for c in self.cases)), 34)
        self.assertGreaterEqual(len(set(c.event_id[:3] for c in self.cases)), 20)
        by_event = {}
        for case in self.cases:
            by_event.setdefault(case.event_id, set()).add(case.kind)
        self.assertTrue(all(kinds == {"paraphrase", "contradiction"}
                            for kinds in by_event.values()))
        self.assertTrue(all(c.source_anchor for c in self.cases))

    def test_all_query_identifiers_hide_case_ids_and_labels(self):
        queries = _probes(self.cases)
        self.assertEqual(len(queries), 68)
        self.assertEqual({p.cue.cue_id for p in queries},
                         {"unseen-evaluation-query"})
        self.assertEqual(len({p.cue.surface for p in queries}), 68)
        for probe in queries:
            self.assertNotIn(probe.event_id, probe.cue.surface)

    def test_reject_unbalanced_or_modified_challenge(self):
        records = [json.loads(line) for line in CHALLENGE.read_text().splitlines()]
        with tempfile.TemporaryDirectory() as temp:
            file = Path(temp) / "challenge.jsonl"
            file.write_text("\n".join(json.dumps(r) for r in records[:-1]) + "\n")
            with self.assertRaises(ValueError):
                load_challenge(file, self.memories)
            records[0]["authoring"] = "independent_human_reviewed"
            file.write_text("\n".join(json.dumps(r) for r in records) + "\n")
            with self.assertRaises(ValueError):
                load_challenge(file, self.memories)

    def test_trial_splits_and_holdout_episode_ids_are_disjoint(self):
        for seed in (31, 37, 43):
            train, validation, test = episode_split(self.memories, seed)
            self.assertFalse({m.episode_id for m in train} &
                             {m.episode_id for m in validation})
            self.assertFalse({m.episode_id for m in train} &
                             {m.episode_id for m in test})
            self.assertFalse({m.episode_id for m in validation} &
                             {m.episode_id for m in test})
            known = {m.event_id for m in train}
            heldout = {m.event_id for m in test}
            positive_ids = {c.event_id for c in self.cases
                            if c.kind == "paraphrase"}
            self.assertGreaterEqual(len(positive_ids & known), 10)
            self.assertGreaterEqual(len(positive_ids & heldout), 2)

    def test_real_corpus_smoke_with_seven_frozen_methods(self):
        result = run(seeds=(31,), cue_units=96, memory_units=64, density=0.7)
        self.assertEqual(result["challenge_case_count"], 68)
        self.assertEqual(result["challenge_event_count"], 34)
        self.assertIn("NOT blinded", result["authoring_status"])
        trial = result["trials"][0]
        self.assertEqual(trial["counts"]["trained_positive_challenge"],
                         trial["counts"]["trained_contradictions"])
        self.assertEqual(len(trial["modes"]), 7)
        self.assertIn("positive_top1", trial["word_retrieval_reference"])
        for mode in trial["modes"].values():
            self.assertEqual(
                mode["positive_paraphrases"]["n"],
                trial["counts"]["trained_positive_challenge"]
            )
            self.assertEqual(
                mode["contradictory_prompts"]["n"],
                trial["counts"]["trained_contradictions"]
            )
            self.assertTrue(0 <= mode["positive_paraphrases"]["oracle_top1"] <= 1)
            self.assertTrue(
                0 <= mode["contradictory_prompts"]["false_acceptance_rate"] <= 1
            )
            self.assertTrue(
                0 <= mode["unimprinted_episode_prompts"]["false_acceptance_rate"] <= 1
            )
        self.assertEqual(
            trial["modes"]["token_unmodified"]["positive_paraphrases"]["oracle_top1"],
            0.0
        )


if __name__ == "__main__":
    unittest.main()
