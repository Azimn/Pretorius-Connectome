"""Live-source integration: cognition port consumes existing verified Vector Fly.

This deliberately rebuilds a canonical L2 and SQLite index on a clean runner,
and checks the 450-memory source without altering it.
"""
from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pretorius_connectome.cognition_port import (
    MemoryCognitionPort, MemoryAccessDenied, MemorySourceMismatch,
)
from pretorius_connectome.shared_memory_l2 import build_cache
from pretorius_connectome.vector_store import build_store, VectorFlyStore
from pretorius_connectome.vector_cognition_adapter import VectorFlyCognitionSource


class CanonicalVectorFlyPortTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.cache = root / "l2"
        build_cache(cls.cache, seed=31)
        cls.all_db = root / "all.sqlite"
        cls.train_db = root / "train.sqlite"
        build_store(cls.all_db, cls.cache, scope="all")
        build_store(cls.train_db, cls.cache, scope="train")
        cls.store = VectorFlyStore(cls.all_db, cls.cache)
        cls.source = VectorFlyCognitionSource(cls.store)
        cls.port = MemoryCognitionPort()
        cls.port.register(cls.source)
        cls.token = cls.port.grant_self(
            subject_id="pretorius", namespace="pretorius.reconstructed.v12"
        )

    @classmethod
    def tearDownClass(cls):
        cls.store.close()
        cls.tmp.cleanup()

    def test_real_source_retrieval_equals_unmodified_vector_fly(self):
        q = "camphor beetle brass key"
        expected = self.store.search(q, top_k=5)
        actual = self.port.search(
            token=self.token, subject_id="pretorius",
            namespace="pretorius.reconstructed.v12", query=q, top_k=5
        )
        self.assertEqual([x.record_id for x in actual],
                         [x["event_id"] for x in expected])
        self.assertEqual(actual[0].record_id, "E01-001")
        self.assertEqual(actual[0].origin_class, "reconstructed_prehistory")
        self.assertEqual(actual[0].index_scope, "all")
        self.assertEqual(actual[0].text, expected[0]["source_excerpt"])
        self.assertFalse(actual[0].eligible_for_lived_write)
        self.assertEqual(actual[0].epistemic_status, "candidate_not_entailment")

    def test_complete_source_is_accessible_only_as_external_evidence(self):
        raw = self.store.get("E01-001")
        entry = self.port.get(
            token=self.token, subject_id="pretorius",
            namespace="pretorius.reconstructed.v12", record_id="E01-001"
        )
        self.assertEqual(entry.text, raw["memory_text"])
        self.assertEqual(entry.source_hash, self.store.info()["source_git_blob"])
        self.assertEqual(entry.channel, "external_memory_evidence")
        self.assertFalse(entry.eligible_for_lived_write)
        with self.assertRaises(MemoryAccessDenied):
            self.port.get(token=self.token, subject_id="kiki",
                          namespace="pretorius.reconstructed.v12",
                          record_id="E01-001")

    def test_search_candidate_must_match_hydrated_original(self):
        original = self.store.search("camphor beetle brass key", top_k=1)[0]
        wrong = dict(original, source_excerpt="A falsified paragraph")
        with patch.object(self.store, "search", return_value=[wrong]):
            with self.assertRaises(MemorySourceMismatch):
                self.port.search(
                    token=self.token, subject_id="pretorius",
                    namespace="pretorius.reconstructed.v12",
                    query="camphor beetle brass key", top_k=1,
                )

    def test_training_scope_cannot_hydrate_heldout_source(self):
        with VectorFlyStore(self.train_db, self.cache) as restricted:
            source = VectorFlyCognitionSource(restricted)
            port = MemoryCognitionPort()
            port.register(source)
            token = port.grant_self(
                subject_id="pretorius", namespace="pretorius.reconstructed.v12"
            )
            heldout = restricted.cache.manifest["split_event_ids"]["test"][0]
            self.assertEqual(restricted.info()["scope"], "train")
            self.assertIsNone(port.get(
                token=token, subject_id="pretorius",
                namespace="pretorius.reconstructed.v12", record_id=heldout
            ))


if __name__ == "__main__":
    unittest.main()
