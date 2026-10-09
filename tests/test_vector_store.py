"""End-to-end persistent Vector Fly DB: real 450-record source, no invented facts."""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pretorius_connectome.shared_memory_l2 import SharedCache, build_cache
from pretorius_connectome.vector_store import build_store, VectorFlyStore


class VectorFlyDatabaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.cache_dir = cls.root / "seed31"
        build_cache(cls.cache_dir, seed=31)
        cls.cache = SharedCache(cls.cache_dir)
        cls.train_db = cls.root / "train.sqlite"
        cls.all_db = cls.root / "all.sqlite"
        cls.train_report = build_store(cls.train_db, cls.cache_dir, scope="train")
        cls.all_report = build_store(cls.all_db, cls.cache_dir, scope="all")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_training_split_has_no_heldout_episodes(self):
        expected = set(self.cache.manifest["fit_event_ids"])
        with VectorFlyStore(self.train_db, self.cache_dir) as store:
            self.assertEqual(store.info()["indexed_documents"], len(expected))
            for i in self.cache.ids:
                self.assertEqual(store.get(i) is not None, i in expected)
            train_episodes = {
                m["episode_id"] for m in self.cache.records if m["event_id"] in expected
            }
            test_episodes = {
                m["episode_id"] for m in self.cache.records
                if m["event_id"] in self.cache.manifest["split_event_ids"]["test"]
            }
            self.assertFalse(train_episodes & test_episodes)
            self.assertEqual(store.verify_vectors()["verified_documents"], len(expected))

    def test_all_corpus_search_exactly_matches_fitted_source_vectors(self):
        self.assertEqual(self.all_report["indexed_documents"], 450)
        self.assertEqual(self.all_report["vector_dim"], 8192)
        queries = [
            self.cache.records[0]["memory_text"][:150],
            self.cache.records[103]["memory_text"][:150],
            "camphor beetle brass key",
        ]
        with VectorFlyStore(self.all_db, self.cache_dir) as store:
            self.assertEqual(store.info()["scope"], "all")
            self.assertEqual(store.verify_vectors()["verified_documents"], 450)
            for query in queries:
                actual = store.search(query, top_k=7)
                q = self.cache.query(query)
                scores = np.asarray((self.cache.docs @ q.T).toarray()).ravel()
                winners = sorted(
                    (i for i in range(len(scores)) if scores[i] > 0),
                    key=lambda i: (-scores[i], self.cache.ids[i]),
                )[:7]
                self.assertEqual([row["event_id"] for row in actual],
                                 [self.cache.ids[i] for i in winners])
                for got, pos in zip(actual, winners):
                    self.assertAlmostEqual(got["similarity"], float(scores[pos]),
                                           delta=1e-10)
                    self.assertEqual(got["provenance"], "reconstructed")
                    self.assertIn("not claim entailment", got["evidence_status"])
                    self.assertEqual(got["source_excerpt"],
                                     self.cache.records[pos]["memory_text"][:360])

    def test_source_record_retrieval_metadata_and_filters(self):
        target = self.cache.records[0]
        with VectorFlyStore(self.all_db, self.cache_dir) as store:
            self.assertEqual(store.get(target["event_id"]), target)
            self.assertIsNone(store.get("E99-NONEXISTENT"))
            found = store.search(
                target["memory_text"][:150],
                episode_id=target["episode_id"], provenance="reconstructed",
            )
            self.assertTrue(found)
            self.assertTrue(all(row["episode_id"] == target["episode_id"]
                                for row in found))
            self.assertEqual(store.search(
                target["memory_text"], episode_id="UNKNOWN-E99"), [])
            self.assertEqual(store.search(target["memory_text"],
                                          provenance="verified-lived"), [])
            self.assertEqual(store.search("zzzzqzxzy unrecoverableword12345"), [])
            with self.assertRaises(ValueError):
                store.search(" ")
            with self.assertRaises(ValueError):
                store.search("laboratory", top_k=0)

    def test_database_and_cache_are_immutable_and_split_pinned(self):
        with self.assertRaises(FileExistsError):
            build_store(self.train_db, self.cache_dir, scope="train")
        with VectorFlyStore(self.train_db, self.cache_dir) as store:
            event = self.cache.manifest["split_event_ids"]["validation"][0]
            self.assertIsNone(store.get(event))
        wrong_dir = self.root / "seed37"
        build_cache(wrong_dir, seed=37)
        with self.assertRaisesRegex(ValueError, "provenance mismatch"):
            VectorFlyStore(self.all_db, wrong_dir)

    def test_tampered_postings_and_canonical_text_fail_closed(self):
        altered = self.root / "forged.sqlite"
        altered.write_bytes(self.train_db.read_bytes())
        with sqlite3.connect(altered) as conn:
            row = conn.execute(
                "SELECT feature_id,doc_id FROM postings LIMIT 1"
            ).fetchone()
            conn.execute(
                "UPDATE postings SET weight=weight+0.25 "
                "WHERE feature_id=? AND doc_id=?", row
            )
        with self.assertRaisesRegex(ValueError, "Vector postings differ"):
            VectorFlyStore(altered, self.cache_dir)
        altered.write_bytes(self.train_db.read_bytes())
        with sqlite3.connect(altered) as conn:
            conn.execute("UPDATE memories SET record_json='{}' WHERE doc_id=("
                         "SELECT MIN(doc_id) FROM memories)")
        with self.assertRaisesRegex(ValueError, "canonical L1"):
            VectorFlyStore(altered, self.cache_dir)

    def test_database_metadata_and_rebuild_are_reproducible(self):
        other_path = self.root / "train-second.sqlite"
        other = build_store(other_path, self.cache_dir, scope="train")
        self.assertEqual(self.train_report["nonzero_postings"],
                         other["nonzero_postings"])
        self.assertEqual(self.train_report["indexed_documents"],
                         other["indexed_documents"])
        with VectorFlyStore(self.train_db, self.cache_dir) as a, \
                VectorFlyStore(other_path, self.cache_dir) as b:
            self.assertEqual(a.meta, b.meta)
            for query in ("beetle brass key", "forgotten agreement", "laboratory"):
                self.assertEqual(a.search(query), b.search(query))
        # File-level SQLite bytes are not required to stay identical across
        # different SQLite library versions; source and vector equality is.


if __name__ == "__main__":
    unittest.main()
