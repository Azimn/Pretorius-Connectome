"""Portable shared memory schema, hash, split and FlyWire consumer tests."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
import shutil
import tempfile
import unittest
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from scipy import sparse

from pretorius_connectome.associative import AssociativeMemory, Topology
from pretorius_connectome.imprinting import load_v12
from pretorius_connectome.shared_memory_l2 import (
    SOURCE, SIDECARS, SHARDS, SOURCE_BLOB, SharedCache, build_cache, git_blob,
)


class SharedMemoryL2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.manifest = build_cache(cls.root / "seed31", seed=31)
        cls.cache = SharedCache(cls.root / "seed31")
        cls.memories = load_v12(SOURCE, SIDECARS)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_l1_provenance_order_and_sensory_candidate_status(self):
        self.assertEqual(git_blob(SOURCE.read_bytes()), SOURCE_BLOB)
        self.assertEqual(len(self.cache.records), 450)
        self.assertEqual(len({x["episode_id"] for x in self.cache.records}), 27)
        self.assertEqual([x["chronological_order"] for x in self.cache.records], list(range(1,451)))
        self.assertTrue(all(x["provenance"] == "reconstructed" for x in self.cache.records))
        self.assertTrue(all(x["provenance"] == "reconstructed" for x in self.cache.records))
        self.assertTrue(all(x["memory_text"] for x in self.cache.records))
        self.cache.assert_original()

    def test_training_fit_is_episode_disjoint(self):
        splits = self.manifest["split_event_ids"]
        ids = set(self.cache.ids)
        self.assertEqual(set().union(*(set(v) for v in splits.values())), ids)
        groups = list(self.manifest["split_episode_ids"].values())
        self.assertTrue(all(not(set(groups[i]) & set(groups[j])) for i in range(3) for j in range(i+1,3)))
        self.assertEqual(self.manifest["fit_event_ids"], splits["train"])
        self.assertLess(len(splits["train"]), 450)
        self.assertGreater(self.cache.docs.nnz, 0)

    def test_query_and_corpus_vectorization_identity(self):
        event = self.cache.records[0]
        actual = self.cache.query(event["memory_text"])
        row = self.cache.subset([event["event_id"]])
        np.testing.assert_allclose(actual.toarray(), row.toarray(), rtol=0, atol=1e-12)
        self.assertEqual(self.cache.query("camphor beetle").shape[1], self.manifest["vector_dim"])
        other = SharedCache(self.root / "seed31")
        np.testing.assert_array_equal(other.docs.toarray(), self.cache.docs.toarray())
        self.assertEqual(other.manifest, self.manifest)

    def test_tampering_and_wrong_fit_fail_closed(self):
        destination = self.root / "corrupt"
        shutil.copytree(self.root / "seed31", destination)
        with (destination / "records.jsonl").open("ab") as f:
            f.write(b"\n")
        with self.assertRaisesRegex(ValueError, "Corrupt|mismatched"):
            SharedCache(destination)
        shutil.rmtree(destination)
        shutil.copytree(self.root / "seed31", destination)
        m = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
        m["fit_event_ids"] = m["fit_event_ids"][1:]
        (destination / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
        with self.assertRaises(ValueError):
            SharedCache(destination)
        with self.assertRaises(ValueError):
            self.cache.require_training_set(["not-train"])

    def test_associative_legacy_matches_cached_training_subset(self):
        fit_ids = self.manifest["fit_event_ids"]
        by_id = {x.event_id: x for x in self.memories}
        train = [by_id[x] for x in fit_ids]
        topo = Topology.synthetic(n=128, degree=4, seed=5)
        original = AssociativeMemory(train, topo, steps=1, activity_cap=32)
        shared = AssociativeMemory(train, topo, steps=1, activity_cap=32, shared_cache=self.cache)
        for mode in ("lexical","graph","hybrid"):
            np.testing.assert_allclose(original.score("camphor beetle brass key", mode),
                                       shared.score("camphor beetle brass key", mode),
                                       rtol=1e-9, atol=1e-9)
        np.testing.assert_array_equal(topo.synapse_counts,
                                      topo.permuted_null().synapse_counts)
        with self.assertRaises(ValueError):
            AssociativeMemory(train[:-1], topo, shared_cache=self.cache)

    def test_self_consistent_noncanonical_record_rejected(self):
        # Attack: change content and update the checksum in the same cache
        # manifest; both shards appear internally valid but L1 is immutable.
        destination = self.root / "forged-narrative"
        shutil.copytree(self.root / "seed31", destination)
        rows = (destination / "records.jsonl").read_text(encoding="utf-8").splitlines()
        changed = json.loads(rows[0])
        changed["memory_text"] += " Invented autobiography."
        rows[0] = json.dumps(changed, sort_keys=True, ensure_ascii=False)
        file = destination / "records.jsonl"
        file.write_text("\n".join(rows) + "\n", encoding="utf-8")
        meta_path = destination / "manifest.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["shard_sha256"]["records.jsonl"] = hashlib.sha256(file.read_bytes()).hexdigest()
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "canonical pinned L1"):
            SharedCache(destination)

    def test_forged_idf_with_updated_manifest_hash_rejected(self):
        # Attack: the IDF and checksum are both replaced, while the declared
        # training split remains unchanged.
        destination = self.root / "forged-idf"
        shutil.copytree(self.root / "seed31", destination)
        idf_path = destination / "idf.npy"
        idf = np.load(idf_path, allow_pickle=False)
        idf[0] += 0.1
        np.save(idf_path, idf, allow_pickle=False)
        meta_path = destination / "manifest.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["shard_sha256"]["idf.npy"] = hashlib.sha256(idf_path.read_bytes()).hexdigest()
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "train-only narratives"):
            SharedCache(destination)

    def test_forged_split_seed_and_labels_rejected(self):
        destination = self.root / "forged-seed"
        shutil.copytree(self.root / "seed31", destination)
        meta_path = destination / "manifest.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["random_seed"] = 37
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "seeded episode partition"):
            SharedCache(destination)

    def test_source_bytes_or_cache_split_change_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            e = Path(folder) / "source.jsonl"
            e.write_bytes(SOURCE.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "Unpinned"):
                build_cache(Path(folder) / "built", events_path=e, sidecars_path=SIDECARS)


if __name__ == "__main__":
    unittest.main()
