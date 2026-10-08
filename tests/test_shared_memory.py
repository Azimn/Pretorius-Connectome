"""Shared L1 + BC01 L2 source integrity and exact FlyWire retrieval parity."""
from __future__ import annotations
from pathlib import Path
import json
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from pretorius_connectome.shared_memory import (
    export_bundle, load_bundle, bundle_to_memories, lexical_vector,
    SOURCE_BLOB, ANNOTATION_BLOB,
)
from pretorius_connectome.imprinting import load_v12
from pretorius_connectome.associative import AssociativeMemory, Topology
from pretorius_connectome.pilot02 import episode_split
from scripts.run_imprinting_pilot import EVENTS, SIDECARS, verify_sources
from scripts.run_associative_memory import _locked_inputs


class SharedMemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        verify_sources(EVENTS, SIDECARS)
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.manifest = export_bundle(EVENTS, SIDECARS, cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_exact_v12_coverage_and_unchanged_source(self):
        m, rows, matrix = load_bundle(self.root)
        originals = load_v12(EVENTS, SIDECARS)
        self.assertEqual(len(rows), 450)
        self.assertEqual(matrix.shape, (450, 256))
        self.assertEqual(m["source_git_blob"], SOURCE_BLOB)
        self.assertEqual(m["annotation_git_blob"], ANNOTATION_BLOB)
        self.assertEqual([r["source"]["event_id"] for r in rows],
                         [r.event_id for r in originals])
        self.assertEqual([r["source"]["memory_text"] for r in rows],
                         [r.memory_text for r in originals])
        self.assertEqual(m["fit_event_ids"], [])

    def test_deterministic_exact_replay_and_query_compatible(self):
        with tempfile.TemporaryDirectory() as second:
            m2 = export_bundle(EVENTS, SIDECARS, Path(second))
            self.assertEqual(self.manifest, m2)
            for name in ("l1_records.jsonl", "bc01_lexical_256.npy"):
                self.assertEqual((self.root / name).read_bytes(),
                                 (Path(second) / name).read_bytes())
        _, rows, matrix = load_bundle(self.root)
        for n in (0, 33, 449):
            np.testing.assert_array_equal(
                matrix[n], lexical_vector(rows[n]["source"]["memory_text"]))
        self.assertTrue(np.all(np.isfinite(lexical_vector("new unseen query"))))

    def test_episode_split_exact_holdout_and_fit_scope(self):
        _, records, _ = load_bundle(self.root)
        memories = bundle_to_memories(records)
        train, val, test = episode_split(memories, 31)
        parts = [{m.episode_id for m in subset} for subset in (train, val, test)]
        self.assertFalse(parts[0] & parts[1])
        self.assertFalse(parts[1] & parts[2])
        self.assertFalse(parts[0] & parts[2])
        self.assertEqual(sum(map(len, (train, val, test))), 450)

    def test_lexical_retrieval_and_topology_are_unchanged(self):
        originals, _ = _locked_inputs()
        via_bundle, _ = _locked_inputs(self.root)
        train_original, _, _ = episode_split(originals, 31)
        train_bundle, _, _ = episode_split(via_bundle, 31)
        topology = Topology.synthetic(n=64, degree=3, seed=17)
        before = topology.synapse_counts.copy()
        a = AssociativeMemory(train_original, topology)
        b = AssociativeMemory(train_bundle, topology)
        for question in ("laboratory window specimen", "strange millstream map"):
            for mode in ("lexical", "graph", "hybrid"):
                np.testing.assert_array_equal(a.score(question, mode),
                                              b.score(question, mode))
        np.testing.assert_array_equal(topology.synapse_counts, before)
        np.testing.assert_array_equal(a.topology.indices, b.topology.indices)

    def test_tamper_and_mismatched_encoder_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            for name in ("l1_records.jsonl", "bc01_lexical_256.npy", "manifest.json"):
                (path / name).write_bytes((self.root / name).read_bytes())
            with (path / "bc01_lexical_256.npy").open("ab") as handle:
                handle.write(b"tamper")
            with self.assertRaises(ValueError):
                load_bundle(path)
            (path / "bc01_lexical_256.npy").write_bytes(
                (self.root / "bc01_lexical_256.npy").read_bytes()
            )
            manifest_path = path / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["encoder_name"] = "some-other-vector-space"
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                load_bundle(path)

    def test_source_rewrite_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            events = out / "modified.jsonl"
            events.write_bytes(EVENTS.read_bytes() + b"\n")
            with self.assertRaises(ValueError):
                export_bundle(events, SIDECARS, out / "bundle")


if __name__ == "__main__":
    unittest.main()
