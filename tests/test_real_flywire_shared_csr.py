"""No false-positive claims from the real CSR integrity guard."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from pretorius_connectome.associative import Topology
from scripts.verify_real_flywire_shared_csr import StructureGuard, array_sha256


class OriginalFlyWireCSRIntegrityTests(unittest.TestCase):
    def fixture(self, folder):
        t = Topology.synthetic(n=64, degree=4, seed=17)
        p = Path(folder) / "csr.npz"
        np.savez(p, root_ids=t.root_ids, indptr=t.indptr,
                 indices=t.indices, synapse_counts=t.synapse_counts)
        return t, p

    def test_unchanged_source_and_arrays_verify(self):
        with tempfile.TemporaryDirectory() as d:
            t, p = self.fixture(d)
            guard = StructureGuard(p)
            guard.check("load")
            _ = guard.topology.transition()
            guard.check("after-computational-operator")
            self.assertEqual(guard.m, len(t.indices))
            self.assertEqual(guard.contacts, int(t.synapse_counts.sum()))
            self.assertEqual(
                guard.hashes["synapse_counts"], array_sha256(t.synapse_counts)
            )

    def test_original_in_memory_synapse_mutation_fails_closed(self):
        with tempfile.TemporaryDirectory() as d:
            _, p = self.fixture(d)
            guard = StructureGuard(p)
            guard.topology.synapse_counts[0] += 1
            with self.assertRaisesRegex(AssertionError, "synapse_counts"):
                guard.check("deliberate-mutation")

    def test_original_connection_mutation_fails_closed(self):
        with tempfile.TemporaryDirectory() as d:
            _, p = self.fixture(d)
            guard = StructureGuard(p)
            guard.topology.indices[0] = (guard.topology.indices[0] + 1) % guard.n
            with self.assertRaisesRegex(AssertionError, "indices"):
                guard.check("deliberate-mutation")

    def test_original_disk_npz_rewrite_fails_closed(self):
        with tempfile.TemporaryDirectory() as d:
            t, p = self.fixture(d)
            guard = StructureGuard(p)
            changed = t.synapse_counts.copy()
            changed[0] += 1
            np.savez(p, root_ids=t.root_ids, indptr=t.indptr,
                     indices=t.indices, synapse_counts=changed)
            with self.assertRaisesRegex(AssertionError, "NPZ"):
                guard.check("deliberate-rewrite")

    def test_source_structure_not_confused_with_learned_overlay(self):
        with tempfile.TemporaryDirectory() as d:
            _, p = self.fixture(d)
            guard = StructureGuard(p)
            weights = guard.topology.transition()
            weights.data *= 0.95
            guard.check("learned-operator-not-anatomical-rewrite")


if __name__ == "__main__":
    unittest.main()
