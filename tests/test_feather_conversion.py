"""Synthetic regression tests for Feather-to-CSR conversion."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from convert_flywire_feather import convert


class FeatherConversionTests(unittest.TestCase):
    def test_duplicate_edges_and_isolated_neuron(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            roots = base / "roots.npy"
            feather_path = base / "connections.feather"
            output = base / "graph.npz"
            np.save(roots, np.array([30, 10, 20, 40], dtype=np.uint64))
            feather.write_feather(pa.table({
                "pre_pt_root_id": [30, 10, 30, 10],
                "post_pt_root_id": [10, 30, 10, 20],
                "syn_count": [2, 3, 4, 1],
            }), feather_path)
            report = convert(feather_path, roots, output)
            with np.load(output, allow_pickle=False) as graph:
                self.assertEqual(graph["root_ids"].tolist(), [10, 20, 30, 40])
                self.assertEqual(graph["indptr"].tolist(), [0, 2, 2, 3, 3])
                self.assertEqual(graph["indices"].tolist(), [1, 2, 0])
                self.assertEqual(graph["synapse_counts"].tolist(), [1, 3, 6])
            self.assertEqual(report["directed_pairs"], 3)
            self.assertEqual(report["synaptic_contacts"], 10)

    def test_unknown_neuron_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            np.save(base / "roots.npy", np.array([10, 20], dtype=np.uint64))
            feather.write_feather(pa.table({
                "pre_pt_root_id": [10], "post_pt_root_id": [30],
                "syn_count": [2],
            }), base / "connections.feather")
            with self.assertRaises(ValueError):
                convert(base / "connections.feather", base / "roots.npy", base / "out.npz")


    def test_mushroom_body_neuropil_rows_preserve_counts_and_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            roots = base / "roots.npy"
            feather_path = base / "connections.feather"
            subset = base / "mb.npz"
            np.save(roots, np.array([40, 30, 20, 10], dtype=np.uint64))
            feather.write_feather(pa.table({
                "pre_pt_root_id": [30, 10, 30, 10, 40],
                "post_pt_root_id": [10, 30, 10, 20, 10],
                "syn_count": [2, 3, 4, 1, 5],
                "neuropil": ["MB_CA_L", "MB_VL_R", "AL_L", "MB_ML_R", "LH_L"],
            }), feather_path)
            report = convert(feather_path, roots, subset,
                             neuropil_prefix="MB_", prune_to_region=True)
            self.assertEqual(report["source_rows"], 3)
            self.assertEqual(report["original_release_rows"], 5)
            self.assertEqual(report["synaptic_contacts"], 6)
            self.assertEqual(report["neuropil_prefix"], "MB_")
            self.assertTrue(report["pruned_to_incident_neurons"])
            with np.load(subset, allow_pickle=False) as graph:
                self.assertEqual(graph["root_ids"].tolist(), [10, 20, 30])
                self.assertEqual(graph["indptr"].tolist(), [0, 2, 2, 3])
                self.assertEqual(graph["indices"].tolist(), [1, 2, 0])
                self.assertEqual(graph["synapse_counts"].tolist(), [1, 3, 2])
            unpruned = convert(feather_path, roots, base / "fullroot.npz",
                               neuropil_prefix="MB_")
            self.assertEqual(unpruned["neurons"], 4)
            with np.load(base / "fullroot.npz", allow_pickle=False) as graph:
                self.assertEqual(graph["root_ids"].tolist(), [10, 20, 30, 40])
            with self.assertRaises(ValueError):
                convert(feather_path, roots, base / "bad.npz",
                        neuropil_prefix="MB_NOT_REAL")
            with self.assertRaises(ValueError):
                convert(feather_path, roots, base / "bad2.npz",
                        prune_to_region=True)



if __name__ == "__main__":
    unittest.main()
