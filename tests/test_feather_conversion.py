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


if __name__ == "__main__":
    unittest.main()
