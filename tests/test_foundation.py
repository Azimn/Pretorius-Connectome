"""Regression tests for the synthetic substrate and FlyWire CSV reader."""
import csv
import gzip
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pretorius_connectome.substrate import Edge, SparseRecurrentNetwork
from pretorius_connectome.flywire_import import summarize_connections


class SubstrateTests(unittest.TestCase):
    def test_deterministic_replay_and_reset(self):
        net = SparseRecurrentNetwork(3, [Edge(0, 1, 0.8), Edge(1, 2, -0.4)])
        first = [net.step([1, 0, 0]), net.step(), net.step()]
        net.reset()
        second = [net.step([1, 0, 0]), net.step(), net.step()]
        self.assertEqual(first, second)
        self.assertGreater(first[1][1], 0)
        self.assertLess(first[2][2], 0)

    def test_rejects_invalid_network(self):
        with self.assertRaises(ValueError):
            SparseRecurrentNetwork(0, [])
        with self.assertRaises(ValueError):
            SparseRecurrentNetwork(2, [Edge(0, 2, 1)])
        with self.assertRaises(ValueError):
            SparseRecurrentNetwork(2, [], leak=1.1)
        with self.assertRaises(ValueError):
            SparseRecurrentNetwork(2, []).step([1])

    def test_csv_and_gzip_equivalent(self):
        content = "pre_root_id,post_root_id,syn_count\n10,20,3\n20,30,5\n"
        with tempfile.TemporaryDirectory() as tmp:
            plain = Path(tmp) / "connections.csv"
            packed = Path(tmp) / "connections.csv.gz"
            plain.write_text(content, encoding="utf-8")
            with gzip.open(packed, "wt", encoding="utf-8") as handle:
                handle.write(content)
            a = summarize_connections(plain)
            b = summarize_connections(packed)
            self.assertEqual(a["neuron_ids_in_connections"], 3)
            self.assertEqual(a["connection_rows"], 2)
            self.assertEqual(a["synaptic_contacts"], 8)
            for key in ("neuron_ids_in_connections", "connection_rows", "synaptic_contacts"):
                self.assertEqual(a[key], b[key])

    def test_invalid_csv_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.csv"
            path.write_text("pre_root_id,post_root_id,syn_count\n1,2,-1\n")
            with self.assertRaises(ValueError):
                summarize_connections(path)
            path.write_text("a,b,c\n1,2,3\n")
            with self.assertRaises(ValueError):
                summarize_connections(path)


if __name__ == "__main__":
    unittest.main()
