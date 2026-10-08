"""Tests for deterministic FlyWire connectivity CSR conversion."""
import gzip
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pretorius_connectome.csr_import import load_csr


class CSRTests(unittest.TestCase):
    def test_sorted_ids_aggregated_edges_and_compression(self):
        csv_data = (
            "pre_root_id,post_root_id,syn_count\n"
            "30,10,2\n10,30,3\n30,10,4\n10,20,1\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            plain = Path(tmp) / "edges.csv"
            zipped = Path(tmp) / "edges.csv.gz"
            plain.write_text(csv_data, encoding="utf-8")
            with gzip.open(zipped, "wt", encoding="utf-8") as stream:
                stream.write(csv_data)
            first = load_csr(plain)
            second = load_csr(zipped)
            for a, b in zip(first, second):
                self.assertEqual(a.tolist(), b.tolist())
            roots, ptr, targets, counts = first
            self.assertEqual(roots.tolist(), [10, 20, 30])
            self.assertEqual(ptr.tolist(), [0, 2, 2, 3])
            self.assertEqual(targets.tolist(), [1, 2, 0])
            self.assertEqual(counts.tolist(), [1, 3, 6])

    def test_invalid_counts_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.csv"
            path.write_text("pre_root_id,post_root_id,syn_count\n1,2,0\n")
            with self.assertRaises(ValueError):
                load_csr(path)


if __name__ == "__main__":
    unittest.main()
