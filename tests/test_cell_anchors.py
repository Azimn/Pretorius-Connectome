"""Falsifiable input-population controls; no real-anatomy claim from fixtures."""
from __future__ import annotations
from pathlib import Path
import tempfile
import unittest
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pretorius_connectome.associative import Topology, AssociativeMemory
from pretorius_connectome.cell_anchors import (
    git_blob, select_cell_class_pool, degree_stratified_control,
)
from pretorius_connectome.imprinting import Cue, Memory


def memories():
    return [
        Memory("A", "E1", "I carried the brass lantern through the laboratory.",
               (Cue("a", "brass"),)),
        Memory("B", "E1", "The old companion visited near the cathedral at dawn.",
               (Cue("b", "cathedral"),)),
        Memory("C", "E2", "A warm rainy garden sheltered the purple flowers.",
               (Cue("c", "garden"),)),
    ]


class ClassAnchorTests(unittest.TestCase):
    def setUp(self):
        self.top = Topology.synthetic(n=96, degree=6, seed=3)

    def make_fixture(self, folder, duplicate=False):
        header = "root_id\tcell_class\tcell_type\tcell_sub_class\n"
        lines = [
            f"{i}\t{'Kenyon cell' if i % 9 == 0 else 'other'}\t{'KC' if i % 9 == 0 else 'U'}\troot\n"
            for i in range(96)
        ]
        if duplicate:
            lines.append(lines[0])
        blob = (header + "".join(lines)).encode("utf-8")
        path = Path(folder) / "annotations.tsv"
        path.write_bytes(blob)
        return path, git_blob(blob)

    def test_annotation_pins_and_exact_ids(self):
        with tempfile.TemporaryDirectory() as folder:
            path, digest = self.make_fixture(folder)
            actual, audit = select_cell_class_pool(
                path, self.top, expected_blob=digest,
            )
            self.assertEqual(len(actual), 11)
            np.testing.assert_array_equal(actual, np.arange(0, 96, 9))
            self.assertEqual(audit["matched_label_counts"], {"Kenyon cell": 11})
            self.assertEqual(audit["annotations_joined"], 96)
            with self.assertRaisesRegex(ValueError, "pinned"):
                select_cell_class_pool(path, self.top)
            path.write_bytes(path.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "pinned"):
                select_cell_class_pool(path, self.top, expected_blob=digest)

    def test_stratified_control_is_disjoint_and_deterministic(self):
        chosen = np.arange(0, 96, 9, dtype=np.int64)
        a, report = degree_stratified_control(self.top, chosen, seed=7)
        b, report2 = degree_stratified_control(self.top, chosen[::-1], seed=7)
        self.assertEqual(report["pool_size"], len(chosen))
        self.assertGreaterEqual(report["exact_degree_bin_fraction"], 0)
        self.assertLessEqual(report["exact_degree_bin_fraction"], 1)
        self.assertEqual(len(a), len(chosen))
        self.assertEqual(len(np.unique(a)), len(a))
        self.assertFalse(np.intersect1d(a, chosen).size)
        # Pool membership and its structural properties are independent of the
        # order the selected neuron IDs are supplied in; seeds fix random draw.
        np.testing.assert_array_equal(a, b)
        self.assertEqual(report, report2)
        with self.assertRaises(ValueError):
            degree_stratified_control(self.top, [0, 0])
        with self.assertRaises(ValueError):
            degree_stratified_control(self.top, np.arange(90))

    def test_full_pool_exactly_preserves_historical_graph_scores(self):
        original = AssociativeMemory(memories(), self.top)
        anchored = AssociativeMemory(
            memories(), self.top,
            anchor_pool=np.arange(len(self.top.root_ids))[::-1]
        )
        for mode in ("lexical", "graph", "hybrid"):
            np.testing.assert_array_equal(
                original.score("brass lantern", mode),
                anchored.score("brass lantern", mode),
            )
        for invalid in (
            [], [0, 0], [-1, 1], [len(self.top.root_ids)],
            [0.5, 1.1]
        ):
            with self.assertRaises(ValueError):
                AssociativeMemory(memories(), self.top, anchor_pool=invalid)
        with self.assertRaises(ValueError):
            AssociativeMemory(memories(), anchor_pool=[0])


if __name__ == "__main__":
    unittest.main()
