"""Reproducible cross-project L1 source adapter acceptance tests."""
import json
from pathlib import Path
import tempfile
import unittest
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pretorius_connectome.imprinting import load_v12
from pretorius_connectome.shared_memory import (
    SOURCE_BLOB, SOURCE_EVENTS, export_l1, read_l1, git_blob_sha
)

EVENTS = ROOT / "memories/current/Pretorius_v12_450_Events_Complete.jsonl"
SIDECARS = ROOT / "memories/annotations/v12_450_sidecars.jsonl"


class SharedMemoryTests(unittest.TestCase):
    def test_full_450_frozen_export_replay_and_existing_flywire_consumer(self):
        self.assertEqual(git_blob_sha(EVENTS.read_bytes()), SOURCE_BLOB)
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a", Path(tmp) / "b"
            one = export_l1(EVENTS, a)
            two = export_l1(EVENTS, b)
            self.assertEqual(one, two)
            self.assertEqual((a / "manifest.json").read_bytes(),
                             (b / "manifest.json").read_bytes())
            self.assertEqual((a / "pretorius_l1_v1.jsonl.gz").read_bytes(),
                             (b / "pretorius_l1_v1.jsonl.gz").read_bytes())
            rows = read_l1(a / "pretorius_l1_v1.jsonl.gz", a / "manifest.json")
            self.assertEqual(len(rows), SOURCE_EVENTS)
            self.assertEqual(
                [r["event_id"] for r in rows],
                [json.loads(s)["event_id"] for s in EVENTS.read_text(encoding="utf-8").splitlines()],
            )
            original = load_v12(EVENTS, SIDECARS)
            shared = load_v12(a / "pretorius_l1_v1.jsonl.gz", SIDECARS,
                              shared_manifest=a / "manifest.json")
            self.assertEqual(original, shared)
            self.assertEqual(len({m.episode_id for m in shared}), 27)

    def test_changes_to_source_archive_and_manifest_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "pretorius_l1_v1.jsonl.gz"
            manifest_path = root / "manifest.json"
            export_l1(EVENTS, root)
            canonical = archive.read_bytes()
            archive.write_bytes(canonical[:-1] + bytes([canonical[-1] ^ 1]))
            with self.assertRaisesRegex(ValueError, "Corrupted"):
                read_l1(archive, manifest_path)
            archive.write_bytes(canonical)
            manifest = json.loads(manifest_path.read_text())
            manifest["record_ids_ordered"][0] = "E00-001"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "event order"):
                read_l1(archive, manifest_path)
            manifest["record_ids_ordered"][0] = "E09-001"
            manifest["source_git_blob"] = "0" * 40
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "mismatched"):
                read_l1(archive, manifest_path)
            bad = root / "bad-source.jsonl"
            bad.write_bytes(EVENTS.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "Unpinned"):
                export_l1(bad, root / "bad-output")

if __name__ == "__main__":
    unittest.main()
