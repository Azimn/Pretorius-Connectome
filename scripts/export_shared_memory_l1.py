#!/usr/bin/env python3
"""Export verified original v12 memories as portable no-model L1 data."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pretorius_connectome.shared_memory import export_l1  # noqa: E402

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path,
        default=ROOT / "memories/current/Pretorius_v12_450_Events_Complete.jsonl",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "artifacts/shared_memory/v1",
    )
    args = parser.parse_args()
    data = export_l1(args.source, args.output_dir)
    print(json.dumps({
        "record_count": data["records"],
        "source_git_blob": data["source_git_blob"],
        "archive": str(args.output_dir / data["archive_filename"]),
        "manifest": str(args.output_dir / "manifest.json"),
        "gzip_sha256": data["gzip_sha256"],
        "payload_sha256": data["payload_sha256"],
    }, indent=2))

if __name__ == "__main__":
    main()
