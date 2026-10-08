#!/usr/bin/env python3
"""Export/validate reusable Pretorius v12 L1 + stateless BC01 L2 offline."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pretorius_connectome.shared_memory import export_bundle, load_bundle

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--validate-only", action="store_true")
    a = p.parse_args()
    if a.validate_only:
        manifest, rows, _ = load_bundle(a.output)
    else:
        manifest = export_bundle(
            ROOT / "memories/current/Pretorius_v12_450_Events_Complete.jsonl",
            ROOT / "memories/annotations/v12_450_sidecars.jsonl", a.output
        )
        _, rows, _ = load_bundle(a.output)
    print(json.dumps({
        "status": "reused immutable L1 and BC01 hashed lexical L2, not semantic",
        "records": len(rows),
        "first": rows[0]["source"]["event_id"],
        "last": rows[-1]["source"]["event_id"],
        "schema": manifest["schema_version"], "encoder": manifest["encoder_name"],
        "artifact_shas": manifest["files"],
        "source_blob": manifest["source_git_blob"],
    }, indent=2))

if __name__ == "__main__":
    main()
