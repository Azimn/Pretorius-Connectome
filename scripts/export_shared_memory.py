#!/usr/bin/env python3
"""Extend canonical compressed Pretorius L1 with deterministic BC01 lexical L2."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pretorius_connectome.shared_memory import export_l1, read_l1
from pretorius_connectome.shared_features_bc01 import build_bc01_cache, load_bc01_cache


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--validate-only", action="store_true")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    archive = a.output / "pretorius_l1_v1.jsonl.gz"
    manifest = a.output / "manifest.json"
    if not a.validate_only:
        if not (archive.exists() and manifest.exists()):
            export_l1(ROOT / "memories/current/Pretorius_v12_450_Events_Complete.jsonl",
                      a.output)
        build_bc01_cache(a.output)
    records = read_l1(archive, manifest)
    bc_meta, vector = load_bc01_cache(a.output)
    print(json.dumps({
        "source_layer": "canonical existing compressed L1",
        "source_records": len(records),
        "source_episodes": len({r["episode_id"] for r in records}),
        "feature_layer": bc_meta["schema_version"],
        "feature_dim": int(vector.shape[1]),
        "input_fit_count": len(bc_meta["fit_event_ids"]),
        "record_order_identical": bc_meta["record_ids_ordered"] ==
                                 [r["event_id"] for r in records],
        "artifact_sha256": bc_meta["artifact_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
