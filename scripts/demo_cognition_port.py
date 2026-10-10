#!/usr/bin/env python3
"""Run the existing Vector Fly index through the subject-bound cognition port.

This does not create an index, modify memories, launch a neural simulation, or
contact any LLM. Database and cache must have been built and verified first.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pretorius_connectome.cognition_port import MemoryCognitionPort
from pretorius_connectome.vector_cognition_adapter import VectorFlyCognitionSource
from pretorius_connectome.vector_store import VectorFlyStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    with VectorFlyStore(args.database, args.cache_dir) as store:
        bridge = MemoryCognitionPort()
        bridge.register(VectorFlyCognitionSource(store))
        capability = bridge.grant_self(
            subject_id="pretorius", namespace="pretorius.reconstructed.v12"
        )
        try:
            candidates = bridge.search(
                token=capability, subject_id="pretorius",
                namespace="pretorius.reconstructed.v12",
                query=args.query, top_k=args.top_k,
            )
            output = {
                "interface": "memory-cognition-port/0.1",
                "source_scope": store.info()["scope"],
                "read_only": True,
                "admission_to_lived_memory": False,
                "candidates": [asdict(item) for item in candidates],
            }
            print(json.dumps(output, indent=2, sort_keys=True, ensure_ascii=False))
        finally:
            bridge.revoke(capability)


if __name__ == "__main__":
    main()
