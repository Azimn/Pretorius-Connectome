#!/usr/bin/env python3
"""Measured CPU/size parity of canonical bundle vs original load and BC hashing.

Run on immutable canonical data; all numbers are observations, not forecasts.
No inference API, and synthetic graph condition explicitly labeled.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
import numpy as np

from pretorius_connectome.shared_memory import (
    export_bundle, load_bundle, lexical_vector, bundle_to_memories,
)
from pretorius_connectome.associative import AssociativeMemory, Topology
from pretorius_connectome.imprinting import load_v12
from pretorius_connectome.pilot02 import episode_split
from scripts.run_imprinting_pilot import EVENTS, SIDECARS, verify_sources

def seconds(fn):
    begin = perf_counter()
    result = fn()
    return result, round(perf_counter() - begin, 6)

def run(bundle: Path) -> dict:
    verify_sources(EVENTS, SIDECARS)
    native, read_native = seconds(lambda: load_v12(EVENTS, SIDECARS))
    exported, write_time = seconds(lambda: export_bundle(EVENTS, SIDECARS, bundle))
    (manifest, records, matrix), verified_read = seconds(lambda: load_bundle(bundle))
    converted = bundle_to_memories(records)
    if [(x.event_id, x.memory_text) for x in native] != [
        (x.event_id, x.memory_text) for x in converted
    ]:
        raise ValueError("cache source equality failure")
    (_, _, src_t), baseline_encoding_time = seconds(
        lambda: (None, None, np.stack([lexical_vector(x.memory_text) for x in native]))
    )
    if not np.array_equal(src_t, matrix):
        raise ValueError("shared sensory feature mismatch")
    topo = Topology.synthetic(n=128, degree=4, seed=7)
    counts = topo.synapse_counts.copy()
    train_a, validation, test = episode_split(native, 31)
    train_b, _, _ = episode_split(converted, 31)
    a, native_fit = seconds(lambda: AssociativeMemory(train_a, topo))
    b, bundle_fit = seconds(lambda: AssociativeMemory(train_b, topo))
    for q in ("millstream map and specimen drawer", "cathedral laboratory betrayal"):
        for mode in ("lexical", "graph", "hybrid"):
            if not np.array_equal(a.score(q, mode), b.score(q, mode)):
                raise ValueError("FlyWire retrieval changed under shared L1")
    if not np.array_equal(topo.synapse_counts, counts):
        raise ValueError("synthetic graph synapse counts mutated")
    return {
        "study": "shared memory v1 CPU engineering measurement",
        "source_blob": manifest["source_git_blob"],
        "sidecar_blob": manifest["annotation_git_blob"],
        "reusable_events": len(records),
        "episode_split": {
            "train": len(train_a), "validation": len(validation), "test": len(test),
        },
        "timings_seconds": {
            "original_load": read_native,
            "artifact_export_including_lexical_hashing": write_time,
            "artifact_read_full_source_and_feature_verification": verified_read,
            "uncached_lexical_hashing_all_events": baseline_encoding_time,
            "original_train_only_tfidf_plus_synthetic_graph_fit": native_fit,
            "shared_L1_train_only_tfidf_plus_same_synthetic_graph_fit": bundle_fit,
        },
        "bundle_size_bytes": sum(x.stat().st_size for x in bundle.iterdir() if x.is_file()),
        "peak_rss_kb_linux": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "exact_original_vs_shared_retrieval_match": True,
        "exact_bc_hashed_feature_replay": True,
        "synthetic_graph_not_flywire": True,
        "biological_synapse_changed": False,
        "fairness": (
            "All costs include validation. L2 hashing stateless; FlyWire TF-IDF "
            "remains split-fit and has no cross-split feature-cache reuse. "
            "Small-run times are platform/thermal dependent, not universal speedups."
        ),
    }

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path,
                   default=Path("results/shared_memory/v1"))
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    result = run(args.bundle)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload)

if __name__ == "__main__":
    main()
