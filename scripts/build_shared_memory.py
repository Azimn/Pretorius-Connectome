#!/usr/bin/env python3
"""Build and test a checksum-verified, episode-fit-pinned shared TF-IDF cache."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
import tracemalloc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pretorius_connectome.shared_memory import (  # noqa: E402
    SOURCE, SIDECARS, SharedCache, build_cache,
)


def run(directory: Path, seed: int, source: Path, annotations: Path) -> dict:
    tracemalloc.start()
    before = time.perf_counter()
    manifest = build_cache(directory, seed=seed, events_path=source, sidecars_path=annotations)
    elapsed_build = time.perf_counter() - before
    loaded = SharedCache(directory)
    loaded.assert_original(source, annotations)
    elapsed_total = time.perf_counter() - before
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    example = "camphor beetle brass key"
    row = loaded.query(example)
    checksum = {name: value for name, value in manifest["shard_sha256"].items()}
    return {
        "status": "operational lexical cache; NOT semantic embeddings",
        "schema_version": manifest["schema_version"],
        "source_git_blob": manifest["source_git_blob"],
        "encoder_name": manifest["encoder_name"],
        "encoder_code_hash": manifest["encoder_code_hash"],
        "seed": seed, "documents": len(loaded.ids),
        "fit_documents": len(manifest["fit_event_ids"]),
        "vector_dim": manifest["vector_dim"], "nonzeros": manifest["nonzero_features"],
        "query_nonzeros": int(row.nnz),
        "building_seconds": round(elapsed_build, 6),
        "build_plus_reload_seconds": round(elapsed_total, 6),
        "python_tracemalloc_peak_bytes": peak,
        "shards": checksum,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, default=Path("data/derived/shared-memory-v1"))
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--source", type=Path, default=SOURCE)
    ap.add_argument("--annotations", type=Path, default=SIDECARS)
    ap.add_argument("--report", type=Path)
    args = ap.parse_args()
    result = run(args.output_dir, args.seed, args.source, args.annotations)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
