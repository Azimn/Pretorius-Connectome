#!/usr/bin/env python3
"""Directly audit original FlyWire v783 CSR through shared-feature consumers.

The actual whole-brain CSR is required for a biological finding. A separate
--synthetic-test path exercises the verifier but never claims FlyWire evidence.
Neither the original CSR nor frozen autobiographical source may be rewritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from pretorius_connectome.associative import AssociativeMemory, Topology
from pretorius_connectome.imprinting import load_v12
from pretorius_connectome.pilot02 import episode_split
from pretorius_connectome.shared_features_bc01 import load_bc01_cache, SCHEMA as BC_SCHEMA
from pretorius_connectome.shared_memory_l2 import (
    SharedCache, SCHEMA as L2_SCHEMA, ENCODER as L2_ENCODER,
    SOURCE, SIDECARS, L1_ARCHIVE, L1_MANIFEST,
)
from pretorius_connectome.shared_memory import read_l1
from scripts.download_flywire_v783 import FILES as RELEASE_FILES, verify as verify_release

FIELDS = ("root_ids", "indptr", "indices", "synapse_counts")
EXPECTED_WHOLE_BRAIN = (139255, 15091983)
PROBES = ("laboratory map and specimen", "the cathedral and Clara")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(2 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(memoryview(np.ascontiguousarray(value)).cast("B")).hexdigest()


class StructureGuard:
    """Byte and array checksums recorded before any shared feature is loaded."""
    def __init__(self, path: Path):
        self.path = path
        self.disk_sha256 = file_sha256(path)
        self.topology = Topology.read(path)
        self.arrays = {name: getattr(self.topology, name).copy() for name in FIELDS}
        self.hashes = {name: array_sha256(data) for name, data in self.arrays.items()}
        self.n = len(self.arrays["root_ids"])
        self.m = len(self.arrays["indices"])
        self.contacts = int(np.sum(self.arrays["synapse_counts"], dtype=np.int64))

    def check(self, checkpoint: str):
        # Match the actual source NPZ bytes and independent unmutated values.
        if file_sha256(self.path) != self.disk_sha256:
            raise AssertionError(f"{checkpoint}: source biological NPZ was rewritten")
        with np.load(self.path, allow_pickle=False) as original:
            if set(original.files) != set(FIELDS):
                raise AssertionError(f"{checkpoint}: biological NPZ field set changed")
            for name in FIELDS:
                initial = self.arrays[name]
                live = getattr(self.topology, name)
                on_disk = original[name]
                if (initial.shape != live.shape or initial.dtype != live.dtype
                    or array_sha256(live) != self.hashes[name]
                    or array_sha256(on_disk) != self.hashes[name]
                    or not np.array_equal(initial, live)
                    or not np.array_equal(initial, on_disk)):
                    raise AssertionError(f"{checkpoint}: biological {name} mutated")
        if self.n != len(self.topology.root_ids) or self.m != len(self.topology.indices):
            raise AssertionError(f"{checkpoint}: biological graph geometry changed")
        if self.contacts != int(np.sum(self.topology.synapse_counts, dtype=np.int64)):
            raise AssertionError(f"{checkpoint}: total biological synapses changed")


def audit(topology_path: Path, shared_l2: Path, bc01_dir: Path,
          *, biological: bool, release_dir: Path | None = None) -> dict:
    source_file_hash = file_sha256(SOURCE)
    source_l1_hash = file_sha256(L1_ARCHIVE)
    source_manifest_hash = file_sha256(L1_MANIFEST)
    guard = StructureGuard(topology_path)
    if biological:
        if (guard.n, guard.m) != EXPECTED_WHOLE_BRAIN:
            raise ValueError("Only the pinned whole-brain v783 CSR can pass the real audit")
        if release_dir is None:
            raise ValueError("Real audit requires original verified FlyWire publisher files")
        for name, (md5, size) in RELEASE_FILES.items():
            if not verify_release(release_dir / name, md5, size):
                raise ValueError("Real FlyWire source publisher checksum mismatch: " + name)
    guard.check("verified-original-CSR")

    original_memories = load_v12(SOURCE, SIDECARS)
    records = read_l1(L1_ARCHIVE, L1_MANIFEST)
    shared_memories = load_v12(L1_ARCHIVE, SIDECARS, shared_manifest=L1_MANIFEST)
    if (len(original_memories) != len(records) or len(records) != 450
        or len({r["episode_id"] for r in records}) != 27
        or original_memories != shared_memories):
        raise AssertionError("Canonical L1 is not exactly equivalent to parsed original")
    guard.check("after-canonical-L1-consumption")

    bc_meta, bc_values = load_bc01_cache(bc01_dir)
    if (bc_meta["schema_version"] != BC_SCHEMA
        or bc_values.shape != (450, 256)
        or bc_meta["fit_event_ids"] != []
        or bc_meta["record_ids_ordered"] != [r["event_id"] for r in records]):
        raise AssertionError("BC01 lexical cache parity or source linkage failed")
    guard.check("after-BC01-256-cache-consumption")

    cache = SharedCache(shared_l2)
    cache.assert_original(SOURCE, SIDECARS)
    if (L2_SCHEMA != "pretorius.shared-features.v2"
        or L2_ENCODER != "sklearn-tfidf-word12-rankstable-v2"
        or cache.manifest["schema_version"] != L2_SCHEMA
        or cache.manifest["encoder_name"] != L2_ENCODER
        or cache.manifest["random_seed"] != 31):
        raise AssertionError("Incorrect or superseded TF-IDF cache version/split")
    train, validation, test = episode_split(original_memories, 31)
    cache.require_training_set([m.event_id for m in train])
    if ({m.episode_id for m in train} & {m.episode_id for m in validation}
        or {m.episode_id for m in train} & {m.episode_id for m in test}
        or {m.episode_id for m in validation} & {m.episode_id for m in test}):
        raise AssertionError("Episode holdout leakage")
    guard.check("after-source-validated-FlyWire-TFIDF-v2-consumption")

    # Real graph diffusion is actually executed, not merely inspected.
    # The TF-IDF v2 encoder is different from the historical uncached v1:
    # do not report their retrieval scores as equal.
    engine = AssociativeMemory(train, guard.topology, shared_cache=cache)
    graph_output = []
    for query in PROBES:
        for mode in ("lexical", "graph", "hybrid"):
            ranked = engine.rank(query, mode, top_k=3)
            if (len(ranked) != 3 or not all(np.isfinite(x["score"]) for x in ranked)
                or any(x["event_id"] not in engine.ids for x in ranked)):
                raise AssertionError("Invalid actual graph retrieval output")
            graph_output.append({
                "query": query,
                "mode": mode,
                "top_three_event_ids": [x["event_id"] for x in ranked],
                "top_three_scores": [x["score"] for x in ranked],
            })
    guard.check("after-actual-sparse-graph-retrieval")

    if (file_sha256(SOURCE) != source_file_hash
        or file_sha256(L1_ARCHIVE) != source_l1_hash
        or file_sha256(L1_MANIFEST) != source_manifest_hash):
        raise AssertionError("Immutable source corpus or L1 bytes were rewritten")
    guard.check("final")
    return {
        "status": "PASS: original CSR and original source bytes unchanged",
        "experiment": "Pretorius 450-memory shared-feature direct CSR invariance",
        "topology_kind": "verified original FlyWire whole-brain v783" if biological
                         else "synthetic nonbiological fixture",
        "biological_claim_allowed": bool(biological),
        "original_topology_sha256": guard.disk_sha256,
        "original_csr_array_sha256": guard.hashes,
        "neuron_count": guard.n,
        "directed_aggregate_edges": guard.m,
        "original_integer_synaptic_contacts": guard.contacts,
        "original_synaptic_counts_intact": True,
        "original_edges_and_root_ids_intact": True,
        "original_csr_npz_bytes_intact": True,
        "source_l0_sha256": source_file_hash,
        "source_l1_archive_sha256": source_l1_hash,
        "source_l1_manifest_sha256": source_manifest_hash,
        "records": len(records),
        "episodes": 27,
        "shared_feature_conditions": {
            "BC01": {
                "encoder": bc_meta["encoder_name"],
                "schema": bc_meta["schema_version"],
                "vectors": list(bc_values.shape),
                "artifact_sha256": bc_meta["artifact_sha256"],
            },
            "FlyWire": {
                "encoder": cache.manifest["encoder_name"],
                "schema": cache.manifest["schema_version"],
                "fit_seed": 31,
                "training_records": len(train),
                "shard_sha256": cache.manifest["shard_sha256"],
            },
        },
        "graph_probes": graph_output,
        "limitation": (
            "This is a direct source-and-anatomical immutability check. It neither "
            "writes autobiographical content into biological synaptic weights "
            "nor establishes neural recall, semantics, consciousness or speedup."
        ),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--topology", type=Path, help="original verified whole-brain CSR NPZ")
    group.add_argument("--synthetic-test", action="store_true", help="nonbiological fixture only")
    p.add_argument("--release-dir", type=Path, help="original publisher-verified files")
    p.add_argument("--shared-l2", required=True, type=Path)
    p.add_argument("--bc01-dir", required=True, type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if args.synthetic_test:
        if args.release_dir is not None:
            p.error("--release-dir is only for real FlyWire")
        target = ROOT / "data/derived/verification_synthetic_fixture.npz"
        target.parent.mkdir(parents=True, exist_ok=True)
        fake = Topology.synthetic(n=96, degree=6, seed=77)
        np.savez(target, root_ids=fake.root_ids, indptr=fake.indptr,
                 indices=fake.indices, synapse_counts=fake.synapse_counts)
    else:
        if not args.release_dir:
            p.error("--release-dir required for real biological verification")
        target = args.topology
    result = audit(target, args.shared_l2, args.bc01_dir,
                   biological=not args.synthetic_test,
                   release_dir=args.release_dir)
    output = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
