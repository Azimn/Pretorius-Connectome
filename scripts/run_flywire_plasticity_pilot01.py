#!/usr/bin/env python3
"""FlyWire sparse activity-trace plasticity Pilot 01, fully source-pinned.

Synaptic plasticity here means a training-dependent numerical overlay on the
existing directed connectivity, not a biological learning mechanism.
Results are exploratory on the already-inspected Pilot04 challenge.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from pretorius_connectome.associative import (
    AssociativeMemory, Topology, paired_case_diagnostics, summarize_challenge,
)
from pretorius_connectome.plasticity import apply_trace_overlay
from pretorius_connectome.pilot02 import episode_split
from pretorius_connectome.shared_memory_l2 import SharedCache, SCHEMA, ENCODER
from scripts.run_associative_memory import _locked_inputs, _threshold
from scripts.run_imprinting_pilot05 import _decisions

L1 = ROOT / "artifacts/shared_memory/v1/pretorius_l1_v1.jsonl.gz"
L1_MANIFEST = ROOT / "artifacts/shared_memory/v1/manifest.json"


def benchmark(topology: Topology, *, seeds=(31, 37, 43),
              gain=2.0, activity_cap=256, cache_root: Path | None = None) -> dict:
    memories, cases = _locked_inputs(L1, L1_MANIFEST)
    decisions = _decisions()
    if cache_root is None and "synthetic" not in topology.provenance:
        raise ValueError("real FlyWire experiment requires source-pinned deterministic v2 L2")
    trials = []
    for seed in seeds:
        train, validation, test = episode_split(memories, seed)
        cache = None
        if cache_root is not None:
            cache = SharedCache(cache_root / f"seed{seed}")
            cache.assert_original(ROOT / "memories/current/Pretorius_v12_450_Events_Complete.jsonl",
                                  ROOT / "memories/annotations/v12_450_sidecars.jsonl")
            if cache.manifest["random_seed"] != seed:
                raise ValueError("episode-fit cache seed mismatch")
            if (cache.manifest["schema_version"] != "pretorius.shared-features.v2"
                    or cache.manifest["encoder_name"] != "sklearn-tfidf-word12-rankstable-v2"
                    or SCHEMA != "pretorius.shared-features.v2"
                    or ENCODER != "sklearn-tfidf-word12-rankstable-v2"):
                raise ValueError("Real FlyWire study requires deterministic shared TF-IDF L2 v2")
            cache.require_training_set([m.event_id for m in train])

        real = AssociativeMemory(train, topology, shared_cache=cache)
        rewired = AssociativeMemory(
            train, topology.permuted_null(seed + 900), shared_cache=cache
        )
        train_ids = {m.event_id for m in train}
        absent_ids = {m.event_id for m in test}
        positives = [decisions[m.event_id] for m in train[:120]]
        negatives = [decisions[m.event_id] for m in validation[:120]]
        if not positives or not negatives:
            raise ValueError("empty split for calibration")

        def evaluate(engine, mode):
            return summarize_challenge(
                engine, cases, train_ids, absent_ids,
                _threshold(engine, mode, positives, negatives), mode
            )

        # All control predictions/acceptance thresholds are measured BEFORE
        # learning. The training rule never sees the challenge cases.
        results = {
            "narrative_tfidf": evaluate(real, "lexical"),
            "real_frozen_graph": evaluate(real, "graph"),
            "real_frozen_hybrid": evaluate(real, "hybrid"),
            "rewired_frozen_graph": evaluate(rewired, "graph"),
            "rewired_frozen_hybrid": evaluate(rewired, "hybrid"),
        }
        real_audit = apply_trace_overlay(
            real, gain=gain, activity_cap=activity_cap
        )
        null_audit = apply_trace_overlay(
            rewired, gain=gain, activity_cap=activity_cap
        )
        results.update({
            "real_plastic_graph": evaluate(real, "graph"),
            "real_plastic_hybrid": evaluate(real, "hybrid"),
            "rewired_plastic_graph": evaluate(rewired, "graph"),
            "rewired_plastic_hybrid": evaluate(rewired, "hybrid"),
        })
        pairs = {
            "real_plastic_vs_real_frozen": (
                "real_frozen_graph", "real_plastic_graph"),
            "rewired_plastic_vs_rewired_frozen": (
                "rewired_frozen_graph", "rewired_plastic_graph"),
            "real_plastic_vs_rewired_plastic": (
                "rewired_plastic_graph", "real_plastic_graph"),
            "real_plastic_vs_lexical": (
                "narrative_tfidf", "real_plastic_graph"),
            "real_plastic_hybrid_vs_rewired_plastic_hybrid": (
                "rewired_plastic_hybrid", "real_plastic_hybrid"),
        }
        trials.append({
            "seed": seed,
            "l2_encoder": cache.manifest["encoder_name"] if cache else "historical-uncached-v1",
            "l2_encoder_hash": cache.manifest["encoder_code_hash"] if cache else None,
            "l2_shard_sha256": cache.manifest["shard_sha256"] if cache else None,
            "train_events": len(train),
            "validation_events": len(validation),
            "test_events": len(test),
            "training_episode_ids": sorted({m.episode_id for m in train}),
            "validation_episode_ids": sorted({m.episode_id for m in validation}),
            "test_episode_ids": sorted({m.episode_id for m in test}),
            "real_learning": real_audit,
            "rewired_learning": null_audit,
            "methods": results,
            "paired_diagnostics": {
                name: paired_case_diagnostics(results[control], results[treatment])
                for name, (control, treatment) in pairs.items()
            },
        })
    return {
        "study": "FlyWire sparse trace plasticity Pilot 01 (exploratory)",
        "topology": topology.provenance,
        "neuron_count": int(len(topology.root_ids)),
        "raw_connection_entries": int(len(topology.indices)),
        "source_events": len(memories),
        "source_archive": str(L1.relative_to(ROOT)),
        "source_manifest": str(L1_MANIFEST.relative_to(ROOT)),
        "shared_l2": "deterministic train-only TF-IDF v2" if cache_root else "historical-uncached-v1 (synthetic only)",
        "shared_l2_schema": SCHEMA if cache_root else None,
        "shared_l2_encoder": ENCODER if cache_root else "historical-uncached-v1",
        "seeds": list(seeds),
        "challenge_status": ("Previously examined assistant-authored Pilot04 cases; "
                             "NOT blind, independently reviewed, or confirmatory."),
        "hypothesis": ("real topology plastic > rewired topology plastic for "
                       "correct-and-accepted positives, without increased "
                       "contradiction and absent-event false acceptance"),
        "plasticity_semantics": ("Only reweights original directed edges from "
                                 "train-only lexical pre/post coactivation; "
                                 "not STDP, neuromodulation or learned semantics."),
        "parameters": {"gain": gain, "trace_activity_cap": activity_cap,
                       "graph_steps": 2, "graph_activity_cap": 256,
                       "diffusion": 0.4, "hybrid_fraction": 0.25},
        "trials": trials,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    topo = ap.add_mutually_exclusive_group(required=True)
    topo.add_argument("--topology", type=Path,
                      help="checksum-verified original FlyWire-derived CSR")
    topo.add_argument("--synthetic-test", action="store_true",
                      help="nonbiological test fixture only")
    ap.add_argument("--seeds", default="31,37,43")
    ap.add_argument("--gain", type=float, default=2.0)
    ap.add_argument("--trace-activity-cap", type=int, default=256)
    ap.add_argument("--shared-cache-root", type=Path,
                    help="seed31/, seed37/, etc. with independently fitted L2")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    seeds = tuple(int(n) for n in args.seeds.split(","))
    if not seeds or len(seeds) != len(set(seeds)):
        ap.error("unique, nonempty seed list required")
    if not args.synthetic_test and args.shared_cache_root is None:
        ap.error("real FlyWire requires --shared-cache-root with pinned v2 features")
    graph = Topology.synthetic() if args.synthetic_test else Topology.read(args.topology)
    report = benchmark(
        graph, seeds=seeds, gain=args.gain,
        activity_cap=args.trace_activity_cap,
        cache_root=args.shared_cache_root,
    )
    output = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    compact = {
        "topology": report["topology"], "neuron_count": report["neuron_count"],
        "trials": [{
            "seed": t["seed"],
            "real_updated_edges": t["real_learning"]["edges_with_positive_trace"],
            "rewired_updated_edges": t["rewired_learning"]["edges_with_positive_trace"],
            "metrics": {
                m: {k: v for k, v in stat.items() if k != "case_results"}
                for m, stat in t["methods"].items()
            },
            "paired_diagnostics": t["paired_diagnostics"],
        } for t in report["trials"]],
    }
    print(json.dumps(compact, indent=2))
    if args.output:
        print("Complete source-identified case records: " + str(args.output))


if __name__ == "__main__":
    main()
