#!/usr/bin/env python3
"""Run a complete, labeled hybrid autobiographical-retrieval comparison.

All numeric findings must come from executed JSON, never inferred from code.
Synthetic fixture requires --synthetic-test. Actual FlyWire requires --topology.
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
    AssociativeMemory, Topology, summarize_challenge, paired_case_diagnostics,
)
from pretorius_connectome.shared_memory import SharedCache
from pretorius_connectome.imprinting import load_v12
from pretorius_connectome.pilot02 import episode_split, calibrate
from pretorius_connectome.pilot04 import load_challenge
from scripts.run_imprinting_pilot import EVENTS, SIDECARS, verify_sources
from scripts.run_imprinting_pilot04 import CHALLENGE, CHALLENGE_BLOB
from scripts.run_imprinting_pilot05 import _decisions
from scripts.run_imprinting_pilot import git_blob_sha


def _locked_inputs():
    verify_sources(EVENTS, SIDECARS)
    if git_blob_sha(CHALLENGE) != CHALLENGE_BLOB:
        raise ValueError("published post-hoc challenge checksum changed")
    memories = load_v12(EVENTS, SIDECARS)
    cases = load_challenge(CHALLENGE, memories)
    return memories, cases


def _threshold(engine, method, positives, negatives):
    pos = [
        {"known": True, "top_cosine": float(max(engine.score(x, method)))}
        for x in positives
    ]
    neg = [
        {"known": False, "top_cosine": float(max(engine.score(x, method)))}
        for x in negatives
    ]
    return calibrate(pos, neg)


def benchmark(topology, *, seeds=(31,), steps=2, activity_cap=256,
              diffusion=0.4, hybrid_fraction=0.25, shared_cache=None) -> dict:
    memories, cases = _locked_inputs()
    decisions = _decisions()
    trials = []
    for seed in seeds:
        train, validation, test = episode_split(memories, seed)
        if shared_cache is not None:
            if shared_cache.manifest["random_seed"] != seed:
                raise ValueError("Cache fit seed differs from benchmark seed")
            shared_cache.require_training_set([m.event_id for m in train])
        original = AssociativeMemory(
            train, topology, steps=steps, activity_cap=activity_cap,
            diffusion=diffusion, hybrid_fraction=hybrid_fraction,
            shared_cache=shared_cache,
        )
        randomized = AssociativeMemory(
            train, topology.permuted_null(seed + 900),
            steps=steps, activity_cap=activity_cap,
            diffusion=diffusion, hybrid_fraction=hybrid_fraction,
            shared_cache=shared_cache,
        )
        known_ids = {m.event_id for m in train}
        absent_ids = {m.event_id for m in test}
        # Source decisions are NOT searchable training documents and calibration
        # never sees any of the 68 previously authored challenge queries.
        positives = [decisions[m.event_id] for m in train[:120]]
        negatives = [decisions[m.event_id] for m in validation[:120]]
        if not positives or not negatives:
            raise ValueError("empty calibration split")
        methods = {
            "tfidf_word_narrative": (original, "lexical"),
            "topology_graph_only": (original, "graph"),
            "topology_hybrid": (original, "hybrid"),
            "target_stub_null_graph": (randomized, "graph"),
            "target_stub_null_hybrid": (randomized, "hybrid"),
        }
        method_results = {
            label: summarize_challenge(
                engine, cases, known_ids, absent_ids,
                _threshold(engine, mode, positives, negatives),
                mode,
            )
            for label, (engine, mode) in methods.items()
        }
        pairings = {
            "graph_vs_lexical": ("tfidf_word_narrative", "topology_graph_only"),
            "hybrid_vs_lexical": ("tfidf_word_narrative", "topology_hybrid"),
            "graph_vs_rewired": ("target_stub_null_graph", "topology_graph_only"),
            "hybrid_vs_rewired": ("target_stub_null_hybrid", "topology_hybrid"),
        }
        trials.append({
            "seed": seed,
            "training_events": len(train),
            "validation_events": len(validation),
            "test_events": len(test),
            "methods": method_results,
            "paired_diagnostics": {
                name: paired_case_diagnostics(
                    method_results[reference], method_results[challenger])
                for name, (reference, challenger) in pairings.items()
            },
        })
    return {
        "study": "connectome-constrained autobiographical retrieval v1.1",
        "topology": topology.provenance,
        "neuron_count": len(topology.root_ids),
        "raw_connection_entries": len(topology.indices),
        "source_events": len(memories),
        "shared_cache": ({"encoder": shared_cache.manifest["encoder_name"],
                          "encoder_code_hash": shared_cache.manifest["encoder_code_hash"],
                          "fit_seed": shared_cache.manifest["random_seed"],
                          "source_blob": shared_cache.manifest["source_git_blob"]}
                         if shared_cache is not None else None),
        "source_challenge_cases": len(cases),
        "challenge_status": (
            "POST-HOC REUSED assistant-authored unreviewed Pilot04. "
            "Not independent validation. Labels are evaluation-only."
        ),
        "graph_semantics": (
            "Directed log1p contact-count graph diffusion, positive-only, "
            "not biophysical dynamics and no learned biological plasticity."
        ),
        "null_semantics": (
            "Permutation of target stubs retains source outdegree and "
            "destination stub counts, but not weighted indegree or "
            "simple-graph uniqueness."
        ),
        "memory_semantics": (
            "Original full narratives remain authoritative searchable documents; "
            "event IDs are only evaluator targets, not encoded features. "
            "Scores never establish truth of an assertion."
        ),
        "parameters": {
            "steps": steps, "activity_cap": activity_cap,
            "diffusion": diffusion, "hybrid_fraction": hybrid_fraction,
        },
        "trials": trials,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--topology", type=Path, help="verified converted FlyWire CSR NPZ")
    group.add_argument("--synthetic-test", action="store_true",
                       help="explicit nonbiological demonstration fixture")
    task = parser.add_mutually_exclusive_group(required=True)
    task.add_argument("--query", type=str, help="retrieve inspectable source excerpts")
    task.add_argument("--benchmark", action="store_true",
                      help="evaluate lexical, biological, and target-stub null controls")
    parser.add_argument("--mode", choices=("lexical", "graph", "hybrid"),
                        default="hybrid")
    parser.add_argument("--seeds", default="31",
                        help="comma separated seeded episode splits, e.g. 31,37,43")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--activity-cap", type=int, default=256)
    parser.add_argument("--diffusion", type=float, default=0.4)
    parser.add_argument("--hybrid-fraction", type=float, default=0.25)
    parser.add_argument("--shared-cache", type=Path,
                        help="Optional exact train-fit shared memory cache; benchmark only")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    topology = (Topology.synthetic() if args.synthetic_test
                else Topology.read(args.topology))
    shared_cache = SharedCache(args.shared_cache) if args.shared_cache else None
    if shared_cache is not None:
        shared_cache.assert_original(EVENTS, SIDECARS)
        if not args.benchmark:
            parser.error("Shared cache requires an episode-split benchmark")
    if args.benchmark:
        seeds = tuple(int(x) for x in args.seeds.split(","))
        if not seeds or len(seeds) != len(set(seeds)):
            parser.error("seeds must be nonempty and unique")
        result = benchmark(
            topology, seeds=seeds, steps=args.steps,
            activity_cap=args.activity_cap,
            diffusion=args.diffusion, hybrid_fraction=args.hybrid_fraction,
            shared_cache=shared_cache,
        )
    else:
        records, _ = _locked_inputs()
        engine = AssociativeMemory(
            records, topology, steps=args.steps,
            activity_cap=args.activity_cap, diffusion=args.diffusion,
            hybrid_fraction=args.hybrid_fraction,
        )
        result = {
            "query": args.query,
            "topology": topology.provenance,
            "retrieval_mode": args.mode,
            "provenance_status": "source excerpt only, no truth verification",
            "matches": engine.rank(args.query, args.mode, top_k=5),
        }
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    if args.benchmark:
        compact = {
            "topology": result["topology"],
            "neuron_count": result["neuron_count"],
            "raw_connection_entries": result["raw_connection_entries"],
            "trials": [
                {"seed": trial["seed"], "metrics": {
                    name: {k: value for k, value in stats.items()
                           if k != "case_results"}
                    for name, stats in trial["methods"].items()
                }, "paired_diagnostics": trial["paired_diagnostics"]}
                for trial in result["trials"]
            ],
        }
        print(json.dumps(compact, indent=2))
        if args.output:
            print("Full per-case results: " + str(args.output))
    else:
        print(text)


if __name__ == "__main__":
    main()
