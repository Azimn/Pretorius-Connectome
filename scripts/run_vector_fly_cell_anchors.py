#!/usr/bin/env python3
"""Vector Fly Pilot 02: opt-in anatomical neuron-class cue anchoring.

Exploratory only. Source classes are not synonymous with learned memory,
language concepts, sensory function, or validated biophysical transfer.
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
from pretorius_connectome.cell_anchors import (
    ANNOTATION_BLOB, select_cell_class_pool, degree_stratified_control,
)
from pretorius_connectome.plasticity import apply_trace_overlay
from pretorius_connectome.pilot02 import episode_split
from pretorius_connectome.shared_memory_l2 import SharedCache, SCHEMA, ENCODER
from scripts.run_associative_memory import _locked_inputs, _threshold
from scripts.run_imprinting_pilot05 import _decisions

L1 = ROOT / "artifacts/shared_memory/v1/pretorius_l1_v1.jsonl.gz"
L1_MANIFEST = ROOT / "artifacts/shared_memory/v1/manifest.json"

METHODS = (
    "tfidf_lexical",
    "unrestricted_frozen_graph",
    "cell_class_frozen_graph",
    "degree_pool_frozen_graph",
    "cell_class_learned_graph",
    "degree_pool_learned_graph",
)


def benchmark(topology: Topology, annotations: Path, cache_root: Path, *,
              seeds=(31, 37, 43), match="Kenyon", field="cell_class",
              gain=2.0) -> dict:
    if "FlyWire-derived" not in topology.provenance:
        raise ValueError("Real annotation benchmark requires measured FlyWire CSR")
    source_memories, cases = _locked_inputs(L1, L1_MANIFEST)
    decisions = _decisions()
    real_pool, selection = select_cell_class_pool(
        annotations, topology, field=field, pattern=match
    )
    trials = []
    for seed in seeds:
        train, validation, test = episode_split(source_memories, int(seed))
        cache = SharedCache(cache_root / f"seed{seed}")
        cache.assert_original(
            ROOT / "memories/current/Pretorius_v12_450_Events_Complete.jsonl",
            ROOT / "memories/annotations/v12_450_sidecars.jsonl",
        )
        if (cache.manifest.get("random_seed") != seed
                or cache.manifest.get("schema_version") != "pretorius.shared-features.v2"
                or cache.manifest.get("encoder_name") != "sklearn-tfidf-word12-rankstable-v2"
                or SCHEMA != "pretorius.shared-features.v2"
                or ENCODER != "sklearn-tfidf-word12-rankstable-v2"):
            raise ValueError("Version-pinned train-only deterministic L2 v2 required")
        cache.require_training_set([m.event_id for m in train])
        null_pool, null_audit = degree_stratified_control(
            topology, real_pool, seed=seed + 1200
        )
        unrestricted = AssociativeMemory(train, topology, shared_cache=cache)
        restricted = AssociativeMemory(
            train, topology, shared_cache=cache, anchor_pool=real_pool
        )
        null = AssociativeMemory(
            train, topology, shared_cache=cache, anchor_pool=null_pool
        )
        train_ids = {m.event_id for m in train}
        absent_ids = {m.event_id for m in test}
        positives = [decisions[m.event_id] for m in train[:120]]
        negatives = [decisions[m.event_id] for m in validation[:120]]

        def eval_engine(engine, mode):
            return summarize_challenge(
                engine, cases, train_ids, absent_ids,
                _threshold(engine, mode, positives, negatives),
                mode,
            )

        methods = {
            "tfidf_lexical": eval_engine(unrestricted, "lexical"),
            "unrestricted_frozen_graph": eval_engine(unrestricted, "graph"),
            "cell_class_frozen_graph": eval_engine(restricted, "graph"),
            "degree_pool_frozen_graph": eval_engine(null, "graph"),
        }
        real_overlay = apply_trace_overlay(restricted, gain=gain)
        null_overlay = apply_trace_overlay(null, gain=gain)
        methods["cell_class_learned_graph"] = eval_engine(restricted, "graph")
        methods["degree_pool_learned_graph"] = eval_engine(null, "graph")
        comparisons = {
            "cell_class_vs_control_frozen": ("degree_pool_frozen_graph", "cell_class_frozen_graph"),
            "cell_class_vs_control_learned": ("degree_pool_learned_graph", "cell_class_learned_graph"),
            "cell_class_frozen_vs_lexical": ("tfidf_lexical", "cell_class_frozen_graph"),
            "cell_class_learned_vs_frozen": ("cell_class_frozen_graph", "cell_class_learned_graph"),
            "control_learned_vs_frozen": ("degree_pool_frozen_graph", "degree_pool_learned_graph"),
        }
        trials.append({
            "seed": int(seed),
            "train_events": len(train),
            "validation_events": len(validation),
            "test_events": len(test),
            "l2_encoder_hash": cache.manifest["encoder_code_hash"],
            "l2_shards": cache.manifest["shard_sha256"],
            "matched_control": null_audit,
            "class_learning_audit": real_overlay,
            "control_learning_audit": null_overlay,
            "methods": methods,
            "paired_diagnostics": {
                name: paired_case_diagnostics(methods[reference], methods[challenger])
                for name, (reference, challenger) in comparisons.items()
            },
        })
    return {
        "study": "Vector Fly neuron-class-constrained cue anchors Pilot 02",
        "status": "exploratory post-hoc; no confirmed biological memory or persona identity",
        "source_events": 450,
        "source_l1": str(L1.relative_to(ROOT)),
        "source_l2_schema": SCHEMA,
        "topology": topology.provenance,
        "neurons": len(topology.root_ids),
        "directed_connections": len(topology.indices),
        "cell_annotation": selection,
        "annotation_git_blob": ANNOTATION_BLOB,
        "methods": list(METHODS),
        "parameters": {"gain": gain, "cue_hash_person": "pt-graph-v1",
                       "steps": 2, "activity_cap": 256,
                       "null": "same-size disjoint nearest log2-total-degree pool"},
        "limitations": ("Previously examined, author-written retrieval prompts. "
                        "Class labels and hash-based anchors do not constitute "
                        "biologically learned semantic or sensory representations. "
                        "Degree-bin null is not a cell-class-preserving rewired graph, "
                        "and mismatched degree bins must be reported."),
        "trials": trials,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--seeds", default="31,37,43")
    parser.add_argument("--class-field", default="cell_class",
                        choices=("cell_class", "cell_type", "cell_sub_class"))
    parser.add_argument("--class-match", default="Kenyon")
    parser.add_argument("--gain", type=float, default=2.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    seeds = tuple(int(x) for x in args.seeds.split(","))
    if not seeds or len(set(seeds)) != len(seeds):
        parser.error("Unique nonempty seeds required")
    result = benchmark(
        Topology.read(args.topology), args.annotations, args.cache_root,
        seeds=seeds, match=args.class_match, field=args.class_field, gain=args.gain,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps({
        "study": result["study"],
        "class_pool": result["cell_annotation"]["matched_neurons"],
        "neurons": result["neurons"],
        "seeds": [row["seed"] for row in result["trials"]],
        "methods": list(METHODS),
        "full_json": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
