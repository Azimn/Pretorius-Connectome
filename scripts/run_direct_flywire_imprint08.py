#!/usr/bin/env python3
"""Direct FlyWire synaptic imprinting Pilot 08: source-linked, retrieval-free readout.

BC01 sensory vectors are *lexical*, not validated semantic embeddings. A
trained synaptic state returns a distributed 256D vector, NOT narrative text.
Event identification uses an explicitly evaluator-only external codebook.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from pretorius_connectome.associative import Topology
from pretorius_connectome.direct_flywire_imprint import (
    DirectFlywireOverlay, select_features, fingerprint,
)
from pretorius_connectome.imprinting import load_v12, order_by_episode
from pretorius_connectome.pilot02 import episode_split
from pretorius_connectome.shared_features_bc01 import (
    load_bc01_cache, SCHEMA as BC_SCHEMA,
)
from pretorius_connectome.shared_memory import read_l1
from scripts.run_imprinting_pilot import verify_sources

SOURCE = ROOT / "memories/current/Pretorius_v12_450_Events_Complete.jsonl"
SIDECARS = ROOT / "memories/annotations/v12_450_sidecars.jsonl"
L1_DIR = ROOT / "artifacts/shared_memory/v1"
REAL_SHA = {
    "root_ids": "84bc9ec32c4f797e3f6e6557b30b5fc7f447b5852aeb9553f7dd18ed6bd3321d",
    "indptr": "bb91b075324c5971600d3c1bc296dabdfc95a18d695c0293a7d09647cff61595",
    "indices": "3180cc427eb92389f22afde12d99b19d6676443aa3b9ffd25589c97013cd404b",
    "synapse_counts": "ef00868a5a54e3c9bdf1c4d64966bdec6fe5946a6f6909937801afcbdc44385e",
}


def surfaces(memory) -> list[str]:
    return [c.surface for c in memory.cues if c.surface]


def train_cue(memory) -> str:
    forms = surfaces(memory)
    if len(forms) < 2:
        raise ValueError("Need two literal source cues for leave-one-cue-out training")
    return " ".join(forms[:-1])


def probe_cue(memory) -> str:
    forms = surfaces(memory)
    if len(forms) < 2:
        raise ValueError("Need two literal source cues for leave-one-cue-out testing")
    return forms[-1]


def score_probes(model: DirectFlywireOverlay, probes: list, reference_ids: list[str],
                 oracle_targets: np.ndarray) -> list[dict]:
    """Offline-only scorer, never passed into model construction or inference."""
    if oracle_targets.shape != (len(reference_ids), 256):
        raise ValueError("Evaluator needs the separate canonical content targets")
    id_to_index = {event_id: i for i, event_id in enumerate(reference_ids)}
    output = []
    for memory in probes:
        # Neural model receives only raw cue text. No narrative or ID passed in.
        response = model.infer(probe_cue(memory))
        norm = float(np.linalg.norm(response))
        similarity = oracle_targets @ response
        best = int(np.argmax(similarity))
        predicted = reference_ids[best] if norm > 1e-10 else None
        value = float(similarity[best]) if predicted is not None else 0.0
        output.append({
            "event_id": memory.event_id,
            "predicted": predicted,
            "known": memory.event_id in id_to_index,
            "correct": predicted == memory.event_id if memory.event_id in id_to_index else None,
            "best_cosine": round(value, 7),
            "readout_norm": round(norm, 7),
            "target_cosine": (
                round(float(similarity[id_to_index[memory.event_id]]), 7)
                if memory.event_id in id_to_index else None
            ),
        })
    return output


def calibration(known: list[dict], unknown: list[dict]) -> float:
    """Select acceptance threshold using training-heldout and validation only."""
    if not known or not unknown:
        raise ValueError("Insufficient calibration examples")
    positive = np.asarray([r["best_cosine"] for r in known])
    negative = np.asarray([r["best_cosine"] for r in unknown])
    values = np.sort(np.unique(np.concatenate((positive, negative))))
    cutoffs = [-1.000001, *map(float, (values[:-1] + values[1:]) / 2), 1.000001]
    return float(max(
        cutoffs,
        key=lambda t: (
            np.mean(positive > t) + np.mean(negative <= t),
            -abs(t),
        ),
    ))


def metrics(known: list[dict], unknown: list[dict], threshold: float) -> dict:
    return {
        "positive_n": len(known),
        "positive_correct_top1": round(
            sum(row["correct"] is True for row in known) / max(1, len(known)), 6
        ),
        "positive_correct_and_accepted": round(
            sum(row["correct"] is True and row["best_cosine"] > threshold
                for row in known) / max(1, len(known)), 6
        ),
        "positive_mean_target_cosine": round(
            sum(float(row["target_cosine"]) for row in known) / max(1, len(known)), 6
        ),
        "absent_n": len(unknown),
        "absent_false_acceptance": (
            round(sum(row["best_cosine"] > threshold for row in unknown)
                  / len(unknown), 6) if unknown else None
        ),
        "readout_nonzero_positive": sum(row["readout_norm"] > 0 for row in known),
    }


def benchmark(topology: Topology, bc01_dir: Path, *, seed: int = 31,
              cells_per_feature: int = 32, full_corpus: bool = False,
              checkpoint: Path | None = None, real_data: bool = False) -> dict:
    verify_sources(SOURCE, SIDECARS)
    if real_data and (
        len(topology.root_ids) != 139255 or len(topology.indices) != 15091983
        or fingerprint(topology) != REAL_SHA
        or int(topology.synapse_counts.sum(dtype=np.int64)) != 54492922
    ):
        raise ValueError("Original full-brain FlyWire v783 biological graph checksum mismatch")
    baseline = fingerprint(topology)
    memories = load_v12(SOURCE, SIDECARS)
    pinned_l1 = read_l1(L1_DIR / "pretorius_l1_v1.jsonl.gz",
                        L1_DIR / "manifest.json")
    if (len(memories) != 450 or len(pinned_l1) != 450
        or any(m.event_id != item["event_id"] or
               m.memory_text != item["memory_text"] or
               m.episode_id != item["episode_id"]
               for m, item in zip(memories, pinned_l1))):
        raise AssertionError("Frozen complete L1 archive not identical to source")
    meta, vectors = load_bc01_cache(bc01_dir)
    if (meta["schema_version"] != BC_SCHEMA or vectors.shape != (450, 256)
        or meta["record_ids_ordered"] != [m.event_id for m in memories]):
        raise ValueError("BC01 cache source/encoder/ordered records mismatch")
    position = {m.event_id: i for i, m in enumerate(memories)}
    train, validation, test = episode_split(memories, seed)
    if full_corpus:
        train = memories
        validation = []
        test = []
    learned = DirectFlywireOverlay(
        topology, seed=seed, cells_per_feature=cells_per_feature
    )
    shuffled = DirectFlywireOverlay(
        topology, seed=seed, cells_per_feature=cells_per_feature
    )
    untouched = DirectFlywireOverlay(
        topology, seed=seed, cells_per_feature=cells_per_feature
    )
    if not np.array_equal(learned.pre_cells, shuffled.pre_cells):
        raise AssertionError("Treatment and shuffled control do not share neuron assignment")
    ordered = order_by_episode(list(train), seed=20261008)
    shift = max(1, len(ordered) // 3)
    shuffled_order = np.roll(np.arange(len(ordered)), shift)
    for i, memory in enumerate(ordered):
        learned.imprint(train_cue(memory), vectors[position[memory.event_id]])
        wrong = ordered[int(shuffled_order[i])]
        if wrong.event_id == memory.event_id:
            raise AssertionError("Shuffled content control contains identical pair")
        shuffled.imprint(train_cue(memory), vectors[position[wrong.event_id]])
    for model in (learned, shuffled, untouched):
        model.assert_original_unchanged()
    if fingerprint(topology) != baseline:
        raise AssertionError("Original source biological anatomy changed after learning")
    ckpt = None
    if checkpoint is not None:
        ckpt = learned.save(checkpoint)
        recovered = DirectFlywireOverlay.load(topology, checkpoint)
        for memory in ordered[:3]:
            if not np.array_equal(recovered.infer(probe_cue(memory)),
                                  learned.infer(probe_cue(memory))):
                raise AssertionError("Checkpoint replay lost distributed neural readout")
        if not np.array_equal(recovered.delta, learned.delta):
            raise AssertionError("Checkpoint lost synaptic weight values")
    # An evaluator-only text-derived dictionary, outside the neural model.
    # It cannot be accessed by DirectFlywireOverlay.infer().
    ids = [m.event_id for m in ordered]
    candidate_vectors = np.stack([
        select_features(vectors[position[m.event_id]], top_k=learned.content_features)
        for m in ordered
    ])
    rng = np.random.default_rng(seed + 991)
    shuffled_eval = rng.permutation(len(ordered))
    cut = max(1, len(shuffled_eval) // 2)
    calibrate_mem = [ordered[int(i)] for i in shuffled_eval[:cut]]
    evaluate_mem = [ordered[int(i)] for i in shuffled_eval[cut:]]
    if full_corpus:
        evaluate_mem = ordered
    methods = {}
    outputs = {}
    for name, model in (
        ("learned_real_or_synthetic_graph", learned),
        ("shuffled_content_same_graph", shuffled),
        ("zero_overlay_same_graph", untouched),
    ):
        known_cal = score_probes(model, calibrate_mem, ids, candidate_vectors)
        positives = score_probes(model, evaluate_mem, ids, candidate_vectors)
        absent_cal = score_probes(model, validation, ids, candidate_vectors)
        absents = score_probes(model, test, ids, candidate_vectors)
        outputs[name] = {"known": positives, "absent": absents}
        methods[name] = {
            "metrics_at_learned_calibrated_threshold": None,
            "training_updates": model.imprints,
            "edge_update_events": model.edge_update_events,
            "nonzero_overlay_edges": model.modified_edges,
            "calibration_known": known_cal,
            "calibration_absent": absent_cal,
        }
    primary = methods["learned_real_or_synthetic_graph"]
    threshold = calibration(
        primary["calibration_known"], primary["calibration_absent"]
    ) if not full_corpus else 0.0
    for name, info in methods.items():
        info["metrics_at_learned_calibrated_threshold"] = metrics(
            outputs[name]["known"], outputs[name]["absent"], threshold
        )
        del info["calibration_known"]
        del info["calibration_absent"]
    return {
        "experiment": "Direct synaptic imprinting Pilot 08 (heteroassociative)",
        "evidence_status": (
            "Measured real FlyWire v783 original whole-brain topology"
            if real_data else "Synthetic computational smoke; not biological evidence"
        ),
        "claim_boundary": (
            "Real directed-edge-constrained signed numerical learning; model infer "
            "uses cue-only lexical hashing and an isolated learned synaptic overlay. "
            "Content readout is 256D lexical coordinates, not native-language memory. "
            "Candidate-event identification is an EXTERNAL oracle-assisted diagnostic. "
            "No STDP, autonomous autobiographical recall, semantic reasoning, "
            "independently reviewed prompts, or fly cognition is claimed."
        ),
        "graph": {
            "neurons": len(topology.root_ids),
            "raw_aggregated_directed_edges": len(topology.indices),
            "original_integer_synaptic_contacts": int(topology.synapse_counts.sum(dtype=np.int64)),
            "array_sha256": baseline,
            "original_anatomy_unchanged": fingerprint(topology) == baseline,
        },
        "input": {
            "canonical_source_events": 450,
            "source_episodes": 27,
            "representation": meta["encoder_name"],
            "cache_artifact_sha256": meta["artifact_sha256"],
            "source_archive": "artifacts/shared_memory/v1/pretorius_l1_v1.jsonl.gz",
            "readout": "signed sparse top-32 of original BC01 256D lexical content",
            "cue": "leave-one-literal-recall-cue-out; unsigned textual lookup forbidden",
            "episodes_disjoint": not full_corpus,
        },
        "training": {
            "seed": seed,
            "imprinted_events": len(ordered),
            "heldout_calibration_episode_count": len({m.episode_id for m in validation}),
            "heldout_test_episode_count": len({m.episode_id for m in test}),
            "neuron_cells_per_feature": cells_per_feature,
            "input_features_per_cue": learned.cue_features,
            "content_features_per_memory": learned.content_features,
            "learning_rate": learned.rate,
            "weight_cap": learned.weight_cap,
            "full_corpus_development_mode": full_corpus,
            "checkpoint_sha256": ckpt,
        },
        "calibration_threshold": threshold,
        "methods": methods,
        "external_evaluator_case_results": outputs,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--topology", type=Path)
    g.add_argument("--synthetic-test", action="store_true")
    p.add_argument("--bc01-dir", required=True, type=Path)
    p.add_argument("--seed", type=int, default=31)
    p.add_argument("--cells-per-feature", type=int)
    p.add_argument("--full-corpus", action="store_true")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.synthetic_test:
        topo = Topology.synthetic(n=8192, degree=16, seed=15)
    else:
        topo = Topology.read(a.topology)
    result = benchmark(
        topo, a.bc01_dir, seed=a.seed,
        cells_per_feature=a.cells_per_feature or (8 if a.synthetic_test else 32),
        full_corpus=a.full_corpus, checkpoint=a.checkpoint,
        real_data=not a.synthetic_test,
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    print(json.dumps({
        "experiment": result["experiment"],
        "evidence_status": result["evidence_status"],
        "trained_memories": result["training"]["imprinted_events"],
        "graph_neurons": result["graph"]["neurons"],
        "original_anatomy_unchanged": result["graph"]["original_anatomy_unchanged"],
        "metrics": {key: x["metrics_at_learned_calibrated_threshold"]
                    for key, x in result["methods"].items()},
        "complete_case_file": str(a.output),
    }, indent=2))


if __name__ == "__main__":
    main()
