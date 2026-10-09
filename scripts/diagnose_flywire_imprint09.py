#!/usr/bin/env python3
"""Pilot 09: isolate exact-trained-cue storage from heldout-cue transfer.

This is an evaluation-only diagnostic of Pilot08's frozen learned synaptic
overlay, not a new learned rule or a new parameter-search benchmark. The model
receives only cue strings in infer(); event identities and source vectors exist
only outside the model in the oracle evaluator.
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
from pretorius_connectome.direct_flywire_imprint import DirectFlywireOverlay, fingerprint, select_features
from pretorius_connectome.imprinting import load_v12, order_by_episode
from pretorius_connectome.pilot02 import episode_split
from pretorius_connectome.shared_features_bc01 import load_bc01_cache
from scripts.run_direct_flywire_imprint08 import (
    SOURCE, SIDECARS, L1_DIR, REAL_SHA, train_cue, probe_cue, score_probes
)
from scripts.run_imprinting_pilot import verify_sources

PILOT08_RUN = 37863818653
PILOT08_CHECKPOINT_SHA256 = "e6edf8dd68e84540140612dbcbc42f827e00ce707d9f2947adf3342e215fcb6d"
PILOT08_EXPERIMENT = "Direct synaptic imprinting Pilot 08 (heteroassociative)"
SCENARIOS = ("trained_cue", "withheld_last_cue", "absent_episode_last_cue")
CONDITIONS = ("trained_source", "shuffled_content", "frozen_untrained")


def audit_original_real(topology: Topology) -> None:
    if (len(topology.root_ids) != 139255 or len(topology.indices) != 15091983
        or fingerprint(topology) != REAL_SHA
        or int(np.sum(topology.synapse_counts, dtype=np.int64)) != 54492922):
        raise ValueError("Not checksum-verified actual whole-brain FlyWire v783")


def select_probe_ids(train: list, seed: int) -> tuple[list, list]:
    """Reuse EXACT Pilot08 test subset, without reselecting favorable cases."""
    ordered = order_by_episode(list(train), seed=20261008)
    rng = np.random.default_rng(seed + 991)
    indices = rng.permutation(len(ordered))
    cut = max(1, len(indices) // 2)
    return ordered, [ordered[int(i)] for i in indices[cut:]]


def case_vectors(model: DirectFlywireOverlay, cases: list,
                 candidate_ids: list[str], target_vectors: np.ndarray,
                 kind: str) -> list[dict]:
    """No candidate data is passed to model.infer() or embedded in its state."""
    id_to_pos = {event_id: j for j, event_id in enumerate(candidate_ids)}
    outputs = []
    for memory in cases:
        cue = train_cue(memory) if kind == "trained_cue" else probe_cue(memory)
        prediction = model.infer(cue)
        norm = float(np.linalg.norm(prediction))
        sims = target_vectors @ prediction
        best = int(np.argmax(sims))
        proposed = candidate_ids[best] if norm > 1e-10 else None
        labels_exist = memory.event_id in id_to_pos
        row = {
            "event_id": memory.event_id,
            "predicted": proposed,
            "correct": bool(labels_exist and proposed == memory.event_id) if labels_exist else None,
            "known": labels_exist,
            "best_cosine": round(float(sims[best]), 7) if proposed is not None else 0.0,
            "readout_norm": round(norm, 7),
            "target_cosine": (
                round(float(sims[id_to_pos[memory.event_id]]), 7)
                if labels_exist else None
            ),
        }
        if labels_exist:
            before = model._input(train_cue(memory))
            withheld = model._input(probe_cue(memory))
            left = set(np.flatnonzero(before).tolist())
            right = set(np.flatnonzero(withheld).tolist())
            row["cue_feature_overlap_count"] = len(left & right)
            row["cue_feature_cosine"] = round(float(before @ withheld), 7)
        outputs.append(row)
    return outputs


def summarize(rows: list[dict], threshold: float) -> dict:
    if not rows:
        raise ValueError("Cannot evaluate empty case group")
    known = all(x["known"] for x in rows)
    if not known and any(x["known"] for x in rows):
        raise ValueError("Mixed known and absent labels")
    output = {
        "count": len(rows),
        "nonzero_responses": sum(x["readout_norm"] > 0 for x in rows),
        "fraction_accepted": round(
            sum(x["best_cosine"] > threshold for x in rows) / len(rows), 6
        ),
        "mean_best_cosine": round(
            float(np.mean([x["best_cosine"] for x in rows])), 6
        ),
    }
    if known:
        output.update({
            "correct_top1": round(sum(x["correct"] is True for x in rows) / len(rows), 6),
            "correct_and_accepted": round(
                sum(x["correct"] is True and x["best_cosine"] > threshold
                    for x in rows) / len(rows), 6
            ),
            "mean_target_cosine": round(float(np.mean(
                [x["target_cosine"] for x in rows])), 6),
            "mean_cross_cue_input_cosine": round(float(np.mean(
                [x["cue_feature_cosine"] for x in rows])), 6),
            "fraction_cross_cue_no_shared_feature": round(
                sum(x["cue_feature_overlap_count"] == 0 for x in rows) / len(rows), 6
            ),
        })
    return output


def diagnose(topology: Topology, bc01_dir: Path, checkpoint: Path,
             original_report: dict, *, real: bool) -> dict:
    verify_sources(SOURCE, SIDECARS)
    if real:
        audit_original_real(topology)
        if sha256(checkpoint.read_bytes()).hexdigest() != PILOT08_CHECKPOINT_SHA256:
            raise ValueError("Not exact original Pilot08 learned FlyWire checkpoint")
    original = fingerprint(topology)
    original_memories = load_v12(SOURCE, SIDECARS)
    train, validation, test = episode_split(original_memories, 31)
    source_order, eval_probe = select_probe_ids(train, 31)
    meta, vectors = load_bc01_cache(bc01_dir)
    if (len(original_memories) != len(meta["record_ids_ordered"]) or
        meta["record_ids_ordered"] != [m.event_id for m in original_memories]
        or len(original_memories) != 450):
        raise ValueError("Shared BC01 cache misaligned with canonical autobiography")
    if (original_report.get("experiment") != PILOT08_EXPERIMENT
        or original_report.get("training", {}).get("seed") != 31
        or original_report["training"]["imprinted_events"] != len(train)
        or original_report["input"]["representation"] != meta["encoder_name"]
        or original_report["training"]["checkpoint_sha256"] !=
           sha256(checkpoint.read_bytes()).hexdigest()
        or (real and original_report["graph"]["array_sha256"] != REAL_SHA)):
        raise ValueError("Historical Pilot08 dataset/provenance/checkpoint mismatch")
    model = DirectFlywireOverlay.load(topology, checkpoint)
    if model.imprints != len(train):
        raise ValueError("Checkpoint does not represent exact full train split")
    if real and model.cells_per_feature != 32:
        raise ValueError("Whole-brain model lacks original mapping parameters")
    position = {m.event_id: i for i, m in enumerate(original_memories)}
    target_vectors = np.stack([
        select_features(vectors[position[m.event_id]], top_k=model.content_features)
        for m in source_order
    ])
    ids = [m.event_id for m in source_order]

    def new_model() -> DirectFlywireOverlay:
        return DirectFlywireOverlay(
            topology, seed=model.seed,
            cells_per_feature=model.cells_per_feature,
            cue_features=model.cue_features, content_features=model.content_features,
            rate=model.rate, weight_cap=model.weight_cap,
        )

    shuffled = new_model()
    shifted = np.roll(np.arange(len(source_order)), max(1, len(source_order) // 3))
    for i, memory in enumerate(source_order):
        wrong = source_order[int(shifted[i])]
        if wrong.event_id == memory.event_id:
            raise AssertionError("Shuffled training contains self-assignment")
        shuffled.imprint(train_cue(memory), vectors[position[wrong.event_id]])
    frozen = new_model()
    if (not np.array_equal(model.pre_cells, shuffled.pre_cells)
        or not np.array_equal(model.post_cells, shuffled.post_cells)
        or frozen.modified_edges != 0
        or model.modified_edges != original_report["methods"][
            "learned_real_or_synthetic_graph"]["nonzero_overlay_edges"]):
        raise AssertionError("Original neural placement, state or training mismatched")
    calibration_threshold = float(original_report["calibration_threshold"])
    conditions = {"trained_source": model,
                  "shuffled_content": shuffled,
                  "frozen_untrained": frozen}
    cases_by_group = {
        "trained_cue": eval_probe,
        "withheld_last_cue": eval_probe,
        "absent_episode_last_cue": test,
    }
    summaries = {}
    case_results = {}
    for name, substrate in conditions.items():
        case_results[name] = {}
        summaries[name] = {}
        for group in SCENARIOS:
            rows = case_vectors(
                substrate, cases_by_group[group], ids, target_vectors, group
            )
            case_results[name][group] = rows
            summaries[name][group] = summarize(rows, calibration_threshold)
        summaries[name]["modified_synaptic_edges"] = substrate.modified_edges
        substrate.assert_original_unchanged()

    # The archive must exactly reproduce every original Pilot08 held-out probe.
    original_positive = original_report["external_evaluator_case_results"][
        "learned_real_or_synthetic_graph"]["known"]
    original_absent = original_report["external_evaluator_case_results"][
        "learned_real_or_synthetic_graph"]["absent"]
    def same_original(a: list[dict], b: list[dict]) -> bool:
        # Pilot09 adds cue-transfer diagnostics, absent in the frozen Pilot08.
        # Compare every original field rather than extras introduced by Pilot09.
        return len(a) == len(b) and all(
            {key: row[key] for key in reference} == reference
            for row, reference in zip(a, b)
        )

    if (not same_original(
            case_results["trained_source"]["withheld_last_cue"], original_positive)
        or not same_original(
            case_results["trained_source"]["absent_episode_last_cue"], original_absent)):
        raise AssertionError("Saved biological state does not reproduce original cases")
    if fingerprint(topology) != original:
        raise AssertionError("Diagnostic touched original source connectivity")
    cross_cos = summaries["trained_source"]["withheld_last_cue"][
        "mean_cross_cue_input_cosine"
    ]
    return {
        "study": "Direct FlyWire Pilot09: frozen Pilot08 exact-cue versus cross-cue audit",
        "status": "real v783 diagnostic" if real else "synthetic test only",
        "data_source": {
            "original_workflow_run_id": PILOT08_RUN,
            "original_checkpoint_sha256": sha256(checkpoint.read_bytes()).hexdigest(),
            "canonical_memory_count": 450,
            "train_events": len(train),
            "evaluation_trained_events": len(eval_probe),
            "validation_absent_events": len(validation),
            "evaluation_absent_events": len(test),
            "bc01_artifact_sha256": meta["artifact_sha256"],
            "topology_array_sha256": original,
            "anatomical_synapse_counts_intact": True,
            "frozen_calibration_threshold": calibration_threshold,
        },
        "question": (
            "Can the existing learned overlay recover a remembered content vector "
            "from its own training cue? If so, how much of that signal transfers "
            "to a distinct literal cue from the same training memory?"
        ),
        "primary_interpretation_boundary": (
            "Trained-cue and withheld-cue prompts are sourced from existing "
            "unreviewed lexical recall-cue sidecars. A distinct literal cue "
            "is not a human-validated semantic paraphrase. Network inference "
            "is cue-only and returns lexical 256D features. External oracle "
            "target codebooks are used only for diagnostic identification."
        ),
        "cross_cue_mean_input_cosine": cross_cos,
        "comparisons": summaries,
        "case_results": case_results,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    gate = p.add_mutually_exclusive_group(required=True)
    gate.add_argument("--topology", type=Path)
    gate.add_argument("--synthetic-test", action="store_true")
    p.add_argument("--bc01-dir", type=Path, required=True)
    p.add_argument("--pilot08-checkpoint", type=Path)
    p.add_argument("--pilot08-result", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.synthetic_test:
        from scripts.run_direct_flywire_imprint08 import benchmark
        topo = Topology.synthetic(n=8192, degree=16, seed=15)
        archive = args.output.parent / "direct09-synthetic-base.npz"
        v08 = benchmark(topo, args.bc01_dir, seed=31, cells_per_feature=8,
                        checkpoint=archive, real_data=False)
    else:
        if not (args.pilot08_checkpoint and args.pilot08_result):
            p.error("Real v783 audit requires exact Pilot08 checkpoint and case JSON")
        topo = Topology.read(args.topology)
        archive = args.pilot08_checkpoint
        v08 = json.loads(args.pilot08_result.read_text(encoding="utf-8"))
    output = diagnose(topo, args.bc01_dir, archive, v08, real=not args.synthetic_test)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps({
        "status": output["status"],
        "train_events": output["data_source"]["train_events"],
        "inference": output["comparisons"],
        "case_file": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
