#!/usr/bin/env python3
"""Pilot11: frozen real FlyWire imprint, unseen literal cue and open-set rejection.

The checkpoint is NEVER trained or modified. The model sees only a cue at
inference and returns a signed 256D lexical vector. The memory-ID dictionary
is an explicitly external oracle for offline diagnostics, not model state.
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
    DirectFlywireOverlay, fingerprint, select_features,
)
from pretorius_connectome.imprinting import load_v12, order_by_episode
from pretorius_connectome.pilot02 import episode_split
from pretorius_connectome.shared_features_bc01 import load_bc01_cache, SCHEMA
from pretorius_connectome.shared_memory import read_l1
from scripts.run_direct_flywire_imprint08 import (
    SOURCE, SIDECARS, L1_DIR, REAL_SHA, surfaces,
)
from scripts.run_direct_flywire_imprint10 import (
    binding_cues, choose_probes, REFERENCE_PILOT08_THRESHOLD,
)
from scripts.run_imprinting_pilot import verify_sources

PILOT10_SOURCE_RUN = 37869353939
FROZEN_CHECKPOINT_SHA = "fbc2f329993d414cde3ce3a1271a1083098c6513410373e124362045a427c86a"
FROZEN_CASE_SHA = "ce2d463a72fa58a1e65265cf50f961f9e585610d058b9701d26791e033e7cea8"
CHECKPOINT = ROOT / "artifacts/imprinting/checkpoints/pilot10-original-v783-run37869353939.npz"
CASES = ROOT / "results/imprinting/runs/direct-flywire-imprint-pilot10-run37869353939.json"
VALIDATION_FPR_CAP = 0.10


def unseen_literal_source_cue(memory):
    """Return a source-owned cue that was NEVER presented in Pilot10 training."""
    all_cues = surfaces(memory)
    if len(all_cues) < 4:
        return None
    trained_positions = {0, len(all_cues) // 2, len(all_cues) - 1}
    trained_texts = set(binding_cues(memory))
    for i, literal in enumerate(all_cues):
        if i not in trained_positions and literal not in trained_texts:
            return literal
    return None


def score_rows(model, records, ids, target_vectors, cue_fn, condition):
    """External-oracle event identification after self-contained neural infer."""
    positions = {event_id: i for i, event_id in enumerate(ids)}
    results = []
    for memory in records:
        literal_cue = cue_fn(memory)
        if not literal_cue:
            raise ValueError("Unsupported source cue in selected evaluation")
        # This is the only information given to the numerical neural substrate.
        vec = model.infer(literal_cue)
        similarity = target_vectors @ vec  # Offline scorer, never model input.
        best = int(np.argmax(similarity))
        nonzero = bool(np.linalg.norm(vec) > 1e-10)
        predicted = ids[best] if nonzero else None
        true_index = positions.get(memory.event_id)
        row = {
            "event_id": memory.event_id,
            "condition": condition,
            "source_literal_cue": literal_cue,
            "known_train_event": true_index is not None,
            "predicted": predicted,
            "correct": bool(predicted == memory.event_id) if true_index is not None else None,
            "best_cosine": round(float(similarity[best]), 7) if nonzero else 0.0,
            "target_cosine": (round(float(similarity[true_index]), 7)
                              if true_index is not None else None),
            "nonzero_model_response": nonzero,
        }
        if condition == "unseen_fourth_cue":
            probe = model._input(literal_cue)
            known = [model._input(cue) for cue in binding_cues(memory)]
            active = set(map(int, np.flatnonzero(probe)))
            trained_active = set().union(
                *(set(map(int, np.flatnonzero(v))) for v in known)
            )
            row["unseen_shared_feature_count"] = len(active & trained_active)
            row["unseen_max_feature_cosine"] = round(
                max(float(probe @ v) for v in known), 7
            )
        results.append(row)
    return results


def calibrated_gate(calibration_known, calibration_absent, *,
                    max_validation_fpr=VALIDATION_FPR_CAP):
    """Set evaluator-only threshold; never consult heldout test data.

    The model cannot access this threshold or the oracle candidate codebook.
    All thresholds use strict similarity > threshold for positive acceptance.
    """
    if not 0 <= max_validation_fpr < 1:
        raise ValueError("Invalid validation false-positive ceiling")
    if not calibration_known or not calibration_absent:
        raise ValueError("Nonempty source calibration examples required")
    combined = sorted({float(row["best_cosine"]) for row in
                       (calibration_known + calibration_absent)})
    candidates = [min(combined) - 1e-6]
    candidates += [(a + b) / 2 for a, b in zip(combined[:-1], combined[1:])]
    candidates += [max(combined) + 1e-6]
    allowed = int(np.floor(max_validation_fpr * len(calibration_absent)))
    feasible = []
    for t in candidates:
        f = sum(row["nonzero_model_response"] and row["best_cosine"] > t
                for row in calibration_absent)
        if f <= allowed:
            good = sum(row["correct"] is True and row["best_cosine"] > t
                       for row in calibration_known)
            accepted = sum(row["nonzero_model_response"] and row["best_cosine"] > t
                           for row in calibration_known)
            feasible.append((good, accepted, -t, t, f))
    if not feasible:
        raise AssertionError("No valid validation-only rejection threshold")
    good, accepted, _, chosen, f = max(feasible)
    return {
        "threshold": float(chosen),
        "constraint": "maximum 10% validation-episode false acceptance, not test-tuned",
        "calibration_known_events": len(calibration_known),
        "calibration_absent_events": len(calibration_absent),
        "calibration_correct_and_accepted": good,
        "calibration_positive_accepted": accepted,
        "calibration_false_acceptances": f,
        "calibration_false_acceptance_rate": round(f / len(calibration_absent), 6),
    }


def summarize(rows, threshold):
    if not rows:
        return {"n": 0, "no_eligible_source_cases": True}
    known = all(r["known_train_event"] for r in rows)
    if any(r["known_train_event"] != known for r in rows):
        raise ValueError("Known/absent probes mixed")
    accepted = [r["nonzero_model_response"] and r["best_cosine"] > threshold
                for r in rows]
    result = {
        "n": len(rows),
        "nonzero_output": sum(x["nonzero_model_response"] for x in rows),
        "acceptance_rate": round(sum(accepted) / len(rows), 6),
        "mean_best_cosine": round(float(np.mean(
            [x["best_cosine"] for x in rows])), 6),
    }
    if known:
        result.update({
            "correct_top1": round(sum(r["correct"] is True for r in rows) / len(rows), 6),
            "correct_and_accepted": round(sum(r["correct"] is True and a
                for r, a in zip(rows, accepted)) / len(rows), 6),
            "mean_target_cosine": round(float(np.mean(
                [r["target_cosine"] for r in rows])), 6),
        })
    else:
        result["absent_false_acceptance"] = result["acceptance_rate"]
    if any("unseen_shared_feature_count" in x for x in rows):
        result["no_shared_source_feature_count"] = sum(
            r["unseen_shared_feature_count"] == 0 for r in rows
        )
        result["no_shared_source_feature_rate"] = round(
            result["no_shared_source_feature_count"] / len(rows), 6
        )
        result["source_feature_overlap_mean"] = round(float(np.mean([
            r["unseen_shared_feature_count"] for r in rows
        ])), 6)
    return result


def compare_source_case_rows(now, prior, *, condition):
    if len(now) != len(prior):
        raise AssertionError(condition + ": source event count changed")
    for row, baseline in zip(now, prior):
        if (row["event_id"] != baseline["event_id"]
            or row["predicted"] != baseline["predicted"]
            or row["correct"] != baseline["correct"]
            or row["nonzero_model_response"] !=
            (baseline["response_norm"] > 0)
            or abs(row["best_cosine"] - baseline["best_cosine"]) > 2e-6
            or (row["target_cosine"] is not None and abs(
                row["target_cosine"] - baseline["target_cosine"]) > 2e-6)):
            raise AssertionError(
                "Frozen Pilot10 trained last-cue/absent case replay drift: "
                + condition + ":" + row["event_id"])


def run(topology, checkpoint, original_report, bc01_dir, *, real=False):
    if real and (
        len(topology.root_ids) != 139255
        or len(topology.indices) != 15091983
        or int(topology.synapse_counts.sum(dtype=np.int64)) != 54492922
        or fingerprint(topology) != REAL_SHA
        or sha256(checkpoint.read_bytes()).hexdigest() != FROZEN_CHECKPOINT_SHA
    ):
        raise ValueError("Unverified original FlyWire v783 or Pilot10 learned checkpoint")
    source_hash = fingerprint(topology)
    verify_sources(SOURCE, SIDECARS)
    memories = load_v12(SOURCE, SIDECARS)
    l1 = read_l1(L1_DIR / "pretorius_l1_v1.jsonl.gz", L1_DIR / "manifest.json")
    if (len(memories) != 450 or len(l1) != 450
        or any(m.event_id != b["event_id"] or m.memory_text != b["memory_text"]
               or m.episode_id != b["episode_id"] for m, b in zip(memories, l1))):
        raise ValueError("Source corpus L0/L1 version mismatch")
    meta, vecs = load_bc01_cache(bc01_dir)
    if (meta.get("schema_version") != SCHEMA
        or vecs.shape != (450, 256)
        or meta["record_ids_ordered"] != [m.event_id for m in memories]
        or original_report["source"]["bc01_artifact_sha256"] !=
            meta["artifact_sha256"]
        or original_report["source"]["original_anatomy_array_sha256"] != source_hash):
        raise ValueError("Source encoder or topology differs from frozen Pilot10")
    train, val, test = episode_split(memories, 31)
    if (len(train), len(val), len(test)) != (317, 62, 71):
        raise AssertionError("Episode-disjoint split changed")
    ordered = order_by_episode(train.copy(), seed=20261008)
    positives = choose_probes(ordered, 31)  # original 159 evaluation positives
    positive_ids = {m.event_id for m in positives}
    calibration_train = [m for m in ordered if m.event_id not in positive_ids]
    selected_unseen = [m for m in positives if unseen_literal_source_cue(m) is not None]
    model = DirectFlywireOverlay.load(topology, checkpoint)
    if (model.imprints != 951 or model.seed != 31
        or model.cue_features != 8 or model.content_features != 32
        or (real and (model.cells_per_feature != 32 or
                      model.modified_edges != 28938))):
        raise ValueError("Loaded checkpoint is not original 3-cue Pilot10 state")
    id_to_pos = {m.event_id: i for i, m in enumerate(memories)}
    candidates = np.stack([
        select_features(vecs[id_to_pos[m.event_id]],
                        top_k=model.content_features) for m in ordered
    ])
    ids = [m.event_id for m in ordered]
    tests = {
        "calibration_trained": score_rows(
            model, calibration_train, ids, candidates,
            lambda m: binding_cues(m)[-1], "calibration_trained"
        ),
        "calibration_absent": score_rows(
            model, val, ids, candidates,
            lambda m: binding_cues(m)[-1], "calibration_absent"
        ),
        "test_trained": score_rows(
            model, positives, ids, candidates,
            lambda m: binding_cues(m)[-1], "test_trained"
        ),
        "test_absent": score_rows(
            model, test, ids, candidates,
            lambda m: binding_cues(m)[-1], "test_absent"
        ),
        "paired_trained_unseen_subset": score_rows(
            model, selected_unseen, ids, candidates,
            lambda m: binding_cues(m)[-1], "paired_trained_unseen_subset"
        ),
        "unseen_fourth_cue": score_rows(
            model, selected_unseen, ids, candidates,
            unseen_literal_source_cue, "unseen_fourth_cue"
        ),
    }
    prior = original_report["oracle_only_case_results"]["multicue_source"]
    compare_source_case_rows(
        tests["test_trained"], prior["last_trained_cue"],
        condition="original_last_trained_cue"
    )
    compare_source_case_rows(
        tests["test_absent"], prior["absent_episode_last_cue"],
        condition="original_absent_episode"
    )
    calibration = calibrated_gate(tests["calibration_trained"],
                                   tests["calibration_absent"])
    old = REFERENCE_PILOT08_THRESHOLD
    summaries = {
        name: {
            "original_pilot08_threshold": summarize(rows, old),
            "validation_constrained_threshold": summarize(
                rows, calibration["threshold"]
            ),
        }
        for name, rows in tests.items()
    }
    if fingerprint(topology) != source_hash:
        raise AssertionError("Original biological fly CSR changed")
    return {
        "study": "FlyWire Pilot11 original learned state: unseen source cue and validation-only rejection",
        "topology_status": ("source-verified original complete FlyWire v783"
                            if real else "synthetic structural-mask smoke only"),
        "limitations": (
            "Retrieval is from frozen source-constrained signed synaptic overlay. "
            "Neural forward pass receives cue only and outputs BC01 256D lexical vector. "
            "Event identities and rank are evaluated using an EXTERNAL oracle-only "
            "candidate codebook. Untrained fourth source cue is a different literal "
            "source field, not independently validated semantic paraphrase. "
            "Threshold is offline post-hoc diagnostic, not an autonomous neural "
            "open-set or truth-fact judgement. No confirmation or fly physiology."
        ),
        "evidence": {
            "original_pilot10_real_run": PILOT10_SOURCE_RUN,
            "checkpoint_sha256": sha256(checkpoint.read_bytes()).hexdigest(),
            "original_case_json_sha256": original_report.get("_input_sha256"),
            "anatomical_csr_array_sha256": source_hash,
            "original_anatomy_unchanged": True,
            "events": len(memories), "train_events": len(train),
            "test_positive_events": len(positives),
            "validation_unknown_events": len(val),
            "test_unknown_events": len(test),
            "eligible_unseen_source_events": len(selected_unseen),
            "source_unseen_cue_rule": (
                "first literal source recall cue whose index and text were not "
                "among Pilot10 first/middle/last imprint presentations"),
        },
        "calibration": calibration,
        "comparisons": summaries,
        "case_results": tests,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    graph = p.add_mutually_exclusive_group(required=True)
    graph.add_argument("--topology", type=Path)
    graph.add_argument("--synthetic-test", action="store_true")
    p.add_argument("--bc01-dir", required=True, type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--prior-cases", type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    if args.synthetic_test:
        from scripts.run_direct_flywire_imprint10 import benchmark
        import tempfile
        topo = Topology.synthetic(n=8192, degree=16, seed=15)
        with tempfile.TemporaryDirectory() as temp:
            cp = Path(temp) / "synthetic_pilot10_source.npz"
            prior = benchmark(topo, args.bc01_dir, seed=31,
                              cells_per_feature=8, checkpoint=cp)
            prior["_input_sha256"] = "synthetic current-run only"
            outcome = run(topo, cp, prior, args.bc01_dir)
    else:
        topo = Topology.read(args.topology)
        cp = args.checkpoint or CHECKPOINT
        prior_file = args.prior_cases or CASES
        if sha256(prior_file.read_bytes()).hexdigest() != FROZEN_CASE_SHA:
            p.error("Original main-archived real Pilot10 case JSON SHA mismatch")
        prior = json.loads(prior_file.read_text(encoding="utf-8"))
        prior["_input_sha256"] = FROZEN_CASE_SHA
        outcome = run(topo, cp, prior, args.bc01_dir, real=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(outcome, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps({
        "status": outcome["topology_status"],
        "eligible_unseen_source_cues": outcome["evidence"][
            "eligible_unseen_source_events"],
        "threshold": outcome["calibration"]["threshold"],
        "comparisons": outcome["comparisons"],
        "case_evidence": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
