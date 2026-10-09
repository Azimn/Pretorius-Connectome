#!/usr/bin/env python3
"""Pilot13: train-order memory interference and representational capacity assay.

Separates the capacity/interference question from semantic cue transfer.
Only the original pinned BC01 cue and CONTENT codes are used. Training is
progressive: at every stage exactly the same frozen ordered 317 source events
and first/middle/last literal source-cue presentations, with predeclared
stage cutoffs. All evaluation has one fixed 317-target EXTERNAL oracle universe,
never an event-ID table in the neural infer(cue) model.

Matched arms: real anatomy vs fixed degree-preserving rewired anatomy,
true cue->content vs deranged pairing. Neither biological integer CSR nor
any previous Pilot08-Pilot12 checkpoint is edited.
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
from pretorius_connectome.rewire12 import (
    rewire_effective_edges, synthetic_aggregated_fixture,
)
from pretorius_connectome.shared_features_bc01 import (
    load_bc01_cache, SCHEMA,
)
from pretorius_connectome.shared_memory import read_l1
from scripts.run_direct_flywire_imprint08 import (
    SOURCE, SIDECARS, L1_DIR, REAL_SHA, surfaces,
)
from scripts.run_direct_flywire_imprint10 import (
    binding_cues, choose_probes, REFERENCE_PILOT08_THRESHOLD,
)
from scripts.diagnose_flywire_imprint11 import unseen_literal_source_cue
from scripts.run_imprinting_pilot import verify_sources

SEED = 31
REWIRE_SEED = 73
CUTS = (0, 16, 32, 64, 128, 256, 317)
ANCHORS = 16
NEWEST = 16
ARM_NAMES = (
    "original_learned", "rewired_learned",
    "original_deranged", "rewired_deranged",
)
SOURCE_PILOT10_SHA = "ce2d463a72fa58a1e65265cf50f961f9e585610d058b9701d26791e033e7cea8"
SOURCE_PILOT10_CHECKPOINT_SHA = "fbc2f329993d414cde3ce3a1271a1083098c6513410373e124362045a427c86a"


def rank_measured(model, records, ids, targets, *, learned_ids):
    """Oracle-only diagnostic after a model-only signed-vector forward pass.

    The model never receives IDs, target codebook or correct labels. The
    candidate pool is ALWAYS all 317 final training events, at stage zero too.
    """
    pos = {event_id: i for i, event_id in enumerate(ids)}
    out = []
    for record in records:
        response = model.infer(binding_cues(record)[-1])
        norm = float(np.linalg.norm(response))
        sims = targets @ response
        top = int(np.argmax(sims))
        guess = ids[top] if norm > 1e-10 else None
        mine = pos.get(record.event_id)
        known = mine is not None
        own = float(sims[mine]) if known else None
        competitor = None
        gap = None
        if known:
            if len(sims) < 2:
                raise ValueError("External candidate oracle must hold 2+ events")
            second = np.concatenate((sims[:mine], sims[mine+1:]))
            competitor = float(np.max(second))
            gap = own - competitor
        score = float(sims[top]) if guess is not None else 0.0
        out.append({
            "event_id": record.event_id,
            "was_imprinted_by_this_stage": record.event_id in learned_ids,
            "has_oracle_target": known,
            "predicted_event_id": guess,
            "correct_top1": bool(guess == record.event_id) if known else None,
            "best_cosine": round(score, 7),
            "self_content_cosine": round(own, 7) if known else None,
            "best_competing_cosine": round(competitor, 7) if known else None,
            "true_content_margin": round(gap, 7) if known else None,
            "neural_response_nonzero": bool(norm > 1e-10),
            "accepted_at_unchanged_pilot08_threshold": bool(
                guess is not None and score > REFERENCE_PILOT08_THRESHOLD
            ),
        })
    return out


def describe(rows):
    if not rows:
        return {"n": 0}
    known = all(r["has_oracle_target"] for r in rows)
    if any(r["has_oracle_target"] != known for r in rows):
        raise AssertionError("Do not mix target-available and truly absent events")
    out = {
        "n": len(rows),
        "nonzero_outputs": sum(r["neural_response_nonzero"] for r in rows),
        "accepted_at_original_threshold": sum(
            r["accepted_at_unchanged_pilot08_threshold"] for r in rows
        ),
        "mean_best_cosine": round(float(np.mean([
            r["best_cosine"] for r in rows
        ])), 6),
    }
    if known:
        out.update({
            "correct_top1": sum(r["correct_top1"] for r in rows),
            "correct_and_accepted": sum(
                r["correct_top1"] and
                r["accepted_at_unchanged_pilot08_threshold"] for r in rows
            ),
            "mean_target_cosine": round(float(np.mean([
                r["self_content_cosine"] for r in rows
            ])), 6),
            "mean_target_margin": round(float(np.mean([
                r["true_content_margin"] for r in rows
            ])), 6),
            "positive_target_margin": sum(
                r["true_content_margin"] > 0 for r in rows
            ),
        })
    else:
        out["absent_false_acceptances"] = out["accepted_at_original_threshold"]
    return out


def rank_auc(source_rows, absent_rows):
    """Threshold-free separability of known vs episode-absent best cosine.

    Ties score 0.5, no calibration or test-set threshold selection. This
    external oracle metric does NOT represent an internal neural judgment.
    """
    if not source_rows or not absent_rows:
        return None
    a = np.asarray([r["best_cosine"] for r in source_rows], dtype=np.float64)
    b = np.asarray([r["best_cosine"] for r in absent_rows], dtype=np.float64)
    comp = a[:, None] - b[None, :]
    return round(float((np.count_nonzero(comp > 0)
                       + 0.5 * np.count_nonzero(comp == 0)) / comp.size), 6)


def source_state(topology, bc01_dir, real):
    verify_sources(SOURCE, SIDECARS)
    original_hash = fingerprint(topology)
    if real and (
        len(topology.root_ids) != 139255
        or len(topology.indices) != 15091983
        or int(topology.synapse_counts.sum(dtype=np.int64)) != 54492922
        or original_hash != REAL_SHA
    ):
        raise ValueError("Real FlyWire requires exact original publisher v783 CSR")
    memories = load_v12(SOURCE, SIDECARS)
    l1 = read_l1(L1_DIR / "pretorius_l1_v1.jsonl.gz",
                 L1_DIR / "manifest.json")
    if (len(memories) != 450 or len(l1) != 450 or any(
        a.event_id != b["event_id"] or
        a.episode_id != b["episode_id"] or
        a.memory_text != b["memory_text"]
        for a, b in zip(memories, l1)
    )):
        raise ValueError("Source v12 narrative and version-pinned L1 mismatch")
    meta, vectors = load_bc01_cache(bc01_dir)
    if (meta["schema_version"] != SCHEMA
        or meta["record_ids_ordered"] != [x.event_id for x in memories]
        or vectors.shape != (450, 256)
        or (real and meta["artifact_sha256"] !=
            "65ae81401a17ec68adc3900395f226734618be4ffb2bae24745d989bd2a17625")):
        raise ValueError("Original BC01 neural source features changed")
    train, validation, absent = episode_split(memories, SEED)
    ordered = order_by_episode(train.copy(), seed=20261008)
    original_probes = choose_probes(ordered, SEED)
    if (len(ordered), len(validation), len(absent), len(original_probes)) != (
        317, 62, 71, 159
    ):
        raise AssertionError("Historical episode split or positive probes changed")
    position = {m.event_id: i for i, m in enumerate(memories)}
    codebook = np.stack([
        select_features(vectors[position[m.event_id]], top_k=32)
        for m in ordered
    ])
    return (
        original_hash, memories, meta, vectors, position,
        ordered, original_probes, validation, absent, codebook,
    )


def compare_original_pilot10_rows(actual, previous, label):
    if len(actual) != len(previous):
        raise AssertionError(label + ": original positive/absent count changed")
    for now, before in zip(actual, previous):
        if (now["event_id"] != before["event_id"]
            or now["predicted_event_id"] != before["predicted"]
            or now["correct_top1"] != before["correct"]
            or now["neural_response_nonzero"] !=
                (before["response_norm"] > 0)
            or abs(now["best_cosine"] - before["best_cosine"]) > 2e-6):
            raise AssertionError(label + ": Pilot10 exact original case drift at "
                                 + now["event_id"])


def run(topology, bc01_dir, *, real=False, cells_per_feature=32,
        previous_cases=None, previous_checkpoint=None,
        weights_dir=None):
    (original_hash, memories, meta, vectors, pos,
     ordered, probes, validation, absent, targets) = source_state(
        topology, bc01_dir, real
    )
    if CUTS[0] != 0 or CUTS[-1] != len(ordered):
        raise AssertionError("Predetermined model exposure stages are invalid")
    model_kwargs = {
        "seed": SEED, "cells_per_feature": cells_per_feature,
        "cue_features": 8, "content_features": 32, "rate": 0.7 / 3,
    }
    template = DirectFlywireOverlay(topology, **model_kwargs)
    null, null_report = rewire_effective_edges(
        topology, template.pre_cells, template.post_cells,
        seed=REWIRE_SEED, swaps_per_edge=2,
    )
    conditions = {
        name: DirectFlywireOverlay(
            null if name.startswith("rewired") else topology, **model_kwargs
        )
        for name in ARM_NAMES
    }
    # Same source->target derangement as the previous Pilot12 shuffled control.
    permutation = np.roll(np.arange(len(ordered)),
                          max(1, len(ordered) // 3))
    ids = [m.event_id for m in ordered]
    anchor_cases = ordered[:ANCHORS]
    test_eligible = [m for m in probes if unseen_literal_source_cue(m)]
    if len(test_eligible) != 31:
        raise AssertionError("Historical fourth-source-cue pool changed")

    stages = []
    history = {name: {} for name in ARM_NAMES}
    for step, cutoff in enumerate(CUTS):
        if step:
            start = CUTS[step - 1]
            for i in range(start, cutoff):
                source = ordered[i]
                deranged = ordered[int(permutation[i])]
                if source.event_id == deranged.event_id:
                    raise AssertionError("Deranged control was not deranged")
                for condition, model in conditions.items():
                    target = deranged if condition.endswith("deranged") else source
                    content = vectors[pos[target.event_id]]
                    for cue in binding_cues(source):
                        model.imprint(cue, content)
        learned_ids = set(ids[:cutoff])
        # The external 317-event oracle candidate universe never contracts.
        # Anchor study is purposefully all initially learned 16 source records.
        groups = {
            "first16_anchors": anchor_cases,
            "newest16": ordered[max(0, cutoff - NEWEST):cutoff],
            "source_test_known_learned": [
                m for m in probes if m.event_id in learned_ids
            ],
            "next16_not_yet_learned": ordered[
                cutoff:min(cutoff+NEWEST, len(ordered))
            ],
            "episode_absent_71": absent,
        }
        outcomes, aggregate = {}, {}
        for name, model in conditions.items():
            outcomes[name] = {}
            aggregate[name] = {
                "learned_nonzero_synaptic_edges": model.modified_edges,
                "imprint_presentations": model.imprints,
                "edge_update_events": model.edge_update_events,
                "groups": {},
            }
            for group, records in groups.items():
                values = rank_measured(
                    model, records, ids, targets, learned_ids=learned_ids
                )
                # Keep full original rows for independent paired longitudinal
                # comparison. No inference-ever access to source narratives.
                outcomes[name][group] = values
                aggregate[name]["groups"][group] = describe(values)
            aggregate[name]["oracle_known_absent_auc"] = rank_auc(
                outcomes[name]["source_test_known_learned"],
                outcomes[name]["episode_absent_71"]
            )
            if model.imprints != cutoff * 3:
                raise AssertionError("Arms differ in actual memory exposures")
            model.assert_original_unchanged()
        stages.append({
            "trained_source_memories": cutoff,
            "first_imprint_at_step": CUTS[step-1] if step else None,
            "total_source_cue_exposures_per_arm": cutoff * 3,
            "measured": aggregate,
            "original_case_rows_external_oracle_only": outcomes,
        })
        for condition, arm in outcomes.items():
            history[condition][str(cutoff)] = arm["first16_anchors"]

    expected_final = None
    terminal_checkpoint = None
    if real:
        if previous_cases is None or previous_checkpoint is None:
            raise ValueError("Real test MUST be anchored to original Pilot10 cases and weights")
        case_file = Path(previous_cases)
        weight_file = Path(previous_checkpoint)
        if (sha256(case_file.read_bytes()).hexdigest() != SOURCE_PILOT10_SHA
            or sha256(weight_file.read_bytes()).hexdigest() !=
                SOURCE_PILOT10_CHECKPOINT_SHA):
            raise ValueError("Exact original Pilot10 real evidence pins differ")
        source_case = json.loads(case_file.read_text(encoding="utf-8"))
        learned = conditions["original_learned"]
        replay = DirectFlywireOverlay.load(topology, weight_file)
        difference = np.abs(learned.delta.astype(np.float64) -
                            replay.delta.astype(np.float64))
        exact = np.array_equal(learned.delta, replay.delta)
        changed = int(np.count_nonzero(difference))
        max_difference = float(np.max(difference))
        support_identical = np.array_equal(
            np.flatnonzero(learned.delta), np.flatnonzero(replay.delta)
        )
        print("FROZEN_REAL_SOURCE_DIAGNOSTIC", json.dumps({
            "actual_nonzero": learned.modified_edges,
            "expected_nonzero": replay.modified_edges,
            "actual_imprints": learned.imprints,
            "expected_imprints": replay.imprints,
            "learned_state_exact": exact,
            "different_float32_entries": changed,
            "max_absolute_difference": max_difference,
            "same_nonzero_edge_support": support_identical,
        }), flush=True)
        if (replay.modified_edges != 28938
            or learned.modified_edges not in (28938,28939)
            or max_difference > 5e-7
            or learned.imprints != 951 or replay.imprints != 951):
            raise AssertionError(
                "Progressive Pilot13 failed bounded Pilot10 numerical source replay: "+
                f"modified={learned.modified_edges}/{replay.modified_edges}, "
                f"imprints={learned.imprints}/{replay.imprints}, "
                f"diff_positions={changed}, max_float32_abs_diff={max_difference}, "
                f"identical_nonzero_edge_support={support_identical}"
            )
        final_rows = rank_measured(
            learned, probes, ids, targets, learned_ids=set(ids)
        )
        earlier = source_case["oracle_only_case_results"]["multicue_source"]
        compare_original_pilot10_rows(
            final_rows, earlier["last_trained_cue"], "source positive"
        )
        absent_rows = rank_measured(
            learned, absent, ids, targets, learned_ids=set(ids)
        )
        compare_original_pilot10_rows(
            absent_rows, earlier["absent_episode_last_cue"], "absent episode"
        )
        expected_final = {
            "loaded_full_original_checkpoint_matches_weights_exactly": bool(exact),
            "loaded_full_original_checkpoint_within_predeclared_numerical_tolerance": True,
            "predeclared_max_float32_absolute_tolerance": 5e-7,
            "max_float32_absolute_difference": max_difference,
            "different_float32_positions": changed,
            "identical_source_active_edge_support": bool(support_identical),
            "original_159_cases_identical": True,
            "original_71_absent_cases_identical": True,
            "terminal_changed_edge_positions": learned.modified_edges,
            "source_original_checkpoint_sha256": SOURCE_PILOT10_CHECKPOINT_SHA,
            "prior_original_cases_sha256": SOURCE_PILOT10_SHA,
        }
    ckpts = {}
    if weights_dir:
        for name, model in conditions.items():
            dst = Path(weights_dir) / (name + ".npz")
            ckpts[name] = {"sha256": model.save(dst),
                           "path": str(dst),
                           "source_cue_encoder": meta["encoder_name"]}
            reread = DirectFlywireOverlay.load(model.topology, dst)
            if not np.array_equal(reread.delta, model.delta):
                raise AssertionError("Saved original-edge synaptic weights differ")
    if fingerprint(topology) != original_hash:
        raise AssertionError("Original publisher biological neurons/synapses changed")
    return {
        "study": "Pilot13 source-linked synaptic memory capacity and interference",
        "evidence_class": (
            "verified original full FlyWire v783"
            if real else "synthetic neural fixture NOT original biological data"
        ),
        "limitations": (
            "Only the original signed 256D BC01 lexical cue and narrative "
            "content features were tested. Trained familiar literal cues are "
            "not independent semantic queries. The neural inference receives "
            "only a cue string and a sparse learned synaptic overlay; memory "
            "event IDs/rankings and all targets reside in an EXTERNAL "
            "oracle-only evaluator, not the model. Staged training shows "
            "interference under a constant 317-event oracle candidate pool; "
            "it is not a claim of biological fly episodic memory. The "
            "rewired null exactly matches eligible binary in/out degree and "
            "edge-slot count, not every output neuron's contact-weighted "
            "incoming strength or final learned weight density."
        ),
        "source": {
            "canonical_event_count": len(memories),
            "canonical_episode_count": len({m.episode_id for m in memories}),
            "trained_event_count": len(ordered),
            "test_probe_event_count": len(probes),
            "validation_absent_count": len(validation),
            "test_absent_count": len(absent),
            "eligible_never_trained_fourth_literal_cues": len(test_eligible),
            "original_BC01_cache_sha256": meta["artifact_sha256"],
            "original_anatomical_array_sha256": original_hash,
            "original_anatomical_arrays_unchanged": fingerprint(topology) == original_hash,
            "original_neurons": len(topology.root_ids),
            "original_directed_pairs": len(topology.indices),
            "original_integer_synapse_contacts": int(
                topology.synapse_counts.sum(dtype=np.int64)
            ),
        },
        "predeclared": {
            "training_order": "original Pilot10 order_by_episode seed 20261008",
            "source_episode_split_seed": SEED,
            "probe_selection_seed": SEED + 991,
            "stages": CUTS,
            "anchor_early_train_events": ANCHORS,
            "latest_cohort": NEWEST,
            "fixed_oracle_candidate_count_at_all_stages": 317,
            "three_source_authored_cue_exposures_per_memory": True,
            "source_content_rate_per_exposure": 0.7 / 3,
            "fixed_historical_acceptance_threshold": REFERENCE_PILOT08_THRESHOLD,
            "graph_null_seed": REWIRE_SEED,
            "graph_null_swaps_per_edge": 2,
            "no_threshold_fitting_or_encoder_tuning_in_this_pilot": True,
        },
        "rewired_null": null_report,
        "frozen_original_pilot10_parity": expected_final,
        "learned_checkpoint_manifest": ckpts,
        "stages": stages,
        "longitudinal_early_memory_cases": history,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--topology", type=Path)
    g.add_argument("--synthetic-test", action="store_true")
    p.add_argument("--bc01-dir", type=Path, required=True)
    p.add_argument("--previous-real-cases", type=Path)
    p.add_argument("--previous-real-checkpoint", type=Path)
    p.add_argument("--cells-per-feature", type=int)
    p.add_argument("--weights-dir", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    topology = (
        synthetic_aggregated_fixture(n=8192, degree=16, seed=15)
        if args.synthetic_test else Topology.read(args.topology)
    )
    result = run(
        topology, args.bc01_dir, real=not args.synthetic_test,
        cells_per_feature=args.cells_per_feature or (8 if args.synthetic_test else 32),
        previous_cases=args.previous_real_cases,
        previous_checkpoint=args.previous_real_checkpoint,
        weights_dir=args.weights_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    for stage in result["stages"]:
        cols = [str(stage["trained_source_memories"])]
        for name in ARM_NAMES:
            n = stage["measured"][name]["groups"]["first16_anchors"]
            cols.append(name + ":" + str(n["correct_top1"]) + "/" + str(n["n"]))
        print("STAGE", " ".join(cols))
    print("RAW_ORIGINAL_CASES", args.output)


if __name__ == "__main__":
    main()
