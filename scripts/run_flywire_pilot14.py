#!/usr/bin/env python3
"""Pilot14: predeclared test of synaptic usage protection versus fixed baselines.

No semantic-encoder change, source prose in neural inference, event-ID routers
or post hoc tuning. Measures whether a local exposure-counter-mediated
stability rule slows early trained memory displacement, and whether it also
harms new learning, rejection and source content discrimination.
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
from pretorius_connectome.imprinting import order_by_episode
from pretorius_connectome.metaplastic14 import (
    UsageProtectedOverlay, MatchedSlotLinear,
)
from pretorius_connectome.rewire12 import (
    rewire_effective_edges, synthetic_aggregated_fixture,
)
from scripts.run_flywire_pilot13_capacity import (
    source_state, rank_measured, describe, rank_auc,
    compare_original_pilot10_rows, SEED, REWIRE_SEED,
    CUTS, SOURCE_PILOT10_SHA, SOURCE_PILOT10_CHECKPOINT_SHA,
)
from scripts.run_direct_flywire_imprint10 import binding_cues
from scripts.diagnose_flywire_imprint11 import unseen_literal_source_cue

CONDITIONS = (
    "original_baseline", "rewired_baseline",
    "original_beta1", "original_beta4", "rewired_beta1",
    "original_beta1_deranged", "matched_slot_linear",
)
EVALUATION_GROUPS = (
    "early16_familiar", "newest16_familiar",
    "original159_learned_familiar", "original31_truly_unseen",
    "absent71_episode",
)
ELIGIBLE_FOURTH_COUNT = 31
MAX_BASELINE_NUMERIC_TOLERANCE = 5e-7


def predeclared_models(original, rewired, *, group, n_eligible):
    common = {
        "seed": SEED, "cells_per_feature": group,
        "cue_features": 8, "content_features": 32,
        "rate": 0.7 / 3,
    }
    return {
        "original_baseline": DirectFlywireOverlay(original, **common),
        "rewired_baseline": DirectFlywireOverlay(rewired, **common),
        "original_beta1": UsageProtectedOverlay(
            original, protection_beta=1.0, **common
        ),
        "original_beta4": UsageProtectedOverlay(
            original, protection_beta=4.0, **common
        ),
        "rewired_beta1": UsageProtectedOverlay(
            rewired, protection_beta=1.0, **common
        ),
        "original_beta1_deranged": UsageProtectedOverlay(
            original, protection_beta=1.0, **common
        ),
        "matched_slot_linear": MatchedSlotLinear(
            slot_count=n_eligible, seed=REWIRE_SEED,
            cue_features=8, content_features=32, rate=0.7 / 3,
        ),
    }


def unseen_rows(model, records, ids, targets, learned_ids):
    """True fourth literal source-cue test, still EXTERNAL source oracle."""
    source_to_pos = {m: j for j, m in enumerate(ids)}
    result = []
    for record in records:
        cue = unseen_literal_source_cue(record)
        if not cue:
            raise ValueError("No originally untrained fourth source cue")
        output = model.infer(cue)
        similarities = targets @ output
        index = int(np.argmax(similarities))
        active = float(np.linalg.norm(output)) > 1e-10
        guess = ids[index] if active else None
        target = source_to_pos.get(record.event_id)
        if target is None:
            raise ValueError("Original positive source event absent from codebook")
        other = float(np.max(np.concatenate(
            (similarities[:target], similarities[target + 1:])
        )))
        own = float(similarities[target])
        result.append({
            "event_id": record.event_id,
            "cue_kind": "genuinely_untrained_fourth_original_literal_source",
            "was_imprinted_by_this_stage": record.event_id in learned_ids,
            "has_oracle_target": True,
            "predicted_event_id": guess,
            "correct_top1": bool(guess == record.event_id),
            "best_cosine": round(float(similarities[index]), 7) if active else 0.,
            "self_content_cosine": round(own, 7),
            "best_competing_cosine": round(other, 7),
            "true_content_margin": round(own - other, 7),
            "neural_response_nonzero": active,
            "accepted_at_unchanged_pilot08_threshold": bool(
                active and float(similarities[index]) > .0082783
            ),
        })
    return result


def matched_group_stats(rows):
    summary = describe(rows)
    return summary


def parity_with_original_pilot10(model, graph, cases, weights, source_ids,
                                 targets, positive, negatives):
    """No tolerance on identity decisions; finite original float allowance.

    Some source-verified previous GitHub runners differ in <=2.38e-7 due
    to floating roundoff. Do NOT silently call that byte-identical.
    """
    if sha256(Path(weights).read_bytes()).hexdigest() != SOURCE_PILOT10_CHECKPOINT_SHA:
        raise ValueError("Wrong original historical learned source checkpoint")
    if sha256(Path(cases).read_bytes()).hexdigest() != SOURCE_PILOT10_SHA:
        raise ValueError("Wrong original historical original-case evidence")
    old = DirectFlywireOverlay.load(graph, weights)
    if old.imprints != 951 or model.imprints != 951:
        raise AssertionError("Original trained exposure budget not reproduced")
    delta = np.abs(model.delta.astype(np.float64) - old.delta.astype(np.float64))
    maxdiff = float(np.max(delta))
    changed = int(np.count_nonzero(delta))
    if maxdiff > MAX_BASELINE_NUMERIC_TOLERANCE:
        raise AssertionError("Historical original source baseline numerical deviation exceeds predeclared roundoff bound")
    prior = json.loads(Path(cases).read_text(encoding="utf-8"))
    rows = rank_measured(model, positive, source_ids, targets,
                         learned_ids=set(source_ids))
    unknown = rank_measured(model, negatives, source_ids, targets,
                            learned_ids=set(source_ids))
    ref = prior["oracle_only_case_results"]["multicue_source"]
    compare_original_pilot10_rows(rows, ref["last_trained_cue"],
                                  "Pilot14 frozen original positive events")
    compare_original_pilot10_rows(unknown, ref["absent_episode_last_cue"],
                                  "Pilot14 frozen original episode-absent events")
    return {
        "historical_checkpoint_sha256": SOURCE_PILOT10_CHECKPOINT_SHA,
        "original_full_weight_array_byte_identical": bool(changed == 0),
        "original_full_weight_array_max_float_abs_difference": maxdiff,
        "original_full_weight_array_different_float_positions": changed,
        "source_case_predictions_same_all_159": True,
        "absent_case_predictions_same_all_71": True,
        "tolerance_for_numerical_roundoff_predeclared": MAX_BASELINE_NUMERIC_TOLERANCE,
        "original_learned_951_presentations": True,
    }


def run(graph, bc01_dir, *, real=False, group=32, source_cases=None,
        source_checkpoint=None, weights_dir=None):
    (source_sha, memories, meta, bc01, positions, train, positives,
     validation, negatives, targets) = source_state(graph, bc01_dir, real)
    ids = [m.event_id for m in train]
    chosen_unseen = [m for m in positives if unseen_literal_source_cue(m)]
    if len(chosen_unseen) != ELIGIBLE_FOURTH_COUNT:
        raise AssertionError("Original unseen source cue eligibility changed")
    expected = DirectFlywireOverlay(
        graph, seed=SEED, cells_per_feature=group, rate=0.7/3
    )
    rewired, rewiring = rewire_effective_edges(
        graph, expected.pre_cells, expected.post_cells,
        seed=REWIRE_SEED, swaps_per_edge=2
    )
    n_slots = rewiring["originally_eligible_edges"]
    models = predeclared_models(graph, rewired, group=group, n_eligible=n_slots)
    for model in models.values():
        if not isinstance(model, MatchedSlotLinear):
            if (not np.array_equal(model.pre_cells, expected.pre_cells)
                or not np.array_equal(model.post_cells, expected.post_cells)):
                raise AssertionError("Feature neuron populations differ among source models")
    # Frozen shift by 1/3 training corpus, exactly Pilot12/Pilot13.
    shift = np.roll(np.arange(len(train)), max(1, len(train) // 3))
    anchor = train[:16]
    stages = []
    for j, cut in enumerate(CUTS):
        if j:
            for idx in range(CUTS[j-1], cut):
                record = train[idx]
                shuffled_record = train[int(shift[idx])]
                if record.event_id == shuffled_record.event_id:
                    raise AssertionError("Deranged condition hit own target")
                for name, model in models.items():
                    target_record = (shuffled_record
                                     if name.endswith("_deranged") else record)
                    content = bc01[positions[target_record.event_id]]
                    for literal_cue in binding_cues(record):
                        model.imprint(literal_cue, content)
        learned = set(ids[:cut])
        subsets = {
            "early16_familiar": anchor,
            "newest16_familiar": train[max(0, cut - 16):cut],
            "original159_learned_familiar": [
                m for m in positives if m.event_id in learned
            ],
            "original31_truly_unseen": [
                m for m in chosen_unseen if m.event_id in learned
            ],
            "absent71_episode": negatives,
        }
        stage = {
            "trained_events": cut,
            "presentations_per_condition": 3 * cut,
            "arms": {},
        }
        for name, model in models.items():
            results = {}
            summaries = {}
            for group_name in EVALUATION_GROUPS:
                records = subsets[group_name]
                rows = (
                    unseen_rows(model, records, ids, targets, learned)
                    if group_name == "original31_truly_unseen"
                    else rank_measured(model, records, ids, targets,
                                       learned_ids=learned)
                )
                results[group_name] = rows
                summaries[group_name] = matched_group_stats(rows)
            if model.imprints != cut * 3:
                raise AssertionError("Equal cue presentation counts required")
            if not isinstance(model, MatchedSlotLinear):
                model.assert_original_unchanged()
            stage["arms"][name] = {
                "cases": results, "summaries": summaries,
                "cumulative_edge_update_operations": model.edge_update_events,
                "nonzero_learned_weights": model.modified_edges,
                "total_trainable_parameter_slots": (
                    n_slots if not isinstance(model, MatchedSlotLinear)
                    else model.slot_count
                ),
                "biological_edge_exposure_distribution": (
                    model.exposure_diagnostics
                    if isinstance(model, UsageProtectedOverlay) else None
                ),
                "oracle_known_vs_absent_auc": rank_auc(
                    results["original159_learned_familiar"],
                    results["absent71_episode"]
                ),
            }
        stages.append(stage)

    parity = None
    if real:
        if source_cases is None or source_checkpoint is None:
            raise ValueError("Real original FlyWire requires pinned Pilot10 baseline cases and checkpoint")
        parity = parity_with_original_pilot10(
            models["original_baseline"], graph, source_cases,
            source_checkpoint, ids, targets, positives, negatives,
        )
    checks = {}
    if weights_dir:
        for name, model in models.items():
            path = Path(weights_dir) / (name + ".npz")
            if isinstance(model, UsageProtectedOverlay):
                digest = model.save_protected(path)
                reread = UsageProtectedOverlay.load_protected(model.topology, path)
                if (not np.array_equal(reread.delta, model.delta)
                    or not np.array_equal(reread.edge_exposures,
                                          model.edge_exposures)):
                    raise AssertionError("Protected source synapse archive did not exactly replay")
            elif isinstance(model, MatchedSlotLinear):
                digest = model.save_linear(path)
                with np.load(path, allow_pickle=False) as saved:
                    if (not np.array_equal(saved["weights"], model.weights)
                        or not np.array_equal(saved["mask"], model.mask)):
                        raise AssertionError("Nonneural comparison weights/mask failed replay")
            else:
                digest = model.save(path)
                reread = DirectFlywireOverlay.load(model.topology, path)
                if not np.array_equal(reread.delta, model.delta):
                    raise AssertionError("Original-rule source synapse weights failed checkpoint replay")
            checks[name] = {
                "sha256": digest, "path": str(path),
                "model_kind": ("parameter_matched_non_neural_linear"
                               if isinstance(model, MatchedSlotLinear)
                               else "usage_protected_synapse"
                               if isinstance(model, UsageProtectedOverlay)
                               else "original_rule_synapse"),
            }
    if fingerprint(graph) != source_sha:
        raise AssertionError("Publisher's full original anatomical arrays were mutated")
    return {
        "study": "Pilot14 usage-dependent synaptic stability and matched-slot linear source control",
        "source_type": ("verified complete original FlyWire v783"
                        if real else "synthetic numerical mask only, NOT original fly data"),
        "scientific_limits": (
            "All arms use original BC01 signed lexical input and target code. "
            "Usage counts are local numeric edge state, not a biological fly "
            "metaplasticity law. A pretrained semantic encoder and independently "
            "authored paraphrase test are NOT used. The linear control has "
            "matched scalar parameter slot COUNT, not connectivity, numeric "
            "contact strength or effective learning-rate budget. All source "
            "event identities, target vector rankings and absence thresholds "
            "are EXTERNAL oracle-only evaluation, never the inference model. "
            "Fixed old acceptance threshold measures false accepts only as a "
            "historical diagnostic, not a calibrated autobiographical truth gate."
        ),
        "source": {
            "canonical_memories": len(memories),
            "original_train_memories": len(train),
            "original_test_familiar": len(positives),
            "original_test_absent_episode": len(negatives),
            "original_validation_absent": len(validation),
            "original_untrained_source_fourth_cues": len(chosen_unseen),
            "original_bc01_artifact_sha256": meta["artifact_sha256"],
            "original_anatomical_array_sha256": source_sha,
            "original_anatomy_unchanged": fingerprint(graph) == source_sha,
            "root_neurons": len(graph.root_ids),
            "directed_pairs": len(graph.indices),
            "original_integer_contacts": int(graph.synapse_counts.sum(dtype=np.int64)),
        },
        "preregistered": {
            "training_stages": CUTS, "fixed_external_candidate_count": 317,
            "cue_exposures_per_memory": 3, "learning_rate_per_cue": 0.7/3,
            "all_input_top_features": 8, "all_content_top_features": 32,
            "protected_beta_values": [1.0, 4.0],
            "baseline_beta": 0.0,
            "linear_mask_seed": REWIRE_SEED,
            "contact_strengths_NOT_matched_in_linear_control": True,
            "graph_null_seed": REWIRE_SEED,
            "original_159_test_not_used_for_tuning": True,
            "original_71_absent_test_not_used_for_tuning": True,
        },
        "rewired_null": rewiring,
        "original_terminal_baseline_replay": parity,
        "complete_learned_checkpoints": checks,
        "stage_case_evidence": stages,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--topology", type=Path)
    g.add_argument("--synthetic-test", action="store_true")
    p.add_argument("--bc01-dir", type=Path, required=True)
    p.add_argument("--source-pilot10-cases", type=Path)
    p.add_argument("--source-pilot10-checkpoint", type=Path)
    p.add_argument("--cells-per-feature", type=int)
    p.add_argument("--weights-dir", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    graph = (synthetic_aggregated_fixture(n=8192, degree=16, seed=15)
             if a.synthetic_test else Topology.read(a.topology))
    result = run(
        graph, a.bc01_dir, real=not a.synthetic_test,
        group=a.cells_per_feature or (8 if a.synthetic_test else 32),
        source_cases=a.source_pilot10_cases,
        source_checkpoint=a.source_pilot10_checkpoint,
        weights_dir=a.weights_dir,
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    for stage in result["stage_case_evidence"]:
        part = []
        for name in CONDITIONS:
            group = stage["arms"][name]["summaries"]["early16_familiar"]
            part.append(f"{name}:{group['correct_top1']}/{group['n']}")
        print("STAGE", stage["trained_events"], " ".join(part), flush=True)
    print("PILOT14_ORIGINAL_SOURCE_CASES", a.output)


if __name__ == "__main__":
    main()
