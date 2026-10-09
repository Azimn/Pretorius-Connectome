#!/usr/bin/env python3
"""Pilot12: source-linked 2x2 frozen-encoder x degree-rewired FlyWire assay.

All conditions re-train the same original 317 autobiographical events using
exactly the same 3 explicitly source-authored cues and rate 0.7/3; only the
cue encoder and whether the fixed learnable directed edge support is original
or degree-preserving rewired differ. Content target BC01 remains fixed.

Output memory IDs are ALWAYS external oracle-only evaluations. MiniLM is a
pretrained *non-fly* sentence encoder, never evidence of semantics stored in
synaptic weights. The original full biological CSR is read-only.
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
    fingerprint, select_features,
)
from pretorius_connectome.imprinting import load_v12, order_by_episode
from pretorius_connectome.pilot02 import episode_split
from pretorius_connectome.rewire12 import rewire_effective_edges
from pretorius_connectome.semantic_cue12 import (
    CueOnlyOverlay, FrozenMiniLMCues, MODEL, REVISION, ONNX_NAME,
    PROJECTION_SEED, projection_sha256,
)
from pretorius_connectome.shared_features_bc01 import load_bc01_cache, SCHEMA
from pretorius_connectome.shared_memory import read_l1
from scripts.run_direct_flywire_imprint08 import (
    SOURCE, SIDECARS, L1_DIR, REAL_SHA, surfaces,
)
from scripts.run_direct_flywire_imprint10 import (
    binding_cues, cue_drop_last_token, choose_probes,
    REFERENCE_PILOT08_THRESHOLD,
)
from scripts.diagnose_flywire_imprint11 import (
    unseen_literal_source_cue, calibrated_gate,
)
from scripts.run_imprinting_pilot import verify_sources

SEED = 31
REWIRE_SEED = 73
MODEL_IDS = (
    "real_bc01",
    "degree_rewired_bc01",
    "real_minilm",
    "degree_rewired_minilm",
    "real_minilm_shuffled",
    "degree_rewired_minilm_shuffled",
)
CASE_TYPES = (
    "familiar_trained_last",
    "controlled_last_cue_token_deletion",
    "paired_familiar_for_unseen",
    "genuinely_untrained_fourth_source_cue",
    "absent_heldout_episode",
)
SOURCE_PILOT10_SHA = "ce2d463a72fa58a1e65265cf50f961f9e585610d058b9701d26791e033e7cea8"


def cue_for(case_type: str, memory):
    if case_type in ("familiar_trained_last", "paired_familiar_for_unseen",
                     "absent_heldout_episode"):
        return binding_cues(memory)[-1]
    if case_type == "controlled_last_cue_token_deletion":
        return cue_drop_last_token(binding_cues(memory)[-1])
    if case_type == "genuinely_untrained_fourth_source_cue":
        return unseen_literal_source_cue(memory)
    raise ValueError("Unrecognized cue probe kind")


def measured_rows(model, records, event_ids, candidate_targets, case_type):
    """The oracle table is used strictly OUTSIDE model.infer()."""
    id_to_index = {event_id: i for i, event_id in enumerate(event_ids)}
    outputs = []
    for memory in records:
        cue = cue_for(case_type, memory)
        if not cue:
            raise ValueError("Never-presented literal cue absent in selected source record")
        response = model.infer(cue)  # No memory text or source identity passed.
        similarities = candidate_targets @ response
        active = bool(np.linalg.norm(response) > 1e-10)
        best = int(np.argmax(similarities))
        predicted = event_ids[best] if active else None
        index = id_to_index.get(memory.event_id)
        row = {
            "event_id": memory.event_id,
            "cue_type": case_type,
            "known_event": index is not None,
            "predicted": predicted,
            "correct": bool(predicted == memory.event_id) if index is not None else None,
            "best_cosine": round(float(similarities[best]), 7) if active else 0.0,
            "target_cosine": (
                round(float(similarities[index]), 7)
                if index is not None else None),
            "nonzero_model_response": active,
        }
        if case_type == "genuinely_untrained_fourth_source_cue":
            source_coords = model._input(cue)
            exposures = [model._input(q) for q in binding_cues(memory)]
            used = set().union(
                *(set(np.flatnonzero(v).tolist()) for v in exposures))
            row["same_encoder_trained_input_overlap"] = len(
                set(np.flatnonzero(source_coords).tolist()) & used
            )
            row["largest_same_encoder_trained_cue_cosine"] = round(
                max(float(source_coords @ v) for v in exposures), 7
            )
        outputs.append(row)
    return outputs


def summary(rows, threshold):
    if not rows:
        return {"n": 0, "eligible_source_cues": False}
    known = all(r["known_event"] for r in rows)
    if any(r["known_event"] != known for r in rows):
        raise AssertionError("Event presence labels mixed in one test")
    accepted = [r["nonzero_model_response"] and r["best_cosine"] > threshold
                for r in rows]
    result = {
        "n": len(rows),
        "nonzero_responses": sum(r["nonzero_model_response"] for r in rows),
        "accepted_fraction": round(sum(accepted) / len(rows), 6),
        "mean_best_cosine": round(float(np.mean([r["best_cosine"] for r in rows])), 6),
    }
    if known:
        result["correct_top1"] = round(
            sum(r["correct"] for r in rows) / len(rows), 6
        )
        result["correct_and_accepted"] = round(
            sum(r["correct"] and accept for r, accept in zip(rows, accepted))
            / len(rows), 6
        )
        result["mean_target_cosine"] = round(float(
            np.mean([r["target_cosine"] for r in rows])), 6
        )
    else:
        result["absent_false_acceptance"] = result["accepted_fraction"]
    if any("same_encoder_trained_input_overlap" in r for r in rows):
        result["no_own_trained_feature_overlap"] = sum(
            r["same_encoder_trained_input_overlap"] == 0 for r in rows
        )
    return result


def train_condition(topology, encoder, ordered, position, bc01,
                    *, shuffled: bool, cells_per_feature: int):
    model = CueOnlyOverlay(
        topology, cue_encoder=encoder,
        seed=SEED, cells_per_feature=cells_per_feature,
        cue_features=8, content_features=32, rate=0.7/3,
    )
    derangement = np.roll(np.arange(len(ordered)),
                          max(1, len(ordered) // 3))
    for i, memory in enumerate(ordered):
        content_memory = ordered[int(derangement[i])] if shuffled else memory
        if shuffled and content_memory.event_id == memory.event_id:
            raise AssertionError("Not an actual cue/content derangement")
        content_vec = bc01[position[content_memory.event_id]]
        for cue in binding_cues(memory):
            model.imprint(cue, content_vec)
    if model.imprints != len(ordered) * 3:
        raise AssertionError("All conditions need exactly 951 presentations")
    return model


def run(topology, bc01_dir, *, real=False, cells_per_feature=32,
        output_weights=None, source_pilot10=None, semantic_backend=None):
    verify_sources(SOURCE, SIDECARS)
    source_sha = fingerprint(topology)
    if real and (
        len(topology.root_ids) != 139255
        or len(topology.indices) != 15091983
        or int(topology.synapse_counts.sum(dtype=np.int64)) != 54492922
        or source_sha != REAL_SHA
    ):
        raise ValueError("Original complete FlyWire v783 biological arrays mismatch")
    originals = load_v12(SOURCE, SIDECARS)
    frozen = read_l1(L1_DIR / "pretorius_l1_v1.jsonl.gz",
                     L1_DIR / "manifest.json")
    if (len(originals) != len(frozen) or len(frozen) != 450
        or any(a.event_id != b["event_id"] or
               a.episode_id != b["episode_id"] or
               a.memory_text != b["memory_text"]
               for a, b in zip(originals, frozen))):
        raise ValueError("Original canonical L0 and L1 evidence mismatch")
    meta, bc01 = load_bc01_cache(bc01_dir)
    if (meta["schema_version"] != SCHEMA or bc01.shape != (450,256)
        or meta["record_ids_ordered"] != [m.event_id for m in originals]):
        raise ValueError("Source BC01 256D content feature mismatch")
    train, validation, absent = episode_split(originals, SEED)
    ordered = order_by_episode(train.copy(), seed=20261008)
    tests = choose_probes(ordered, SEED)
    test_ids = {m.event_id for m in tests}
    calibration_positive = [m for m in ordered if m.event_id not in test_ids]
    genuinely_unseen = [m for m in tests if unseen_literal_source_cue(m)]
    if (len(train),len(validation),len(absent),len(tests),
        len(calibration_positive)) != (317,62,71,159,158):
        raise AssertionError("Original evaluation partitions changed")
    position = {m.event_id: i for i,m in enumerate(originals)}
    candidate_vectors = np.stack([
        select_features(bc01[position[m.event_id]], top_k=32)
        for m in ordered
    ])
    ids = [m.event_id for m in ordered]

    # Reference feature-neuron population is fixed, seed31, for ALL graphs.
    template = CueOnlyOverlay(topology, seed=SEED,
                              cells_per_feature=cells_per_feature, rate=.7/3)
    null_graph, null_meta = rewire_effective_edges(
        topology, template.pre_cells, template.post_cells,
        seed=REWIRE_SEED, swaps_per_edge=2,
    )
    if fingerprint(topology) != source_sha:
        raise AssertionError("Original source graph mutated to produce rewired null")
    if null_meta["originally_eligible_edges"] != null_meta["rewired_eligible_edges"]:
        raise AssertionError("Unequal learnable directed synaptic support")

    if semantic_backend is None:
        semantic_backend = FrozenMiniLMCues()
    # Batch local CPU transformer prewarm only source literal cues. The cache
    # contains NO event IDs, targets, narratives or retrieved memory records.
    required_texts = []
    for m in ordered:
        required_texts.extend(binding_cues(m))
    for m in tests:
        required_texts.append(cue_drop_last_token(binding_cues(m)[-1]))
    for m in genuinely_unseen:
        required_texts.append(unseen_literal_source_cue(m))
    for m in validation + absent:
        required_texts.append(binding_cues(m)[-1])
    semantic_backend.prewarm(required_texts)
    initial_model_forwards = semantic_backend.forward_batches
    if not hasattr(semantic_backend, "onnx_sha256"):
        raise ValueError("Semantic encoder must expose pinned ONNX weights SHA-256")

    groups = {
        "familiar_trained_last": tests,
        "controlled_last_cue_token_deletion": tests,
        "paired_familiar_for_unseen": genuinely_unseen,
        "genuinely_untrained_fourth_source_cue": genuinely_unseen,
        "absent_heldout_episode": absent,
    }
    all_results, stats, checkpoints = {}, {}, {}
    for name in MODEL_IDS:
        use_null = name.startswith("degree_rewired")
        use_semantic = "minilm" in name
        shuffled = name.endswith("_shuffled")
        graph = null_graph if use_null else topology
        encoder = semantic_backend if use_semantic else None
        model = train_condition(
            graph, encoder, ordered, position, bc01,
            shuffled=shuffled, cells_per_feature=cells_per_feature,
        )
        if (model.pre_cells.shape != template.pre_cells.shape
            or not np.array_equal(model.pre_cells, template.pre_cells)
            or not np.array_equal(model.post_cells, template.post_cells)):
            raise AssertionError("Graph conditions do not share fixed neurons")
        if output_weights is not None and name in (
            "real_minilm", "degree_rewired_minilm"):
            target = Path(output_weights) / (name + ".npz")
            checkpoints[name] = {
                "sha256": model.save(target),
                "relative_path": str(target),
                "external_encoder": MODEL + "@" + REVISION,
                "encoder_projection_sha256": projection_sha256(),
            }
            # Reparse numerical state, attach the PINNED frozen same encoder.
            reread = CueOnlyOverlay.load(graph, target)
            reread.cue_encoder = semantic_backend
            for probe in tests[:3]:
                if not np.array_equal(
                    reread.infer(binding_cues(probe)[-1]),
                    model.infer(binding_cues(probe)[-1])):
                    raise AssertionError("Source-bound neural checkpoint changed on reload")

        known_calibration = measured_rows(
            model, calibration_positive, ids, candidate_vectors,
            "familiar_trained_last"
        )
        absent_validation = measured_rows(
            model, validation, ids, candidate_vectors,
            "absent_heldout_episode"
        )
        # Uses original Pilot11 rejection protocol, *no* test leakage.
        gate = calibrated_gate(known_calibration, absent_validation)
        records, summaries = {}, {}
        for kind, cases in groups.items():
            output = measured_rows(model, cases, ids, candidate_vectors, kind)
            records[kind] = output
            summaries[kind] = {
                "original_pilot08_threshold": summary(
                    output, REFERENCE_PILOT08_THRESHOLD
                ),
                "validation_only_threshold": summary(
                    output, gate["threshold"]
                ),
            }
        all_results[name] = records
        stats[name] = {
            "learned_edge_positions": model.modified_edges,
            "synaptic_edge_update_events": model.edge_update_events,
            "presentations": model.imprints,
            "trainable_original_support_slots": null_meta[
                "originally_eligible_edges"],
            "encoder": "pinned-MiniLM-ONNX" if use_semantic else "original-BC01-lexical",
            "graph": "binary-degree-rewired-null" if use_null else "original-v783",
            "target_labels_shuffled": shuffled,
            "gate": gate,
            "summaries": summaries,
        }
        model.assert_original_unchanged()
        del model

    # Reproducibility baseline: original biological real BC01 source condition
    # must reproduce archived 159 positive and 71 absent numerical decisions.
    if real:
        if source_pilot10 is None:
            raise ValueError("Source original real Pilot10 case evidence is required")
        archived = Path(source_pilot10)
        if sha256(archived.read_bytes()).hexdigest() != SOURCE_PILOT10_SHA:
            raise ValueError("Frozen original real Pilot10 source case SHA differs")
        reference = json.loads(archived.read_text(encoding="utf-8"))
        base = reference["oracle_only_case_results"]["multicue_source"]
        for test_case, old_name in [
            ("familiar_trained_last", "last_trained_cue"),
            ("absent_heldout_episode", "absent_episode_last_cue"),
        ]:
            now_rows = all_results["real_bc01"][test_case]
            old_rows = base[old_name]
            if len(now_rows) != len(old_rows):
                raise AssertionError("Baseline source case count changed")
            for new, old in zip(now_rows, old_rows):
                if (new["event_id"] != old["event_id"]
                    or new["predicted"] != old["predicted"]
                    or new["correct"] != old["correct"]
                    or abs(new["best_cosine"]-old["best_cosine"]) > 2e-6):
                    raise AssertionError(
                        "New Pilot12 BC01 benchmark lost source-bound "
                        "Pilot10 identical-cue replay at " + new["event_id"]
                    )
    if fingerprint(topology) != source_sha:
        raise AssertionError("Publisher's original biological graph has changed")
    return {
        "study": "Pretorius Pilot12 frozen semantic cue encoder vs degree-rewired real FlyWire",
        "status": ("original source-verified full FlyWire v783" if real
                   else "synthetic graph only, NOT biological evidence"),
        "scientific_limitations": (
            "Pinned pretrained MiniLM contributes EXTERNAL language prior. "
            "BC01 narrative content targets, learning rule and 3 source-cue "
            "training exposures are unchanged. Neural infer only outputs "
            "256D signed features from original-edge constrained learned "
            "weights. Source IDs and memory rankings use EXTERNAL oracle-only "
            "candidate vectors; model cannot reconstruct narrative prose. "
            "Untrained fourth cue is original unreviewed source surface, not "
            "an independently authored semantic paraphrase. Rewiring controls "
            "binary eligible source/destination degrees and synaptic slots, "
            "but not contact-weighted incoming target strength or learned "
            "nonzero weight counts. No biological STDP, genuine recall, "
            "or character/personhood is claimed."
        ),
        "source": {
            "source_memories": len(originals),
            "episode_count": len({m.episode_id for m in originals}),
            "train": len(train), "validation_absent": len(validation),
            "test_absent": len(absent), "test_known": len(tests),
            "calibration_known": len(calibration_positive),
            "eligible_never_seen_cue_events": len(genuinely_unseen),
            "BC01_encoder": meta["encoder_name"],
            "BC01_artifact_sha256": meta["artifact_sha256"],
            "original_graph_array_sha256": source_sha,
            "original_graph_neurons": len(topology.root_ids),
            "original_directed_edges": len(topology.indices),
            "original_integer_synaptic_contacts": int(
                topology.synapse_counts.sum(dtype=np.int64)),
            "original_graph_unchanged": fingerprint(topology) == source_sha,
        },
        "frozen_pretrained_cue_encoder": {
            "model": MODEL,
            "revision": REVISION,
            "backend": ONNX_NAME,
            "onnx_sha256": semantic_backend.onnx_sha256,
            "project_384_to_256_sha256": projection_sha256(),
            "projection_seed": PROJECTION_SEED,
            "batch_calls_after_prewarm": initial_model_forwards,
            "all_cue_cache_has_no_labels_or_targets": True,
        },
        "rewired_degree_matched_control": null_meta,
        "model_checkpoint_metadata": checkpoints,
        "conditions": stats,
        "per_case_external_oracle_evidence": all_results,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    graph = p.add_mutually_exclusive_group(required=True)
    graph.add_argument("--topology", type=Path)
    graph.add_argument("--synthetic-test", action="store_true")
    p.add_argument("--bc01-dir", type=Path, required=True)
    p.add_argument("--source-pilot10", type=Path)
    p.add_argument("--cells-per-feature", type=int)
    p.add_argument("--output-weights", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    topo = (Topology.synthetic(n=8192, degree=16, seed=15)
            if a.synthetic_test else Topology.read(a.topology))
    result = run(
        topo, a.bc01_dir, real=not a.synthetic_test,
        source_pilot10=a.source_pilot10,
        cells_per_feature=a.cells_per_feature or (8 if a.synthetic_test else 32),
        output_weights=a.output_weights,
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    for key, condition in result["conditions"].items():
        familiar = condition["summaries"]["familiar_trained_last"][
            "original_pilot08_threshold"]
        never_seen = condition["summaries"]["genuinely_untrained_fourth_source_cue"][
            "original_pilot08_threshold"]
        absent = condition["summaries"]["absent_heldout_episode"][
            "validation_only_threshold"]
        print(f"{key:31s} familiar_top1={familiar.get('correct_top1')} "
              f"never_seen_top1={never_seen.get('correct_top1')} "
              f"absent_false_acceptance_valgate={absent.get('absent_false_acceptance')} "
              f"changed_edges={condition['learned_edge_positions']}")
    print("RAW_CASES", a.output)


if __name__ == "__main__":
    main()
