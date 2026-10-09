#!/usr/bin/env python3
"""Pilot 10: learn several independent literal cues for one content vector.

This pilot isolates explicit multi-cue binding versus 3 equal-budget repetitions
of Pilot08's concatenated cue. It does NOT test unseen semantic paraphrases.
DirectFlywireOverlay.infer receives ONLY a cue, no autobiography, source event
ID, document retrieval index or external decoder targets. The original FlyWire
CSR and all integer anatomical synapse counts remain untouched.
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
    SOURCE, SIDECARS, L1_DIR, REAL_SHA, train_cue, surfaces,
)
from scripts.run_imprinting_pilot import verify_sources

CONDITIONS = ("multicue_source", "repeated_concat_control",
              "multicue_shuffled_targets", "zero_overlay")
PROBES = ("first_trained_cue", "last_trained_cue",
          "last_trained_cue_drop_token", "absent_episode_last_cue")
REFERENCE_PILOT08_THRESHOLD = 0.0082783
PRESENTATIONS_PER_EVENT = 3
ORIGINAL_RATE = 0.7


def binding_cues(memory) -> tuple[str, str, str]:
    """Three source-authored, not algorithmically generated cue exposures.

    448/450 source events have >=3 literal cues; 2 have exactly two. For the
    two-cue edge cases repeat the last cue, disclose this in training metadata.
    More-than-three cue lists sample first, middle and last deterministically.
    """
    forms = surfaces(memory)
    if len(forms) < 2:
        raise ValueError("source record lacks two literal recall cues")
    if len(forms) == 2:
        return forms[0], forms[1], forms[1]
    return forms[0], forms[len(forms) // 2], forms[-1]


def cue_drop_last_token(cue: str) -> str:
    """Controlled lexical deletion, not semantic or human paraphrase."""
    units = cue.split()
    if len(units) >= 2:
        return " ".join(units[:-1])
    if len(cue) > 1:
        return cue[:-1]
    return cue + "x"


def choose_probes(ordered: list, seed: int) -> list:
    rng = np.random.default_rng(seed + 991)
    selection = rng.permutation(len(ordered))
    return [ordered[int(i)] for i in selection[max(1, len(selection) // 2):]]


def score(model: DirectFlywireOverlay, records: list, ids: list[str],
          targets: np.ndarray, *, scenario: str) -> list[dict]:
    """Candidate ID lookup is explicitly offline and outside neural inference."""
    id_position = {key: i for i, key in enumerate(ids)}
    output = []
    for row in records:
        if scenario == "first_trained_cue":
            prompt = binding_cues(row)[0]
        elif scenario == "last_trained_cue":
            prompt = binding_cues(row)[-1]
        elif scenario == "last_trained_cue_drop_token":
            prompt = cue_drop_last_token(binding_cues(row)[-1])
        elif scenario == "absent_episode_last_cue":
            prompt = binding_cues(row)[-1]
        else:
            raise ValueError("Unknown cue test")
        response = model.infer(prompt)
        norm = float(np.linalg.norm(response))
        sim = targets @ response
        best = int(np.argmax(sim))
        predicted = ids[best] if norm > 1e-10 else None
        known = row.event_id in id_position
        output.append({
            "event_id": row.event_id,
            "scenario": scenario,
            "predicted": predicted,
            "known": known,
            "correct": bool(predicted == row.event_id) if known else None,
            "best_cosine": round(float(sim[best]), 7) if predicted else 0.0,
            "target_cosine": (
                round(float(sim[id_position[row.event_id]]), 7) if known else None
            ),
            "response_norm": round(norm, 7),
            "accepted_at_frozen_pilot08_threshold": (
                bool(predicted is not None and
                     float(sim[best]) > REFERENCE_PILOT08_THRESHOLD)
            ),
        })
    return output


def summarize(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("Cannot summarize empty challenge")
    known = all(x["known"] for x in rows)
    if any(x["known"] != known for x in rows):
        raise ValueError("Known and unknown events cannot share a summary")
    result = {
        "cases": len(rows),
        "nonzero_responses": sum(x["response_norm"] > 0 for x in rows),
        "acceptance_rate": round(sum(x["accepted_at_frozen_pilot08_threshold"]
                                     for x in rows) / len(rows), 6),
        "mean_best_cosine": round(float(np.mean(
            [x["best_cosine"] for x in rows])), 6),
    }
    if known:
        result.update({
            "correct_top1": round(sum(x["correct"] for x in rows) / len(rows), 6),
            "correct_and_accepted": round(sum(
                x["correct"] and x["accepted_at_frozen_pilot08_threshold"]
                for x in rows) / len(rows), 6),
            "mean_target_cosine": round(float(np.mean(
                [x["target_cosine"] for x in rows])), 6),
        })
    else:
        result["absent_false_acceptance"] = result["acceptance_rate"]
    return result


def benchmark(topology: Topology, bc01_dir: Path, *, seed: int = 31,
              real: bool = False, cells_per_feature: int = 32,
              checkpoint: Path | None = None) -> dict:
    verify_sources(SOURCE, SIDECARS)
    original = fingerprint(topology)
    if real and (
        len(topology.root_ids) != 139255 or len(topology.indices) != 15091983
        or original != REAL_SHA
        or int(topology.synapse_counts.sum(dtype=np.int64)) != 54492922
    ):
        raise ValueError("Real experiment must use checksum-verified original v783")
    corpus = load_v12(SOURCE, SIDECARS)
    l1 = read_l1(L1_DIR / "pretorius_l1_v1.jsonl.gz", L1_DIR / "manifest.json")
    if (len(corpus) != len(l1) or len(corpus) != 450
        or any(m.event_id != x["event_id"] or m.episode_id != x["episode_id"]
               or m.memory_text != x["memory_text"]
               for m, x in zip(corpus, l1))):
        raise ValueError("Original frozen L0 and L1 autobiographies differ")
    meta, vectors = load_bc01_cache(bc01_dir)
    if (meta.get("schema_version") != SCHEMA or vectors.shape != (450, 256)
        or meta.get("record_ids_ordered") != [m.event_id for m in corpus]):
        raise ValueError("Incompatible BC01 canonical source lexical features")
    train, validation, test = episode_split(corpus, seed)
    ordered = order_by_episode(train.copy(), seed=20261008)
    selected = choose_probes(ordered, seed)
    loc = {m.event_id: i for i, m in enumerate(corpus)}
    targets = np.stack([
        select_features(vectors[loc[m.event_id]], top_k=32)
        for m in ordered
    ])
    event_ids = [m.event_id for m in ordered]
    samples_with_two_cues = sum(len(surfaces(m)) == 2 for m in ordered)
    learning_rate = ORIGINAL_RATE / PRESENTATIONS_PER_EVENT
    settings = {
        "seed": seed, "cells_per_feature": cells_per_feature,
        "rate": learning_rate, "cue_features": 8,
        "content_features": 32,
    }
    models = {key: DirectFlywireOverlay(topology, **settings)
              for key in CONDITIONS}
    order_shift = np.roll(np.arange(len(ordered)),
                          max(1, len(ordered) // 3))
    for i, memory in enumerate(ordered):
        cue_set = binding_cues(memory)
        content = vectors[loc[memory.event_id]]
        shuffled_memory = ordered[int(order_shift[i])]
        if shuffled_memory.event_id == memory.event_id:
            raise AssertionError("Shuffled pairing contains exact target")
        for source_cue in cue_set:
            models["multicue_source"].imprint(source_cue, content)
            models["multicue_shuffled_targets"].imprint(
                source_cue, vectors[loc[shuffled_memory.event_id]]
            )
        combined = train_cue(memory)
        for _ in cue_set:
            models["repeated_concat_control"].imprint(combined, content)
    if any(fingerprint(topology) != original
           for _ in models.values()):
        raise AssertionError("Biological integer connectivity was changed")
    for m in models.values():
        m.assert_original_unchanged()
    learned_ckpt = None
    if checkpoint is not None:
        learned_ckpt = models["multicue_source"].save(checkpoint)
        reconstructed = DirectFlywireOverlay.load(topology, checkpoint)
        if (not np.array_equal(reconstructed.delta, models["multicue_source"].delta)
            or any(not np.array_equal(
                reconstructed.infer(binding_cues(sample)[-1]),
                models["multicue_source"].infer(binding_cues(sample)[-1])
            ) for sample in selected[:3])):
            raise AssertionError("Saved synaptic imprint checkpoint fails exact replay")

    evaluations = {}
    summaries = {}
    for key, model in models.items():
        evaluations[key] = {}
        summaries[key] = {
            "imprint_presentations": model.imprints,
            "edge_update_events": model.edge_update_events,
            "modified_original_synapses": model.modified_edges,
        }
        for scenario in PROBES:
            sample = test if scenario == "absent_episode_last_cue" else selected
            cases = score(model, sample, event_ids, targets, scenario=scenario)
            evaluations[key][scenario] = cases
            summaries[key][scenario] = summarize(cases)
    if fingerprint(topology) != original:
        raise AssertionError("Original anatomy changed during evaluation")
    return {
        "study": "Pretorius direct FlyWire Pilot10 multi-cue binding",
        "topology_status": ("verified original full FlyWire v783"
                            if real else "synthetic computational fixture only"),
        "scientific_boundary": (
            "Multiple source-authored literal cues explicitly trained to predict one "
            "256D signed lexical content vector. Novel-drop probes are deterministic "
            "lexical perturbations, not independent semantic paraphrases. Neural "
            "inference uses only cue and learned source-edge synaptic deltas. "
            "Event identification is EXTERNAL oracle-assisted evaluation. "
            "No autonomous prose recollection, fly cognition or STDP."
        ),
        "source": {
            "canonical_memories": len(corpus),
            "source_episodes": len({m.episode_id for m in corpus}),
            "encoder": meta["encoder_name"],
            "bc01_artifact_sha256": meta["artifact_sha256"],
            "original_anatomy_array_sha256": original,
            "original_anatomy_unchanged": fingerprint(topology) == original,
            "neurons": len(topology.root_ids),
            "aggregate_directed_edges": len(topology.indices),
            "integer_synapse_contacts": int(topology.synapse_counts.sum(dtype=np.int64)),
        },
        "split": {
            "seed": seed,
            "imprinted_train": len(train),
            "validation_unknown": len(validation),
            "test_unknown": len(test),
            "selected_train_test": len(selected),
            "train_events_with_only_two_literal_cues": samples_with_two_cues,
            "cue_exposures_per_training_memory": PRESENTATIONS_PER_EVENT,
            "input_rate_per_presentation": learning_rate,
            "total_nominal_rate_per_memory": ORIGINAL_RATE,
            "source_cues_chosen": "first, middle, last; second repeated if only two",
            "primary_positive_scenario": "last_trained_cue is explicitly trained, not held out",
            "lexical_perturbation": "delete last whitespace token, or final character",
            "fixed_acceptance_threshold_source": (
                "Original Pilot08 seed31 heldout-episode calibration, not refitted"),
            "fixed_acceptance_threshold": REFERENCE_PILOT08_THRESHOLD,
            "checkpoints": {"multicue_source_sha256": learned_ckpt},
        },
        "comparisons": summaries,
        "oracle_only_case_results": evaluations,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--topology", type=Path)
    group.add_argument("--synthetic-test", action="store_true")
    p.add_argument("--bc01-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=31)
    p.add_argument("--cells-per-feature", type=int)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    model = (Topology.synthetic(n=8192, degree=16, seed=15)
             if a.synthetic_test else Topology.read(a.topology))
    result = benchmark(
        model, a.bc01_dir, seed=a.seed,
        cells_per_feature=a.cells_per_feature or (8 if a.synthetic_test else 32),
        checkpoint=a.checkpoint, real=not a.synthetic_test,
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    print(json.dumps({
        "study": result["study"],
        "topology": result["topology_status"],
        "source_events": result["source"]["canonical_memories"],
        "training_events": result["split"]["imprinted_train"],
        "anatomical_edges_unchanged": result["source"]["original_anatomy_unchanged"],
        "summary": result["comparisons"],
        "full_original_cases": str(a.output),
    }, indent=2))


if __name__ == "__main__":
    main()
