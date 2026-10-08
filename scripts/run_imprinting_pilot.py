#!/usr/bin/env python3
"""Pilot 01: comparable load and control assays on an immutable v12 corpus."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pretorius_connectome.imprinting import (  # noqa: E402
    SynapticOverlay, evaluate, load_v12, order_by_episode,
    retrieval_only, shuffled_targets,
)

EVENTS = ROOT / "memories/current/Pretorius_v12_450_Events_Complete.jsonl"
SIDECARS = ROOT / "memories/annotations/v12_450_sidecars.jsonl"
PINNED_BLOBS = {
    "events": "718dcc2d5ba4feccdef1690d447edfcebaa9bfb5",
    "sidecars": "ad32025166c382caf13e07c7e3b0863eb89e1adb",
}
SOURCE_COMMIT = "60c8ee8dd78158a8f2d7c73b9eccb3b1f121291f"


def git_blob_sha(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def verify_sources(events: Path, sidecars: Path) -> None:
    for label, path in (("events", events), ("sidecars", sidecars)):
        actual = git_blob_sha(path)
        if actual != PINNED_BLOBS[label]:
            raise ValueError(f"v12 {label} SHA mismatch: {actual}; expected {PINNED_BLOBS[label]}")


def run(seeds: tuple[int, ...] = (0, 1, 2),
        loads: tuple[int, ...] = (50, 100, 200, 450),
        cue_units: int = 512, memory_units: int = 256,
        density: float = 0.55) -> dict:
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique and nonempty")
    if not loads or tuple(sorted(set(loads))) != loads or loads[-1] > 450:
        raise ValueError("loads must be increasing unique values within 450")
    verify_sources(EVENTS, SIDECARS)
    corpus = load_v12(EVENTS, SIDECARS)
    # One fixed, episode-balanced curriculum for all seeds and all conditions.
    ordered = order_by_episode(corpus, seed=20261008)
    results = []
    for seed in seeds:
        learned = SynapticOverlay(cue_units, memory_units, density, seed)
        shuffled = SynapticOverlay(cue_units, memory_units, density, seed)
        untouched = SynapticOverlay(cue_units, memory_units, density, seed)
        wrong_texts = shuffled_targets(ordered, seed + 17031)
        early_learned = early_shuffled = None
        for index, memory in enumerate(ordered, 1):
            learned.imprint(memory.cues, memory.memory_text)
            shuffled.imprint(memory.cues, wrong_texts[index - 1])
            if index not in loads:
                continue
            candidates = ordered[:index]
            controls = {
                "learned": evaluate(learned, candidates),
                "shuffled_pairs": evaluate(shuffled, candidates),
                "unmodified": evaluate(untouched, candidates),
                "retrieval_only_exact_id": retrieval_only(candidates, candidates),
            }
            # Fixed 50-candidate decoding isolates some synaptic interference
            # from the increased distractor count in the full candidate set.
            if index >= 50:
                first_fifty = ordered[:50]
                controls["fixed50_learned"] = evaluate(learned, first_fifty)
                controls["fixed50_shuffled"] = evaluate(shuffled, first_fifty)
                if index == 50:
                    early_learned = controls["fixed50_learned"]["top1"]
                    early_shuffled = controls["fixed50_shuffled"]["top1"]
                controls["fixed50_change_from_50"] = (
                    round(controls["fixed50_learned"]["top1"] - early_learned, 6)
                    if early_learned is not None else None
                )
            results.append({
                "seed": seed, "load": index,
                "episode_count": len({m.episode_id for m in candidates}),
                **controls,
            })
    return {
        "experiment": "pretorius-imprinting-pilot01",
        "interpretation": (
            "Associative content-fingerprint identification with an oracle "
            "evaluator; not natural-language recall, semantic understanding, "
            "FlyWire simulation, or behavioral identity."
        ),
        "source_commit": SOURCE_COMMIT,
        "source_git_blobs": PINNED_BLOBS,
        "corpus_events": len(corpus),
        "corpus_episodes": len({m.episode_id for m in corpus}),
        "cue_annotation_status": "unreviewed_candidate",
        "source_cue_count": sum(len(m.cues) for m in corpus),
        "seeds": list(seeds),
        "loads": list(loads),
        "topology": {
            "kind": "synthetic_fixed_bipartite_boolean_mask",
            "cue_units": cue_units,
            "memory_units": memory_units,
            "density": density,
            "overlay": "masked_additive_hebbian",
            "target": "opaque_sha256_text_fingerprint",
        },
        "probe": "one already-imprinted cue per trained event, within-item assay",
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--loads", default="50,100,200,450")
    parser.add_argument("--cue-units", type=int, default=512)
    parser.add_argument("--memory-units", type=int, default=256)
    parser.add_argument("--density", type=float, default=0.55)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    record = run(
        tuple(int(n) for n in args.seeds.split(",")),
        tuple(int(n) for n in args.loads.split(",")),
        args.cue_units, args.memory_units, args.density,
    )
    payload = json.dumps(record, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
