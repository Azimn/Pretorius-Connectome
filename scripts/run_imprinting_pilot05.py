#!/usr/bin/env python3
"""Pilot 05A: post-hoc full-narrative retrieval and masked neural association."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from pretorius_connectome.imprinting import load_v12  # noqa: E402
from pretorius_connectome.pilot04 import load_challenge  # noqa: E402
from pretorius_connectome.pilot05 import METHODS, evaluate_seed  # noqa: E402
from scripts.run_imprinting_pilot import (  # noqa: E402
    EVENTS, SIDECARS, PINNED_BLOBS, SOURCE_COMMIT, git_blob_sha, verify_sources,
)
from scripts.run_imprinting_pilot04 import CHALLENGE, CHALLENGE_BLOB  # noqa: E402


def _decisions() -> dict[str, str]:
    """Read decision field from same locked source, never from challenge labels."""
    result = {}
    with EVENTS.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            event_id = record["event_id"]
            if event_id in result:
                raise ValueError("duplicate source decision event")
            text = record.get("decisions")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"missing decision text for {event_id}")
            result[event_id] = text
    if len(result) != 450:
        raise ValueError("source decision set differs from archive")
    return result


def run(seeds: tuple[int, ...] = (31, 37, 43),
        cue_units: int = 512, memory_units: int = 256,
        density: float = 0.55) -> dict:
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be nonempty and distinct")
    verify_sources(EVENTS, SIDECARS)
    if git_blob_sha(CHALLENGE) != CHALLENGE_BLOB:
        raise ValueError("locked Pilot 04 test prompts changed")
    memories = load_v12(EVENTS, SIDECARS)
    cases = load_challenge(CHALLENGE, memories)
    decisions = _decisions()
    return {
        "experiment": "pretorius-pilot05a-posthoc-narrative-diagnostic",
        "source_commit": SOURCE_COMMIT,
        "source_git_blobs": PINNED_BLOBS,
        "frozen_previous_challenge_blob": CHALLENGE_BLOB,
        "challenge_status": "REUSED previously observed Pilot04 benchmark, not new blind holdout",
        "corpus_events": len(memories),
        "challenge_cases": len(cases),
        "seeds": list(seeds),
        "methods": list(METHODS),
        "calibration_input": (
            "frozen original 'decisions' field from train and validation episodes; "
            "NO Pilot 04 challenge query used for calibration"
        ),
        "model_difference": (
            "full narrative texts explicitly exposed as training input to TF-IDF, "
            "BM25, and neural imprints; former pilots only saw original cue phrases"
        ),
        "neural": {
            "fixed_synthetic_mask": True,
            "cue_units": cue_units, "memory_units": memory_units, "density": density,
            "plasticity": "masked_sum_of_hebbian_outer_products",
            "target": "opaque_sha256_narrative_fingerprint",
            "decoder": "evaluator_only_oracle_fingerprint_codebook",
        },
        "interpretation_limit": (
            "A diagnostic, not evidence of semantic understanding or "
            "independent paraphrase generalization. Lexical retrieval can "
            "return evidence sentences but cannot verify truth or "
            "contradiction. The synthetic overlay is not FlyWire."
        ),
        "trials": [
            evaluate_seed(memories, decisions, cases, seed,
                          cue_units, memory_units, density)
            for seed in seeds
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="31,37,43")
    parser.add_argument("--cue-units", type=int, default=512)
    parser.add_argument("--memory-units", type=int, default=256)
    parser.add_argument("--density", type=float, default=0.55)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    record = run(
        seeds=tuple(int(s) for s in args.seeds.split(",")),
        cue_units=args.cue_units, memory_units=args.memory_units,
        density=args.density,
    )
    body = json.dumps(record, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(body, encoding="utf-8")
    print(body)


if __name__ == "__main__":
    main()
