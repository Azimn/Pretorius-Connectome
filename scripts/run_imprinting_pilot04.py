#!/usr/bin/env python3
"""Pilot 04: assistant-authored semantic and counterfactual challenge, frozen v12."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from pretorius_connectome.imprinting import load_v12  # noqa: E402
from pretorius_connectome.pilot04 import load_challenge, evaluate_semantic_trial  # noqa: E402
from scripts.run_imprinting_pilot import (  # noqa: E402
    EVENTS, SIDECARS, PINNED_BLOBS, SOURCE_COMMIT, git_blob_sha, verify_sources,
)

CHALLENGE = ROOT / "experiments/pilot04/challenge_v1.jsonl"
CHALLENGE_BLOB = "1e1c6f0273519bf3cd5868404ddfa5f783df6abd"


def run(seeds: tuple[int, ...] = (31, 37, 43),
        cue_units: int = 512, memory_units: int = 256,
        density: float = 0.55) -> dict:
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be nonempty and distinct")
    verify_sources(EVENTS, SIDECARS)
    actual = git_blob_sha(CHALLENGE)
    if actual != CHALLENGE_BLOB:
        raise ValueError(
            f"semantic challenge file changed: {actual} != {CHALLENGE_BLOB}"
        )
    memories = load_v12(EVENTS, SIDECARS)
    cases = load_challenge(CHALLENGE, memories)
    return {
        "experiment": "pretorius-imprinting-pilot04",
        "recorded_date": "2026-10-08",
        "source_commit": SOURCE_COMMIT,
        "source_git_blobs": PINNED_BLOBS,
        "challenge_git_blob": CHALLENGE_BLOB,
        "challenge_case_count": len(cases),
        "challenge_event_count": len({c.event_id for c in cases}),
        "authoring_status": (
            "assistant-written after inspecting original memories; "
            "NOT blinded, human-validated or independently adjudicated"
        ),
        "seeds": list(seeds),
        "cue_units": cue_units, "fingerprint_units": memory_units,
        "mask_density": density,
        "semantic_limitations": (
            "Targets remain opaque hash fingerprints, evaluated against an "
            "external oracle codebook. Paraphrases and contradictions are "
            "assistant-authored stress prompts, not independent expert labels. "
            "An abstention score from swapped-cue calibration does not verify "
            "logical contradiction or memory content. No FlyWire connectivity, "
            "natural-language recollection, or behavioral agent is present."
        ),
        "trials": [
            evaluate_semantic_trial(memories, cases, seed,
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
    record = run(tuple(int(s) for s in args.seeds.split(",")),
                 args.cue_units, args.memory_units, args.density)
    content = json.dumps(record, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")
    print(content)


if __name__ == "__main__":
    main()
