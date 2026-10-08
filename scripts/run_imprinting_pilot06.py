#!/usr/bin/env python3
"""Pilot 06A: reproducible post-hoc sentence-evidence gate on frozen memories."""
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
from pretorius_connectome.pilot06 import METHODS, evaluate_seed  # noqa: E402
from scripts.run_imprinting_pilot import (  # noqa: E402
    EVENTS, SIDECARS, PINNED_BLOBS, SOURCE_COMMIT, git_blob_sha, verify_sources,
)
from scripts.run_imprinting_pilot04 import CHALLENGE, CHALLENGE_BLOB  # noqa: E402
from scripts.run_imprinting_pilot05 import _decisions  # noqa: E402


def run(seeds: tuple[int, ...] = (31, 37, 43),
        cue_units: int = 512, memory_units: int = 256,
        density: float = 0.55) -> dict:
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be distinct and nonempty")
    verify_sources(EVENTS, SIDECARS)
    if git_blob_sha(CHALLENGE) != CHALLENGE_BLOB:
        raise ValueError("Pilot 04 challenge has changed")
    memories = load_v12(EVENTS, SIDECARS)
    cases = load_challenge(CHALLENGE, memories)
    decisions = _decisions()
    return {
        "experiment": "pretorius-pilot06a-posthoc-evidence-gating",
        "source_commit": SOURCE_COMMIT,
        "source_git_blobs": PINNED_BLOBS,
        "challenge_git_blob": CHALLENGE_BLOB,
        "challenge_status": (
            "Previously examined, assistant-authored, human-unreviewed; "
            "post-hoc exploratory diagnostic, NOT independent validation"
        ),
        "record_count": len(memories),
        "challenge_count": len(cases),
        "seeds": list(seeds),
        "methods": list(METHODS),
        "topology": {
            "neural_conditions_only": True,
            "synthetic_fixed_mask": True,
            "cue_units": cue_units,
            "memory_units": memory_units,
            "density": density,
            "decoder": "external oracle SHA256 narrative fingerprints",
        },
        "gate": (
            "Train-only sentence TF-IDF similarity, calibrated using training "
            "and validation episode decision text, >=2 shared content words, "
            "optional shallow explicit-negation mismatch; neither entailment "
            "nor logical contradiction understanding."
        ),
        "data_leakage": (
            "No challenge case text, event label, kind, or source anchor enters "
            "model fitting, threshold calibration or inference; case ID and "
            "expected event ID are evaluation metadata only."
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
    result = run(
        tuple(int(s) for s in args.seeds.split(",")),
        args.cue_units, args.memory_units, args.density,
    )
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
