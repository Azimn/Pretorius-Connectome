#!/usr/bin/env python3
"""Pilot 02: episode-disjoint rejection and novel-lexical-cue probes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pretorius_connectome.imprinting import load_v12  # noqa: E402
from pretorius_connectome.pilot02 import (  # noqa: E402
    generalization_assay, interference_assay,
)
from scripts.run_imprinting_pilot import (  # noqa: E402
    EVENTS, SIDECARS, PINNED_BLOBS, SOURCE_COMMIT, verify_sources,
)


def run(seeds: tuple[int, ...] = (0, 1, 2),
        cue_units: int = 512, memory_units: int = 256,
        density: float = 0.55, include_interference: bool = True) -> dict:
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("nonempty unique seeds are required")
    verify_sources(EVENTS, SIDECARS)
    corpus = load_v12(EVENTS, SIDECARS)
    heldout = [
        generalization_assay(corpus, seed, cue_units, memory_units, density)
        for seed in seeds
    ]
    load_curves = (
        [
            result for seed in seeds
            for result in interference_assay(
                corpus, seed, (50, 100, 200, 450),
                cue_units, memory_units, density,
            )
        ] if include_interference else []
    )
    return {
        "experiment": "pretorius-imprinting-pilot02",
        "source_commit": SOURCE_COMMIT,
        "source_git_blobs": PINNED_BLOBS,
        "events": len(corpus),
        "episodes": len({m.episode_id for m in corpus}),
        "seeds": list(seeds),
        "topology": {
            "kind": "fixed_synthetic_feedforward_mask",
            "cue_units": cue_units, "fingerprint_units": memory_units,
            "density": density,
        },
        "evaluation_protocol": {
            "group_disjointness": "episode_only, not deduplicated narrative cluster",
            "positive_query": "unseen lexical cue permutation or token deletion",
            "negative_query": "event drawn from withheld episode, never imprinted",
            "calibration": "known imprinted events versus different withheld calibration episodes",
            "threshold_selection": "maximize calibration balanced accuracy, conservative tie",
            "test": "different known-event probes and independent withheld test episodes",
            "decoder": "external oracle SHA-256 narrative fingerprint codebook",
            "caution": "not natural language recall, semantic paraphrase, behavioral identity or FlyWire",
        },
        "generalization": heldout,
        "fixed50_interference": load_curves,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--cue-units", type=int, default=512)
    parser.add_argument("--memory-units", type=int, default=256)
    parser.add_argument("--density", type=float, default=0.55)
    parser.add_argument("--skip-interference", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    record = run(
        seeds=tuple(int(s) for s in args.seeds.split(",")),
        cue_units=args.cue_units, memory_units=args.memory_units,
        density=args.density, include_interference=not args.skip_interference,
    )
    payload = json.dumps(record, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
