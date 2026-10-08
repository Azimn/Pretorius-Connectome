#!/usr/bin/env python3
"""Pilot 03: train-only fitted lexical encoders against unchanged v12 memories."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from pretorius_connectome.imprinting import load_v12  # noqa: E402
from pretorius_connectome.pilot03 import METHODS, evaluate_seed  # noqa: E402
from scripts.run_imprinting_pilot import (  # noqa: E402
    EVENTS, SIDECARS, PINNED_BLOBS, SOURCE_COMMIT, verify_sources,
)


def run(seeds: tuple[int, ...] = (31, 37, 43),
        cue_units: int = 512, memory_units: int = 256,
        density: float = 0.55) -> dict:
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be distinct and nonempty")
    verify_sources(EVENTS, SIDECARS)
    memories = load_v12(EVENTS, SIDECARS)
    return {
        "experiment": "pretorius-imprinting-pilot03",
        "source_commit": SOURCE_COMMIT,
        "source_git_blobs": PINNED_BLOBS,
        "source_record_count": len(memories),
        "source_episode_count": len({m.episode_id for m in memories}),
        "seeds": list(seeds),
        "methods": list(METHODS),
        "fixed_topology": {
            "cue_units": cue_units, "fingerprint_units": memory_units,
            "density": density, "architecture": "synthetic_feedforward_bipartite_mask",
            "target": "sha256_narrative_fingerprint_oracle_decoder",
            "plasticity": "masked_additive_hebbian",
        },
        "limitations": (
            "Episode-disjoint negatives and new lexical perturbations only. "
            "Not semantic paraphrasing, true unseen-event recall, behavioral "
            "identity, biological FlyWire connectivity, or brain simulation. "
            "No training on withheld episode narratives or cues."
        ),
        "trials": [
            evaluate_seed(memories, seed, cue_units, memory_units, density)
            for seed in seeds
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="31,37,43")
    parser.add_argument("--cue-units", type=int, default=512)
    parser.add_argument("--memory-units", type=int, default=256)
    parser.add_argument("--density", type=float, default=0.55)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    record = run(
        seeds=tuple(int(s) for s in args.seeds.split(",")),
        cue_units=args.cue_units, memory_units=args.memory_units,
        density=args.density,
    )
    body = json.dumps(record, sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(body, encoding="utf-8")
    print(body)


if __name__ == "__main__":
    main()
