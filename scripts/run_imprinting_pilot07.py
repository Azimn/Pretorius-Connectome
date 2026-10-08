#!/usr/bin/env python3
"""Pilot 07A -- REAL LOCAL NLI or review packet; no model-call fallback."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from pretorius_connectome.imprinting import load_v12
from pretorius_connectome.pilot04 import load_challenge
from pretorius_connectome.pilot07 import (
    LocalMiniLMNLI, NLI_MODEL_ID, NLI_MODEL_REVISION, evaluate_seed,
)
from pretorius_connectome.pilot07_review import (
    build_packet, write_packet, validate_packet_submission,
)
from scripts.run_imprinting_pilot import (
    EVENTS, SIDECARS, SOURCE_COMMIT, PINNED_BLOBS, verify_sources, git_blob_sha,
)
from scripts.run_imprinting_pilot04 import CHALLENGE, CHALLENGE_BLOB
from scripts.run_imprinting_pilot05 import _decisions


def inputs():
    verify_sources(EVENTS, SIDECARS)
    if git_blob_sha(CHALLENGE) != CHALLENGE_BLOB:
        raise ValueError("Pilot04 frozen challenge changed")
    memories = load_v12(EVENTS, SIDECARS)
    cases = load_challenge(CHALLENGE, memories)
    return memories, cases


def diagnostic(seeds=(31,), threads=2, max_tokens=384):
    """Load actual local checkpoint; cannot be confused with fake unit tests."""
    memories, cases = inputs()
    classifier = LocalMiniLMNLI(max_tokens=max_tokens, threads=threads)
    decisions = _decisions()
    records = [evaluate_seed(memories, decisions, cases, seed, classifier)
               for seed in seeds]
    return {
        "experiment": "pretorius-pilot07a-posthoc-pinned-local-nli",
        "challenge_status": (
            "PREVIOUSLY EXAMINED 68 assistant-authored Pilot04 prompts; "
            "NOT independent human-reviewed validation"
        ),
        "frozen_corpus_commit": SOURCE_COMMIT,
        "frozen_source_blobs": PINNED_BLOBS,
        "challenge_git_blob": CHALLENGE_BLOB,
        "corpus_events": len(memories),
        "challenge_cases": len(cases),
        "model": {
            "id": NLI_MODEL_ID,
            "revision": NLI_MODEL_REVISION,
            "license": "apache-2.0",
            "source_url": "https://huggingface.co/" + NLI_MODEL_ID,
            "runtime": "CPU only; pretrained NLI; no paid API",
            "labels": classifier.ordered_labels,
            "max_token_budget": classifier.max_tokens,
            "software_versions": classifier.versions,
        },
        "explanation": (
            "BM25 selects top-3 ORIGINAL trained-event narratives. "
            "A frozen pretrained SNLI/MultiNLI cross-encoder scores each "
            "story as premise and incoming claim as hypothesis. "
            "The top event is accepted ONLY when the model predicts "
            "entailment above frozen threshold and margins. Predicted "
            "scores are not gold entailment truth or a proof of identity."
        ),
        "trials": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("packet", "validate", "diagnostic"),
                        default="packet")
    parser.add_argument("--directory", type=Path,
                        default=Path("results/pilot07/review_packet_v1"))
    parser.add_argument("--seeds", default="31")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.mode == "packet":
        memories, cases = inputs()
        result = write_packet(build_packet(memories, cases), args.directory)
    elif args.mode == "validate":
        result = validate_packet_submission(args.directory)
    else:
        seeds = tuple(int(x.strip()) for x in args.seeds.split(","))
        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError("seeds must be distinct")
        result = diagnostic(seeds, args.threads, args.max_tokens)
    payload = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
