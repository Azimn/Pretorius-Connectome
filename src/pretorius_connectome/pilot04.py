"""Pilot 04, frozen assistant-authored semantic and contradiction stress assay.

The challenge is independent of training *inputs*, not independent of the
author's exposure to the original source. Human review and blind annotation
have NOT occurred. This is an exploratory generalization check.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from pretorius_connectome.imprinting import Cue, Memory, order_by_episode
from pretorius_connectome.pilot02 import Probe, calibrate, episode_split, lexical_probe, lexical_retrieval, words
from pretorius_connectome.pilot03 import METHODS, _fit_conditions, scores


@dataclass(frozen=True)
class ChallengeCase:
    case_id: str
    event_id: str
    kind: str
    query: str
    source_anchor: str


def load_challenge(path: str | Path, memories: list[Memory]) -> list[ChallengeCase]:
    """Validate 34 paired prompts tied to existing event IDs, no training edits."""
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    if len(records) != 68:
        raise ValueError("Pilot 04 challenge must contain 68 paired records")
    valid = {memory.event_id: memory for memory in memories}
    by_event: dict[str, set[str]] = defaultdict(set)
    unique_cases: set[str] = set()
    unique_queries: set[str] = set()
    output: list[ChallengeCase] = []
    for record in records:
        if set(record) != {"case_id", "event_id", "kind", "query",
                           "source_anchor", "authoring"}:
            raise ValueError("challenge schema has drifted")
        case_id, event_id, kind = (
            record["case_id"], record["event_id"], record["kind"]
        )
        query = record["query"].strip()
        if (not isinstance(case_id, str) or event_id not in valid or
            kind not in ("paraphrase", "contradiction") or
            record["authoring"] != "assistant_authored_unreviewed" or
            not isinstance(query, str) or len(query) < 25 or
            not isinstance(record["source_anchor"], str) or
            len(record["source_anchor"]) < 20 or
            case_id in unique_cases or query.casefold() in unique_queries or
            event_id in query):
            raise ValueError(f"invalid challenge: {case_id}")
        if query.casefold() in {
            c.surface.casefold() for c in valid[event_id].cues if c.surface
        }:
            raise ValueError("identical original cue is not a semantic probe")
        if kind in by_event[event_id]:
            raise ValueError(f"duplicate challenge kind: {event_id}")
        by_event[event_id].add(kind)
        unique_cases.add(case_id)
        unique_queries.add(query.casefold())
        output.append(ChallengeCase(case_id, event_id, kind, query,
                                    record["source_anchor"]))
    if len(by_event) != 34 or any(kinds != {"paraphrase", "contradiction"}
                                  for kinds in by_event.values()):
        raise ValueError("unbalanced challenge event pairings")
    return output


def _probes(cases: list[ChallengeCase]) -> list[Probe]:
    # No event/case ID, challenge kind, or target label enters encoder.
    return [Probe(case.event_id, Cue("unseen-evaluation-query", case.query))
            for case in cases]


def _round(number: float) -> float:
    return round(float(number), 6)


def _positive(records: list[dict], threshold: float) -> dict:
    if not records or any(not r["known"] for r in records):
        raise ValueError("positive benchmark must target known records")
    return {
        "n": len(records),
        "oracle_top1": _round(np.mean([r["correct"] for r in records])),
        "accepted": _round(np.mean([r["top_cosine"] > threshold for r in records])),
        "correct_and_accepted": _round(np.mean([
            bool(r["correct"] and r["top_cosine"] > threshold) for r in records
        ])),
        "mean_target_cosine": _round(np.mean([r["target_cosine"] for r in records])),
    }


def _contradiction(records: list[dict], threshold: float) -> dict:
    """Negative queries describe events counterfactually: target is NOT true."""
    if not records or any(not r["known"] for r in records):
        raise ValueError("contradictions must be tied to trained event IDs")
    return {
        "n": len(records),
        "false_acceptance_rate": _round(np.mean([
            r["top_cosine"] > threshold for r in records
        ])),
        "false_target_confirmation_rate": _round(np.mean([
            r["top_cosine"] > threshold and r["predicted"] == r["event_id"]
            for r in records
        ])),
        "raw_target_choice_rate": _round(np.mean([
            r["predicted"] == r["event_id"] for r in records
        ])),
    }


def _unknown(records: list[dict], threshold: float) -> dict:
    if not records or any(r["known"] for r in records):
        raise ValueError("unimprinted episodes must be absent from codebook")
    return {
        "n": len(records),
        "false_acceptance_rate": _round(np.mean([
            r["top_cosine"] > threshold for r in records
        ])),
    }


def _lexical_contradiction(train: list[Memory],
                           cases: list[ChallengeCase]) -> dict:
    """Text retriever has no abstention; report contradictions as false targets."""
    docs = []
    for memory in train:
        tokens = set()
        for cue in memory.cues:
            if cue.surface:
                tokens.update(words(cue.surface))
        docs.append(tokens)
    total = 0
    matched = 0
    for case in cases:
        q = set(words(case.query))
        overlaps = [len(q & doc) for doc in docs]
        predicted = train[int(np.argmax(overlaps))].event_id
        total += 1
        matched += int(predicted == case.event_id)
    return {
        "n": total,
        "raw_false_target_choice_rate": _round(matched / total),
        "forced_to_answer": True,
    }


def _overlap(train: list[Memory], cases: list[ChallengeCase]) -> dict:
    lookup = {m.event_id: m for m in train}
    ratios = []
    for case in cases:
        cue_tokens = {
            token for cue in lookup[case.event_id].cues
            for token in words(cue.surface)
        }
        challenge_tokens = set(words(case.query))
        ratios.append(
            len(cue_tokens & challenge_tokens) / len(challenge_tokens)
            if challenge_tokens else 0.0
        )
    return {
        "n": len(cases),
        "mean_query_token_overlap_with_own_original_cues": _round(np.mean(ratios)),
        "mean_query_tokens": _round(np.mean([len(words(c.query)) for c in cases])),
    }


def evaluate_semantic_trial(memories: list[Memory],
                            cases: list[ChallengeCase], seed: int,
                            cue_units: int = 512, memory_units: int = 256,
                            density: float = 0.55) -> dict:
    """Freeze trained model and swap-derived abstention thresholds before query."""
    train, validation, test = episode_split(memories, seed)
    ordered = order_by_episode(train, seed=20261008)
    train_ids = {m.event_id for m in ordered}
    test_ids = {m.event_id for m in test}
    # Calibration replicates Pilot 03's protocol with *original* corpus cues;
    # not a single new semantic challenge utterance is used for calibration.
    eligible = [
        p for m in ordered if (p := lexical_probe(m, "swap")) is not None
    ]
    rng = np.random.default_rng(seed + 227)
    permutation = rng.permutation(len(eligible))
    cut = len(permutation) // 2
    calibration_positive = [eligible[int(i)] for i in permutation[:cut]]
    calibration_unknown = [
        p for m in validation if (p := lexical_probe(m, "swap")) is not None
    ]
    if not calibration_positive or not calibration_unknown:
        raise ValueError("no calibration data")

    known_positive = [
        c for c in cases if c.kind == "paraphrase" and c.event_id in train_ids
    ]
    known_negative = [
        c for c in cases if c.kind == "contradiction" and c.event_id in train_ids
    ]
    unknown_positive = [
        c for c in cases if c.kind == "paraphrase" and c.event_id in test_ids
    ]
    if len(known_positive) < 10 or len(known_positive) != len(known_negative):
        raise ValueError("semantic challenge lacks a valid known-event sample")
    if len(unknown_positive) < 2:
        raise ValueError("semantic challenge lacks held-out episode examples")

    trained = _fit_conditions(ordered, seed, cue_units, memory_units, density)
    output: dict = {}
    for mode in METHODS:
        model = trained[mode]
        known_cal = scores(model, ordered, calibration_positive)
        unknown_cal = scores(model, ordered, calibration_unknown)
        threshold = calibrate(known_cal, unknown_cal)
        positive = scores(model, ordered, _probes(known_positive))
        contradictory = scores(model, ordered, _probes(known_negative))
        absent = scores(model, ordered, _probes(unknown_positive))
        output[mode] = {
            "calibration_threshold": _round(threshold),
            "positive_paraphrases": _positive(positive, threshold),
            "contradictory_prompts": _contradiction(contradictory, threshold),
            "unimprinted_episode_prompts": _unknown(absent, threshold),
            "probe_results": [
                {
                    "case_id": c.case_id, "kind": c.kind,
                    "expected_event_id": c.event_id,
                    "predicted_event_id": row["predicted"],
                    "accepted": bool(row["top_cosine"] > threshold),
                    "top_cosine": _round(row["top_cosine"]),
                }
                for c, row in (
                    list(zip(known_positive, positive)) +
                    list(zip(known_negative, contradictory)) +
                    list(zip(unknown_positive, absent))
                )
            ],
        }
    return {
        "seed": seed,
        "counts": {
            "trained_events": len(ordered),
            "trained_positive_challenge": len(known_positive),
            "trained_contradictions": len(known_negative),
            "unknown_episode_challenge": len(unknown_positive),
        },
        "episode_split": {
            "train": sorted({m.episode_id for m in ordered}),
            "validation": sorted({m.episode_id for m in validation}),
            "test": sorted({m.episode_id for m in test}),
        },
        "source_cue_overlap": _overlap(ordered, known_positive),
        "word_retrieval_reference": {
            "positive_top1": lexical_retrieval(ordered, _probes(known_positive)),
            "contradiction": _lexical_contradiction(ordered, known_negative),
        },
        "modes": output,
    }
