"""Pilot 02: withheld-episode rejection, lexical perturbation, and interference.

A synthetic, feedforward, masked associative overlay. The oracle narrative
fingerprint codebook is used only by evaluation, never by the model.
No semantic paraphrase, natural language recall, or behavioral choice is claimed.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
import re

import numpy as np

from pretorius_connectome.imprinting import (
    Cue, Memory, SynapticOverlay, content_fingerprint, cue_vector,
    order_by_episode, shuffled_targets,
)


def words(surface: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[^\W_]+(?:['’][^\W_]+)?", surface.casefold()))


@dataclass(frozen=True)
class Probe:
    event_id: str
    cue: Cue


def lexical_probe(memory: Memory, mode: str = "swap") -> Probe | None:
    """Use one eligible original *surface*, but never its imprinted cue ID.

    Swapping words or deleting one word is lexical perturbation, NOT
    an independently authored paraphrase or true novel semantic cue.
    """
    if mode not in ("swap", "drop"):
        raise ValueError("invalid perturbation mode")
    for source in memory.cues:
        if not source.surface:
            continue
        units = words(source.surface)
        if len(units) < 2:
            continue
        if mode == "swap":
            variant = (units[1], units[0], *units[2:])
        else:
            variant = units[:-1]
        new_phrase = " ".join(variant)
        if new_phrase == " ".join(units):
            continue
        # Not keyed on target event identity, and no previously imprinted ID.
        return Probe(memory.event_id, Cue(f"probe:{mode}:{new_phrase}", new_phrase))
    return None


def episode_split(memories: list[Memory], seed: int) -> tuple[list[Memory], list[Memory], list[Memory]]:
    """Disjoint episode splits. Related scenes ACROSS episodes may still leak."""
    episodes = sorted({m.episode_id for m in memories})
    if len(episodes) < 6:
        raise ValueError("need at least six episodes for episode-disjoint splits")
    rng = np.random.default_rng(seed + 5719)
    order = [episodes[int(i)] for i in rng.permutation(len(episodes))]
    n_unknown = max(2, int(round(len(episodes) * 0.15)))
    validation = set(order[:n_unknown])
    test = set(order[n_unknown:2 * n_unknown])
    train = set(order[2 * n_unknown:])
    if not train or validation & test or test & train or train & validation:
        raise ValueError("invalid episode partition")
    buckets = [
        [m for m in memories if m.episode_id in selected]
        for selected in (train, validation, test)
    ]
    if any(not bucket for bucket in buckets):
        raise ValueError("empty memory partition")
    return tuple(buckets)  # type: ignore[return-value]


def _oracle_scores(model: SynapticOverlay, candidates: list[Memory],
                   probes: list[Probe]) -> list[dict]:
    """Evaluator-only codebook. No event targets or IDs used in plasticity."""
    if not candidates or not probes:
        return []
    fingerprints = np.stack([
        content_fingerprint(m.memory_text, model.memory_units)
        for m in candidates
    ])
    stimuli = np.stack([cue_vector((p.cue,), model.cue_units) for p in probes])
    raw = stimuli @ model.weights.T
    lengths = np.linalg.norm(raw, axis=1)
    normalized = raw / np.maximum(lengths[:, None], 1e-12)
    sims = normalized @ fingerprints.T
    candidate_ids = [m.event_id for m in candidates]
    id_to_pos = {event_id: i for i, event_id in enumerate(candidate_ids)}
    top = np.argmax(sims, axis=1)
    top_scores = sims[np.arange(len(probes)), top]
    if len(candidates) > 1:
        # Runner-up cosine, rather than unnormalized score.
        second = np.partition(sims, -2, axis=1)[:, -2]
    else:
        second = np.zeros(len(probes), dtype=np.float32)
    scored = []
    for row, probe in enumerate(probes):
        index = id_to_pos.get(probe.event_id)
        active = bool(lengths[row] > 1e-12)
        scored.append({
            "event_id": probe.event_id,
            "known": index is not None,
            "predicted": candidate_ids[int(top[row])] if active else None,
            "top_cosine": float(top_scores[row]) if active else 0.0,
            "margin": float(top_scores[row] - second[row]) if active else 0.0,
            "target_cosine": float(sims[row, index]) if index is not None and active else None,
            "correct": bool(active and int(top[row]) == index) if index is not None else None,
        })
    return scored


def _mean(values: list[float]) -> float | None:
    return round(float(np.mean(values)), 6) if values else None


def _positive_summary(scored: list[dict], threshold: float | None = None) -> dict:
    if not scored or any(not s["known"] for s in scored):
        raise ValueError("positive summary requires known probes")
    accepted = [
        s for s in scored
        if threshold is None or s["top_cosine"] > threshold
    ]
    return {
        "n": len(scored),
        "top1_no_abstention": round(sum(s["correct"] for s in scored) / len(scored), 6),
        "true_positive_rate": round(len(accepted) / len(scored), 6),
        "correct_and_accepted": round(sum(s["correct"] for s in accepted) / len(scored), 6),
        "mean_target_cosine": _mean([s["target_cosine"] for s in scored]),
        "mean_margin": _mean([s["margin"] for s in scored]),
        "mean_top_cosine": _mean([s["top_cosine"] for s in scored]),
    }


def _negative_summary(scored: list[dict], threshold: float) -> dict:
    if not scored or any(s["known"] for s in scored):
        raise ValueError("negative summary requires withheld probes")
    return {
        "n": len(scored),
        "false_acceptance_rate": round(
            sum(s["top_cosine"] > threshold for s in scored) / len(scored), 6
        ),
        "mean_top_cosine": _mean([s["top_cosine"] for s in scored]),
        "mean_margin": _mean([s["margin"] for s in scored]),
    }


def calibrate(known: list[dict], unknown: list[dict]) -> float:
    """Choose threshold using calibration data ONLY, never the test episodes."""
    if (not known or not unknown or
            any(not s["known"] for s in known) or
            any(s["known"] for s in unknown)):
        raise ValueError("need positive and negative calibration probes")
    positives = np.asarray([x["top_cosine"] for x in known])
    negatives = np.asarray([x["top_cosine"] for x in unknown])
    unique = sorted(set(float(x) for x in np.concatenate((positives, negatives))))
    candidates = [unique[0] - 1e-6] + [
        (a + b) / 2 for a, b in zip(unique, unique[1:])
    ] + [unique[-1] + 1e-6]
    best_threshold, best_score = candidates[0], -1.0
    for threshold in candidates:
        sensitivity = float(np.mean(positives > threshold))
        specificity = float(np.mean(negatives <= threshold))
        balanced = 0.5 * (sensitivity + specificity)
        # Conservative tie: prefer lower false positives, larger threshold.
        if balanced > best_score + 1e-12 or (
            abs(balanced - best_score) < 1e-12 and threshold > best_threshold
        ):
            best_threshold, best_score = threshold, balanced
    return float(best_threshold)


def lexical_retrieval(candidates: list[Memory], probes: list[Probe]) -> dict:
    """Separate, non-neural word-overlap lookup baseline. No abstention."""
    docs = []
    for memory in candidates:
        tokens = set()
        for cue in memory.cues:
            if cue.surface:
                tokens.update(words(cue.surface))
        docs.append(tokens)
    known = {m.event_id for m in candidates}
    correct = 0
    included = 0
    for probe in probes:
        if probe.event_id not in known:
            continue
        included += 1
        query = set(words(probe.cue.surface))
        overlaps = [len(query & doc) for doc in docs]
        best = candidates[int(np.argmax(overlaps))].event_id
        correct += int(best == probe.event_id)
    return {
        "n": included,
        "top1_no_abstention": round(correct / included, 6) if included else None,
        "method": "literal_token_overlap_max; ties by candidate order; no abstention",
    }


def train_conditions(train: list[Memory], seed: int,
                     cue_units: int, memory_units: int, density: float) -> dict:
    conditions = ("full", "surface_only", "ids_only", "shuffled", "unmodified")
    overlays = {
        name: SynapticOverlay(cue_units, memory_units, density, seed)
        for name in conditions
    }
    shuffled_text = shuffled_targets(train, seed + 113)
    for memory, wrong_text in zip(train, shuffled_text):
        surface = tuple(c for c in memory.cues if c.surface)
        ids = tuple(c for c in memory.cues if not c.surface)
        overlays["full"].imprint(memory.cues, memory.memory_text)
        if surface:
            overlays["surface_only"].imprint(surface, memory.memory_text)
        if ids:
            overlays["ids_only"].imprint(ids, memory.memory_text)
        overlays["shuffled"].imprint(memory.cues, wrong_text)
    return overlays


def generalization_assay(memories: list[Memory], seed: int,
                         cue_units: int = 512, memory_units: int = 256,
                         density: float = 0.55) -> dict:
    train, validation, test = episode_split(memories, seed)
    train = order_by_episode(train, seed=20261008)
    calibration_rng = np.random.default_rng(seed + 227)
    eligible = [p for m in train if (p := lexical_probe(m, "swap")) is not None]
    if len(eligible) < 8:
        raise ValueError("insufficient eligible training cues")
    indices = calibration_rng.permutation(len(eligible))
    cut = max(1, len(indices) // 2)
    calibration_pos = [eligible[int(i)] for i in indices[:cut]]
    test_pos = [eligible[int(i)] for i in indices[cut:]]
    validation_neg = [p for m in validation if (p := lexical_probe(m, "swap"))]
    test_neg = [p for m in test if (p := lexical_probe(m, "swap"))]
    drop_pos = [p for m in train if (p := lexical_probe(m, "drop"))]
    drop_neg = [p for m in test if (p := lexical_probe(m, "drop"))]
    if not validation_neg or not test_neg or not test_pos or not drop_pos or not drop_neg:
        raise ValueError("empty perturbation partition")
    overlays = train_conditions(train, seed, cue_units, memory_units, density)
    output = {}
    for name, model in overlays.items():
        cal_known = _oracle_scores(model, train, calibration_pos)
        cal_unknown = _oracle_scores(model, train, validation_neg)
        threshold = calibrate(cal_known, cal_unknown)
        known = _oracle_scores(model, train, test_pos)
        unknown = _oracle_scores(model, train, test_neg)
        deleted = _oracle_scores(model, train, drop_pos)
        deleted_neg = _oracle_scores(model, train, drop_neg)
        positives = _positive_summary(known, threshold)
        negatives = _negative_summary(unknown, threshold)
        output[name] = {
            "threshold_from_calibration": round(threshold, 8),
            "calibration": {
                "known": _positive_summary(cal_known, threshold),
                "withheld": _negative_summary(cal_unknown, threshold),
            },
            "swap_known_test": positives,
            "swap_withheld_test": negatives,
            "swap_balanced_accuracy": round(
                (positives["true_positive_rate"] +
                 (1 - negatives["false_acceptance_rate"])) / 2, 6),
            "drop_known_secondary": _positive_summary(deleted, threshold),
            "drop_withheld_secondary": _negative_summary(deleted_neg, threshold),
        }
    return {
        "seed": seed,
        "episodes": {
            "train": sorted({m.episode_id for m in train}),
            "calibration_unknown": sorted({m.episode_id for m in validation}),
            "test_unknown": sorted({m.episode_id for m in test}),
        },
        "counts": {
            "train": len(train), "calibration_unknown": len(validation),
            "test_unknown": len(test), "calibration_known_probe": len(calibration_pos),
            "test_known_probe": len(test_pos), "test_unknown_probe": len(test_neg),
        },
        "conditions": output,
        "lexical_retrieval_reference": lexical_retrieval(train, test_pos),
    }


def interference_assay(memories: list[Memory], seed: int,
                       loads: tuple[int, ...] = (50, 100, 200, 450),
                       cue_units: int = 512, memory_units: int = 256,
                       density: float = 0.55) -> list[dict]:
    """Track fixed earliest-50 cosine AND margin as later weights accumulate."""
    if not loads or tuple(sorted(set(loads))) != loads or loads[-1] > len(memories):
        raise ValueError("invalid loads")
    ordered = order_by_episode(memories, seed=20261008)
    first50 = ordered[:50]
    probes = [Probe(m.event_id, m.cues[0]) for m in first50]
    model = SynapticOverlay(cue_units, memory_units, density, seed)
    records = []
    for index, memory in enumerate(ordered, 1):
        model.imprint(memory.cues, memory.memory_text)
        if index in loads:
            measure = _positive_summary(_oracle_scores(model, first50, probes))
            records.append({"load": index, "seed": seed, **measure})
    return records
