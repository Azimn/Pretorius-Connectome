"""Pilot 03 cue-encoding challenger, with train-only feature statistics.

The neural mask, 256-bit oracle targets and additive masked Hebbian rule
remain identical to Pilot 01. This synthetic assay does not implement
semantic memory, FlyWire physiology, or natural-language recollection.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math

import numpy as np

from pretorius_connectome.imprinting import (
    Cue, Memory, SynapticOverlay, _digest, content_fingerprint,
    cue_vector, order_by_episode, shuffled_targets,
)
from pretorius_connectome.pilot02 import (
    Probe, _negative_summary, _positive_summary, calibrate,
    episode_split, lexical_probe, lexical_retrieval, words,
)


METHODS = ("legacy_full", "legacy_surface", "token", "char3", "idf_char3",
           "token_shuffled", "token_unmodified")
CHALLENGES = ("swap", "drop", "typo")


def _trigrams(token: str) -> tuple[str, ...]:
    marked = "^" + token + "$"
    if len(marked) < 3:
        return (marked,)
    return tuple(marked[i:i + 3] for i in range(len(marked) - 2))


def _features(surface: str, kind: str) -> Counter:
    tokens = words(surface)
    if not tokens:
        return Counter()
    if kind == "token":
        return Counter("token:" + token for token in tokens)
    if kind in ("char3", "idf_char3"):
        return Counter("char3:" + gram for token in tokens for gram in _trigrams(token))
    raise ValueError("unknown features")


@dataclass
class CueEncoder:
    """Stateless hashing except for IDF fitted strictly on train events."""
    mode: str
    units: int = 512
    idf: dict[str, float] | None = None
    fitted_events: int = 0

    def __post_init__(self) -> None:
        if self.mode not in ("legacy_full", "legacy_surface", "token",
                             "char3", "idf_char3"):
            raise ValueError("unrecognized encoder")
        if self.units <= 0:
            raise ValueError("units must be positive")

    def fit(self, train: list[Memory]) -> "CueEncoder":
        if not train:
            raise ValueError("no training records")
        self.fitted_events = len(train)
        self.idf = None
        if self.mode == "idf_char3":
            df: Counter = Counter()
            for memory in train:
                present = set()
                for cue in memory.cues:
                    if cue.surface:
                        present.update(_features(cue.surface, "idf_char3"))
                df.update(present)
            self.idf = {
                feature: 1.0 + math.log((len(train) + 1) / (count + 1))
                for feature, count in df.items()
            }
        return self

    def encode(self, cues: tuple[Cue, ...], training: bool = False) -> np.ndarray:
        if self.fitted_events <= 0:
            raise ValueError("fit the encoder using training records first")
        if self.mode == "legacy_full":
            return cue_vector(cues, self.units)
        literal = tuple(c for c in cues if c.surface)
        if not literal:
            raise ValueError("surface cue required")
        if self.mode == "legacy_surface":
            return cue_vector(literal, self.units)
        counts: Counter = Counter()
        for cue in literal:
            counts.update(_features(cue.surface, self.mode))
        if not counts:
            raise ValueError("no token features")
        value = np.zeros(self.units, dtype=np.float32)
        default_idf = 1.0 + math.log(self.fitted_events + 1)
        for feature, occurrences in counts.items():
            digest = _digest(feature)
            position = int.from_bytes(digest[:8], "little") % self.units
            sign = 1.0 if digest[8] & 1 else -1.0
            weight = (
                self.idf.get(feature, default_idf)
                if self.idf is not None else 1.0
            )
            value[position] += occurrences * weight * sign
        magnitude = float(np.linalg.norm(value))
        if magnitude <= 1e-12:
            raise ValueError("all hashed features cancelled")
        return value / magnitude


class EncoderOverlay:
    """Same topology, target, plasticity rule and parameter count per encoder."""
    def __init__(self, encoder: CueEncoder, memory_units: int,
                 density: float, seed: int):
        if encoder.fitted_events <= 0:
            raise ValueError("encoder needs training-only fit")
        self.encoder = encoder
        self.network = SynapticOverlay(encoder.units, memory_units, density, seed)
        self.cue_units = encoder.units
        self.memory_units = memory_units

    @property
    def mask(self) -> np.ndarray:
        return self.network.mask

    @property
    def weights(self) -> np.ndarray:
        return self.network.weights

    def imprint(self, memory: Memory, target_text: str | None = None) -> None:
        x = self.encoder.encode(memory.cues, training=True)
        y = content_fingerprint(
            memory.memory_text if target_text is None else target_text,
            self.memory_units
        )
        self.weights[:] += (y[:, None] * x[None, :]) * self.mask
        self.network.updates += 1

    def query(self, probe: Probe) -> np.ndarray:
        return self.weights @ self.encoder.encode((probe.cue,))


def typo_probe(memory: Memory) -> Probe | None:
    """One deterministic nonsemantic character replacement, no ID reuse."""
    for cue in memory.cues:
        if not cue.surface:
            continue
        tokens = list(words(cue.surface))
        if not tokens:
            continue
        candidates = [(len(token), index) for index, token in enumerate(tokens)
                      if len(token) >= 4]
        if not candidates:
            continue
        _, index = max(candidates)
        word = tokens[index]
        middle = len(word) // 2
        replacement = "x" if word[middle] != "x" else "z"
        tokens[index] = word[:middle] + replacement + word[middle + 1:]
        phrase = " ".join(tokens)
        return Probe(memory.event_id, Cue("probe:typo:" + phrase, phrase))
    return None


def probes(memories: list[Memory], challenge: str) -> list[Probe]:
    if challenge not in CHALLENGES:
        raise ValueError("unknown challenge")
    if challenge == "typo":
        return [p for m in memories if (p := typo_probe(m)) is not None]
    return [p for m in memories if (p := lexical_probe(m, challenge)) is not None]


def scores(model: EncoderOverlay, candidates: list[Memory],
           queries: list[Probe]) -> list[dict]:
    """Decoder-only candidate codebook, not a retriever inside the substrate."""
    if not queries:
        return []
    targets = np.stack([content_fingerprint(m.memory_text, model.memory_units)
                        for m in candidates]).astype(np.float32)
    vectors = np.stack([model.encoder.encode((p.cue,)) for p in queries])
    responses = vectors @ model.weights.T
    norms = np.linalg.norm(responses, axis=1)
    similarities = (responses / np.maximum(norms[:, None], 1e-12)) @ targets.T
    top = similarities.argmax(axis=1)
    best = similarities[np.arange(len(queries)), top]
    if len(candidates) > 1:
        runner_up = np.partition(similarities, -2, axis=1)[:, -2]
    else:
        runner_up = np.zeros(len(queries), dtype=np.float32)
    ids = [m.event_id for m in candidates]
    lookup = {v: i for i, v in enumerate(ids)}
    output = []
    for i, probe in enumerate(queries):
        original_index = lookup.get(probe.event_id)
        active = norms[i] > 1e-12
        output.append({
            "event_id": probe.event_id,
            "known": original_index is not None,
            "predicted": ids[int(top[i])] if active else None,
            "top_cosine": float(best[i]) if active else 0.0,
            "target_cosine": (
                float(similarities[i, original_index]) if active else 0.0
            ) if original_index is not None else None,
            "margin": float(best[i] - runner_up[i]) if active else 0.0,
            "correct": (
                bool(active and int(top[i]) == original_index)
                if original_index is not None else None
            ),
        })
    return output


def _auc(known: list[dict], unknown: list[dict]) -> float:
    """Pairwise AUROC (ties=0.5), a descriptive threshold-free test metric."""
    pos = np.array([r["top_cosine"] for r in known], dtype=np.float64)
    neg = np.array([r["top_cosine"] for r in unknown], dtype=np.float64)
    if not len(pos) or not len(neg):
        raise ValueError("AUC requires positive and negative examples")
    return round(float(((pos[:, None] > neg[None, :]).sum()
                       + 0.5 * (pos[:, None] == neg[None, :]).sum())
                       / (len(pos) * len(neg))), 6)


def _fit_conditions(train: list[Memory], seed: int, cue_units: int,
                    memory_units: int, density: float) -> dict[str, EncoderOverlay]:
    ordered = order_by_episode(train, seed=20261008)
    trained: dict[str, EncoderOverlay] = {}
    mismatched = shuffled_targets(ordered, seed + 113)
    for method in METHODS:
        mode = "token" if method in ("token_shuffled", "token_unmodified") else method
        encoder = CueEncoder(mode, cue_units).fit(ordered)
        model = EncoderOverlay(encoder, memory_units, density, seed)
        if method != "token_unmodified":
            for i, memory in enumerate(ordered):
                model.imprint(
                    memory, mismatched[i] if method == "token_shuffled" else None
                )
        trained[method] = model
    return trained


def evaluate_seed(memories: list[Memory], seed: int,
                  cue_units: int = 512, memory_units: int = 256,
                  density: float = 0.55) -> dict:
    """Comparison is prespecified; no fit and no threshold uses test episodes."""
    train, validation, test = episode_split(memories, seed)
    # Use only training events for learned document frequency.
    ordered = order_by_episode(train, seed=20261008)
    positive = probes(ordered, "swap")
    rng = np.random.default_rng(seed + 227)
    indices = rng.permutation(len(positive))
    cut = len(indices) // 2
    calibration_known = [positive[int(i)] for i in indices[:cut]]
    testing_known = [positive[int(i)] for i in indices[cut:]]
    calibration_unknown = probes(validation, "swap")
    testing_unknown = probes(test, "swap")
    if min(map(len, (calibration_known, testing_known,
                     calibration_unknown, testing_unknown))) == 0:
        raise ValueError("incomplete episode split or cue probes")

    models = _fit_conditions(ordered, seed, cue_units, memory_units, density)
    results: dict = {}
    for method, model in models.items():
        known_cal = scores(model, ordered, calibration_known)
        unknown_cal = scores(model, ordered, calibration_unknown)
        threshold = calibrate(known_cal, unknown_cal)
        challenges = {}
        for challenge in CHALLENGES:
            if challenge == "swap":
                kp, up = testing_known, testing_unknown
            else:
                # Same event IDs as swapped primary test, and corresponding
                # test-negative episodes. No second calibration is performed.
                allowed = {p.event_id for p in testing_known}
                kp = [p for p in probes(ordered, challenge) if p.event_id in allowed]
                up = probes(test, challenge)
            known = scores(model, ordered, kp)
            unknown = scores(model, ordered, up)
            if not known or not unknown:
                raise ValueError(f"empty {challenge} comparison")
            pos = _positive_summary(known, threshold)
            neg = _negative_summary(unknown, threshold)
            challenges[challenge] = {
                "known": pos,
                "unknown": neg,
                "balanced_accuracy": round(
                    (pos["true_positive_rate"] +
                     1 - neg["false_acceptance_rate"]) / 2, 6
                ),
                "discrimination_auc": _auc(known, unknown),
            }
        results[method] = {
            "threshold_from_calibration": round(threshold, 8),
            "calibration": {
                "known": _positive_summary(known_cal, threshold),
                "unknown": _negative_summary(unknown_cal, threshold),
            },
            "tests": challenges,
            "train_events_used_for_encoder_fit": model.encoder.fitted_events,
            "topology_edges": int(model.mask.sum()),
            "updates": model.network.updates,
        }
    return {
        "seed": seed,
        "episodes": {
            "imprinted": sorted({m.episode_id for m in ordered}),
            "calibration_unknown": sorted({m.episode_id for m in validation}),
            "test_unknown": sorted({m.episode_id for m in test}),
        },
        "count": {
            "train_events": len(ordered),
            "calibration_unknown_events": len(validation),
            "test_unknown_events": len(test),
            "calibration_known_probes": len(calibration_known),
            "test_known_probes": len(testing_known),
        },
        "methods": results,
        "lexical_lookup": {
            challenge: lexical_retrieval(
                ordered, testing_known if challenge == "swap"
                else [p for p in probes(ordered, challenge)
                      if p.event_id in {x.event_id for x in testing_known}]
            )
            for challenge in CHALLENGES
        },
    }
