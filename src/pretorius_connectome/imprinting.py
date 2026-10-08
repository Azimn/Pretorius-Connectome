"""Reproducible associative imprinting on a fixed synthetic wiring mask.

This is a constrained engineering assay, NOT a FlyWire-brain simulation,
natural-language episodic recall, or evidence of a biological engram.
The evaluator's target codebook is never placed inside the model.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from hashlib import blake2b, sha256
import json
from pathlib import Path
import re

import numpy as np


@dataclass(frozen=True)
class Cue:
    cue_id: str
    surface: str


@dataclass(frozen=True)
class Memory:
    event_id: str
    episode_id: str
    memory_text: str
    cues: tuple[Cue, ...]


def load_v12(events_path: str | Path, sidecars_path: str | Path, *,
             shared_manifest: str | Path | None = None) -> list[Memory]:
    """Load verified L0 or portable L1 with identical existing cue handling.

    shared_manifest=None leaves every historical experiment unchanged.
    """
    def lines(path: str | Path) -> list[dict]:
        with Path(path).open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    if shared_manifest is None:
        events = lines(events_path)
    else:
        from pretorius_connectome.shared_memory import read_l1
        events = read_l1(events_path, shared_manifest)
    sidecars = lines(sidecars_path)
    if len(events) != len(sidecars) or len(events) != 450:
        raise ValueError("Pilot 01 requires exactly 450 aligned v12 records")
    output: list[Memory] = []
    seen: set[str] = set()
    for event, sidecar in zip(events, sidecars):
        event_id = event["event_id"]
        if event_id in seen or event_id != sidecar["event_id"]:
            raise ValueError("duplicate or misaligned v12 event IDs")
        seen.add(event_id)
        surfaces, cue_ids = sidecar["cue_surface_forms"], sidecar["cue_ids"]
        if (not surfaces or not cue_ids
                or surfaces != event["recall_cues"]
                or sidecar["annotation_status"] != "unreviewed_candidate"):
            raise ValueError(f"invalid cue sidecar for {event_id}")
        if not event["memory_text"].strip():
            raise ValueError(f"empty memory text for {event_id}")
        # v12 sidecars may attach MORE associative registry IDs than surface cues.
        # They are independent candidates, never zipped to literal phrases.
        literal = tuple(
            Cue("surface:" + " ".join(surface.casefold().split()), surface)
            for surface in surfaces
        )
        candidates = tuple(Cue(identifier, "") for identifier in cue_ids)
        output.append(Memory(
            event_id, event["episode_id"], event["memory_text"],
            literal + candidates,
        ))
    return output


def _digest(value: str) -> bytes:
    return blake2b(value.encode("utf-8"), digest_size=16, person=b"pretorius-p01").digest()


def cue_vector(cues: tuple[Cue, ...], units: int) -> np.ndarray:
    """Stable signed feature hashing of known cues; no event ID is a feature."""
    if units <= 0 or not cues:
        raise ValueError("cue units and cues must be nonempty")
    values = np.zeros(units, dtype=np.float32)
    for cue in cues:
        words = re.findall(r"[^\W_]+(?:['’][^\W_]+)?", cue.surface.casefold())
        features = [f"id:{cue.cue_id}"]
        if words:
            features.append(f"phrase:{' '.join(words)}")
        features.extend(f"token:{word}" for word in words)
        features.extend(f"pair:{a}:{b}" for a, b in zip(words, words[1:]))
        for feature in features:
            digest = _digest(feature)
            index = int.from_bytes(digest[:8], "little") % units
            values[index] += 1 if digest[8] & 1 else -1
    magnitude = np.linalg.norm(values)
    if not magnitude:
        raise ValueError("cue features cancelled to zero")
    return values / magnitude


def content_fingerprint(text: str, units: int) -> np.ndarray:
    """Opaque bipolar assay target derived only from the first-person narrative.

    This cryptographic fingerprint is NOT a semantic embedding or a memory.
    A separate evaluator-only codebook is required to score identification.
    """
    if units <= 0 or not text:
        raise ValueError("content fingerprint requires text and positive dimensions")
    key = sha256(text.encode("utf-8")).digest()
    stream = bytearray()
    counter = 0
    while len(stream) * 8 < units:
        stream.extend(sha256(key + counter.to_bytes(4, "little")).digest())
        counter += 1
    bits = np.unpackbits(np.frombuffer(bytes(stream), dtype=np.uint8))[:units]
    return np.where(bits == 1, 1.0, -1.0).astype(np.float32) / np.sqrt(units)


class SynapticOverlay:
    """Learned changes restricted to edges in an immutable synthetic topology.

    The fixed boolean mask defines allowed synapses. Only overlay weights change.
    No target labels, text, cue registry, or episodic database live in this object.
    """

    def __init__(self, cue_units: int = 512, memory_units: int = 256,
                 density: float = 0.55, seed: int = 0):
        if cue_units <= 0 or memory_units <= 0 or not 0 < density <= 1:
            raise ValueError("invalid topology")
        self.cue_units = cue_units
        self.memory_units = memory_units
        rng = np.random.default_rng(seed)
        mask = rng.random((memory_units, cue_units)) < density
        mask.flags.writeable = False
        self.mask = mask
        self.weights = np.zeros(mask.shape, dtype=np.float32)
        self.updates = 0

    def imprint(self, cues: tuple[Cue, ...], text: str, rate: float = 1.0) -> None:
        if not 0 < rate <= 1.0:
            raise ValueError("rate must be in (0, 1]")
        x = cue_vector(cues, self.cue_units)
        y = content_fingerprint(text, self.memory_units)
        active = np.flatnonzero(x)
        self.weights[:, active] += (
            rate * y[:, None] * x[active][None, :] * self.mask[:, active]
        )
        self.updates += 1

    def readout(self, cues: tuple[Cue, ...]) -> np.ndarray:
        x = cue_vector(cues, self.cue_units)
        return self.weights @ x

    def reset_overlay(self) -> None:
        self.weights.fill(0.0)
        self.updates = 0


def order_by_episode(memories: list[Memory], seed: int = 0) -> list[Memory]:
    """Seeded round-robin over episodes, avoiding a childhood-only 50-item run."""
    rng = np.random.default_rng(seed)
    groups: dict[str, list[Memory]] = defaultdict(list)
    for memory in memories:
        groups[memory.episode_id].append(memory)
    for group in groups.values():
        rng.shuffle(group)
    episode_ids = sorted(groups)
    rng.shuffle(episode_ids)
    output: list[Memory] = []
    while len(output) < len(memories):
        for episode_id in episode_ids:
            if groups[episode_id]:
                output.append(groups[episode_id].pop())
    return output


def decode(readout: np.ndarray, candidates: list[Memory], units: int) -> str | None:
    """Oracle evaluator: choose closest candidate narrative fingerprint.

    Candidate texts live ONLY in the evaluator, never the plasticity overlay.
    This is a latent-code identification assay, not an executable recall policy.
    """
    if not candidates or not np.any(readout):
        return None
    targets = np.stack([content_fingerprint(m.memory_text, units) for m in candidates])
    scores = targets @ readout
    return candidates[int(np.argmax(scores))].event_id


def evaluate(overlay: SynapticOverlay, candidates: list[Memory],
             probes: list[Memory] | None = None) -> dict:
    """Score trained events using only one known source cue per probe."""
    if probes is None:
        probes = candidates
    if not probes:
        raise ValueError("probes must not be empty")
    targets = np.stack([
        content_fingerprint(m.memory_text, overlay.memory_units) for m in candidates
    ])
    candidate_ids = [m.event_id for m in candidates]
    correct = 0
    cosines = []
    for memory in probes:
        out = overlay.readout((memory.cues[0],))
        if not np.any(out):
            cosines.append(0.0)
            continue
        index = int(np.argmax(targets @ out))
        if candidate_ids[index] == memory.event_id:
            correct += 1
        desired = content_fingerprint(memory.memory_text, overlay.memory_units)
        cosines.append(float(desired @ out / (np.linalg.norm(out) or 1.0)))
    return {
        "n_probes": len(probes),
        "top1": round(correct / len(probes), 6),
        "mean_target_cosine": round(float(np.mean(cosines)), 6),
    }


def retrieval_only(candidates: list[Memory], probes: list[Memory]) -> float:
    """Non-neural exact cue-ID lookup, reported as a separate reference."""
    reverse: dict[str, list[str]] = defaultdict(list)
    for item in candidates:
        for cue in item.cues:
            reverse[cue.cue_id].append(item.event_id)
    correct = sum(
        bool(reverse.get(item.cues[0].cue_id))
        and reverse[item.cues[0].cue_id][0] == item.event_id
        for item in probes
    )
    return round(correct / len(probes), 6)


def shuffled_targets(memories: list[Memory], seed: int) -> list[str]:
    """Deterministic mismatched narrative control, with no label access in overlay."""
    rng = np.random.default_rng(seed)
    if len(memories) < 2:
        raise ValueError("shuffled control requires at least two memories")
    perm = rng.permutation(len(memories))
    shifted = np.empty(len(memories), dtype=np.int64)
    shifted[perm] = np.roll(perm, 1)  # fixed-point-free shuffled assignment
    return [memories[int(index)].memory_text for index in shifted]
