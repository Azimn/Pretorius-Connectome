"""Shared, source-pinned Pretorius L1 records and reusable BC01 lexical L2.

Canonical exporter only. Never fit TF-IDF on held-out episodes, leak
labels into features, or interpret the BC hashed vectors as semantics.
"""
from __future__ import annotations

from hashlib import blake2b, sha1, sha256
from pathlib import Path
import json
import re

import numpy as np

SCHEMA = "pretorius-shared-memory-v1"
ENCODER = "persona-net-experience-lexical-bc01-v1"
DIM = 256
SOURCE_BLOB = "718dcc2d5ba4feccdef1690d447edfcebaa9bfb5"
ANNOTATION_BLOB = "ad32025166c382caf13e07c7e3b0863eb89e1adb"
SOURCE_COMMIT = "60c8ee8dd78158a8f2d7c73b9eccb3b1f121291f"
TOKEN_REGEX = r"[a-z0-9']+"
ENCODER_SPEC = (
    "lowercase; regex [a-z0-9']+; unigrams plus adjacent token bigrams "
    "joined by ::; blake2b-8 little endian index modulo 256, "
    "sign from bit 8; np.float32 counts then numpy l2 normalize"
)


def _git_sha(raw: bytes) -> str:
    return sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


def _file_sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def lexical_vector(text: str) -> np.ndarray:
    """Bit-compatible with BC01 ExperienceEncoder(sensory_dim=256)."""
    tokens = re.findall(TOKEN_REGEX, text.lower())
    terms = tokens + [a + "::" + b for a, b in zip(tokens, tokens[1:])]
    x = np.zeros(DIM, dtype=np.float32)
    for term in terms:
        value = int.from_bytes(blake2b(term.encode("utf-8"), digest_size=8).digest(),
                               "little", signed=False)
        x[value % DIM] += 1.0 if (value >> 8) & 1 == 0 else -1.0
    if terms:
        norm = float(np.linalg.norm(x))
        if norm > 1e-8:
            x /= norm
    return x


def _lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def export_bundle(events: Path, sidecars: Path, destination: Path) -> dict:
    """Export immutable original event fields, candidate annotations and L2."""
    if _git_sha(events.read_bytes()) != SOURCE_BLOB:
        raise ValueError("source Git blob not pinned")
    if _git_sha(sidecars.read_bytes()) != ANNOTATION_BLOB:
        raise ValueError("sidecar Git blob not pinned")
    source = _lines(events)
    annotation = _lines(sidecars)
    if len(source) != 450 or len(annotation) != 450:
        raise ValueError("expected 450 matching memories and annotations")
    ids = [r["event_id"] for r in source]
    if len(set(ids)) != 450 or len({r["episode_id"] for r in source}) != 27:
        raise ValueError("corpus IDs or episodes changed")
    if any(a["event_id"] != r["event_id"] or
           a["annotation_status"] != "unreviewed_candidate" or
           a["cue_surface_forms"] != r["recall_cues"]
           for r, a in zip(source, annotation)):
        raise ValueError("source-sidecar linkage changed")
    destination.mkdir(parents=True, exist_ok=True)
    source_out = destination / "l1_records.jsonl"
    vectors_out = destination / "bc01_lexical_256.npy"
    with source_out.open("w", encoding="utf-8") as f:
        for record, sidecar in zip(source, annotation):
            row = {"source": record, "annotation": {
                "cue_ids": sidecar["cue_ids"],
                "status": sidecar["annotation_status"],
                "surface_forms": sidecar["cue_surface_forms"],
            }}
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    sensory = np.stack([lexical_vector(r["memory_text"]) for r in source])
    np.save(vectors_out, sensory, allow_pickle=False)
    manifest = {
        "schema_version": SCHEMA,
        "source_repo": "Azimn/Pretorius-Connectome",
        "source_commit": SOURCE_COMMIT,
        "source_git_blob": SOURCE_BLOB,
        "annotation_git_blob": ANNOTATION_BLOB,
        "record_ids_ordered": ids,
        "normalization_version": "exact-source-record-plus-candidate-sidecar-v1",
        "encoder_name": ENCODER,
        "encoder_code_hash": sha256(ENCODER_SPEC.encode()).hexdigest(),
        "model_revision": None,
        "fit_event_ids": [],
        "fit_scope": "stateless lexical hashing; no train data fit",
        "vector_dim": DIM,
        "dtype": str(sensory.dtype),
        "feature_normalization": "L2 float32, same as BC01 ExperienceEncoder",
        "random_seed": None,
        "records_count": len(ids),
        "episodes_count": 27,
        "files": {
            source_out.name: _file_sha(source_out),
            vectors_out.name: _file_sha(vectors_out),
        },
        "limitations": (
            "Deterministic hashed lexical vectors are not semantic embeddings. "
            "No TF-IDF or episode-global IDF was fit; FlyWire retains train-only "
            "TF-IDF fit. Sidecar cues are unreviewed candidates."
        ),
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def load_bundle(destination: Path, *, full_verification: bool = True
               ) -> tuple[dict, list[dict], np.ndarray]:
    """Reject tampering, split/corpus drift, unsupported or mismatched cache."""
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != SCHEMA or
        manifest.get("source_repo") != "Azimn/Pretorius-Connectome" or
        manifest.get("source_git_blob") != SOURCE_BLOB or
        manifest.get("annotation_git_blob") != ANNOTATION_BLOB or
        manifest.get("encoder_name") != ENCODER or
        manifest.get("encoder_code_hash") != sha256(ENCODER_SPEC.encode()).hexdigest()
        or manifest.get("fit_event_ids") != [] or
        manifest.get("fit_scope") != "stateless lexical hashing; no train data fit"
        or manifest.get("vector_dim") != DIM or
        manifest.get("dtype") != "float32" or
        manifest.get("normalization_version") !=
            "exact-source-record-plus-candidate-sidecar-v1"):
        raise ValueError("incompatible shared cache version/encoder/source/split")
    for filename in ("l1_records.jsonl", "bc01_lexical_256.npy"):
        if _file_sha(destination / filename) != manifest.get("files", {}).get(filename):
            raise ValueError("shared cache shard hash mismatch: " + filename)
    data = _lines(destination / "l1_records.jsonl")
    with (destination / "bc01_lexical_256.npy").open("rb") as f:
        vectors = np.load(f, allow_pickle=False)
    ids = [r["source"]["event_id"] for r in data]
    if (len(data) != 450 or len(set(ids)) != 450 or
        ids != manifest.get("record_ids_ordered") or
        len({r["source"]["episode_id"] for r in data}) != 27 or
        vectors.shape != (450, DIM) or vectors.dtype != np.dtype("float32") or
        not np.all(np.isfinite(vectors)) or
        any(r["annotation"]["status"] != "unreviewed_candidate" or
            r["annotation"]["surface_forms"] != r["source"]["recall_cues"]
            for r in data)):
        raise ValueError("cache coverage/order/feature/provenance invalid")
    if full_verification:
        rebuilt = np.stack([
            lexical_vector(r["source"]["memory_text"]) for r in data
        ])
        if not np.array_equal(vectors, rebuilt):
            raise ValueError("cached lexical features differ from source")
    return manifest, data, vectors


def bundle_to_memories(rows: list[dict]):
    """Adapt validated L1 back to the existing FlyWire retrieval source type."""
    from pretorius_connectome.imprinting import Memory, Cue
    return [
        Memory(r["source"]["event_id"], r["source"]["episode_id"],
               r["source"]["memory_text"],
               tuple(Cue("surface:" + " ".join(s.casefold().split()), s)
                     for s in r["source"]["recall_cues"]))
        for r in rows
    ]
