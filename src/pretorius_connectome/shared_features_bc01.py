"""BC01 lexical L2 cache on top of the already-published canonical L1 archive.

Distinct from FlyWire's train-fitted TF-IDF features. L1 remains immutable and
its manifest untouched. The BC01 hash is stateless and exact on every split.
"""
from __future__ import annotations

from hashlib import sha256, blake2b
from pathlib import Path
import json
import re

import numpy as np

from pretorius_connectome.shared_memory import SOURCE_BLOB, read_l1

SCHEMA = "pretorius-bc01-lexical-l2/1"
ENCODER = "persona-net-ExperienceEncoder-sensory256-v1"
SPEC = (
    "regex [a-z0-9']+ lowercase; adjacent unigrams+bigrams a::b; "
    "blake2b digest_size=8 interpreted little-endian; signed bit8; "
    "float32 counts then numpy norm L2"
)
DIM = 256
FILENAME = "bc01_sensory_256.npy"
MANIFEST = "bc01_l2_manifest.json"


def encode_bc_sensory(text: str) -> np.ndarray:
    """Match BioCircuit's existing ExperienceEncoder exactly, no new model."""
    tokens = re.findall(r"[a-z0-9']+", text.lower())
    terms = tokens + [a + "::" + b for a, b in zip(tokens, tokens[1:])]
    vector = np.zeros(DIM, dtype=np.float32)
    for term in terms:
        value = int.from_bytes(blake2b(term.encode("utf-8"), digest_size=8).digest(),
                               "little", signed=False)
        index = value % DIM
        vector[index] += 1.0 if ((value >> 8) & 1) else -1.0
    if terms:
        norm = float(np.linalg.norm(vector))
        if norm > 1e-8:
            vector /= norm
    return vector


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def build_bc01_cache(folder: Path) -> dict:
    """Use existing L1, do not change its payload or manifest."""
    folder = Path(folder)
    archive = folder / "pretorius_l1_v1.jsonl.gz"
    l1_manifest = folder / "manifest.json"
    rows = read_l1(archive, l1_manifest)
    array = np.stack([encode_bc_sensory(row["memory_text"]) for row in rows])
    np.save(folder / FILENAME, array, allow_pickle=False)
    result = {
        "schema_version": SCHEMA,
        "source_git_blob": SOURCE_BLOB,
        "parent_l1_manifest_sha256": _sha(l1_manifest),
        "parent_l1_archive_sha256": _sha(archive),
        "record_ids_ordered": [row["event_id"] for row in rows],
        "encoder_name": ENCODER,
        "encoder_code_hash": sha256(SPEC.encode()).hexdigest(),
        "fit_event_ids": [],
        "fit_scope": "stateless hash, no fit/IDF or episode exposure",
        "model_revision": None,
        "dtype": "float32",
        "feature_normalization": "float32 L2 per event",
        "vector_dim": DIM,
        "random_seed": None,
        "records": len(rows),
        "artifact_name": FILENAME,
        "artifact_sha256": _sha(folder / FILENAME),
        "limitations": "Lexical hashing, not semantic embeddings; distinct from FlyWire split-fitted TF-IDF.",
    }
    (folder / MANIFEST).write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    check, _ = load_bc01_cache(folder)
    if check != result:
        raise AssertionError("L2 round-trip altered manifest")
    return result


def load_bc01_cache(folder: Path) -> tuple[dict, np.ndarray]:
    folder = Path(folder)
    l1_manifest = folder / "manifest.json"
    archive = folder / "pretorius_l1_v1.jsonl.gz"
    meta = json.loads((folder / MANIFEST).read_text(encoding="utf-8"))
    if (meta.get("schema_version") != SCHEMA or
        meta.get("source_git_blob") != SOURCE_BLOB or
        meta.get("encoder_name") != ENCODER or
        meta.get("encoder_code_hash") != sha256(SPEC.encode()).hexdigest() or
        meta.get("fit_event_ids") != [] or
        meta.get("fit_scope") != "stateless hash, no fit/IDF or episode exposure" or
        meta.get("vector_dim") != DIM or meta.get("dtype") != "float32" or
        meta.get("model_revision") is not None or
        meta.get("parent_l1_manifest_sha256") != _sha(l1_manifest) or
        meta.get("parent_l1_archive_sha256") != _sha(archive) or
        meta.get("artifact_sha256") != _sha(folder / FILENAME)):
        raise ValueError("BC01 L2 source/encoder/fit/shard fingerprint mismatch")
    rows = read_l1(archive, l1_manifest)
    if [r["event_id"] for r in rows] != meta.get("record_ids_ordered"):
        raise ValueError("BC01 L2 event order mismatch")
    with (folder / FILENAME).open("rb") as f:
        values = np.load(f, allow_pickle=False)
    if (values.shape != (450, DIM) or values.dtype != np.dtype("float32")
        or not np.all(np.isfinite(values))):
        raise ValueError("BC01 L2 shape/dtype invalid")
    restored = np.stack([encode_bc_sensory(r["memory_text"]) for r in rows])
    if not np.array_equal(values, restored):
        raise ValueError("BC01 L2 cache does not reproduce pinned source")
    return meta, values
