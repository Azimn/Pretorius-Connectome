"""Portable source-pinned L1 records and fit-split-pinned TF-IDF L2 cache.

Owner: Pretorius-Connectome. BioCircuit imports this code from a pinned
checkout rather than maintaining a second normalization/vectorization engine.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer

from pretorius_connectome.imprinting import load_v12
from pretorius_connectome.pilot02 import episode_split
from pretorius_connectome.shared_memory import (
    SOURCE_BLOB, SOURCE_COMMIT, git_blob_sha, read_l1,
)

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "memories/current/Pretorius_v12_450_Events_Complete.jsonl"
SIDECARS = ROOT / "memories/annotations/v12_450_sidecars.jsonl"
SIDECAR_BLOB = "ad32025166c382caf13e07c7e3b0863eb89e1adb"
SCHEMA = "pretorius.shared-features.v1"
L1_DIR = ROOT / "artifacts/shared_memory/v1"
L1_ARCHIVE = L1_DIR / "pretorius_l1_v1.jsonl.gz"
L1_MANIFEST = L1_DIR / "manifest.json"
# Immutable published L1 v1 Git blobs, independent of mutable L2 manifests.
L1_ARCHIVE_BLOB = "5381c3c22ab224bb6ebf9f037489c1b9be564888"
L1_MANIFEST_BLOB = "7a8d05b46b025b1bfe44339b3edbee040df22394"
ENCODER = "sklearn-tfidf-word12-v1"
NORMALIZATION = "canonical-pretorius-autobiography-l1/1"
SHARDS = ("records.jsonl", "vocabulary.json", "idf.npy", "docs.npz")


def git_blob(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canon_json(data) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _pinned_l1_records() -> list[dict]:
    if (git_blob_sha(L1_ARCHIVE.read_bytes()) != L1_ARCHIVE_BLOB
            or git_blob_sha(L1_MANIFEST.read_bytes()) != L1_MANIFEST_BLOB):
        raise ValueError("Canonical L1 artifact differs from published Git blob")
    return read_l1(L1_ARCHIVE, L1_MANIFEST)


def _source_records(events_path: Path, sidecars_path: Path) -> list[dict]:
    """Consume the already committed immutable L1 artifact, never rebuild L1."""
    if (git_blob_sha(events_path.read_bytes()) != SOURCE_BLOB
            or git_blob_sha(sidecars_path.read_bytes()) != SIDECAR_BLOB):
        raise ValueError("Unpinned source or sidecar bytes")
    rows = _pinned_l1_records()
    original = [json.loads(s) for s in events_path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != 450 or any(
        row["event_id"] != original[i]["event_id"]
        or row["memory_text"] != original[i]["memory_text"]
        or row["episode_id"] != original[i]["episode_id"]
        for i, row in enumerate(rows)
    ):
        raise ValueError("Published L1 does not match pinned raw source")
    return rows


def _vectorizer(vocabulary=None):
    return TfidfVectorizer(
        stop_words="english", ngram_range=(1, 2),
        max_features=8192, sublinear_tf=True, norm="l2",
        dtype=np.float64, vocabulary=vocabulary,
    )


def _encoder_hash() -> str:
    return sha256(Path(__file__).read_bytes())


def build_cache(destination: str | Path, seed: int = 31,
                events_path: str | Path = SOURCE,
                sidecars_path: str | Path = SIDECARS) -> dict:
    dest = Path(destination)
    dest.mkdir(parents=True, exist_ok=True)
    rows = _source_records(Path(events_path), Path(sidecars_path))
    originals = load_v12(events_path, sidecars_path)
    train, validation, test = episode_split(originals, int(seed))
    buckets = {
        "train": [x.event_id for x in train],
        "validation": [x.event_id for x in validation],
        "test": [x.event_id for x in test],
    }
    group_map = {x["event_id"]: x["episode_id"] for x in rows}
    sets = [{group_map[k] for k in buckets[n]} for n in ("train", "validation", "test")]
    if not all(sets) or any(sets[a] & sets[b] for a in range(3) for b in range(a+1,3)):
        raise ValueError("Episode split contaminated")
    encoder = _vectorizer()
    encoder.fit([x.memory_text for x in train])  # IDF fit ONLY on train
    docs = encoder.transform([x["memory_text"] for x in rows]).tocsr()
    docs.sort_indices()
    vocab = {str(term): int(index) for term, index in encoder.vocabulary_.items()}
    records_path = dest / SHARDS[0]
    records_path.write_text("".join(canon_json(x)+"\n" for x in rows), encoding="utf-8")
    (dest / SHARDS[1]).write_text(canon_json(vocab)+"\n", encoding="utf-8")
    with (dest / SHARDS[2]).open("wb") as handle:
        np.save(handle, encoder.idf_, allow_pickle=False)
    sparse.save_npz(dest / SHARDS[3], docs, compressed=True)
    manifest = {
        "l1_archive_sha256": sha256(L1_ARCHIVE.read_bytes()),
        "l1_manifest_sha256": sha256(L1_MANIFEST.read_bytes()),
        "schema_version": SCHEMA,
        "source_repo": "Azimn/Pretorius-Connectome",
        "source_commit": SOURCE_COMMIT,
        "source_git_blob": SOURCE_BLOB,
        "annotation_git_blob": SIDECAR_BLOB,
        "record_ids_ordered": [x["event_id"] for x in rows],
        "normalization_version": NORMALIZATION,
        "encoder_name": ENCODER,
        "encoder_code_hash": _encoder_hash(),
        "model_revision": None, "model_checksum": None,
        "fit_event_ids": buckets["train"],
        "split_event_ids": buckets,
        "split_episode_ids": {name: sorted({group_map[k] for k in ids})
                              for name, ids in buckets.items()},
        "vector_dim": int(docs.shape[1]),
        "dtype": str(docs.dtype),
        "feature_normalization": "sklearn-l2",
        "random_seed": int(seed),
        "feature_rows": int(docs.shape[0]),
        "nonzero_features": int(docs.nnz),
        "shard_sha256": {name: sha256((dest / name).read_bytes()) for name in SHARDS},
        "limitations": "TF-IDF lexical features, not semantic entailment. Validation/test source narratives were transformed using train-only IDF, never used for fitting.",
    }
    (dest / "manifest.json").write_text(canon_json(manifest)+"\n", encoding="utf-8")
    return manifest


class SharedCache:
    """Fail-closed loader with identical fitted query transformation."""
    def __init__(self, directory: str | Path):
        self.path = Path(directory)
        manifest = json.loads((self.path / "manifest.json").read_text(encoding="utf-8"))
        self.manifest = manifest
        if (manifest.get("l1_archive_sha256") != sha256(L1_ARCHIVE.read_bytes())
                or manifest.get("l1_manifest_sha256") != sha256(L1_MANIFEST.read_bytes())
                or manifest.get("schema_version") != SCHEMA
                or manifest.get("source_repo") != "Azimn/Pretorius-Connectome"
                or manifest.get("source_commit") != SOURCE_COMMIT
                or manifest.get("source_git_blob") != SOURCE_BLOB
                or manifest.get("annotation_git_blob") != SIDECAR_BLOB
                or manifest.get("normalization_version") != NORMALIZATION
                or manifest.get("encoder_name") != ENCODER
                or manifest.get("encoder_code_hash") != _encoder_hash()
                or manifest.get("model_revision") is not None
                or manifest.get("model_checksum") is not None
                or manifest.get("dtype") != "float64"
                or manifest.get("feature_normalization") != "sklearn-l2"):
            raise ValueError("Incompatible shared memory manifest")
        if set(manifest.get("shard_sha256", {})) != set(SHARDS):
            raise ValueError("Incomplete shard manifest")
        for name in SHARDS:
            if sha256((self.path / name).read_bytes()) != manifest["shard_sha256"][name]:
                raise ValueError("Corrupt or mismatched shared memory shard: " + name)
        self.records = tuple(json.loads(line) for line in
                             (self.path / SHARDS[0]).read_text(encoding="utf-8").splitlines())
        self.ids = tuple(x["event_id"] for x in self.records)
        if (len(self.ids) != 450 or len(set(self.ids)) != 450
                or list(self.ids) != manifest["record_ids_ordered"]
                or [x["chronological_order"] for x in self.records] != list(range(1, 451))
                or any(x["provenance"] != "reconstructed" for x in self.records)):
            raise ValueError("L1 records or order invalid")
        # A bundle's own checksums are not a trusted source of truth. Compare
        # every projected field to the separately pinned, immutable L1 bytes.
        canonical_rows = _pinned_l1_records()
        if list(self.records) != canonical_rows:
            raise ValueError("Shared cache records differ from canonical pinned L1")

        splits = manifest["split_event_ids"]
        if set(splits) != {"train", "validation", "test"}:
            raise ValueError("Invalid split names")
        if set(splits["train"]) != set(manifest["fit_event_ids"]):
            raise ValueError("Inconsistent fit split")
        if sum(len(splits[k]) for k in splits) != 450 or set().union(*(set(v) for v in splits.values())) != set(self.ids):
            raise ValueError("Split event coverage incorrect")
        episodes = {x["event_id"]: x["episode_id"] for x in self.records}
        group_sets = []
        for name, ids in splits.items():
            if len(ids) != len(set(ids)):
                raise ValueError("Duplicate split IDs")
            ep = {episodes[k] for k in ids}
            if sorted(ep) != manifest["split_episode_ids"][name]:
                raise ValueError("Episode split label mismatch")
            group_sets.append(ep)
        if any(group_sets[i] & group_sets[j] for i in range(3) for j in range(i+1,3)):
            raise ValueError("Episode leakage")
        # Split labels must agree with the actual fixed episode-partition
        # algorithm. A changed manifest cannot move validation into training.
        expected_train, expected_validation, expected_test = episode_split(
            load_v12(SOURCE, SIDECARS), int(manifest["random_seed"])
        )
        for split_name, expected in (
            ("train", expected_train), ("validation", expected_validation),
            ("test", expected_test)
        ):
            if splits[split_name] != [x.event_id for x in expected]:
                raise ValueError("Declared training split differs from seeded episode partition")
        vocab = json.loads((self.path / SHARDS[1]).read_text(encoding="utf-8"))
        with (self.path / SHARDS[2]).open("rb") as handle:
            idf = np.load(handle, allow_pickle=False)
        self.docs = sparse.load_npz(self.path / SHARDS[3]).tocsr()
        if (len(vocab) != manifest["vector_dim"]
                or sorted(vocab.values()) != list(range(len(vocab)))
                or idf.shape != (len(vocab),)
                or not np.isfinite(idf).all()
                or self.docs.shape != (450, len(vocab))
                or self.docs.dtype != np.float64
                or not np.isfinite(self.docs.data).all()
                or self.docs.nnz != manifest["nonzero_features"]):
            raise ValueError("Invalid feature shape or statistics")
        # Refit from the independently verified, DECLARED TRAINING records.
        # Verifying only self-consistent shard hashes allows a malicious cache
        # fitted on held-out narratives to pass all existing checks.
        canonical = {row["event_id"]: row["memory_text"] for row in canonical_rows}
        trained = _vectorizer()
        trained.fit([canonical[event_id] for event_id in splits["train"]])
        if (vocab != trained.vocabulary_
                or not np.array_equal(idf, trained.idf_)):
            raise ValueError("Cached vocabulary/IDF was not fitted from train-only narratives")
        self.encoder = _vectorizer(vocabulary=vocab)
        # Restore the fitted transformation WITHOUT refitting on held-out text.
        self.encoder._tfidf = TfidfTransformer(norm="l2", use_idf=True,
                                              smooth_idf=True, sublinear_tf=True)
        self.encoder.idf_ = idf
        self.positions = {eid: n for n, eid in enumerate(self.ids)}
        # Fail closed if the serialized features are not exactly reproducible.
        reproduced = self.encoder.transform([x["memory_text"] for x in self.records]).tocsr()
        if (reproduced.shape != self.docs.shape
                or not np.allclose((reproduced - self.docs).data, 0, atol=1e-12)):
            raise ValueError("Cached TF-IDF documents inconsistent with restored encoder")

    def subset(self, event_ids: list[str] | tuple[str, ...]):
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("Duplicated requested event IDs")
        try:
            indices = [self.positions[eid] for eid in event_ids]
        except KeyError as error:
            raise ValueError("Requested unknown event ID") from error
        return self.docs[indices].tocsr()

    def query(self, text: str):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Query must be nonempty")
        return self.encoder.transform([text]).tocsr()

    def require_training_set(self, event_ids) -> None:
        if set(event_ids) != set(self.manifest["fit_event_ids"]):
            raise ValueError("Cache IDF fit events differ from active training set")

    def assert_original(self, events_path: str | Path = SOURCE,
                        sidecars_path: str | Path = SIDECARS) -> None:
        if (git_blob(Path(events_path).read_bytes()) != SOURCE_BLOB
                or git_blob(Path(sidecars_path).read_bytes()) != SIDECAR_BLOB):
            raise ValueError("Shared cache does not match source bytes")
