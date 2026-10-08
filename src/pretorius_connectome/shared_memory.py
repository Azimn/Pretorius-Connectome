"""Portable L1 autobiography handoff, intentionally NOT a shared embedding.

The original v12 JSONL is the only biographical source of truth. This module
publishes a strictly verified, reproducibly compressed view for other projects.
No identifiers, annotations or action labels enter neural input vectors here.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

SCHEMA = "pretorius-autobiography-l1/1"
SOURCE_REPO = "Azimn/Pretorius-Connectome"
SOURCE_COMMIT = "597fb23473a60eecf2e1b50f79c22bfbea816be5"
SOURCE_BLOB = "718dcc2d5ba4feccdef1690d447edfcebaa9bfb5"
SOURCE_EVENTS = 450
SOURCE_EPISODES = 27
FIELD_NAMES = (
    "event_id", "episode_id", "chronological_order", "memory_text",
    "title", "approximate_date", "participants", "locations", "recall_cues",
    "provenance", "source_anchors", "links_to_prior_events",
    "relationship_changes", "belief_changes", "observations", "decisions",
    "consequences",
)
REQUIRED = (
    "event_id", "episode_id", "chronological_order", "memory_text",
    "title", "recall_cues", "provenance",
)


def git_blob_sha(raw: bytes) -> str:
    return hashlib.sha1(
        b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw
    ).hexdigest()


def _encoded_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _check_rows(rows: list[dict], expected_count: int = SOURCE_EVENTS) -> None:
    if len(rows) != expected_count:
        raise ValueError(f"Expected {expected_count} canonical records")
    ids = [row["event_id"] for row in rows]
    if len(set(ids)) != len(rows):
        raise ValueError("Duplicate shared autobiographical event IDs")
    if [row["chronological_order"] for row in rows] != list(range(1, len(rows) + 1)):
        raise ValueError("Changed chronological event ordering")
    if expected_count == SOURCE_EVENTS and len({row["episode_id"] for row in rows}) != SOURCE_EPISODES:
        raise ValueError("Unexpected count of source episodes")
    for item in rows:
        if not all(key in item for key in REQUIRED):
            raise ValueError("Required source field missing")
        if not isinstance(item["memory_text"], str) or not item["memory_text"].strip():
            raise ValueError("Empty memory narrative")
        if not isinstance(item["recall_cues"], list):
            raise ValueError("Expected editorial recall cues")
        if item["provenance"] != "reconstructed":
            raise ValueError("Unexpected provenance or false lived-history claim")


def export_l1(events_path: str | Path, output_dir: str | Path,
              expected_blob: str = SOURCE_BLOB,
              expected_count: int = SOURCE_EVENTS) -> dict:
    """Produce pinned gzip JSONL and canonical manifest, without fitting features.

    The expected_blob / expected_count overrides exist for isolated unit
    fixtures, not for the production command.
    """
    source_raw = Path(events_path).read_bytes()
    observed_blob = git_blob_sha(source_raw)
    if observed_blob != expected_blob:
        raise ValueError("Unpinned source bytes: Git blob SHA mismatch")
    original = [json.loads(line) for line in source_raw.decode("utf-8").splitlines() if line.strip()]
    _check_rows(original, expected_count)
    # Every emitted field is a direct source field. No rewritten biography.
    rows = [{key: row[key] for key in FIELD_NAMES if key in row} for row in original]
    payload = b"".join(_encoded_json(row) + b"\n" for row in rows)
    compressed = gzip.compress(payload, compresslevel=9, mtime=0)
    payload_sha = hashlib.sha256(payload).hexdigest()
    compressed_sha = hashlib.sha256(compressed).hexdigest()
    manifest = {
        "schema_version": SCHEMA,
        "source_repo": SOURCE_REPO,
        "source_commit_pin": SOURCE_COMMIT,
        "source_git_blob": observed_blob,
        "original_source_sha256": hashlib.sha256(source_raw).hexdigest(),
        "records": len(rows),
        "record_ids_ordered": [item["event_id"] for item in rows],
        "episode_ids_ordered": [item["episode_id"] for item in rows],
        "projection_fields": list(FIELD_NAMES),
        "annotation_git_blob": None,
        "normalization_version": "jsonl-field-projection-v1-no-authored-changes",
        "encoder_name": None,
        "encoder_code_hash": None,
        "model_revision": None,
        "fit_event_ids": [],
        "vector_dim": None,
        "dtype": None,
        "feature_normalization": None,
        "random_seed": None,
        "payload_sha256": payload_sha,
        "gzip_sha256": compressed_sha,
        "payload_size_bytes": len(payload),
        "gzip_size_bytes": len(compressed),
        "archive_filename": "pretorius_l1_v1.jsonl.gz",
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / manifest["archive_filename"]).write_bytes(compressed)
    (output / "manifest.json").write_bytes(_encoded_json(manifest) + b"\n")
    loaded = read_l1(output / manifest["archive_filename"], output / "manifest.json",
                     expected_blob=expected_blob, expected_count=expected_count)
    if loaded != rows:
        raise AssertionError("L1 serialization roundtrip failed")
    return manifest


def read_l1(archive_path: str | Path, manifest_path: str | Path,
            expected_blob: str = SOURCE_BLOB,
            expected_count: int = SOURCE_EVENTS) -> list[dict]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != SCHEMA
            or manifest.get("source_repo") != SOURCE_REPO
            or manifest.get("source_commit_pin") != SOURCE_COMMIT
            or manifest.get("source_git_blob") != expected_blob
            or manifest.get("normalization_version") != "jsonl-field-projection-v1-no-authored-changes"
            or manifest.get("encoder_name") is not None
            or manifest.get("fit_event_ids") != []
            or manifest.get("vector_dim") is not None
            or manifest.get("records") != expected_count):
        raise ValueError("Unsupported, fitted, unpinned or mismatched L1 manifest")
    compressed = Path(archive_path).read_bytes()
    if hashlib.sha256(compressed).hexdigest() != manifest.get("gzip_sha256"):
        raise ValueError("Corrupted or substituted L1 compressed archive")
    raw = gzip.decompress(compressed)
    if (hashlib.sha256(raw).hexdigest() != manifest.get("payload_sha256")
            or len(raw) != manifest.get("payload_size_bytes")):
        raise ValueError("Corrupted normalized L1 content")
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    _check_rows(rows, expected_count)
    if ([item["event_id"] for item in rows] != manifest.get("record_ids_ordered")
            or [item["episode_id"] for item in rows] != manifest.get("episode_ids_ordered")):
        raise ValueError("Manifest event order differs from L1 data")
    if any(set(row) - set(FIELD_NAMES) for row in rows):
        raise ValueError("Unexpected L1 field")
    return rows
