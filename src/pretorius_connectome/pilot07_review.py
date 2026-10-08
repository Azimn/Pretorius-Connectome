"""Pilot 07 independent review packet and strict provenance validation.

This does NOT author claims, manufacture external reviewers, or perform review.
Review independence is a declared human workflow, not a property software can
cryptographically verify. Never label empty templates "human reviewed".
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

REVIEW_KINDS = ("true_paraphrase", "explicit_negation", "changed_outcome",
                "unanswerable_nearby")
TRUTH_LABELS = ("entailed", "contradicted", "not_enough_information",
                "ambiguous_or_invalid")
AUTHOR_HEADERS = ("case_id", "source_event_id", "kind", "author_id", "claim")
REVIEW_HEADERS = ("case_id", "reviewer_id", "label", "evidence_quote")
ADJUDICATE_HEADERS = ("case_id", "adjudicator_id", "final_label",
                     "final_evidence_quote", "adjudication_notes")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def choose_events(memories, excluded_event_ids: set[str], max_events: int = 16):
    """Deterministic episode-interleaved selection; no challenge question text."""
    if not (4 <= max_events <= 100):
        raise ValueError("review packet size must be 4..100")
    groups = defaultdict(list)
    for memory in memories:
        if memory.event_id in excluded_event_ids:
            continue
        groups[memory.episode_id].append(memory)
    for group in groups.values():
        group.sort(key=lambda m: (_sha("P07-event-v1:" + m.event_id), m.event_id))
    episodes = sorted(groups, key=lambda ep: _sha("P07-episode-v1:" + ep))
    chosen = []
    while len(chosen) < max_events and any(groups.values()):
        for ep in episodes:
            if groups[ep] and len(chosen) < max_events:
                chosen.append(groups[ep].pop(0))
    if len(chosen) != max_events:
        raise ValueError("insufficient events outside old Pilot04 challenge")
    return chosen


def build_packet(memories, old_cases, max_events: int = 16) -> dict:
    excluded = {c.event_id for c in old_cases}
    selected = choose_events(memories, excluded, max_events)
    manifest = [
        {
            "source_event_id": m.event_id,
            "episode_id": m.episode_id,
            "memory_text": m.memory_text,
            "memory_text_sha256": _sha(m.memory_text),
            "human_reviewed": False,
        }
        for m in selected
    ]
    authoring = [
        {
            "case_id": f"R07-{i+1:03d}-{j+1}",
            "source_event_id": row["source_event_id"],
            "kind": kind,
            "author_id": "", "claim": "",
        }
        for i, row in enumerate(manifest)
        for j, kind in enumerate(REVIEW_KINDS)
    ]
    return {
        "status": "DRAFT PACKET ONLY -- no authored or reviewed questions",
        "selection_method": "stable episode-interleaved SHA-256, excluding all Pilot04 target IDs",
        "source_events": manifest,
        "authoring_template": authoring,
        "selected_events": len(manifest),
        "draft_cases": len(authoring),
        "excluded_previous_ids": len(excluded),
    }


def _write_csv(path: Path, headers: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def write_packet(packet: dict, dest: Path) -> dict:
    dest.mkdir(parents=True, exist_ok=True)
    manifest_path = dest / "source_events.jsonl"
    with manifest_path.open("w", encoding="utf-8") as f:
        for row in packet["source_events"]:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    authors = packet["authoring_template"]
    _write_csv(dest / "authoring.csv", AUTHOR_HEADERS, authors)
    for reviewer in ("a", "b"):
        _write_csv(
            dest / f"reviewer_{reviewer}.csv", REVIEW_HEADERS,
            [
                {"case_id": c["case_id"], "reviewer_id": "", "label": "",
                 "evidence_quote": ""} for c in authors
            ],
        )
    _write_csv(
        dest / "adjudication.csv", ADJUDICATE_HEADERS,
        [
            {"case_id": c["case_id"], "adjudicator_id": "", "final_label": "",
             "final_evidence_quote": "", "adjudication_notes": ""}
            for c in authors
        ],
    )
    return {
        "paths": [str(p) for p in sorted(dest.iterdir()) if p.is_file()],
        "cases": packet["draft_cases"], "status": packet["status"],
    }


def _read_csv(path: Path, expected: tuple[str, ...]) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if tuple(reader.fieldnames or ()) != expected:
            raise ValueError(f"wrong CSV fields in {path.name}")
        result = list(reader)
    if not result:
        raise ValueError(f"empty CSV: {path.name}")
    return result


def _exact_index(rows: list[dict], key: str, expected_ids: set[str]) -> dict:
    ids = [r[key] for r in rows]
    if len(ids) != len(set(ids)) or set(ids) != expected_ids:
        raise ValueError("review case IDs missing, added or duplicated")
    return {r[key]: r for r in rows}


def validate_packet_submission(dest: Path) -> dict:
    """Reject placeholders/ID reuse, confirm exact quoted spans and count labels.

    A passed validator means the *forms* were filled plausibly. Human reviewer
    identity, independence, and truth of content still need external audit.
    """
    sources = [
        json.loads(line) for line in (dest / "source_events.jsonl").read_text(
            encoding="utf-8"
        ).splitlines() if line.strip()
    ]
    source = {}
    for row in sources:
        if (row["source_event_id"] in source or
            _sha(row["memory_text"]) != row["memory_text_sha256"] or
            row.get("human_reviewed") is not False):
            raise ValueError("source manifest tampered or duplicated")
        source[row["source_event_id"]] = row["memory_text"]
    authors = _read_csv(dest / "authoring.csv", AUTHOR_HEADERS)
    expected_ids = set()
    for j, row in enumerate(sources):
        for k in range(len(REVIEW_KINDS)):
            expected_ids.add(f"R07-{j+1:03d}-{k+1}")
    a = _exact_index(authors, "case_id", expected_ids)
    first = _exact_index(_read_csv(dest / "reviewer_a.csv", REVIEW_HEADERS),
                         "case_id", expected_ids)
    second = _exact_index(_read_csv(dest / "reviewer_b.csv", REVIEW_HEADERS),
                          "case_id", expected_ids)
    finals = _exact_index(_read_csv(dest / "adjudication.csv", ADJUDICATE_HEADERS),
                          "case_id", expected_ids)
    agrees = 0
    for j, source_row in enumerate(sources):
        for k, kind in enumerate(REVIEW_KINDS):
            case_id = f"R07-{j+1:03d}-{k+1}"
            record = a[case_id]
            if (record["source_event_id"] != source_row["source_event_id"] or
                record["kind"] != kind or not record["author_id"].strip() or
                len(record["claim"].strip()) < 15):
                raise ValueError(f"unfilled/altered author question {case_id}")
            reviewers = (first[case_id], second[case_id])
            final = finals[case_id]
            ids = [
                record["author_id"].strip(),
                reviewers[0]["reviewer_id"].strip(),
                reviewers[1]["reviewer_id"].strip(),
                final["adjudicator_id"].strip(),
            ]
            if not all(ids) or len(set(ids)) != 4:
                raise ValueError(f"not four distinct declared participants {case_id}")
            for review in reviewers:
                if review["label"] not in TRUTH_LABELS:
                    raise ValueError(f"bad/unreviewed label {case_id}")
                quote = review["evidence_quote"].strip()
                if quote and quote not in source[record["source_event_id"]]:
                    raise ValueError(f"unsupported quoted passage {case_id}")
                if review["label"] in ("entailed", "contradicted") and not quote:
                    raise ValueError(f"missing quoted evidence {case_id}")
            if final["final_label"] not in TRUTH_LABELS:
                raise ValueError(f"unadjudicated label {case_id}")
            quote = final["final_evidence_quote"].strip()
            if quote and quote not in source[record["source_event_id"]]:
                raise ValueError(f"invalid final evidence {case_id}")
            if final["final_label"] in ("entailed", "contradicted") and not quote:
                raise ValueError(f"no adjudicated source quote {case_id}")
            if not final["adjudication_notes"].strip():
                raise ValueError(f"no adjudication notes {case_id}")
            agrees += int(reviewers[0]["label"] == reviewers[1]["label"])
    return {
        "status": "FORMS PASS STRUCTURAL VALIDATION; external independence/quality audit STILL REQUIRED",
        "cases": len(expected_ids),
        "source_events": len(source),
        "raw_reviewer_agreement": round(agrees / len(expected_ids), 6),
        "independent_gold_ready": False,
    }
