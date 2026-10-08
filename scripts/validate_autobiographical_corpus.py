#!/usr/bin/env python3
"""Check Pretorius v10 archive invariants without third-party packages."""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "memories"

def load_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def validate():
    previous = load_jsonl(ROOT / "current/Pretorius_v9_385_Events_Complete.jsonl")
    current = load_jsonl(ROOT / "current/Pretorius_v10_400_Events_Complete.jsonl")
    additions = load_jsonl(ROOT / "batches/v10/Pretorius_v10_15_New_Events.jsonl")
    annotations = load_jsonl(ROOT / "annotations/v10_400_sidecars.jsonl")
    registry = json.loads((ROOT / "annotations/v10_cue_registry.json").read_text(encoding="utf-8"))
    assert len(previous) == 385 and len(additions) == 15 and len(current) == len(annotations) == 400
    by_id = {e["event_id"]: e for e in current}
    assert len(by_id) == 400 and set(by_id) == {a["event_id"] for a in annotations}
    assert set(e["event_id"] for e in additions) == {f"E26-{i:03}" for i in range(1, 16)}
    assert current[-1]["event_id"] == "E05-021" and current[-1]["approximate_date"] == "1899-06"
    assert [e["chronological_order"] for e in current] == list(range(1,401))
    assert [e["approximate_date"] for e in current] == sorted(e["approximate_date"] for e in current)
    for old in previous:
        new = by_id[old["event_id"]]
        for field in ("event_id","episode_id","approximate_date","title","memory_text","participants","locations","recall_cues","links_to_prior_events"):
            assert old[field] == new[field], (old["event_id"], field)
    for e in additions:
        assert by_id[e["event_id"]] == e
    entry_by_id={c["cue_id"]:c for c in registry["entries"]}
    assert len(entry_by_id)==len(registry["entries"])
    for e, a in zip(current, annotations):
        assert e["event_id"] == a["event_id"] and a["annotation_status"] == "unreviewed_candidate"
        assert a["cue_surface_forms"] == e["recall_cues"]
        for c in a["cue_ids"]:
            assert e["event_id"] in entry_by_id[c]["event_ids"]
        for p in a.get("sensory_percepts",[]):
            assert p["evidence_excerpt"].casefold() in e["memory_text"].casefold()
        for link in a.get("typed_links",[]):
            assert link["target_event_id"] in by_id
    counts=Counter(c for a in annotations for c in a["cue_ids"])
    print(f"PASS: {len(current)} events; {len(additions)} additions; {len(annotations)} sidecars; "
          f"{len(entry_by_id)} cue IDs; {sum(n>1 for n in counts.values())} recurring cue IDs")
if __name__ == "__main__":
    validate()
