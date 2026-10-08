#!/usr/bin/env python3
"""Validate v12 archive against immutable v11 narrative baseline."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]/"memories"
def records(p):
    return [json.loads(s) for s in (ROOT/p).read_text(encoding="utf-8").splitlines() if s.strip()]
def validate_v12():
    old=records("current/Pretorius_v11_430_Events_Complete.jsonl")
    current=records("current/Pretorius_v12_450_Events_Complete.jsonl")
    new=records("batches/v12/Pretorius_v12_E27_20_New_Events.jsonl")
    ann=records("annotations/v12_450_sidecars.jsonl")
    registry=json.loads((ROOT/"annotations/v12_cue_registry.json").read_text(encoding="utf-8"))
    assert (len(old),len(current),len(new),len(ann))==(430,450,20,450)
    by_id={e["event_id"]:e for e in current}
    by_cue={x["cue_id"]:x for x in registry["entries"]}
    assert len(by_id)==450 and len(by_cue)==len(registry["entries"])
    assert set(e["event_id"] for e in new)=={"E27-%03d"%i for i in range(1,21)}
    assert current[-1]["event_id"]=="E05-021" and current[-1]["approximate_date"]=="1899-06"
    assert [e["chronological_order"] for e in current]==list(range(1,451))
    assert [e["approximate_date"] for e in current]==sorted(e["approximate_date"] for e in current)
    for e in old:
        n=by_id[e["event_id"]]
        for field in ("event_id","episode_id","approximate_date","title","memory_text","participants","locations","recall_cues","links_to_prior_events"):
            assert e[field]==n[field],(e["event_id"],field)
    for e in new:
        assert {k:v for k,v in by_id[e["event_id"]].items() if k!="chronological_order"} == {k:v for k,v in e.items() if k!="chronological_order"}
        for target in e["links_to_prior_events"]:
            assert target in by_id and by_id[target]["approximate_date"]<=e["approximate_date"]
    for ev,sc in zip(current,ann):
        assert ev["event_id"]==sc["event_id"] and sc["annotation_status"]=="unreviewed_candidate"
        assert ev["recall_cues"]==sc["cue_surface_forms"]
        for cid in sc["cue_ids"]:
            assert ev["event_id"] in by_cue[cid]["event_ids"]
        for p in sc.get("sensory_percepts",[]):
            if p.get("evidence_excerpt"):
                assert p["evidence_excerpt"].casefold() in ev["memory_text"].casefold()
        for link in sc.get("typed_links",[]):
            assert link["target_event_id"] in by_id
    print("PASS v12: 450 events, 20 E27 additions, 430 unchanged baselines, 450 sidecars, %d cue IDs"%len(by_cue))
if __name__=="__main__":
    validate_v12()
