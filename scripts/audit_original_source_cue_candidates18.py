#!/usr/bin/env python3
"""Pilot18 post-result ORIGINAL candidate cue structure and provenance audit.

DESCRIPTIVE ONLY; not a model training source, an accuracy benchmark, or
independent semantic equivalence annotation. We noticed source cue anchors
are mostly different short OBJECT/SCENE DETAILS, not sentence paraphrases.
Never use these counts for post-hoc fitting or choosing source test cues.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"memories/current/Pretorius_v12_450_Events_Complete.jsonl"
SIDECAR=ROOT/"memories/annotations/v12_450_sidecars.jsonl"


def pieces(text):
    return [s for s in re.split(r"[^A-Za-z0-9_]+",text.casefold()) if s]


def audit(source=SOURCE,sidecar=SIDECAR):
    source_bytes=Path(source).read_bytes()
    sidecar_bytes=Path(sidecar).read_bytes()
    source_events=[json.loads(x) for x in source_bytes.decode("utf-8").splitlines() if x.strip()]
    candidate_cues=[json.loads(x) for x in sidecar_bytes.decode("utf-8").splitlines() if x.strip()]
    if len(source_events)!=450 or len(candidate_cues)!=450:
        raise ValueError("Full pinned source and cues must have 450 original events")
    source_map={x["event_id"]:x for x in source_events}
    if len(source_map)!=450 or {x["event_id"] for x in candidate_cues}!=set(source_map):
        raise ValueError("Original memory IDs do not match source annotation records")
    n_words=[];comparable=[];phrase_hits=0;exact_overlap=0;status={}
    for row in candidate_cues:
        status[row.get("annotation_status","UNKNOWN")]=status.get(
            row.get("annotation_status","UNKNOWN"),0)+1
        cues=row.get("cue_surface_forms",[])
        if len(cues)<2 or any(not isinstance(c,str) or not c.strip() for c in cues):
            raise ValueError("Malformed source cue candidate")
        n_words.extend(len(c.split()) for c in cues)
        if len(cues)<3:continue
        first,middle,last=cues[0],cues[len(cues)//2],cues[-1]
        if len({x.strip().casefold() for x in (first,middle,last)})<3:
            continue
        original_story=source_map[row["event_id"]]["memory_text"]
        comparable.append(row["event_id"])
        first_tokens=set(pieces(first))
        last_tokens=set(pieces(last))
        if first_tokens & last_tokens:
            exact_overlap+=1
        if last.casefold() in original_story.casefold():
            phrase_hits+=1
    n_words.sort()
    return {
        "kind":"POST-HOC original source cue candidate structure audit, NOT human semantic equivalence",
        "source_memory_SHA256":sha256(source_bytes).hexdigest(),
        "sidecar_candidate_annotation_SHA256":sha256(sidecar_bytes).hexdigest(),
        "fictional_original_source_records":len(source_events),
        "candidate_annotation_records":len(candidate_cues),
        "annotation_status_counts":status,
        "cue_candidate_surface_total":len(n_words),
        "cue_surface_words_median":n_words[len(n_words)//2],
        "cue_surface_at_most_three_words":sum(n<=3 for n in n_words),
        "three_distinct_selected_cues_original_records":len(comparable),
        "no_shared_first_last_literal_word_original_records":
            len(comparable)-exact_overlap,
        "original_last_literal_appears_verbatim_in_same_original_memory_text":
            phrase_hits,
        "not_supported_claims":[
            "No annotation is independently reviewed as a human semantic paraphrase",
            "Zero word overlap does NOT prove semantically unrelated anchors",
            "Cue annotations are memory-associated object/detail candidates, not proven paraphrase pairs",
            "No source event, lexical cue or memory prose altered by this audit",
            "This audit is after the Pilot18 null and cannot be used to tune Pilots18 test conditions",
        ],
    }


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,default=SOURCE)
    p.add_argument("--sidecar",type=Path,default=SIDECAR)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    result=audit(args.source,args.sidecar)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2))
