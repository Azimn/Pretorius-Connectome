#!/usr/bin/env python3
"""Archive original-source Pilot19 without fabricating fly neural checkpoints.

Fail-closed exact original input JSONL bytes, source episode-isolated split,
six complete per-case cohorts, annotation exposure and unchanged source IDs.
Index state is deterministic from SOURCE+SIDECARS and versioned code; store
per-arm index SHA + case evidence, not fake learned numeric weights.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path
from pretorius_connectome.reliquary19 import ARMS

ROOT=Path(__file__).resolve().parents[1]
GROUPS=("train_first159","source_third316","old_fourth31",
        "validation_absent62","test_absent71")
SIZES=(159,316,31,62,71)

def verify(d):
    source=d["original_source"]
    if (d.get("study")!=
        "Pilot19 The Reliquary evidence-bound event-anchored external cue retrieval"
        or not d.get("architecture","").startswith("EXTERNAL")
        or tuple(d.get("registered_arms",()))!=ARMS
        or tuple(d.get("registered_groups",()))!=GROUPS
        or (source.get("canonical_events"),source.get("canonical_episodes"),
            source.get("original_train_events"),
            source.get("original_validation_episode_absent"),
            source.get("original_test_episode_absent"),
            source.get("original_train_first_known_queries"),
            source.get("original_train_distinct_third_queries"),
            source.get("original_exploratory_fourth_queries"))!=
            (450,27,317,62,71,159,316,31)
        or source.get("original_annotation_status")!="unreviewed_candidate"
        or source.get("original_source_memory_SHA256")!=sha256(
            (ROOT/"memories/current/Pretorius_v12_450_Events_Complete.jsonl").read_bytes()).hexdigest()
        or source.get("original_candidate_sidecars_SHA256")!=sha256(
            (ROOT/"memories/annotations/v12_450_sidecars.jsonl").read_bytes()).hexdigest()):
        raise ValueError("Pilot19 original source/annotated split identity failed")
    ids=source.get("original_train_event_IDs",[])
    splits=source.get("original_split_episode_ids",{})
    if (len(ids)!=317 or len(set(ids))!=317
        or not splits or set(splits.get("train",[])) & (
            set(splits.get("validation",[]))|set(splits.get("test",[])))):
        raise ValueError("Train/absent original episode leakage")
    arms=d["arm_results"]
    if set(arms)!=set(ARMS):
        raise ValueError("Missing original source search controls")
    ref=None
    for name in ARMS:
        a=arms[name];m=a["metrics"];cases=a["original_case_rows"]
        if (set(m)!=set(GROUPS) or set(cases)!=set(GROUPS)
            or a["provenance"]["source_event_count"]!=317
            or len(a["provenance"]["index_sha256"])!=64
            or not a["rejection_validation_only"]["all_decisions_source_oracle_event_ID_FREE"]
            or m["validation_absent62"]["source_false_accepted_absent"]>6):
            raise ValueError("Missing full original six-arm provenance/rejection")
        for key,n in zip(GROUPS,SIZES):
            if m[key]["n"]!=n or len(cases[key])!=n:
                raise ValueError("Wrong original source cases")
        third_ids=[r["source_expected_event_id"] for r in cases["source_third316"]]
        if ref is None:ref=third_ids
        elif ref!=third_ids:
            raise ValueError("Source third cohort mismatch across arms")
        for key in ("validation_absent62","test_absent71"):
            if any(r["is_original_train_event"] for r in cases[key]):
                raise ValueError("Unknown episode contaminated with original train")
        if any(r["source_annotation_and_story_exposure"]["sidecar_annotation_status"]!=
               "unreviewed_candidate" for r in cases["source_third316"]):
            raise ValueError("Invalid original unreviewed annotation provenance")
    n0=arms["train_two_anchors_only"]["metrics"]["source_third316"]
    ng=arms["reliquary_event_graph"]["metrics"]["source_third316"]
    ne=arms["all_anchors_exact_index"]["metrics"]["source_third316"]
    if (n0["truth_cue_present_in_own_retrieval_index"]!=0
        or ng["truth_cue_present_in_own_retrieval_index"]!=316
        or ng["source_correct_top1"]!=ne["source_correct_top1"]):
        raise ValueError("Source third-cue exposure or same-evidence graph control invalid")

def archive(source_json,run_id):
    if not run_id.isdecimal():raise ValueError("Bad GitHub source run ID")
    raw=source_json.read_bytes()
    d=json.loads(raw)
    verify(d)
    resultdir=ROOT/"results/imprinting"
    p=resultdir/"runs"/("pilot19-reliquary-original-source-run"+run_id+".json")
    p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists() and p.read_bytes()!=raw:
        raise ValueError("Different original source cases under identical run ID")
    p.write_bytes(raw)
    rows=[
        "# Pilot19: THE RELIQUARY — original-source episode association",
        "",
        "[Original source workflow "+run_id+
        "](https://github.com/Azimn/Pretorius-Connectome/actions/runs/"+run_id+").",
        "[Complete original 6-arm per-case source/evidence data](runs/"+p.name+").",
        "Full JSON SHA256: "+sha256(raw).hexdigest(),
        "",
        "**EXPLICIT EXTERNAL SOURCE INDEX: NOT biological fly synapses, "
        "novel semantic understanding, or authenticated sensory memory.** "
        "The 317 source-event IDs are directly indexed, and original "
        "candidate annotation status remains unreviewed_candidate. "
        "Sidecar-present last-cue success is literal *in-index exposure*, "
        "not a genuinely unseen cue test. No new human paraphrase panel.",
        "",
        "| Original source arm | Source third316 correct | Correct AND accepted | "
        "Third literal available in OWN retrieval index | "
        "Third exact source ambiguous | Absent71 falsely accepted | "
        "Wrong-owner provenance detected in third cases |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ARMS:
        m=d["arm_results"][name]["metrics"]
        t=m["source_third316"]
        n=m["test_absent71"]
        rows.append("| "+name+" | "+str(t["source_correct_top1"])+"/316 | "+
            str(t["source_correct_AND_accepted"])+"/316 | "+
            str(t["truth_cue_present_in_own_retrieval_index"])+"/316 | "+
            str(t["source_retrieval_ambiguous"])+"/316 | "+
            str(n["source_false_accepted_absent"])+"/71 | "+
            str(t["retrieval_wrong_owner_provenance_detected"])+"/316 |")
    rows+=["","## Causal and data-quality interpretation","",
        "The original 316 distinct source third literal detail anchors have "
        "been examined in earlier studies and are not semantic paraphrase "
        "annotations. An all-sidecar graph may return the correct experience "
        "because it directly stores the original third literal → event ID "
        "association. Source ownership is represented explicitly, enabling "
        "retrieval of OTHER original event details and annotation provenance, "
        "but an identical-evidence exact index is a required control. Flat "
        "BM25 and narrative-only BM25 have distinct source availability, "
        "so do not credit graph processing for their lower evidence recall. "
        "Wrong-owner graph is deliberately contaminated and should fail "
        "correct event identity. Exact source strings stored in narrative/"
        "sidecars must be classified as exposed, not semantically inferred. "
        "All 450 annotations are unreviewed candidates, not verified first-"
        "person observation; never promote them into subject-owned memories "
        "without evidence review. Source IDs, fields and source status are "
        "original file provenance and remain entirely distinct from real "
        "FlyWire anatomy or separate Synthetic Ipseity subject access.",""]
    (resultdir/"PILOT19_REAL_RELIQUARY_RESULTS.md").write_text(
        "\n".join(rows),encoding="utf-8")
    idx=resultdir/"README.md"
    content=idx.read_text(encoding="utf-8")
    link="[Pilot19 original provenance-bound episode cue index](PILOT19_REAL_RELIQUARY_RESULTS.md)"
    if link not in content:idx.write_text(content.rstrip()+"\n\n"+link+"\n",encoding="utf-8")
    handoff=ROOT/"docs/RESEARCH_HANDOFF.md"
    original=handoff.read_text(encoding="utf-8")
    title="## Pilot19 Reliquary original event-evidence run "+run_id
    if title not in original:
        handoff.write_text(
            original.rstrip()+"\n\n"+title+"\n\n"+
            "[Complete original full source data and provenance-labeled "
            "six-arm result](../results/imprinting/PILOT19_REAL_RELIQUARY_RESULTS.md): "
            "317 source TRAIN events with 159 familiar/316 distinct third "
            "candidate cues/31 prior fourth/62 absent validation/71 absent "
            "test. Original event-ID and narrative/annotation indexing is "
            "EXTERNAL retrieval, not a learned fly model. All-sidecar "
            "success is indexed third-cue knowledge, with same-evidence exact "
            "index, narrative/flat BM25 and wrong owner controls. "
            "No learned neural numeric checkpoint is claimed; original source "
            "index state is deterministic from pinned JSONL and versioned "
            "implementation; exact six index SHA digests are in case JSON.\n",
            encoding="utf-8"
        )

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",required=True,type=Path)
    p.add_argument("--run-id",required=True)
    opt=p.parse_args()
    archive(opt.input,opt.run_id)
