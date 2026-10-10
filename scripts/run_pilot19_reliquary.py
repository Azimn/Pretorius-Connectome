#!/usr/bin/env python3
"""Pilot19 external event association, with exact evidence exposure disclosures.

No FlyWire neuron data or neural language model is involved. The graph directly
indexes source event IDs, candidate unreviewed cue surfaces, participants and
locations of only 317 TRAIN episodes; therefore last-cue success in an all-
sidecar arm is an IN-INDEX literal association, not generalization.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT)]

from pretorius_connectome.reliquary19 import ARMS,Reliquary,normalize,WRONG_OWNER_SHIFT
from pretorius_connectome.imprinting import load_v12,order_by_episode
from pretorius_connectome.pilot02 import episode_split
from scripts.run_direct_flywire_imprint08 import SOURCE,SIDECARS
from scripts.run_direct_flywire_imprint10 import binding_cues,choose_probes
from scripts.diagnose_flywire_imprint11 import unseen_literal_source_cue
from scripts.run_imprinting_pilot import verify_sources

SEED=31
TRAIN_ORDER_SEED=20261008
GROUPS=("train_first159","source_third316","old_fourth31",
        "validation_absent62","test_absent71")


def load_original_source():
    verify_sources(SOURCE,SIDECARS)
    raw_source=SOURCE.read_bytes();raw_annotation=SIDECARS.read_bytes()
    sources=[json.loads(x) for x in raw_source.decode("utf-8").splitlines() if x.strip()]
    annotations=[json.loads(x) for x in raw_annotation.decode("utf-8").splitlines() if x.strip()]
    model_source=load_v12(SOURCE,SIDECARS)
    if len(model_source)!=450 or len(sources)!=450 or len(annotations)!=450:
        raise ValueError("Original source must consist of 450 exact events")
    source_by_id={r["event_id"]:r for r in sources}
    annotation_by_id={a["event_id"]:a for a in annotations}
    if (len(source_by_id)!=450 or len(annotation_by_id)!=450
        or set(source_by_id)!=set(annotation_by_id)
        or set(source_by_id)!={r.event_id for r in model_source}):
        raise ValueError("Original full source and candidate annotation IDs disagree")
    train,valid_absent,test_absent=episode_split(model_source,SEED)
    ordered=order_by_episode(train.copy(),seed=TRAIN_ORDER_SEED)
    familiar=choose_probes(ordered,SEED)
    third=[m for m in ordered if len(
        {normalize(c) for c in binding_cues(m)})==3]
    fourth=[m for m in familiar if unseen_literal_source_cue(m)]
    if (len(ordered),len(valid_absent),len(test_absent),len(familiar),
        len(third),len(fourth))!=(317,62,71,159,316,31):
        raise AssertionError("Historical 450-event/27-episode source split changed")
    if len({m.episode_id for m in ordered}&
           {m.episode_id for m in valid_absent+test_absent}):
        raise AssertionError("Source episode boundary leaked across train/evaluation")
    return {
        "original_source_rows":sources,
        "original_annotation_rows":annotations,
        "source_sha256":sha256(raw_source).hexdigest(),
        "annotations_sha256":sha256(raw_annotation).hexdigest(),
        "train":[source_by_id[m.event_id] for m in ordered],
        "train_annotations":[annotation_by_id[m.event_id] for m in ordered],
        "queries":{
            "train_first159":[(m.event_id,binding_cues(m)[0]) for m in familiar],
            "source_third316":[(m.event_id,binding_cues(m)[-1]) for m in third],
            "old_fourth31":[(m.event_id,unseen_literal_source_cue(m)) for m in fourth],
            "validation_absent62":[(m.event_id,binding_cues(m)[-1])
                                    for m in valid_absent],
            "test_absent71":[(m.event_id,binding_cues(m)[-1])
                              for m in test_absent],
        },
        "train_ids":[m.event_id for m in ordered],
        "episode_ids":{"train":sorted({m.episode_id for m in ordered}),
                       "validation":sorted({m.episode_id for m in valid_absent}),
                       "test":sorted({m.episode_id for m in test_absent})},
    }


def evaluate(model,queries,train_ids,*,threshold,truth_by_id,annotations_by_id):
    original_ids=set(train_ids)
    out=[]
    for expected,query in queries:
        result=model.infer(query)
        is_known=expected in original_ids
        accepted=result["event_id"] is not None and result["confidence"]>threshold
        truth_exposure=(model.indexed_exposure(
            query,expected,
            original_sidecar_cues=annotations_by_id[expected]["cue_surface_forms"],
            original_narrative=truth_by_id[expected]["memory_text"]
        ) if is_known else None)
        evidence=result["evidence"]
        # A contaminated/deranged graph's evidence original owner is explicit.
        if evidence and evidence.get("source_event_id")!=result["event_id"]:
            owner_contamination=True
        else:owner_contamination=False
        out.append({
            "source_expected_event_id":expected,
            "original_source_query_literal":query,
            "is_original_train_event":is_known,
            "returned_event_id":result["event_id"],
            "retrieval_score":result["confidence"],
            "retrieval_ambiguous":result["ambiguous"],
            "accepted_by_validation_only_gate":accepted,
            "original_source_correct_top1":(
                result["event_id"]==expected if is_known else None),
            "original_source_correct_AND_accepted":(
                bool(result["event_id"]==expected and accepted)
                if is_known else None),
            "source_annotation_and_story_exposure":truth_exposure,
            "retrieval_evidence":evidence,
            "wrong_source_ownership_detectable":owner_contamination,
        })
    return out


def summarize(rows):
    known=bool(rows and rows[0]["is_original_train_event"])
    if any(bool(x["is_original_train_event"])!=known for x in rows):
        raise AssertionError("Source group mixes trained and absent events")
    expo=[x["source_annotation_and_story_exposure"] for x in rows if x[
        "source_annotation_and_story_exposure"] is not None]
    return {
        "n":len(rows),
        "source_correct_top1":sum(x["original_source_correct_top1"]
                                   for x in rows) if known else None,
        "source_correct_AND_accepted":sum(x[
            "original_source_correct_AND_accepted"] for x in rows)
            if known else None,
        "source_false_accepted_absent":sum(x[
            "accepted_by_validation_only_gate"] for x in rows)
            if not known else None,
        "source_retrieval_ambiguous":sum(x["retrieval_ambiguous"] for x in rows),
        "truth_cue_present_in_own_retrieval_index":sum(x[
            "cue_in_own_RETRIEVAL_INDEX"] for x in expo),
        "truth_cue_present_in_own_sidecar":sum(x[
            "cue_available_in_own_source_original_sidecar"] for x in expo),
        "truth_cue_verbatim_in_own_narrative":sum(x[
            "cue_verbatim_in_own_original_narrative"] for x in expo),
        "retrieval_wrong_owner_provenance_detected":sum(
            x["wrong_source_ownership_detectable"] for x in rows),
    }


def threshold_from_validation_only(model,valid):
    # External event IDs/correctness are NEVER passed to the rejector.
    scores=sorted(model.infer(query)["confidence"] for _,query in valid)
    if len(scores)!=62:raise AssertionError("Missing original 62 source validation negatives")
    if model.arm in ("train_two_anchors_only","all_anchors_exact_index",
                     "reliquary_event_graph","wrong_owner_graph"):
        # Exact index sees 0 or 1. Predeclared threshold 0 for original
        # existence of ANY literal; counts may exceed six validation FAs.
        # If >=7 negatives match, threshold 1 rejects ALL, is disclosed.
        threshold=0. if sum(v>0 for v in scores)<=6 else 1.
    else:
        threshold=float(scores[-7])
    return {
        "threshold":threshold,
        "validation_62_score_only_false_accept_upper_bound":sum(
            model.infer(q)["event_id"] is not None and
            model.infer(q)["confidence"]>threshold for _,q in valid),
        "all_decisions_source_oracle_event_ID_FREE":True,
        "abstains_when_exact_cue_owned_by_multiple_train_events":True,
    }


def run():
    d=load_original_source()
    train_ids=d["train_ids"]
    indices={arm:Reliquary(d["train"],d["train_annotations"],arm=arm)
             for arm in ARMS}
    # Full original source data are OFFLINE scorer-only truth; not passed
    # into arm-specific cue retrieval after each index has been built.
    truth_by_id={r["event_id"]:r for r in d["train"]}
    annotations_by_id={r["event_id"]:r for r in d["train_annotations"]}
    data={}
    for arm,model in indices.items():
        rejection=threshold_from_validation_only(
            model,d["queries"]["validation_absent62"])
        cohorts={}
        metrics={}
        for kind,queries in d["queries"].items():
            rows=evaluate(model,queries,train_ids,
                          threshold=rejection["threshold"],
                          truth_by_id=truth_by_id,
                          annotations_by_id=annotations_by_id)
            cohorts[kind]=rows
            metrics[kind]=summarize(rows)
        if metrics["validation_absent62"]["source_false_accepted_absent"]>6:
            raise AssertionError("Validation-only gate false-accept bound violated")
        if [metrics[x]["n"] for x in GROUPS]!=[159,316,31,62,71]:
            raise AssertionError("Changed historical evaluation source groups")
        data[arm]={"provenance":model.stored_provenance(),
                   "rejection_validation_only":rejection,
                   "metrics":metrics,"original_case_rows":cohorts}
    if (data["all_anchors_exact_index"]["metrics"]["source_third316"][
        "source_correct_top1"]!=data["reliquary_event_graph"]["metrics"][
        "source_third316"]["source_correct_top1"]):
        raise AssertionError("Same sidecar evidence should give graph/index equal decisions")
    return {
        "study":"Pilot19 The Reliquary evidence-bound event-anchored external cue retrieval",
        "architecture":"EXTERNAL source event ID, candidate cue, narrative and provenance indexes; NOT FlyWire neural memory",
        "original_source":{
            "canonical_events":450,"canonical_episodes":27,
            "original_train_events":317,
            "original_validation_episode_absent":62,
            "original_test_episode_absent":71,
            "original_train_first_known_queries":159,
            "original_train_distinct_third_queries":316,
            "original_exploratory_fourth_queries":31,
            "original_source_memory_SHA256":d["source_sha256"],
            "original_candidate_sidecars_SHA256":d["annotations_sha256"],
            "original_train_event_IDs":train_ids,
            "original_split_episode_ids":d["episode_ids"],
            "original_annotation_status":"unreviewed_candidate",
        },
        "registered_arms":ARMS,
        "registered_groups":GROUPS,
        "wrong_owner_permutation_nonself_offset":WRONG_OWNER_SHIFT,
        "data_leakage_and_nonclaims":[
            "Only 317 source TRAIN events were indexed; original 62 and 71 entire absent episode records were NEVER indexed",
            "Third literal cue in all-anchor sidecars is literally in the index, NOT new or semantic generalization",
            "Graph and exact index see identical sidecar evidence and should tie, NOT proof graph superiority",
            "Original sidecar cues all unreviewed_candidate and cannot establish authenticated perception",
            "Narrative may contain literal third cue; exposure reported per original source case",
            "Source event IDs live INSIDE index intentionally: EXTERNAL retrieval database, not a fly synaptic/LLM model",
            "Original 316 third cues and original 31 fourth cues reused in earlier pilots; not independent human paraphrases",
            "Wrong owner control deliberately poisons event-source links and must be checked for provenance leakage",
            "Original source evidence fields, story and cue surfaces unchanged",
        ],
        "arm_results":data,
    }

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    results=run()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(results,sort_keys=True,indent=2)+"\n",
                           encoding="utf-8")
    for name,item in results["arm_results"].items():
        m=item["metrics"]
        print(name,"third",m["source_third316"]["source_correct_top1"],"/316",
              "third_accessible",m["source_third316"][
                "truth_cue_present_in_own_retrieval_index"],
              "correct_AND_accepted",m["source_third316"][
                "source_correct_AND_accepted"],"absent_false_accept",m[
                    "test_absent71"]["source_false_accepted_absent"],flush=True)

if __name__=="__main__":
    main()
