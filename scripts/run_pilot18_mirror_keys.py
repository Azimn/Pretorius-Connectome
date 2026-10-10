#!/usr/bin/env python3
"""Pilot18 Mirror Keys: train-only semantic metric on original source cue pairs.

Unlike Pilot17, store only FIRST and MIDDLE original source cues, with the
third slot allocated but zeroed across ALL six same-byte routing controls.
For the 448 original events with >=3 surface cues, the last original
source literal was never used in either metric fitting or stored keys
(exclude duplicates). Holdout is the source-authored third view, NOT a
new human-independent paraphrase panel. Old 31 fourth literals remain
exploratory only; no tuning on 159/31/71 heldout source outcomes.

All metric fits use ALL 317 train records' first two cue vectors before
episodic memory allocation, so stages earlier than full 317 would be
transductive with future semantic views. This experiment ONLY reports
FINAL 317-memory load for an unambiguous causal interpretation. The
source contents of 317 train records never enter the metric fitting.
This is EXTERNAL 633600-byte episodic numeric key-value retrieval.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT)]

from pretorius_connectome.mirror_keys18 import (
    TwoViewEpisodicMemory,fit_train_only_projection,
    PROJECTION_REGULARIZATION,PAIR_PENALTY,WRONG_PAIR_SEED
)
from pretorius_connectome.semantic_cue12 import FrozenMiniLMCues,REVISION,projection_sha256
from pretorius_connectome.shared_features_bc01 import encode_bc_sensory
from scripts.run_pilot17_echo_chamber import (
    source_data,ONNX_SHA,BC01_SHA,binding_cues,unseen_literal_source_cue
)

ARMS=(
 "semantic_random_two_views",
 "semantic_pair_discriminant_two_views",
 "semantic_mispaired_discriminant_two_views",
 "semantic_pca_two_views",
 "lexical_random_two_views",
 "semantic_pair_discriminant_wrong_content",
)
GROUPS=(
 "known_first_training_literal159",
 "heldout_third_source_view_all",
 "previously_explored_fourth_source31",
 "absent_episode71",
 "early16_heldout_third",
 "newest16_heldout_third",
)


def withheld_third(memory):
    """Only truly distinct source-authored third literal cue, never trained.

    Source original 2-cue records repeat cue2 in old 3-cue training system
    and are EXCLUDED from prospective two-cue leave-view-out test.
    """
    a,b,c=binding_cues(memory)
    if len({x.strip().casefold() for x in (a,b,c)})!=3:
        return None
    return c


def source_probe_rows(model,records,ids,targets,*,kind):
    pos={event:i for i,event in enumerate(ids)}
    rows=[]
    for m in records:
        if kind=="heldout_third":
            cue=withheld_third(m)
            if cue is None:
                raise ValueError("Invalid third heldout source cue, may leak train literal")
        elif kind=="fourth":
            cue=unseen_literal_source_cue(m)
            if cue is None:raise ValueError("Historical genuine fourth cue not present")
        else:
            cue=binding_cues(m)[0] if kind=="trained_first" else binding_cues(m)[-1]
        out=model.infer(cue)
        scores=targets@out
        idx=int(np.argmax(scores))
        is_active=bool(np.linalg.norm(out)>1e-12)
        own=pos.get(m.event_id)
        rows.append({
            "event_id":m.event_id,
            "kind":kind,
            "oracle_event_known":own is not None,
            "external_predicted_event_id":ids[idx] if is_active else None,
            "external_correct_top1":(
                bool(is_active and ids[idx]==m.event_id)
                if own is not None else None),
            "true_source_content_margin":(
                round(float(scores[own]-
                    np.max(np.delete(scores,own))),7) if own is not None else None),
            "model_native_max_stored_key_score":
                round(float(model.native_novelty_score(cue)),7),
        })
    return rows


def report(rows):
    result={"n":len(rows)}
    if rows and rows[0]["oracle_event_known"]:
        result["correct_top1"]=sum(x["external_correct_top1"] for x in rows)
        result["positive_true_target_margin"]=sum(
            x["true_source_content_margin"]>0 for x in rows)
        result["mean_true_target_margin"]=round(float(np.mean(
            [x["true_source_content_margin"] for x in rows])),6)
    return result


def validation_gate(model,validation_absent,calibration_train,heldout,old_fourth,
                    familiar,absent,ids,targets):
    # Third/fourth/159/71 source test cases do NOT enter threshold selection.
    scores=np.asarray([model.native_novelty_score(binding_cues(m)[-1])
                       for m in validation_absent])
    if len(scores)!=62:raise AssertionError("Incorrect original validation subjects")
    threshold=float(np.sort(scores)[-7])
    cal=np.asarray([model.native_novelty_score(binding_cues(m)[0])
                    for m in calibration_train])
    def accepted(records,kind):
        return source_probe_rows(model,records,ids,targets,kind=kind)
    # Could reuse already-measured rows; these queries never alter threshold.
    a=accepted(heldout,"heldout_third")
    b=accepted(old_fourth,"fourth")
    c=accepted(familiar,"trained_first")
    d=accepted(absent,"absent")
    def joint(rows):
        return sum(x["external_correct_top1"] and
                   x["model_native_max_stored_key_score"]>threshold
                   for x in rows)
    return {
        "validation_negative_n":62,
        "threshold":threshold,
        "validation_false_accepted":int(np.sum(scores>threshold)),
        "training_known_non_evaluation_n":len(calibration_train),
        "training_known_accepted":int(np.sum(cal>threshold)),
        "heldout_third_correct_and_accepted":joint(a),
        "heldout_third_n":len(a),
        "old_fourth_correct_and_accepted":joint(b),
        "old_fourth_n":len(b),
        "familiar_first_correct_and_accepted":joint(c),
        "familiar_first_n":len(c),
        "absent_test_false_accepted":sum(
            x["model_native_max_stored_key_score"]>threshold for x in d),
        "absent_test_n":len(d),
        "oracle_candidate_event_ids_used_to_set_threshold":False,
    }


def run(bc01_dir,*,semantic=None,weights_dir=None):
    (memories,meta,content,pos,ordered,positives,fourth,
     validation,absent,ids,targets)=source_data(bc01_dir)
    semantic=semantic or FrozenMiniLMCues()
    if semantic.onnx_sha256!=ONNX_SHA:
        raise ValueError("Pinned public MiniLM source encoder changed")
    two_train=[binding_cues(m)[:2] for m in ordered]
    heldout=[m for m in ordered if withheld_third(m) is not None]
    if len(heldout)<290:
        raise AssertionError("Original source third holdout eligible unexpectedly small")
    # Strictly prepare metric data from FIRST and MIDDLE source cues only.
    # No last, fourth, absent or any of the frozen source CONTENT vectors.
    flatten=[text for pair in two_train for text in pair]
    semantic.prewarm(flatten)
    X=np.asarray([[semantic(c) for c in pair] for pair in two_train],
                 dtype=np.float32)
    if X.shape!=(317,2,256):
        raise AssertionError("Wrong original 317x2x256 train-only cue matrix")
    projected={
        "semantic_random_two_views":fit_train_only_projection(
            X,algorithm="random"),
        "semantic_pair_discriminant_two_views":fit_train_only_projection(
            X,algorithm="pair_discriminant"),
        "semantic_mispaired_discriminant_two_views":fit_train_only_projection(
            X,algorithm="mispaired_discriminant"),
        "semantic_pca_two_views":fit_train_only_projection(
            X,algorithm="pca"),
        "lexical_random_two_views":fit_train_only_projection(
            X,algorithm="random"),
        "semantic_pair_discriminant_wrong_content":fit_train_only_projection(
            X,algorithm="pair_discriminant"),
    }
    if tuple(projected)!=ARMS:
        raise AssertionError("The registered controls or fit inputs changed")
    if not np.array_equal(
        projected["semantic_pair_discriminant_two_views"],
        projected["semantic_pair_discriminant_wrong_content"]):
        raise AssertionError("Source-shuffled CONTENT must not alter cue metric")
    models={
        name:TwoViewEpisodicMemory(
            encoder=encode_bc_sensory if name.startswith("lexical") else semantic,
            projection=p
        ) for name,p in projected.items()
    }
    wrong=np.roll(np.arange(len(ordered)),max(1,len(ordered)//3))
    for i,m in enumerate(ordered):
        for name,model in models.items():
            target=ordered[int(wrong[i])] if name.endswith("wrong_content") else m
            if name.endswith("wrong_content") and target.event_id==m.event_id:
                raise AssertionError("Negative source target permutation invalid")
            model.add_episode(two_train[i],content[pos[target.event_id]])
    # Public model encoder caching after fitting is NOT metric training.
    # Test third views were never passed to fit_train_only_projection.
    for m in heldout:semantic.prewarm([withheld_third(m)])
    for m in fourth:semantic.prewarm([unseen_literal_source_cue(m)])
    for m in validation+absent:semantic.prewarm([binding_cues(m)[-1]])
    cal=[m for m in ordered if m.event_id not in
         {v.event_id for v in positives}]
    if len(cal)!=158:raise AssertionError("Original non-evaluation training group changed")

    cohorts={
        "known_first_training_literal159":(positives,"trained_first"),
        "heldout_third_source_view_all":(heldout,"heldout_third"),
        "previously_explored_fourth_source31":(fourth,"fourth"),
        "absent_episode71":(absent,"absent"),
        "early16_heldout_third":(
            [m for m in ordered[:16] if withheld_third(m) is not None],
            "heldout_third"),
        "newest16_heldout_third":(
            [m for m in ordered[-16:] if withheld_third(m) is not None],
            "heldout_third")
    }
    casebook={}
    gates={}
    checks={}
    for name,model in models.items():
        casebook[name]={
            group:source_probe_rows(model,rec,ids,targets,kind=kind)
            for group,(rec,kind) in cohorts.items()
        }
        gates[name]=validation_gate(model,validation,cal,heldout,fourth,
                                   positives,absent,ids,targets)
        checks[name]={
            "stored_source_episodes":model.allocated_episodes,
            "third_source_cue_was_NOT_stored":bool(
                np.all(model.keys[:,2,:]==0)),
            "source_key_count_per_record":2,
            "numeric_total_array_bytes":model.numeric_storage_bytes,
            "numeric_active_array_bytes":model.active_storage_bytes,
            "fitted_projection_sha256":__import__("hashlib").sha256(
                np.ascontiguousarray(model.projection).tobytes()).hexdigest(),
            "groups":{group:report(rows) for group,rows in casebook[name].items()}
        }
        if (model.allocated_episodes!=317 or
            model.numeric_storage_bytes!=633600 or
            not checks[name]["third_source_cue_was_NOT_stored"]):
            raise AssertionError("Equal-budget/two-view original source contract broken")
    checkpoint={}
    if weights_dir:
        for name,model in models.items():
            path=Path(weights_dir)/(name+".npz")
            sha=model.save(path,encoder_sha=(
                BC01_SHA if name.startswith("lexical") else ONNX_SHA),
                source_sha=BC01_SHA,algorithm=name)
            reread=TwoViewEpisodicMemory.load(
                path,encoder=model.encoder,
                encoder_sha=BC01_SHA if name.startswith("lexical") else ONNX_SHA,
                source_sha=BC01_SHA)
            if not (np.array_equal(reread.keys,model.keys) and
                    np.array_equal(reread.contents,model.contents) and
                    np.array_equal(reread.projection,model.projection)):
                raise AssertionError("Original learned train-only projection/source memory checkpoint drift")
            checkpoint[name]={"sha256":sha,"path":str(path)}
    return {
        "study":"Pilot18 Mirror Keys two-view-only train-fit discriminant external episodic memory",
        "category":"EXTERNAL numeric allocated episodic memory, NOT FlyWire neuron biology",
        "limitations":(
            "Trains metric on FIRST+MIDDLE source cue views of ALL 317 "
            "TRAIN original fictional autobiographical events only, then "
            "stores just those TWO original cue keys plus original 256D "
            "CONTENT vectors; third key slot is allocated but always ZERO. "
            "Third source-authored literal is heldout for eligible events "
            "and NEVER enters metric fit or stored episode keys. These source "
            "third literals were previously trained/tested in prior pilots, "
            "thus a newly withheld view in this experiment, NOT a new "
            "independently human authored sealed paraphrase panel. Entire "
            "metric fit uses all 317 train events, so no early-load "
            "performance claims. Original 31 fourth source cues have been "
            "examined many times and are EXPLORATORY only. This explicitly "
            "stores source CONTENT values, not original biological FlyWire "
            "synaptic cognition. The offline 317 target codewords and source "
            "event IDs are EXTERNAL oracle-only ranking, never inside model. "
            "At most 6 validation absent false accepts, but test unknown "
            "acceptance and CORRECT+ACCEPTED novel recall must both be read."
        ),
        "original_source":{
            "canonical_records":len(memories),
            "episodes":len({m.episode_id for m in memories}),
            "original_train":len(ordered),
            "two_original_source_train_cues_per_record":True,
            "true_third_distinct_source_heldout":len(heldout),
            "historical_train_literal_positive":len(positives),
            "historical_exploratory_fourth":len(fourth),
            "episode_validation_absent":len(validation),
            "episode_test_absent":len(absent),
            "BC01_source_original_SHA":meta["artifact_sha256"],
            "frozen_MiniLM_ONNX_SHA":semantic.onnx_sha256,
            "frozen_MiniLM_revision":REVISION,
            "frozen_MiniLM_projection_SHA":projection_sha256()
        },
        "preregistered":{
            "frozen_projection_fit_pairs":317,
            "frozen_projection_pair_penalty":PAIR_PENALTY,
            "frozen_projection_regularization":PROJECTION_REGULARIZATION,
            "wrong_positive_pair_seed":WRONG_PAIR_SEED,
            "stored_cue_views_per_record":2,
            "never_fit_or_store_last_third_literal":True,
            "numeric_array_bytes_identical_all_arms":633600,
            "fitted_or_random_256x64_projection_bytes":65536,
            "original_317_contents_stored_directly_in_external_numeric_array":True,
            "source_second_and_first_only_train_metric":True,
            "threshold_from_validation_absent_62_ONLY":True,
            "no_tuning_on_reused_fourth_31_or_absent_71":True,
        },
        "arm_numeric_resources_and_metrics":checks,
        "source_cases_317_episode_memory_FINAL_only":casebook,
        "validation_only_native_novelty_gates":gates,
        "source_model_checkpoints_sha256":checkpoint,
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bc01-dir",required=True,type=Path)
    p.add_argument("--weights-dir",type=Path)
    p.add_argument("--output",required=True,type=Path)
    args=p.parse_args()
    result=run(args.bc01_dir,weights_dir=args.weights_dir)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",
                           encoding="utf-8")
    for name,details in result["arm_numeric_resources_and_metrics"].items():
        a=details["groups"]["heldout_third_source_view_all"]
        print(name,"heldout_third_correct",a["correct_top1"],"/",a["n"],
              "correct_accepted",result["validation_only_native_novelty_gates"][
                  name]["heldout_third_correct_and_accepted"],flush=True)


if __name__=="__main__":
    main()
