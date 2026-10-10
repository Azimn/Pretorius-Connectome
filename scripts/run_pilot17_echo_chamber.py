#!/usr/bin/env python3
"""Pilot17 Chamber of Echoes: EXTERNAL numeric episode traces versus routing nulls.

Source-owned original 450 reconstructed fictional Pretorius autobiographical
records, original 317 train/159 familiar/31 never-trained fourth source cues/
62 validation-absent/71 test-absent split. Model's numeric trace lookup is
EXPLICITLY an external episodic memory architecture, not FlyWire synapses.
Returns original BC01 signed target codeword from its internally stored
numerical trace. EXTERNAL fixed 317-candidate oracle gives the event ID.

All source train cues authored in immutable original source. Pretrained local
MiniLM is fixed at original Pilot12 ONNX revision and digest. Every arm stores
the same 317*(3*64+256) float32 numeric slot array plus the same fixed
256x64 projection (633,600 full allocated bytes), NO source record IDs/text.
Three semantic arms differ only in routing; wrong-target and lexical source
controls distinguish binding and pretrained representation from retrieval.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT)]
from pretorius_connectome.echo_chamber17 import EpisodicTraceMemory
from pretorius_connectome.semantic_cue12 import (
    FrozenMiniLMCues,MODEL,REVISION,projection_sha256)
from pretorius_connectome.shared_features_bc01 import (
    load_bc01_cache,SCHEMA,encode_bc_sensory)
from pretorius_connectome.direct_flywire_imprint import select_features
from pretorius_connectome.imprinting import load_v12,order_by_episode
from pretorius_connectome.pilot02 import episode_split
from pretorius_connectome.shared_memory import read_l1
from scripts.run_direct_flywire_imprint08 import SOURCE,SIDECARS,L1_DIR
from scripts.run_direct_flywire_imprint10 import binding_cues,choose_probes
from scripts.diagnose_flywire_imprint11 import unseen_literal_source_cue
from scripts.run_imprinting_pilot import verify_sources

SEED=31
ONNX_SHA="6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452"
BC01_SHA="65ae81401a17ec68adc3900395f226734618be4ffb2bae24745d989bd2a17625"
CUTS=(0,16,32,64,128,256,317)
ARMS=(
 "semantic_competitive_three_keys",
 "semantic_centroid_same_bytes",
 "semantic_soft_top3_same_bytes",
 "lexical_competitive_three_keys",
 "semantic_competitive_wrong_content",
)
GROUPS=("earliest16","newest16","original159_trained_familiar",
        "original31_untrained_fourth","episode_absent71")


def source_data(bc01_dir):
    verify_sources(SOURCE,SIDECARS)
    records=load_v12(SOURCE,SIDECARS)
    original=read_l1(L1_DIR/"pretorius_l1_v1.jsonl.gz",
                     L1_DIR/"manifest.json")
    if (len(records)!=450 or len(original)!=450 or
        any(a.event_id!=b["event_id"] or
            a.episode_id!=b["episode_id"] or
            a.memory_text!=b["memory_text"]
            for a,b in zip(records,original))):
        raise ValueError("Canonical original L0/L1 source mismatch")
    meta,features=load_bc01_cache(bc01_dir)
    if (meta["schema_version"]!=SCHEMA
        or meta["artifact_sha256"]!=BC01_SHA
        or features.shape!=(450,256)
        or meta["record_ids_ordered"]!=[m.event_id for m in records]):
        raise ValueError("Frozen source original BC01 CONTENT cache mismatch")
    trained,validation,absent=episode_split(records,SEED)
    train=order_by_episode(trained.copy(),seed=20261008)
    positives=choose_probes(train,SEED)
    never=[m for m in positives if unseen_literal_source_cue(m)]
    if (len(train),len(positives),len(never),len(validation),
        len(absent))!=(317,159,31,62,71):
        raise AssertionError("Historical source episode split or cue battery drift")
    pos={m.event_id:i for i,m in enumerate(records)}
    ids=[m.event_id for m in train]
    targets=np.stack([
        select_features(features[pos[m.event_id]],top_k=32)
        for m in train])
    return records,meta,features,pos,train,positives,never,validation,absent,ids,targets


def make_models(semantic):
    return {
        "semantic_competitive_three_keys":EpisodicTraceMemory(
            encoder=semantic,mode="competitive"),
        "semantic_centroid_same_bytes":EpisodicTraceMemory(
            encoder=semantic,mode="centroid"),
        "semantic_soft_top3_same_bytes":EpisodicTraceMemory(
            encoder=semantic,mode="soft_top3"),
        "lexical_competitive_three_keys":EpisodicTraceMemory(
            encoder=encode_bc_sensory,mode="competitive"),
        "semantic_competitive_wrong_content":EpisodicTraceMemory(
            encoder=semantic,mode="competitive"),
    }


def evaluate(model,records,ids,target_matrix,learned,*,fourth=False):
    pos={v:i for i,v in enumerate(ids)}
    results=[]
    for record in records:
        cue=(unseen_literal_source_cue(record) if fourth
             else binding_cues(record)[-1])
        if not cue:raise ValueError("Original source fourth literal missing")
        predicted=model.infer(cue)
        scores=target_matrix@predicted
        idx=int(np.argmax(scores))
        active=bool(np.linalg.norm(predicted)>1e-12)
        own=pos.get(record.event_id)
        other=(float(np.max(np.delete(scores,own))) if own is not None else None)
        self_score=(float(scores[own]) if own is not None else None)
        results.append({
            "event_id":record.event_id,
            "source_original_cue_kind":(
                "fourth_genuinely_never_trained_original_literal"
                if fourth else "third_literal_presented_at_training_if_known"),
            "was_this_source_event_added_at_stage":record.event_id in learned,
            "external_candidate_target_known":own is not None,
            "external_predicted_event_id":ids[idx] if active else None,
            "external_source_correct_top1":(
                bool(ids[idx]==record.event_id and active)
                if own is not None else None),
            "external_own_content_cosine":(
                round(self_score,7) if self_score is not None else None),
            "external_own_content_minus_best_competitor":(
                round(self_score-other,7) if self_score is not None else None),
            "internal_best_key_cosine":round(model.native_novelty_score(cue),7),
            "nonzero_content_response":active
        })
    return results


def summarize(rows):
    if not rows:return {"n":0}
    known=rows[0]["external_candidate_target_known"]
    if any(x["external_candidate_target_known"]!=known for x in rows):
        raise AssertionError("Known and absent source groups mixed")
    return {
        "n":len(rows),
        "correct_top1":sum(x["external_source_correct_top1"] for x in rows)
                         if known else None,
        "positive_target_margin":sum(
            x["external_own_content_minus_best_competitor"]>0 for x in rows)
            if known else None,
        "mean_target_margin":(
            round(float(np.mean([x["external_own_content_minus_best_competitor"]
                                 for x in rows])),6) if known else None),
        "mean_internal_best_key_cosine":round(float(np.mean([
            x["internal_best_key_cosine"] for x in rows])),6),
    }


def gate(model,valid_absent,train_calibration,positive,unseen,negative,ids,targets):
    """Model key-match confidence, calibrated exclusively on 62 validation absent."""
    validate=np.asarray([
        model.native_novelty_score(binding_cues(m)[-1])
        for m in valid_absent])
    if len(validate)!=62:raise AssertionError("No fixed original validation split")
    threshold=float(np.sort(validate)[-7])
    positives=evaluate(model,positive,ids,targets,set(ids))
    unexplored=evaluate(model,unseen,ids,targets,set(ids),fourth=True)
    neg=evaluate(model,negative,ids,targets,set(ids))
    calibration=np.asarray([
        model.native_novelty_score(binding_cues(m)[-1])
        for m in train_calibration])
    return {
        "score":"maximum stored cue-key cosine in compact 64D numeric external episode slots",
        "validation_negative_threshold":threshold,
        "validation_n":62,
        "validation_false_accepted":int(np.count_nonzero(validate>threshold)),
        "known_train_calibration_n":len(train_calibration),
        "known_train_calibration_accepted":int(np.count_nonzero(calibration>threshold)),
        "heldout_familiar159_correct_and_accepted":sum(
            x["external_source_correct_top1"] and
            x["internal_best_key_cosine"]>threshold for x in positives),
        "heldout_fourth31_correct_and_accepted":sum(
            x["external_source_correct_top1"] and
            x["internal_best_key_cosine"]>threshold for x in unexplored),
        "heldout_absent71_false_accepted":sum(
            x["internal_best_key_cosine"]>threshold for x in neg),
        "source_ID_or_CONTENT_oracle_used_to_set_threshold":False,
    }


def run(bc01_dir,*,semantic=None,weights_dir=None):
    (records,meta,features,pos,ordered,positives,fourth,
     validation,absent,ids,targets)=source_data(bc01_dir)
    semantic=semantic or FrozenMiniLMCues()
    if semantic.onnx_sha256!=ONNX_SHA:
        raise ValueError("Original pinned pretrained MiniLM ONNX altered")
    all_text=[]
    for item in ordered:all_text.extend(binding_cues(item))
    for item in fourth:all_text.append(unseen_literal_source_cue(item))
    for item in validation+absent:all_text.append(binding_cues(item)[-1])
    semantic.prewarm(all_text)
    models=make_models(semantic)
    if tuple(models)!=ARMS:raise AssertionError("Changed source model-arm roster")
    derangement=np.roll(np.arange(len(ordered)),max(1,len(ordered)//3))
    stages=[]
    for si,cut in enumerate(CUTS):
        for i in range(CUTS[si-1] if si else 0,cut):
            item=ordered[i]
            wrong=ordered[int(derangement[i])]
            if item.event_id==wrong.event_id:
                raise AssertionError("Not wrong CONTENT negative control")
            for name,model in models.items():
                source=wrong if name.endswith("wrong_content") else item
                model.add_episode(binding_cues(item),
                                  features[pos[source.event_id]])
        learned=set(ids[:cut])
        sets={
            "earliest16":ordered[:16],
            "newest16":ordered[max(0,cut-16):cut],
            "original159_trained_familiar":[
                item for item in positives if item.event_id in learned],
            "original31_untrained_fourth":[
                item for item in fourth if item.event_id in learned],
            "episode_absent71":absent,
        }
        stage={"trained_events":cut,"arms":{}}
        for name,model in models.items():
            if model.allocated_episodes!=cut:
                raise AssertionError("Unequal source train event storage")
            cases={}
            groups={}
            for group,items in sets.items():
                rows=evaluate(model,items,ids,targets,learned,
                              fourth=group=="original31_untrained_fourth")
                cases[group]=rows
                groups[group]=summarize(rows)
            stage["arms"][name]={
                "case_rows_oracle_only":cases,
                "groups":groups,
                "allocated_episode_slots":model.allocated_episodes,
                "numeric_storage_total_bytes":model.numeric_storage_bytes,
                "numeric_storage_active_bytes":model.active_storage_bytes,
                "fixed_projection_included_in_storage_budget":True
            }
        stages.append(stage)
    cal_known=[m for m in ordered if m.event_id not in
               {v.event_id for v in positives}]
    if len(cal_known)!=158:raise AssertionError("No original 158 train calibration")
    gates={name:gate(model,validation,cal_known,positives,fourth,
                     absent,ids,targets) for name,model in models.items()}
    checkpoints={}
    if weights_dir:
        for name,model in models.items():
            dest=Path(weights_dir)/(name+".npz")
            digest=model.save(dest,encoder_sha=(
                ONNX_SHA if name.startswith("semantic") else BC01_SHA))
            copy=EpisodicTraceMemory.load(
                dest,encoder=model.encoder,encoder_sha=(
                    ONNX_SHA if name.startswith("semantic") else BC01_SHA))
            if (not np.array_equal(copy.keys,model.keys)
                or not np.array_equal(copy.contents,model.contents)):
                raise AssertionError("Numeric episode contents/keys not exactly restored")
            checkpoints[name]={"sha256":digest,"path":str(dest)}
    return {
        "study":"Pilot17 Chamber of Echoes allocated numeric episodic routing",
        "memory_architecture":"EXTERNAL numerical episodic key-value retrieval, NOT original FlyWire synapses",
        "limits":(
            "Stores 317 directly allocated event-specific three-key/one-value "
            "numeric episode traces. It DOES perform retrieval and stores the "
            "entire original BC01 CONTENT target codeword for every trained "
            "event. This is a genuine external vector memory, NOT a neural-"
            "only associative recurrent state or full original FlyWire graph. "
            "Original 159 familiar source probes INCLUDE presented training "
            "cues, and a high score for them can be mere exact key matching. "
            "Only 31 previously explored never-trained fourth ORIGINAL literal "
            "cues test limited generalization; no independently human-"
            "authored sealed paraphrases. The 317 source-ID labels and candidate "
            "CONTENT target matching remain EXTERNAL offline oracle-only; "
            "numeric slot indices are NOT source ID labels. Frozen pretrained "
            "MiniLM language semantics are external to Pretorius source. "
            "No native source story text or proven truth/absence gate."
        ),
        "source":{
            "original_canonical_events":len(records),
            "original_episode_count":len({m.episode_id for m in records}),
            "training_episodes_source_memories":len(ordered),
            "original_familiar_evaluation":len(positives),
            "original_never_presented_fourth_cues":len(fourth),
            "original_validation_absent":len(validation),
            "original_test_absent":len(absent),
            "original_BC01_CONTENT_sha256":meta["artifact_sha256"],
            "frozen_source_pretrained_encoder_ONNX_sha256":semantic.onnx_sha256,
            "frozen_source_pretrained_encoder_revision":REVISION,
            "frozen_semantic_projection_sha256":projection_sha256(),
        },
        "preregistered":{
            "stages":CUTS,
            "numeric_source_episode_capacity":317,
            "three_source_keys_per_episode":3,
            "compressed_key_coordinates":64,
            "full_stored_content_coordinates":256,
            "frozen_input_projection_coordinates":256*64,
            "slot_state_float32_bytes":317*(3*64+256)*4,
            "fixed_projection_float32_bytes":256*64*4,
            "total_array_bytes_including_fixed_projection":633600,
            "source_ONNX_and_tokenizer_weights_NOT_included_in_memory_array_budget":True,
            "previous_unmasked_W_float32_P_float64_bytes":65536*4+65536*8,
            "source_CODEBOOK_evaluation_external_only":True,
            "valid_absent_only_threshold":True,
            "no_reused_159_31_71_test_tuning":True,
        },
        "source_cases_all_stages":stages,
        "validation_only_native_key_match_gates":gates,
        "complete_numeric_episode_state_SHA256":checkpoints,
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bc01-dir",type=Path,required=True)
    p.add_argument("--weights-dir",type=Path)
    p.add_argument("--output",type=Path,required=True)
    opt=p.parse_args()
    evidence=run(opt.bc01_dir,weights_dir=opt.weights_dir)
    opt.output.parent.mkdir(parents=True,exist_ok=True)
    opt.output.write_text(json.dumps(evidence,indent=2,sort_keys=True)+"\n",
                          encoding="utf-8")
    for step in evidence["source_cases_all_stages"]:
        print("STAGE",step["trained_events"]," ".join(
            name+":"+str(a["groups"]["earliest16"]["correct_top1"])+"/16"
            for name,a in step["arms"].items()),flush=True)

if __name__=="__main__":
    main()
