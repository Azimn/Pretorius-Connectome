#!/usr/bin/env python3
"""Fail-closed original Pretorius v12 source Pilot17 episodic trace archive.

Original 450 reconstructed source memories are not FlyWire neural biology.
This archives five explicit EXTERNAL numerical memory stores. Verifies full
original L1 and BC01 digest and pinned MiniLM revision, all five 317-episode
numeric memory state checkpoints and original 159/31/71/62 evaluation sets.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
ARMS=(
 "semantic_competitive_three_keys",
 "semantic_centroid_same_bytes",
 "semantic_soft_top3_same_bytes",
 "lexical_competitive_three_keys",
 "semantic_competitive_wrong_content",
)
STAGES=(0,16,32,64,128,256,317)
GROUPS=("earliest16","newest16","original159_trained_familiar",
        "original31_untrained_fourth","episode_absent71")
BC01="65ae81401a17ec68adc3900395f226734618be4ffb2bae24745d989bd2a17625"
ONNX="6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452"

def verify(d):
    source=d["source"];p=d["preregistered"]
    if (d.get("study")!="Pilot17 Chamber of Echoes allocated numeric episodic routing"
        or d.get("memory_architecture")!=
            "EXTERNAL numerical episodic key-value retrieval, NOT original FlyWire synapses"
        or "external vector memory" not in d.get("limits","")
        or (source.get("original_canonical_events"),
            source.get("original_episode_count"),
            source.get("training_episodes_source_memories"),
            source.get("original_familiar_evaluation"),
            source.get("original_never_presented_fourth_cues"),
            source.get("original_validation_absent"),
            source.get("original_test_absent"))!=(450,27,317,159,31,62,71)
        or source.get("original_BC01_CONTENT_sha256")!=BC01
        or source.get("frozen_source_pretrained_encoder_ONNX_sha256")!=ONNX
        or source.get("frozen_source_pretrained_encoder_revision")!=
            "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
        or tuple(p.get("stages",[]))!=STAGES
        or p.get("numeric_source_episode_capacity")!=317
        or p.get("three_source_keys_per_episode")!=3
        or p.get("compressed_key_coordinates")!=64
        or p.get("full_stored_content_coordinates")!=256
        or p.get("frozen_input_projection_coordinates")!=16384
        or p.get("slot_state_float32_bytes")!=568064
        or p.get("fixed_projection_float32_bytes")!=65536
        or p.get("total_array_bytes_including_fixed_projection")!=633600
        or p.get("source_CODEBOOK_evaluation_external_only") is not True
        or p.get("valid_absent_only_threshold") is not True
        or p.get("no_reused_159_31_71_test_tuning") is not True):
        raise ValueError("Not exact pinned original source or predeclared external numeric episode study")
    if (len(d["source_cases_all_stages"])!=len(STAGES)
        or set(d["complete_numeric_episode_state_SHA256"])!=set(ARMS)
        or set(d["validation_only_native_key_match_gates"])!=set(ARMS)):
        raise ValueError("Missing full source stage, arms or trained state")
    previous_anchors=None
    for stage,cutoff in zip(d["source_cases_all_stages"],STAGES):
        if stage.get("trained_events")!=cutoff or set(stage.get("arms",{}))!=set(ARMS):
            raise ValueError("Incorrect original source chronology")
        same=None
        for name in ARMS:
            arm=stage["arms"][name]
            cases=arm.get("case_rows_oracle_only",{})
            if (arm.get("allocated_episode_slots")!=cutoff
                or arm.get("numeric_storage_total_bytes")!=633600
                or arm.get("numeric_storage_active_bytes")!=
                    65536+cutoff*(3*64+256)*4
                or set(cases)!=set(GROUPS)
                or set(arm.get("groups",{}))!=set(GROUPS)
                or arm["groups"]["earliest16"]["n"]!=16
                or arm["groups"]["newest16"]["n"]!=min(cutoff,16)
                or arm["groups"]["episode_absent71"]["n"]!=71
                or any(len(cases[g])!=arm["groups"][g]["n"] for g in GROUPS)):
                raise ValueError("Original stage model byte/source case budgets differ")
            anchors=[x["event_id"] for x in cases["earliest16"]]
            if same is None:same=anchors
            elif same!=anchors:
                raise ValueError("Unequal fixed original source anchor cohort")
            if previous_anchors is None:previous_anchors=anchors
            elif previous_anchors!=anchors:
                raise ValueError("Anchor events changed by stage")
            if cutoff==317 and (arm["groups"]["original159_trained_familiar"]["n"]!=159
                or arm["groups"]["original31_untrained_fourth"]["n"]!=31):
                raise ValueError("Missing previous source test-positive cohort")
    for name,g in d["validation_only_native_key_match_gates"].items():
        if (g.get("validation_n")!=62 or g.get("validation_false_accepted",100)>6
            or g.get("known_train_calibration_n")!=158
            or g.get("source_ID_or_CONTENT_oracle_used_to_set_threshold") is not False):
            raise ValueError("Novelty gate uses wrong source labels/split")
        if (not 0<=g["heldout_familiar159_correct_and_accepted"]<=159
            or not 0<=g["heldout_fourth31_correct_and_accepted"]<=31
            or not 0<=g["heldout_absent71_false_accepted"]<=71):
            raise ValueError("Invalid source-known/absent acceptance counts")

def archive(input,weights,run_id):
    if not run_id.isdecimal():raise ValueError("Invalid source GitHub workflow identifier")
    raw=input.read_bytes()
    d=json.loads(raw)
    verify(d)
    dest=ROOT/"artifacts/imprinting/checkpoints"
    dest.mkdir(parents=True,exist_ok=True)
    for name in ARMS:
        b=(weights/(name+".npz")).read_bytes()
        if (sha256(b).hexdigest()!=d["complete_numeric_episode_state_SHA256"][name]["sha256"]
            or not 0<len(b)<2*1024*1024):
            raise ValueError("Original source numeric episode checkpoint digest failure: "+name)
        save=dest/("pilot17-"+name+"-source-run"+run_id+".npz")
        if save.exists() and save.read_bytes()!=b:
            raise ValueError("Different learned memory state for same source run")
        save.write_bytes(b)
    result_dir=ROOT/"results/imprinting"
    case=result_dir/"runs"/("pilot17-episodic-source-run"+run_id+".json")
    case.parent.mkdir(parents=True,exist_ok=True)
    if case.exists() and case.read_bytes()!=raw:
        raise ValueError("Same source run has different JSON bytes")
    case.write_bytes(raw)
    rows=[
        "# Pilot17 Chamber of Echoes: external episodic memory original source results",
        "",
        "[Original source-locked experiment run "+run_id+
            "](https://github.com/Azimn/Pretorius-Connectome/actions/runs/"+run_id+").",
        "[Complete original source original case JSON](runs/"+case.name+").",
        "Original raw source case JSON SHA256 "+sha256(raw).hexdigest()+".",
        "All five original trained external numeric episodic NPZ checkpoints "+
            "preserved under artifacts/imprinting/checkpoints.",
        "",
        "**NOT FLYWIRE NEURAL LEARNING.** One explicit numeric key-value episode "+
            "memory slot stores all three originally authored training cue keys "+
            "and exact BC01 CONTENT codeword, per source event. Retrieval is "+
            "EXTERNAL episodic memory; evaluator-only source IDs are not "+
            "inside memory. Trained original literal cue success is in-sample "+
            "key matching, not untrained generalization.",
        "Same original 450 fictional reconstructed Pretorius autobiographical "+
            "sources/27 episodes, original 317 trained events, original 159 "+
            "trained cue positives, 31 genuinely never-presented FOURTH "+
            "original literal exploratory source cues, original 62 validation "+
            "episode absences and 71 test-episode absences, pinned local MiniLM.",
        "All arms use **633,600 float32 source array bytes** including fixed "+
            "projection, NOT encoder weights/tokenizer/array overhead.",
        "",
        "| Condition | Early16 correct at initial load16 | Early16 correct at final317 | "+
            "Newest16 correct | Familiar trained159 correct | Fourth never-trained31 "+
            "correct | Familiar correctly accepted after native gate | "+
            "Source absent71 false accepted after native gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    first=d["source_cases_all_stages"][1]["arms"]
    last=d["source_cases_all_stages"][-1]["arms"]
    for name in ARMS:
        a=first[name]["groups"]
        b=last[name]["groups"]
        gate=d["validation_only_native_key_match_gates"][name]
        rows.append("| "+name+" | "+
            str(a["earliest16"]["correct_top1"])+"/16 | "+
            str(b["earliest16"]["correct_top1"])+"/16 | "+
            str(b["newest16"]["correct_top1"])+"/16 | "+
            str(b["original159_trained_familiar"]["correct_top1"])+"/159 | "+
            str(b["original31_untrained_fourth"]["correct_top1"])+"/31 | "+
            str(gate["heldout_familiar159_correct_and_accepted"])+"/159 | "+
            str(gate["heldout_absent71_false_accepted"])+"/71 |")
    rows+=["","## Scientific boundary","",
        "Competitive retrieval may seem successful simply because each "+
        "training cue was explicitly stored and the exact original 256D "+
        "target content vector is directly returned. This is NOT a "+
        "synaptic/biological original FlyWire model and does not demonstrate "+
        "a native narrative decoder, intrinsic source provenance or truthful "+
        "autobiographical self-knowledge. Evaluate all claims against the "+
        "same-size centroid and content-mixture controls, semantic versus "+
        "lexical, wrong source content, and separate ordinary retrieval "+
        "baseline. The same repeatedly explored fourth original literal "+
        "source battery is NOT independent human-authored semantic validation. "+
        "Original 317 event-ID output and heldout correctness are offline "+
        "EXTERNAL oracle-only measurements.",""]
    (result_dir/"PILOT17_REAL_SOURCE_EPISODIC_RESULTS.md").write_text(
        "\n".join(rows),encoding="utf-8")
    readme=result_dir/"README.md"
    txt=readme.read_text(encoding="utf-8")
    link="[Pilot17 source-owned EXTERNAL episodic retrieval](PILOT17_REAL_SOURCE_EPISODIC_RESULTS.md)"
    if link not in txt:readme.write_text(txt.rstrip()+"\n\n"+link+"\n",encoding="utf-8")
    handoff=ROOT/"docs/RESEARCH_HANDOFF.md"
    txt=handoff.read_text(encoding="utf-8")
    marker="## Pilot17 Chamber of Echoes source episode run "+run_id
    if marker not in txt:
        handoff.write_text(
            txt.rstrip()+"\n\n"+marker+"\n\n"+
            "[Original source episodic trace study and all five original "+
            "trained numeric checkpoint files](../results/imprinting/"+
            "PILOT17_REAL_SOURCE_EPISODIC_RESULTS.md) compare competitive"+
            " semantic, centroid equal-budget, soft top3, lexical and wrong"+
            " target retrieval on the previous fixed Pretorius corpus. "+
            "This is **EXTERNAL 633,600-byte key-value episode memory**,"+
            " not original neural FlyWire synaptic imprinting. A native"+
            " semantic key-similarity rejection gate used only the 62"+
            " original source validation episode absences; test counts"+
            " include correct AND accepted, and heldout source absences.\n",
            encoding="utf-8")

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",required=True,type=Path)
    p.add_argument("--weights",required=True,type=Path)
    p.add_argument("--run-id",required=True)
    o=p.parse_args()
    archive(o.input,o.weights,o.run_id)
