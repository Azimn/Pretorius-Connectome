#!/usr/bin/env python3
"""Archive only successful main-source Pilot18 six-arm EXTERNAL episodic evidence.

Require original fictional Pretorius L1/BC01, exact frozen MiniLM ONNX,
two literal training cues ONLY, third key ZERO, six equal 633600-byte
numeric memory conditions, validation-only native confidence and
checksum-verified actual six saved projections+episode key/value arrays.
This is never labeled fly-v783 biological learning.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BC01="65ae81401a17ec68adc3900395f226734618be4ffb2bae24745d989bd2a17625"
ONNX="6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452"
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


def verify(d):
    s=d["original_source"];p=d["preregistered"]
    if (d.get("study")!=
        "Pilot18 Mirror Keys two-view-only train-fit discriminant external episodic memory"
        or d.get("category")!=
            "EXTERNAL numeric allocated episodic memory, NOT FlyWire neuron biology"
        or (s.get("canonical_records"),s.get("episodes"),s.get("original_train"),
            s.get("historical_train_literal_positive"),
            s.get("historical_exploratory_fourth"),
            s.get("episode_validation_absent"),
            s.get("episode_test_absent"))!=(450,27,317,159,31,62,71)
        or not 290<=s.get("true_third_distinct_source_heldout",0)<=315
        or s.get("BC01_source_original_SHA")!=BC01
        or s.get("frozen_MiniLM_ONNX_SHA")!=ONNX
        or s.get("frozen_MiniLM_revision")!=
            "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
        or p.get("frozen_projection_fit_pairs")!=317
        or p.get("frozen_projection_pair_penalty")!=2.
        or p.get("frozen_projection_regularization")!=.002
        or p.get("wrong_positive_pair_seed")!=18031
        or p.get("stored_cue_views_per_record")!=2
        or p.get("never_fit_or_store_last_third_literal") is not True
        or p.get("numeric_array_bytes_identical_all_arms")!=633600
        or p.get("fitted_or_random_256x64_projection_bytes")!=65536
        or p.get("threshold_from_validation_absent_62_ONLY") is not True
        or p.get("no_tuning_on_reused_fourth_31_or_absent_71") is not True):
        raise ValueError("Pilot18 incomplete source, metric or no-third-leak contract")
    checks=d["arm_numeric_resources_and_metrics"]
    cases=d["source_cases_317_episode_memory_FINAL_only"]
    gates=d["validation_only_native_novelty_gates"]
    weights=d["source_model_checkpoints_sha256"]
    if (set(checks)!=set(ARMS) or set(cases)!=set(ARMS)
        or set(gates)!=set(ARMS) or set(weights)!=set(ARMS)):
        raise ValueError("Missing original six predeclared same-byte/source arms")
    reference_ids=None
    for name in ARMS:
        c=checks[name];g=gates[name]
        if (c.get("stored_source_episodes")!=317
            or c.get("third_source_cue_was_NOT_stored") is not True
            or c.get("source_key_count_per_record")!=2
            or c.get("numeric_total_array_bytes")!=633600
            or set(c.get("groups",{}))!=set(GROUPS)
            or set(cases[name])!=set(GROUPS)
            or any(c["groups"][k]["n"]!=len(cases[name][k])
                   for k in GROUPS)
            or c["groups"]["heldout_third_source_view_all"]["n"]!=
                s["true_third_distinct_source_heldout"]
            or c["groups"]["known_first_training_literal159"]["n"]!=159
            or c["groups"]["previously_explored_fourth_source31"]["n"]!=31
            or c["groups"]["absent_episode71"]["n"]!=71
            or g.get("validation_negative_n")!=62
            or g.get("validation_false_accepted",100)>6
            or g.get("training_known_non_evaluation_n")!=158
            or g.get("heldout_third_n")!=s["true_third_distinct_source_heldout"]
            or g.get("familiar_first_n")!=159
            or g.get("old_fourth_n")!=31
            or g.get("absent_test_n")!=71
            or g.get("oracle_candidate_event_ids_used_to_set_threshold") is not False):
            raise ValueError("Original holdout metrics, model native gate or equal storage invalid")
        cohort=[x["event_id"] for x in cases[name]["heldout_third_source_view_all"]]
        if reference_ids is None:reference_ids=cohort
        elif reference_ids!=cohort:
            raise ValueError("Model source heldout third event identities differ")
        for row in cases[name]["heldout_third_source_view_all"]:
            if row["kind"]!="heldout_third":
                raise ValueError("Claimed genuine third source cue actually trained")

def archive(raw_path,weights_dir,run_id):
    if not run_id.isdecimal():raise ValueError("Invalid GitHub source workflow ID")
    b=raw_path.read_bytes()
    d=json.loads(b);verify(d)
    outputs=ROOT/"artifacts/imprinting/checkpoints"
    outputs.mkdir(parents=True,exist_ok=True)
    for name in ARMS:
        p=weights_dir/(name+".npz")
        content=p.read_bytes()
        if (sha256(content).hexdigest()!=
            d["source_model_checkpoints_sha256"][name]["sha256"]
            or not 0<len(content)<3*1024*1024):
            raise ValueError("Original source-trained projection/episode SHA mismatch: "+name)
        with np.load(p,allow_pickle=False) as z:
            if (z["format_version"].tolist()!=[1]
                or z["keys"].shape!=(317,3,64)
                or z["contents"].shape!=(317,256)
                or z["projection"].shape!=(256,64)
                or not np.all(z["keys"][:,2,:]==0)
                or z["settings"].tolist()!=[317,317]
                or str(z["algorithm"])!=name
                or str(z["source_sha"])!=BC01
                or str(z["encoder_sha"])!=(BC01 if name.startswith("lexical") else ONNX)
                or not np.isfinite(z["projection"]).all()):
                raise ValueError("Saved original source episode contains heldout third cue or bad provenance")
        save=outputs/("pilot18-"+name+"-source-run"+run_id+".npz")
        if save.exists() and save.read_bytes()!=content:
            raise ValueError("Changed source learned trace under original run ID")
        save.write_bytes(content)
    folder=ROOT/"results/imprinting"
    run=folder/"runs"/("pilot18-two-view-source-run"+run_id+".json")
    run.parent.mkdir(parents=True,exist_ok=True)
    if run.exists() and run.read_bytes()!=b:
        raise ValueError("Existing raw source case bytes differ")
    run.write_bytes(b)
    lines=[
        "# Pilot18 Mirror Keys — original two-view semantic address metric",
        "",
        "[Source original successful 450-Pretorius-memory experiment "+run_id+
        "](https://github.com/Azimn/Pretorius-Connectome/actions/runs/"+run_id+").",
        "[Complete original case-level six-arm JSON](runs/"+run.name+").",
        "Original JSON SHA256: "+sha256(b).hexdigest(),
        "Exactly six original 317-episode learned external numeric key/value and "
        "train-fit projection NPZ files are permanently archived under "
        "artifacts/imprinting/checkpoints.",
        "",
        "**NOT full FlyWire biological neural memory.** All conditions are "
        "EXTERNAL numeric episodic stores. Original first+middle source-authored "
        "literal cue vectors train the 64D metric and become the only stored keys; "
        "each ORIGINAL distinct third literal is a withheld source view, never "
        "fit nor stored. Re-used original 31 fourth cues are exploratory. "
        "Numeric storage exactly **633,600 float32 bytes** per arm, including "
        "the 256x64 fixed or learned projection. Original BC01 CONTENT directly "
        "stored per source episode; original 317 target/event-ID ranking is "
        "offline EXTERNAL oracle-only.",
        "",
        "| Arm, final 317 memories | Source first TRAINED 159 correct | "
        "Third withheld original literal correct | First16 eligible third correct | "
        "Newest16 eligible third correct | Old fourth31 correct | "
        "Third correctly identified AND accepted | Absent71 false accepted |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ARMS:
        c=d["arm_numeric_resources_and_metrics"][name]["groups"]
        gate=d["validation_only_native_novelty_gates"][name]
        lines.append("| "+name+" | "+
            str(c["known_first_training_literal159"]["correct_top1"])+"/159 | "+
            str(c["heldout_third_source_view_all"]["correct_top1"])+"/"+
                str(c["heldout_third_source_view_all"]["n"])+" | "+
            str(c["early16_heldout_third"]["correct_top1"])+"/"+
                str(c["early16_heldout_third"]["n"])+" | "+
            str(c["newest16_heldout_third"]["correct_top1"])+"/"+
                str(c["newest16_heldout_third"]["n"])+" | "+
            str(c["previously_explored_fourth_source31"]["correct_top1"])+"/31 | "+
            str(gate["heldout_third_correct_and_accepted"])+"/"+
                str(gate["heldout_third_n"])+" | "+
            str(gate["absent_test_false_accepted"])+"/71 |")
    lines+=["","## Interpretation boundary","",
        "Training all original 317 paired cue sources before metric fitting "
        "does not demonstrate online continual metric learning. Heldout LAST "
        "source views are distinct within this experimental training but "
        "the same corpus and literals have been examined in preceding pilots, "
        "NOT a blinded independent human paraphrase test. Any positive "
        "third-view result must outperform unchanged-storage random/PCA/"
        "mispaired/lexical and wrong-CONTENT controls, while preserving "
        "absent rejection. The model relies on direct external retrieval of "
        "stored numerical CONTENT codewords and is not Pretorius's fly "
        "synaptic brain nor a natural-language autobiographical decoder.",""]
    (folder/"PILOT18_SOURCE_MIRROR_KEYS_RESULTS.md").write_text(
        "\n".join(lines),encoding="utf-8")
    idx=folder/"README.md"
    old=idx.read_text(encoding="utf-8")
    link="[Pilot18 original two-view semantic metric and withheld source cues](PILOT18_SOURCE_MIRROR_KEYS_RESULTS.md)"
    if link not in old:idx.write_text(old.rstrip()+"\n\n"+link+"\n",encoding="utf-8")
    handoff=ROOT/"docs/RESEARCH_HANDOFF.md"
    old=handoff.read_text(encoding="utf-8")
    marker="## Pilot18 train-only Mirror Keys source run "+run_id
    if marker not in old:
        handoff.write_text(
            old.rstrip()+"\n\n"+marker+"\n\n"+
            "[Original source measured Mirror Keys outcomes]"
            "(../results/imprinting/PILOT18_SOURCE_MIRROR_KEYS_RESULTS.md) "
            "record exact original BC01 source, 317 train events, only "
            "first/middle metric fitting and learned numeric keys with third "
            "original literal source cue heldout, six equal array-size "
            "633600-byte external memory arms, same 62-validation-absent "
            "native threshold, complete original case evidence and six "
            "actually learned numeric projection/episode checkpoints. "
            "No full FlyWire v783 synaptic simulation or new human-authored "
            "paraphrase proof is asserted.\n",encoding="utf-8")

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",required=True,type=Path)
    p.add_argument("--weights",required=True,type=Path)
    p.add_argument("--run-id",required=True)
    args=p.parse_args()
    archive(args.input,args.weights,args.run_id)
