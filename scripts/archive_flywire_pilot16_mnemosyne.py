#!/usr/bin/env python3
"""Archive original source-verified Pilot16 Mnemosyne case and trained W/P state.

NEVER label synthetic generated masks as complete FlyWire biology. Fail-closed
validation requires original published four-array SHA, pinned local ONNX SHA,
original unchanged 450 reconstructed source corpus, historical BC01 training
baseline, model-only confidence gate, 10 arms and 7 stage source case sets.
Original real v783 is only a BINARY SOURCE-FEATURE MASK prior, not a direct
15 million original synapse memory readout for Mnemosyne arms.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REAL={
 "root_ids":"84bc9ec32c4f797e3f6e6557b30b5fc7f447b5852aeb9553f7dd18ed6bd3321d",
 "indptr":"bb91b075324c5971600d3c1bc296dabdfc95a18d695c0293a7d09647cff61595",
 "indices":"3180cc427eb92389f22afde12d99b19d6676443aa3b9ffd25589c97013cd404b",
 "synapse_counts":"ef00868a5a54e3c9bdf1c4d64966bdec6fe5946a6f6909937801afcbdc44385e",
}
ONNX="6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452"
BC01="65ae81401a17ec68adc3900395f226734618be4ffb2bae24745d989bd2a17625"
STAGES=(0,16,32,64,128,256,317)
ARMS=(
    "original_BC01_direct_hebb",
    "original_MiniLM_top8_hebb",
    "mnemosyne_dense_rls_original",
    "mnemosyne_dense_rls_rewired",
    "mnemosyne_dense_rls_random_pair_mask",
    "mnemosyne_dense_rls_unmasked_linear",
    "mnemosyne_sparse64_rls_original",
    "mnemosyne_dense_delta_original",
    "mnemosyne_dense_rls_wrong_content",
    "mnemosyne_BC01_dense_rls_original"
)
GROUPS=(
 "earliest16","newest16","original159_learned_familiar",
 "original31_fourth_never_trained","absent71_episode"
)


def verify(d):
    s=d["original_source"]
    p=d["preregistered"]
    mask=d["original_fly_feature_pair_mask"]
    null=d["rewired_binary_degree_preserving_null"]
    gates=d["validation_only_model_native_novelty_gates"]
    if (d.get("study")!=
        "Mnemosyne Pilot16: frozen dense semantic cue learned residual content and source mask controls"
        or d.get("evidence")!="original verified complete publisher FlyWire v783"
        or "EXTERNAL oracle-only" not in d.get("limitations","")
        or s.get("original_biology_array_sha256")!=REAL
        or s.get("original_anatomy_unchanged") is not True
        or (s.get("root_neurons"),s.get("aggregate_directed_pairs"),
            s.get("integer_anatomical_synapses"))!=(139255,15091983,54492922)
        or (s.get("canonical_events"),s.get("canonical_episodes"),
            s.get("trained_source_events"),s.get("heldout_known_test"),
            s.get("validation_absent"),s.get("test_absent"),
            s.get("eligible_never_trained_fourth_literal"))!=(450,27,317,159,62,71,31)
        or s.get("BC01_cache_sha256")!=BC01
        or p.get("frozen_ONNX_sha256")!=ONNX
        or p.get("frozen_ONNX_revision")!="1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
        or tuple(p.get("memory_load_stages",[]))!=STAGES
        or p.get("source_train_presentations_per_event")!=3
        or p.get("fixed_external_source_content_candidates")!=317
        or p.get("input_novelty_validation_only_62_absences") is not True
        or p.get("no_159_31_71_test_tuning") is not True
        or p.get("note_extra_P_inverse_covariance_state_65536_scalars") is not True
        or mask.get("eligible_original_directed_synapse_slots")!=50920
        or mask.get("original_csr_sha256")!=REAL
        or null.get("originally_eligible_edges")!=50920
        or null.get("rewired_eligible_edges")!=50920
        or null.get("source_graph_unchanged") is not True
        or null.get("binary_indegrees_equal") is not True
        or null.get("binary_outdegrees_equal") is not True):
        raise ValueError("Source original biological/oracle/pinned semantic evidence mismatch")
    steps=d["full_original_source_case_stages"]
    if (len(steps)!=len(STAGES)
        or set(d["learned_numerical_state_checkpoint_SHA256"])!=set(ARMS)
        or set(gates)!=set(ARMS)):
        raise ValueError("Incomplete source stages, learned model checkpoints or novelty gates")
    for stage,cut in zip(steps,STAGES):
        if (stage.get("trained_events")!=cut
            or stage.get("cue_presentations_per_arm")!=cut*3
            or set(stage.get("arms",{}))!=set(ARMS)):
            raise ValueError("Wrong source stages or model-arm parity")
        compare=None
        for arm in ARMS:
            x=stage["arms"][arm]
            cases=x["cases_external_oracle_only"]
            if (set(cases)!=set(GROUPS) or set(x["groups"])!=set(GROUPS)
                or x["cue_presentations"]!=cut*3
                or x["groups"]["earliest16"]["n"]!=16
                or x["groups"]["newest16"]["n"]!=min(cut,16)
                or x["groups"]["absent71_episode"]["n"]!=71
                or any(len(cases[group])!=x["groups"][group]["n"] for group in GROUPS)):
                raise ValueError("Missing complete paired original source evaluation cohorts")
            early=[q["event_id"] for q in cases["earliest16"]]
            if compare is None:compare=early
            elif compare!=early:
                raise ValueError("Early original source events not paired across conditions")
    final=steps[-1]["arms"]
    if (final["original_BC01_direct_hebb"]["groups"]["original159_learned_familiar"]["correct_top1"]!=20
        or final["original_BC01_direct_hebb"]["groups"]["original31_fourth_never_trained"]["n"]!=31):

        raise ValueError("Original historical source Pilot10 changed")
    for arm,g in gates.items():
        if arm in ("original_BC01_direct_hebb","original_MiniLM_top8_hebb"):
            if g is not None:
                raise ValueError("Incorrect novelty gate on old no-input-covariance source model")
        else:
            if (g is None or g["validation_absent_n"]!=62
                or g["validation_absent_false_accepted"]>6
                or g["source_test_original_familiar_n"]!=159
                or g["source_test_original_absent_n"]!=71
                or g["source_test_original_unseen_fourth_n"]!=31
                or g["uses_external_event_codebook_in_gate"] is not False):
                raise ValueError("Model-native novelty calibration used test set or source candidate codebook")


def archive(raw,models,run_id):
    if not run_id.isdecimal():raise ValueError("Bad Actions run id")
    b=raw.read_bytes()
    d=json.loads(b)
    verify(d)
    dest=ROOT/"artifacts/imprinting/checkpoints"
    dest.mkdir(parents=True,exist_ok=True)
    for arm in ARMS:
        src=models/(arm+".npz")
        wb=src.read_bytes()
        expected=d["learned_numerical_state_checkpoint_SHA256"][arm]["sha256"]
        if sha256(wb).hexdigest()!=expected or not 0<len(wb)<12*1024*1024:
            raise ValueError("Learned original source numerical state invalid: "+arm)
        path=dest/("pilot16-"+arm+"-v783-run"+run_id+".npz")
        if path.exists() and path.read_bytes()!=wb:
            raise ValueError("Conflicting same-run trained numeric memory source")
        path.write_bytes(wb)
    folder=ROOT/"results/imprinting"
    run_dir=folder/"runs";run_dir.mkdir(parents=True,exist_ok=True)
    original_file=run_dir/("flywire-pilot16-mnemosyne-run"+run_id+".json")
    if original_file.exists() and original_file.read_bytes()!=b:
        raise ValueError("Conflicting same-run raw original case bytes")
    original_file.write_bytes(b)
    lines=[
        "# Pilot16 Mnemosyne original full-FlyWire feature-mask associative memory",
        "",
        "Complete source-verified [actual publisher original FlyWire v783 run "+run_id+
            "](https://github.com/Azimn/Pretorius-Connectome/actions/runs/"+run_id+").",
        "[Complete raw seven-stage ten-arm 450-source-event case JSON](runs/"+
            original_file.name+").",
        "Original case JSON SHA-256: "+sha256(b).hexdigest(),
        "All TEN original source trained weight/covariance NPZ states are preserved"+
            " under artifacts/imprinting/checkpoints with exact original digests in JSON.",
        "",
        "CRITICAL: Except legacy direct Hebbian conditions, Mnemosyne learns on a"+
            " **BINARY 256×256 SOURCE-FEATURE connectivity MASK extracted from the"+
            " genuine full-FlyWire original 15M-edge anatomical CSR**, not on all"+
            " original synapse weights. This is a feature-level biological support"+
            " prior, not direct original-neuron memory modeling.",
        "Original feature-pair structural mask permits "+
            str(d["original_fly_feature_pair_mask"]["effective_learned_feature_pairs"])+
            " W scalar positions, compared with **50,920 original eligible individual"+
            " source/readout directed neuron edges**. Each residual model also"+
            " stores 65,536 separate float64 inverse input-covariance parameters.",
        "",
        "| Original source-linked condition | Earliest16 correct at load16 | "+
            "Earliest16 correct at load317 | Newest16 correct | Familiar159 correct | "+
            "Genuinely untrained fourth source cue31 correct | Native model gate correct"+
            " and accepted familiar | Native absent71 wrongly accepted |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    first=d["full_original_source_case_stages"][1]["arms"]
    final=d["full_original_source_case_stages"][-1]["arms"]
    for arm in ARMS:
        a=first[arm]["groups"];z=final[arm]["groups"]
        gate=d["validation_only_model_native_novelty_gates"][arm]
        lines.append("| "+arm+" | "+
                     str(a["earliest16"]["correct_top1"])+"/16 | "+
                     str(z["earliest16"]["correct_top1"])+"/16 | "+
                     str(z["newest16"]["correct_top1"])+"/16 | "+
                     str(z["original159_learned_familiar"]["correct_top1"])+"/159 | "+
                     str(z["original31_fourth_never_trained"]["correct_top1"])+"/31 | "+
                     (str(gate["source_test_correct_and_accepted_familiar"])+"/159"
                      if gate else "not supported") +" | "+
                     (str(gate["source_test_absent_false_accepted"])+"/71"
                      if gate else "not supported")+" |")
    lines+=["","## Explicit limits","",
            "Original MiniLM pretrained semantic cue knowledge is not encoded"+
            " in fly source neurons, but borrowed externally from a frozen pretrained"+
            " encoder. The learned output is only a 256D BC01 signed CONTENT"+
            " vector. Actual event ID and content scores are produced only"+
            " by an EXTERNAL fixed 317-source codeword diagnostic scorer."+
            " The learned model-native input coverage gate is calibrated with"+
            " 62 validation absent source events, not the 71 original heldout"+
            " absent events or old 159/31 probes. Its low false accept rate"+
            " cannot be separated from correctly accepted known cases."+
            " Re-used source literal cue sets are exploratory, NOT independent"+
            " human-written semantic confirmatory prompts. No autonomous"+
            " autobiographical recollection, insect cognition or definite"+
            " FlyWire topological advantage is implied.",""]
    (folder/"PILOT16_REAL_MNEMOSYNE_RESULTS.md").write_text(
        "\n".join(lines),encoding="utf-8")
    index=folder/"README.md"
    old=index.read_text(encoding="utf-8")
    link="[Pilot16 original real-v783 Mnemosyne source feature memory](PILOT16_REAL_MNEMOSYNE_RESULTS.md)"
    if link not in old:index.write_text(old.rstrip()+"\n\n"+link+"\n",encoding="utf-8")
    handoff=ROOT/"docs/RESEARCH_HANDOFF.md"
    old=handoff.read_text(encoding="utf-8")
    marker="## Pilot16 original whole-FlyWire source feature-mask Mnemosyne "+run_id
    if marker not in old:
        handoff.write_text(
            old.rstrip()+"\n\n"+marker+"\n\n"+
            "[Original source verified Pilot16 Mnemosyne results]"
            "(../results/imprinting/PILOT16_REAL_MNEMOSYNE_RESULTS.md)"+
            " preserve actual ten learned model W/P/cue original evidence"+
            " and the full source case JSON. This is frozen MiniLM dense cue"+
            " representation plus supervised error-corrective source content"+
            " learning subject to binary pairwise support derived from the"+
            " actual original FlyWire graph, NOT a neuron-level memory model."+
            " Comparisons include rewired and random equal-pair controls,"+
            " unmasked nonneural exact ridge, sparse semantic projection,"+
            " wrong source content, lexical dense and original additive."+
            " Source event identity remains EXTERNAL oracle-only.\n",
            encoding="utf-8"
        )

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",required=True,type=Path)
    p.add_argument("--weights",required=True,type=Path)
    p.add_argument("--run-id",required=True)
    o=p.parse_args()
    archive(o.input,o.weights,o.run_id)
