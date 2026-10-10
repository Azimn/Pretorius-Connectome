#!/usr/bin/env python3
"""Fail-closed archive of original real FlyWire v783 Pilot15 numeric source cases.

Check exact source original graph, pinned old Pilot10 baseline case identity,
eight arm/source cohorts, 7 memory loads, double-slot controls and all eight
actual learned checkpoints. Only successful MAIN Actions workflows qualify.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
ARRAY_SHA={
 "root_ids":"84bc9ec32c4f797e3f6e6557b30b5fc7f447b5852aeb9553f7dd18ed6bd3321d",
 "indptr":"bb91b075324c5971600d3c1bc296dabdfc95a18d695c0293a7d09647cff61595",
 "indices":"3180cc427eb92389f22afde12d99b19d6676443aa3b9ffd25589c97013cd404b",
 "synapse_counts":"ef00868a5a54e3c9bdf1c4d64966bdec6fe5946a6f6909937801afcbdc44385e",
}
CUTS=(0,16,32,64,128,256,317)
ARMS=(
 "original_additive","original_beta4",
 "original_dual_beta1","original_dual_beta4",
 "rewired_dual_beta1","original_dual_beta1_deranged",
 "matched_slot_linear","matched_slot_dual_linear",
)
GROUPS=(
 "early16_familiar","newest16_familiar",
 "original159_learned_familiar","original31_truly_unseen",
 "absent71_episode"
)


def verify(d):
    s=d["source"]; pr=d["predeclared"]
    null=d["degree_preserved_rewired_null"]
    replay=d["original_pilot10_source_baseline_replay"]
    if (d.get("study")!="Pilot15 dual timescale fast and stable source associative synaptic traces"
        or d.get("status")!="original real publisher-verified full FlyWire-v783"
        or "EXTERNAL oracle-only" not in d.get("limits","")
        or s.get("original_graph_array_sha256")!=ARRAY_SHA
        or s.get("original_graph_unchanged") is not True
        or (s.get("canonical_memories"),s.get("episode_count"),
            s.get("original_train"),s.get("original_familiar_test"),
            s.get("original_test_absent"),s.get("original_validation_absent"),
            s.get("fourth_source_unseen_eligible"))!=(450,27,317,159,71,62,31)
        or s.get("BC01_original_sha256")!=
            "65ae81401a17ec68adc3900395f226734618be4ffb2bae24745d989bd2a17625"
        or (s.get("original_neurons"),s.get("original_directed_edges"),
            s.get("original_integer_synapses"))!=(139255,15091983,54492922)
        or tuple(pr.get("stages",[]))!=CUTS
        or (pr.get("slow_trace_fraction"),pr.get("fast_trace_fraction"),
            pr.get("fast_decay_per_completed_three_cue_memory"))!=(.6,.4,.97)
        or pr.get("slow_usage_beta_values")!=[1.0,4.0]
        or pr.get("original_graph_single_slots")!=50920
        or pr.get("dual_trace_slots")!=101840
        or pr.get("external_fixed_source_target_candidates")!=317
        or pr.get("no_historical_159_71_31_test_tuning") is not True
        or null.get("original_csr_array_sha256")!=ARRAY_SHA
        or null.get("source_graph_unchanged") is not True
        or null.get("binary_indegrees_equal") is not True
        or null.get("binary_outdegrees_equal") is not True
        or null.get("originally_eligible_edges")!=50920
        or null.get("rewired_eligible_edges")!=50920
        or null.get("changed_edge_destinations",0)<25000
        or replay is None
        or replay.get("historical_checkpoint_sha256")!=
            "fbc2f329993d414cde3ce3a1271a1083098c6513410373e124362045a427c86a"
        or replay.get("source_case_predictions_same_all_159") is not True
        or replay.get("absent_case_predictions_same_all_71") is not True
        or replay.get("original_learned_951_presentations") is not True
        or replay.get("original_full_weight_array_max_float_abs_difference",1)>5e-7):
        raise ValueError("Pilot15 missing original publisher-v783, source-aligned or original baseline evidence")
    stages=d["cases_at_all_stages_EXTERNAL_only"]
    if len(stages)!=len(CUTS) or set(d["all_model_checkpoint_sha256"])!=set(ARMS):
        raise ValueError("Incomplete preregistered source stages or trained model states")
    initial_ids=None
    for stage,cut in zip(stages,CUTS):
        if (stage.get("trained_events")!=cut
            or stage.get("presentations_per_arm")!=cut*3
            or set(stage.get("all_arms",{}))!=set(ARMS)):
            raise ValueError("Source event chronological stages or arm counts changed")
        source_ids=None
        for name in ARMS:
            x=stage["all_arms"][name]
            if (x.get("source_cue_presentations")!=cut*3
                or set(x["cases"])!=set(GROUPS)
                or set(x["summaries"])!=set(GROUPS)
                or x["independent_trainable_scalar_slots"] !=
                    (101840 if "dual" in name else 50920)
                or x["summaries"]["early16_familiar"]["n"]!=16
                or x["summaries"]["newest16_familiar"]["n"]!=min(cut,16)
                or x["summaries"]["absent71_episode"]["n"]!=71
                or any(len(x["cases"][k])!=x["summaries"][k]["n"] for k in GROUPS)):
                raise ValueError("Unequal source case size or wrong single/double scalar resource budget")
            ids=[x["event_id"] for x in x["cases"]["early16_familiar"]]
            if source_ids is None:source_ids=ids
            elif source_ids != ids:
                raise ValueError("Different early-memory subjects compared across arms")
            if initial_ids is None:initial_ids=ids
            elif initial_ids!=ids:
                raise ValueError("Pilot15 early memory identities varied by training load")
            if name in ("original_dual_beta1","original_dual_beta4",
                        "rewired_dual_beta1","original_dual_beta1_deranged",
                        "matched_slot_dual_linear"):
                diag=x["trace_diagnostics"]
                if (diag is None or diag.get("completed_memories")!=cut
                    or diag.get("independent_numeric_trace_slots",
                                diag.get("independent_numeric_trainable_slots"))!=101840):
                    raise ValueError("New memory fast trace/capacity not accounted")
        for group in GROUPS:
            ids=[r["event_id"] for r in stage["all_arms"][ARMS[0]]["cases"][group]]
            for name in ARMS[1:]:
                if ids!=[r["event_id"] for r in stage["all_arms"][name]["cases"][group]]:
                    raise ValueError("Unpaired source event cohorts among original/rewired/nonneural conditions")
    final=stages[-1]["all_arms"]["original_additive"]["summaries"]
    if (final["original159_learned_familiar"]["correct_top1"]!=20
        or final["absent71_episode"]["accepted_at_original_threshold"]!=56
        or final["original31_truly_unseen"]["n"]!=31):
        raise ValueError("Changed exact original source baseline after 317 trained memories")


def archive(raw,models,run_id):
    if not run_id.isdecimal():
        raise ValueError("Invalid Actions id")
    b=raw.read_bytes()
    d=json.loads(b)
    verify(d)
    checkpoint_dir=ROOT/"artifacts/imprinting/checkpoints"
    checkpoint_dir.mkdir(parents=True,exist_ok=True)
    for name in ARMS:
        orig=models/(name+".npz")
        content=orig.read_bytes()
        expected=d["all_model_checkpoint_sha256"][name]["sha256"]
        if sha256(content).hexdigest()!=expected:
            raise ValueError(name+": learned model checkpoint digest mismatch")
        if not 0<len(content)<15*1024*1024:
            raise ValueError(name+": unexpected original source model footprint")
        dest=checkpoint_dir/("pilot15-"+name+"-v783-run"+run_id+".npz")
        if dest.exists() and dest.read_bytes()!=content:
            raise ValueError("Conflicting learned source checkpoint for same original run")
        dest.write_bytes(content)
    base=ROOT/"results/imprinting"
    run_dir=base/"runs"
    run_dir.mkdir(parents=True,exist_ok=True)
    case=run_dir/("flywire-pilot15-real-v783-run"+run_id+".json")
    if case.exists() and case.read_bytes()!=b:
        raise ValueError("Previously archived case evidence bytes differ")
    case.write_bytes(b)
    lines=[
        "# Pilot15 original complete FlyWire v783: dual-timescale memory",
        "",
        "[Source-verified successful original biological numerical run "+run_id+
            "](https://github.com/Azimn/Pretorius-Connectome/actions/runs/"+run_id+").",
        "Original [complete eight-arm case-level seven-stage JSON](runs/"+case.name+").",
        "Original case JSON SHA-256: "+sha256(b).hexdigest(),
        "All EIGHT original source model/checkpoint NPZ files are committed under "+
            "artifacts/imprinting/checkpoints; individual SHA-256s are in the raw JSON.",
        "",
        "All models trained exactly the original 317 reconstructed autobiographical "+
            "source records on original BC01 256D cue and CONTENT vectors; "+
            "the same 159 familiar, 31 previously untrained fourth original "+
            "source cues and 71 episode-absent evaluation cases are preserved.",
        "Original immutable biological source: 139,255 root neurons, "+
            "15,091,983 directed edges and 54,492,922 original integer "+
            "synaptic contacts; four original CSR array hashes pinned.",
        "IMPORTANT: two source or non-neural traces require **101,840** "+
            "trainable scalar numeric slots versus **50,920** in the single "+
            "channel baseline. Numerical resource and update norm differs.",
        "",
        "| Model condition | First16 correct at load16 | First16 correct at load317 | "+
            "Newest16 correct | Familiar159 correct | Untrained fourth cue31 correct | "+
            "Absent71 falsely accepted |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    stages=d["cases_at_all_stages_EXTERNAL_only"]
    for name in ARMS:
        first=stages[1]["all_arms"][name]["summaries"]
        end=stages[-1]["all_arms"][name]["summaries"]
        lines.append("| "+name+" | "+
                     str(first["early16_familiar"]["correct_top1"])+"/16 | "+
                     str(end["early16_familiar"]["correct_top1"])+"/16 | "+
                     str(end["newest16_familiar"]["correct_top1"])+"/16 | "+
                     str(end["original159_learned_familiar"]["correct_top1"])+"/159 | "+
                     str(end["original31_truly_unseen"]["correct_top1"])+"/31 | "+
                     str(end["absent71_episode"]["accepted_at_original_threshold"])+"/71 |")
    lines += [
        "",
        "## Evidence boundary",
        "",
        "Source original and rewired anatomical graphs are not modified. "+
            "The original source-based synaptic numerical model never sees "+
            "event IDs, source narratives, candidate memory codewords "+
            "or a source event retrieval index during inference; every "+
            "source ID and similarity threshold is an EXTERNAL oracle-only "+
            "evaluation. Dual traces double learned scalar source-state "+
            "capacity; the non-neural dual linear control matches source "+
            "parameter SLOT COUNT only, not contact strength, synaptic "+
            "update magnitude, computer cost or fly-specific neural motifs. "+
            "Fourth source-cue evaluation has been used exploratorily in "+
            "prior pilots; this does not establish independently heldout "+
            "human semantic generalization or biological fly learning.",
        "",
    ]
    (base/"PILOT15_REAL_DUAL_TRACE_RESULTS.md").write_text("\n".join(lines),encoding="utf-8")
    idx=base/"README.md"
    txt=idx.read_text(encoding="utf-8")
    link="[Pilot15 original real FlyWire dual trace](PILOT15_REAL_DUAL_TRACE_RESULTS.md)"
    if link not in txt:
        idx.write_text(txt.rstrip()+"\n\n"+link+"\n",encoding="utf-8")
    handoff=ROOT/"docs/RESEARCH_HANDOFF.md"
    txt=handoff.read_text(encoding="utf-8")
    marker="## Original FlyWire Pilot15 dual trace source run "+run_id
    if marker not in txt:
        handoff.write_text(
            txt.rstrip()+"\n\n"+marker+"\n\n"+
            "[Original source case-level real Pilot15 results]"
            "(../results/imprinting/PILOT15_REAL_DUAL_TRACE_RESULTS.md) "+
            "now compare eight preregistered numerical fast/slow source "+
            "synaptic trace, degree-switched biological connection, deranged "+
            "content pairing and exact single/double numeric slot matched "+
            "nonneural matrices. Full 317 original training cases, paired "+
            "early/new 16, previously used source 159/31/71 and externally "+
            "assigned content identity remain source-pinned. All eight actual "+
            "learned numerical states and the original case JSON are in "+
            "permanent Git history, with their exact SHA and original "+
            "anatomical CSR immutability. No source labels are held by "+
            "the neural model and this is not native narrative recall.\n",
            encoding="utf-8"
        )

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",required=True,type=Path)
    p.add_argument("--weights",required=True,type=Path)
    p.add_argument("--run-id",required=True)
    a=p.parse_args()
    archive(a.input,a.weights,a.run_id)
