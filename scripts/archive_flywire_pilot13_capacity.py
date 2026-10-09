#!/usr/bin/env python3
"""Archive exact real whole-FlyWire Pilot13 case evidence and four learned states.

Fail-closed provenance verification: original source arrays, frozen Pilot10
terminal checkpoint replay, all stages/arms and exact external-oracle case
IDs. Never archive synthetic numerical findings as whole-brain results.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARRAY_SHA = {
    "root_ids": "84bc9ec32c4f797e3f6e6557b30b5fc7f447b5852aeb9553f7dd18ed6bd3321d",
    "indptr": "bb91b075324c5971600d3c1bc296dabdfc95a18d695c0293a7d09647cff61595",
    "indices": "3180cc427eb92389f22afde12d99b19d6676443aa3b9ffd25589c97013cd404b",
    "synapse_counts": "ef00868a5a54e3c9bdf1c4d64966bdec6fe5946a6f6909937801afcbdc44385e",
}
CUTS = (0,16,32,64,128,256,317)
ARMS = ("original_learned","rewired_learned","original_deranged","rewired_deranged")
GROUPS = ("first16_anchors","newest16","source_test_known_learned",
          "next16_not_yet_learned","episode_absent_71")


def verify(data):
    source=data["source"]
    prec=data["predeclared"]
    null=data["rewired_null"]
    parity=data["frozen_original_pilot10_parity"]
    if (data.get("study")!="Pilot13 source-linked synaptic memory capacity and interference"
        or data.get("evidence_class")!="verified original full FlyWire v783"
        or "EXTERNAL" not in data.get("limitations","")
        or source.get("original_anatomical_array_sha256")!=ARRAY_SHA
        or source.get("original_anatomical_arrays_unchanged") is not True
        or (source.get("original_neurons"),source.get("original_directed_pairs"),
            source.get("original_integer_synapse_contacts"))!=(139255,15091983,54492922)
        or (source.get("canonical_event_count"),source.get("canonical_episode_count"),
            source.get("trained_event_count"),source.get("test_probe_event_count"),
            source.get("validation_absent_count"),source.get("test_absent_count"))!=
            (450,27,317,159,62,71)
        or source.get("original_BC01_cache_sha256")!=
            "65ae81401a17ec68adc3900395f226734618be4ffb2bae24745d989bd2a17625"
        or parity is None
        or parity.get("loaded_full_original_checkpoint_matches_weights_exactly") is not True
        or parity.get("original_159_cases_identical") is not True
        or parity.get("original_71_absent_cases_identical") is not True
        or parity.get("source_original_checkpoint_sha256")!=
            "fbc2f329993d414cde3ce3a1271a1083098c6513410373e124362045a427c86a"
        or parity.get("prior_original_cases_sha256")!=
            "ce2d463a72fa58a1e65265cf50f961f9e585610d058b9701d26791e033e7cea8"):
        raise ValueError("Not pinned original biological source and original Pilot10 terminal replay")
    if (tuple(prec["stages"])!=CUTS
        or prec["fixed_oracle_candidate_count_at_all_stages"]!=317
        or prec["anchor_early_train_events"]!=16
        or prec["latest_cohort"]!=16
        or prec["fixed_historical_acceptance_threshold"]!=0.0082783
        or prec["no_threshold_fitting_or_encoder_tuning_in_this_pilot"] is not True
        or null.get("source_graph_unchanged") is not True
        or null.get("original_csr_array_sha256")!=ARRAY_SHA
        or null.get("binary_outdegrees_equal") is not True
        or null.get("binary_indegrees_equal") is not True
        or null.get("originally_eligible_edges")!=null.get("rewired_eligible_edges")
        or null.get("changed_edge_destinations",0)<=0):
        raise ValueError("Original preregistration stages, candidate universe or rewiring null altered")
    stages=data["stages"]
    if len(stages)!=len(CUTS):
        raise ValueError("Staged source evidence missing")
    previous_anchor={name:None for name in ARMS}
    for stage,cut in zip(stages,CUTS):
        if (stage["trained_source_memories"]!=cut
            or stage["total_source_cue_exposures_per_arm"]!=cut*3
            or set(stage["measured"])!=set(ARMS)
            or set(stage["original_case_rows_external_oracle_only"])!=set(ARMS)):
            raise ValueError("Wrong memory-load order, learned exposure count, or arms")
        for name in ARMS:
            m=stage["measured"][name]
            group=stage["original_case_rows_external_oracle_only"][name]
            if (m["imprint_presentations"]!=cut*3
                or set(m["groups"])!=set(GROUPS)
                or set(group)!=set(GROUPS)
                or any(len(group[g])!=m["groups"][g]["n"] for g in GROUPS)
                or len(group["first16_anchors"])!=16
                or len(group["episode_absent_71"])!=71
                or len(group["newest16"])!=(min(cut,16))
                or len(group["next16_not_yet_learned"])!=min(317-cut,16)):
                raise ValueError("Source case shape or per-arm training stage mismatch")
            ids=[r["event_id"] for r in group["first16_anchors"]]
            if previous_anchor[name] is not None and ids!=previous_anchor[name]:
                raise ValueError("Anchor case identities not identical across memory loads")
            previous_anchor[name]=ids
            if any(r["was_imprinted_by_this_stage"] is not False
                   for r in group["next16_not_yet_learned"]):
                raise ValueError("Leakage: novel next-source events already trained")
            if cut>=16 and any(r["was_imprinted_by_this_stage"] is not True
                               for r in group["first16_anchors"]):
                raise ValueError("Anchors not actually exposed")
            if (cut==0 and m["learned_nonzero_synaptic_edges"]!=0
                or cut==317 and m["learned_nonzero_synaptic_edges"]<=0):
                raise ValueError("Frozen no-training or trained overlay invalid")
        for g in GROUPS:
            baseline=[r["event_id"] for r in
                      stage["original_case_rows_external_oracle_only"][ARMS[0]][g]]
            if any(baseline != [
                    r["event_id"] for r in
                    stage["original_case_rows_external_oracle_only"][arm][g]]
                    for arm in ARMS[1:]):
                raise ValueError("Not paired source case IDs among matched arm conditions")
    last=stages[-1]["measured"]["original_learned"]["groups"]
    if (last["source_test_known_learned"]["n"]!=159
        or last["source_test_known_learned"]["correct_top1"]!=20
        or last["episode_absent_71"]["absent_false_acceptances"]!=56
        or parity["terminal_changed_edge_positions"]!=28938):
        raise ValueError("Original Pilot10 baseline terminal numerical scores shifted")
    if set(data["learned_checkpoint_manifest"])!=set(ARMS):
        raise ValueError("Missing any learned original source-bound synaptic checkpoint")


def archive(raw:Path,weights:Path,run_id:str):
    data_bytes=raw.read_bytes()
    evidence=json.loads(data_bytes)
    verify(evidence)
    if not run_id.isdecimal():
        raise ValueError("Invalid Actions run ID")
    checkpoint_dir=ROOT/"artifacts/imprinting/checkpoints"
    checkpoint_dir.mkdir(parents=True,exist_ok=True)
    for name in ARMS:
        original=weights/(name+".npz")
        b=original.read_bytes()
        if sha256(b).hexdigest()!=evidence["learned_checkpoint_manifest"][name]["sha256"]:
            raise ValueError("Original source synaptic checkpoint hash mismatch: "+name)
        if not 0<len(b)<10*1024*1024:
            raise ValueError("Unexpected original learned checkpoint footprint")
        saved=checkpoint_dir/("pilot13-"+name+"-v783-run"+run_id+".npz")
        if saved.exists() and saved.read_bytes()!=b:
            raise ValueError("Existing learned source weights conflict with run")
        saved.write_bytes(b)
    results=ROOT/"results/imprinting"
    original_dir=results/"runs"
    original_dir.mkdir(parents=True,exist_ok=True)
    case_file=original_dir/("flywire-pilot13-capacity-run"+run_id+".json")
    if case_file.exists() and case_file.read_bytes()!=data_bytes:
        raise ValueError("Conflicting identical run case bytes")
    case_file.write_bytes(data_bytes)
    lines=[
        "# Pilot13 real original FlyWire v783 synaptic capacity and interference",
        "",
        "Original real [full whole-brain Actions run "+run_id+
            "](https://github.com/Azimn/Pretorius-Connectome/actions/runs/"+run_id+").",
        "Complete original [source-paired seven-stage four-arm case JSON](runs/"+
            case_file.name+").",
        "Original JSON SHA-256: "+sha256(data_bytes).hexdigest(),
        "The four actual original synaptic checkpoints are committed under "+
            "artifacts/imprinting/checkpoints with SHA-256s recorded in this original JSON.",
        "",
        "The 450-source-event canonical v12 corpus, 317 learning events, 159 "+
            "previous source-positive probes, 71 truly absent test-episode queries "+
            "and original fixed 317-event external ranking universe remain unchanged.",
        "The original 139,255-neuron, 15,091,983-pair, 54,492,922-contact FlyWire v783 "+
            "anatomical CSR remained unchanged. Learned synapses are separate numeric deltas.",
        "",
        "| Original source memories learned | Original: early-16 correct | "+
            "Original: early-16 mean source margin | Rewired: early-16 correct | "+
            "Deranged original: early-16 correct | Original learned synaptic edges | "+
            "Original absent false accept (of 71) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for stage in evidence["stages"]:
        m=stage["measured"]
        original=m["original_learned"]
        rewired=m["rewired_learned"]
        wrong=m["original_deranged"]
        lines.append("| "+str(stage["trained_source_memories"])+" | "+
                     str(original["groups"]["first16_anchors"].get("correct_top1",0))+"/16 | "+
                     str(original["groups"]["first16_anchors"].get("mean_target_margin","n/a"))+" | "+
                     str(rewired["groups"]["first16_anchors"].get("correct_top1",0))+"/16 | "+
                     str(wrong["groups"]["first16_anchors"].get("correct_top1",0))+"/16 | "+
                     str(original["learned_nonzero_synaptic_edges"])+" | "+
                     str(original["groups"]["episode_absent_71"]["absent_false_acceptances"])+"/71 |")
    lines.extend([
        "",
        "**Causal boundary:** These are original signed lexical BC01 literal-source-cue "+
        "numerical assays and do not independently test semantic or naturally phrased "+
        "memory recall. Memory event IDs and cosine rankings exist exclusively in an "+
        "EXTERNAL evaluator with all 317 target codewords, never in neural infer(cue). "+
        "The rewired null matches eligible source/target binary degrees and writable "+
        "edge slots, not contact-weighted output degree or terminal nonzero learned "+
        "delta count. Stagewise anchor interference is exploratory, single-seed and "+
        "source-specific, not proof of fly cognition or a definitive Pretorius.",
        "",
    ])
    (results/"PILOT13_REAL_CAPACITY_RESULTS.md").write_text(
        "\n".join(lines),encoding="utf-8"
    )
    index=results/"README.md"
    old=index.read_text(encoding="utf-8") if index.exists() else "# Imprinting results\n"
    entry="[Pilot13 original real-v783 staged memory capacity](PILOT13_REAL_CAPACITY_RESULTS.md)"
    if entry not in old:
        index.write_text(old.rstrip()+"\n\n"+entry+"\n",encoding="utf-8")
    handoff=ROOT/"docs/RESEARCH_HANDOFF.md"
    old=handoff.read_text(encoding="utf-8")
    marker="## Pilot13 source-cue interference, real full FlyWire run "+run_id
    if marker not in old:
        handoff.write_text(
            old.rstrip()+"\n\n"+marker+"\n\n"+
            "[Complete case-level real biological-connectivity capacity results]"+
            "(../results/imprinting/PILOT13_REAL_CAPACITY_RESULTS.md) "+
            "contain seven predeclared memory loads, paired original-vs-rewired "+
            "correct/deranged associations, anchor stability, source-target cosine "+
            "margins, unchanged original v783 anatomy and all learned numerical weights. "+
            "The terminal original source branch reproduced the *entire* original "+
            "Pilot10 synaptic delta array and historical source event decisions, "+
            "preventing source drift. This is external-oracle-assisted lexical "+
            "numerical memory measurement, not autonomous source recollection.\n",
            encoding="utf-8"
        )


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--weights",type=Path,required=True)
    p.add_argument("--run-id",required=True)
    args=p.parse_args()
    archive(args.input,args.weights,args.run_id)
