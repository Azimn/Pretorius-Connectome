#!/usr/bin/env python3
"""Fail-closed original whole FlyWire Pilot14 source and seven-weight archiver.

Permanently copy original full case JSON bytes, all numerical trained edge
overlays and non-neural fixed-slot masked weights. Synthetic output or invalid
original source baseline will not pass archival verification.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHA = {
    "root_ids": "84bc9ec32c4f797e3f6e6557b30b5fc7f447b5852aeb9553f7dd18ed6bd3321d",
    "indptr": "bb91b075324c5971600d3c1bc296dabdfc95a18d695c0293a7d09647cff61595",
    "indices": "3180cc427eb92389f22afde12d99b19d6676443aa3b9ffd25589c97013cd404b",
    "synapse_counts": "ef00868a5a54e3c9bdf1c4d64966bdec6fe5946a6f6909937801afcbdc44385e",
}
ARMS = (
    "original_baseline", "rewired_baseline", "original_beta1",
    "original_beta4", "rewired_beta1", "original_beta1_deranged",
    "matched_slot_linear",
)
CUTS = (0, 16, 32, 64, 128, 256, 317)
GROUPS = (
    "early16_familiar", "newest16_familiar",
    "original159_learned_familiar", "original31_truly_unseen",
    "absent71_episode",
)


def verify_report(d):
    s, prior, n = d["source"], d["preregistered"], d["rewired_null"]
    replay = d["original_terminal_baseline_replay"]
    if (d.get("study") !=
        "Pilot14 usage-dependent synaptic stability and matched-slot linear source control"
        or d.get("source_type") != "verified complete original FlyWire v783"
        or "EXTERNAL oracle-only" not in d.get("scientific_limits", "")
        or s.get("original_anatomical_array_sha256") != SHA
        or s.get("original_anatomy_unchanged") is not True
        or (s.get("root_neurons"),s.get("directed_pairs"),
            s.get("original_integer_contacts"))!=(139255,15091983,54492922)
        or (s.get("canonical_memories"),s.get("original_train_memories"),
            s.get("original_test_familiar"),s.get("original_test_absent_episode"),
            s.get("original_validation_absent"),
            s.get("original_untrained_source_fourth_cues")) != (450,317,159,71,62,31)
        or s.get("original_bc01_artifact_sha256") !=
            "65ae81401a17ec68adc3900395f226734618be4ffb2bae24745d989bd2a17625"
        or prior.get("training_stages") != list(CUTS)
        or prior.get("protected_beta_values") != [1.0,4.0]
        or prior.get("fixed_external_candidate_count") != 317
        or prior.get("all_input_top_features") != 8
        or prior.get("all_content_top_features") != 32
        or prior.get("original_159_test_not_used_for_tuning") is not True
        or prior.get("original_71_absent_test_not_used_for_tuning") is not True
        or n.get("original_csr_array_sha256") != SHA
        or n.get("source_graph_unchanged") is not True
        or n.get("binary_indegrees_equal") is not True
        or n.get("binary_outdegrees_equal") is not True
        or n.get("originally_eligible_edges") != 50920
        or n.get("rewired_eligible_edges") != 50920
        or n.get("changed_edge_destinations",0) < 25000
        or replay is None
        or replay.get("historical_checkpoint_sha256") !=
            "fbc2f329993d414cde3ce3a1271a1083098c6513410373e124362045a427c86a"
        or replay.get("original_full_weight_array_max_float_abs_difference",1) > 5e-7
        or replay.get("source_case_predictions_same_all_159") is not True
        or replay.get("absent_case_predictions_same_all_71") is not True
        or replay.get("original_learned_951_presentations") is not True):
        raise ValueError("Not original publisher-verified v783 source and pinned historical baseline")
    steps = d["stage_case_evidence"]
    if len(steps) != len(CUTS):
        raise ValueError("Expected all seven source memory capacity stages")
    for data, cutoff in zip(steps,CUTS):
        if (data.get("trained_events") != cutoff
            or data.get("presentations_per_condition") != 3*cutoff
            or set(data.get("arms",{})) != set(ARMS)):
            raise ValueError("Unequal source exposure or missing control model")
        anchors = None
        for arm in ARMS:
            payload = data["arms"][arm]
            groups = payload["cases"]
            if (set(groups) != set(GROUPS)
                or set(payload["summaries"]) != set(GROUPS)
                or payload["total_trainable_parameter_slots"] != 50920):
                raise ValueError("Arm controls or matched original trainable slot count changed")
            for kind, size in (
                ("early16_familiar", 16),
                ("newest16_familiar", min(cutoff,16)),
                ("absent71_episode", 71),
            ):
                if (len(groups[kind]) != size
                    or payload["summaries"][kind]["n"] != size):
                    raise ValueError("Missing original paired evaluator cases")
            ids=[x["event_id"] for x in groups["early16_familiar"]]
            if anchors is None:
                anchors=ids
            elif ids != anchors:
                raise ValueError("Unpaired earliest 16 source memories")
            if payload["cumulative_edge_update_operations"] < 0:
                raise ValueError("Invalid negative learned update tally")
    last=steps[-1]["arms"]
    if (last["original_baseline"]["summaries"]["original159_learned_familiar"]["correct_top1"]!=20
        or last["original_baseline"]["summaries"]["absent71_episode"]["accepted_at_original_threshold"]!=56
        or len(last["original_baseline"]["cases"]["original31_truly_unseen"])!=31
        or last["original_baseline"]["nonzero_learned_weights"] < 28938
        or set(d.get("complete_learned_checkpoints",{})) != set(ARMS)):
        raise ValueError("Original historical Pilot10 end baseline or checkpoint manifest changed")
    for name in ("original_beta1","original_beta4","rewired_beta1",
                 "original_beta1_deranged"):
        exposure = last[name]["biological_edge_exposure_distribution"]
        if (exposure is None or exposure["max_exposures_of_one_directed_edge"] < 2
            or exposure["ever_touched_directed_edges"] < 1):
            raise ValueError("No original local synapse exposure traces for protected arm")


def archive(path, weight_dir, run_id):
    raw=path.read_bytes()
    result=json.loads(raw)
    verify_report(result)
    if not run_id.isdecimal():
        raise ValueError("Actions original source run_id malformed")
    checkpoint_dir=ROOT/"artifacts/imprinting/checkpoints"
    checkpoint_dir.mkdir(parents=True,exist_ok=True)
    for name in ARMS:
        src=weight_dir/(name+".npz")
        b=src.read_bytes()
        if sha256(b).hexdigest() != result["complete_learned_checkpoints"][name]["sha256"]:
            raise ValueError("Learned source checkpoint hash mismatch: "+name)
        if not 0<len(b)<12*1024*1024:
            raise ValueError("Unexpected learned model checkpoint size")
        dest=checkpoint_dir/("pilot14-"+name+"-v783-run"+run_id+".npz")
        if dest.exists() and dest.read_bytes()!=b:
            raise ValueError("Different weights previously archived for same original run")
        dest.write_bytes(b)
    out=ROOT/"results/imprinting"
    target=out/"runs"/("flywire-pilot14-real-v783-run"+run_id+".json")
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists() and target.read_bytes()!=raw:
        raise ValueError("Original same-run case evidence bytes differ")
    target.write_bytes(raw)
    report=[
        "# Pilot14 original full FlyWire v783: synaptic usage protection",
        "",
        "[Publisher-verified whole-v783 original run "+run_id+
        "](https://github.com/Azimn/Pretorius-Connectome/actions/runs/"+run_id+").",
        "Complete original [all source case JSON](runs/"+target.name+").",
        "Original full JSON SHA-256: "+sha256(raw).hexdigest(),
        "All SEVEN original trained model checkpoints, including protected synaptic "+
        "exposure counters, preserved under artifacts/imprinting/checkpoints.",
        "",
        "Original 450 autobiographical reconstructed source events, 317 trained, "+
        "159 familiar original source test probes, 31 original untrained fourth "+
        "literal cue cases, and 71 never-imprinted test-episode absences.",
        "Original biology: 139,255 neurons, 15,091,983 directed pairs, "+
        "54,492,922 integer synaptic contacts, all original publisher arrays unchanged.",
        "All neural model-only inference is cue-to-signed-256D content features. "+
        "Source event IDs and accepted guesses are EXTERNAL oracle-only diagnostics.",
        "",
        "| Arm | Early correct after 16 (of 16) | Early correct after 317 (of 16) | "+
        "Newest16 correct at 317 | All original 159 familiar correct | "+
        "Original 31 novel source cues correct | Absent71 false accepted |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    first=steps[1]["arms"] if (steps:=result["stage_case_evidence"]) else {}
    end=steps[-1]["arms"]
    for name in ARMS:
        a=first[name]["summaries"]
        b=end[name]["summaries"]
        report.append("| "+name+" | "+
                      str(a["early16_familiar"]["correct_top1"])+"/16 | "+
                      str(b["early16_familiar"]["correct_top1"])+"/16 | "+
                      str(b["newest16_familiar"]["correct_top1"])+"/16 | "+
                      str(b["original159_learned_familiar"]["correct_top1"])+"/159 | "+
                      str(b["original31_truly_unseen"]["correct_top1"])+"/31 | "+
                      str(b["absent71_episode"]["accepted_at_original_threshold"])+"/71 |")
    report += [
        "",
        "**Boundaries:** β-protection changes the effective write gain, and any "+
        "retained early event identity must be weighed against newly imprinted "+
        "event recall. The nonneural linear control is matched only in scalar "+
        "parameter COUNT, not computation, anatomical contact strengths or "+
        "plasticity gain. The real FlyWire-configuration rewired null preserves "+
        "selected binary source/target degrees and edge slots, not weighted target "+
        "strength. Neither source cue transfer nor autonomous recollection, "+
        "true fly metaplasticity or definitive Pretorius is established.",
        "",
    ]
    (out/"PILOT14_REAL_SYNAPTIC_STABILITY_RESULTS.md").write_text(
        "\n".join(report),encoding="utf-8")
    index=out/"README.md"
    text=index.read_text(encoding="utf-8")
    link="[Pilot14 real FlyWire synaptic stability](PILOT14_REAL_SYNAPTIC_STABILITY_RESULTS.md)"
    if link not in text:
        index.write_text(text.rstrip()+"\n\n"+link+"\n",encoding="utf-8")
    handoff=ROOT/"docs/RESEARCH_HANDOFF.md"
    content=handoff.read_text(encoding="utf-8")
    marker="## Whole-FlyWire Pilot14 synaptic stability source run "+run_id
    if marker not in content:
        handoff.write_text(
            content.rstrip()+"\n\n"+marker+"\n\n"+
            "[Original measured Pilot14 cases and all seven source-bound synaptic states]"
            "(../results/imprinting/PILOT14_REAL_SYNAPTIC_STABILITY_RESULTS.md) "+
            "now compare original Hebbian numerical learning, local exposure-count "+
            "protected writing, original-derived rewired controls, wrong-content "+
            "negative control and a 50,920-scalar slot-count-matched non-neural "+
            "linear comparator across the fixed 0–317 source-event memory loads. "+
            "The original Pilot10 source baseline reproduces exact original "+
            "159-known and 71-absent classifications with separately reported "+
            "numerical roundoff. There is no candidate event-ID codebook "+
            "inside neural inference, only in the EXTERNAL scorer.\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",required=True,type=Path)
    p.add_argument("--weights",required=True,type=Path)
    p.add_argument("--run-id",required=True)
    a=p.parse_args()
    archive(a.input,a.weights,a.run_id)
