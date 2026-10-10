#!/usr/bin/env python3
"""Pilot15: fixed fast/slow numeric source association vs seven prior controls.

Predeclared original 450-event source, 317 selected training events, canonical
L0/L1/BC01, 3 first/middle/last cues per event, 7 memory-load checkpoints.
Only a separate external oracle identifies event IDs; model inference gets
ONLY a cue and fixed trained weights, never candidate memories or source IDs.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT)]

from pretorius_connectome.associative import Topology
from pretorius_connectome.direct_flywire_imprint import (
    DirectFlywireOverlay,fingerprint
)
from pretorius_connectome.metaplastic14 import (
    UsageProtectedOverlay,MatchedSlotLinear
)
from pretorius_connectome.dual_trace15 import (
    DualTraceSynapses,DualTraceLinear
)
from pretorius_connectome.rewire12 import (
    rewire_effective_edges,synthetic_aggregated_fixture
)
from scripts.run_flywire_pilot13_capacity import (
    source_state,rank_measured,describe,rank_auc,SEED,REWIRE_SEED,
    CUTS,SOURCE_PILOT10_SHA,SOURCE_PILOT10_CHECKPOINT_SHA
)
from scripts.run_flywire_pilot14 import (
    unseen_rows,parity_with_original_pilot10
)
from scripts.run_direct_flywire_imprint10 import binding_cues
from scripts.diagnose_flywire_imprint11 import unseen_literal_source_cue

ARMS=(
    "original_additive","original_beta4",
    "original_dual_beta1","original_dual_beta4",
    "rewired_dual_beta1","original_dual_beta1_deranged",
    "matched_slot_linear","matched_slot_dual_linear",
)
GROUPS=(
    "early16_familiar","newest16_familiar",
    "original159_learned_familiar","original31_truly_unseen",
    "absent71_episode",
)
FAST_DECAY=.97
SLOW_SHARE=.6
RATE=.7/3


def models_for(original,rewired,group,n_slots):
    kw=dict(seed=SEED,cells_per_feature=group,
            cue_features=8,content_features=32,rate=RATE)
    dualkw=dict(
        seed=SEED,cells_per_feature=group,cue_features=8,
        content_features=32,rate=RATE,fast_decay=FAST_DECAY,
        slow_share=SLOW_SHARE
    )
    return {
        "original_additive":DirectFlywireOverlay(original,**kw),
        "original_beta4":UsageProtectedOverlay(
            original,protection_beta=4.0,**kw),
        "original_dual_beta1":DualTraceSynapses(
            original,beta=1.0,**dualkw),
        "original_dual_beta4":DualTraceSynapses(
            original,beta=4.0,**dualkw),
        "rewired_dual_beta1":DualTraceSynapses(
            rewired,beta=1.0,**dualkw),
        "original_dual_beta1_deranged":DualTraceSynapses(
            original,beta=1.0,**dualkw),
        "matched_slot_linear":MatchedSlotLinear(
            slot_count=n_slots,seed=REWIRE_SEED,rate=RATE),
        "matched_slot_dual_linear":DualTraceLinear(
            slot_count=n_slots,seed=REWIRE_SEED,rate=RATE,
            fast_decay=FAST_DECAY,slow_share=SLOW_SHARE),
    }


def save_source_model(name,model,path):
    if isinstance(model,DualTraceSynapses):
        digest=model.save_dual(path)
        replay=DualTraceSynapses.load_dual(model.topology,path)
        if (not np.array_equal(replay.fast.delta,model.fast.delta)
            or not np.array_equal(replay.slow.delta,model.slow.delta)
            or not np.array_equal(
                replay.slow.edge_exposures,model.slow.edge_exposures)
            or replay.completed_memories!=model.completed_memories):
            raise AssertionError("Dual source synaptic checkpoint failed exact round-trip")
        return digest,"double_eligible_numeric_trace_slots"
    if isinstance(model,DualTraceLinear):
        digest=model.save_dual_linear(path)
        with np.load(path,allow_pickle=False) as z:
            if (not np.array_equal(z["fast_weights"],model.fast.weights)
                or not np.array_equal(z["slow_weights"],model.slow.weights)
                or not np.array_equal(z["mask"],model.slow.mask)):
                raise AssertionError("Dual nonneural checkpoint failed exact weight/mask roundtrip")
        return digest,"double_non_neural_trainable_slots"
    if isinstance(model,UsageProtectedOverlay):
        digest=model.save_protected(path)
        replay=UsageProtectedOverlay.load_protected(model.topology,path)
        if (not np.array_equal(replay.delta,model.delta)
            or not np.array_equal(replay.edge_exposures,model.edge_exposures)):
            raise AssertionError("Protected source checkpoint failed roundtrip")
        return digest,"single_eligible_numeric_trace_slots"
    if isinstance(model,MatchedSlotLinear):
        digest=model.save_linear(path)
        with np.load(path,allow_pickle=False) as z:
            if (not np.array_equal(z["weights"],model.weights)
                or not np.array_equal(z["mask"],model.mask)):
                raise AssertionError("Nonneural comparator saved weights/mask disagree")
        return digest,"single_non_neural_trainable_slots"
    digest=model.save(path)
    replay=DirectFlywireOverlay.load(model.topology,path)
    if not np.array_equal(replay.delta,model.delta):
        raise AssertionError("Original Pilot10 baseline source weights failed reload")
    return digest,"single_eligible_numeric_trace_slots"


def run(graph,bc01_dir,*,real=False,group=32,source_cases=None,
        source_checkpoint=None,weights_dir=None):
    (original_sha,memories,meta,bc01,pos,train,positives,
     validation,negatives,targets)=source_state(graph,bc01_dir,real)
    ids=[m.event_id for m in train]
    unseen=[m for m in positives if unseen_literal_source_cue(m)]
    if len(unseen)!=31:
        raise AssertionError("Historical original fourth literal source cue set changed")
    template=DirectFlywireOverlay(
        graph,seed=SEED,cells_per_feature=group,rate=RATE
    )
    rewired,rewiring=rewire_effective_edges(
        graph,template.pre_cells,template.post_cells,
        seed=REWIRE_SEED,swaps_per_edge=2
    )
    slots=rewiring["originally_eligible_edges"]
    models=models_for(graph,rewired,group,slots)
    if set(models)!=set(ARMS):
        raise AssertionError("Original eight preregistered model arms missing")
    permutation=np.roll(np.arange(len(train)),max(1,len(train)//3))
    stages=[]
    for stage_index,cut in enumerate(CUTS):
        if stage_index:
            for i in range(CUTS[stage_index-1],cut):
                original=train[i]
                deranged=train[int(permutation[i])]
                if deranged.event_id==original.event_id:
                    raise AssertionError("Deranged targets cannot be their own cue's source")
                for name,model in models.items():
                    target=(deranged if name.endswith("_deranged") else original)
                    y=bc01[pos[target.event_id]]
                    for cue in binding_cues(original):
                        model.imprint(cue,y)
                    if isinstance(model,(DualTraceSynapses,DualTraceLinear)):
                        model.finish_memory()
        learned=set(ids[:cut])
        source_groups={
            "early16_familiar":train[:16],
            "newest16_familiar":train[max(0,cut-16):cut],
            "original159_learned_familiar":[
                m for m in positives if m.event_id in learned],
            "original31_truly_unseen":[
                m for m in unseen if m.event_id in learned],
            "absent71_episode":negatives,
        }
        record={
            "trained_events":cut,
            "presentations_per_arm":3*cut,
            "all_arms":{},
        }
        for name,model in models.items():
            cases={}
            summaries={}
            for group_name in GROUPS:
                source_records=source_groups[group_name]
                rows=(unseen_rows(model,source_records,ids,targets,learned)
                      if group_name=="original31_truly_unseen"
                      else rank_measured(
                          model,source_records,ids,targets,
                          learned_ids=learned))
                cases[group_name]=rows
                summaries[group_name]=describe(rows)
            if model.imprints!=3*cut:
                raise AssertionError("All model arms require same three-cue presentations")
            if isinstance(model,(DualTraceSynapses,DualTraceLinear)):
                if model.completed_memories!=cut:
                    raise AssertionError("A source record did not complete fast-channel decay")
            if isinstance(model,(DirectFlywireOverlay,DualTraceSynapses)):
                model.assert_original_unchanged()
            record["all_arms"][name]={
                "cases":cases,"summaries":summaries,
                "source_cue_presentations":model.imprints,
                "numeric_update_operations":model.edge_update_events,
                "nonzero_union_weights":model.modified_edges,
                "independent_trainable_scalar_slots":(
                    slots*2 if isinstance(model,(DualTraceSynapses,DualTraceLinear))
                    else slots
                ),
                "trace_diagnostics":(
                    model.exposure_diagnostics
                    if isinstance(model,(DualTraceSynapses,DualTraceLinear))
                    else model.exposure_diagnostics
                    if isinstance(model,UsageProtectedOverlay) else None
                ),
                "known_vs_absent_EXTERNAL_oracle_auc":rank_auc(
                    cases["original159_learned_familiar"],
                    cases["absent71_episode"]
                ),
            }
        stages.append(record)
    baseline=None
    if real:
        if source_cases is None or source_checkpoint is None:
            raise ValueError("Original whole-FlyWire replay requires pinned real Pilot10 evidence")
        baseline=parity_with_original_pilot10(
            models["original_additive"],graph,source_cases,
            source_checkpoint,ids,targets,positives,negatives
        )
    checkpoints={}
    if weights_dir:
        for name,model in models.items():
            p=Path(weights_dir)/(name+".npz")
            h,kind=save_source_model(name,model,p)
            checkpoints[name]={"sha256":h,"path":str(p),"kind":kind}
    if fingerprint(graph)!=original_sha:
        raise AssertionError("Original biological root neurons and synaptic contacts changed")
    return {
        "study":"Pilot15 dual timescale fast and stable source associative synaptic traces",
        "status":("original real publisher-verified full FlyWire-v783"
                  if real else "SYNTHETIC generated source graph, NOT original fly biology"),
        "limits":(
            "Eight predeclared arms use the original 256D BC01 signed lexical "
            "cue and CONTENT vectors. Two-channel states DOUBLE the trainable "
            "numerical synaptic scalar slots, and the two-channel nonneural "
            "matrix comparator is double-slot matched, not gradient, contact "
            "or morphology matched. Slow/fast blending normalizes each "
            "channel BEFORE combining. The original 317 event-ID candidates, "
            "content target similarity ranks and rejection thresholds are "
            "EXTERNAL oracle-only evaluation, NEVER model-native episodic "
            "retrieval. The same explored 159/71/31 source probe sets are "
            "NOT independent confirmations. No human-semantic unseen cues, "
            "fly learning physiology, selfhood or narrative recall is claimed."
        ),
        "source":{
            "canonical_memories":len(memories),
            "episode_count":len({m.episode_id for m in memories}),
            "original_train":len(train),
            "original_familiar_test":len(positives),
            "original_test_absent":len(negatives),
            "original_validation_absent":len(validation),
            "fourth_source_unseen_eligible":len(unseen),
            "BC01_original_sha256":meta["artifact_sha256"],
            "original_graph_array_sha256":original_sha,
            "original_graph_unchanged":fingerprint(graph)==original_sha,
            "original_neurons":len(graph.root_ids),
            "original_directed_edges":len(graph.indices),
            "original_integer_synapses":int(graph.synapse_counts.sum(dtype=np.int64))
        },
        "predeclared":{
            "stages":CUTS,"slow_trace_fraction":SLOW_SHARE,
            "fast_trace_fraction":1-SLOW_SHARE,
            "fast_decay_per_completed_three_cue_memory":FAST_DECAY,
            "slow_usage_beta_values":[1.0,4.0],
            "nominal_source_rate":RATE,
            "all_train_source_cues_exactly_three_per_record":True,
            "external_fixed_source_target_candidates":317,
            "no_historical_159_71_31_test_tuning":True,
            "original_graph_single_slots":slots,
            "dual_trace_slots":2*slots,
            "source_and_linear_comparisons_NOT_update_norm_matched":True
        },
        "degree_preserved_rewired_null":rewiring,
        "original_pilot10_source_baseline_replay":baseline,
        "all_model_checkpoint_sha256":checkpoints,
        "cases_at_all_stages_EXTERNAL_only":stages
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument("--topology",type=Path)
    g.add_argument("--synthetic-test",action="store_true")
    p.add_argument("--bc01-dir",type=Path,required=True)
    p.add_argument("--source-pilot10-cases",type=Path)
    p.add_argument("--source-pilot10-checkpoint",type=Path)
    p.add_argument("--cells-per-feature",type=int)
    p.add_argument("--weights-dir",type=Path)
    p.add_argument("--output",required=True,type=Path)
    a=p.parse_args()
    graph=(synthetic_aggregated_fixture(n=8192,degree=16,seed=15)
           if a.synthetic_test else Topology.read(a.topology))
    result=run(
        graph,a.bc01_dir,real=not a.synthetic_test,
        group=a.cells_per_feature or (8 if a.synthetic_test else 32),
        source_cases=a.source_pilot10_cases,
        source_checkpoint=a.source_pilot10_checkpoint,
        weights_dir=a.weights_dir
    )
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",
                        encoding="utf-8")
    for stage in result["cases_at_all_stages_EXTERNAL_only"]:
        print("STAGE",stage["trained_events"]," ".join(
            f"{arm}:{stage['all_arms'][arm]['summaries']['early16_familiar']['correct_top1']}/16"
            for arm in ARMS),flush=True)
    print("ORIGINAL_CASE_EVIDENCE",a.output)


if __name__=="__main__":
    main()
