#!/usr/bin/env python3
"""Pilot16 Mnemosyne: dense MiniLM + error-corrective memory on FlyWire mask.

IMPORTANT: Original full FlyWire v783 is reduced to a binary 256x256
feature-pair connectivity constraint. This is a biologically derived SUPPORT
prior, NOT the unchanged direct 15M-edge neural readout of Pilots08-15.

Original source 450 v12 events, L1, original BC01 targets and episode splits
are fixed. MiniLM is frozen pinned local ONNX. RLS is masked approximate RLS
on graph-derived pair supports; the unconstrained nonneural ridge arm is exact
sequential least squares. An online inverse covariance P (65,536 scalars)
is extra learning state and is counted in every arm.

The model receives ONLY a literal cue, never source event IDs, source text
archive, candidate memory codebook or expected event IDs during inference.
Event IDs and all correctness are calculated EXTERNALLY for measurement.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import sys
from hashlib import sha256
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT)]

from pretorius_connectome.associative import Topology
from pretorius_connectome.direct_flywire_imprint import (
    DirectFlywireOverlay,fingerprint,select_features
)
from pretorius_connectome.mnemosyne16 import (
    ResidualAssociativeMemory,actual_feature_support,random_feature_support,
    normalized
)
from pretorius_connectome.semantic_cue12 import (
    CueOnlyOverlay,FrozenMiniLMCues,MODEL,REVISION,projection_sha256
)
from pretorius_connectome.shared_features_bc01 import encode_bc_sensory
from pretorius_connectome.rewire12 import (
    rewire_effective_edges,synthetic_aggregated_fixture
)
from scripts.run_flywire_pilot13_capacity import (
    source_state,rank_measured,describe,rank_auc,SEED,REWIRE_SEED,CUTS
)
from scripts.run_direct_flywire_imprint10 import binding_cues
from scripts.diagnose_flywire_imprint11 import unseen_literal_source_cue

ONNX_SHA="6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452"
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
    "mnemosyne_BC01_dense_rls_original",
)
GROUPS=(
    "earliest16","newest16","original159_learned_familiar",
    "original31_fourth_never_trained","absent71_episode"
)


def make_models(graph,rewired,source_mask,rewired_mask,*,semantic,group):
    pair_count=int(source_mask.sum())
    common=dict(ridge=1.,rate=.4)
    return {
        "original_BC01_direct_hebb":DirectFlywireOverlay(
            graph,seed=SEED,cells_per_feature=group,rate=.7/3),
        "original_MiniLM_top8_hebb":CueOnlyOverlay(
            graph,cue_encoder=semantic,seed=SEED,cells_per_feature=group,
            cue_features=8,content_features=32,rate=.7/3),
        "mnemosyne_dense_rls_original":ResidualAssociativeMemory(
            encoder=semantic,mask=source_mask,rule="rls",**common),
        "mnemosyne_dense_rls_rewired":ResidualAssociativeMemory(
            encoder=semantic,mask=rewired_mask,rule="rls",**common),
        "mnemosyne_dense_rls_random_pair_mask":ResidualAssociativeMemory(
            encoder=semantic,mask=random_feature_support(pair_count,seed=REWIRE_SEED),
            rule="rls",**common),
        "mnemosyne_dense_rls_unmasked_linear":ResidualAssociativeMemory(
            encoder=semantic,mask=None,rule="rls",**common),
        "mnemosyne_sparse64_rls_original":ResidualAssociativeMemory(
            encoder=semantic,mask=source_mask,rule="rls",
            representation="kc64",separation_k=64,**common),
        "mnemosyne_dense_delta_original":ResidualAssociativeMemory(
            encoder=semantic,mask=source_mask,rule="delta",**common),
        "mnemosyne_dense_rls_wrong_content":ResidualAssociativeMemory(
            encoder=semantic,mask=source_mask,rule="rls",**common),
        "mnemosyne_BC01_dense_rls_original":ResidualAssociativeMemory(
            encoder=encode_bc_sensory,mask=source_mask,rule="rls",**common)
    }


def score_rows(model,records,ids,targets,known,*,fourth=False):
    """Source case identity comes ONLY from outer offline oracle evaluator."""
    pos={e:i for i,e in enumerate(ids)}
    out=[]
    for item in records:
        cue=(unseen_literal_source_cue(item) if fourth
             else binding_cues(item)[-1])
        if not cue:raise ValueError("Requested truly untrained cue absent")
        response=model.infer(cue)
        sims=targets@response
        norm=float(np.linalg.norm(response))
        best=int(np.argmax(sims))
        guess=ids[best] if norm>1e-10 else None
        index=pos.get(item.event_id)
        own=float(sims[index]) if index is not None else None
        alt=(float(max(np.max(sims[:index]) if index else -np.inf,
                      np.max(sims[index+1:]) if index+1<len(sims) else -np.inf))
             if index is not None else None)
        native=(float(model.cue_familiarity(cue))
                if isinstance(model,ResidualAssociativeMemory) else None)
        out.append({
            "event_id":item.event_id,
            "source_probe_kind":("never_presented_original_fourth_source_literal"
                                 if fourth else "trained_last_source_literal"),
            "already_trained":item.event_id in known,
            "known_to_external_317_event_oracle":index is not None,
            "model_native_cue_familiarity":native,
            "predicted_external_event_id":guess,
            "correct_external_top1":bool(guess==item.event_id) if index is not None else None,
            "external_top_cosine":round(float(sims[best]),7) if norm>1e-10 else 0.,
            "external_correct_target_cosine":round(own,7) if own is not None else None,
            "external_correct_target_margin":round(own-alt,7) if own is not None else None,
            "nonzero_model_content_signal":bool(norm>1e-10),
        })
    return out


def stats(rows):
    if not rows:return {"n":0}
    known=all(r["known_to_external_317_event_oracle"] for r in rows)
    if any(r["known_to_external_317_event_oracle"]!=known for r in rows):
        raise AssertionError("Do not mix known and absent case populations")
    x={"n":len(rows),"model_active":sum(r["nonzero_model_content_signal"] for r in rows),
       "mean_external_best_cosine":round(float(np.mean([
           r["external_top_cosine"] for r in rows])),6)}
    if known:
        x.update({
            "correct_top1":sum(bool(r["correct_external_top1"]) for r in rows),
            "mean_content_margin":round(float(np.mean([
                r["external_correct_target_margin"] for r in rows])),6),
            "positive_content_margin":sum(
                r["external_correct_target_margin"]>0 for r in rows)
        })
    if rows[0]["model_native_cue_familiarity"] is not None:
        x["mean_model_native_familiarity"]=round(float(np.mean([
            r["model_native_cue_familiarity"] for r in rows])),6)
    return x


def validation_only_novelty_gate(model,validation_absent,calibration_trained):
    """Threshold uses 62 VALIDATION absent examples, no heldout test cases.

    90% empirical negative rejection guaranteed on the validation records.
    Familiar known train examples are for descriptive acceptance only and
    never determine threshold. Distinct from external codebook scores.
    """
    if not isinstance(model,ResidualAssociativeMemory):
        return None
    neg=np.asarray([
        model.cue_familiarity(binding_cues(m)[-1])
        for m in validation_absent],dtype=float)
    # Candidate threshold chosen so at most 6/62 validation absences exceed.
    threshold=float(np.sort(neg)[-7])
    train=np.asarray([model.cue_familiarity(binding_cues(m)[-1])
                       for m in calibration_trained],dtype=float)
    return {
        "threshold":threshold,
        "validation_absent_n":len(neg),
        "validation_absent_false_accepted":int(np.count_nonzero(neg>threshold)),
        "calibration_train_n":len(train),
        "calibration_train_accepted":int(np.count_nonzero(train>threshold)),
        "uses_external_event_codebook_in_gate":False,
    }


def run(graph,bc01_dir,*,real=False,group=32,weights_dir=None,
        semantic=None):
    (original_sha,original,meta,bc01,pos,ordered,positives,
     validation,absent,targets)=source_state(graph,bc01_dir,real)
    ids=[m.event_id for m in ordered]
    selected=[m for m in positives if unseen_literal_source_cue(m)]
    if len(selected)!=31:raise AssertionError("Original source fourth cues changed")
    original_mask,mask_evidence=actual_feature_support(
        graph,seed=SEED,cells_per_feature=group)
    template=DirectFlywireOverlay(graph,seed=SEED,cells_per_feature=group,rate=.7/3)
    null,null_meta=rewire_effective_edges(
        graph,template.pre_cells,template.post_cells,
        seed=REWIRE_SEED,swaps_per_edge=2)
    rewired_mask,rewired_mask_evidence=actual_feature_support(
        null,seed=SEED,cells_per_feature=group)
    if (mask_evidence["eligible_original_directed_synapse_slots"]!=
        null_meta["originally_eligible_edges"]
        or rewired_mask_evidence["eligible_original_directed_synapse_slots"]!=
        null_meta["rewired_eligible_edges"]
        or fingerprint(graph)!=original_sha):
        raise AssertionError("Modified original publisher graph or changed eligible mask accounting")

    semantic=semantic or FrozenMiniLMCues()
    if semantic.onnx_sha256!=ONNX_SHA:
        raise ValueError("Pinned previous original Pilot12 actual MiniLM source weights changed")
    all_text=[]
    for m in ordered:
        all_text.extend(binding_cues(m))
    for m in selected:all_text.append(unseen_literal_source_cue(m))
    for m in validation+absent:all_text.append(binding_cues(m)[-1])
    semantic.prewarm(all_text)
    models=make_models(graph,null,original_mask,rewired_mask,semantic=semantic,group=group)
    if tuple(models)!=ARMS:raise AssertionError("Predeclared model conditions changed")
    shift=np.roll(np.arange(len(ordered)),max(1,len(ordered)//3))
    stages=[]
    for i,cut in enumerate(CUTS):
        if i:
            for j in range(CUTS[i-1],cut):
                item=ordered[j];wrong=ordered[int(shift[j])]
                if wrong.event_id==item.event_id:
                    raise AssertionError("Negative control not deranged")
                for name,model in models.items():
                    content=bc01[pos[(wrong if name.endswith("wrong_content") else item).event_id]]
                    for cue in binding_cues(item):
                        model.imprint(cue,content)
        known=set(ids[:cut])
        groups={
            "earliest16":ordered[:16],
            "newest16":ordered[max(0,cut-16):cut],
            "original159_learned_familiar":[m for m in positives if m.event_id in known],
            "original31_fourth_never_trained":[m for m in selected if m.event_id in known],
            "absent71_episode":absent,
        }
        stage={"trained_events":cut,"cue_presentations_per_arm":cut*3,"arms":{}}
        for name,model in models.items():
            cases={}
            reports={}
            for g in GROUPS:
                rows=score_rows(model,groups[g],ids,targets,known,
                                fourth=g=="original31_fourth_never_trained")
                cases[g]=rows
                reports[g]=stats(rows)
            if (model.n_presentations if isinstance(model,ResidualAssociativeMemory)
                else model.imprints)!=cut*3:
                raise AssertionError("Source cue budgets are not paired")
            stage["arms"][name]={
                "cases_external_oracle_only":cases,
                "groups":reports,
                "trained_W_nonzero":(
                    model.nonzero_trained_weights
                    if isinstance(model,ResidualAssociativeMemory)
                    else model.modified_edges),
                "allocated_learned_numeric_scalars":(
                    model.allocated_numeric_scalars
                    if isinstance(model,ResidualAssociativeMemory)
                    else mask_evidence["eligible_original_directed_synapse_slots"]),
                "cue_presentations":cut*3,
            }
        stages.append(stage)
    calibration_known=[m for m in ordered if m.event_id not in
                       {p.event_id for p in positives}]
    if len(calibration_known)!=158 or len(validation)!=62:
        raise AssertionError("Training calibration sample is no longer original")
    terminal_gates={}
    for name,model in models.items():
        gate=validation_only_novelty_gate(model,validation,calibration_known)
        if gate is None:
            terminal_gates[name]=None;continue
        threshold=gate["threshold"]
        familiar=stages[-1]["arms"][name]["cases_external_oracle_only"]["original159_learned_familiar"]
        missing=stages[-1]["arms"][name]["cases_external_oracle_only"]["absent71_episode"]
        unseen=stages[-1]["arms"][name]["cases_external_oracle_only"]["original31_fourth_never_trained"]
        gate.update({
            "source_test_absent_false_accepted":sum(
                x["model_native_cue_familiarity"]>threshold for x in missing),
            "source_test_correct_and_accepted_familiar":sum(
                x["correct_external_top1"] and
                x["model_native_cue_familiarity"]>threshold for x in familiar),
            "source_test_correct_and_accepted_unseen_fourth":sum(
                x["correct_external_top1"] and
                x["model_native_cue_familiarity"]>threshold for x in unseen),
            "source_test_original_familiar_n":159,
            "source_test_original_absent_n":71,
            "source_test_original_unseen_fourth_n":31,
        })
        terminal_gates[name]=gate
    checkpoints={}
    if weights_dir:
        for name,model in models.items():
            dest=Path(weights_dir)/(name+".npz")
            if isinstance(model,ResidualAssociativeMemory):
                digest=model.save(
                    dest,source_graph_sha=(
                        null_meta["original_csr_array_sha256"] if "rewired" in name
                        else original_sha),
                    encoder_sha=(semantic.onnx_sha256
                                 if "BC01_dense" not in name else "BC01-original")
                )
                reread=ResidualAssociativeMemory.load(
                    dest,encoder=model.encoder,source_graph_sha=(
                        null_meta["original_csr_array_sha256"] if "rewired" in name
                        else original_sha),
                    encoder_sha=(semantic.onnx_sha256
                                 if "BC01_dense" not in name else "BC01-original")
                )
                if not np.array_equal(reread.W.astype(np.float32),model.W.astype(np.float32)):
                    raise AssertionError("Model source numeric checkpoints not replayed")
            else:
                digest=model.save(dest)
            checkpoints[name]={"sha256":digest,"path":str(dest)}
    if fingerprint(graph)!=original_sha:
        raise AssertionError("Source original full FlyWire biological arrays mutated")
    return {
        "study":"Mnemosyne Pilot16: frozen dense semantic cue learned residual content and source mask controls",
        "evidence":("original verified complete publisher FlyWire v783"
                    if real else "synthetic artificial connectivity, NOT fly biology"),
        "limitations":(
            "Actual published FlyWire graph is COLLAPSED into a 256x256 binary "
            "feature-mask prior for error-corrective regression. It is NOT a "
            "direct 15M-edge neural simulation. Covariance 65,536 extra numeric "
            "scalars are used for novelty/learning. MiniLM pretrained external "
            "language weights supply semantic features, not source episodes. "
            "Correct event labels and full original 317 CONTENT candidate "
            "rankings are EXTERNAL oracle-only, never available inside "
            "model.infer(cue). Source 159/31/71 were already examined in "
            "earlier pilots and remain EXPLORATORY. kc64 rotated sparse code "
            "is an analogy, not biologically reconstructed Kenyon neurons. "
            "No autonomous narrative, self-awareness or independent source "
            "memory confidence is demonstrated by correct oracle ID alone."
        ),
        "original_source":{
            "canonical_events":len(original),"canonical_episodes":len({m.episode_id for m in original}),
            "trained_source_events":len(ordered),"heldout_known_test":len(positives),
            "validation_absent":len(validation),"test_absent":len(absent),
            "eligible_never_trained_fourth_literal":len(selected),
            "BC01_cache_sha256":meta["artifact_sha256"],
            "original_biology_array_sha256":original_sha,
            "original_anatomy_unchanged":fingerprint(graph)==original_sha,
            "root_neurons":len(graph.root_ids),
            "aggregate_directed_pairs":len(graph.indices),
            "integer_anatomical_synapses":int(graph.synapse_counts.sum(dtype=np.int64)),
        },
        "preregistered":{
            "memory_load_stages":CUTS,"source_train_presentations_per_event":3,
            "fixed_external_source_content_candidates":317,
            "frozen_ONNX_model":MODEL,"frozen_ONNX_revision":REVISION,
            "frozen_ONNX_sha256":semantic.onnx_sha256,
            "frozen_MiniLM_projection_sha256":projection_sha256(),
            "output_is_original_BC01_32_signed_features":True,
            "orthogonal_sparse_input_separation_k":64,
            "RLS_inverse_covariance_ridge":1.,"SGD_residual_learning_rate":.4,
            "no_159_31_71_test_tuning":True,
            "input_novelty_validation_only_62_absences":True,
            "source_256x256_pair_mask_collapses_individual_anatomical_synapses":True,
            "note_extra_P_inverse_covariance_state_65536_scalars":True
        },
        "original_fly_feature_pair_mask":mask_evidence,
        "rewired_fly_feature_pair_mask":rewired_mask_evidence,
        "feature_pair_random_mask_n":int(mask_evidence["effective_learned_feature_pairs"]),
        "rewired_binary_degree_preserving_null":null_meta,
        "full_original_source_case_stages":stages,
        "validation_only_model_native_novelty_gates":terminal_gates,
        "learned_numerical_state_checkpoint_SHA256":checkpoints
    }

def main():
    p=argparse.ArgumentParser(description=__doc__)
    a=p.add_mutually_exclusive_group(required=True)
    a.add_argument("--topology",type=Path)
    a.add_argument("--synthetic-test",action="store_true")
    p.add_argument("--bc01-dir",type=Path,required=True)
    p.add_argument("--cells-per-feature",type=int)
    p.add_argument("--weights-dir",type=Path)
    p.add_argument("--output",type=Path,required=True)
    opt=p.parse_args()
    graph=(synthetic_aggregated_fixture(n=8192,degree=16,seed=15)
           if opt.synthetic_test else Topology.read(opt.topology))
    outcome=run(graph,opt.bc01_dir,real=not opt.synthetic_test,
                group=opt.cells_per_feature or (8 if opt.synthetic_test else 32),
                weights_dir=opt.weights_dir)
    opt.output.parent.mkdir(parents=True,exist_ok=True)
    opt.output.write_text(json.dumps(outcome,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    for stage in outcome["full_original_source_case_stages"]:
        print("STAGE",stage["trained_events"]," ".join(
            str(name)+":"+str(stage["arms"][name]["groups"]["earliest16"]["correct_top1"])+
            "/"+str(stage["arms"][name]["groups"]["earliest16"]["n"])
            for name in ARMS),flush=True)

if __name__=="__main__":
    main()
