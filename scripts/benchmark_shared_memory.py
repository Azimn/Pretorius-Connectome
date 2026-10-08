#!/usr/bin/env python3
"""Measure cached-versus-uncached BC lexical preprocessing and FlyWire parity."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import resource
import sys
from time import perf_counter
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
sys.path.insert(0,str(ROOT))
from pretorius_connectome.shared_memory import read_l1
from pretorius_connectome.shared_features_bc01 import (
    encode_bc_sensory,build_bc01_cache,load_bc01_cache,
)
from pretorius_connectome.imprinting import load_v12
from pretorius_connectome.associative import AssociativeMemory,Topology
from pretorius_connectome.pilot02 import episode_split

EVENTS=ROOT/"memories/current/Pretorius_v12_450_Events_Complete.jsonl"
SIDECARS=ROOT/"memories/annotations/v12_450_sidecars.jsonl"

def measure(fun):
    t=perf_counter()
    value=fun()
    return value,round(perf_counter()-t,6)

def run(folder:Path):
    archive=folder/"pretorius_l1_v1.jsonl.gz"
    manifest=folder/"manifest.json"
    original,t_original=measure(lambda:load_v12(EVENTS,SIDECARS))
    shared,t_l1=measure(lambda:load_v12(archive,SIDECARS,shared_manifest=manifest))
    if original!=shared:
        raise ValueError("L1 original/source equality failed")
    raw,t_hash=measure(lambda:np.stack([
        encode_bc_sensory(m.memory_text) for m in original
    ]))
    (meta,cached),t_cache=measure(lambda:load_bc01_cache(folder))
    if not np.array_equal(raw,cached):
        raise ValueError("BC01 lexical sensory cached parity failure")
    topo=Topology.synthetic(n=128,degree=4,seed=8)
    counts=topo.synapse_counts.copy()
    train,validation,test=episode_split(original,31)
    train_l1,_,_=episode_split(shared,31)
    old,t_fit0=measure(lambda:AssociativeMemory(train,topo))
    new,t_fit1=measure(lambda:AssociativeMemory(train_l1,topo))
    for q in ("millstream map and turned glove","cathedral laboratory"):
        for mode in ("lexical","graph","hybrid"):
            if not np.array_equal(old.score(q,mode),new.score(q,mode)):
                raise ValueError("shared L1 changed retrieved score")
    np.testing.assert_array_equal(counts,topo.synapse_counts)
    return {
        "status":"measured CPU baseline; no semantic model",
        "source_blob":meta["source_git_blob"],
        "record_count":len(original),
        "train_events":len(train),
        "validation_events":len(validation),
        "test_events":len(test),
        "cache_bytes":sum(p.stat().st_size for p in folder.iterdir() if p.is_file()),
        "time_seconds":{
            "read_original":t_original,
            "read_shared_compressed_L1":t_l1,
            "uncached_BC01_all_450":t_hash,
            "read_and_verify_BC01_L2_all_450":t_cache,
            "TFIDF_graph_original_train_only":t_fit0,
            "TFIDF_graph_shared_L1_train_only":t_fit1,
        },
        "peak_rss_kb":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "exact_L1_retrieval_parity":True,
        "exact_BC01_feature_parity":True,
        "biological_topology_mutation":False,
        "topology":"synthetic_test_only",
        "limitation":"Read and VERIFY includes all 450 hashes; time is not an unconditional speedup.",
    }

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle",type=Path,
                   default=ROOT/"artifacts/shared_memory/v1")
    p.add_argument("--output",type=Path)
    args=p.parse_args()
    result=run(args.bundle)
    text=json.dumps(result,indent=2,sort_keys=True)+"\n"
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(text,encoding="utf-8")
    print(text)

if __name__=="__main__":
    main()
