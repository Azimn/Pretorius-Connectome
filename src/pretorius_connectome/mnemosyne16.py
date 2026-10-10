"""Pilot16 Mnemosyne: cue-only learned residual associative memory.

The actual original whole-FlyWire graph is collapsed into a SOURCE->CONTENT
feature-pair mask. This discards individual neuron/synapse morphology, and
must NOT be called direct fly-physiology simulation. Online error-corrective
learning uses a full dense pretrained semantic cue, NOT top-8 features.
The input inverse-covariance P is extra numerical state, accounted for.
Source event IDs, narrative targets and retrieval codebook never enter infer.
"""
from __future__ import annotations
from hashlib import sha256
from pathlib import Path
import numpy as np
from pretorius_connectome.direct_flywire_imprint import DirectFlywireOverlay,fingerprint
DIM=256

def normalized(v):
    a=np.asarray(v,dtype=np.float64)
    if a.shape!=(DIM,) or not np.isfinite(a).all():
        raise ValueError("Expected 256 finite feature coordinates")
    n=float(np.linalg.norm(a))
    return a/n if n>1e-12 else np.zeros(DIM,dtype=np.float64)

def actual_feature_support(topology,*,seed=31,cells_per_feature=32):
    """Each allowed scalar pair has >=1 actual pre->post original fly edge."""
    m=DirectFlywireOverlay(topology,seed=seed,cells_per_feature=cells_per_feature,rate=.7/3)
    mask=np.zeros((DIM,DIM),dtype=bool)
    edge_count=0
    for feature in range(DIM):
        for cell in m.pre_cells[feature]:
            start,stop=int(topology.indptr[cell]),int(topology.indptr[cell+1])
            outputs=m.post_feature[topology.indices[start:stop]]
            good=outputs>=0
            edge_count+=int(good.sum())
            mask[feature,outputs[good].astype(int)]=True
    return mask,{"eligible_original_directed_synapse_slots":edge_count,
                 "effective_learned_feature_pairs":int(mask.sum()),
                 "original_csr_sha256":fingerprint(topology),
                 "neuron_mapping_seed":seed,"cells_per_feature":cells_per_feature}

def random_feature_support(count,*,seed=73):
    if not 1<=count<=DIM*DIM:raise ValueError("Invalid feature-pair budget")
    rng=np.random.default_rng(seed)
    m=np.zeros(DIM*DIM,dtype=bool)
    m[rng.choice(DIM*DIM,count,replace=False)]=True
    return m.reshape(DIM,DIM)

def projection(seed=14016):
    q,_=np.linalg.qr(np.random.default_rng(seed).standard_normal((DIM,DIM)))
    return q

class ResidualAssociativeMemory:
    """Learn a cue->256D CONTENT vector without any source-event ID lookup.

    RLS uses input inverse covariance to correct current output prediction,
    suppressing redundant overlapping input updates. With a binary support
    mask the update is MASKED APPROXIMATE RLS, not exact constrained ridge.
    Delta uses error correction SGD; Hebb is additive control. All variants
    carry P for model-native cue novelty and disclose that extra 65,536-state
    matrix as part of memory budget. kc64 does a fixed orthogonal rotation
    and top-64 sparsification (a functional pattern-separation analogy).
    """
    def __init__(self,*,encoder,mask=None,rule="rls",representation="dense",
                 ridge=1.,rate=.4,separation_k=64,seed=14016):
        if (rule not in ("rls","delta","hebb") or representation not in ("dense","kc64")
            or not 0<ridge<10000 or not 0<rate<=2 or not 1<=separation_k<=DIM):
            raise ValueError("Invalid source-encoder learning configuration")
        self.encoder=encoder;self.rule=rule;self.representation=representation
        self.ridge=float(ridge);self.rate=float(rate)
        self.separation_k=separation_k;self.seed=seed
        self.mask=(np.ones((DIM,DIM),dtype=bool) if mask is None
                   else np.asarray(mask,dtype=bool).copy())
        if self.mask.shape!=(DIM,DIM) or not self.mask.any():
            raise ValueError("No valid source-feature support")
        self.W=np.zeros((DIM,DIM),dtype=np.float64)
        self.P=np.eye(DIM,dtype=np.float64)/self.ridge
        self.Q=projection(seed) if representation=="kc64" else None
        self.n_presentations=0;self.n_updates=0

    @property
    def allocated_numeric_scalars(self):
        return int(self.mask.sum())+self.P.size

    @property
    def nonzero_trained_weights(self):
        return int(np.count_nonzero(self.W))

    def cue_vector(self,cue):
        if not isinstance(cue,str) or not cue.strip():raise ValueError("Empty cue")
        x=normalized(self.encoder(cue))
        if self.Q is not None:
            rotated=x@self.Q
            ids=np.lexsort((np.arange(DIM),-np.abs(rotated)))[:self.separation_k]
            kept=np.zeros(DIM);kept[ids]=rotated[ids]
            x=normalized(kept)
        return x

    def imprint(self,cue,content):
        x=self.cue_vector(cue);y=normalized(content)
        if not x.any() or not y.any():raise ValueError("Missing source signal")
        prediction=x@self.W
        px=self.P@x
        k=px/(1.+float(x@px))
        if self.rule=="rls":
            change=np.outer(k,y-prediction)
        elif self.rule=="delta":
            change=self.rate*np.outer(x,y-prediction)
        else:
            change=self.rate*np.outer(x,y)
        self.W+=np.where(self.mask,change,0)
        self.P-=np.outer(k,px)
        self.P=(self.P+self.P.T)*.5
        self.n_presentations+=1
        self.n_updates+=int(np.count_nonzero(self.mask & (change!=0)))
        if not np.isfinite(self.W).all() or not np.isfinite(self.P).all():
            raise ArithmeticError("Nonfinite source learned numerical state")

    def infer(self,cue):
        return normalized(self.cue_vector(cue)@self.W).astype(np.float32)

    def cue_familiarity(self,cue):
        """Model-native input coverage, never external event-ID similarity."""
        x=self.cue_vector(cue)
        return float(np.clip(1.-self.ridge*float(x@self.P@x),0.,1.))

    def save(self,path,*,source_graph_sha=None,encoder_sha=None):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        source=source_graph_sha or {}
        np.savez_compressed(
            path,format_version=np.asarray([1],dtype=np.int64),
            W=self.W.astype(np.float32),P=self.P,mask=self.mask,
            Q=self.Q if self.Q is not None else np.empty((0,0)),
            settings=np.asarray([self.n_presentations,self.n_updates,
                                 self.separation_k,self.seed],dtype=np.int64),
            rule=np.asarray(self.rule),representation=np.asarray(self.representation),
            ridge=np.float64(self.ridge),rate=np.float64(self.rate),
            source_sha=np.asarray([source.get(x,"") for x in
                ("root_ids","indptr","indices","synapse_counts")],dtype="U64"),
            encoder_sha=np.asarray(encoder_sha or "",dtype="U64")
        )
        return sha256(path.read_bytes()).hexdigest()

    @classmethod
    def load(cls,path,*,encoder,source_graph_sha=None,encoder_sha=None):
        with np.load(path,allow_pickle=False) as z:
            if z["format_version"].tolist()!=[1]:raise ValueError("Unsupported state")
            expected=[(source_graph_sha or {}).get(x,"") for x in
                      ("root_ids","indptr","indices","synapse_counts")]
            if z["source_sha"].tolist()!=expected or str(z["encoder_sha"])!=(encoder_sha or ""):
                raise ValueError("Wrong source original biology or pretrained encoder")
            settings=z["settings"]
            m=cls(encoder=encoder,mask=z["mask"],rule=str(z["rule"]),
                  representation=str(z["representation"]),ridge=float(z["ridge"]),
                  rate=float(z["rate"]),separation_k=int(settings[2]),
                  seed=int(settings[3]))
            if (z["W"].shape!=(DIM,DIM) or z["P"].shape!=(DIM,DIM)
                or not np.isfinite(z["W"]).all() or not np.isfinite(z["P"]).all()
                or not np.array_equal(z["Q"],m.Q if m.Q is not None else np.empty((0,0)))):
                raise ValueError("Malformed learned source memory")
            m.W=z["W"].astype(np.float64);m.P=z["P"].copy()
            m.n_presentations=int(settings[0]);m.n_updates=int(settings[1])
            if np.any(m.W[~m.mask]!=0):raise ValueError("Learned forbidden edge pair")
            return m
