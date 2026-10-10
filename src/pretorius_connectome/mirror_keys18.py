"""Pilot18 Mirror Keys: train-only pair-discriminant semantic memory addresses.

Unlike Pilot17's three *stored* literal source cues, Pilot18 stores only
TWO original-authored views of each source memory, and holds the third
literal cue entirely out of metric fitting and numeric key storage.
A train-only supervised contrastive scatter matrix may learn a better
64D cue-address projection. It is NOT end-to-end neural metric learning:
the specified generalized discriminant projection is an analytically
fitted linear matrix using positive within-record pairs and between-record
negatives (between-class scatter). No fourth source cues or absent
validation/test prompts enter projection fitting.

Every model allocates 317 * (3 * 64 + 256) float32 episodic slots and a
256x64 float32 projection, exactly the SAME 633600 array bytes as Pilot17.
Third 64D key slots are identically ZERO and MASKED in every condition,
rather than accidentally scoring as fabricated 0 cosine keys.

This is explicitly EXTERNAL numeric episodic key/value retrieval, NOT
FlyWire synaptic computation, event-ID lookup, source story reconstruction
or human semantic paraphrase evidence.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import numpy as np
from pretorius_connectome.echo_chamber17 import (
    DIM, KEY_DIM, MAX_EPISODES, SEED, unit, random_projection
)
from pretorius_connectome.direct_flywire_imprint import select_features

ALGORITHMS=("random","pair_discriminant","mispaired_discriminant","pca")
PROJECTION_REGULARIZATION=0.002
PAIR_PENALTY=2.0
WRONG_PAIR_SEED=18031


def _canonical_eigh(matrix,k=KEY_DIM):
    eigenvalues,vecs=np.linalg.eigh((matrix+matrix.T)*0.5)
    order=np.argsort(-eigenvalues,kind="stable")[:k]
    q=vecs[:,order]
    # Eliminate arbitrary eigenvector sign flips on different LAPACK builds.
    for i in range(q.shape[1]):
        lead=int(np.argmax(np.abs(q[:,i])))
        if q[lead,i]<0:q[:,i]*=-1
    return q.astype(np.float32)


def fit_train_only_projection(paired_vectors, *, algorithm,seed=SEED):
    """Paired embeddings shape (events,2,256); no episode IDs or test cues.

    Between-event negatives enter via covariance of class centroids.
    Within-event positives enter via covariance of two cue differences.
    This is discriminant scatter projection rather than triplet-loss SGD.
    """
    if algorithm not in ALGORITHMS:
        raise ValueError("Unregistered metric algorithm")
    v=np.asarray(paired_vectors,dtype=np.float64)
    if (v.ndim!=3 or v.shape[1:]!=(2,DIM) or len(v)<3
        or not np.isfinite(v).all()):
        raise ValueError("Projection requires >=3 finite TWO-CUE train groups")
    v=v/np.maximum(np.linalg.norm(v,axis=2,keepdims=True),1e-12)
    if algorithm=="random":return random_projection(seed)
    if algorithm=="pca":
        x=v.reshape((-1,DIM))
        x=x-x.mean(axis=0,keepdims=True)
        return _canonical_eigh(x.T@x/len(x))
    if algorithm=="mispaired_discriminant":
        shuffle=np.random.default_rng(WRONG_PAIR_SEED).permutation(len(v))
        # Wrong pseudo-event pairing is fixed from TRAIN views alone, never
        # the source event contents or withheld third cue.
        v=np.stack([v[:,0],v[shuffle,1]],axis=1)
    midpoint=v.mean(axis=1)
    centered=midpoint-midpoint.mean(axis=0,keepdims=True)
    between=centered.T@centered/len(v)
    differences=(v[:,0]-v[:,1])/np.sqrt(2.0)
    within=differences.T@differences/len(v)
    # Orthogonal fixed-rank supervised contrastive scatter metric:
    # maximize between-memory variation, penalize within-memory mismatch.
    matrix=between-PAIR_PENALTY*within
    matrix+=np.eye(DIM)*PROJECTION_REGULARIZATION
    return _canonical_eigh(matrix)


class TwoViewEpisodicMemory:
    """A label-free externally allocated 2-cue numeric trace with fitted map."""

    def __init__(self,*,encoder,projection,max_episodes=MAX_EPISODES,
                 wrong_content=False):
        p=np.asarray(projection,dtype=np.float32)
        if p.shape!=(DIM,KEY_DIM) or not np.isfinite(p).all():
            raise ValueError("Expected a trained/pinned 256x64 address map")
        if not 1<=max_episodes<=MAX_EPISODES:
            raise ValueError("Invalid frozen source episode capacity")
        self.encoder=encoder
        self.projection=p.copy()
        self.keys=np.zeros((max_episodes,3,KEY_DIM),dtype=np.float32)
        self.contents=np.zeros((max_episodes,DIM),dtype=np.float32)
        self.max_episodes=max_episodes
        self.allocated_episodes=0
        self.wrong_content=bool(wrong_content)

    def embed(self,cue):
        if not isinstance(cue,str) or not cue.strip():
            raise ValueError("No blank source cue")
        raw=np.asarray(self.encoder(cue),dtype=np.float32)
        if raw.shape!=(DIM,) or not np.isfinite(raw).all():
            raise ValueError("Encoder must output valid source 256D cue")
        return unit(unit(raw)@self.projection)

    def add_episode(self,train_cues,content):
        if len(train_cues)!=2 or not all(
            isinstance(c,str) and c.strip() for c in train_cues):
            raise ValueError("Only the first two TRAIN source literal cues allowed")
        if self.allocated_episodes==self.max_episodes:
            raise OverflowError("No original source-episode capacity remaining")
        val=np.asarray(content,dtype=np.float32)
        if val.shape!=(DIM,) or not np.isfinite(val).all():
            raise ValueError("Invalid source 256D target content")
        i=self.allocated_episodes
        self.keys[i,:2]=np.stack([self.embed(c) for c in train_cues])
        self.contents[i]=select_features(val,top_k=32)
        self.allocated_episodes+=1

    def route_scores(self,cue):
        if not self.allocated_episodes:return np.empty((0,),dtype=np.float32)
        # The untrained third slot is excluded, not a synthetic zero-cue key.
        return np.max(self.keys[:self.allocated_episodes,:2]@self.embed(cue),axis=1)

    def native_novelty_score(self,cue):
        a=self.route_scores(cue)
        return float(a.max()) if len(a) else -1.

    def infer(self,cue):
        scores=self.route_scores(cue)
        if not len(scores):return np.zeros(DIM,dtype=np.float32)
        return self.contents[int(np.argmax(scores))].copy()

    @property
    def numeric_storage_bytes(self):
        return self.keys.nbytes+self.contents.nbytes+self.projection.nbytes

    @property
    def active_storage_bytes(self):
        return (self.allocated_episodes*(3*KEY_DIM+DIM)*4
                + self.projection.nbytes)

    def save(self,path,*,encoder_sha,algorithm,source_sha):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(
            path,format_version=np.asarray([1],dtype=np.int64),
            keys=self.keys[:self.allocated_episodes],
            contents=self.contents[:self.allocated_episodes],
            projection=self.projection,
            settings=np.asarray([self.max_episodes,self.allocated_episodes],dtype=np.int64),
            algorithm=np.asarray(algorithm),
            encoder_sha=np.asarray(encoder_sha,dtype="U64"),
            source_sha=np.asarray(source_sha,dtype="U64"),
        )
        return sha256(path.read_bytes()).hexdigest()

    @classmethod
    def load(cls,path,*,encoder,encoder_sha,source_sha):
        with np.load(path,allow_pickle=False) as z:
            if (z["format_version"].tolist()!=[1]
                or str(z["encoder_sha"])!=encoder_sha
                or str(z["source_sha"])!=source_sha):
                raise ValueError("Frozen source/encoder/checkpoint mismatch")
            capacity,n=map(int,z["settings"])
            obj=cls(encoder=encoder,projection=z["projection"],max_episodes=capacity)
            if (not 0<=n<=capacity or z["keys"].shape!=(n,3,KEY_DIM)
                or z["contents"].shape!=(n,DIM)
                or not np.all(z["keys"][:,2,:]==0)
                or not np.isfinite(z["keys"]).all()
                or not np.isfinite(z["contents"]).all()):
                raise ValueError("Third heldout cue entered source memory or corrupted state")
            obj.keys[:n]=z["keys"]
            obj.contents[:n]=z["contents"]
            obj.allocated_episodes=n
            return obj
