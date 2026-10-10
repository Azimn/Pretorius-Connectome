"""Pilot17 Chamber of Echoes: explicitly EXTERNAL, numeric episodic memory.

Unlike direct FlyWire imprinting or Mnemosyne's shared learned matrix, this
module ALLOCATES ONE numeric memory slot PER TRAINED SOURCE EVENT.
Each slot stores exactly three frozen input cue vectors (compressed to 64
coordinates apiece) and one original source CONTENT codeword (256 float32
coordinates). It stores no narrative text or event-ID label, but it IS a
retrieval-based external episodic database. This fact is never disguised.

Three source-memory routing laws: max-competitive key match, averaged-key
centroid (same stored keys and bytes) and top-3 weighted content mixture.
All use the same pretrained encoder, original source training and fixed
random projection. Cannot claim fly physiology, a neural-only memory,
autonomous narrative or independently authored semantic generalization.

The frozen 256x64 projection costs 16,384 extra fixed floats. 317 source
slots cost 317*(3*64+256)=142,016 trainable/stored numeric floats. Total
including fixed projection: 158,400 float32 = 633,600 bytes, compared to
the previous unmasked W(float32)+P(float64) 786,432 bytes. This is a
storage budget comparison, not matched compute/representation capacity.
"""
from __future__ import annotations
from hashlib import sha256
from pathlib import Path
import numpy as np
from pretorius_connectome.direct_flywire_imprint import select_features

DIM=256
KEY_DIM=64
CUES_PER_EPISODE=3
MAX_EPISODES=317
SEED=17031

def unit(v):
    a=np.asarray(v,dtype=np.float32)
    n=float(np.linalg.norm(a))
    return a/np.float32(n) if n>1e-12 else np.zeros_like(a)

def random_projection(seed=SEED):
    q,_=np.linalg.qr(np.random.default_rng(seed).standard_normal((DIM,KEY_DIM)))
    return q.astype(np.float32)

class EpisodicTraceMemory:
    """Trainable retrieval slots, NOT native fly numeric synapses.

    add_episode(cues, content) accepts only the THREE literal cue texts and
    the original 256D content code, no episode IDs. infer(cue) internally
    searches stored numeric keys/values and returns learned CONTENT.
    None of the evaluated 317 source labels is held in a model index.
    """

    def __init__(self,*,encoder,mode="competitive",
                 max_episodes=MAX_EPISODES,seed=SEED,temperature=.12):
        if mode not in ("competitive","centroid","soft_top3"):
            raise ValueError("Unknown predeclared routing control")
        if max_episodes<1 or max_episodes>MAX_EPISODES:
            raise ValueError("Invalid memory storage contract")
        if temperature<=0 or not np.isfinite(temperature):
            raise ValueError("Invalid fixed routing temperature")
        self.encoder=encoder
        self.mode=mode
        self.seed=seed
        self.temperature=float(temperature)
        self.projection=random_projection(seed)
        self.keys=np.zeros((max_episodes,CUES_PER_EPISODE,KEY_DIM),dtype=np.float32)
        self.contents=np.zeros((max_episodes,DIM),dtype=np.float32)
        self.allocated_episodes=0
        self.max_episodes=max_episodes
        self.last_route=None

    def _cue(self,cue):
        if not isinstance(cue,str) or not cue.strip():
            raise ValueError("Source cue required")
        encoded=np.asarray(self.encoder(cue),dtype=np.float32)
        if encoded.shape!=(DIM,) or not np.isfinite(encoded).all():
            raise ValueError("Frozen cue encoder must return finite 256D features")
        return unit(unit(encoded)@self.projection)

    def add_episode(self,cues,source_content):
        if len(cues)!=CUES_PER_EPISODE:
            raise ValueError("Exactly three original literal source cues required")
        if self.allocated_episodes==self.max_episodes:
            raise OverflowError("Episodic storage capacity full")
        y=np.asarray(source_content,dtype=np.float32)
        if y.shape!=(DIM,) or not np.isfinite(y).all():
            raise ValueError("Source content is original finite 256D BC01 vector")
        y=select_features(y,top_k=32)
        i=self.allocated_episodes
        self.keys[i]=np.stack([self._cue(c) for c in cues])
        self.contents[i]=y
        self.allocated_episodes+=1

    def _scores(self,cue):
        if not self.allocated_episodes:
            return np.empty((0,),dtype=np.float32)
        x=self._cue(cue)
        sims=self.keys[:self.allocated_episodes]@x
        if self.mode=="centroid":
            centroids=np.mean(self.keys[:self.allocated_episodes],axis=1)
            centroids=centroids/np.maximum(
                np.linalg.norm(centroids,axis=1,keepdims=True),1e-12)
            return centroids@x
        # Best of all three original authored surface forms, per episode.
        return np.max(sims,axis=1)

    def native_novelty_score(self,cue):
        scores=self._scores(cue)
        return float(np.max(scores)) if len(scores) else -1.

    def infer(self,cue):
        scores=self._scores(cue)
        if not len(scores):
            return np.zeros(DIM,dtype=np.float32)
        best=int(np.argmax(scores))
        # Retrieval indices here are numerical SLOT positions, not external
        # source-event IDs. The label-free model does know its stored values.
        if self.mode!="soft_top3":
            return self.contents[best].copy()
        k=min(3,len(scores))
        best_ids=np.argsort(-scores,kind="stable")[:k]
        logits=(scores[best_ids]-np.max(scores[best_ids]))/self.temperature
        weights=np.exp(logits)
        weights/=weights.sum()
        return unit(weights@self.contents[best_ids])

    @property
    def numeric_storage_bytes(self):
        return int(self.keys.nbytes+self.contents.nbytes+
                   self.projection.nbytes)

    @property
    def active_storage_bytes(self):
        return int(self.allocated_episodes*(3*KEY_DIM+DIM)*4+
                   self.projection.nbytes)

    def save(self,path,*,encoder_sha):
        path=Path(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(
            path,format_version=np.asarray([1],dtype=np.int64),
            keys=self.keys[:self.allocated_episodes],
            contents=self.contents[:self.allocated_episodes],
            projection=self.projection,
            settings=np.asarray([self.seed,self.max_episodes,
                                 self.allocated_episodes],dtype=np.int64),
            mode=np.asarray(self.mode),temperature=np.asarray(self.temperature),
            encoder_sha=np.asarray(encoder_sha,dtype="U64"))
        return sha256(path.read_bytes()).hexdigest()

    @classmethod
    def load(cls,path,*,encoder,encoder_sha):
        with np.load(path,allow_pickle=False) as z:
            if z["format_version"].tolist()!=[1] or str(z["encoder_sha"])!=encoder_sha:
                raise ValueError("Wrong source encoder or trace checkpoint version")
            seed,capacity,n=map(int,z["settings"])
            obj=cls(encoder=encoder,mode=str(z["mode"]),seed=seed,
                    max_episodes=capacity,temperature=float(z["temperature"]))
            if (not 0<=n<=capacity or z["keys"].shape!=(n,3,KEY_DIM)
                or z["contents"].shape!=(n,DIM)
                or not np.array_equal(obj.projection,z["projection"])
                or not np.isfinite(z["keys"]).all()
                or not np.isfinite(z["contents"]).all()):
                raise ValueError("Malformed source-only episodic numeric traces")
            obj.keys[:n]=z["keys"]
            obj.contents[:n]=z["contents"]
            obj.allocated_episodes=n
        return obj
