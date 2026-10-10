"""The Reliquary (Pilot19): provenance-bound explicit EXTERNAL event retrieval.

Six predeclared arms vary ORIGINAL available evidence. An exact third anchor
can be correctly retrieved by an all-sidecar index only because the literal
candidate is INDEXED. That is not unseen-cue generalization.

Event graph paths are literal: cue surface -> indexed candidate detail ->
source event -> another original event detail/participant/location.
The graph is intentionally comparable with a simple exact inverted index.
No source event content outside the 317 trained events enters these stores.
"""
from __future__ import annotations
from collections import Counter,defaultdict
import hashlib
import json
import math
import re

ARMS=("train_two_anchors_only","narrative_bm25",
      "all_anchors_flat_bm25","all_anchors_exact_index",
      "reliquary_event_graph","wrong_owner_graph")
WRONG_OWNER_SHIFT=79

def normalize(s):
    return " ".join(re.findall(r"[a-z0-9]+",s.casefold()))

def words(s):
    return normalize(s).split()

def _best_score(results):
    if not results:return {"event_id":None,"confidence":0.,"ambiguous":False,"evidence":None}
    candidates=sorted(results,key=lambda x:(-x[1],x[0]))
    val=candidates[0][1]
    if val<=0:return {"event_id":None,"confidence":0.,"ambiguous":False,"evidence":None}
    tied=[x for x in candidates if abs(x[1]-val)<1e-9]
    if len(tied)>1:
        return {"event_id":None,"confidence":round(float(val),8),
                "ambiguous":True,"evidence":None}
    return {"event_id":candidates[0][0],"confidence":round(float(val),8),
            "ambiguous":False,"evidence":candidates[0][2]}

class Reliquary:
    """Explicit source-event ID index, NOT fly synapses or neural cue inference."""

    def __init__(self,records,annotations,*,arm):
        if arm not in ARMS:raise ValueError("Unregistered source retrieval arm")
        if len(records)<2 or len(records)!=len(annotations):
            raise ValueError("Only same-length original event/annotation train corpus")
        self.arm=arm
        self.events={}
        self.annotations={}
        self.documents={}
        self.exact_index=defaultdict(set)
        self.candidate_paths=defaultdict(list)
        self.graph={}
        self.tf={}
        self.df=Counter()
        self.doc_lengths={}
        self.mean_length=1.
        self.source_checksum=None
        ids=[r["event_id"] for r in records]
        if len(ids)!=len(set(ids)) or set(ids)!={a["event_id"] for a in annotations}:
            raise ValueError("Original annotation/source event identity mismatch")
        self.ids=list(ids)
        self.indexed_source_event_count=len(ids)
        annotation_by_id={a["event_id"]:a for a in annotations}
        for r in records:
            id=r["event_id"]
            a=annotation_by_id[id]
            if a.get("annotation_status")!="unreviewed_candidate":
                raise ValueError("Candidate source annotation provenance altered")
            if not isinstance(r.get("memory_text"),str) or not r["memory_text"]:
                raise ValueError("Original source event narrative missing")
            self.events[id]={"memory_text":r["memory_text"],
                             "participants":list(r.get("participants",[])),
                             "locations":list(r.get("locations",[])),
                             "episode_id":r["episode_id"]}
            self.annotations[id]={"cue_surface_forms":list(a["cue_surface_forms"]),
                                  "annotation_status":a["annotation_status"]}
            self.graph[id]={"node_type":"source_episode",
                            "participants":list(r.get("participants",[])),
                            "locations":list(r.get("locations",[])),
                            "candidate_details":[],
                            "provenance_field":"source_event"}
        for index,id in enumerate(ids):
            cues=self.annotations[id]["cue_surface_forms"]
            if not cues or any(not normalize(c) for c in cues):
                raise ValueError("Original source candidates empty")
            if arm=="train_two_anchors_only":
                chosen=(cues[0],cues[len(cues)//2])
                self._add_exact(id,chosen,source_owner=id)
            elif arm=="narrative_bm25":
                self.documents[id]=self.events[id]["memory_text"]
            elif arm=="all_anchors_flat_bm25":
                self.documents[id]=" ".join(cues)
            elif arm in ("all_anchors_exact_index","reliquary_event_graph"):
                self._add_exact(id,cues,source_owner=id)
            else:
                # Same 317 SOURCE DETAIL candidates but assigned to WRONG
                # source event. Does not alter original input annotations.
                assigned=ids[(index+WRONG_OWNER_SHIFT)%len(ids)]
                self._add_exact(assigned,cues,source_owner=id)
        if arm in ("narrative_bm25","all_anchors_flat_bm25"):
            for id,doc in self.documents.items():
                terms=words(doc)
                self.tf[id]=Counter(terms)
                self.doc_lengths[id]=len(terms)
                self.df.update(self.tf[id].keys())
            self.mean_length=sum(self.doc_lengths.values())/len(self.doc_lengths)
        self.index_fingerprint=hashlib.sha256(json.dumps({
            "arm":arm,"ids":ids,
            "documents":self.documents,
            "cue_edges":sorted((term,sorted(owner)) for term,owner
                               in self.exact_index.items()),
            "source_owners":sorted((term,entry["source_event_id"],
                                    entry["retrieval_event_id"])
                                   for term,entries in self.candidate_paths.items()
                                   for entry in entries)
        },sort_keys=True).encode("utf-8")).hexdigest()

    def _add_exact(self,retrieval_owner,cues,*,source_owner):
        for cue in cues:
            surface=normalize(cue)
            self.exact_index[surface].add(retrieval_owner)
            detail={"node_type":"original_candidate_detail",
                    "source_event_id":source_owner,
                    "retrieval_event_id":retrieval_owner,
                    "source_annotation_status":"unreviewed_candidate",
                    "source_field":"cue_surface_forms",
                    "literal_cue":cue}
            self.candidate_paths[surface].append(detail)
            self.graph[retrieval_owner]["candidate_details"].append(detail)

    def _bm25(self,query,id):
        query_counts=Counter(words(query))
        score=0.
        document_size=self.doc_lengths[id]
        for token,qf in query_counts.items():
            tf=self.tf[id].get(token,0)
            if not tf:continue
            n=self.indexed_source_event_count
            df=self.df.get(token,0)
            idf=math.log(1+(n-df+.5)/(df+.5))
            denom=tf+1.2*(.25+.75*document_size/self.mean_length)
            score+=qf*idf*tf*2.2/denom
        return score

    def infer(self,query):
        """Returns named source ID openly: this is indexed retrieval."""
        key=normalize(query)
        if not key:raise ValueError("Blank original source detail query")
        if self.arm in ("narrative_bm25","all_anchors_flat_bm25"):
            source="memory_text" if self.arm=="narrative_bm25" else "cue_surface_forms"
            return _best_score([
                (id,self._bm25(query,id),
                 {"retrieval_event_id":id,"source_event_id":id,
                  "source_field":source,"annotation_status":(
                      None if source=="memory_text" else "unreviewed_candidate"),
                  "matching_query_terms":sorted(set(words(query)) & set(self.tf[id]))})
                for id in self.ids])
        if key not in self.exact_index:
            return _best_score([])
        candidates=[]
        for id in self.exact_index[key]:
            edges=[e for e in self.candidate_paths[key]
                   if e["retrieval_event_id"]==id]
            evidence=edges[0].copy()
            if self.arm in ("reliquary_event_graph","wrong_owner_graph"):
                # Provenance-aware literal one-hop ASSOCIATION. A candidate
                # is attached to an explicit event, not semantically equated.
                other=[x["literal_cue"] for x in
                       self.graph[id]["candidate_details"]
                       if normalize(x["literal_cue"])!=key]
                evidence.update({
                    "graph_path_types":["original_candidate_detail",
                                        "retrieval_source_event",
                                        "other_event_candidate_detail"],
                    "associated_other_detail":other[0] if other else None,
                    "original_event_participants":self.graph[id]["participants"],
                    "original_event_locations":self.graph[id]["locations"],
                })
            candidates.append((id,1.,evidence))
        return _best_score(candidates)

    def indexed_exposure(self,query,truth_event_id):
        """Per-case truth provenance; never score success as unobserved cue."""
        key=normalize(query)
        sidecar=self.annotations[truth_event_id]["cue_surface_forms"]
        narrative=self.events[truth_event_id]["memory_text"]
        return {
            "cue_available_in_own_source_original_sidecar":
                key in [normalize(s) for s in sidecar],
            "cue_verbatim_in_own_original_narrative":
                key in normalize(narrative),
            "cue_in_own_RETRIEVAL_INDEX":(
                key in {normalize(x["literal_cue"]) for x in
                        self.graph[truth_event_id]["candidate_details"]}
                if self.arm not in ("narrative_bm25","all_anchors_flat_bm25")
                else (key in normalize(self.documents[truth_event_id]))
            ),
            "cue_index_mode":self.arm,
            "sidecar_annotation_status":
                self.annotations[truth_event_id]["annotation_status"],
        }

    def stored_provenance(self):
        return {"architecture":"EXTERNAL event-owned source retrieval",
                "source_event_count":self.indexed_source_event_count,
                "arm":self.arm,"index_sha256":self.index_fingerprint,
                "index_document_token_count":sum(self.doc_lengths.values()),
                "index_candidate_edges":sum(
                    len(x) for x in self.candidate_paths.values()),
                "graph_event_nodes":len(self.graph),
                "graph_participant_links":sum(
                    len(x["participants"]) for x in self.graph.values()),
                "graph_location_links":sum(
                    len(x["locations"]) for x in self.graph.values())}
