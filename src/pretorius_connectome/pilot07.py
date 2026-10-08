"""Pilot 07A: optional local CPU NLI over BM25-retrieved real narratives.

Uses frozen pre-trained NLI, not a hand-written negation proxy. The old
assistant-authored Pilot04 challenge is used ONLY for posthoc diagnostics.
Model inference is optional for basic unit tests but required for an actual
Pilot07A *measured* NLI result. No silent fallback is permitted.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
from typing import Protocol

import numpy as np

from pretorius_connectome.imprinting import Memory, order_by_episode
from pretorius_connectome.pilot02 import calibrate, episode_split
from pretorius_connectome.pilot04 import ChallengeCase
from pretorius_connectome.pilot05 import NarrativeModels, _round, _summary


NLI_MODEL_ID = "cross-encoder/nli-MiniLM2-L6-H768"
NLI_MODEL_REVISION = "c847a3c0e1cad93a5343183ef183f3044e3fc7c2"
NLI_LABELS = ("contradiction", "entailment", "neutral")
ENTAIL_MIN = 0.60
ENTAIL_MARGIN = 0.12
CONTRADICTION_MIN = 0.60
TOP_K = 3
MAX_TOKENS = 384


class NLIScorer(Protocol):
    def score_pairs(self, premises: list[str], hypotheses: list[str]
                    ) -> np.ndarray: ...


class LocalMiniLMNLI:
    """Load a specified model revision on CPU, with an audited label mapping."""
    def __init__(self, model_id: str = NLI_MODEL_ID,
                 revision: str = NLI_MODEL_REVISION,
                 max_tokens: int = MAX_TOKENS, threads: int = 2):
        if model_id != NLI_MODEL_ID or revision != NLI_MODEL_REVISION:
            raise ValueError("pilot checkpoint identity is frozen")
        if max_tokens < 64 or max_tokens > 512:
            raise ValueError("token budget must stay within 64..512")
        try:
            import torch
            import transformers
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
        except ImportError as exc:
            raise RuntimeError(
                "Real NLI requires 'torch', 'transformers' and a downloaded "
                "pinned model; neither a heuristic nor a stub is substituted."
            ) from exc
        torch.set_num_threads(threads)
        self.torch = torch
        self.model_id = model_id
        self.revision = revision
        self.max_tokens = max_tokens
        self.versions = {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "numpy": np.__version__,
        }
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=revision, trust_remote_code=False,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id, revision=revision, use_safetensors=True,
            trust_remote_code=False,
        )
        id2label = {int(i): str(label).lower()
                    for i, label in self.model.config.id2label.items()}
        self.ordered_labels = [id2label[i] for i in range(len(id2label))]
        if set(self.ordered_labels) != set(NLI_LABELS) or len(id2label) != 3:
            raise ValueError(f"checkpoint NLI labels mismatch: {id2label}")
        self.model.to("cpu")
        self.model.eval()

    def score_pairs(self, premises: list[str], hypotheses: list[str],
                    batch_size: int = 8) -> np.ndarray:
        if len(premises) != len(hypotheses):
            raise ValueError("NLI premise/hypothesis pairing length mismatch")
        if not premises:
            return np.empty((0, 3), dtype=np.float32)
        output = []
        for start in range(0, len(premises), batch_size):
            a = premises[start:start + batch_size]
            b = hypotheses[start:start + batch_size]
            data = self.tokenizer(
                a, b, padding=True, truncation="only_first",
                max_length=self.max_tokens, return_tensors="pt",
            )
            with self.torch.inference_mode():
                logits = self.model(**data).logits
                probabilities = self.torch.softmax(logits, dim=-1)
            # Canonical order is fixed, independent of checkpoint column order.
            indices = [self.ordered_labels.index(label) for label in NLI_LABELS]
            output.append(probabilities[:, indices].cpu().numpy())
        return np.concatenate(output).astype(np.float32)


@dataclass(frozen=True)
class RetrievalCandidate:
    event_id: str
    story: str
    rank: int
    score: float


def retrieve_top_k(model: NarrativeModels, query: str,
                   k: int = TOP_K) -> list[RetrievalCandidate]:
    if k <= 0:
        raise ValueError("top_k must be positive")
    scores = model.rank([query], "bm25_narrative")[0]
    # Stable ranking also makes ties reproducible across environments.
    indexes = np.argsort(-scores, kind="stable")[:min(k, len(scores))]
    return [
        RetrievalCandidate(
            event_id=model.identifiers[int(i)],
            story=model.train[int(i)].memory_text,
            rank=position + 1, score=float(scores[int(i)]),
        )
        for position, i in enumerate(indexes)
    ]


def triage_pairs(
    candidates: list[list[RetrievalCandidate]],
    claims: list[str], scorer: NLIScorer,
) -> list[list[dict]]:
    if len(candidates) != len(claims):
        raise ValueError("candidate and claim lengths differ")
    flat = [(case_index, candidate) for case_index, group in enumerate(candidates)
            for candidate in group]
    if not flat:
        raise ValueError("empty candidate pool")
    # Both the premise and hypothesis contain only original train narratives
    # and query strings respectively. No expected event ID, case kind, source
    # anchor or truth label is passed to the NLI model.
    probs = scorer.score_pairs(
        [candidate.story for _, candidate in flat],
        [claims[case_index] for case_index, _ in flat],
    )
    if probs.shape != (len(flat), 3):
        raise ValueError(f"unexpected NLI probability shape: {probs.shape}")
    if (np.any(~np.isfinite(probs)) or np.any(probs < 0)
        or np.any(probs > 1) or not
        np.allclose(probs.sum(axis=1), 1.0, atol=1e-3)):
        raise ValueError("invalid NLI probability output")
    results = [[] for _ in candidates]
    for (case_index, candidate), triple in zip(flat, probs):
        results[case_index].append({
            "event_id": candidate.event_id,
            "rank": candidate.rank,
            "bm25_score": _round(candidate.score),
            "contradiction": _round(triple[0]),
            "entailment": _round(triple[1]),
            "neutral": _round(triple[2]),
        })
    return results


def _policy(rows: list[dict], rerank: bool) -> tuple[str | None, bool, str]:
    """Choose only justified entailment; contradiction is advisory not proof."""
    candidates = rows if rerank else rows[:1]
    ranked = sorted(candidates, key=lambda r: (-r["entailment"], r["rank"]))
    winner = ranked[0]
    ent, contra, neutral = (
        winner["entailment"], winner["contradiction"], winner["neutral"]
    )
    if (ent >= ENTAIL_MIN and
        ent > contra + ENTAIL_MARGIN and
        ent > neutral + ENTAIL_MARGIN):
        return winner["event_id"], True, "model_entailment"
    if (contra >= CONTRADICTION_MIN and
        contra > ent + ENTAIL_MARGIN):
        return None, False, "model_possible_contradiction"
    return None, False, "insufficient_evidence"


def evaluate_seed(memories: list[Memory], decisions: dict[str, str],
                  cases: list[ChallengeCase], seed: int,
                  scorer: NLIScorer, top_k: int = TOP_K,
                  cue_units: int = 512) -> dict:
    """Previously seen challenge diagnostic, NLI threshold frozen a priori."""
    train, validation, test = episode_split(memories, seed)
    train = order_by_episode(train, seed=20261008)
    model = NarrativeModels(train, seed, cue_units=cue_units,
                            memory_units=256, density=0.55)
    train_ids = {m.event_id for m in train}
    absent_ids = {m.event_id for m in test}
    groups = {
        "positive": [c for c in cases
                     if c.event_id in train_ids and c.kind == "paraphrase"],
        "contradiction": [c for c in cases
                          if c.event_id in train_ids and c.kind == "contradiction"],
        "absent": [c for c in cases
                   if c.event_id in absent_ids and c.kind == "paraphrase"],
    }
    if min(len(x) for x in groups.values()) < 2:
        raise ValueError("insufficient episode-disjoint challenge coverage")
    # BM25 threshold is calibrated only on original decision fields, exactly
    # as Pilot05A; NLI decision cutoffs above are predeclared and NOT tuned.
    pos_cal = model.score_batch([decisions[m.event_id] for m in train],
                                "bm25_narrative")
    neg_cal = model.score_batch([decisions[m.event_id] for m in validation],
                                "bm25_narrative")
    bm25_threshold = calibrate(
        [{"known": True, "top_cosine": r["score"]} for r in pos_cal],
        [{"known": False, "top_cosine": r["score"]} for r in neg_cal],
    )

    evaluation = []
    for group, records in groups.items():
        for record in records:
            evaluation.append((group, record))
    raw_candidates = [retrieve_top_k(model, record.query, top_k)
                      for _, record in evaluation]
    all_scores = triage_pairs(raw_candidates,
                              [record.query for _, record in evaluation], scorer)
    policies = ("bm25_calibrated", "nli_top1", "nli_rerank_top3", "reject_all")
    all_rows: dict[str, dict[str, list[dict]]] = {
        policy: {group: [] for group in groups} for policy in policies
    }
    for case_index, ((group, record), choices) in enumerate(zip(evaluation, all_scores)):
        first = choices[0]
        candidates_by_id = {c.event_id: c for c in raw_candidates[case_index]}
        for policy in policies:
            if policy == "bm25_calibrated":
                prediction = first["event_id"]
                accepted = bool(first["bm25_score"] > bm25_threshold)
                verdict = "retrieved_without_verification" if accepted else "insufficient_evidence"
            elif policy == "reject_all":
                prediction, accepted, verdict = None, False, "insufficient_evidence"
            else:
                prediction, accepted, verdict = _policy(
                    choices, rerank=policy == "nli_rerank_top3"
                )
            all_rows[policy][group].append({
                "case_id": record.case_id,
                "target": record.event_id,  # evaluator metadata only
                "expected_kind": group,     # evaluator metadata only
                "predicted": prediction,
                "accepted": accepted,
                "verdict": verdict,
                "retrieval_top1_event": first["event_id"],
                "candidate_probs": choices,
                "source_excerpt": (
                    candidates_by_id[prediction].story[:400]
                    if accepted and prediction in candidates_by_id else None
                ),
            })
    summaries = {}
    for policy in policies:
        sets = all_rows[policy]
        summaries[policy] = {
            "positive": _summary(sets["positive"], "positive"),
            "contradiction": _summary(sets["contradiction"], "contradiction"),
            "absent": _summary(sets["absent"], "absent"),
            "possible_contradictions": sum(
                row["verdict"] == "model_possible_contradiction"
                for row in sets["contradiction"]
            ),
        }
    return {
        "seed": seed, "train_events": len(train),
        "validation_unknown_events": len(validation),
        "test_unknown_events": len(test),
        "nli_candidate_count": sum(len(x) for x in raw_candidates),
        "bm25_threshold": _round(bm25_threshold),
        "thresholds": {
            "entailment_min": ENTAIL_MIN, "entailment_margin": ENTAIL_MARGIN,
            "contradiction_min": CONTRADICTION_MIN,
            "top_k": top_k,
        },
        "summaries": summaries, "case_results": all_rows,
    }
