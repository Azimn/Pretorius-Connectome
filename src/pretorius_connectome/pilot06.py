"""Pilot 06A: evidence-bounded retrieval, explicit unknown, and polarity checks.

No pretrained entailment model is used. A lexical sentence match never proves
semantic support. The benchmark is a previously seen assistant-authored set;
the measured result is a POST-HOC engineering diagnostic, not independent NLI.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from pretorius_connectome.imprinting import Memory, order_by_episode
from pretorius_connectome.pilot02 import calibrate, episode_split, words
from pretorius_connectome.pilot04 import ChallengeCase
from pretorius_connectome.pilot05 import NarrativeModels, _round, _summary


POLARITY_TERMS = frozenset({"not", "never", "no", "without", "neither", "nor"})
METHODS = (
    "bm25_calibrated",
    "bm25_sentence_only",
    "bm25_sentence_polarity",
    "tfidf_sentence_polarity",
    "hebb_word_sentence_polarity",
    "reject_all",
)


def sentences(text: str) -> list[str]:
    """Sentence spans derived only from the source narrative."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def significant_tokens(text: str) -> set[str]:
    return {word for word in words(text)
            if word not in ENGLISH_STOP_WORDS and word not in POLARITY_TERMS
            and len(word) > 2}


def polarity(text: str) -> bool:
    """Very narrow lexical negation heuristic, not logical truth verification."""
    return bool(POLARITY_TERMS & set(words(text)))


@dataclass
class EvidenceMatch:
    event_id: str | None
    sentence_index: int | None
    quote: str | None
    score: float
    lexical_anchors: int
    negation_mismatch: bool


class EvidenceIndex:
    """Build sentence-level TF-IDF ONLY from imprinted narrative documents."""
    def __init__(self, memories: list[Memory]):
        if not memories:
            raise ValueError("empty trained narrative index")
        self.memories = memories
        self.positions = {m.event_id: i for i, m in enumerate(memories)}
        self.sentence_spans = [sentences(m.memory_text) for m in memories]
        self.flat_sentences = [
            sentence for document in self.sentence_spans for sentence in document
        ]
        self.flat_owner = [
            (i, j) for i, doc in enumerate(self.sentence_spans)
            for j in range(len(doc))
        ]
        self.by_event: dict[str, list[int]] = {}
        for index, (doc, _) in enumerate(self.flat_owner):
            self.by_event.setdefault(memories[doc].event_id, []).append(index)
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), sublinear_tf=True, stop_words="english",
            max_features=20000, norm="l2",
        )
        self.matrix = self.vectorizer.fit_transform(self.flat_sentences)

    def match(self, query: str, event_id: str | None) -> EvidenceMatch:
        if event_id is None or event_id not in self.positions:
            return EvidenceMatch(None, None, None, 0.0, 0, False)
        indices = self.by_event[event_id]
        candidate = self.vectorizer.transform([query])
        similarities = (candidate @ self.matrix[indices].T).toarray()[0]
        local = int(np.argmax(similarities))
        score = float(similarities[local])
        quote = self.flat_sentences[indices[local]]
        anchors = len(significant_tokens(query) & significant_tokens(quote))
        return EvidenceMatch(
            event_id=event_id,
            sentence_index=self.flat_owner[indices[local]][1],
            quote=quote,
            score=score,
            lexical_anchors=anchors,
            negation_mismatch=polarity(query) != polarity(quote),
        )


def _verification(method: str, match: EvidenceMatch, threshold: float) -> str:
    """A conservative three-valued *heuristic*, not textual entailment."""
    if method == "reject_all" or match.event_id is None:
        return "insufficient_evidence"
    if match.score <= threshold or match.lexical_anchors < 2:
        return "insufficient_evidence"
    if method.endswith("_polarity") and match.negation_mismatch:
        return "possible_contradiction"
    return "lexically_supported"


def _make_rows(
    retrieval: NarrativeModels, evidence: EvidenceIndex, method: str,
    cases: list[ChallengeCase], retrieval_threshold: float,
    evidence_threshold: float,
) -> list[dict]:
    if not cases:
        return []
    base = (
        "hebb_word_narrative" if method == "hebb_word_sentence_polarity" else
        "tfidf_word_narrative" if method == "tfidf_sentence_polarity" else
        "bm25_narrative"
    )
    ranked = retrieval.score_batch([c.query for c in cases], base)
    rows = []
    for case, prediction in zip(cases, ranked):
        match = evidence.match(case.query, prediction["predicted"])
        if method == "bm25_calibrated":
            accepted = bool(prediction["score"] > retrieval_threshold)
            verdict = "retrieved_without_verification" if accepted else "insufficient_evidence"
        else:
            verdict = _verification(method, match, evidence_threshold)
            accepted = verdict == "lexically_supported"
        rows.append({
            "case_id": case.case_id,
            "target": case.event_id,
            "predicted": prediction["predicted"],
            "accepted": accepted,
            "verdict": verdict,
            "retrieval_score": _round(prediction["score"])
                if np.isfinite(prediction["score"]) else None,
            "evidence_score": _round(match.score),
            "lexical_anchors": match.lexical_anchors,
            "sentence_index": match.sentence_index,
            "evidence_event_id": match.event_id,
            "evidence_quote": match.quote[:300] if match.quote else None,
            "negation_mismatch": match.negation_mismatch,
        })
    return rows


def _calibrate(
    retrieval: NarrativeModels, evidence: EvidenceIndex,
    training: list[Memory], validation: list[Memory],
    decisions: dict[str, str], retrieval_method: str,
) -> tuple[float, float, dict]:
    """Separate calibration texts; no old challenge prompts are used."""
    known = retrieval.score_batch([decisions[m.event_id] for m in training], retrieval_method)
    absent = retrieval.score_batch([decisions[m.event_id] for m in validation], retrieval_method)
    known_evidence = [
        evidence.match(decisions[m.event_id], result["predicted"])
        for m, result in zip(training, known)
    ]
    absent_evidence = [
        evidence.match(decisions[m.event_id], result["predicted"])
        for m, result in zip(validation, absent)
    ]
    def threshold(good: list[float], bad: list[float]) -> float:
        return calibrate(
            [{"known": True, "top_cosine": x} for x in good],
            [{"known": False, "top_cosine": x} for x in bad],
        )
    retrieval_t = threshold([r["score"] for r in known], [r["score"] for r in absent])
    evidence_t = threshold([r.score for r in known_evidence],
                           [r.score for r in absent_evidence])
    return retrieval_t, evidence_t, {
        "known": len(known), "withheld": len(absent),
        "retrieval_known_acceptance": _round(np.mean([
            r["score"] > retrieval_t for r in known
        ])),
        "retrieval_unknown_acceptance": _round(np.mean([
            r["score"] > retrieval_t for r in absent
        ])),
        "evidence_known_acceptance_before_anchor_gate": _round(np.mean([
            r.score > evidence_t for r in known_evidence
        ])),
        "evidence_unknown_acceptance_before_anchor_gate": _round(np.mean([
            r.score > evidence_t for r in absent_evidence
        ])),
    }


def evaluate_seed(
    memories: list[Memory], decisions: dict[str, str],
    cases: list[ChallengeCase], seed: int,
    cue_units: int = 512, memory_units: int = 256,
    density: float = 0.55,
) -> dict:
    train, validation, test = episode_split(memories, seed)
    train = order_by_episode(train, seed=20261008)
    known_ids = {m.event_id for m in train}
    test_ids = {m.event_id for m in test}
    positives = [c for c in cases if c.kind == "paraphrase" and c.event_id in known_ids]
    contradictions = [c for c in cases if c.kind == "contradiction" and c.event_id in known_ids]
    unlearned = [c for c in cases if c.kind == "paraphrase" and c.event_id in test_ids]
    if min(len(positives), len(contradictions), len(unlearned)) < 2:
        raise ValueError("insufficient split coverage")
    if {c.event_id for c in positives} != {c.event_id for c in contradictions}:
        raise ValueError("challenge pair imbalance")

    retrieval = NarrativeModels(train, seed, cue_units, memory_units, density)
    evidence = EvidenceIndex(train)
    thresholds = {}
    calibration = {}
    for base in ("bm25_narrative", "tfidf_word_narrative", "hebb_word_narrative"):
        retrieval_t, evidence_t, metrics = _calibrate(
            retrieval, evidence, train, validation, decisions, base
        )
        thresholds[base] = (retrieval_t, evidence_t)
        calibration[base] = metrics

    output = {}
    for method in METHODS:
        base = (
            "hebb_word_narrative" if method == "hebb_word_sentence_polarity" else
            "tfidf_word_narrative" if method == "tfidf_sentence_polarity" else
            "bm25_narrative"
        )
        t_retrieval, t_evidence = thresholds[base]
        score_rows = {}
        for label, selection in (
            ("positive", positives),
            ("contradiction", contradictions),
            ("absent", unlearned),
        ):
            score_rows[label] = _make_rows(
                retrieval, evidence, method, selection,
                t_retrieval, t_evidence
            )
        output[method] = {
            "retrieval_calibration_threshold": _round(t_retrieval),
            "sentence_calibration_threshold": _round(t_evidence),
            "summary": {
                "positive": _summary(score_rows["positive"], "positive"),
                "contradiction": _summary(score_rows["contradiction"], "contradiction"),
                "absent": _summary(score_rows["absent"], "absent"),
                "positive_abstention": _round(np.mean([
                    not r["accepted"] for r in score_rows["positive"]
                ])),
                "explicit_possible_contradiction_count": sum(
                    r["verdict"] == "possible_contradiction"
                    for r in score_rows["contradiction"]
                ),
            },
            "rows": score_rows,
        }
    return {
        "seed": seed,
        "train_events": len(train),
        "calibration_unknown_events": len(validation),
        "test_unknown_events": len(test),
        "trained_positive_cases": len(positives),
        "trained_counterfactual_cases": len(contradictions),
        "heldout_episode_cases": len(unlearned),
        "trained_source_sentences": len(evidence.flat_sentences),
        "train_only_evidence_vocab": len(evidence.vectorizer.vocabulary_),
        "calibration": calibration,
        "conditions": output,
    }
