"""Pilot 05A: post-hoc narrative-grounded retrieval vs masked imprinting.

This module deliberately uses the already seen Pilot 04 challenge for a
DIAGNOSTIC only. Full narrative documents become the *training input*, unlike
the cue-only pilots. A lexical TF-IDF encoder is not semantic understanding.
Weights alone are not a memory retriever: fingerprint decoding uses an oracle.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

from pretorius_connectome.imprinting import Memory, SynapticOverlay, content_fingerprint, order_by_episode
from pretorius_connectome.pilot02 import episode_split, calibrate, words
from pretorius_connectome.pilot04 import ChallengeCase


METHODS = ("bm25_narrative", "tfidf_word_narrative", "tfidf_char_narrative",
           "hebb_word_narrative", "hebb_char_narrative",
           "hebb_word_shuffled", "hebb_word_unmodified")
WORD_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", flags=re.UNICODE)


def _round(x: float) -> float:
    return round(float(x), 6)


def _summary(rows: list[dict], kind: str) -> dict:
    if not rows:
        raise ValueError("empty scoring set")
    if kind == "positive":
        return {
            "n": len(rows),
            "top1": _round(np.mean([r["predicted"] == r["target"] for r in rows])),
            "accepted": _round(np.mean([r["accepted"] for r in rows])),
            "correct_and_accepted": _round(np.mean([
                r["accepted"] and r["predicted"] == r["target"] for r in rows
            ])),
        }
    if kind == "contradiction":
        return {
            "n": len(rows),
            "false_acceptance": _round(np.mean([r["accepted"] for r in rows])),
            "false_target_confirmation": _round(np.mean([
                r["accepted"] and r["predicted"] == r["target"] for r in rows
            ])),
            "raw_target_choice": _round(np.mean([
                r["predicted"] == r["target"] for r in rows
            ])),
        }
    if kind == "absent":
        return {
            "n": len(rows),
            "false_acceptance": _round(np.mean([r["accepted"] for r in rows])),
        }
    raise ValueError("unknown summary kind")


@dataclass
class NarrativeModels:
    """All statistics are fit on imprinted episodes, never challenge prompts."""
    train: list[Memory]
    seed: int
    cue_units: int = 512
    memory_units: int = 256
    density: float = 0.55

    def __post_init__(self) -> None:
        if not self.train:
            raise ValueError("no narrative training source")
        docs = [m.memory_text for m in self.train]
        self.identifiers = [m.event_id for m in self.train]
        self.identifier_set = set(self.identifiers)
        self.word_vectorizer = TfidfVectorizer(
            max_features=self.cue_units, stop_words="english",
            ngram_range=(1, 2), sublinear_tf=True, norm="l2",
        )
        self.char_vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5),
            max_features=self.cue_units, sublinear_tf=True, norm="l2",
        )
        self.doc_word = self.word_vectorizer.fit_transform(docs)
        self.doc_char = self.char_vectorizer.fit_transform(docs)
        self.bm25 = BM25Okapi([list(words(doc)) for doc in docs])
        self.targets = np.stack([
            content_fingerprint(doc, self.memory_units) for doc in docs
        ])
        self.overlays = {}
        for method in ("hebb_word_narrative", "hebb_char_narrative",
                       "hebb_word_shuffled", "hebb_word_unmodified"):
            self.overlays[method] = SynapticOverlay(
                self.cue_units, self.memory_units, self.density, self.seed
            )
        # One deterministic, fixed-point-free permutation; no narrative IDs
        # or target labels enter the neural weights.
        rng = np.random.default_rng(self.seed + 33017)
        perm = rng.permutation(len(docs))
        swapped = np.empty(len(docs), dtype=np.int64)
        swapped[perm] = np.roll(perm, 1)
        for i, target_index in enumerate(swapped):
            if int(target_index) == i:
                raise ValueError("shuffled control has fixed point")
        self.shuffled_mapping = swapped.tolist()
        for name, x, desired in (
            ("hebb_word_narrative", self.doc_word, self.targets),
            ("hebb_char_narrative", self.doc_char, self.targets),
            ("hebb_word_shuffled", self.doc_word, self.targets[swapped]),
        ):
            dense = x.toarray().astype(np.float32, copy=False)
            if dense.shape[1] < self.cue_units:
                dense = np.pad(dense, ((0, 0), (0, self.cue_units - dense.shape[1])))
            # Matrix product is the same sum of outer products as per-event
            # masked additive Hebbian learning. The mask never changes.
            trained = self.overlays[name]
            trained.weights[:] = ((desired.T @ dense) * trained.mask)
            trained.updates = len(docs)
        self.train_doc_count = len(docs)
        self.word_features = len(self.word_vectorizer.vocabulary_)
        self.char_features = len(self.char_vectorizer.vocabulary_)

    def _vector(self, texts: list[str], kind: str) -> np.ndarray:
        v = self.word_vectorizer if kind == "word" else self.char_vectorizer
        sparse = v.transform(texts)
        dense = sparse.toarray().astype(np.float32)
        if dense.shape[1] < self.cue_units:
            # The neural mask remains fixed-size even for tiny unit-test corpora.
            dense = np.pad(dense, ((0, 0), (0, self.cue_units - dense.shape[1])))
        return dense

    def rank(self, texts: list[str], method: str) -> np.ndarray:
        """Higher rank score is better. No query target or label access."""
        if method == "bm25_narrative":
            return np.stack([
                self.bm25.get_scores(list(words(text)))
                for text in texts
            ]).astype(np.float32)
        if method == "tfidf_word_narrative":
            return (self.word_vectorizer.transform(texts) @ self.doc_word.T).toarray()
        if method == "tfidf_char_narrative":
            return (self.char_vectorizer.transform(texts) @ self.doc_char.T).toarray()
        if method in self.overlays:
            x = self._vector(texts, "char" if method == "hebb_char_narrative" else "word")
            raw = x @ self.overlays[method].weights.T
            magnitude = np.linalg.norm(raw, axis=1)
            normed = raw / np.maximum(magnitude[:, None], 1e-12)
            result = normed @ self.targets.T
            # A completely unmodified substrate emits no identity at all.
            return np.where(magnitude[:, None] > 1e-12, result, -np.inf)
        raise ValueError("unknown retrieval method")

    def _best(self, scores: np.ndarray) -> tuple[str | None, float, float]:
        if not np.isfinite(scores).any():
            return None, float("-inf"), 0.0
        winner = int(np.argmax(scores))
        top = float(scores[winner])
        if len(scores) > 1:
            second = float(np.partition(scores, -2)[-2])
        else:
            second = 0.0
        return self.identifiers[winner], top, top - second

    def score_batch(self, texts: list[str], method: str) -> list[dict]:
        ranked = self.rank(texts, method)
        results = []
        for row in ranked:
            predicted, score, margin = self._best(row)
            results.append({
                "predicted": predicted,
                "score": score,
                "margin": margin,
            })
        return results

    def evidence(self, predicted: str | None, query: str) -> str | None:
        """Return inspectable narrative snippet ONLY for lexical retrievers.

        This is not fact verification or entailment. The full narrative source
        remains accessible to retrieval-based methods, unlike the Hebbian mask.
        """
        if predicted is None or predicted not in self.identifier_set:
            return None
        text = self.train[self.identifiers.index(predicted)].memory_text
        sentences = re.split(r"(?<=[.!?])\s+", text)
        query_words = set(words(query))
        selected = max(sentences, key=lambda sentence: len(query_words & set(words(sentence))))
        return selected[:240]


def evaluate_seed(memories: list[Memory], decision_text: dict[str, str],
                  cases: list[ChallengeCase], seed: int,
                  cue_units: int = 512, memory_units: int = 256,
                  density: float = 0.55) -> dict:
    """Post-hoc results on known Pilot 04 prompts, no challenge-tuned thresholds."""
    train, validation, test = episode_split(memories, seed)
    train = order_by_episode(train, seed=20261008)
    for item in train + validation + test:
        if not decision_text.get(item.event_id):
            raise ValueError(f"missing locked source decision {item.event_id}")
    model = NarrativeModels(train, seed, cue_units, memory_units, density)
    trained_ids = set(model.identifiers)
    test_ids = {m.event_id for m in test}
    known_pos = [c for c in cases if c.event_id in trained_ids and c.kind == "paraphrase"]
    known_neg = [c for c in cases if c.event_id in trained_ids and c.kind == "contradiction"]
    missing_pos = [c for c in cases if c.event_id in test_ids and c.kind == "paraphrase"]
    if min(len(known_pos), len(known_neg), len(missing_pos)) <= 0:
        raise ValueError("empty challenge split")

    rows = {}
    for method in METHODS:
        cal_pos = model.score_batch([decision_text[m.event_id] for m in train], method)
        cal_neg = model.score_batch([decision_text[m.event_id] for m in validation], method)
        if method == "hebb_word_unmodified":
            threshold = float("inf")
        else:
            threshold = calibrate(
                [{"known": True, "top_cosine": r["score"]} for r in cal_pos],
                [{"known": False, "top_cosine": r["score"]} for r in cal_neg],
            )
        outcomes: dict[str, list[dict]] = {}
        for group, selected in (
            ("positive", known_pos), ("contradiction", known_neg),
            ("absent", missing_pos)
        ):
            scored = model.score_batch([c.query for c in selected], method)
            outcomes[group] = [
                {
                    "case_id": c.case_id,
                    "target": c.event_id,
                    "predicted": r["predicted"],
                    "score": _round(r["score"]) if np.isfinite(r["score"]) else None,
                    "margin": _round(r["margin"]) if np.isfinite(r["margin"]) else None,
                    "accepted": bool(r["score"] > threshold),
                    "evidence": (
                        model.evidence(r["predicted"], c.query)
                        if method in ("bm25_narrative", "tfidf_word_narrative",
                                      "tfidf_char_narrative")
                        else None
                    ),
                }
                for c, r in zip(selected, scored)
            ]
        rows[method] = {
            "threshold": _round(threshold) if np.isfinite(threshold) else None,
            "summary": {
                "positive": _summary(outcomes["positive"], "positive"),
                "contradiction": _summary(outcomes["contradiction"], "contradiction"),
                "absent": _summary(outcomes["absent"], "absent"),
            },
            "calibration": {
                "known_count": len(cal_pos),
                "unknown_count": len(cal_neg),
                "known_acceptance": _round(np.mean([r["score"] > threshold for r in cal_pos])),
                "unknown_false_acceptance": _round(np.mean([r["score"] > threshold for r in cal_neg])),
            },
            "case_results": outcomes,
        }
    return {
        "seed": seed,
        "trained_events": len(train),
        "validation_unknown_events": len(validation),
        "test_unknown_events": len(test),
        "candidate_fingerprint_units": memory_units,
        "cue_input_units": cue_units,
        "train_only_word_vocab": model.word_features,
        "train_only_char_vocab": model.char_features,
        "method_results": rows,
    }
