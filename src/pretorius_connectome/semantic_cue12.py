"""Pilot12 frozen, local ONNX sentence encoder for cue-only neural inference.

This encoder knows no autobiographical event IDs, narrative targets or retrieval
indices. MiniLM weights and tokenizer are pinned to a published revision.
Frozen 384D sentence embeddings are projected to the *same* 256D signed
coordinates used by the original fly synaptic readout. Only input encoding
changes; original BC01 256D narrative CONTENT targets are preserved.

An ephemeral cue->vector cache avoids repeating deterministic public model
inference; it never stores source records, event IDs, target vectors or rankings.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import numpy as np

from pretorius_connectome.direct_flywire_imprint import DirectFlywireOverlay, select_features
from pretorius_connectome.shared_features_bc01 import encode_bc_sensory

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
ONNX_NAME = "onnx/model.onnx"
PROJECTION_SEED = 20261008


def project_sentence_vectors(embeddings: np.ndarray) -> np.ndarray:
    """Frozen input-only 384->256 projection, independent of all event labels."""
    source = np.asarray(embeddings, dtype=np.float32)
    if source.ndim != 2 or source.shape[1] != 384 or not np.isfinite(source).all():
        raise ValueError("Expected finite pretrained 384D sentence representations")
    matrix = np.random.default_rng(PROJECTION_SEED).standard_normal(
        (384, 256)).astype(np.float32) / np.sqrt(np.float32(384.0))
    projected = source @ matrix
    norms = np.linalg.norm(projected, axis=1, keepdims=True)
    return (projected / np.maximum(norms, 1e-12)).astype(np.float32)


def projection_sha256() -> str:
    matrix = np.random.default_rng(PROJECTION_SEED).standard_normal(
        (384, 256)).astype(np.float32) / np.sqrt(np.float32(384.0))
    return sha256(np.ascontiguousarray(matrix).tobytes()).hexdigest()


class FrozenMiniLMCues:
    """CPU ONNX; source text enters only through stateless pretrained tokenizer.

    Loading encoder does NOT load an autobiography or set of cue-target pairs.
    Batch prewarm is a computational optimization, never a retrieval index.
    """

    def __init__(self, *, cache_dir: str | Path | None = None):
        try:
            from huggingface_hub import hf_hub_download
            import onnxruntime as ort
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Pilot12 needs huggingface_hub, onnxruntime and transformers; "
                "no paid API is involved"
            ) from exc
        model_path = hf_hub_download(
            repo_id=MODEL, filename=ONNX_NAME, revision=REVISION,
            cache_dir=cache_dir,
        )
        self.onnx_sha256 = sha256(Path(model_path).read_bytes()).hexdigest()
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL, revision=REVISION, cache_dir=cache_dir
        )
        self.session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"],
            sess_options=ort.SessionOptions(),
        )
        self.input_names = {node.name for node in self.session.get_inputs()}
        self.outputs = self.session.get_outputs()
        self.cache: dict[str, np.ndarray] = {}
        self.forward_batches = 0

    def _batch(self, texts: list[str]) -> np.ndarray:
        tokens = self.tokenizer(
            texts, padding=True, truncation=True, max_length=256,
            return_tensors="np",
        )
        feed = {
            key: np.asarray(tokens[key], dtype=np.int64)
            for key in self.input_names if key in tokens
        }
        if set(feed) != self.input_names:
            raise ValueError("Tokenizer/model ONNX inputs disagree")
        outputs = self.session.run(None, feed)
        # Canonical MiniLM ONNX may expose pooled 2D output or 3D tokens.
        candidates = [a for a in outputs if
                      isinstance(a, np.ndarray) and a.shape[-1] == 384]
        if not candidates:
            raise ValueError("Pinned MiniLM ONNX did not return hidden size 384")
        last_hidden = next((a for a in candidates if a.ndim == 3), None)
        if last_hidden is None:
            pooled = next(a for a in candidates if a.ndim == 2)
        else:
            mask = tokens["attention_mask"][:, :, None].astype(np.float32)
            pooled = np.sum(last_hidden.astype(np.float32) * mask, axis=1) / (
                np.maximum(np.sum(mask, axis=1), 1.0)
            )
        pooled = np.asarray(pooled, dtype=np.float32)
        pooled /= np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)
        self.forward_batches += 1
        return project_sentence_vectors(pooled)

    def prewarm(self, texts, *, batch_size: int = 32):
        unique = list(dict.fromkeys(t for t in texts
                                    if isinstance(t, str) and t.strip()
                                    and t not in self.cache))
        for start in range(0, len(unique), batch_size):
            batch = unique[start:start + batch_size]
            encoded = self._batch(batch)
            for cue, vector in zip(batch, encoded):
                self.cache[cue] = vector

    def __call__(self, cue: str) -> np.ndarray:
        if not isinstance(cue, str) or not cue.strip():
            raise ValueError("Source cue must be nonempty text")
        if cue not in self.cache:
            self.prewarm([cue])
        return self.cache[cue]


class CueOnlyOverlay(DirectFlywireOverlay):
    """Original immutable synaptic learning rule with an alternate cue encoder.

    Checkpoints MUST be labeled in the outer Pilot12 manifest: legacy Pilot08
    save()/load() does not encode a cue encoder identity. Never load this as
    a BC01-only model or imply pretrained language knowledge lives in fly edges.
    """

    def __init__(self, topology, *, cue_encoder=None, **kwargs):
        super().__init__(topology, **kwargs)
        self.cue_encoder = cue_encoder or encode_bc_sensory

    def _input(self, cue: str) -> np.ndarray:
        return select_features(self.cue_encoder(cue), top_k=self.cue_features)
