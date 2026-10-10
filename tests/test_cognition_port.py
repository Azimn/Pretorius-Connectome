"""Synthetic two-subject isolation and read-only cognition evidence tests."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
from hashlib import sha256
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pretorius_connectome.cognition_port import (
    MemoryAccessDenied, MemoryCandidate, MemoryCognitionPort,
    MemoryPortError, MemorySourceMismatch, SourceDescriptor,
)


class LedgerFixture:
    def __init__(self, subject):
        self.descriptor = SourceDescriptor(
            subject_id=subject, namespace=subject + ".events.v1",
            source_version="v1", source_hash=sha256(subject.encode()).hexdigest(),
            encoder_id="synthetic-exact-v1", index_scope="subject",
        )
        self.records = {
            subject + "-1": MemoryCandidate(
                subject, self.descriptor.namespace, subject + "-1",
                self.descriptor.source_hash, "synthetic_fixture",
                subject + " remembers a brass key.", 0.8,
            ),
        }
        self.calls = 0

    def search(self, query, *, top_k):
        self.calls += 1
        return list(self.records.values())[:top_k] if "key" in query else []

    def get(self, record_id):
        self.calls += 1
        return self.records.get(record_id)


class MemoryCognitionPortTests(unittest.TestCase):
    def setUp(self):
        self.port = MemoryCognitionPort()
        self.ada = LedgerFixture("ada")
        self.bryn = LedgerFixture("bryn")
        self.port.register(self.ada)
        self.port.register(self.bryn)
        self.ada_token = self.port.grant_self(subject_id="ada", namespace="ada.events.v1")
        self.bryn_token = self.port.grant_self(subject_id="bryn", namespace="bryn.events.v1")

    def test_independent_subjects_have_separate_retrieval_and_provenance(self):
        a = self.port.search(token=self.ada_token, subject_id="ada", namespace="ada.events.v1", query="key")
        b = self.port.search(token=self.bryn_token, subject_id="bryn", namespace="bryn.events.v1", query="key")
        self.assertEqual([x.record_id for x in a], ["ada-1"])
        self.assertEqual([x.record_id for x in b], ["bryn-1"])
        self.assertNotEqual(a[0].source_hash, b[0].source_hash)
        self.assertEqual(a[0].epistemic_status, "candidate_not_entailment")
        self.assertEqual(a[0].channel, "external_memory_evidence")
        self.assertFalse(a[0].eligible_for_lived_write)
        with self.assertRaises(FrozenInstanceError):
            a[0].subject_id = "bryn"
        self.assertEqual(self.port.get(token=self.ada_token, subject_id="ada", namespace="ada.events.v1", record_id="ada-1").record_id, "ada-1")
        self.assertIsNone(self.port.get(token=self.ada_token, subject_id="ada", namespace="ada.events.v1", record_id="bryn-1"))

    def test_wrong_owner_rejected_before_backend_called(self):
        for token, subject, namespace in (
            (self.ada_token, "bryn", "ada.events.v1"),
            (self.ada_token, "ada", "bryn.events.v1"),
            (self.bryn_token, "ada", "ada.events.v1"),
        ):
            with self.assertRaises(MemoryAccessDenied):
                self.port.search(token=token, subject_id=subject, namespace=namespace, query="key")
        with self.assertRaises(MemoryAccessDenied):
            self.port.grant_self(subject_id="ada", namespace="bryn.events.v1")
        self.assertEqual(self.ada.calls + self.bryn.calls, 0)

    def test_revocation_and_stale_source_cannot_retrieve(self):
        self.port.revoke(self.ada_token)
        with self.assertRaises(MemoryAccessDenied):
            self.port.get(token=self.ada_token, subject_id="ada", namespace="ada.events.v1", record_id="ada-1")
        self.assertEqual(self.ada.calls, 0)
        self.ada_token = self.port.grant_self(subject_id="ada", namespace="ada.events.v1")
        self.ada.descriptor = SourceDescriptor(subject_id="ada", namespace="ada.events.v1", source_version="v2", source_hash=self.ada.descriptor.source_hash, encoder_id="synthetic-exact-v1", index_scope="subject")
        with self.assertRaises(MemoryAccessDenied):
            self.port.search(token=self.ada_token, subject_id="ada", namespace="ada.events.v1", query="key")
        self.assertEqual(self.ada.calls, 0)

    def test_wrong_owner_and_content_hash_are_not_silently_accepted(self):
        self.ada.records["ada-1"] = MemoryCandidate(
            "bryn", "ada.events.v1", "ada-1", self.ada.descriptor.source_hash,
            "synthetic_fixture", "forged memory", 0.7,
        )
        with self.assertRaises(MemorySourceMismatch):
            self.port.search(token=self.ada_token, subject_id="ada", namespace="ada.events.v1", query="key")
        self.ada.records["ada-1"] = MemoryCandidate(
            "ada", "ada.events.v1", "ada-1", "0" * 64,
            "synthetic_fixture", "forged memory", 0.7,
        )
        with self.assertRaises(MemorySourceMismatch):
            self.port.search(token=self.ada_token, subject_id="ada", namespace="ada.events.v1", query="key")

    def test_bad_scores_and_duplicate_candidates_fail_closed(self):
        original = self.ada.records["ada-1"]
        for score in (float("nan"), float("inf"), -0.2, True):
            self.ada.records["ada-1"] = MemoryCandidate("ada", "ada.events.v1", "ada-1", self.ada.descriptor.source_hash, "synthetic_fixture", "key", score)
            with self.assertRaises(MemorySourceMismatch):
                self.port.search(token=self.ada_token, subject_id="ada", namespace="ada.events.v1", query="key")
        self.ada.records["ada-1"] = original
        self.ada.records["duplicate"] = original
        with self.assertRaises(MemorySourceMismatch):
            self.port.search(token=self.ada_token, subject_id="ada", namespace="ada.events.v1", query="key")

    def test_missing_or_invalid_capability_and_invalid_requests(self):
        with self.assertRaises(MemoryAccessDenied):
            self.port.search(token="wrong", subject_id="ada", namespace="ada.events.v1", query="key")
        for q in ("", " ", "x" * 2001):
            with self.assertRaises(MemoryPortError):
                self.port.search(token=self.ada_token, subject_id="ada", namespace="ada.events.v1", query=q)
        for n in (0, 21, True, 1.5):
            with self.assertRaises(MemoryPortError):
                self.port.search(token=self.ada_token, subject_id="ada", namespace="ada.events.v1", query="key", top_k=n)
        with self.assertRaises(MemoryPortError):
            self.port.get(token=self.ada_token, subject_id="ada", namespace="ada.events.v1", record_id="../bryn")

    def test_get_enforces_exact_requested_record_identity(self):
        self.ada.get = lambda _: MemoryCandidate("ada", "ada.events.v1", "different", self.ada.descriptor.source_hash, "synthetic_fixture", "key")
        with self.assertRaises(MemorySourceMismatch):
            self.port.get(token=self.ada_token, subject_id="ada", namespace="ada.events.v1", record_id="ada-1")

    def test_duplicate_source_registration_rejected(self):
        with self.assertRaises(MemoryPortError):
            self.port.register(LedgerFixture("ada"))


if __name__ == "__main__":
    unittest.main()
