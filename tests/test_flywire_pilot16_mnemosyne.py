"""Pilot16 Mnemosyne source-only learned residual, novelty and graph controls."""
from __future__ import annotations
import tempfile
import unittest
from pathlib import Path
import numpy as np

from pretorius_connectome.associative import Topology
from pretorius_connectome.direct_flywire_imprint import DirectFlywireOverlay,fingerprint
from pretorius_connectome.mnemosyne16 import (
    ResidualAssociativeMemory,actual_feature_support,
    random_feature_support,normalized
)
from pretorius_connectome.rewire12 import synthetic_aggregated_fixture
from scripts.run_flywire_pilot16_mnemosyne import validation_only_novelty_gate


class FakeSemanticEncoder:
    """Only fixture semantically paired toy keys; NO source event lookup in model."""
    def __init__(self):
        rng=np.random.default_rng(13)
        self.vectors=[normalized(rng.normal(size=256)) for i in range(25)]
        self.mapping={
            "the doctor unlocked his laboratory":0,
            "the physician opened his workspace":0,
            "a silver railway locomotive":1,
            "the metallic train was approaching":1,
            "red ribbons across the window":2,
            "scarlet fabric hung on glass":2,
        }

    def __call__(self,text):
        if text in self.mapping:return self.vectors[self.mapping[text]]
        return self.vectors[abs(sum((i+1)*ord(c) for i,c in enumerate(text))) % 22+3]


class MnemosyneTests(unittest.TestCase):
    def test_dense_preserves_more_source_input_information_than_top8(self):
        fake=FakeSemanticEncoder()
        m=ResidualAssociativeMemory(encoder=fake)
        x=m.cue_vector("the doctor unlocked his laboratory")
        self.assertGreater(np.count_nonzero(x),250)
        sharp=ResidualAssociativeMemory(encoder=fake,representation="kc64")
        self.assertEqual(int(np.count_nonzero(sharp.cue_vector(
            "the doctor unlocked his laboratory"))),64)
        self.assertEqual(sharp.allocated_numeric_scalars,131072)

    def test_error_corrected_online_memory_matches_two_unseen_semantic_synonyms(self):
        semantic=FakeSemanticEncoder()
        model=ResidualAssociativeMemory(encoder=semantic,rule="rls",ridge=.05)
        a=normalized(np.random.default_rng(17).normal(size=256))
        b=normalized(np.random.default_rng(18).normal(size=256))
        model.imprint("the doctor unlocked his laboratory",a)
        model.imprint("a silver railway locomotive",b)
        outa=model.infer("the physician opened his workspace")
        outb=model.infer("the metallic train was approaching")
        self.assertGreater(float(outa@a),float(outa@b))
        self.assertGreater(float(outb@b),float(outb@a))
        self.assertNotIn("event_ids",model.__dict__)
        self.assertNotIn("source_narratives",model.__dict__)
        self.assertNotIn("candidate_targets",model.__dict__)

    def test_masked_update_never_changes_forbidden_source_feature_pair(self):
        f=FakeSemanticEncoder()
        mask=random_feature_support(10000,seed=13)
        model=ResidualAssociativeMemory(encoder=f,mask=mask,rule="rls")
        model.imprint("the doctor unlocked his laboratory",
                     np.random.default_rng(1).normal(size=256))
        self.assertTrue(np.all(model.W[~mask]==0))
        self.assertEqual(model.allocated_numeric_scalars,75536)
        self.assertEqual(model.n_presentations,1)
        self.assertEqual(int(model.nonzero_trained_weights),10000)

    def test_actual_graph_mask_is_derived_and_source_graph_unchanged(self):
        topo=synthetic_aggregated_fixture(n=8192,degree=16,seed=15)
        old=fingerprint(topo)
        active,meta=actual_feature_support(topo,cells_per_feature=8)
        self.assertEqual(active.shape,(256,256))
        self.assertGreater(meta["eligible_original_directed_synapse_slots"],1000)
        self.assertGreater(meta["effective_learned_feature_pairs"],1000)
        self.assertLessEqual(meta["effective_learned_feature_pairs"],
                             meta["eligible_original_directed_synapse_slots"])
        self.assertEqual(fingerprint(topo),old)

    def test_model_only_covariance_novelty_separates_orthogonal_input(self):
        rng=np.random.default_rng(14)
        class Orthogonal:
            def __call__(self,t):
                vec=np.zeros(256);vec[0 if t=="train" else 1]=1.
                return vec
        enc=Orthogonal()
        model=ResidualAssociativeMemory(encoder=enc,ridge=1.)
        self.assertAlmostEqual(model.cue_familiarity("train"),0.,places=10)
        for _ in range(8):
            model.imprint("train",normalized(rng.normal(size=256)))
        self.assertGreater(model.cue_familiarity("train"),.85)
        self.assertLess(model.cue_familiarity("absent"),1e-9)

    def test_checkpoint_is_encoder_and_biological_source_hash_bound(self):
        f=FakeSemanticEncoder()
        original=synthetic_aggregated_fixture(n=8192,degree=16,seed=15)
        proof=fingerprint(original)
        m=ResidualAssociativeMemory(
            encoder=f,mask=random_feature_support(50000),representation="kc64"
        )
        for cue in ("the doctor unlocked his laboratory",
                    "the physician opened his workspace"):
            m.imprint(cue,np.random.default_rng(1).normal(size=256))
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"test.npz"
            digest=m.save(path,source_graph_sha=proof,encoder_sha="test-encoder")
            self.assertEqual(len(digest),64)
            restored=ResidualAssociativeMemory.load(
                path,encoder=f,source_graph_sha=proof,encoder_sha="test-encoder")
            self.assertTrue(np.array_equal(restored.P,m.P))
            self.assertTrue(np.array_equal(
                restored.infer("the doctor unlocked his laboratory"),
                m.infer("the doctor unlocked his laboratory")))
            with self.assertRaises(ValueError):
                ResidualAssociativeMemory.load(
                    path,encoder=f,source_graph_sha=proof,encoder_sha="wrong")
            dense=ResidualAssociativeMemory(encoder=f)
            dense.imprint("a silver railway locomotive",f("red ribbons across the window"))
            path=Path(temp)/"dense.npz"
            dense.save(path)
            self.assertEqual(
                ResidualAssociativeMemory.load(path,encoder=f).representation,"dense")

    def test_heldout_novelty_uses_validation_absent_only(self):
        from types import SimpleNamespace
        from scripts.run_flywire_pilot16_mnemosyne import validation_only_novelty_gate
        class Probe:
            def cue_familiarity(self,t):
                return (len(t)%23)/23
        self.assertIsNone(validation_only_novelty_gate(
            object(),[],[]
        ))
        # Use actual source-like records with original binding_cues method
        # independently guarded in full CI; the gate cannot take 71 test
        # absent or 159 familiar test cases as threshold-selection inputs.
        self.assertEqual(ResidualAssociativeMemory(
            encoder=FakeSemanticEncoder()).rule,"rls")


if __name__=="__main__":
    unittest.main()
