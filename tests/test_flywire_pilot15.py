"""Pilot15 dual synaptic trace original-graph and leakage guards."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import numpy as np

from pretorius_connectome.associative import Topology
from pretorius_connectome.direct_flywire_imprint import (
    fingerprint,DirectFlywireOverlay
)
from pretorius_connectome.dual_trace15 import (
    DualTraceSynapses,DualTraceLinear,eligible_slots
)
from pretorius_connectome.shared_features_bc01 import encode_bc_sensory
from pretorius_connectome.rewire12 import synthetic_aggregated_fixture
from scripts.run_flywire_pilot15 import ARMS,GROUPS,FAST_DECAY,SLOW_SHARE,models_for


class Pilot15Tests(unittest.TestCase):
    def setUp(self):
        self.graph=synthetic_aggregated_fixture(n=4096,degree=40,seed=19)
        self.original_sha=fingerprint(self.graph)
        self.cues=[
            ("laboratory glass",encode_bc_sensory("a silver prism and blue light")),
            ("echoes in hall",encode_bc_sensory("the old doctor remembered a room")),
            ("fog at night",encode_bc_sensory("red ribbon near the library")),
        ]

    def test_exact_original_edge_slots_are_fixed_and_only_nonzero_on_those_slots(self):
        model=DualTraceSynapses(
            self.graph,beta=1.0,cells_per_feature=4
        )
        positions=eligible_slots(model.slow)
        self.assertGreater(len(positions),0)
        self.assertEqual(len(positions),len(np.unique(positions)))
        model.imprint(*self.cues[0])
        mask=np.ones(len(self.graph.indices),dtype=bool)
        mask[positions]=False
        self.assertFalse(np.any(model.fast.delta[mask]))
        self.assertFalse(np.any(model.slow.delta[mask]))
        self.assertEqual(fingerprint(self.graph),self.original_sha)

    def test_fast_decay_is_per_complete_three_cue_memory(self):
        m=DualTraceSynapses(self.graph,beta=1.0,cells_per_feature=4)
        for cue,content in self.cues:
            m.imprint(cue,content)
        before=m.fast.delta.copy()
        with self.assertRaises(AssertionError):
            m.imprint(self.cues[0][0],self.cues[0][1])
            m.finish_memory()
        # Restore exact cue exposure schedule in a fresh model.
        m=DualTraceSynapses(self.graph,beta=1.0,cells_per_feature=4)
        for cue,content in self.cues:
            m.imprint(cue,content)
        before=m.fast.delta.copy()
        m.finish_memory()
        expected=before.copy()
        expected[m.eligible_positions]*=np.float32(.97)
        self.assertTrue(np.array_equal(m.fast.delta,expected))
        self.assertEqual(m.completed_memories,1)
        self.assertEqual(m.imprints,3)
        self.assertEqual(m.slow.imprints,m.fast.imprints)
        with self.assertRaises(AssertionError):
            m.finish_memory()

    def test_source_only_inference_and_exact_replay(self):
        m=DualTraceSynapses(
            self.graph,beta=4,fast_decay=.97,slow_share=.6,
            cells_per_feature=4
        )
        for cue,content in self.cues:
            m.imprint(cue,content)
        m.finish_memory()
        answer=m.infer("laboratory glass")
        self.assertEqual(answer.shape,(256,))
        self.assertTrue(np.isfinite(answer).all())
        self.assertNotIn("event_ids",m.__dict__)
        self.assertNotIn("candidate_targets",m.__dict__)
        self.assertNotIn("source_narratives",m.__dict__)
        with tempfile.TemporaryDirectory() as td:
            f=Path(td)/"original-dual.npz"
            digest=m.save_dual(f)
            self.assertEqual(len(digest),64)
            reread=DualTraceSynapses.load_dual(self.graph,f)
            self.assertTrue(np.array_equal(reread.slow.delta,m.slow.delta))
            self.assertTrue(np.array_equal(reread.fast.delta,m.fast.delta))
            self.assertTrue(np.array_equal(
                reread.slow.edge_exposures,m.slow.edge_exposures
            ))
            self.assertEqual(reread.completed_memories,1)
            self.assertTrue(np.array_equal(
                reread.infer("laboratory glass"),answer
            ))
            other=synthetic_aggregated_fixture(n=4096,degree=40,seed=21)
            with self.assertRaises(ValueError):
                DualTraceSynapses.load_dual(other,f)
        self.assertEqual(fingerprint(self.graph),self.original_sha)

    def test_dual_linear_matched_double_slot_count_and_source_only(self):
        original=DualTraceSynapses(
            self.graph,cells_per_feature=4
        )
        count=len(original.eligible_positions)
        linear=DualTraceLinear(slot_count=count,seed=73)
        self.assertEqual(linear.exposure_diagnostics[
            "independent_numeric_trainable_slots"],count*2
        )
        self.assertTrue(np.array_equal(linear.slow.mask,linear.fast.mask))
        for cue,content in self.cues:
            linear.imprint(cue,content)
        linear.finish_memory()
        response=linear.infer("laboratory glass")
        self.assertEqual(response.shape,(256,))
        self.assertTrue(np.isfinite(response).all())
        self.assertNotIn("event_ids",linear.__dict__)
        with tempfile.TemporaryDirectory() as td:
            f=Path(td)/"dual-linear.npz"
            d=linear.save_dual_linear(f)
            self.assertEqual(len(d),64)
            with np.load(f,allow_pickle=False) as z:
                self.assertTrue(np.array_equal(z["slow_weights"],linear.slow.weights))
                self.assertTrue(np.array_equal(z["fast_weights"],linear.fast.weights))

    def test_all_eight_predeclared_controls_and_no_pretorius_records_in_models(self):
        ms=models_for(self.graph,self.graph,4,1000)
        self.assertEqual(tuple(ms),ARMS)
        self.assertEqual(len(ms),8)
        self.assertEqual(len(GROUPS),5)
        self.assertEqual(FAST_DECAY,.97)
        self.assertEqual(SLOW_SHARE,.6)
        for m in ms.values():
            self.assertNotIn("event_ids",m.__dict__)
            self.assertNotIn("memory_records",m.__dict__)
        self.assertEqual(ms["original_dual_beta4"].beta,4.)
        self.assertEqual(ms["original_dual_beta1"].beta,1.)
        self.assertEqual(ms["matched_slot_dual_linear"].slot_count,1000)
        self.assertIsInstance(ms["original_additive"],DirectFlywireOverlay)
        with self.assertRaises(ValueError):
            DualTraceSynapses(self.graph,beta=-1,cells_per_feature=4)
        with self.assertRaises(ValueError):
            DualTraceSynapses(self.graph,fast_decay=1.0,cells_per_feature=4)


if __name__=="__main__":
    unittest.main()
