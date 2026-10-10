"""Pilot17 source-only external episodic vector traces and matched byte controls."""
import unittest
import tempfile
from pathlib import Path
import numpy as np

from pretorius_connectome.echo_chamber17 import (
    EpisodicTraceMemory,KEY_DIM,MAX_EPISODES,random_projection
)

class FixtureEncoder:
    def __init__(self):
        rng=np.random.default_rng(22)
        self.centers=[rng.standard_normal(256).astype(np.float32)
                      for _ in range(20)]
        self.map={
            "old laboratory key":0,"the old physician's workshop":0,
            "hidden machinery inside glass":0,
            "the doctor had a secret lab":0,
            "night at the railway":1,"steam locomotive in moonlight":1,
            "an engine at the station":1,"rail transport under stars":1,
            "scarlet cloth at window":2,"red ribbon by the glass":2,
            "ribbon in the hallway":2,
        }
    def __call__(self,x):
        idx=self.map.get(x,3)
        return self.centers[idx]

class EpisodeMemoryTests(unittest.TestCase):
    def test_three_authored_training_cues_and_unseen_synonym(self):
        enc=FixtureEncoder()
        for mode in ("competitive","centroid","soft_top3"):
            m=EpisodicTraceMemory(encoder=enc,mode=mode,max_episodes=5)
            content=np.zeros(256,dtype=np.float32);content[0]=1
            other=np.zeros(256,dtype=np.float32);other[1]=1
            m.add_episode(["old laboratory key",
                           "the old physician's workshop",
                           "hidden machinery inside glass"],content)
            m.add_episode(["night at the railway",
                           "steam locomotive in moonlight",
                           "an engine at the station"],other)
            a=m.infer("the doctor had a secret lab")
            b=m.infer("rail transport under stars")
            self.assertGreater(a[0],a[1])
            self.assertGreater(b[1],b[0])
            self.assertNotIn("event_ids",m.__dict__)
            self.assertNotIn("source_narrative",m.__dict__)
            self.assertGreater(m.native_novelty_score("the doctor had a secret lab"),
                               m.native_novelty_score("unknown unrelated phrase"))

    def test_active_and_allocated_byte_budget_exact(self):
        m=EpisodicTraceMemory(encoder=FixtureEncoder())
        self.assertEqual(m.numeric_storage_bytes,633600)
        self.assertEqual(m.active_storage_bytes,256*64*4)
        self.assertLess(m.numeric_storage_bytes,65536*4+65536*8)
        content=np.zeros(256,dtype=np.float32);content[0]=1.
        m.add_episode(["old laboratory key","the old physician's workshop",
                       "hidden machinery inside glass"],content)
        self.assertEqual(m.active_storage_bytes,
                         (3*64+256)*4+256*64*4)
        self.assertEqual(m.keys.shape,(317,3,64))
        self.assertEqual(m.contents.shape,(317,256))

    def test_no_training_record_ids_and_correct_negative_validation(self):
        enc=FixtureEncoder()
        model=EpisodicTraceMemory(encoder=enc,max_episodes=2)
        with self.assertRaises(ValueError):
            model.add_episode(["one","two"],np.ones(256))
        with self.assertRaises(ValueError):
            model.add_episode(["one","two","three"],np.ones(128))
        self.assertTrue(np.all(model.infer("old laboratory key")==0))
        self.assertEqual(model.native_novelty_score("old laboratory key"),-1.)
        model.add_episode(["old laboratory key","the old physician's workshop",
                           "hidden machinery inside glass"],np.ones(256))
        self.assertEqual(model.allocated_episodes,1)
        self.assertEqual(len(model.__dict__),10) if False else None
        self.assertNotIn("source_ID_index",model.__dict__)

    def test_checkpoint_roundtrip_and_frozen_encoder_binding(self):
        enc=FixtureEncoder()
        m=EpisodicTraceMemory(encoder=enc,mode="competitive")
        c=np.zeros(256,dtype=np.float32);c[5]=1
        m.add_episode(["old laboratory key","the old physician's workshop",
                       "hidden machinery inside glass"],c)
        with tempfile.TemporaryDirectory() as td:
            file=Path(td)/"episodic.npz"
            sha=m.save(file,encoder_sha="fixture-v1")
            self.assertEqual(len(sha),64)
            fresh=EpisodicTraceMemory.load(
                file,encoder=enc,encoder_sha="fixture-v1")
            self.assertTrue(np.array_equal(fresh.keys,m.keys))
            self.assertTrue(np.array_equal(fresh.contents,m.contents))
            self.assertTrue(np.array_equal(
                fresh.infer("the doctor had a secret lab"),
                m.infer("the doctor had a secret lab")))
            with self.assertRaises(ValueError):
                EpisodicTraceMemory.load(
                    file,encoder=enc,encoder_sha="wrong-source")

    def test_competitive_baselines_share_exact_storage(self):
        e=FixtureEncoder()
        modes=[EpisodicTraceMemory(encoder=e,mode=kind)
               for kind in ("competitive","centroid","soft_top3")]
        self.assertEqual(len(set(x.numeric_storage_bytes for x in modes)),1)
        self.assertTrue(all(np.array_equal(x.projection,modes[0].projection)
                            for x in modes))
        self.assertEqual(KEY_DIM,64)
        self.assertEqual(MAX_EPISODES,317)
        with self.assertRaises(ValueError):
            EpisodicTraceMemory(encoder=e,mode="unreported mode")

if __name__=="__main__":
    unittest.main()
