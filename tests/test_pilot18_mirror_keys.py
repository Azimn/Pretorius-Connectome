"""Pilot18 train-only semantic addressing source and capacity safeguards."""
from __future__ import annotations
import tempfile
import unittest
from pathlib import Path
import numpy as np
from pretorius_connectome.mirror_keys18 import (
    TwoViewEpisodicMemory,fit_train_only_projection,
    PROJECTION_REGULARIZATION,PAIR_PENALTY
)

class SourceEncoder:
    def __init__(self):
        rng=np.random.default_rng(180)
        self.classes=rng.normal(size=(12,256)).astype(np.float32)
        self.calls=[]
        self.names={
            "physician workshop":0,"doctor laboratory":0,
            "the scientist entered his lab":0,
            "railway crossing":1,"steam locomotive":1,
            "rail transport beneath moonlight":1,
            "ribbon by glass":2,"red cloth beside window":2,
            "a scarlet fabric by the pane":2,
        }
    def __call__(self,cue):
        self.calls.append(cue)
        ix=self.names.get(cue,11)
        return self.classes[ix]

class TrainOnlyMetricTests(unittest.TestCase):
    def setUp(self):
        self.encoder=SourceEncoder()
        rng=np.random.default_rng(18)
        bases=rng.normal(size=(15,256))
        self.X=np.stack([
            np.stack([bases[i]+rng.normal(size=256)*.15,
                      bases[i]+rng.normal(size=256)*.15])
            for i in range(15)]).astype(np.float32)
        self.content=np.eye(256,dtype=np.float32)

    def test_fitting_uses_only_supplied_two_cues_and_is_deterministic(self):
        for mode in ("random","pair_discriminant","mispaired_discriminant","pca"):
            q=fit_train_only_projection(self.X,algorithm=mode)
            repeat=fit_train_only_projection(self.X,algorithm=mode)
            self.assertEqual(q.shape,(256,64))
            self.assertTrue(np.array_equal(q,repeat))
            self.assertTrue(np.isfinite(q).all())
            self.assertTrue(np.allclose(q.T@q,np.eye(64),atol=1e-4))
        with self.assertRaises(ValueError):
            fit_train_only_projection(np.zeros((15,3,256)),algorithm="pair_discriminant")

    def test_third_source_cue_unavailable_to_numeric_train_slots(self):
        for mode in ("random","pair_discriminant"):
            # Previous iteration's *inference* intentionally encoded the
            # heldout query; clear instrumentation before next training.
            self.encoder.calls.clear()
            model=TwoViewEpisodicMemory(
                encoder=self.encoder,
                projection=fit_train_only_projection(self.X,algorithm=mode),
                max_episodes=5)
            model.add_episode(["physician workshop","doctor laboratory"],
                              self.content[0])
            model.add_episode(["railway crossing","steam locomotive"],
                              self.content[1])
            self.assertNotIn("the scientist entered his lab",
                             self.encoder.calls)
            self.assertTrue(np.all(model.keys[:,2,:]==0))
            unseen=model.infer("the scientist entered his lab")
            self.assertGreater(float(unseen[0]),float(unseen[1]))
            self.assertEqual(model.numeric_storage_bytes,
                5*(3*64+256)*4+256*64*4)
            self.assertNotIn("event_ids",model.__dict__)
            self.assertNotIn("memory_narratives",model.__dict__)

    def test_metric_training_within_pair_labels_changes_projection(self):
        source=fit_train_only_projection(self.X,algorithm="pair_discriminant")
        deranged=fit_train_only_projection(self.X,algorithm="mispaired_discriminant")
        random=fit_train_only_projection(self.X,algorithm="random")
        self.assertFalse(np.array_equal(source,deranged))
        self.assertFalse(np.array_equal(source,random))
        self.assertEqual(PAIR_PENALTY,2.)
        self.assertEqual(PROJECTION_REGULARIZATION,.002)

    def test_source_bound_checkpoint_roundtrip_and_no_three_key_leak(self):
        q=fit_train_only_projection(self.X,algorithm="pair_discriminant")
        enc=self.encoder
        model=TwoViewEpisodicMemory(encoder=enc,projection=q,max_episodes=4)
        model.add_episode(["physician workshop","doctor laboratory"],
                          self.content[0])
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"source-episode.npz"
            digest=model.save(path,encoder_sha="test-onnx-sha",
                              algorithm="pair_discriminant",
                              source_sha="source-bc01")
            self.assertEqual(len(digest),64)
            reread=TwoViewEpisodicMemory.load(
                path,encoder=enc,encoder_sha="test-onnx-sha",
                source_sha="source-bc01")
            self.assertTrue(np.array_equal(reread.keys,model.keys))
            self.assertTrue(np.array_equal(reread.contents,model.contents))
            self.assertTrue(np.array_equal(reread.projection,model.projection))
            with self.assertRaises(ValueError):
                TwoViewEpisodicMemory.load(path,encoder=enc,
                                           encoder_sha="wrong",source_sha="source-bc01")
            with np.load(path,allow_pickle=False) as z:
                self.assertTrue(np.all(z["keys"][:,2,:]==0))

    def test_unseen_third_is_excluded_from_unique_source_view_protocol(self):
        from scripts.run_pilot18_mirror_keys import withheld_third
        from unittest.mock import patch
        with patch("scripts.run_pilot18_mirror_keys.binding_cues",
                   return_value=("first","middle","middle")):
            self.assertIsNone(withheld_third(object()))
        with patch("scripts.run_pilot18_mirror_keys.binding_cues",
                   return_value=("first","middle","third")):
            self.assertEqual(withheld_third(object()),"third")
        with patch("scripts.run_pilot18_mirror_keys.binding_cues",
                   return_value=("first","MIDDLE","middle")):
            self.assertIsNone(withheld_third(object()))

if __name__=="__main__":
    unittest.main()
