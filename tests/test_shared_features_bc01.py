"""Shared BC01 sensory cache extends canonical compressed L1 without changing it."""
from __future__ import annotations
import json
from pathlib import Path
import tempfile
import unittest
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from pretorius_connectome.shared_memory import export_l1, read_l1
from pretorius_connectome.shared_features_bc01 import (
    build_bc01_cache, load_bc01_cache, encode_bc_sensory,
)
from pretorius_connectome.imprinting import load_v12
from pretorius_connectome.associative import AssociativeMemory, Topology
from pretorius_connectome.pilot02 import episode_split

EVENTS=ROOT/"memories/current/Pretorius_v12_450_Events_Complete.jsonl"
SIDECARS=ROOT/"memories/annotations/v12_450_sidecars.jsonl"


class SharedBC01Features(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t=tempfile.TemporaryDirectory()
        cls.folder=Path(cls.t.name)
        export_l1(EVENTS,cls.folder)
        cls.original_l1={name:(cls.folder/name).read_bytes()
                         for name in ("manifest.json","pretorius_l1_v1.jsonl.gz")}
        cls.metadata=build_bc01_cache(cls.folder)

    @classmethod
    def tearDownClass(cls):
        cls.t.cleanup()

    def test_existing_canonical_L1_files_not_changed(self):
        for name,data in self.original_l1.items():
            self.assertEqual((self.folder/name).read_bytes(),data)
        m,values=load_bc01_cache(self.folder)
        rows=read_l1(self.folder/"pretorius_l1_v1.jsonl.gz",
                     self.folder/"manifest.json")
        self.assertEqual(len(rows),450)
        self.assertEqual(values.shape,(450,256))
        self.assertEqual(m["fit_event_ids"],[])
        self.assertEqual(m["record_ids_ordered"],[r["event_id"] for r in rows])

    def test_repeatable_bytes_and_feature_replay(self):
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td)
            export_l1(EVENTS,folder)
            second=build_bc01_cache(folder)
            self.assertEqual(self.metadata,second)
            for name in ("bc01_sensory_256.npy","bc01_l2_manifest.json"):
                self.assertEqual((self.folder/name).read_bytes(),
                                 (folder/name).read_bytes())
        _,values=load_bc01_cache(self.folder)
        rows=read_l1(self.folder/"pretorius_l1_v1.jsonl.gz",
                     self.folder/"manifest.json")
        for index in (0,50,449):
            np.testing.assert_array_equal(values[index],
                                          encode_bc_sensory(rows[index]["memory_text"]))

    def test_train_episode_holdouts_and_unchanged_FlyWire_scores(self):
        a=load_v12(EVENTS,SIDECARS)
        b=load_v12(self.folder/"pretorius_l1_v1.jsonl.gz",
                   SIDECARS,shared_manifest=self.folder/"manifest.json")
        self.assertEqual(a,b)
        train,validation,test=episode_split(b,31)
        self.assertFalse({m.episode_id for m in train}&
                         {m.episode_id for m in validation})
        self.assertFalse({m.episode_id for m in train}&
                         {m.episode_id for m in test})
        topo=Topology.synthetic(n=64,degree=3,seed=5)
        counts=topo.synapse_counts.copy()
        train_base,_,_=episode_split(a,31)
        one=AssociativeMemory(train_base,topo)
        two=AssociativeMemory(train,topo)
        for mode in ("lexical","graph","hybrid"):
            np.testing.assert_array_equal(one.score("cathedral laboratory",mode),
                                          two.score("cathedral laboratory",mode))
        np.testing.assert_array_equal(topo.synapse_counts,counts)

    def test_reject_bad_shard_and_fit_manifest(self):
        manifest=self.folder/"bc01_l2_manifest.json"
        original=manifest.read_bytes()
        try:
            payload=json.loads(original)
            payload["fit_event_ids"]=["E01-001"]
            manifest.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                load_bc01_cache(self.folder)
        finally:
            manifest.write_bytes(original)
        shard=self.folder/"bc01_sensory_256.npy"
        raw=shard.read_bytes()
        try:
            shard.write_bytes(raw+b"attack")
            with self.assertRaises(ValueError):
                load_bc01_cache(self.folder)
        finally:
            shard.write_bytes(raw)


if __name__=="__main__":
    unittest.main()
