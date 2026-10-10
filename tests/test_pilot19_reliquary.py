"""Reliquary19 exact original provenance and explicit source-index exposure tests."""
import unittest
from pretorius_connectome.reliquary19 import (
    ARMS,Reliquary,normalize
)

def record(id,text,participant="Pretorius"):
    return {"event_id":id,"episode_id":"TEST","memory_text":text,
            "participants":[participant],"locations":["Laboratory"]}
def cue(id,items):
    return {"event_id":id,"annotation_status":"unreviewed_candidate",
            "cue_surface_forms":list(items)}

class EventProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.records=[
            record("one","He carried a millstream map in the office."),
            record("two","The cracked lens was beside a beetle in the drawer."),
            record("three","A mercury barometer tipped near the stove."),
            record("four","A glass vessel struck the narrow bench."),
        ]
        self.cues=[
            cue("one",["turned glove","millstream map","green tape"]),
            cue("two",["camphor","beetle","cracked lens"]),
            cue("three",["barometer","mercury","stove"]),
            cue("four",["glass vessel","bench","sawdust"])
        ]
    def test_flat_exact_and_graph_use_same_original_cue_evidence(self):
        flat=Reliquary(self.records,self.cues,arm="all_anchors_exact_index")
        graph=Reliquary(self.records,self.cues,arm="reliquary_event_graph")
        for query,expected in [("green tape","one"),("cracked lens","two"),
                               ("stove","three"),("sawdust","four")]:
            self.assertEqual(flat.infer(query)["event_id"],expected)
            self.assertEqual(graph.infer(query)["event_id"],expected)
            self.assertEqual(graph.infer(query)["evidence"]["source_event_id"],expected)
        proof=graph.infer("green tape")["evidence"]
        self.assertEqual(proof["source_annotation_status"],"unreviewed_candidate")
        self.assertEqual(proof["associated_other_detail"],"turned glove")
        self.assertEqual(proof["original_event_locations"],["Laboratory"])
        self.assertIn("source_episode",proof["graph_path_types"][1])
    def test_third_cue_not_available_when_only_first_and_middle_indexed(self):
        train=Reliquary(self.records,self.cues,arm="train_two_anchors_only")
        self.assertIsNone(train.infer("green tape")["event_id"])
        self.assertIsNone(train.infer("cracked lens")["event_id"])
        self.assertFalse(train.indexed_exposure("green tape","one")[
            "cue_in_own_RETRIEVAL_INDEX"])
        self.assertTrue(train.indexed_exposure("green tape","one")[
            "cue_available_in_own_source_original_sidecar"])
        self.assertFalse(train.indexed_exposure("green tape","one")[
            "cue_verbatim_in_own_original_narrative"])
    def test_original_memory_narrative_exposure_is_separate_from_sidecar(self):
        narrative=Reliquary(self.records,self.cues,arm="narrative_bm25")
        self.assertEqual(narrative.infer("millstream map")["event_id"],"one")
        self.assertIsNone(narrative.infer("green tape")["event_id"])
        self.assertEqual(narrative.infer("cracked lens")["event_id"],"two")
    def test_wrong_ownership_never_masquerades_as_true_source_provenance(self):
        wrong=Reliquary(self.records,self.cues,arm="wrong_owner_graph")
        result=wrong.infer("green tape")
        self.assertIsNotNone(result["event_id"])
        self.assertNotEqual(result["event_id"],"one")
        self.assertEqual(result["evidence"]["source_event_id"],"one")
        self.assertEqual(result["evidence"]["source_annotation_status"],
                         "unreviewed_candidate")
    def test_no_test_or_absent_records_are_indexed(self):
        train=Reliquary(self.records[:2],self.cues[:2],
                        arm="reliquary_event_graph")
        self.assertEqual(train.indexed_source_event_count,2)
        self.assertIsNone(train.infer("mercury")["event_id"])
        self.assertNotIn("three",train.events)
    def test_ambiguous_exact_source_cue_abstains(self):
        changed=[self.cues[0],
                 cue("two",["camphor","beetle","green tape"]),
                 self.cues[2],self.cues[3]]
        index=Reliquary(self.records,changed,arm="all_anchors_exact_index")
        z=index.infer("green tape")
        self.assertTrue(z["ambiguous"])
        self.assertIsNone(z["event_id"])
        self.assertEqual(z["confidence"],1.)
    def test_reject_empty_source_cue_and_metadata_drift(self):
        with self.assertRaises(ValueError):
            Reliquary(self.records,self.cues,arm="fabricated_model")
        with self.assertRaises(ValueError):
            Reliquary(self.records,self.cues[:3],arm="reliquary_event_graph")
        model=Reliquary(self.records,self.cues,arm="all_anchors_exact_index")
        with self.assertRaises(ValueError):
            model.infer("   ")
        wrong=self.cues.copy()
        wrong[0]={**wrong[0],"annotation_status":"independently_verified"}
        with self.assertRaises(ValueError):
            Reliquary(self.records,wrong,arm="all_anchors_exact_index")
    def test_flat_same_evidence_can_retrieve_present_candidate(self):
        bm=Reliquary(self.records,self.cues,arm="all_anchors_flat_bm25")
        self.assertEqual(bm.infer("green tape")["event_id"],"one")
        self.assertEqual(bm.infer("cracked lens")["event_id"],"two")
        self.assertEqual(bm.stored_provenance()["index_document_token_count"],
                         sum(len(normalize(" ".join(v["cue_surface_forms"])).split())
                             for v in self.cues))
        self.assertEqual(set(ARMS),set((
            "train_two_anchors_only","narrative_bm25",
            "all_anchors_flat_bm25","all_anchors_exact_index",
            "reliquary_event_graph","wrong_owner_graph")))

if __name__=="__main__":
    unittest.main()
