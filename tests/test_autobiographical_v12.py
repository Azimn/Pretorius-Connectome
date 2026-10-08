"""Regression test for the v12 full corpus and its sensory index."""
import unittest
from scripts.validate_autobiographical_corpus_v12 import validate_v12

class V12CorpusTest(unittest.TestCase):
    def test_v12_integrity(self):
        validate_v12()
