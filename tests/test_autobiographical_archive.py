"""Archive invariants, including the pre-1899 stop and preserved v9 memories."""
import unittest
from scripts.validate_autobiographical_corpus import validate

class TestAutobiographicalArchive(unittest.TestCase):
    def test_400_memory_checkpoint(self):
        validate()
