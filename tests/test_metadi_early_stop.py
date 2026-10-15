import sys
import math
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from rdkit import Chem

from article.siderophore_fragmentation import decompose
from metadisassembler import MetaDisassembler


class MetaDiEarlyStopTests(unittest.TestCase):
    def test_optimizer_remains_exhaustive_by_default(self):
        md = MetaDisassembler()
        md.history = [{"source": 0}]
        md.cpds = [SimpleNamespace(mol=Chem.MolFromSmiles("CC")) for _ in range(3)]
        first = {"target": [1], "cut_bond_id": [3]}
        later = {"target": [2], "cut_bond_id": [4]}
        md.seed_list = [first, later]
        md.relation = []
        md.result = []
        md.frag_limit = 10

        md._optimize_seed()

        self.assertEqual(md.result, [first, later])

    def test_optimizer_keeps_only_first_complete_tree(self):
        md = MetaDisassembler()
        md.history = [{"source": 0}]
        md.cpds = [SimpleNamespace(mol=Chem.MolFromSmiles("CC")) for _ in range(3)]
        first = {"target": [1], "cut_bond_id": [3]}
        later = {"target": [2], "cut_bond_id": [4]}
        md.seed_list = [first, later]
        md.relation = []
        md.result = []
        md.frag_limit = 10
        md.stop_after_first = True

        md._optimize_seed()

        self.assertEqual(md.result, [first])

    def test_optimizer_skips_filtered_trees_before_stopping(self):
        md = MetaDisassembler()
        md.history = [{"source": 0}]
        md.cpds = [SimpleNamespace(mol=Chem.MolFromSmiles("C=CO")) for _ in range(3)]
        rejected = {"target": [1], "cut_bond_id": [3]}
        accepted = {"target": [2], "cut_bond_id": [4]}
        md.seed_list = [rejected, accepted]
        md.relation = []
        md.result = []
        md.frag_limit = 10
        md.stop_after_first = True

        with patch.object(md, "_has_unnatural_carbon", side_effect=[True, False]):
            md._optimize_seed()

        self.assertEqual(md.result, [accepted])

    def test_geass_hard_codes_first_tree_mode(self):
        class FakeMetaDisassembler:
            instance = None

            def __init__(self):
                self.result = [{}]
                self.calls = []
                self.frag_limit = None
                FakeMetaDisassembler.instance = self

            def input_query(self, smiles):
                return True

            def _initialize(self):
                self.frag_limit = 12
                return True

            def disassemble(self, **kwargs):
                self.calls.append(kwargs)
                self._initialize()
                return [Chem.MolToInchi(Chem.MolFromSmiles("CC"))]

        module = SimpleNamespace(MetaDisassembler=FakeMetaDisassembler)

        with patch.dict(sys.modules, {"metadisassembler": module}):
            fragments, _ = decompose("MetaDi", Chem.MolFromSmiles("CC"), "CC")

        md = FakeMetaDisassembler.instance
        self.assertEqual(len(fragments), 1)
        self.assertTrue(md.stop_after_first)
        self.assertEqual(md.frag_limit, 30)
        self.assertEqual(md.calls, [{"return_inchis": True, "result_id": 0, "write_output": False}])
        self.assertEqual(md.time_limit, math.inf)


if __name__ == "__main__":
    unittest.main()
