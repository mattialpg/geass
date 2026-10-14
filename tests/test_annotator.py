import unittest
from unittest.mock import patch

from rdkit import Chem

from annotator import AnnotationEngine
from new_geass import Geass, mol2svg


class AnnotationPropertyTests(unittest.TestCase):
    def assert_annotation_fields(self, mol, expected_status):
        result = AnnotationEngine.annotate_fragment(mol, threshold=0.35)
        self.assertEqual(result["Annotation_status"], expected_status)
        self.assertNotIn("Annotation", result)

    def test_direct_annotation_returns_canonical_status_field(self):
        with patch.object(AnnotationEngine, "check_MUDB", return_value="C00001"):
            self.assert_annotation_fields(Chem.MolFromSmiles("CC"), "Direct")

    def test_special_annotation_returns_canonical_status_field(self):
        self.assert_annotation_fields(Chem.MolFromSmiles("C"), "Special")

    def test_unmatched_annotation_returns_canonical_status_field(self):
        with patch.object(AnnotationEngine, "check_MUDB", return_value=""):
            self.assert_annotation_fields(Chem.MolFromSmiles("*C"), "None")


class GeassAnnotationPropertyTests(unittest.TestCase):
    def test_connection_graph_maps_status_to_annotation_node_field(self):
        mol = Chem.MolFromSmiles("CC")
        mol.SetProp("ISOSMILES", "CC")
        mol.SetProp("MUID", "C00001")
        mol.SetProp("Similarity", "1.0")
        mol.SetProp("Annotation_status", "Special")
        mol.SetProp("isNP", "Yes")

        geass = Geass.__new__(Geass)
        geass.frag_mols = [mol]
        geass.bonds_to_break = {}
        geass.atom_frag = {}

        graph = geass.create_connection_graph(img=False)

        self.assertEqual(graph.nodes[0]["Annotation"], "Special")

    def test_special_svg_reads_annotation_status(self):
        mol = Chem.MolFromSmiles("C")
        mol.SetProp("Annotation_status", "Special")
        mol.SetProp("MUID", "C00019")

        mol2svg(mol)

        self.assertEqual(mol.GetAtomWithIdx(0).GetProp("atomLabel"), "RCH<sub>3</sub>")


if __name__ == "__main__":
    unittest.main()
