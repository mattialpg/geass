import unittest

from rdkit import Chem

from article.siderophore_fragmentation import (
    DEFAULT_TILE_SIZE,
    _grid_images,
    _multiplicity_legends,
    group_identical_fragments,
)


class SiderophoreFigureTests(unittest.TestCase):
    def test_default_tiles_reduce_horizontal_fragment_spacing(self):
        self.assertLess(DEFAULT_TILE_SIZE[0], 200)

    def test_groups_identical_structures_and_counts_stereoisomers_separately(self):
        fragments = [
            Chem.MolFromSmiles("CC"),
            Chem.MolFromSmiles("CC"),
            Chem.MolFromSmiles("CO"),
            Chem.MolFromSmiles("C[C@H](O)F"),
            Chem.MolFromSmiles("C[C@@H](O)F"),
        ]

        grouped = group_identical_fragments(fragments)

        self.assertEqual([count for _, count in grouped], [2, 1, 1, 1])

    def test_grid_legend_contains_only_multiplicity(self):
        fragments = [Chem.MolFromSmiles("CC"), Chem.MolFromSmiles("CO")]

        _, svg = _grid_images(fragments, [2, 1], per_row=2, tile=(120, 120))

        self.assertEqual(_multiplicity_legends([2, 1]), ["2x", "1x"])
        self.assertNotIn("atoms", svg)


if __name__ == "__main__":
    unittest.main()
