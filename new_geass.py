import os, sys
from pathlib import Path
import pandas as pd
import networkx as nx
import numpy as np

from rdkit import Chem

workdir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(workdir)
from utils import *

from annotation import AnnotationEngine
from fragmentation import FragmentationEngine

#* Nomenclature
#* fragment        >  mol with dummy atom(s)
#* building unit   >  fragment whose dummy atoms (*) have been replaced with real atoms 
#* building block  >  molecule formed by two or more building units


class Geass:
    _resources_loaded = False

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.mol = None
        self.bonds_to_break = {}
        self.frag_mols = []
        self.atom_frag = {}
        self.G = nx.Graph()

        # Initialise engines
        self.annotator = AnnotationEngine()
        self.fragmenter = FragmentationEngine(workdir)

    def perpare_molecule(self):
        """
        Prepare and sanitise a molecule for fragmentation.
        Neutralises formal charges if present.
        Splits multi-component molecules, retaining only the largest organic component.
        """
        if '.' in Chem.MolToSmiles(self.mol):
            components = Chem.GetMolFrags(self.mol, asMols=True)
            components = [c for c in components if any(a.GetAtomicNum() == 6 for a in c.GetAtoms())]
            if len(components) != 1:
                print("Warning: Complex molecule with multiple components detected.")
                self.frag_mols = [self.mol]
                return self.frag_mols
            self.mol = components[0]

        # Neutralise only if formal charges are present
        if any(a.GetFormalCharge() != 0 for a in self.mol.GetAtoms()):
            self.mol = neutralize_atoms(self.mol)


    def get_fragments(self, query_mol, as_smiles=False, annotate=True):
        """
        Fragment a molecule into potential metabolic units (MUs).
        Cleans, fragments, deduplicates, and classifies fragments.
        """
        # --- Reset transient state for a new molecule ---
        self.mol = None
        self.bonds_to_break.clear()
        self.frag_mols.clear()
        self.atom_frag.clear()
        self.G.clear()

        # --- Step 1: Prepare molecule ---
        self.mol = Chem.Mol(query_mol)
        # self.perpare_molecule()

        # --- Step 2: Determine which reaction apply ---
        self.bonds_to_break = self.fragmenter.get_bonds_to_break(self.mol)
        if not self.bonds_to_break:
            self.annotator.annotate_fragment(self.mol)
            if as_smiles:
                return Chem.MolToSmiles(self.mol)
            return [self.mol]

        # --- Step 3: Run fragmentation ---
        self.frag_mols, self.atom_frag = self.fragmenter.fragment_molecule(self.mol)

        # # --- Step 4: Combine if isotopic coverage incomplete ---
        # total_isotopes = {atom.GetIsotope() for atom in self.mol.GetAtoms() if atom.GetIsotope()}
        # covered_isotopes = set().union(*[{atom.GetIsotope() for atom in f.GetAtoms()
        #     if atom.GetIsotope()} for f in self.frag_mols])

        # if covered_isotopes != total_isotopes:
        #     print("Warning: Incomplete atom coverage; some fragments may be missing.")
        #     combined = self.combine_frags_with_timeout(unique_frags, timeout=10)
        #     if combined:
        #         self.frag_mols = combined

        # --- Step 5: Clean and annotate ---
        for frag in self.frag_mols:
            remove_isotopes(frag)
            if annotate:
                self.annotator.annotate_fragment(frag)

        if as_smiles:
            return [Chem.MolToSmiles(m) for m in self.frag_mols]
        return self.frag_mols


from visualisation import create_connection_graph, show_connection_graph
Geass.create_connection_graph = create_connection_graph
Geass.show_connection_graph = show_connection_graph