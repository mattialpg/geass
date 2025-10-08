import os, re, sys, threading
from itertools import combinations
from copy import deepcopy

import pandas as pd
import networkx as nx
import numpy as np
import yaml
import joblib

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs


current_file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_file_dir)
from utils import *

import logging
logger = logging.getLogger(__name__)


#* Nomenclature
#* fragment        >  mol with dummy atom(s)
#* building unit   >  fragment whose dummy atoms (*) have been replaced with real atoms 
#* building block  >  molecule formed by two or more building units

#* For consistency, reaction products must be written in parentheses and separated by a dot.
#* Molecules in the resulting Mol object are then separated by Chem.GetMolFrags.
#* In the reaction smarts, the breaking bond must always be between atom 0 and 1.

# Load reaction SMARTS patterns from YAML
reactions_path = os.path.join(current_file_dir, "reactions.yaml")
with open(reactions_path, "r") as f:
    reaction_smarts = yaml.safe_load(f)
    reactions = {k: AllChem.ReactionFromSmarts(v) for k, v in reaction_smarts.items()}


class Geass:
    _df_BUL = None
    _df_UUL = None
    _xgb_model = None

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.bonds_to_break = {}
        self.frag_mols = []
        self.atom_to_frag = {}
        self.G = nx.Graph()

        if Geass._df_BUL is None or Geass._df_UUL is None or Geass._xgb_model is None:
            Geass._initialize_resources()

    @classmethod
    def _initialize_resources(cls):
        try:
            cls._df_BUL = pd.read_pickle(f"{current_file_dir}/data/Database_BUL.pkl")
            cls._df_UUL = pd.read_pickle(f"{current_file_dir}/data/Database_UUL.pkl")
            cls._xgb_model = joblib.load(f"{current_file_dir}/data/xgb_classification.pkl")

            # Pre-cache NumPy representations for fast vector math
            cls._BUL_num_atoms = cls._df_BUL['Num_Atoms'].to_numpy()
            cls._BUL_features = np.stack(cls._df_BUL['Feature_vec'].values)
            cls._UUL_smiles = set(cls._df_UUL['ISOSMILES'].values)
        except Exception as e:
            print(f"Error loading resources: {e}")


    from functools import lru_cache
    @staticmethod
    @lru_cache(maxsize=50000)
    def _cached_vector_from_smiles(smiles):
        mol = Chem.MolFromSmiles(smiles)
        gen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=2048)
        fp_vec = gen.GetFingerprint(mol)
        arr = np.zeros((2048,), dtype=int)
        DataStructs.ConvertToNumpyArray(fp_vec, arr)
        desc_vec = np.array([
            rdMolDescriptors.CalcNumRings(mol),
            rdMolDescriptors.CalcNumAromaticRings(mol),
            rdMolDescriptors.CalcNumAliphaticRings(mol),
            sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 6 and a.GetIsAromatic()),
            sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 6 and not a.GetIsAromatic())
        ])
        return np.concatenate([desc_vec, arr])


    def find_best_match(self, mol):
        """
        Find the best matching building unit (BU) for a given molecule.
        Uses cached fingerprints, pre-cached database arrays,
        and vectorised NumPy operations for speed.
        """
        try:
            smiles = Chem.MolToSmiles(mol)
            fv_u = Geass._cached_vector_from_smiles(smiles)
            num_atoms_u = mol.GetNumHeavyAtoms()

            # Quickly filter BUL entries by atom-count proximity (NumPy-based)
            mask = np.abs(Geass._BUL_num_atoms - num_atoms_u) <= 5
            if not np.any(mask):
                return '', np.nan

            # Vectorised probability prediction
            fvs_filtered = Geass._BUL_features[mask]
            probs = Geass._xgb_model.predict_proba(np.abs(fvs_filtered - fv_u[None, :]))[:, 1]

            # Select highest-probability candidate
            best_idx = np.argmax(probs)
            bul_subset = Geass._df_BUL.iloc[np.flatnonzero(mask)]
            best_id = bul_subset.iloc[best_idx]['ID']
            best_prob = probs[best_idx]

            if best_id:
                return f"~{best_id}", best_prob
            return '', np.nan

        except Exception as e:
            if self.verbose:
                print(f"find_best_match error: {e}")
            return '', np.nan


    def check_BUL(self, mol):
        """Check if molecule has an exact match in the Building Unit Library (BUL)."""

        inchi = Chem.MolToInchi(mol)
        df_match = Geass._df_BUL.loc[Geass._df_BUL['InChI'] == inchi]

        if not df_match.empty:
            bu_id = df_match.iloc[0]['ID']
        else:
            bu_id = ""

        return bu_id
         

    def check_UUL(self, mol, threshold=0.86):
        """Check if molecule is in the Unknown Unit Library (UUL)."""

        smiles = Chem.MolToSmiles(mol)
        inchi = Chem.MolToInchi(mol)
        bu_id = ""

        # --- Special case for methyl group ---
        if smiles == "*C":
            return "Similar", "[C00019]"

        # --- Check for exact match in UUL ---
        df_match = Geass._df_UUL.loc[Geass._df_UUL['InChI'] == inchi]
        if len(df_match) > 0:
            prob = float(df_match.iloc[0]['Probability'])
            if prob >= 0.85:
                bu_id = f"~{df_match.iloc[0]['ID_Similar']}"

        return bu_id


    def annotate_fragment(self, mol, threshold=0.85):
        smiles = Chem.MolToSmiles(mol)

        bu_id = self.check_BUL(mol)
        if bu_id == "":
            bu_id = self.check_UUL(mol, threshold)
        if bu_id == "":
            bu_id, bu_prob = self.find_best_match(mol)
        
        if bu_id == "":
            isNP = "No"
        elif bu_id.startswith("~"):
            isNP = "Similar"
        else:
            isNP = "Yes"

        mol.SetProp("ISOSMILES", smiles)
        mol.SetProp("isNP", isNP)
        mol.SetProp("BU_ID", bu_id)


    def get_bonds_to_break(self, query_mol):
        self.bonds_to_break = {}
        for rxn_name, rxn in reactions.items():
            matches = query_mol.GetSubstructMatches(Chem.Mol(rxn.GetReactantTemplate(0).ToBinary()))
            for match in matches:
                self.bonds_to_break[match] = rxn_name
        return self.bonds_to_break


    def fragment_molecule(self, query_mol):

        for idx, atom in enumerate(query_mol.GetAtoms()):
            atom.SetIsotope(idx + 1)
            atom.SetProp('_protected', '1')
        final_prods = [query_mol]
        
        skipped_bonds = []
        for bond, rxn_name in self.bonds_to_break.items():
            bond_broken = False
            rxn = reactions[rxn_name]
            new_prods = []
            for i, prod in enumerate(final_prods):

                if bond_broken:
                    new_prods.append(prod)
                    continue

                # Check presence of bond in the molecule
                isotopes = [atom.GetIsotope() for atom in prod.GetAtoms()]
                if not all((i + 1) in isotopes for i in bond):
                    new_prods.append(prod)
                    continue

                # Unprotect atoms in the bond to be broken
                for atom in prod.GetAtoms():
                    if (atom.GetIsotope() - 1) in bond:
                        atom.ClearProp('_protected')
                
                inter = rxn.RunReactants((prod,))

                if len(inter) == 0:
                    new_prods.append(prod)
                    continue
                prod_new = inter[0][0]

                frags = Chem.GetMolFrags(prod_new, asMols=True, sanitizeFrags=True)
                new_prods.extend(frags)
                bond_broken = True

            final_prods = new_prods
        
            if bond_broken == False:
                skipped_bonds.append(bond)

        # Remove skipped bonds to avoid confusion in the connection graph
        if skipped_bonds:
            for b in skipped_bonds:
                if b in self.bonds_to_break:
                    del self.bonds_to_break[b]

        # Create a dictionary mapping query_mol atoms to fragment molecules
        for i, frag in enumerate(final_prods):
            for atom in frag.GetAtoms():
                iso = atom.GetIsotope()
                if iso:
                    self.atom_to_frag[iso - 1] = i
        self.frag_mols = final_prods
        return self.frag_mols


    # def combine_frags_with_timeout(self, isotope_prods, timeout):
    #     def target():
    #         try:
    #             for r in range(2, len(isotope_prods.keys()) + 1):
    #                 for combo in combinations(isotope_prods.keys(), r):
    #                     if sorted(flatten(combo)) == list(range(1, self.query_mol.GetNumAtoms() + 1)):
    #                         return [isotope_prods[x] for x in combo]  #<-- Arbitrarily choosing the first combo; TODO: fix
    #         except: pass

    #     timer = threading.Timer(timeout, lambda: None)
    #     thread = threading.Thread(target=target)

    #     thread.start()
    #     timer.start()

    #     thread.join(timeout)
    #     timer.cancel()

    #     if thread.is_alive():
    #         return []
    #     return target()


    def get_fragments(self, query_mol, check_unknowns=True, as_smiles=False):
        """
        Fragment a molecule into potential building units (BUs).
        Cleans, fragments, deduplicates, and classifies fragments.
        """

        # --- Reset transient state for a new molecule ---
        self.bonds_to_break.clear()
        self.frag_mols.clear()
        self.atom_to_frag.clear()
        self.G.clear()

        mol = Chem.Mol(query_mol)

        # --- Step 1: Prepare and sanitise molecule ---
        if '.' in Chem.MolToSmiles(mol):
            components = Chem.GetMolFrags(mol, asMols=True)
            components = [c for c in components if any(a.GetAtomicNum() == 6 for a in c.GetAtoms())]
            if len(components) != 1:
                print("Warning: Complex molecule with multiple components detected.")
                self.frag_mols = [query_mol]
                if as_smiles:
                    return [Chem.MolToSmiles(m) for m in self.frag_mols]
                return self.frag_mols
            mol = components[0]

        # Neutralise only if formal charges are present
        if any(a.GetFormalCharge() != 0 for a in mol.GetAtoms()):
            mol = neutralize_atoms(mol)

        # --- Step 3: Determine which reactions apply ---
        self.get_bonds_to_break(mol)
        if not self.bonds_to_break:
            self.annotate_fragment(mol)
            self.frag_mols = [mol]
            if as_smiles:
                return [Chem.MolToSmiles(m) for m in self.frag_mols]
            return self.frag_mols

        # --- Step 4: Label atoms with isotopes for tracking ---
        for i, atom in enumerate(mol.GetAtoms()):
            atom.SetIsotope(i + 1)

        # --- Step 5: Run fragmentation ---
        self.frag_mols = self.fragment_molecule(mol)

        # --- Step 6: Deduplicate fragments by canonical SMILES ---
        unique_frags = {}
        for frag in self.frag_mols:
            smi = Chem.MolToSmiles(frag)
            unique_frags[smi] = frag
        self.frag_mols = list(unique_frags.values())

        # # --- Step 7: Combine if isotopic coverage incomplete ---
        total_isotopes = {atom.GetIsotope() for atom in mol.GetAtoms() if atom.GetIsotope()}
        covered_isotopes = set().union(*[{atom.GetIsotope() for atom in f.GetAtoms()
            if atom.GetIsotope()} for f in self.frag_mols])

        if covered_isotopes != total_isotopes:
            print("Warning: Incomplete atom coverage; some fragments may be missing.")
        #     combined = self.combine_frags_with_timeout(unique_frags, timeout=10)
        #     if combined:
        #         self.frag_mols = combined

        # --- Step 8: Clean and annotate ---
        for frag in self.frag_mols:
            remove_isotopes(frag)
            self.annotate_fragment(frag, check_unknowns)

        if as_smiles:
            return [Chem.MolToSmiles(m) for m in self.frag_mols]
        return self.frag_mols


from visualisation import create_connection_graph, show_connection_graph
Geass.create_connection_graph = create_connection_graph
Geass.show_connection_graph = show_connection_graph