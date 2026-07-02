from pathlib import Path
import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import rdFingerprintGenerator
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# from functools import lru_cache
# from classifier.feature_utils import calc_feature_vec


_cache_fp = {}
def calculate_fp(mol, method):
    method = method.lower()

    if method == "atompair":
        key = (method, 2048)
        if key not in _cache_fp:
            _cache_fp[key] = rdFingerprintGenerator.GetAtomPairGenerator(
                fpSize=2048, includeChirality=False)
        fp = _cache_fp[key].GetFingerprint(mol)
        arr = np.zeros((2048,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr

    if method == "map4":
        from map4 import MAP4Calculator
        key = (method, 1024)
        if key not in _cache_fp:
            _cache_fp[key] = MAP4Calculator(dimensions=1024)
        smiles0 = Chem.MolToSmiles(mol, isomericSmiles=False)
        mol0 = Chem.MolFromSmiles(smiles0)
        fp = _cache_fp[key].calculate(mol0)
        return np.asarray(fp)


class AnnotationEngine:
    method = 'map4'
    df_MUDB = None
    _muid_by_inchikey = None
    _muid_ids = None

    @classmethod
    def initialize_resources(cls):
        if cls.df_MUDB is not None and getattr(cls, "_loaded_method", None) == cls.method:
            return

        base = Path(__file__).resolve().parent
        cls.df_MUDB = pd.read_pickle(base / 'data/MUDB.pkl')
        cls.MUDB_fps = np.load(base / f"data/MUDB_{cls.method}.npz")["fp"]
        cls.df_MUDB["FP"] = list(cls.MUDB_fps)

        assert len(cls.df_MUDB) == cls.MUDB_fps.shape[0]

        # Use a compact matrix for faster atompair operations
        if cls.method == "atompair":
            cls._fp_mat = cls.MUDB_fps.astype(np.uint16, copy=False)
            cls._fp_row_sums = cls._fp_mat.sum(axis=1, dtype=np.uint16)
        else:
            cls._fp_mat = cls.MUDB_fps
            cls._fp_row_sums = None
        cls._num_atoms = cls.df_MUDB["Num_Atoms"].to_numpy()
        cls._muid_ids = cls.df_MUDB["ID"].to_numpy(dtype=object)

        mudb_idx = (cls.df_MUDB)
        cls._muid_by_inchikey = dict(zip(mudb_idx["InChIKey"], mudb_idx["ID"]))
        cls._loaded_method = cls.method

    @classmethod
    def check_MUDB(cls, inchikey):
        return cls._muid_by_inchikey.get(inchikey, "")

    @classmethod
    def find_similar(cls, q_mol):
        q_atoms = q_mol.GetNumHeavyAtoms()
        mask = np.abs(cls._num_atoms - q_atoms) < 6
        if not np.any(mask):
            return "", 0.0

        fp_q = np.asarray(calculate_fp(q_mol, cls.method))

        cand = cls._fp_mat[mask]
        if cls.method == "map4":
            sims = (cand == fp_q).mean(axis=1)
        elif cls.method == "atompair":
            fp_q = fp_q.astype(np.uint16, copy=False)
            inter = cand @ fp_q
            q_sum = np.uint16(fp_q.sum(dtype=np.uint16))
            union = cls._fp_row_sums[mask] + q_sum - inter
            sims = inter / np.maximum(union, 1)

        best_idx_in_cand = int(np.argmax(sims))
        best_sim = float(sims[best_idx_in_cand])
        cand_idx = np.flatnonzero(mask)
        row_idx = cand_idx[best_idx_in_cand]

        return cls._muid_ids[row_idx], round(best_sim, 2)

    @classmethod
    def annotate_fragment(cls, mol, threshold):
        """ Annotate RDKit mol with molecular properties """
        smiles = Chem.MolToSmiles(mol)

        # Hard-coded special fragments
        special = {
            'C': 'C00019',
            '*C(C)=O': 'C00024',
            '*OS(=O)(=O)O': 'C00053',
            '*OP(=O)(O)O': 'C00002',
            '*OP(=O)(O)OP(=O)(O)O': 'C00002',
            '*OP(=O)(O)OP(=O)(O)OP(=O)(O)O': 'C00002',
            }
        if smiles in special.keys():
            mol.SetProp("ISOSMILES", smiles)
            mol.SetProp("isNP", "Yes")
            mol.SetProp("MUID", special[smiles])
            mol.SetProp("Similarity", "1.0")
            mol.SetProp("Annotation", "Special")
            return 

        # Step 1: check MUDB for an exact InChIKey match
        inchikey = Chem.MolToInchiKey(mol)
        muid = cls.check_MUDB(inchikey)
        isNP = "Yes"
        sim = 1.0
        ann = "Direct"

        # Step 2: if not found, check UUDB for similarity
        # if muid == "":
        #     muid = cls.checkUUDB(smiles, inchi, threshold)
        #     isNP = "Similar"

        # Step 3: if still not found, use model to find best match
        if muid == '':
            muid, sim = cls.find_similar(mol)
            ann = "Similarity"
            if sim == 1:
                isNP = 'Yes'
            elif sim >= threshold:
                isNP = 'Similar'
            else:
                muid = inchikey
                sim = 0.0
                isNP = 'No'
                ann = 'None'

        # Step 5: annotate RDKit Mol
        mol.SetProp("ISOSMILES", smiles)
        mol.SetProp("isNP", isNP)
        mol.SetProp("MUID", muid)
        mol.SetProp("Similarity", str(sim))
        mol.SetProp("Annotation", ann)
        # mol.SetProp("Original_atoms", self.atom_frag)
