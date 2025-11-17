from pathlib import Path
import numpy as np
import pandas as pd, joblib
from functools import lru_cache
from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors, rdFingerprintGenerator


class AnnotationEngine:
    _df_MUDB = None
    _df_UUDB = None
    _xgb_model = None
    _MUDB_num_atoms = None
    _MUDB_feat_vec = None
    _UUDB_smiles = None

    @classmethod
    def initialize_resources(cls, base):
        base = Path(base)
        cls._df_MUDB = pd.read_pickle(base / "data/MUDB.pkl")
        cls._df_UUDB = pd.read_pickle(base / "data/UUDB.pkl")
        cls._xgb_model = joblib.load(base / "data/xgb_classification.pkl")
        cls._MUDB_num_atoms = cls._df_MUDB["Num_Atoms"].to_numpy()
        cls._MUDB_feat_vec = np.stack(cls._df_MUDB["Feature_vec"].values)
        cls._UUDB_smiles = set(cls._df_UUDB["ISOSMILES"].values)

    @staticmethod
    @lru_cache(maxsize=50000)
    def calc_feat_vector(mol):
        gen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=2048)
        fp_vec = gen.GetFingerprint(mol)
        arr = np.zeros((2048,), dtype=int)
        DataStructs.ConvertToNumpyArray(fp_vec, arr)
        descr_vec = np.array([
            rdMolDescriptors.CalcNumRings(mol),
            rdMolDescriptors.CalcNumAromaticRings(mol),
            rdMolDescriptors.CalcNumAliphaticRings(mol),
            sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 6 and a.GetIsAromatic()),
            sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 6 and not a.GetIsAromatic())])
        return np.concatenate([descr_vec, arr])

    @classmethod
    def annotate_fragment(cls, mol, threshold=0.85, verbose=False):
        """Annotate RDKit Mol with MU_ID and NP classification."""
        # Ensure resources are loaded before use
        if cls._df_MUDB is None or cls._df_UUDB is None or cls._xgb_model is None:
            base = Path(__file__).resolve().parent
            cls.initialize_resources(base)
            print("Annotation resources loaded.")

        smiles = Chem.MolToSmiles(mol)
        inchi = Chem.MolToInchi(mol)

        # Step 1: check MUDB for an exact InChI match
        mu_id = cls.check_MUDB(inchi)

        # Step 2: if not found, check UUDB for similarity
        if mu_id == "":
            mu_id = cls.check_UUDB(smiles, inchi, threshold)

        # Step 3: if still not found, use model to find best match
        if mu_id == "":
            mu_id, prob = cls.find_best_match(mol, threshold)

        # Step 4: assign NP classification
        if mu_id == "":
            isNP = "No"
        elif mu_id.startswith("~"):
            isNP = "Similar"
        else:
            isNP = "Yes"

        # Step 5: annotate RDKit Mol
        mol.SetProp("ISOSMILES", smiles)
        mol.SetProp("isNP", isNP)
        mol.SetProp("MU_ID", mu_id)
        # mol.SetProp("Original_atoms", self.atom_frag)

        return mol


    @classmethod
    def check_MUDB(cls, inchi):
        """Exact match in MUDB."""
        df_match = cls._df_MUDB.loc[cls._df_MUDB["InChI"] == inchi]
        if df_match.empty:
            return ""
        return df_match.iloc[0]["ID"]


    @classmethod
    def check_UUDB(cls, smiles, inchi, threshold):
        """Check similarity in UUDB."""
        if smiles in {"*C", "C*"}:
            return "<C00019>"
        df_match = cls._df_UUDB.loc[cls._df_UUDB["InChI"] == inchi]
        if not df_match.empty:
            prob = float(df_match.iloc[0]["Probability"])
            if prob >= threshold:
                return f"~{df_match.iloc[0]['ID_Similar']}"
        return ""


    @classmethod
    def find_best_match(cls, mol, threshold, verbose=False):
        """Predict best match using the XGB model."""
        try:
            num_atoms = mol.GetNumHeavyAtoms()
            mask = np.abs(cls._MUDB_num_atoms - num_atoms) < 6
            if not np.any(mask):
                return "", np.nan

            feat_vec_mol = cls.calc_feat_vector(mol)
            feat_vec_mus = cls._MUDB_feat_vec[mask]
            diffs = np.abs(feat_vec_mus - feat_vec_mol[None, :])
            probs = cls._xgb_model.predict_proba(diffs)[:, 1]

            best_idx = np.argmax(probs)
            subset = cls._df_MUDB.iloc[np.flatnonzero(mask)]
            best_id = subset.iloc[best_idx]["ID"]
            best_prob = probs[best_idx]

            if best_prob >= threshold:
                return f"~{best_id}", best_prob
            return "", np.nan
        except Exception as e:
            return "", np.nan

