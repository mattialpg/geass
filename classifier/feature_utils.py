import numpy as np
from rdkit import DataStructs
from rdkit.Chem import rdMolDescriptors, rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


_normalizer = rdMolStandardize.Normalizer()
_uncharger = rdMolStandardize.Uncharger()
_gen_ap = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=2048)

def standardize_mol(mol):
    if mol is None:
        return None
    mol = _normalizer.normalize(mol)
    mol = _uncharger.uncharge(mol)
    mol = rdMolStandardize.Cleanup(mol)
    return mol

def calc_feature_vec(mol):
    if mol is None:
        return None

    fp = _gen_ap.GetFingerprint(mol)
    fp_arr = np.zeros((2048,), dtype=np.int32)
    DataStructs.ConvertToNumpyArray(fp, fp_arr)

    desc_vec = np.array([
        rdMolDescriptors.CalcNumRings(mol),
        rdMolDescriptors.CalcNumAromaticRings(mol),
        rdMolDescriptors.CalcNumAliphaticRings(mol),
        sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 6 and a.GetIsAromatic()),
        sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 6 and not a.GetIsAromatic())
    ], dtype=np.int32)

    return np.concatenate([desc_vec, fp_arr])
