import pandas as pd
import pickle
from tqdm.auto import tqdm
from collections import defaultdict
from typing import List
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator, PandasTools
from rdkit import RDLogger; RDLogger.DisableLog('rdApp.*')

from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)

# ── Config ────────────────────────────────────────────────────────────────────
METHOD = 'map4'
MIN_SIM = 0.35 if METHOD == 'map4' else 0.76


# ── Per-row fingerprint function (top-level so pandarallel can pickle it) ─────
def _mol_to_fp(mol):
    if METHOD == "map4":
        from map4 import MAP4Calculator
        calc = MAP4Calculator(dimensions=1024)
        smi0 = Chem.MolToSmiles(mol, isomericSmiles=False)
        mol0 = Chem.MolFromSmiles(smi0)
        return np.array(calc.calculate(mol0)).astype(np.uint32)
    else:
        gen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=2048, includeChirality=False)
        arr = np.zeros(2048, dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(gen.GetFingerprint(mol), arr)
        return arr


# ── Fingerprint computation ───────────────────────────────────────────────────
def compute_fp_matrix(mol_series):
    mat = np.stack(mol_series.parallel_apply(_mol_to_fp))
    if METHOD == "atompair":
        mat = mat.astype(np.uint16, copy=False)
        return mat, mat.sum(axis=1, dtype=np.uint32)
    return mat, None


# ── Canonical map via connected components on perfect-match subgraph ──────────
def build_canonical_map(muid, rows, cols, vals):
    n       = len(muid)
    perfect = vals == 1.0
    adj     = sp.csr_matrix((np.ones(perfect.sum()), (rows[perfect], cols[perfect])), shape=(n, n))
    _, labels = connected_components(adj, directed=False)

    groups = defaultdict(list)
    for idx, lbl in enumerate(labels):
        groups[lbl].append(muid[idx])

    return {node: min(members) for members in groups.values() for node in members}


# ── Pack to upper-triangle CSR via scipy ──────────────────────────────────────
def pack_csr(muid, rows, cols, vals, canonical_map):
    canonical_nodes = sorted(set(canonical_map.values()))
    can_to_idx      = {n: i for i, n in enumerate(canonical_nodes)}
    N               = len(canonical_nodes)

    # Remap original indices to canonical indices
    can_rows = np.array([can_to_idx[canonical_map[muid[r]]] for r in rows], dtype=np.int32)
    can_cols = np.array([can_to_idx[canonical_map[muid[c]]] for c in cols], dtype=np.int32)

    # Build symmetric sparse matrix; take element-wise max across both triangles
    mat = sp.csr_matrix((vals, (can_rows, can_cols)), shape=(N, N))
    mat = mat.maximum(mat.T)

    # Extract upper triangle only
    mat = sp.triu(mat, k=1).tocsr()
    mat.sort_indices()

    return {
        "format":  "upper_csr_v1",
        "nodes":   canonical_nodes,
        "row_ptr": mat.indptr.astype(np.int32),
        "col_idx": mat.indices.astype(np.int32),
        "values":  mat.data.astype(np.float32),
    }


# ── Main ──────────────────────────────────────────────────────────────────────
# Recover MU smiles for assigned nodes 
df_fragments = pd.read_pickle(f"/home/mattia/geass/NPA_fragmentation/NPA_fragments_{METHOD}.pkl")
MUDB = pd.read_pickle("/home/mattia/geass/data/MUDB.pkl")
df_nodes = df_fragments[["Frag_ISOSMILES", "Frag_IDs"]].explode(["Frag_ISOSMILES", "Frag_IDs"])
df_nodes = df_nodes.merge(MUDB[["ID", "ISOSMILES"]].rename(columns={"ID": "Frag_IDs"}),
    on="Frag_IDs", how="left", suffixes=("", "_MU"))
df_nodes["Frag_ISOSMILES"] = df_nodes["ISOSMILES"].fillna(df_nodes["Frag_ISOSMILES"])
df_nodes.loc[df_nodes["Frag_IDs"] == "C00019", "Frag_ISOSMILES"] = "C"
df_nodes = df_nodes.drop_duplicates("Frag_IDs").drop(columns=["ISOSMILES"]).reset_index(drop=True)
del df_fragments
del MUDB

PandasTools.AddMoleculeColumnToFrame(df_nodes, 'Frag_ISOSMILES')
muid      = df_nodes["Frag_IDs"].tolist()
num_atoms = df_nodes["ROMol"].apply(lambda m: m.GetNumHeavyAtoms()).tolist()

node_weights        = {node: float(na) for node, na in zip(muid, num_atoms)}
fp_mat, fp_row_sums = compute_fp_matrix(df_nodes["ROMol"])
del df_nodes

# ── Atom-count buckets for candidate pruning ──────────────────────────────────
atom_buckets = defaultdict(list)
for idx, a in enumerate(num_atoms):
    atom_buckets[a].append(idx)

# ── Similarity loop ───────────────────────────────────────────────────────────
edge_rows: List[int]   = []
edge_cols: List[int]   = []
edge_vals: List[float] = []

for i in tqdm(range(len(muid)), desc="Pairwise similarity"):
    ai       = num_atoms[i]
    cand_idx = np.array([j for a in range(ai - 5, ai + 6)
                           for j in atom_buckets.get(a, ()) if j > i], dtype=np.int32)
    if cand_idx.size == 0:
        continue

    fp_i     = fp_mat[i]
    cand_fps = fp_mat[cand_idx]

    if METHOD == "map4":
        sims = (cand_fps == fp_i).mean(axis=1, dtype=np.float32)
    else:
        inter = cand_fps @ fp_i.astype(np.uint32)
        union = fp_row_sums[cand_idx].astype(np.uint32) + fp_row_sums[i] - inter
        sims  = (inter / np.maximum(union, 1)).astype(np.float32)

    for h in np.where(sims >= MIN_SIM)[0]:
        edge_rows.append(i)
        edge_cols.append(int(cand_idx[h]))
        edge_vals.append(float(sims[h]))

rows = np.array(edge_rows, dtype=np.int32);   del edge_rows
cols = np.array(edge_cols, dtype=np.int32);   del edge_cols
vals = np.array(edge_vals, dtype=np.float32); del edge_vals

# ── Canonicalisation ──────────────────────────────────────────────────────────
canonical_map = build_canonical_map(muid, rows, cols, vals)
csr           = pack_csr(muid, rows, cols, vals, canonical_map)
del rows, cols, vals

node_similarity = {
    "similar_nodes": csr,
    "canonical_map": canonical_map,
    "metadata": {"min_node_similarity": MIN_SIM, "representation": "upper_csr_v1"},
}

# ── Save ──────────────────────────────────────────────────────────────────────
with open("/home/mattia/geass/NPA_subgraphs/prep_files/NPA_node_weights.pkl", "wb") as f:
    pickle.dump(node_weights, f, protocol=pickle.HIGHEST_PROTOCOL)

with open(f"/home/mattia/geass/NPA_subgraphs/prep_files/NPA_node_similarity_{METHOD}.pkl", "wb") as f:
    pickle.dump(node_similarity, f, protocol=pickle.HIGHEST_PROTOCOL)

print("Done.")
