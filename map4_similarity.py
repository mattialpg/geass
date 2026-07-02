import os
import sys
import tempfile

import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from numba import njit, prange
from scipy import sparse
from rdkit.Chem import PandasTools
from pandarallel import pandarallel


MAP4_INPUT = "/home/mattia/geass/NPA_fragmentation/NPA_fragments_map4.pkl"
SIM_OUTPUT_BASE = "/home/mattia/geass/NPA_analysis/NPA_sim_matrix_map4"
SPARSE_THRESHOLD = 0.05

BLOCK_SIZE = 512       # larger blocks amortise numba call overhead
N_JOBS = -1        # use all cores; set to e.g. 8 to cap
CUTS = np.round(np.linspace(0.20, 0.90, 15), 2)

pandarallel.initialize(progress_bar=True)
sys.path.append("/home/mattia/")
from aiddtools import fingerprint_tools as fpt


# ── Numba kernel ──────────────────────────────────────────────────────────────
# Computes MAP4 Tanimoto (= fraction of equal positions) between two blocks.
# No intermediate 3-D tensor: accumulates directly into a float32 output array.
@njit(parallel=True, cache=True)
def _map4_sim_block(block_i, block_j, out):
    """out[a, b] = mean(block_i[a] == block_j[b])"""
    ni, d = block_i.shape
    nj    = block_j.shape[0]
    inv_d = np.float32(1.0 / d)
    for a in prange(ni):
        for b in range(nj):
            count = np.uint32(0)
            for k in range(d):
                if block_i[a, k] == block_j[b, k]:
                    count += 1
            out[a, b] = np.float32(count) * inv_d


# ── Worker called by joblib ───────────────────────────────────────────────────
def _fill_block(i0, j0, n, fps, memmap_path):
    i1 = min(i0 + BLOCK_SIZE, n)
    j1 = min(j0 + BLOCK_SIZE, n)

    out = np.empty((i1 - i0, j1 - j0), dtype=np.float32)
    _map4_sim_block(fps[i0:i1], fps[j0:j1], out)

    # Write directly into the on-disk memmap (each block touches disjoint cells)
    mat = np.memmap(memmap_path, dtype=np.float32, mode="r+", shape=(n, n))
    mat[i0:i1, j0:j1] = out
    if i0 != j0:
        mat[j0:j1, i0:i1] = out.T
    mat.flush()


# ── Fingerprints ──────────────────────────────────────────────────────────────
def compute_map4():
    df = pd.read_pickle(MAP4_INPUT)
    PandasTools.AddMoleculeColumnToFrame(df, "ISOSMILES")
    df["FP_map4"] = df["ROMol"].parallel_apply(
        lambda mol: fpt.calculate_fp(mol, "map4", as_numpy=True))
    df = df[["NPA_ID", "FP_map4"]]
    df = df[df["FP_map4"].notna()].reset_index(drop=True)

    ids = df["NPA_ID"].to_numpy()
    fps = np.asarray(df["FP_map4"].tolist(), dtype=np.uint32)
    return ids, fps


# ── Similarity matrix ─────────────────────────────────────────────────────────
def build_similarity_matrix(fps):
    n = fps.shape[0]

    # Warm up numba (compile once on a tiny slice before spawning workers)
    _dummy = np.empty((2, 2), dtype=np.float32)
    _map4_sim_block(fps[:2], fps[:2], _dummy)

    # Write the matrix to a memmap so workers share it without copying to RAM
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mmap")
    tmp.close()
    mat = np.memmap(tmp.name, dtype=np.float32, mode="w+", shape=(n, n))
    np.fill_diagonal(mat, 1.0)
    mat.flush()
    del mat   # workers will re-open it

    # Upper-triangle block pairs (diagonal included for off-diagonal symmetry)
    blocks = [(i0, j0)
              for i0 in range(0, n, BLOCK_SIZE)
              for j0 in range(i0, n, BLOCK_SIZE)]

    Parallel(n_jobs=N_JOBS, prefer="threads")(
        delayed(_fill_block)(i0, j0, n, fps, tmp.name)
        for i0, j0 in tqdm(blocks, desc="Similarity blocks")
    )

    # Load the completed matrix back into RAM
    sim_matrix = np.array(np.memmap(tmp.name, dtype=np.float32, mode="r", shape=(n, n)))
    os.unlink(tmp.name)
    return sim_matrix


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ids, fps = compute_map4()
    sim_matrix = build_similarity_matrix(fps)

    np.clip(sim_matrix, 0.0, 1.0, out=sim_matrix)
    if SPARSE_THRESHOLD > 0.0:
        sim_matrix[sim_matrix < SPARSE_THRESHOLD] = 0.0
    np.fill_diagonal(sim_matrix, 1.0)

    sim_csr = sparse.csr_matrix(sim_matrix.astype(np.float32, copy=False))
    sim_csr.eliminate_zeros()

    sparse.save_npz(f"{SIM_OUTPUT_BASE}.npz", sim_csr, compressed=True)
    np.save(f"{SIM_OUTPUT_BASE}_ids.npy", np.asarray(ids))

if __name__ == "__main__":
    main()
