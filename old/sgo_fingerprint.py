"""
sgo_fingerprint.py
==================
Fingerprint-based approximation of the sgo graph similarity metric.

Each molecule is encoded once into an SgoFingerprint dict; pairwise
scoring then requires no graph traversal and no similarity-database
lookups — only sparse dict operations.

Design
------
Three sparse vector components are precomputed per molecule:

  fn_soft : {label -> float}
      Soft node fingerprint.  For each node i with label l_i and size
      s_i, contributes s_i * σ(l_i, f_k) to fn_soft[f_k] for every
      fragment type f_k with σ >= NODE_MIN_SIMILARITY (self included,
      σ = 1).  Encodes which fragment types are chemically "present",
      weighted by mass and similarity.

  fd_soft : {label -> float}
      Degree-weighted version of fn_soft.  Same spread, but each
      contribution is multiplied by the node's normalised degree
      d̃_i = deg_i / max(n-1, 1).  Encodes the connectivity role of
      each fragment type.

  fe : {(label_a, label_b) -> float}
      Edge fingerprint.  For each inter-fragment bond (i, j) adds
      0.5 * (s_i + s_j) to the canonical (sorted) label-pair entry.
      Encodes which fragment-type pairs are directly bonded.

Two auxiliary sets:

  support   : frozenset of actual fragment labels in the molecule.
  reachable : frozenset of fn_soft.keys() — labels reachable from
              this molecule's fragments via σ >= NODE_MIN_SIMILARITY.

Relationship to sgo_graphs
--------------------------
  Exact   : identical-graph score (= 1.0)
            size penalty
            degree normalisation and ρ-attenuation
            adjacency coverage scaling
            adaptive weight normalisation

  Approximate : node penalty — uses Dice on soft fingerprints instead
                of greedy one-to-one matching.  The σ-spread
                double-counts cross-type similarity, causing a mild
                over-estimate of the overlap for soft-similar graphs.
                Error decreases with fragment diversity; zero for
                disjoint vocabularies; zero for identical graphs.

                adjacency penalty — uses Dice on edge-type fingerprints.
                Loses information about which specific nodes are bonded;
                two molecules with the same edge-type multiset but
                different topologies score identically on this term.

Advantages
----------
  Symmetry      : all operations are symmetric by construction.
  Precomputation: O(n * k) once per molecule; scoring is O(nnz).
  Matrix compute: full N×N via pairwise sparse dict scoring.
  Metric props  : L2 distance on L2-normalised fn_soft is a true metric.
  ANN-ready     : fingerprint vectors can be indexed with FAISS / ScaNN.
"""

import bisect
import pickle
from collections import defaultdict

import numpy as np
import scipy.sparse as sp
from joblib import Parallel, delayed
from tqdm import tqdm

# Shared with sgo_similarity — import or copy as needed.
from sgo_similarity import (
    EPS,
    MIN_SIZE_RATIO,
    NODE_MIN_SIMILARITY,
    MAX_NEIGHBORS_PER_NODE,
    STRONG_SIMILARITY_FLOOR,
    CHUNK_SIZE,
    _filtered_neighbours,
    _get_similarity_backend,
)


# ------------------------------------------------------------------ #
# 1. Sparse dict helpers                                               #
# ------------------------------------------------------------------ #

def _dict_l1_overlap(da, db):
    """
    Σ_k min(da[k], db[k]) over shared keys.  O(min(nnz_a, nnz_b)).
    Iterates the smaller dict to minimise cache misses.
    """
    if len(da) > len(db):
        da, db = db, da
    return sum(min(v, db[k]) for k, v in da.items() if k in db)


def _dict_l1(d):
    """L1 norm of a non-negative dict."""
    return sum(d.values())


# ------------------------------------------------------------------ #
# 2. Fingerprint construction                                          #
# ------------------------------------------------------------------ #

def build_sgo_fingerprint(
    graph_data,
    node_similarity,
    node_sizes=None,
    node_min_similarity=NODE_MIN_SIMILARITY,
    max_neighbors=MAX_NEIGHBORS_PER_NODE,
    strong_floor=STRONG_SIMILARITY_FLOOR,
):
    """
    Compute a self-contained fingerprint from a preprocessed graph dict.

    Parameters
    ----------
    graph_data       : dict produced by preprocess_graph / SgoSimilarity.preprocess_graph
    node_similarity  : the node_similarity dict (same as in sgo_graphs)
    node_sizes       : optional {MUID -> float} weight map
    node_min_similarity, max_neighbors, strong_floor : same semantics as
                       in sgo_graphs / _filtered_neighbours

    Returns
    -------
    dict with keys:
      fn_soft   {label: float}          soft node fingerprint
      fd_soft   {label: float}          degree-weighted fingerprint
      fe        {(la,lb): float}        edge fingerprint
      support   frozenset               actual fragment labels
      reachable frozenset               fn_soft.keys()
      fn_norm   float                   L1 norm of fn_soft
      S         float                   true total node size
      n         int                     node count
      e         int                     edge count
    """
    labels     = graph_data["node_labels"]
    sizes      = graph_data["node_size_list"]
    degrees    = graph_data["degree_list"]
    edge_pairs = graph_data["edge_pairs"]
    n          = graph_data["n_nodes"]
    e          = graph_data["n_edges"]
    S          = graph_data["total_size"]

    similar_nodes = node_similarity["similar_nodes"]
    max_deg = max(n - 1, 1)

    fn_soft = defaultdict(float)
    fd_soft = defaultdict(float)
    support = set()

    for i, label in enumerate(labels):
        s      = sizes[i]
        d_norm = degrees[i] / max_deg
        support.add(label)

        # Self contribution (σ = 1.0 by definition)
        fn_soft[label] += s
        fd_soft[label] += s * d_norm

        # Soft contributions to neighbours in the similarity graph
        neighbours = _filtered_neighbours(
            similar_nodes, label,
            min_similarity=node_min_similarity,
            max_neighbors=max_neighbors,
            strong_floor=strong_floor,
        )
        for nb_label, sim in neighbours.items():
            fn_soft[nb_label] += s * sim
            fd_soft[nb_label] += s * d_norm * sim

    # Edge fingerprint: canonical (sorted) label pairs
    fe = defaultdict(float)
    for u, v in edge_pairs:
        la, lb = labels[u], labels[v]
        key    = (la, lb) if la <= lb else (lb, la)
        fe[key] += 0.5 * (sizes[u] + sizes[v])

    fn_soft = dict(fn_soft)
    fd_soft = dict(fd_soft)
    fe      = dict(fe)
    fn_norm = _dict_l1(fn_soft) if fn_soft else 0.0

    return {
        "fn_soft":  fn_soft,
        "fd_soft":  fd_soft,
        "fe":       fe,
        "support":  frozenset(support),
        "reachable": frozenset(fn_soft.keys()),
        "fn_norm":  fn_norm,
        "S":        S,
        "n":        n,
        "e":        e,
    }


# ------------------------------------------------------------------ #
# 3. Pairwise scoring                                                  #
# ------------------------------------------------------------------ #

def score_pair_fp(
    fp_a, fp_b,
    w_nodes=0.50, w_size=0.15, w_degree=0.15, w_adj=0.20,
):
    """
    Score two fingerprints.  Pure dict operations, O(nnz).

    Node penalty
    ------------
    Dice coefficient on soft fingerprints:

        overlap  = Σ_k min(fn_a[k], fn_b[k])
        P_node   = 1 - overlap / max(||fn_a||_1, ||fn_b||_1)

    This is symmetric, equals 0 for disjoint vocabularies, and equals 1
    for identical graphs.  For soft-similar graphs it slightly
    over-estimates overlap vs the greedy one-to-one matching in
    sgo_graphs because the σ-spread double-counts cross-type similarity.
    The error is bounded and shrinks with fragment diversity.

    Degree penalty
    --------------
    Coverage-weighted L1 distance on normalised degree vectors.
    d̃[k] = fd_soft[k] / fn_soft[k] is the average normalised degree
    of fragments that contribute to position k.  Weighted by the matched
    soft mass min(fn_a[k], fn_b[k]) at each shared key.

    Adjacency penalty
    -----------------
    Dice on edge-type fingerprints, scaled by node-match coverage —
    same structure as _adjacency_penalty in sgo_graphs.

    Adaptive normalisation
    ----------------------
    Identical to sgo_graphs: inactive terms (undefined degree, topology,
    or zero size difference) are excluded from both numerator and
    normalisation denominator.

    Zero conditions
    ---------------
    Returns 0.0 if size ratio < MIN_SIZE_RATIO (same gate as sgo_graphs).
    """
    na, nb = fp_a["n"], fp_b["n"]
    ea, eb = fp_a["e"], fp_b["e"]

    # ---- size ratio gate ---------------------------------------------
    if max(na, nb) > 0 and min(na, nb) / max(na, nb) < MIN_SIZE_RATIO:
        return 0.0

    fn_a, fn_b = fp_a["fn_soft"], fp_b["fn_soft"]
    fd_a, fd_b = fp_a["fd_soft"], fp_b["fd_soft"]
    fe_a, fe_b = fp_a["fe"],      fp_b["fe"]
    S_a,  S_b  = fp_a["S"],       fp_b["S"]
    fn_norm_a  = fp_a["fn_norm"]
    fn_norm_b  = fp_b["fn_norm"]

    # ---- node penalty (Dice on soft fingerprints) --------------------
    size_denom_soft = max(fn_norm_a, fn_norm_b, EPS)
    overlap_node    = _dict_l1_overlap(fn_a, fn_b)
    node_penalty    = 1.0 - overlap_node / size_denom_soft

    # ---- size penalty ------------------------------------------------
    size_denom   = max(S_a, S_b, EPS)
    size_penalty = abs(S_a - S_b) / size_denom

    # ---- degree penalty ----------------------------------------------
    # d̃[k] = fd[k] / fn[k]: average normalised degree at position k.
    # Weighted by min(fn_a[k], fn_b[k]) — matched soft mass at k.
    # ρ attenuates across size-asymmetric pairs (same as sgo_graphs Fix 3).
    size_ratio = min(na, nb) / max(na, nb)
    deg_num = deg_den = 0.0
    for k in fn_a.keys() & fn_b.keys():
        w_k   = min(fn_a[k], fn_b[k])
        d_a_k = fd_a.get(k, 0.0) / (fn_a[k] + EPS)
        d_b_k = fd_b.get(k, 0.0) / (fn_b[k] + EPS)
        deg_num += w_k * abs(d_a_k - d_b_k)
        deg_den += w_k
    degree_penalty = size_ratio * (deg_num / deg_den) if deg_den > EPS else 0.0

    # ---- adjacency penalty -------------------------------------------
    T_a = _dict_l1(fe_a)
    T_b = _dict_l1(fe_b)
    if T_a > EPS or T_b > EPS:
        preserved   = _dict_l1_overlap(fe_a, fe_b)
        adj_raw     = 1.0 - preserved / max(T_a, T_b, EPS)
        coverage    = overlap_node / size_denom_soft
        adj_penalty = coverage * adj_raw
    else:
        adj_penalty = 0.0

    # ---- adaptive normalisation (identical to sgo_graphs) ------------
    active_w_size   = w_size   if size_penalty   > EPS else 0.0
    active_w_degree = w_degree if max(na, nb) > 1      else 0.0
    active_w_adj    = w_adj    if max(ea, eb) > 0       else 0.0

    active_denom = w_nodes + active_w_size + active_w_degree + active_w_adj

    distance = (
        w_nodes         * node_penalty
        + active_w_size   * size_penalty
        + active_w_degree * degree_penalty
        + active_w_adj    * adj_penalty
    ) / active_denom

    return max(0.0, 1.0 - distance)


# ------------------------------------------------------------------ #
# 4. Pairwise matrix (Dice, exact fingerprint path)                   #
# ------------------------------------------------------------------ #

def _fp_chunk(chunk, fingerprints, order_list, sorted_n_nodes, n_nodes,
              w_nodes, w_size, w_degree, w_adj):
    results = []
    for i in chunk:
        fp_a = fingerprints[i]
        na   = int(n_nodes[i])
        lo   = int(np.ceil(na * MIN_SIZE_RATIO))
        hi   = int(np.floor(na / MIN_SIZE_RATIO))
        left  = bisect.bisect_left(sorted_n_nodes, lo)
        right = bisect.bisect_right(sorted_n_nodes, hi)

        for pos in range(left, right):
            j = order_list[pos]
            if j <= i:
                continue
            score = score_pair_fp(fp_a, fingerprints[j],
                                  w_nodes, w_size, w_degree, w_adj)
            if score > 0.0:
                results.append((i, j, float(score)))
    return results


def compute_fp_similarity_matrix(
    fingerprints,
    w_nodes=0.50, w_size=0.15, w_degree=0.15, w_adj=0.20,
    n_jobs=-1,
    chunk_size=CHUNK_SIZE,
):
    """
    Full N×N similarity matrix using Dice node penalty (pairwise loop).

    Symmetric by construction.  Consistent with updated sgo_graphs for
    the cases analysed: singletons, functionalised analogues, disjoint
    vocabularies.  Approximates sgo_graphs for multi-node graphs (see
    module docstring for approximation details).

    Parallelised with joblib; fingerprints are serialised once per chunk
    (same batching strategy as sgo_similarity._compute_chunk).
    """
    n = len(fingerprints)
    sim_matrix = np.zeros((n, n), dtype=np.float32)
    np.fill_diagonal(sim_matrix, 1.0)

    n_nodes = np.array([fp["n"] for fp in fingerprints], dtype=np.int32)
    order   = np.argsort(n_nodes, kind="mergesort")
    order_list     = order.tolist()
    sorted_n_nodes = n_nodes[order].tolist()

    chunks = [
        list(range(i, min(i + chunk_size, n)))
        for i in range(0, n, chunk_size)
    ]

    chunk_results = Parallel(n_jobs=n_jobs)(
        delayed(_fp_chunk)(
            chunk, fingerprints, order_list, sorted_n_nodes, n_nodes,
            w_nodes, w_size, w_degree, w_adj,
        )
        for chunk in tqdm(chunks, desc="fp matrix (Dice)")
    )

    for chunk_rows in chunk_results:
        for i, j, score in chunk_rows:
            score = float(np.clip(score, 0.0, 1.0))
            sim_matrix[i, j] = score
            sim_matrix[j, i] = score

    np.clip(sim_matrix, 0.0, 1.0, out=sim_matrix)
    return sim_matrix


# ------------------------------------------------------------------ #
# 5. Sparse matrix helper                                              #
# ------------------------------------------------------------------ #

def _build_fp_sparse_matrices(fingerprints):
    """
    Encode fingerprints as scipy CSR matrices for vectorised scoring.

    Returns
    -------
    F_node      : (N, F) float32 CSR, L2-normalised soft node fingerprints
    F_edge      : (N, E) float32 CSR, L1-normalised edge fingerprints
    frag_index  : {label -> col index} for node matrix
    edge_index  : {(la,lb) -> col index} for edge matrix
    """
    N = len(fingerprints)

    # Fragment index: all labels that appear in any fingerprint's fn_soft
    all_labels = set()
    for fp in fingerprints:
        all_labels.update(fp["fn_soft"].keys())
    frag_index = {label: i for i, label in enumerate(sorted(all_labels, key=str))}
    F = len(frag_index)

    # Node matrix
    node_rows, node_cols, node_vals = [], [], []
    for i, fp in enumerate(fingerprints):
        for label, val in fp["fn_soft"].items():
            col = frag_index.get(label)
            if col is not None:
                node_rows.append(i)
                node_cols.append(col)
                node_vals.append(val)

    F_node = sp.csr_matrix(
        (node_vals, (node_rows, node_cols)), shape=(N, F), dtype=np.float64
    )
    # L2-normalise rows so F_node @ F_node.T = cosine similarity matrix
    norms = np.sqrt(np.asarray(F_node.power(2).sum(axis=1)).ravel())
    norms = np.maximum(norms, EPS)
    F_node = F_node.multiply(1.0 / norms[:, None]).tocsr().astype(np.float32)

    # Edge matrix
    all_edge_keys = set()
    for fp in fingerprints:
        all_edge_keys.update(fp["fe"].keys())
    edge_index = {key: i for i, key in enumerate(sorted(all_edge_keys, key=str))}
    E = max(len(edge_index), 1)

    edge_rows, edge_cols, edge_vals = [], [], []
    for i, fp in enumerate(fingerprints):
        row_sum = max(_dict_l1(fp["fe"]), EPS)
        for key, val in fp["fe"].items():
            col = edge_index.get(key)
            if col is not None:
                edge_rows.append(i)
                edge_cols.append(col)
                edge_vals.append(val / row_sum)

    if edge_vals:
        F_edge = sp.csr_matrix(
            (edge_vals, (edge_rows, edge_cols)), shape=(N, E), dtype=np.float32
        )
    else:
        F_edge = sp.csr_matrix((N, E), dtype=np.float32)

    return F_node, F_edge, frag_index, edge_index


# ------------------------------------------------------------------ #
# 6. Class wrapper                                                     #
# ------------------------------------------------------------------ #

class SgoFingerprintSimilarity:
    """
    Manages precomputed fingerprints for a molecule database and provides
    pairwise fingerprint-based similarity matrix computation.

    Usage
    -----
    From an existing SgoSimilarity instance:

        engine = SgoFingerprintSimilarity.from_sgo(sgo)
        engine.build_fingerprints(graph_data_list)

        # Pairwise-exact matrix (symmetric, consistent with sgo_graphs):
        sim_matrix = engine.compute_similarity_matrix()

        # Score a new molecule against the database:
        fp_query = engine.fingerprint(query_graph_data)
        scores   = [engine.score_pair(fp_query, fp) for fp in engine.fingerprints]

        # Sparse matrices for FAISS / ScaNN / ANN indexing:
        F_node, F_edge, frag_idx, edge_idx = engine.build_sparse_matrices()
    """

    def __init__(
        self,
        node_similarity,
        node_sizes=None,
        node_min_similarity=NODE_MIN_SIMILARITY,
        max_neighbors=MAX_NEIGHBORS_PER_NODE,
        strong_floor=STRONG_SIMILARITY_FLOOR,
    ):
        self.node_similarity     = node_similarity
        self.node_sizes          = node_sizes or {}
        self.node_min_similarity = node_min_similarity
        self.max_neighbors       = max_neighbors
        self.strong_floor        = strong_floor
        self.fingerprints        = []

    @classmethod
    def from_sgo(cls, sgo_instance):
        """Initialise from an existing SgoSimilarity instance."""
        return cls(
            node_similarity     = sgo_instance.node_similarity,
            node_sizes          = sgo_instance.node_sizes,
        )

    def fingerprint(self, graph_data):
        """Compute the fingerprint for a single preprocessed graph dict."""
        return build_sgo_fingerprint(
            graph_data,
            self.node_similarity,
            self.node_sizes,
            node_min_similarity = self.node_min_similarity,
            max_neighbors       = self.max_neighbors,
            strong_floor        = self.strong_floor,
        )

    def build_fingerprints(self, graph_data_list):
        """
        Precompute and store fingerprints for a list of graph dicts.
        O(N * n_avg * k) total.  Returns the fingerprint list.
        """
        self.fingerprints = [
            self.fingerprint(gd)
            for gd in tqdm(graph_data_list, desc="building fingerprints")
        ]
        return self.fingerprints

    def score_pair(self, fp_a, fp_b,
                   w_nodes=0.50, w_size=0.15, w_degree=0.15, w_adj=0.20):
        """Score two fingerprint dicts.  O(nnz)."""
        return score_pair_fp(fp_a, fp_b, w_nodes, w_size, w_degree, w_adj)

    def compute_similarity_matrix(
        self,
        w_nodes=0.50, w_size=0.15, w_degree=0.15, w_adj=0.20,
        n_jobs=-1,
        chunk_size=CHUNK_SIZE,
    ):
        """
        Pairwise-exact N×N matrix using Dice node penalty.
        Symmetric by construction.  Faithful to score_pair_fp for all
        penalty terms including degree.
        """
        return compute_fp_similarity_matrix(
            self.fingerprints,
            w_nodes=w_nodes, w_size=w_size, w_degree=w_degree, w_adj=w_adj,
            n_jobs=n_jobs, chunk_size=chunk_size,
        )

    def build_sparse_matrices(self):
        """
        Return (F_node, F_edge, frag_index, edge_index) for ANN indexing.

        F_node is an (N, F) L2-normalised sparse CSR matrix suitable for
        direct use with FAISS (after .toarray()) or scipy nearest-neighbour
        search.  Cosine similarity between two rows equals their inner
        product (dot product of L2-normalised vectors).
        """
        return _build_fp_sparse_matrices(self.fingerprints)


if __name__ == "__main__":
    GRAPHS_PATH = "/home/mattia/geass/NPA_fragmentation/NPA_graphs_map4.pkl"
    SIMILARITY_PATH = "/home/mattia/geass/NPA_subgraphs/prep_files/NPA_node_similarity_map4.pkl"
    SIZES_PATH = "/home/mattia/geass/NPA_subgraphs/prep_files/NPA_node_weights.pkl"
    SIM_OUTPUT_PATH = "/home/mattia/geass/NPA_analysis/NPA_sim_matrix_sgofp.pkl"

    # Lazy import here to avoid module import cycles at import time.
    from sgo_similarity import SgoSimilarity

    sgo_engine = SgoSimilarity.from_files(
        graphs_path=GRAPHS_PATH,
        similarity_path=SIMILARITY_PATH,
        sizes_path=SIZES_PATH)
    ids = list(sgo_engine.npa_graphs.keys())
    graph_data_list = [sgo_engine.preprocess_graph(sgo_engine.npa_graphs[gid]) for gid in ids]

    fp_engine = SgoFingerprintSimilarity.from_sgo(sgo_engine)
    fingerprints = fp_engine.build_fingerprints(graph_data_list)
    sim_matrix = fp_engine.compute_similarity_matrix(n_jobs=-1, chunk_size=CHUNK_SIZE)

    with open(SIM_OUTPUT_PATH, "wb") as f:
        pickle.dump({"ids": ids, "sim_matrix": sim_matrix}, f, protocol=pickle.HIGHEST_PROTOCOL)
