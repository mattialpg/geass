import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.sparse import csr_matrix, load_npz
from scipy.sparse.csgraph import connected_components

SIM_METHOD = "map4"
ANN_INPUT = "/home/mattia/geass/NPA_fragmentation/NPA_fragments_map4.pkl"

CUTS = np.round(np.linspace(0.20, 0.90, 15), 2)
THRESHOLD = 0.7
CLUSTER_METHOD = "average_linkage"   # "average_linkage" or "connected_components"


def load_similarity_payload():
    if SIM_METHOD not in {"sgo", "map4"}:
        raise ValueError("SIM_METHOD must be 'sgo' or 'map4'.")

    base = f"/home/mattia/geass/NPA_analysis/NPA_sim_matrix_{SIM_METHOD}"
    ids = np.load(f"{base}_ids.npy", allow_pickle=True)
    sim_matrix = load_npz(f"{base}.npz").toarray().astype(np.float32, copy=False)
    return ids, sim_matrix


def _prefilter(ids, sim_matrix):
    """Keep only rows/cols with at least one off-diagonal similarity >= THRESHOLD."""
    off_diag_max = np.where(np.eye(len(ids), dtype=bool), -1.0, sim_matrix).max(axis=1)
    mask = off_diag_max >= THRESHOLD
    del off_diag_max
    return ids[mask], sim_matrix[np.ix_(mask, mask)]


def _relabel_by_first_appearance(labels):
    out = np.empty(len(labels), dtype=np.int32)
    mapping = {}
    next_id = 1
    for i, c in enumerate(labels):
        if c not in mapping:
            mapping[c] = next_id
            next_id += 1
        out[i] = mapping[c]
    return out


def _cluster_average_linkage(ids_filtered, sim_filtered):
    df_clusters = pd.DataFrame({"NPA_ID": ids_filtered})
    if sim_filtered.shape[0] < 2:
        for t in CUTS:
            df_clusters[f"cluster_id_t{int(t * 100):02d}"] = "1"
        return df_clusters

    # Symmetrise in-place, then extract condensed directly as float64
    sim_filtered += sim_filtered.T
    sim_filtered *= 0.5
    np.clip(sim_filtered, 0.0, 1.0, out=sim_filtered)
    np.fill_diagonal(sim_filtered, 1.0)

    iu = np.triu_indices(sim_filtered.shape[0], k=1)
    condensed = (1.0 - sim_filtered[iu]).astype(np.float64)
    np.maximum(condensed, 0.0, out=condensed)
    del sim_filtered   # free before linkage

    Z = linkage(condensed, method="average")
    del condensed
    leaf_order = dendrogram(Z, no_plot=True)["leaves"]

    for t in CUTS:
        clusters = fcluster(Z, t=t, criterion="distance")
        ordered_clusters = clusters[leaf_order]
        cluster_map = {cid: i + 1 for i, cid in enumerate(pd.unique(ordered_clusters))}
        df_clusters[f"cluster_id_t{int(t * 100):02d}"] = pd.array(
            [str(cluster_map[c]) for c in clusters], dtype="string"
        )
    return df_clusters


def _cluster_connected_components(ids_filtered, sim_filtered):
    df_clusters = pd.DataFrame({"NPA_ID": ids_filtered})
    if sim_filtered.shape[0] == 0:
        return df_clusters

    off_diag_max = np.where(np.eye(len(ids_filtered), dtype=bool), -1.0, sim_filtered).max(axis=1)
    for t in CUTS:
        # Per-cut prefilter: keep only nodes with at least one off-diagonal
        # similarity >= current threshold t.
        mask = off_diag_max >= t

        col = pd.array([pd.NA] * len(ids_filtered), dtype="string")
        if not mask.any():
            df_clusters[f"cluster_id_t{int(t * 100):02d}"] = col
            continue

        sim_t = sim_filtered[np.ix_(mask, mask)]
        # Build sparse adjacency directly — no full boolean matrix materialised
        rows, cols = np.where(sim_t >= t)
        off_diag = rows != cols
        adj = csr_matrix(
            (np.ones(off_diag.sum(), dtype=np.bool_), (rows[off_diag], cols[off_diag])),
            shape=sim_t.shape)
        _, labels = connected_components(adj, directed=False, return_labels=True)
        col[np.where(mask)[0]] = _relabel_by_first_appearance(labels).astype(str)
        df_clusters[f"cluster_id_t{int(t * 100):02d}"] = col
    del off_diag_max
    return df_clusters


def cluster_from_similarity(ids, sim_matrix, method=CLUSTER_METHOD):
    # Clip in-place on the float32 matrix — no dtype upgrade on the full matrix
    np.clip(sim_matrix, 0.0, 1.0, out=sim_matrix)

    if method == "average_linkage":
        ids_filtered, sim_filtered = _prefilter(ids, sim_matrix)
        del sim_matrix   # full matrix no longer needed
        return _cluster_average_linkage(ids_filtered, sim_filtered)
    if method == "connected_components":
        return _cluster_connected_components(ids, sim_matrix)
    raise ValueError(f"Unknown CLUSTER_METHOD='{method}'.")


def main():
    ids, sim_matrix = load_similarity_payload()
    df_clusters = cluster_from_similarity(ids, sim_matrix, method=CLUSTER_METHOD)
    cluster_cols = [c for c in df_clusters.columns if c.startswith("cluster_id_t")]

    df_fragments = pd.read_pickle(ANN_INPUT)[["NPA_ID", "ISOSMILES"]]
    df_clusters = df_fragments.merge(df_clusters, on="NPA_ID", how="right")
    df_clusters = df_clusters.fillna("0")

    cluster_output = f"/home/mattia/geass/NPA_analysis/NPA_clusters_{SIM_METHOD}.csv"
    df_clusters.to_csv(cluster_output, index=False)


if __name__ == "__main__":
    main()
