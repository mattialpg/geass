from __future__ import annotations

import pickle
from typing import Any

import networkx as nx
import numpy as np

NODE_SIM_PATH = "/home/mattia/geass/NPA_subgraphs/prep_files/NPA_node_similarity_atomp.pkl"


def load_similarity_matrix(similarity_pkl_path: str) -> Any:
    with open(similarity_pkl_path, "rb") as f:
        return pickle.load(f)


def _node_label_from_attrs(node: Any, attrs: dict[str, Any]) -> Any:
    if "MUID" in attrs:
        return attrs["MUID"]
    if "label" in attrs:
        return attrs["label"]
    return node


def _build_sparse_backend(similar_nodes: Any) -> dict[str, Any]:
    if isinstance(similar_nodes, dict) and similar_nodes.get("format") == "upper_csr_v1":
        nodes = similar_nodes["nodes"]
        return {
            "use_csr": True,
            "node_to_idx": {node: i for i, node in enumerate(nodes)},
            "row_ptr": np.asarray(similar_nodes["row_ptr"]),
            "col_idx": np.asarray(similar_nodes["col_idx"]),
            "values": np.asarray(similar_nodes["values"]),
        }
    return {"use_csr": False}


def _similarity_lookup(
    a: Any,
    b: Any,
    similar_nodes: Any,
    backend: dict[str, Any],
) -> float:
    if a == b:
        return 1.0

    if not backend["use_csr"]:
        da = similar_nodes.get(a, {}) if isinstance(similar_nodes, dict) else {}
        db = similar_nodes.get(b, {}) if isinstance(similar_nodes, dict) else {}
        return float(max(da.get(b, 0.0), db.get(a, 0.0)))

    ia = backend["node_to_idx"].get(a)
    ib = backend["node_to_idx"].get(b)
    if ia is None or ib is None:
        return 0.0
    if ia > ib:
        ia, ib = ib, ia

    start = int(backend["row_ptr"][ia])
    end = int(backend["row_ptr"][ia + 1])
    if start >= end:
        return 0.0

    cols = backend["col_idx"][start:end]
    pos = int(np.searchsorted(cols, ib))
    if pos < (end - start) and int(cols[pos]) == ib:
        return float(backend["values"][start + pos])
    return 0.0


def build_pair_similarity_matrix_from_node_similarity(
    g1: nx.Graph,
    g2: nx.Graph,
    node_similarity: dict[str, Any],
) -> np.ndarray:
    canonical_map = node_similarity.get("canonical_map", {})
    similar_nodes = node_similarity.get("similar_nodes", {})
    backend = _build_sparse_backend(similar_nodes)

    labels1 = []
    for node, attrs in g1.nodes(data=True):
        raw = _node_label_from_attrs(node, attrs)
        labels1.append(canonical_map.get(raw, raw))

    labels2 = []
    for node, attrs in g2.nodes(data=True):
        raw = _node_label_from_attrs(node, attrs)
        labels2.append(canonical_map.get(raw, raw))

    n1 = len(labels1)
    n2 = len(labels2)
    sim = np.zeros((n1, n2), dtype=float)

    for i in range(n1):
        a = labels1[i]
        for j in range(n2):
            b = labels2[j]
            sim[i, j] = _similarity_lookup(a, b, similar_nodes, backend)

    return sim


def validate_and_clip_similarity(sim: np.ndarray, n1: int, n2: int) -> np.ndarray:
    if sim.shape != (n1, n2):
        raise ValueError(f"Similarity matrix shape {sim.shape} does not match ({n1}, {n2})")
    return np.clip(sim, 0.0, 1.0)


def greedy_node_matching(sim: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n1, n2 = sim.shape
    flat_order = np.argsort(sim, axis=None)[::-1]

    matched_rows = np.zeros(n1, dtype=bool)
    matched_cols = np.zeros(n2, dtype=bool)

    match_i: list[int] = []
    match_j: list[int] = []
    match_s: list[float] = []

    for flat_idx in flat_order:
        i, j = np.unravel_index(flat_idx, sim.shape)
        if not matched_rows[i] and not matched_cols[j]:
            matched_rows[i] = True
            matched_cols[j] = True
            match_i.append(i)
            match_j.append(j)
            match_s.append(float(sim[i, j]))

        if matched_rows.all() or matched_cols.all():
            break

    return (
        np.asarray(match_i, dtype=int),
        np.asarray(match_j, dtype=int),
        np.asarray(match_s, dtype=float),
    )


def node_penalty(match_s: np.ndarray, n1: int, n2: int) -> float:
    return 1.0 - (float(match_s.sum()) / max(n1, n2))


def size_penalty(n1: int, n2: int) -> float:
    return abs(n1 - n2) / max(n1, n2)


def degree_penalty(
    g1: nx.Graph,
    g2: nx.Graph,
    nodes1: list[Any],
    nodes2: list[Any],
    match_i: np.ndarray,
    match_j: np.ndarray,
) -> float:
    m = len(match_i)
    if m == 0:
        return 1.0

    deg1 = dict(g1.degree())
    deg2 = dict(g2.degree())
    degree_diff = np.empty(m, dtype=float)

    for k in range(m):
        d1 = deg1[nodes1[int(match_i[k])]]
        d2 = deg2[nodes2[int(match_j[k])]]
        degree_diff[k] = abs(d1 - d2) / max(d1, d2, 1)

    return float(np.mean(degree_diff))


def adjacency_penalty(
    g1: nx.Graph,
    g2: nx.Graph,
    nodes1: list[Any],
    nodes2: list[Any],
    match_i: np.ndarray,
    match_j: np.ndarray,
) -> float:
    e1 = g1.number_of_edges()
    e2 = g2.number_of_edges()
    if e1 == 0 and e2 == 0:
        return 0.0

    m = len(match_i)
    if m == 0:
        return 1.0 - (0.0 / max(e1, e2, 1))

    g1_to_g2 = {nodes1[int(match_i[k])]: nodes2[int(match_j[k])] for k in range(m)}
    preserved_edges = 0

    for u, v in g1.edges():
        if u in g1_to_g2 and v in g1_to_g2:
            if g2.has_edge(g1_to_g2[u], g1_to_g2[v]):
                preserved_edges += 1

    return 1.0 - (preserved_edges / max(e1, e2, 1))


def combine_distance(
    node_pen: float,
    size_pen: float,
    degree_pen: float,
    adj_pen: float,
    w_nodes: float = 0.50,
    w_size: float = 0.15,
    w_degree: float = 0.15,
    w_adj: float = 0.20,
) -> tuple[float, float]:
    distance = (
        w_nodes * node_pen +
        w_size * size_pen +
        w_degree * degree_pen +
        w_adj * adj_pen
    )
    similarity = max(0.0, 1.0 - distance)
    return float(distance), float(similarity)


def graph_similarity_from_matrix(
    g1: nx.Graph,
    g2: nx.Graph,
    sim: np.ndarray,
    w_nodes: float = 0.50,
    w_size: float = 0.15,
    w_degree: float = 0.15,
    w_adj: float = 0.20,
) -> dict[str, Any]:
    nodes1 = list(g1.nodes())
    nodes2 = list(g2.nodes())
    n1 = len(nodes1)
    n2 = len(nodes2)

    sim = validate_and_clip_similarity(sim, n1, n2)
    match_i, match_j, match_s = greedy_node_matching(sim)

    p_node = node_penalty(match_s, n1, n2)
    p_size = size_penalty(n1, n2)
    p_degree = degree_penalty(g1, g2, nodes1, nodes2, match_i, match_j)
    p_adj = adjacency_penalty(g1, g2, nodes1, nodes2, match_i, match_j)

    distance, similarity = combine_distance(
        p_node, p_size, p_degree, p_adj,
        w_nodes=w_nodes, w_size=w_size, w_degree=w_degree, w_adj=w_adj,
    )

    return {
        "distance": distance,
        "similarity": similarity,
        "penalties": {
            "node_penalty": p_node,
            "size_penalty": p_size,
            "degree_penalty": p_degree,
            "adjacency_penalty": p_adj,
        },
        "matches": {
            "i": match_i,
            "j": match_j,
            "s": match_s,
        },
    }


def graph_similarity_from_pickle(
    g1: nx.Graph,
    g2: nx.Graph,
    similarity_pkl_path: str = NODE_SIM_PATH,
    w_nodes: float = 0.50,
    w_size: float = 0.15,
    w_degree: float = 0.15,
    w_adj: float = 0.20,
) -> dict[str, Any]:
    payload = load_similarity_matrix(similarity_pkl_path)
    if isinstance(payload, dict) and "similar_nodes" in payload:
        sim = build_pair_similarity_matrix_from_node_similarity(g1, g2, payload)
    else:
        sim = np.asarray(payload, dtype=float)
    return graph_similarity_from_matrix(
        g1,
        g2,
        sim,
        w_nodes=w_nodes,
        w_size=w_size,
        w_degree=w_degree,
        w_adj=w_adj,
    )
