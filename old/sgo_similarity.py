
# --------- Soft Graph Overlap (SGO) ------------
from collections import Counter, defaultdict
import networkx as nx

def preprocess_graph(graph):
    node_muids = [attrs["MUID"] for _, attrs in graph.nodes(data=True)]

    edge_muids = [
        tuple(sorted((graph.nodes[u]["MUID"], graph.nodes[v]["MUID"])))
        for u, v in graph.edges()
    ]

    node_counter = Counter(node_muids)
    edge_counter = Counter(edge_muids)

    node_set = set(node_counter)
    edge_set = set(edge_counter)

    edge_neighbour_index = defaultdict(set)
    for j, (x, y) in enumerate(edge_muids):
        edge_neighbour_index[x].add(j)
        edge_neighbour_index[y].add(j)

    return {
        "n_nodes": len(node_muids),
        "n_edges": len(edge_muids),
        "node_muids": node_muids,
        "edge_muids": edge_muids,
        "node_counter": node_counter,
        "edge_counter": edge_counter,
        "node_set": node_set,
        "edge_set": edge_set,
        "edge_neighbour_index": edge_neighbour_index,
    }


def sgo_nodes(graph_a_data, graph_b_data, node_similarity, B, normalisation="dice"):
    sim_dict = node_similarity["sim_dict"]
    similar_nodes = node_similarity["similar_nodes"]
    threshold = node_similarity["node_threshold"]

    nodes_a = graph_a_data["node_muids"]
    nodes_b = graph_b_data["node_muids"]

    if not nodes_a and not nodes_b:
        return 1.0
    if not nodes_a or not nodes_b:
        return 0.0

    count_a = graph_a_data["node_counter"]
    count_b = graph_b_data["node_counter"]

    exact = sum(min(count_a[n], count_b[n]) for n in count_a.keys() & count_b.keys())

    rem_a = []
    for n, c in count_a.items():
        k = c - min(c, count_b.get(n, 0))
        if k > 0:
            rem_a.extend([n] * k)

    rem_b = []
    for n, c in count_b.items():
        k = c - min(c, count_a.get(n, 0))
        if k > 0:
            rem_b.extend([n] * k)

    if not rem_a or not rem_b:
        matched = float(exact)
    else:
        node_to_b = defaultdict(set)
        for j, n in enumerate(rem_b):
            node_to_b[n].add(j)

        B.clear()

        for i, u in enumerate(rem_a):
            cand = set()
            for s in similar_nodes.get(u, ()):
                cand.update(node_to_b.get(s, ()))

            if not cand:
                continue

            sim_u = sim_dict.get(u, {})
            add_edge = B.add_edge

            for j in cand:
                v = rem_b[j]
                w = sim_u.get(v, 0.0)
                if w >= threshold:
                    add_edge(("a", i), ("b", j), weight=w)

        if B.number_of_edges() == 0:
            matched = float(exact)
        else:
            matching = nx.algorithms.matching.max_weight_matching(
                B, maxcardinality=False, weight="weight"
            )
            soft = sum(B[u][v]["weight"] for u, v in matching)
            matched = float(exact) + soft

    if normalisation == "dice":
        return 2.0 * matched / (len(nodes_a) + len(nodes_b))

    denom = len(nodes_a) + len(nodes_b) - matched
    return matched / denom if denom > 0 else 1.0


def sgo_edges(graph_a_data, graph_b_data, node_similarity, B, node_weights, normalisation="jaccard"):
    sim_dict = node_similarity["sim_dict"]
    similar_nodes = node_similarity["similar_nodes"]
    threshold = node_similarity["node_threshold"]

    edges_a = graph_a_data["edge_muids"]
    edges_b = graph_b_data["edge_muids"]

    if not edges_a and not edges_b:
        return 1.0
    if not edges_a or not edges_b:
        return 0.0

    count_a = graph_a_data["edge_counter"]
    count_b = graph_b_data["edge_counter"]

    exact = sum(min(count_a[e], count_b[e]) for e in count_a.keys() & count_b.keys())

    rem_a = []
    for e, c in count_a.items():
        k = c - min(c, count_b.get(e, 0))
        if k > 0:
            rem_a.extend([e] * k)

    rem_b = []
    for e, c in count_b.items():
        k = c - min(c, count_a.get(e, 0))
        if k > 0:
            rem_b.extend([e] * k)

    if not rem_a or not rem_b:
        matched = float(exact)
    else:
        node_to_edges = defaultdict(set)
        for j, (x, y) in enumerate(rem_b):
            node_to_edges[x].add(j)
            node_to_edges[y].add(j)

        B.clear()

        for i, (u, v) in enumerate(rem_a):
            cand = set()

            for s in similar_nodes.get(u, ()):
                cand.update(node_to_edges.get(s, ()))
            for s in similar_nodes.get(v, ()):
                cand.update(node_to_edges.get(s, ()))

            if not cand:
                continue

            sim_u = sim_dict.get(u, {})
            sim_v = sim_dict.get(v, {})

            w_u = node_weights.get(u, 1.0)
            w_v = node_weights.get(v, 1.0)
            w_tot = w_u + w_v

            add_edge = B.add_edge

            for j in cand:
                x, y = rem_b[j]

                score_xy = (
                    w_u * sim_u.get(x, 0.0) +
                    w_v * sim_v.get(y, 0.0)
                ) / w_tot

                score_yx = (
                    w_u * sim_u.get(y, 0.0) +
                    w_v * sim_v.get(x, 0.0)
                ) / w_tot

                w = max(score_xy, score_yx)

                if w >= threshold:
                    add_edge(("a", i), ("b", j), weight=w)

        if B.number_of_edges() == 0:
            matched = float(exact)
        else:
            matching = nx.algorithms.matching.max_weight_matching(
                B, maxcardinality=False, weight="weight"
            )
            soft = sum(B[u][v]["weight"] for u, v in matching)
            matched = float(exact) + soft

    if normalisation == "dice":
        return 2.0 * matched / (len(edges_a) + len(edges_b))

    denom = len(edges_a) + len(edges_b) - matched
    return matched / denom if denom > 0 else 1.0


def sgo_graphs(graph_a_data, graph_b_data, node_similarity, node_weights, alpha=0.5, normalisation="jaccard"):
    na = graph_a_data["n_nodes"]
    nb = graph_b_data["n_nodes"]

    if min(na, nb) / max(na, nb) < 0.3:
        return 0.0

    ea = graph_a_data["n_edges"]
    eb = graph_b_data["n_edges"]

    if ea > 0 and eb > 0 and min(ea, eb) / max(ea, eb) < 0.3:
        return 0.0

    B = nx.Graph()

    node_score = sgo_nodes(graph_a_data, graph_b_data, node_similarity, B, normalisation)
    edge_score = sgo_edges(graph_a_data, graph_b_data, node_similarity, B, node_weights, normalisation)

    return alpha * node_score + (1.0 - alpha) * edge_score


if __name__ == "__main__":
    import bisect
    import pickle

    import numpy as np
    from tqdm.auto import tqdm

    # Load graphs
    with open("../NPA_fragmentation/NPA_graphs.pkl", "rb") as f:
        npa_graphs = pickle.load(f)

    ids = list(npa_graphs.keys())
    n = len(ids)

    # Load node similarity data
    with open("prep_files/NPA_node_similarity_sparse_atomp.pkl", "rb") as f:
        node_similarity = pickle.load(f)
    with open("prep_files/NPA_node_weights.pkl", "rb") as f:
        node_weights = pickle.load(f)

    # Precompute graph representations once
    graph_data_list = [preprocess_graph(npa_graphs[gid]) for gid in ids]

    # Flatten metadata for faster access
    n_nodes = np.fromiter((g["n_nodes"] for g in graph_data_list), dtype=np.int32, count=n)
    n_edges = np.fromiter((g["n_edges"] for g in graph_data_list), dtype=np.int32, count=n)

    # Sort indices by node count so we only scan size-compatible candidates
    order = np.argsort(n_nodes, kind="mergesort")
    order_list = order.tolist()
    sorted_n_nodes = n_nodes[order]

    # Map original index -> position in node-sorted order
    rank_in_order = np.empty(n, dtype=np.int32)
    rank_in_order[order] = np.arange(n, dtype=np.int32)

    # Initialise similarity matrix
    sim_matrix = np.zeros((n, n), dtype=np.float32)
    np.fill_diagonal(sim_matrix, 1.0)

    for i in tqdm(range(n), total=n):
        ga = graph_data_list[i]
        na = n_nodes[i]
        ea = n_edges[i]

        # Safe node-count ratio filter window
        lo_nodes = int(np.ceil(na * 0.3))
        hi_nodes = int(np.floor(na / 0.3))

        left = bisect.bisect_left(sorted_n_nodes, lo_nodes)
        right = bisect.bisect_right(sorted_n_nodes, hi_nodes)

        for pos in range(left, right):
            j = order_list[pos]

            if j <= i:
                continue

            nb = n_nodes[j]
            eb = n_edges[j]

            # Safe edge-count ratio filter
            if ea > 0 and eb > 0 and min(ea, eb) / max(ea, eb) < 0.3:
                continue

            score = sgo_graphs(ga, graph_data_list[j], node_similarity, node_weights)

            if score != 0.0:
                sim_matrix[i, j] = score
                sim_matrix[j, i] = score

    # Save matrix
    with open("NPA_graph_similarity_atomp.pkl", "wb") as f:
        pickle.dump({"ids": ids, "sim_matrix": sim_matrix},
            f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Remove molecules with no neighbour above a threshold
    with open("NPA_graph_similarity_atomp.pkl", "rb") as f:
        data = pickle.load(f)

    ids = np.array(data["ids"])
    sim = np.asarray(data["sim_matrix"], dtype=np.float32)

    threshold = 0.8

    # Ignore self-similarity on the diagonal
    sim_wo_diag = sim.copy()
    np.fill_diagonal(sim_wo_diag, 0.0)

    # Keep molecules that have at least one neighbour above threshold
    mask = (sim_wo_diag >= threshold).any(axis=1)

    ids_filtered = ids[mask]
    sim_filtered = sim[mask][:, mask]

    with open("NPA_graph_similarity_atomp_filtered.pkl", "wb") as f:
        pickle.dump({"ids": ids_filtered.tolist(), "sim_matrix": sim_filtered},
            f, protocol=pickle.HIGHEST_PROTOCOL)