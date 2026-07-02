import pickle
import bisect

import networkx as nx
import numpy as np
from tqdm.auto import tqdm


def _build_similarity_backend(similar_nodes):
    use_csr = isinstance(similar_nodes, dict) and similar_nodes.get("format") == "upper_csr_v1"
    if not use_csr:
        return {"use_csr": False}
    nodes = similar_nodes["nodes"]
    return {
        "use_csr": True,
        "node_to_idx": {node: i for i, node in enumerate(nodes)},
        "row_ptr": np.asarray(similar_nodes["row_ptr"]),
        "col_idx": np.asarray(similar_nodes["col_idx"]),
        "values": np.asarray(similar_nodes["values"]),
    }


def _make_helpers(node_weights, canonical_map, similar_nodes, backend):
    def canonical(label):
        return canonical_map.get(label, label)

    def node_label(attrs):
        if "label" in attrs:
            return attrs["label"]
        return attrs.get("MUID")

    def node_weight(attrs):
        if "num_heavy_atoms" in attrs:
            return float(attrs["num_heavy_atoms"])
        label = node_label(attrs)
        if label is None:
            return 1.0
        c = canonical(label)
        return float(node_weights.get(c, node_weights.get(label, 1.0)))

    def node_similarity_value(label_a, label_b):
        if label_a is None or label_b is None:
            return 0.0
        a = canonical(label_a)
        b = canonical(label_b)
        if a == b:
            return 1.0

        if not backend["use_csr"]:
            da = similar_nodes.get(a, {})
            db = similar_nodes.get(b, {})
            return max(da.get(b, 0.0), db.get(a, 0.0))

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

    return node_label, node_weight, node_similarity_value


def _build_cost_functions(node_label, node_weight, node_similarity_value):
    def node_del_cost(a):
        return node_weight(a)

    def node_ins_cost(a):
        return node_weight(a)

    def node_subst_cost(a, b):
        wa = node_weight(a)
        wb = node_weight(b)
        sim = node_similarity_value(node_label(a), node_label(b))

        if sim > 0.0:
            cost = (1.0 - sim) * min(wa, wb)
        else:
            cost = abs(wa - wb) + 1.0
        return min(cost, wa + wb)

    def edge_subst_cost(a, b):
        return 0.0 if a == b else 1.0

    return node_del_cost, node_ins_cost, node_subst_cost, edge_subst_cost


def _normalize_similarity(g1, g2, ged, node_weight):
    node_weight_total = (
        sum(node_weight(d) for _, d in g1.nodes(data=True)) +
        sum(node_weight(d) for _, d in g2.nodes(data=True))
    )
    edge_weight_total = g1.number_of_edges() + g2.number_of_edges()
    c_max = node_weight_total + edge_weight_total

    if c_max <= 0:
        similarity = 1.0
    else:
        similarity = 1.0 - (float(ged) / float(c_max))
        similarity = max(0.0, min(1.0, similarity))

    return {"ged": float(ged), "similarity": similarity, "c_max": float(c_max)}


def _prepare_ged_context(node_similarity, node_weights):
    canonical_map = node_similarity.get("canonical_map", {})
    similar_nodes = node_similarity["similar_nodes"]
    backend = _build_similarity_backend(similar_nodes)

    node_label, node_weight, node_similarity_value = _make_helpers(
        node_weights, canonical_map, similar_nodes, backend
    )
    node_del_cost, node_ins_cost, node_subst_cost, edge_subst_cost = _build_cost_functions(
        node_label, node_weight, node_similarity_value
    )
    return {
        "node_weight": node_weight,
        "node_del_cost": node_del_cost,
        "node_ins_cost": node_ins_cost,
        "node_subst_cost": node_subst_cost,
        "edge_subst_cost": edge_subst_cost,
    }


def _compute_ged_with_context(g1, g2, context, timeout=None):
    ged = nx.graph_edit_distance(
        g1,
        g2,
        node_del_cost=context["node_del_cost"],
        node_ins_cost=context["node_ins_cost"],
        node_subst_cost=context["node_subst_cost"],
        edge_del_cost=lambda a: 1.0,
        edge_ins_cost=lambda a: 1.0,
        edge_subst_cost=context["edge_subst_cost"],
        timeout=timeout,
    )

    if ged is None:
        return {"ged": None, "similarity": None, "c_max": None}

    return _normalize_similarity(g1, g2, ged, context["node_weight"])


def compute_ged(g1, g2, context, timeout=None):
    return _compute_ged_with_context(g1, g2, context, timeout=timeout)


if __name__ == "__main__":
    graphs_path = "/home/mattia/geass/NPA_fragmentation/NPA_graphs_ann.pkl"
    node_similarity_path = "/home/mattia/geass/NPA_subgraphs/prep_files/NPA_node_similarity_atomp.pkl"
    node_weights_path = "/home/mattia/geass/NPA_subgraphs/prep_files/NPA_node_weights.pkl"
    out_path = "/home/mattia/geass/NPA_subgraphs/prep_files/NPA_graph_similarity_ged.pkl"

    with open(graphs_path, "rb") as f:
        npa_graphs = pickle.load(f)
    with open(node_similarity_path, "rb") as f:
        node_similarity = pickle.load(f)
    with open(node_weights_path, "rb") as f:
        node_weights = pickle.load(f)
    context = _prepare_ged_context(node_similarity, node_weights)

    ids = list(npa_graphs.keys())
    n = len(ids)

    n_nodes = np.fromiter((npa_graphs[gid].number_of_nodes() for gid in ids), dtype=np.int32, count=n)
    n_edges = np.fromiter((npa_graphs[gid].number_of_edges() for gid in ids), dtype=np.int32, count=n)
    node_label_sets = []
    for gid in ids:
        g = npa_graphs[gid]
        labels = set()
        for _, attrs in g.nodes(data=True):
            label = attrs.get("MUID", attrs.get("label"))
            if label is not None:
                labels.add(label)
        node_label_sets.append(labels)

    order = np.argsort(n_nodes, kind="mergesort")
    order_list = order.tolist()
    sorted_n_nodes = n_nodes[order]

    sim_matrix = np.zeros((n, n), dtype=np.float32)
    np.fill_diagonal(sim_matrix, 1.0)

    for i in tqdm(range(n), total=n):
        ga = npa_graphs[ids[i]]
        na = n_nodes[i]
        ea = n_edges[i]

        lo_nodes = int(np.ceil(na * 0.3))
        hi_nodes = int(np.floor(na / 0.3))
        left = bisect.bisect_left(sorted_n_nodes, lo_nodes)
        right = bisect.bisect_right(sorted_n_nodes, hi_nodes)

        for pos in range(left, right):
            j = order_list[pos]
            if j <= i:
                continue

            eb = n_edges[j]
            if ea > 0 and eb > 0 and min(ea, eb) / max(ea, eb) < 0.3:
                continue

            if node_label_sets[i].isdisjoint(node_label_sets[j]):
                continue

            gb = npa_graphs[ids[j]]
            result = _compute_ged_with_context(ga, gb, context)
            score = result["similarity"]
            if score is None:
                continue

            sim_matrix[i, j] = score
            sim_matrix[j, i] = score

    with open(out_path, "wb") as f:
        pickle.dump(sim_matrix, f, protocol=pickle.HIGHEST_PROTOCOL)
