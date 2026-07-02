from collections import Counter, defaultdict
import bisect
import pickle
import os
from scipy import sparse

import numpy as np
from joblib import Parallel, delayed
from tqdm.auto import tqdm


# ------------------------------------------------------------------ #
# Constants                                                            #
# ------------------------------------------------------------------ #

SIMILARITY_BACKEND_CACHE = {}
NODE_FILTERED_NEIGHBOUR_CACHE = {}

EPS = 1e-12
MIN_SIZE_RATIO = 0.5
CHUNK_SIZE = 200
NODE_MIN_SIMILARITY = 0.35
MAX_NEIGHBORS_PER_NODE = 50
STRONG_SIMILARITY_FLOOR = 0.7

NODE_FILTERED_NEIGHBOUR_CACHE_MAXSIZE = 200_000

# SOFT_MATCH_ALPHA removed: matched_size now uses sim directly (smooth,
# no binary exact/soft split).  See Fix 1 in sgo_graphs.


# ------------------------------------------------------------------ #
# 2. Similarity backend                                                #
#                                                                      #
# Requires node similarity data in upper_csr_v1 format.                #
# The upper-triangular CSR is expanded on first use into a symmetric    #
# adjacency dict for O(1) lookups.                                      #
# ------------------------------------------------------------------ #


def _get_similarity_backend(similar_nodes):
    """
    Build and cache the similarity backend for similar_nodes.

    The upper-triangular CSR is expanded once into a
    symmetric {label: {neighbour: weight}} adjacency dict so that all
    subsequent lookups are two O(1) dict accesses.
    """
    key = id(similar_nodes)
    backend = SIMILARITY_BACKEND_CACHE.get(key)
    if backend is not None:
        return backend

    if not isinstance(similar_nodes, dict) or similar_nodes.get("format") != "upper_csr_v1":
        raise ValueError("node_similarity['similar_nodes'] must use format='upper_csr_v1'")

    nodes = similar_nodes["nodes"]
    row_ptr = np.asarray(similar_nodes["row_ptr"])
    col_idx = np.asarray(similar_nodes["col_idx"])
    values = np.asarray(similar_nodes["values"])

    adj = defaultdict(dict)
    for i, label_i in enumerate(nodes):
        start = int(row_ptr[i])
        end = int(row_ptr[i + 1])
        for k in range(start, end):
            j = int(col_idx[k])
            w = float(values[k])
            label_j = nodes[j]
            adj[label_i][label_j] = w
            adj[label_j][label_i] = w   # fill lower triangle

    backend = {"adj": dict(adj)}

    SIMILARITY_BACKEND_CACHE[key] = backend
    return backend


def _filtered_neighbours(
    similar_nodes, node,
    min_similarity=NODE_MIN_SIMILARITY,
    max_neighbors=MAX_NEIGHBORS_PER_NODE,
    strong_floor=STRONG_SIMILARITY_FLOOR,
):
    """
    Runtime node-similarity filtering:
    - keep only similarities >= min_similarity
    - keep top-k neighbours by similarity
    - always keep neighbours >= strong_floor, even beyond top-k
    """
    cache_key = (id(similar_nodes), node, float(min_similarity), int(max_neighbors), float(strong_floor))
    cached = NODE_FILTERED_NEIGHBOUR_CACHE.get(cache_key)
    if cached is not None:
        return cached

    raw = _get_similarity_backend(similar_nodes)["adj"].get(node, {})

    base = [(nb, float(w)) for nb, w in raw.items() if float(w) >= min_similarity]
    if not base:
        NODE_FILTERED_NEIGHBOUR_CACHE[cache_key] = {}
        return {}

    strong = {nb: w for nb, w in base if w >= strong_floor}

    if max_neighbors is None or max_neighbors <= 0 or len(base) <= max_neighbors:
        kept = {nb: w for nb, w in base}
    else:
        top = sorted(base, key=lambda x: x[1], reverse=True)[:max_neighbors]
        kept = {nb: w for nb, w in top}

    for nb, w in strong.items():
        if w > kept.get(nb, 0.0):
            kept[nb] = w

    if len(NODE_FILTERED_NEIGHBOUR_CACHE) >= NODE_FILTERED_NEIGHBOUR_CACHE_MAXSIZE:
        NODE_FILTERED_NEIGHBOUR_CACHE.clear()

    NODE_FILTERED_NEIGHBOUR_CACHE[cache_key] = kept
    return kept


def _node_size(node_sizes, node):
    """Return the weight of node, defaulting to 1.0 if absent."""
    return float(node_sizes.get(node, 1.0))


# ------------------------------------------------------------------ #
# 5. Node scoring                                                      #
# ------------------------------------------------------------------ #

def build_node_candidate_lists(rem_a, rem_b, similar_nodes, node_sizes):
    """
    Build per-node candidate match lists for the soft node assignment.

    Each candidate weight is sim * min(size_a, size_b).

    The known neighbours of u are visited in O(degree(u)).
    """
    node_to_b = defaultdict(list)
    for j, n_b in enumerate(rem_b):
        node_to_b[n_b].append(j)

    candidate_lists = []
    has_any_candidate = False

    adj = _get_similarity_backend(similar_nodes)["adj"]

    for u in rem_a:
        wu = _node_size(node_sizes, u)
        row = []
        seen = set()

        # O(degree): self-match + adj neighbours that appear in rem_b
        candidates = {u: 1.0}
        for t, w in adj.get(u, {}).items():
            if t in node_to_b:
                candidates[t] = w

        for s, sim in candidates.items():
            js = node_to_b.get(s)
            if not js:
                continue

            w = sim * min(wu, _node_size(node_sizes, s))
            if w <= 0.0:
                continue

            for j in js:
                if j not in seen:
                    seen.add(j)
                    row.append((j, w))

        if row:
            row.sort(key=lambda x: x[1], reverse=True)
            has_any_candidate = True

        candidate_lists.append(row)

    return candidate_lists, has_any_candidate


# ------------------------------------------------------------------ #
# 6. Edge scoring                                                      #
# ------------------------------------------------------------------ #

def build_edge_candidate_lists(rem_a, rem_b, similar_nodes, node_sizes):
    """
    Build per-edge candidate match lists for the soft edge assignment.

    Each candidate weight is the best-orientation weighted average of endpoint
    similarities. Only known neighbours of each endpoint are visited.
    """
    node_to_edges = defaultdict(list)
    for j, (x, y) in enumerate(rem_b):
        node_to_edges[x].append(j)
        node_to_edges[y].append(j)

    node_to_edges_get = node_to_edges.get
    node_sizes_get = node_sizes.get
    target_nodes = set(node_to_edges)
    candidate_lists = []
    has_any_candidate = False

    adj = _get_similarity_backend(similar_nodes)["adj"]

    for u, v in rem_a:
        # sim_u/sim_v: target node -> similarity with u/v (self-similarity seeded at 1.0)
        sim_u = {u: 1.0}
        sim_v = {v: 1.0}

        # O(degree): iterate only known neighbours, filter to target nodes
        for t, wu in adj.get(u, {}).items():
            if t in target_nodes:
                sim_u[t] = wu
        for t, wv in adj.get(v, {}).items():
            if t in target_nodes:
                sim_v[t] = wv

        w_u = node_sizes_get(u, 1.0)
        w_v = node_sizes_get(v, 1.0)
        w_tot = w_u + w_v

        row = []
        seen = set()

        for endpoint_sim in (sim_u, sim_v):
            for s in endpoint_sim:
                for j in node_to_edges_get(s, ()):
                    if j in seen:
                        continue
                    seen.add(j)
                    x, y = rem_b[j]
                    # Score both orientations of the candidate edge and take the best
                    score_xy = (w_u * sim_u.get(x, 0.0) + w_v * sim_v.get(y, 0.0)) / w_tot
                    score_yx = (w_u * sim_u.get(y, 0.0) + w_v * sim_v.get(x, 0.0)) / w_tot
                    w = max(score_xy, score_yx)
                    if w > 0.0:
                        row.append((j, w))

        if row:
            row.sort(key=lambda x: x[1], reverse=True)
            has_any_candidate = True

        candidate_lists.append(row)

    return candidate_lists, has_any_candidate


def _greedy_node_match(
    graph_a_data, graph_b_data, node_similarity,
    min_similarity=NODE_MIN_SIMILARITY,
    max_neighbors=MAX_NEIGHBORS_PER_NODE,
    strong_floor=STRONG_SIMILARITY_FLOOR,
):
    similar_nodes = node_similarity["similar_nodes"]
    labels_a = graph_a_data["node_labels"]
    labels_b = graph_b_data["node_labels"]

    if not labels_a or not labels_b:
        return [], 0.0

    # Use precomputed label->index map when available; build on-the-fly otherwise.
    positions_b = graph_b_data.get("positions")
    if positions_b is None:
        positions_b = defaultdict(list)
        for j, label in enumerate(labels_b):
            positions_b[label].append(j)

    candidates = []
    seen = set()

    for i, label_a in enumerate(labels_a):
        label_candidates = {label_a: 1.0}

        for label_b, sim in _filtered_neighbours(
            similar_nodes, label_a,
            min_similarity=min_similarity,
            max_neighbors=max_neighbors,
            strong_floor=strong_floor,
        ).items():
            if label_b in positions_b and sim > 0.0:
                label_candidates[label_b] = sim

        for label_b, sim in label_candidates.items():
            for j in positions_b.get(label_b, ()):
                key = (i, j)
                if key not in seen:
                    seen.add(key)
                    candidates.append((sim, i, j))

    candidates.sort(reverse=True)

    matched_a = set()
    matched_b = set()
    pairs = []
    sim_sum = 0.0

    for sim, i, j in candidates:
        if i in matched_a or j in matched_b:
            continue
        matched_a.add(i)
        matched_b.add(j)
        pairs.append((i, j, sim))
        sim_sum += sim
        if len(matched_a) == len(labels_a) or len(matched_b) == len(labels_b):
            break

    return pairs, sim_sum


def _adjacency_penalty(graph_a_data, graph_b_data, matched_pairs, node_sizes=None):
    # With fewer than two matched nodes we cannot evaluate edge preservation.
    # Treat adjacency as unknown instead of maximally penalising.
    if len(matched_pairs) < 2:
        return 0.0

    # ---- edge set for b -----------------------------------------------
    # Use the precomputed frozenset when available; build on-the-fly otherwise.
    edge_pairs_b = graph_b_data.get("edge_pairs_set")
    if edge_pairs_b is None:
        edge_pairs_b = {(min(i, j), max(i, j)) for i, j in graph_b_data["edge_pairs"]}

    # ---- per-node sizes -----------------------------------------------
    sizes_a = graph_a_data.get("node_size_list")
    sizes_b = graph_b_data.get("node_size_list")
    if sizes_a is None:
        labels_a = graph_a_data["node_labels"]
        sizes_a = [_node_size(node_sizes, labels_a[k]) for k in range(len(labels_a))]
    if sizes_b is None:
        labels_b = graph_b_data["node_labels"]
        sizes_b = [_node_size(node_sizes, labels_b[k]) for k in range(len(labels_b))]

    # ---- total edge sizes ---------------------------------------------
    total_edge_size_a = graph_a_data.get("total_edge_size")
    if total_edge_size_a is None:
        total_edge_size_a = sum(
            0.5 * (sizes_a[u] + sizes_a[v])
            for u, v in graph_a_data["edge_pairs"])
    total_edge_size_b = graph_b_data.get("total_edge_size")
    if total_edge_size_b is None:
        total_edge_size_b = sum(
            0.5 * (sizes_b[u] + sizes_b[v])
            for u, v in graph_b_data["edge_pairs"])

    # ---- preserved edge size ------------------------------------------
    map_a_to_b = {i: (j, sim) for i, j, sim in matched_pairs}

    preserved_size = 0.0
    for u, v in graph_a_data["edge_pairs"]:
        mu = map_a_to_b.get(u)
        mv = map_a_to_b.get(v)
        if mu is None or mv is None:
            continue
        ju, su = mu
        jv, sv = mv
        if (min(ju, jv), max(ju, jv)) in edge_pairs_b:
            edge_size = 0.5 * (sizes_a[u] + sizes_a[v])
            # Weight preserved edges by the minimum endpoint match quality,
            # consistent with the smooth sim contribution in sgo_graphs.
            preserved_size += edge_size * min(float(su), float(sv))

    denom = max(total_edge_size_a, total_edge_size_b, EPS)
    return 1.0 - (preserved_size / denom)


# ------------------------------------------------------------------ #
# 7. Graph scoring                                                     #
# ------------------------------------------------------------------ #

def sgo_graphs(
    graph_a_data, graph_b_data, node_similarity, node_sizes,
    w_nodes=0.50, w_size=0.15, w_degree=0.15, w_adj=0.20,
    node_min_similarity=NODE_MIN_SIMILARITY,
    max_neighbors_per_node=MAX_NEIGHBORS_PER_NODE,
    strong_similarity_floor=STRONG_SIMILARITY_FLOOR,
):
    """
    Combined graph similarity using adaptive additive penalty accumulation.

    """
    na = graph_a_data["n_nodes"]
    nb = graph_b_data["n_nodes"]
    ea = graph_a_data["n_edges"]
    eb = graph_b_data["n_edges"]

    if na == 0 and nb == 0:
        return 1.0
    if max(na, nb) > 0 and min(na, nb) / max(na, nb) < MIN_SIZE_RATIO:
        return 0.0

    matched_pairs, _ = _greedy_node_match(
        graph_a_data, graph_b_data, node_similarity,
        min_similarity=node_min_similarity,
        max_neighbors=max_neighbors_per_node,
        strong_floor=strong_similarity_floor,)

    total_size_a = graph_a_data["total_size"]
    total_size_b = graph_b_data["total_size"]
    size_denom = max(total_size_a, total_size_b, EPS)
    sizes_a = graph_a_data["node_size_list"]
    sizes_b = graph_b_data["node_size_list"]

    matched_size = 0.0
    for i, j, sim in matched_pairs:
        sa = sizes_a[i]
        sb = sizes_b[j]
        matched_size += sim * min(sa, sb)

    node_penalty = 1.0 - (matched_size / size_denom)
    size_penalty = abs(total_size_a - total_size_b) / size_denom

    degree_penalty = 0.0
    if matched_pairs:
        degree_num = 0.0
        degree_den = 0.0
        for i, j, _ in matched_pairs:
            norm_d1 = graph_a_data["degree_list"][i] / max(na - 1, 1)
            norm_d2 = graph_b_data["degree_list"][j] / max(nb - 1, 1)
            sa = sizes_a[i]
            sb = sizes_b[j]
            w = min(sa, sb)
            degree_num += w * abs(norm_d1 - norm_d2) * min(na, nb) / max(na, nb)
            degree_den += w
        degree_penalty = (degree_num / degree_den) if degree_den > 0 else 0.0

    adj_penalty = _adjacency_penalty(graph_a_data, graph_b_data, matched_pairs, node_sizes)

    # Scale adjacency contribution by size-weighted node-match coverage.
    matched_size_raw = 0.0
    for i, j, _ in matched_pairs:
        sa = sizes_a[i]
        sb = sizes_b[j]
        matched_size_raw += min(sa, sb)
    match_coverage = matched_size_raw / size_denom
    adj_penalty *= match_coverage

    active_w_size = w_size   if size_penalty   > EPS else 0.0
    active_w_degree = w_degree if max(na, nb) > 1      else 0.0
    active_w_adj = w_adj    if max(ea, eb) > 0       else 0.0

    active_denom = w_nodes + active_w_size + active_w_degree + active_w_adj

    distance = (
        w_nodes        * node_penalty
        + active_w_size   * size_penalty
        + active_w_degree * degree_penalty
        + active_w_adj    * adj_penalty
    ) / active_denom

    return max(0.0, 1.0 - distance)


# ------------------------------------------------------------------ #
# 8. Parallel worker                                                   #
# ------------------------------------------------------------------ #

def _compute_row(i, graph_data_list, node_similarity, node_sizes,
                 order_list, sorted_n_nodes, n_nodes, n_edges, threshold):
    """
    Compute all pairwise scores for graph i against every j > i that
    passes the MIN_SIZE_RATIO filter.  Returns a list of (i, j, score).
    """
    ga = graph_data_list[i]
    na = int(n_nodes[i])

    # Binary-search window: only graphs whose node count is within the ratio
    lo = int(np.ceil(na * MIN_SIZE_RATIO))
    hi = int(np.floor(na / MIN_SIZE_RATIO))
    left = bisect.bisect_left(sorted_n_nodes, lo)
    right = bisect.bisect_right(sorted_n_nodes, hi)

    reachable_a = ga["reachable_nodes"]
    count_a_keys = ga["node_counter"].keys()

    results = []
    for pos in range(left, right):
        j = order_list[pos]
        if j <= i:
            continue

        gb = graph_data_list[j]
        count_b_keys = gb["node_counter"].keys()
        reachable_b = gb["reachable_nodes"]

        # Skip pairs with no possible exact/soft node match according to
        # reachable-node sets.
        if not (reachable_a & count_b_keys) and not (reachable_b & count_a_keys):
            continue

        # Cheap upper bound based on unavoidable size-driven node penalty.
        total_size_a = ga["total_size"]
        total_size_b = gb["total_size"]
        size_denom = max(total_size_a, total_size_b, EPS)
        size_penalty = abs(total_size_a - total_size_b) / size_denom
        active_w_size = 0.15 if size_penalty > EPS else 0.0
        sim_upper_bound = 1.0 - (0.50 + active_w_size) * size_penalty
        if sim_upper_bound + EPS < threshold:
            continue

        score = sgo_graphs(ga, gb, node_similarity, node_sizes)
        if score != 0.0:
            results.append((i, j, float(score)))

    return results


def _compute_chunk(row_indices, graph_data_list, node_similarity, node_sizes,
                   order_list, sorted_n_nodes, n_nodes, n_edges, threshold):
    """
    Process a batch of row indices in a single worker call.

    Batching multiple rows per task dramatically reduces the number of times
    joblib has to pickle and ship graph_data_list and node_similarity across
    process boundaries.
    """
    results = []
    for i in row_indices:
        results.extend(
            _compute_row(i, graph_data_list, node_similarity, node_sizes,
                         order_list, sorted_n_nodes, n_nodes, n_edges, threshold))
    return results


# ------------------------------------------------------------------ #
# 9. Class wrapper                                                     #
# ------------------------------------------------------------------ #

class SgoSimilarity:
    DEFAULT_GRAPHS_PATH = "/home/mattia/geass/NPA_fragmentation/NPA_graphs_map4.pkl"
    DEFAULT_SIMILARITY_PATH = "/home/mattia/geass/NPA_subgraphs/prep_files/NPA_node_similarity_map4.pkl"
    DEFAULT_SIZES_PATH = "/home/mattia/geass/NPA_subgraphs/prep_files/NPA_node_weights.pkl"

    def __init__(self, graphs_path=DEFAULT_GRAPHS_PATH,
                similarity_path=DEFAULT_SIMILARITY_PATH,
                sizes_path=DEFAULT_SIZES_PATH):

        with open(graphs_path, "rb") as f:
            npa_graphs = pickle.load(f)
        with open(similarity_path, "rb") as f:
            node_similarity = pickle.load(f)
        with open(sizes_path, "rb") as f:
            node_sizes = pickle.load(f)

        self.npa_graphs = npa_graphs
        self.node_similarity = node_similarity
        self.node_sizes = node_sizes
        self._fp_engine = None
        self.canonical_map = node_similarity.get("canonical_map", {})
        similar_nodes = node_similarity.get("similar_nodes")
        self._node_similarity_ids = set(similar_nodes["nodes"])

    @classmethod
    def from_files(
        cls,
        graphs_path=DEFAULT_GRAPHS_PATH,
        similarity_path=DEFAULT_SIMILARITY_PATH,
        sizes_path=DEFAULT_SIZES_PATH,
    ):
        return cls(
            graphs_path=graphs_path,
            similarity_path=similarity_path,
            sizes_path=sizes_path)

    def preprocess_graph(self, graph):
        node_items = list(graph.nodes(data=True))
        node_index = {node: i for i, (node, _) in enumerate(node_items)}
        node_muids = [
            self.canonical_map.get(attrs["MUID"], attrs["MUID"])
            for _, attrs in node_items]

        edge_muids = [
            tuple(sorted((
                self.canonical_map.get(graph.nodes[u]["MUID"], graph.nodes[u]["MUID"]),
                self.canonical_map.get(graph.nodes[v]["MUID"], graph.nodes[v]["MUID"])
            )))
            for u, v in graph.edges()]
        edge_pairs = [
            (node_index[u], node_index[v])
            for u, v in graph.edges()]

        degree_list = [0] * len(node_muids)
        for i, j in edge_pairs:
            degree_list[i] += 1
            degree_list[j] += 1

        node_counter = Counter(node_muids)
        edge_counter = Counter(edge_muids)

        edge_neighbour_index = defaultdict(set)
        for j, (x, y) in enumerate(edge_muids):
            edge_neighbour_index[x].add(j)
            edge_neighbour_index[y].add(j)

        _get = self.node_sizes.get if self.node_sizes is not None else None
        node_size_list = (
            [float(_get(m, 1.0)) for m in node_muids]
            if _get is not None
            else [1.0] * len(node_muids))
        total_size = sum(node_size_list)

        positions: dict = {}
        for idx, label in enumerate(node_muids):
            if label in positions:
                positions[label].append(idx)
            else:
                positions[label] = [idx]

        edge_pairs_set = {(min(u, v), max(u, v)) for u, v in edge_pairs}

        total_edge_size = sum(
            0.5 * (node_size_list[u] + node_size_list[v])
            for u, v in edge_pairs)

        return {
            "n_nodes":              len(node_muids),
            "n_edges":              len(edge_muids),
            "node_counter":         node_counter,
            "edge_counter":         edge_counter,
            "edge_neighbour_index": edge_neighbour_index,
            "node_labels":          node_muids,
            "edge_pairs":           edge_pairs,
            "degree_list":          degree_list,
            "reachable_nodes":      None,
            "node_size_list":       node_size_list,
            "total_size":           total_size,
            "positions":            positions,
            "edge_pairs_set":       edge_pairs_set,
            "total_edge_size":      total_edge_size}

    def score_pair(self, graph_a_data, graph_b_data):
        return sgo_graphs(
            graph_a_data,
            graph_b_data,
            self.node_similarity,
            self.node_sizes)

    def score_pair_by_id(self, id_a, id_b):
        ga = self.preprocess_graph(self.npa_graphs[id_a])
        gb = self.preprocess_graph(self.npa_graphs[id_b])
        return self.score_pair(ga, gb)


    def compute_similarity_matrix_sparse(self, n_jobs=-1, chunk_size=CHUNK_SIZE,
            threshold=0.05):
        ids = list(self.npa_graphs.keys())
        n = len(ids)

        graph_data_list = [self.preprocess_graph(self.npa_graphs[gid]) for gid in ids]

        adj = _get_similarity_backend(self.node_similarity["similar_nodes"])["adj"]
        for g in graph_data_list:
            own = set(g["node_counter"])
            soft = {nb for node in own for nb in adj.get(node, {})}
            g["reachable_nodes"] = own | soft

        n_nodes = np.fromiter((g["n_nodes"] for g in graph_data_list), dtype=np.int32, count=n)
        n_edges = np.fromiter((g["n_edges"] for g in graph_data_list), dtype=np.int32, count=n)

        order = np.argsort(n_nodes, kind="mergesort")
        order_list = order.tolist()
        sorted_n_nodes = n_nodes[order].tolist()

        chunks = [list(range(i, min(i + chunk_size, n))) for i in range(0, n, chunk_size)]

        # Start with identity; off-diagonal values are streamed in batches.
        sim_csr = sparse.identity(n, dtype=np.float32, format="csr")

        # Process chunks in bounded batches to avoid materializing all pair
        # results in memory at once.
        if n_jobs in (None, 0):
            n_jobs_eff = 1
        elif n_jobs == -1:
            n_jobs_eff = os.cpu_count() or 1
        else:
            n_jobs_eff = abs(int(n_jobs))
        batch_chunk_count = max(1, 4 * n_jobs_eff)
        with Parallel(n_jobs=n_jobs) as parallel:
            for batch_start in tqdm(range(0, len(chunks), batch_chunk_count), total=(len(chunks) + batch_chunk_count - 1) // batch_chunk_count, desc="chunk_batches"):
                chunk_batch = chunks[batch_start: batch_start + batch_chunk_count]
                batch_results = parallel(
                    delayed(_compute_chunk)(
                        chunk, graph_data_list, self.node_similarity, self.node_sizes,
                        order_list, sorted_n_nodes, n_nodes, n_edges, threshold)
                    for chunk in chunk_batch)

                # off-diagonal, thresholded, mirrored for symmetry
                batch_rows = []
                batch_cols = []
                batch_vals = []
                for chunk_rows in batch_results:
                    for i, j, score in chunk_rows:
                        s = float(np.clip(score, 0.0, 1.0))
                        if s < threshold:
                            continue
                        batch_rows.append(i); batch_cols.append(j); batch_vals.append(s)
                        batch_rows.append(j); batch_cols.append(i); batch_vals.append(s)

                if batch_vals:
                    batch_csr = sparse.coo_matrix(
                        (np.asarray(batch_vals, dtype=np.float32),
                         (np.asarray(batch_rows, dtype=np.int32),
                          np.asarray(batch_cols, dtype=np.int32))),
                        shape=(n, n),
                    ).tocsr()
                    sim_csr += batch_csr

        sim_csr.sum_duplicates()
        np.clip(sim_csr.data, 0.0, 1.0, out=sim_csr.data)
        return ids, sim_csr


# ------------------------------------------------------------------ #
# 10. Entry point                                                    #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    GRAPHS_PATH = "/home/mattia/geass/NPA_fragmentation/NPA_graphs_map4.pkl"
    SIMILARITY_PATH = "/home/mattia/geass/NPA_subgraphs/prep_files/NPA_node_similarity_map4.pkl"
    SIZES_PATH = "/home/mattia/geass/NPA_subgraphs/prep_files/NPA_node_weights.pkl"
    OUTPUT_FNAME = "/home/mattia/geass/NPA_analysis/NPA_sim_matrix_sgo"

    engine = SgoSimilarity.from_files(GRAPHS_PATH, SIMILARITY_PATH, SIZES_PATH)

    ids, sim_csr = engine.compute_similarity_matrix_sparse(threshold=0.05)
    sparse.save_npz(f"{OUTPUT_FNAME}.npz", sim_csr, compressed=True)
    np.save(f"{OUTPUT_FNAME}_ids.npy", np.array(ids))
