import networkx as nx
import numpy as np
from rdkit import Chem
from networkx.algorithms.isomorphism import ISMAGS

_MAP4_CALCULATOR = None
_MAP4_FP_CACHE = {}
SIM_THRESHOLD = 0.20


def _get_map4_calculator():
    global _MAP4_CALCULATOR
    if _MAP4_CALCULATOR is None:
        from map4 import MAP4Calculator
        _MAP4_CALCULATOR = MAP4Calculator(dimensions=1024)
    return _MAP4_CALCULATOR


def _map4_fp_from_smiles(smiles):
    if not smiles:
        return None
    cached = _MAP4_FP_CACHE.get(smiles)
    if cached is not None:
        return cached
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    can_smi = Chem.MolToSmiles(mol, isomericSmiles=False)
    can_mol = Chem.MolFromSmiles(can_smi)
    if can_mol is None:
        return None
    fp = np.asarray(_get_map4_calculator().calculate(can_mol), dtype=np.uint32)
    _MAP4_FP_CACHE[smiles] = fp
    return fp


def _map4_similarity(smiles_a, smiles_b):
    fp_a = _map4_fp_from_smiles(smiles_a)
    fp_b = _map4_fp_from_smiles(smiles_b)
    if fp_a is None or fp_b is None:
        return 0.0
    return float((fp_a == fp_b).mean())


def _default_node_metadata_with_smiles(smiles, similarity=0.0):
    iso_smiles = smiles or ""
    return {
        "SMILES": iso_smiles,
        "ISOSMILES": iso_smiles,
        "MUID": "",
        "Similarity": float(similarity),
        "Annotation": "",
        "isNP": "No",
        "img": "",
    }


def maximum_common_subgraphs(
    G,
    H,
    node_attr="MUID",
    edge_attr="reaction",
    node_similarity_path=None,
    sim_threshold=SIM_THRESHOLD,
    allow_similar_nodes=True,
):
    def node_match(a, b):
        la = a.get(node_attr)
        lb = b.get(node_attr)
        if la == lb:
            return True
        if not allow_similar_nodes:
            return False
        return _map4_similarity(a.get("SMILES"), b.get("SMILES")) >= float(sim_threshold)

    def edge_match(a, b):
        return a.get(edge_attr) == b.get(edge_attr)

    matcher = ISMAGS(G, H, node_match=node_match, edge_match=edge_match)
    mappings = list(matcher.largest_common_subgraph())

    results = []
    for mapping in mappings:
        nodes_G = list(mapping.keys())
        nodes_H = list(mapping.values())

        subgraph_G = G.subgraph(nodes_G).copy()
        subgraph_H = H.subgraph(nodes_H).copy()

        for g_node, h_node in mapping.items():
            data_g = G.nodes[g_node]
            data_h = H.nodes[h_node]
            if data_g.get(node_attr) == data_h.get(node_attr):
                continue
            smi_g = data_g.get("SMILES") or ""
            smi_h = data_h.get("SMILES") or ""
            rep_smiles = smi_g or smi_h
            sim_score = _map4_similarity(smi_g, smi_h)
            rep_data = _default_node_metadata_with_smiles(rep_smiles, similarity=sim_score)
            subgraph_G.nodes[g_node].clear()
            subgraph_G.nodes[g_node].update(rep_data)
            subgraph_H.nodes[h_node].clear()
            subgraph_H.nodes[h_node].update(rep_data)

        results.append(
            {
                "subgraph_G": subgraph_G,
                "subgraph_H": subgraph_H,
                "mapping_G_to_H": dict(mapping),
                "mapping_H_to_G": {v: k for k, v in mapping.items()},
                "n_nodes": subgraph_G.number_of_nodes(),
                "n_edges": subgraph_G.number_of_edges()
            }
        )

    return results


def get_best_intersection(
    G,
    H,
    node_attr="MUID",
    edge_attr="reaction",
    node_similarity_path=None,
    sim_threshold=SIM_THRESHOLD,
    allow_similar_nodes=True,
):
    results = maximum_common_subgraphs(
        G,
        H,
        node_attr=node_attr,
        edge_attr=edge_attr,
        node_similarity_path=node_similarity_path,
        sim_threshold=sim_threshold,
        allow_similar_nodes=allow_similar_nodes,
    )

    if not results:
        return None

    return max(
        results,
        key=lambda r: (r["n_nodes"], r["n_edges"])
    )


def _normalise_node_label(value):
    if isinstance(value, list):
        return tuple(value)
    if isinstance(value, set):
        return tuple(sorted(value))
    if isinstance(value, dict):
        return tuple(sorted(value.items()))
    return value


def _candidate_signature(
    graph,
    node_attr="MUID",
    edge_attr="reaction"
):
    node_counts = {}
    for _, data in graph.nodes(data=True):
        label = _normalise_node_label(data.get(node_attr))
        node_counts[label] = node_counts.get(label, 0) + 1

    edge_counts = {}
    for u, v, data in graph.edges(data=True):
        lu = _normalise_node_label(graph.nodes[u].get(node_attr))
        lv = _normalise_node_label(graph.nodes[v].get(node_attr))
        if lu <= lv:
            edge_key = (lu, lv, _normalise_node_label(data.get(edge_attr)))
        else:
            edge_key = (lv, lu, _normalise_node_label(data.get(edge_attr)))
        edge_counts[edge_key] = edge_counts.get(edge_key, 0) + 1

    return (
        tuple(sorted(node_counts.items())),
        tuple(sorted(edge_counts.items()))
    )


def _deduplicate_candidates(
    candidates,
    node_attr="MUID",
    edge_attr="reaction"
):
    unique = {}
    for candidate in candidates:
        sig = _candidate_signature(
            candidate["graph"],
            node_attr=node_attr,
            edge_attr=edge_attr
        )

        current = unique.get(sig)
        if current is None:
            unique[sig] = candidate
            continue

        current_score = (
            current["graph"].number_of_nodes(),
            current["graph"].number_of_edges()
        )
        new_score = (
            candidate["graph"].number_of_nodes(),
            candidate["graph"].number_of_edges()
        )

        if new_score > current_score:
            unique[sig] = candidate

    return list(unique.values())


def _next_available_node_id(graph):
    int_ids = [n for n in graph.nodes() if isinstance(n, int)]
    return (max(int_ids) + 1) if int_ids else 0


def _pick_representative_node_data(node_data_list):
    if not node_data_list:
        return {}
    if len(node_data_list) == 1:
        return _default_node_metadata_with_smiles(
            node_data_list[0].get("SMILES", ""),
            similarity=1.0
        )

    smiles = [d.get("SMILES", "") for d in node_data_list]
    best_idx = 0
    best_score = -1.0
    for i, smi_i in enumerate(smiles):
        score = 0.0
        for j, smi_j in enumerate(smiles):
            if i == j:
                score += 1.0
            else:
                score += _map4_similarity(smi_i, smi_j)
        if score > best_score:
            best_score = score
            best_idx = i
    rep_smiles = node_data_list[best_idx].get("SMILES", "")
    avg_sim = sum(_map4_similarity(rep_smiles, smi) for smi in smiles) / max(len(smiles), 1)
    return _default_node_metadata_with_smiles(rep_smiles, similarity=avg_sim)


def _cluster_by_similarity(records, similarity_threshold):
    clusters = []
    for rec in records:
        rec_smiles = rec["node_data"].get("SMILES", "")
        assigned = False
        for cluster in clusters:
            rep_smiles = cluster["records"][0]["node_data"].get("SMILES", "")
            if _map4_similarity(rec_smiles, rep_smiles) >= similarity_threshold:
                cluster["records"].append(rec)
                assigned = True
                break
        if not assigned:
            clusters.append({"records": [rec]})
    return clusters


def _build_augmented_template(
    consensus,
    source_graphs,
    consensus_to_source_maps,
    node_attr="MUID",
    edge_attr="reaction",
    similarity_threshold=0.35
):
    template = consensus.copy()
    next_node_id = _next_available_node_id(template)
    added_node_owner = {}

    terminal_records = {}
    path_variants = {}
    path_internal_nodes_by_graph = {}

    for graph_idx, (graph, c2s) in enumerate(zip(source_graphs, consensus_to_source_maps)):
        if graph is None:
            continue
        inv = {src: cons for cons, src in c2s.items()}
        for cons_node, src_node in c2s.items():
            if src_node not in graph:
                continue
            for nbr in graph.neighbors(src_node):
                if nbr in inv:
                    continue
                edge_data = dict(graph.get_edge_data(src_node, nbr) or {})
                node_data = dict(graph.nodes[nbr])
                key = (
                    cons_node,
                    _normalise_node_label(edge_data.get(edge_attr))
                )
                terminal_records.setdefault(key, []).append(
                    {
                        "graph_idx": graph_idx,
                        "source_node": nbr,
                        "node_data": node_data,
                        "edge_data": edge_data
                    }
                )

        for u_c, v_c in consensus.edges():
            if u_c not in c2s or v_c not in c2s:
                continue
            u_s = c2s[u_c]
            v_s = c2s[v_c]
            if u_s not in graph or v_s not in graph:
                continue

            try:
                path = nx.shortest_path(graph, source=u_s, target=v_s)
            except nx.NetworkXNoPath:
                continue

            if len(path) <= 2:
                continue

            internal = path[1:-1]
            if any(node in inv for node in internal):
                continue
            path_internal_nodes_by_graph.setdefault(graph_idx, set()).update(internal)

            edge_label_seq = tuple(
                _normalise_node_label(
                    (graph.get_edge_data(path[i], path[i + 1]) or {}).get(edge_attr)
                )
                for i in range(len(path) - 1)
            )
            endpoint_key = (u_c, v_c) if u_c <= v_c else (v_c, u_c)
            key = (endpoint_key, len(internal), edge_label_seq)
            path_variants.setdefault(key, []).append((graph, path))

    terminal_clusters_by_anchor = {}

    for key, recs in terminal_records.items():
        anchor, _ = key
        filtered_recs = [
            rec
            for rec in recs
            if rec["source_node"] not in path_internal_nodes_by_graph.get(rec["graph_idx"], set())
        ]
        if not filtered_recs:
            continue
        clusters = _cluster_by_similarity(filtered_recs, similarity_threshold)
        terminal_clusters_by_anchor.setdefault(anchor, [])
        for cluster in clusters:
            counts_by_graph = {}
            for rec in cluster["records"]:
                gid = rec["graph_idx"]
                counts_by_graph[gid] = counts_by_graph.get(gid, 0) + 1
            max_count = max(counts_by_graph.values()) if counts_by_graph else 1
            rep_node_data = _pick_representative_node_data(
                [rec["node_data"] for rec in cluster["records"]]
            )
            rep_edge_data = dict(cluster["records"][0]["edge_data"])
            added_nodes = []
            for _ in range(max_count):
                while template.has_node(next_node_id):
                    next_node_id += 1
                new_node = next_node_id
                next_node_id += 1
                template.add_node(new_node, **rep_node_data)
                template.add_edge(anchor, new_node, **rep_edge_data)
                added_node_owner[new_node] = anchor
                added_nodes.append(new_node)
            terminal_clusters_by_anchor[anchor].append(
                {
                    "records": cluster["records"],
                    "added_nodes": added_nodes,
                }
            )

    for anchor, clusters in terminal_clusters_by_anchor.items():
        if len(clusters) < 2:
            continue
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                ci = clusters[i]
                cj = clusters[j]
                edge_data = None
                for ri in ci["records"]:
                    for rj in cj["records"]:
                        if ri["graph_idx"] != rj["graph_idx"]:
                            continue
                        graph = source_graphs[ri["graph_idx"]]
                        if graph.has_edge(ri["source_node"], rj["source_node"]):
                            edge_data = dict(
                                graph.get_edge_data(ri["source_node"], rj["source_node"]) or {}
                            )
                            break
                    if edge_data is not None:
                        break
                if edge_data is None:
                    continue
                ni = ci["added_nodes"][0]
                nj = cj["added_nodes"][0]
                if not template.has_edge(ni, nj):
                    template.add_edge(ni, nj, **edge_data)

    for key, variants in path_variants.items():
        endpoints, internal_len, _ = key
        u_c, v_c = endpoints
        if internal_len <= 0:
            continue
        prev = u_c
        graph0, path0 = variants[0]
        for pos in range(internal_len):
            candidates = [dict(g.nodes[p[1 + pos]]) for g, p in variants]
            rep_node_data = _pick_representative_node_data(candidates)
            while template.has_node(next_node_id):
                next_node_id += 1
            new_node = next_node_id
            next_node_id += 1
            template.add_node(new_node, **rep_node_data)
            edge_data = dict(graph0.get_edge_data(path0[pos], path0[pos + 1]) or {})
            template.add_edge(prev, new_node, **edge_data)
            prev = new_node
        last_edge = dict(graph0.get_edge_data(path0[-2], path0[-1]) or {})
        template.add_edge(prev, v_c, **dict(last_edge))

    consensus_node_set = set(consensus.nodes())
    for graph_idx, (graph, c2s) in enumerate(zip(source_graphs, consensus_to_source_maps)):
        if graph is None:
            continue
        mapped_src_nodes = set(c2s.values())
        for anchor_c, anchor_s in c2s.items():
            if anchor_s not in graph:
                continue
            attached_template_nodes = [
                t for t in template.neighbors(anchor_c)
                if t not in consensus_node_set and added_node_owner.get(t) == anchor_c
            ]
            for x in graph.neighbors(anchor_s):
                if x in mapped_src_nodes:
                    continue
                x_data = dict(graph.nodes[x])
                x_smi = x_data.get("SMILES", "")
                x_candidates = [
                    t for t in attached_template_nodes
                    if _map4_similarity(template.nodes[t].get("SMILES", ""), x_smi) >= similarity_threshold
                ]
                if not x_candidates:
                    continue
                x_t = x_candidates[0]
                for y in graph.neighbors(x):
                    if y == anchor_s or y in mapped_src_nodes:
                        continue
                    y_data = dict(graph.nodes[y])
                    y_smi = y_data.get("SMILES", "")
                    # Reuse existing neighbors of x_t first (prevents duplicates across graphs).
                    y_candidates = [
                        t for t in template.neighbors(x_t)
                        if t not in consensus_node_set
                        and added_node_owner.get(t) == anchor_c
                        and _map4_similarity(template.nodes[t].get("SMILES", ""), y_smi) >= similarity_threshold
                    ]
                    if not y_candidates:
                        y_candidates = [
                            t for t in attached_template_nodes
                            if t not in consensus_node_set
                            and t != x_t
                            and added_node_owner.get(t) == anchor_c
                            and _map4_similarity(template.nodes[t].get("SMILES", ""), y_smi) >= similarity_threshold
                        ]
                    if not y_candidates:
                        y_candidates = [
                            t for t in template.nodes()
                            if added_node_owner.get(t) == anchor_c
                            if _map4_similarity(template.nodes[t].get("SMILES", ""), y_smi) >= similarity_threshold
                        ]
                    if not y_candidates:
                        while template.has_node(next_node_id):
                            next_node_id += 1
                        y_t = next_node_id
                        next_node_id += 1
                        template.add_node(y_t, **_pick_representative_node_data([y_data]))
                        added_node_owner[y_t] = anchor_c
                        edge_anchor_y = dict(graph.get_edge_data(anchor_s, y) or {})
                        if edge_anchor_y:
                            template.add_edge(anchor_c, y_t, **edge_anchor_y)
                        y_candidates = [y_t]
                    edge_xy = dict(graph.get_edge_data(x, y) or {})
                    if not edge_xy:
                        continue
                    y_t = y_candidates[0]
                    if not template.has_edge(x_t, y_t):
                        template.add_edge(x_t, y_t, **edge_xy)

    # Remove redundant shortcut edges from consensus anchors to added nodes when
    # an alternate path already exists (e.g., anchor-X-Y instead of anchor-Y direct).
    consensus_nodes_now = set(consensus.nodes())
    for u, v in list(template.edges()):
        u_cons = u in consensus_nodes_now
        v_cons = v in consensus_nodes_now
        if u_cons == v_cons:
            continue
        anchor = u if u_cons else v
        leaf = v if u_cons else u
        if not template.has_edge(anchor, leaf):
            continue
        temp = template.copy()
        temp.remove_edge(anchor, leaf)
        if not nx.has_path(temp, anchor, leaf):
            continue
        try:
            path = nx.shortest_path(temp, source=anchor, target=leaf)
        except nx.NetworkXNoPath:
            continue
        if len(path) < 3:
            continue
        internal = path[1:-1]
        if any(node in consensus_nodes_now for node in internal):
            continue
        if nx.is_connected(temp):
            template = temp

    consensus_nodes = set(consensus.nodes())
    if template.number_of_nodes() > 0 and not nx.is_connected(template):
        next_node_id = _next_available_node_id(template)
        while not nx.is_connected(template):
            components = list(nx.connected_components(template))
            comp_index = {}
            for idx, comp_nodes in enumerate(components):
                for node in comp_nodes:
                    comp_index[node] = idx

            best = None
            for graph, c2s in zip(source_graphs, consensus_to_source_maps):
                c_nodes = [n for n in consensus_nodes if n in c2s and n in comp_index]
                for i, u_c in enumerate(c_nodes):
                    for v_c in c_nodes[i + 1:]:
                        if comp_index[u_c] == comp_index[v_c]:
                            continue
                        u_s = c2s[u_c]
                        v_s = c2s[v_c]
                        if u_s not in graph or v_s not in graph:
                            continue
                        try:
                            path = nx.shortest_path(graph, source=u_s, target=v_s)
                        except nx.NetworkXNoPath:
                            continue
                        score = len(path)
                        if best is None or score < best[0]:
                            best = (score, graph, path, u_c, v_c)

            if best is None:
                break

            _, graph, path, u_c, v_c = best
            prev = u_c
            for idx, src_node in enumerate(path[1:-1], start=1):
                while template.has_node(next_node_id):
                    next_node_id += 1
                new_node = next_node_id
                next_node_id += 1
                template.add_node(new_node, **dict(graph.nodes[src_node]))
                edge_data = dict(graph.get_edge_data(path[idx - 1], path[idx]) or {})
                template.add_edge(prev, new_node, **dict(edge_data))
                prev = new_node
            last_edge = dict(graph.get_edge_data(path[-2], path[-1]) or {})
            template.add_edge(prev, v_c, **dict(last_edge))

    return template


def cluster_consensus_mcs(
    cluster_graphs,
    node_attr="MUID",
    edge_attr="reaction",
    max_candidates=100,
    sort_graphs=True,
    node_similarity_path=None,
    sim_threshold=SIM_THRESHOLD,
    allow_similar_nodes=True,
):
    cluster_graphs = [g for g in cluster_graphs if g is not None]

    if not cluster_graphs:
        return []

    if len(cluster_graphs) == 1:
        g = cluster_graphs[0].copy()
        return [
            {
                "graph": g,
                "template_graph": g.copy(),
                "sim_threshold": sim_threshold,
                "lineage": [],
                "n_nodes": g.number_of_nodes(),
                "n_edges": g.number_of_edges()
            }
        ]

    if sort_graphs:
        cluster_graphs = sorted(
            cluster_graphs,
            key=lambda g: (g.number_of_nodes(), g.number_of_edges())
        )

    seed_graph = cluster_graphs[0].copy()
    candidates = [
        {
            "graph": seed_graph,
            "template_graph": seed_graph.copy(),
            "seed_graph": seed_graph.copy(),
            "source_graphs": [seed_graph.copy()],
            "sim_threshold": sim_threshold,
            "lineage": [],
            "n_nodes": seed_graph.number_of_nodes(),
            "n_edges": seed_graph.number_of_edges()
        }
    ]

    for step_idx, next_graph in enumerate(cluster_graphs[1:], start=1):
        new_candidates = []

        for candidate in candidates:
            pairwise_results = maximum_common_subgraphs(
                candidate["graph"],
                next_graph,
                node_attr=node_attr,
                edge_attr=edge_attr,
                node_similarity_path=node_similarity_path,
                sim_threshold=sim_threshold,
                allow_similar_nodes=allow_similar_nodes,
            )

            for pair_res in pairwise_results:
                new_graph = pair_res["subgraph_G"].copy()

                new_candidates.append(
                    {
                        "graph": new_graph,
                        "template_graph": candidate["template_graph"],
                        "seed_graph": candidate["seed_graph"],
                        "source_graphs": candidate["source_graphs"] + [next_graph],
                        "sim_threshold": candidate.get(
                            "sim_threshold",
                            sim_threshold
                        ),
                        "lineage": candidate["lineage"]
                        + [
                            {
                                "step": step_idx,
                                "other_graph": next_graph,
                                "mapping_G_to_H": pair_res["mapping_G_to_H"],
                                "mapping_H_to_G": pair_res["mapping_H_to_G"],
                                "subgraph_other": pair_res["subgraph_H"]
                            }
                        ],
                        "n_nodes": new_graph.number_of_nodes(),
                        "n_edges": new_graph.number_of_edges()
                    }
                )

        if not new_candidates:
            return []

        new_candidates = _deduplicate_candidates(
            new_candidates,
            node_attr=node_attr,
            edge_attr=edge_attr
        )

        new_candidates = sorted(
            new_candidates,
            key=lambda c: (c["n_nodes"], c["n_edges"]),
            reverse=True
        )

        if max_candidates is not None:
            new_candidates = new_candidates[:max_candidates]

        candidates = new_candidates

    return candidates


def get_best_cluster_consensus(
    cluster_graphs,
    node_attr="MUID",
    edge_attr="reaction",
    max_candidates=100,
    sort_graphs=True,
    node_similarity_path=None,
    sim_threshold=SIM_THRESHOLD,
    allow_similar_nodes=True,
):
    candidates = cluster_consensus_mcs(
        cluster_graphs=cluster_graphs,
        node_attr=node_attr,
        edge_attr=edge_attr,
        max_candidates=max_candidates,
        sort_graphs=sort_graphs,
        node_similarity_path=node_similarity_path,
        sim_threshold=sim_threshold,
        allow_similar_nodes=allow_similar_nodes,
    )

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda c: (c["n_nodes"], c["n_edges"])
    )


def extract_consensus_graph(result):
    if result is None:
        return None
    consensus = result["graph"]
    template = result.get("template_graph")
    source_graphs = result.get("source_graphs")
    lineage = result.get("lineage", [])
    seed_graph = result.get("seed_graph")
    sim_threshold = result['sim_threshold']

    if source_graphs and seed_graph is not None:
        consensus_nodes = set(consensus.nodes())
        maps = [{n: n for n in consensus_nodes if n in seed_graph}]
        for step in lineage:
            mapping = step.get("mapping_G_to_H", {})
            maps.append({n: mapping[n] for n in consensus_nodes if n in mapping})
        if len(maps) == len(source_graphs):
            template = _build_augmented_template(
                consensus=consensus,
                source_graphs=source_graphs,
                consensus_to_source_maps=maps,
                similarity_threshold=sim_threshold
            )

    if template is None:
        return consensus

    reconstructed = template.copy()
    consensus_nodes = set(consensus.nodes())

    empty_node = {
        "SMILES": "",
        "MUID": "",
        "Similarity": 0.0,
        "Annotation": "",
        "isNP": "No",
        "img": "",
    }

    for n in reconstructed.nodes():
        if n in consensus_nodes:
            reconstructed.nodes[n].clear()
            reconstructed.nodes[n].update(dict(consensus.nodes[n]))
        else:
            node_data = dict(reconstructed.nodes[n])
            iso_smiles = node_data.get("ISOSMILES", node_data.get("SMILES", ""))
            reconstructed.nodes[n].clear()
            reconstructed.nodes[n].update(empty_node)
            reconstructed.nodes[n]["SMILES"] = iso_smiles
            reconstructed.nodes[n]["ISOSMILES"] = iso_smiles

    return reconstructed


def get_all_pairwise_mcs(
    G,
    H,
    node_attr="MUID",
    edge_attr="reaction",
    node_similarity_path=None,
    sim_threshold=SIM_THRESHOLD,
    allow_similar_nodes=True,
):
    return maximum_common_subgraphs(
        G,
        H,
        node_attr=node_attr,
        edge_attr=edge_attr,
        node_similarity_path=node_similarity_path,
        sim_threshold=sim_threshold,
        allow_similar_nodes=allow_similar_nodes,
    )
