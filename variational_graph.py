import networkx as nx
import numpy as np
from map4 import MAP4Calculator
from rdkit import Chem

from new_geass import mol2svg


_MAP4_CALCULATOR = MAP4Calculator(dimensions=1024)
_FP_CACHE = {}


def _get_map4_fp(node_data):
    fp = node_data.get("MAP4")
    if fp is not None:
        arr = np.asarray(fp)
        if arr.size > 0:
            return arr

    smiles = node_data.get("SMILES") or ""
    if not smiles:
        return None

    cached = _FP_CACHE.get(smiles)
    if cached is not None:
        return cached

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    smiles0 = Chem.MolToSmiles(mol, isomericSmiles=False)
    mol0 = Chem.MolFromSmiles(smiles0)
    if mol0 is None:
        return None

    fp = np.asarray(_MAP4_CALCULATOR.calculate(mol0), dtype=np.uint32)
    _FP_CACHE[smiles] = fp
    return fp


def _map4_sim(fp_a, fp_b):
    if fp_a is None or fp_b is None:
        return 0.0

    a = np.asarray(fp_a)
    b = np.asarray(fp_b)
    if a.shape != b.shape or a.size == 0:
        return 0.0

    return float((a == b).mean())


def _node_smiles(data):
    return data.get("SMILES") or ""


def _update_representative_muid(V, v, graphs):
    muid_counts = {}
    for mgid, mnode in V.nodes[v]["members"]:
        muid = graphs[mgid].nodes[mnode].get("MUID", "")
        if muid:
            muid_counts[muid] = muid_counts.get(muid, 0) + 1

    if muid_counts:
        V.nodes[v]["MUID"] = max(muid_counts.items(), key=lambda kv: kv[1])[0]


def _update_weighted_medoid(V, v, graphs):
    members = V.nodes[v]["members"]
    reps = []
    for mgid, mnode in members:
        data = graphs[mgid].nodes[mnode]
        smiles = _node_smiles(data)
        fp = _get_map4_fp(data)
        if fp is None or not smiles:
            continue
        reps.append((smiles, fp, data.get("MUID", ""), mgid))

    if not reps:
        return

    graph_weights = {}
    for _, _, _, gid in reps:
        graph_weights[gid] = graph_weights.get(gid, 0) + 1

    best_idx = 0
    best_score = -1.0
    for i, (_, fp_i, _, _) in enumerate(reps):
        score = 0.0
        for _, fp_j, _, gid_j in reps:
            score += float(graph_weights.get(gid_j, 1)) * _map4_sim(fp_i, fp_j)
        if score > best_score:
            best_score = score
            best_idx = i

    best_smiles, _, best_muid, _ = reps[best_idx]
    V.nodes[v]["SMILES"] = best_smiles
    if best_muid:
        V.nodes[v]["MUID"] = best_muid


def build_variational_graph(graphs, threshold=0.20):
    V = nx.Graph()
    original_to_v = {}
    next_vid = 0

    for gid, G in enumerate(graphs):
        for n, data in G.nodes(data=True):
            fp = _get_map4_fp(data)
            if fp is None:
                continue

            best_v = None
            best_score = -1.0

            for v, vdata in V.nodes(data=True):
                member_gids = {member[0] for member in vdata.get("members", [])}
                if gid in member_gids:
                    continue

                scores = []
                for member_gid, member_node in vdata.get("members", []):
                    member_data = graphs[member_gid].nodes[member_node]
                    scores.append(_map4_sim(fp, _get_map4_fp(member_data)))
                score = float(np.mean(scores)) if scores else 0.0

                if score > best_score:
                    best_score = score
                    best_v = v

            if best_v is not None and best_score > threshold:
                original_to_v[(gid, n)] = best_v
                V.nodes[best_v]["members"].append((gid, n))
                member_gids = {x[0] for x in V.nodes[best_v]["members"]}
                V.nodes[best_v]["is_support"] = "Yes" if len(member_gids) > 1 else "No"
                _update_representative_muid(V, best_v, graphs)
                _update_weighted_medoid(V, best_v, graphs)
            else:
                vid = next_vid
                next_vid += 1
                smiles = _node_smiles(data)
                V.add_node(
                    vid,
                    members=[(gid, n)],
                    is_support="No",
                    SMILES=smiles,
                    MUID=data.get("MUID", ""),
                    img="",
                )
                original_to_v[(gid, n)] = vid

    for gid, G in enumerate(graphs):
        for u, v in G.edges():
            if (gid, u) not in original_to_v or (gid, v) not in original_to_v:
                continue

            vu = original_to_v[(gid, u)]
            vv = original_to_v[(gid, v)]
            if vu == vv:
                continue

            if not V.has_edge(vu, vv):
                V.add_edge(vu, vv, _graphs=set())
            V[vu][vv]["_graphs"].add(gid)

    for _, _, data in V.edges(data=True):
        graphs_set = data.get("_graphs", set())
        data["graphs"] = sorted(graphs_set)
        data["is_support"] = "Yes" if len(graphs_set) > 1 else "No"
        data.pop("_graphs", None)

    for _, data in V.nodes(data=True):
        member_gids = {m[0] for m in data.get("members", []) if isinstance(m, (list, tuple)) and len(m) >= 1}
        data["is_support"] = "Yes" if len(member_gids) > 1 else "No"

    for _, data in V.nodes(data=True):
        smiles = data.get("SMILES", "")
        if smiles and not data.get("img"):
            mol = Chem.MolFromSmiles(smiles)
            data["img"] = mol2svg(mol) if mol is not None else ""

    return V


def show_variation_graph(V, highlight_gid=None):
        from ipycytoscape import CytoscapeWidget
        layout = 'cose'

        def _node_has_gid(node_data, gid):
            members = node_data.get("members", [])
            for m in members:
                if isinstance(m, (list, tuple)) and len(m) >= 1:
                    try:
                        if int(m[0]) == int(gid):
                            return True
                    except Exception:
                        if m[0] == gid:
                            return True
            return False

        def _edge_has_gid(edge_data, gid):
            graphs = edge_data.get("graphs", [])
            if isinstance(graphs, str):
                try:
                    graphs = [int(x.strip()) for x in graphs.split(",") if x.strip() != ""]
                except Exception:
                    graphs = [x.strip() for x in graphs.split(",") if x.strip() != ""]
            if isinstance(graphs, (set, list, tuple)):
                for g in graphs:
                    try:
                        if int(g) == int(gid):
                            return True
                    except Exception:
                        if g == gid:
                            return True
            return False

        V_plot = V.copy()
        if highlight_gid is not None:
            for _, data in V_plot.nodes(data=True):
                in_path = _node_has_gid(data, highlight_gid)
                data["path_node"] = "Yes" if in_path else "No"
                data["path_focus"] = "Yes"

            for u, v, data in V_plot.edges(data=True):
                in_path = _edge_has_gid(data, highlight_gid)
                data["path_edge"] = "Yes" if in_path else "No"
                data["path_focus"] = "Yes"

        cyto = CytoscapeWidget()
        cyto.layout.height = '600px'
        cyto.layout.width = '1400px'
        cyto.set_layout(name=layout,
            nodeSpacing=20, avoidOverlap=True,
            padding=50, maxSimulationTime=6000)
        cyto.graph.add_graph_from_networkx(V_plot)

        for node in cyto.graph.nodes:
            node.data['label'] = node.data.get('MUID', '')

        cyto.set_style([
            {
                'selector': 'node',
                'css': {
                    'content': 'data(MUID)',
                    'width': 50,
                    'height': 50,
                    'border-width': 1.5,
                    'border-color': "#989898",
                    'background-color': '#FFFFFF',
                    'background-width': '80%',
                    'background-height': '80%',
                    'background-image': 'data(img)',
                    'font-family': 'helvetica',
                    'font-size': '10px',
                    'text-margin-y': '8px',
                    'text-halign': 'center',
                    'text-valign': 'bottom',
                    'text-background-color': '#FFFFFF',
                    'text-background-shape': 'roundrectangle',
                    'text-background-opacity': 1,
                    'text-background-padding': 2
                }
            },
            {
                'selector': "edge",
                'css': {
                    'curve-style': 'bezier',
                    'line-color': '#989898',
                    'width': 1.5,
                    'font-family': 'helvetica',
                    'font-size': '10px',
                    'text-valign': 'center',
                    'text-background-color': '#ffffff',
                    'text-background-shape': 'roundrectangle',
                    'text-background-opacity': 1,
                    'text-background-padding': 2
                }
            },
            {
                'selector': "node[is_support = 'Yes'], edge[is_support = 'Yes']",
                'css': {'border-color': '#0a59c1', 'line-color': '#0a59c1'}
            },
            {
                'selector': "node[path_node = 'Yes'], edge[path_edge = 'Yes']",
                'css': {'border-color': '#0057b8', 'line-color': '#0057b8'}
            },
            {
                'selector': "node[path_node = 'Yes']",
                'css': {'border-width': 4}
            },
            {
                'selector': "edge[path_edge = 'Yes']",
                'css': {'width': 3.5}
            },
            {
                'selector': "node[path_focus = 'Yes'][path_node = 'No']",
                'css': {'opacity': 0.25}
            },
            {
                'selector': "edge[path_focus = 'Yes'][path_edge = 'No']",
                'css': {'opacity': 0.15}
            }
        ])

        for _, data in V_plot.nodes(data=True):
            for k, v in list(data.items()):
                if isinstance(v, (list, tuple, set, dict)):
                    data[k] = str(v)  # GraphML needs scalar attrs
        for _, _, data in V_plot.edges(data=True):
            for k, v in list(data.items()):
                if isinstance(v, (list, tuple, set, dict)):
                    data[k] = str(v)
        nx.write_graphml(V_plot, "V.graphml")

        return cyto
