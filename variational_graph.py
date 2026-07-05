import networkx as nx
from rdkit import Chem

from new_geass import mol2svg


def _node_key(data):
    key = data.get("MUID")
    return str(key) if key else None


def _update_representative_muid(V, v, graphs):
    muid_counts = {}
    for mgid, mnode in V.nodes[v]["members"]:
        muid = graphs[mgid].nodes[mnode].get("MUID", "")
        if muid:
            muid_counts[muid] = muid_counts.get(muid, 0) + 1

    if muid_counts:
        V.nodes[v]["MUID"] = max(muid_counts.items(), key=lambda kv: kv[1])[0]


def build_variational_graph(graphs):
    V = nx.Graph()
    original_to_v = {}
    key_to_vids = {}
    next_vid = 0

    for gid, G in enumerate(graphs):
        for n, data in G.nodes(data=True):
            key = _node_key(data)
            if key is None:
                continue

            best_v = None
            for vid in key_to_vids.get(key, []):
                member_gids = {member_gid for member_gid, _ in V.nodes[vid]["members"]}
                if gid not in member_gids:
                    best_v = vid
                    break

            if best_v is None:
                best_v = next_vid
                next_vid += 1
                key_to_vids.setdefault(key, []).append(best_v)
                V.add_node(
                    best_v,
                    members=[],
                    is_support="No",
                    SMILES=data.get("SMILES", ""),
                    MUID=str(data.get("MUID", "")),
                    img="",
                    key=key,
                )

            original_to_v[(gid, n)] = best_v
            V.nodes[best_v]["members"].append((gid, n))
            member_gids = {x[0] for x in V.nodes[best_v]["members"]}
            V.nodes[best_v]["is_support"] = "Yes" if len(member_gids) > 1 else "No"
            _update_representative_muid(V, best_v, graphs)

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


def show_variational_graph(V, highlight_gid=None):
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
