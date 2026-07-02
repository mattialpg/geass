from rdkit import Chem
from rdkit.Chem import AllChem

from urllib import parse
import numpy as np
import networkx as nx
import plotly.graph_objects as go

from utils import *


def reset_view(mol):
    """ Reset coordinates for display """
    Chem.rdDepictor.SetPreferCoordGen(True)
    Chem.rdDepictor.Compute2DCoords(mol)
    Chem.rdDepictor.StraightenDepiction(mol)
    return mol


def mol2svg(mol):
    try:
        Chem.Kekulize(mol)

        atoms = list(mol.GetAtoms())
        heavy_atoms = [a.GetAtomicNum() for a in atoms]

        # Handle CH3–X case
        if len(atoms) == 2 and 6 in heavy_atoms:
            atom_C = next(a for a in atoms if a.GetAtomicNum() == 6)
            atom_X = next(a for a in atoms if a.GetAtomicNum() != 6)

            atom_C.SetProp("atomLabel", "CH<sub>3</sub>")
            if atom_X.GetAtomicNum() == 0:
                atom_X.SetProp("atomLabel", "R")

            AllChem.Compute2DCoords(mol)
            conf = mol.GetConformer()

            pos_C = np.array(conf.GetAtomPosition(atom_C.GetIdx()))
            pos_X = np.array(conf.GetAtomPosition(atom_X.GetIdx()))
            conf.SetAtomPosition(atom_C.GetIdx(), pos_C + 0.2 * (pos_X - pos_C))

        else:
            AllChem.Compute2DCoords(mol)

        # Draw SVG
        mol = reset_view(mol)
        drawer = rdMolDraw2D.MolDraw2DSVG(70, 70)
        opts = drawer.drawOptions()
        opts.bondLineWidth = 1.0
        opts.useFractionalBondWidths = True
        opts.clearBackground = False
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
        drawer.FinishDrawing()
        svg_string = drawer.GetDrawingText()


        return 'data:image/svg+xml;charset=utf-8,' + parse.quote(svg_string, safe="")

    except:
        return ''


def create_connection_graph(self):
    G = nx.Graph()

    for i, frag_mol in enumerate(self.frag_mols):
        G.add_nodes_from([(i, {'SMILES': Chem.MolToSmiles(frag_mol),
                                    'isNP': frag_mol.GetProp('isNP'),
                                    'MU_ID': frag_mol.GetProp('MU_ID'),
                                    # 'atoms': [k for k, v in self.atom_frag.items() if v == i],
                                    'img': mol2svg(frag_mol)
                                    })])
    
    # Create edge/attributes dictionary 
    edge_dict = {}
    for k, v in self.bonds_to_break.items():
        if k[0] in self.atom_frag and k[1] in self.atom_frag:
            f1 = self.atom_frag[k[0]]
            f2 = self.atom_frag[k[1]]
            edge_dict[(f1, f2)] = v
            #<-- This last condition arises because, for symmetrical envs,
            #<-- only one of the two possible breakage contained in bonds_to_break
            #<-- is actually carried out.
    #TODO Fantastic-biofrag should overcome this fundamental limitation by creating a branching point
    #TODO the fragmentation tree when symmetrical patterns produce asymmetrical blocks.

    for edge, attribute in edge_dict.items():
        G.add_edge(*edge, reaction=attribute)
    
    return G


def show_connection_graph(G, method='cytoscape'):
    #! ----------- Plot with Cytoscape -----------!#
    if method  == 'cytoscape':
        #* get ideas from https://giter.vip/cytoscape/ipycytoscape
        #* and https://dash.plotly.com/cytoscape/layout

        cyto = CytoscapeWidget()
        cyto.layout.height = '800px'
        cyto.layout.width = '1400px'
        cyto.set_layout(name='cose')
        cyto.graph.add_graph_from_networkx(G)

        # Update node labels to use the 'MU_ID' attribute
        for node in cyto.graph.nodes:
            node.data['label'] = node.data.get('MU_ID', '')

        cyto.set_style([{'selector': 'node',
                        'css': {
                            # 'content': 'data(count)',
                            'content': 'data(MU_ID)',  # Use MU_ID for labels
                            'width': 50,
                            'height': 50,
                            'border-width': 1.5,
                            'background-color': '#FFFFFF',
                            # 'background-fit': 'contain',
                            'background-width': '90%',
                            'background-height': '90%',
                            'background-image': 'data(img)',
                            'font-family': 'helvetica',
                            'font-size': '10px',
                            'text-halign': 'center',
                            'text-valign': 'bottom',
                            'text-margin-y': '8px'}},

                        {'selector': "node[isNP = 'Yes']",
                        'css': {'border-color': '#009b73'}},
                        
                        {'selector': "node[isNP = 'No']",
                        'css': {'border-color': '#d10808'}},

                        {'selector': "node[isNP = 'Similar']",
                        'css': {'border-color': '#ffc32d'}},
                        
                        {'selector': 'edge',
                        'css': {
                            'curve-style': 'bezier',
                            'line-color': '#009b73',
                            'width': 1.5,
                            'label': 'data(reaction)',
                            'font-family': 'helvetica',
                            'font-size': '10px',
                            'text-valign': 'center',
                            'text-background-color': '#ffffff',
                            'text-background-opacity': 1,
                            'text-background-shape': 'roundrectangle',
                            'padding': '50px'}},
                        
                        {'selector': "edge[reaction='']",
                        'css': {'line-color': '#d10808'}},

                        # {'selector': 'node:hover',
                        #  'style': {'label': 'data(atoms)',
                        #            'overlay-opacity': 0.1,
                        #            'overlay-color': 'black'}}
                        ])

        return cyto

    elif method == 'yEd':
        import random
        from yfiles_jupyter_graphs import GraphWidget
        import networkx as nx

        # --- Preprocess graph: random jitter to break symmetry ---
        for node in G.nodes():
            G.nodes[node]["pos"] = (random.random() * 0.1, random.random() * 0.1)

        # Optional: reorder nodes by degree to space hubs
        G = nx.Graph(sorted(G.degree, key=lambda x: -x[1]))
        
        def custom_images(node):
            props = node.get('properties', {})
            img = props.get('img')
            print(img)
            style = {'fill': 'transparent', 'stroke': 'transparent'}
            style['image'] = img
            return style

        w = GraphWidget(graph=G)
        w.node_label_mapping = ''  # Hide labels
        w.set_node_styles_mapping(custom_images)
        w.set_overview(enabled=False)
        w.set_sidebar(enabled=False)

        # --- Enhanced organic layout configuration ---
        w.organic_layout(
            # preferred_edge_length=250,
            # node_overlaps_allowed=False,
            # node_size_consideration=True,
            # edge_crossing_avoidance=True,
            # deterministic=True,
            # consider_node_labels=True,
            # act_on_selection=False,
            # minimum_node_distance=40
        )

        # --- Optional: rerun layout twice to further relax edges ---
        # w.organic_layout(preferred_edge_length=150)

        w.show()

    #! ------------- Plot with Plotly ------------!#
    if method == 'plotly':
        # --- Compute RDKit 2D coordinates ---
        AllChem.Compute2DCoords(mol)
        coords = mol.GetConformer().GetPositions()  # (n_atoms, 3)

        # --- Compute node positions as barycenters of their atoms ---
        raw_pos = {}; scale = 40
        for node in G.nodes():
            atoms = G.nodes[node].get('atoms', [])
            if atoms:
                atom_coords = np.array([coords[i][:2] for i in atoms])
                barycenter = atom_coords.mean(axis=0)
                raw_pos[node] = barycenter * scale
            else:
                raw_pos[node] = np.array([1, 1])  # fallback

        # --- Collect ranges for normalisation ---
        xs = np.array([raw_pos[n][0] for n in G.nodes()])
        ys = np.array([raw_pos[n][1] for n in G.nodes()])

        xmin, xmax = xs.min(), xs.max()
        ymin, ymax = ys.min(), ys.max()

        # Avoid zero-division; if molecule collapsed to a point, widen artificially
        if xmax == xmin:
            xmax = xmin + 1.0
        if ymax == ymin:
            ymax = ymin + 1.0

        # --- Normalise into [0,1] ---
        norm_pos = {}
        for n in G.nodes():
            x, y = raw_pos[n]
            nx = (x - xmin) / (xmax - xmin)
            ny = (y - ymin) / (ymax - ymin)
            norm_pos[n] = np.array([nx, ny])

        # Optional: scale to plot-friendly size
        plot_scale = 500
        pos = {n: norm_pos[n] * plot_scale for n in G.nodes()}

        # --- Extract coordinates for plotting ---
        x_vals = [pos[n][0] for n in G.nodes()]
        y_vals = [pos[n][1] for n in G.nodes()]

        # --- Edges ---
        edge_x = []
        edge_y = []
        for e in G.edges():
            x0, y0 = pos[e[0]]
            x1, y1 = pos[e[1]]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1.7, color='black'),
            mode='lines',
            hoverinfo='none'
        )

        # --- Nodes ---
        node_colors = [
            'green' if G.nodes[n].get('isNP') == 'Yes' else 'red'
            for n in G.nodes()
        ]

        node_trace = go.Scatter(
            x=x_vals, y=y_vals,
            mode='markers',
            marker=dict(
                size=100,
                color='white',
                line=dict(width=1.7, color=node_colors)
            ),
            text=[
                f"Node {n}<br>isNP: {G.nodes[n].get('isNP')}<br>"
                f"Atoms: {G.nodes[n].get('atoms')}"
                for n in G.nodes()
            ],
            hoverinfo='text'
        )

        # --- Overlay images ---
        image_annotations = []
        for node, data in G.nodes(data=True):
            x, y = pos[node]
            if 'img' in data and data['img']:
                image_annotations.append(
                    dict(
                        source=data['img'],
                        xref="x", yref="y",
                        x=x, y=y,
                        sizex=200, sizey=200,
                        xanchor="center", yanchor="middle",
                        layer="above"
                    )
                )

        # --- Build figure ---
        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(
            images=image_annotations,
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
            paper_bgcolor="white",
            plot_bgcolor="white",
            # margin=dict(t=0, b=0, l=0, r=0),
            width=700, height=500)

        fig.update_xaxes(constrain='domain', fixedrange=True)
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        fig.update_yaxes(fixedrange=True)
        fig.update_layout(autosize=False)


        fig.show()



