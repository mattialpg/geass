from rdkit import Chem
from rdkit.Chem import AllChem

from urllib import parse
import numpy as np
import networkx as nx

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
    except:
        pass

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
    drawer = rdMolDraw2D.MolDraw2DSVG(600, 600)
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    svg_string = drawer.GetDrawingText()

    return 'data:image/svg+xml;charset=utf-8,' + parse.quote(svg_string, safe="")


def create_connection_graph(self):

    for i, frag_mol in enumerate(self.frag_mols):
        self.G.add_nodes_from([(i, {'SMILES': Chem.MolToSmiles(frag_mol),
                                    'isNP': frag_mol.GetProp('isNP'),
                                    'BU_ID': frag_mol.GetProp('BU_ID'),
                                    'img': mol2svg(frag_mol)
                                    })])

    # Create edge/attributes dictionary
    edge_dict = {}
    for k, v in self.bonds_to_break.items():
        if k[0] in self.atom_to_frag and k[1] in self.atom_to_frag:
            f1 = self.atom_to_frag[k[0]]
            f2 = self.atom_to_frag[k[1]]
            edge_dict[(f1, f2)] = v
            #<-- This last condition arises because, for symmetrical envs,
            #<-- only one of the two possible breakage contained in bonds_to_break
            #<-- is actually carried out.
    #TODO Fantastic-biofrag should overcome this fundamental limitation by creating a branching point
    #TODO the fragmentation tree when symmetrical patterns produce asymmetrical blocks.

    for edge, attribute in edge_dict.items():
        self.G.add_edge(*edge, reaction=attribute)
    
    return self.G


def show_connection_graph(self, method='cytoscape'):

    self.G = self.create_connection_graph()

    if method == 'yEd':
        import random
        from yfiles_jupyter_graphs import GraphWidget

        # --- Preprocess graph: random jitter to break symmetry ---
        for node in self.G.nodes():
            self.G.nodes[node]["pos"] = (random.random() * 0.1, random.random() * 0.1)

        # Optional: reorder nodes by degree to space hubs
        self.G = nx.Graph(sorted(self.G.degree, key=lambda x: -x[1]))
        
        def custom_images(id, node):
            label = list(self.G.nodes())[id]
            img = self.G.nodes[label]['img']
            return {'img': img}

        w = GraphWidget(graph=self.G)
        w.node_label_mapping = ''  # Hide labels
        w.set_node_styles_mapping(custom_images)
        w.set_overview(enabled=False)
        w.set_sidebar(enabled=False)

        # --- Enhanced organic layout configuration ---
        w.organic_layout(
            preferred_edge_length=250,
            node_overlaps_allowed=False,
            node_size_consideration=True,
            edge_crossing_avoidance=True,
            deterministic=True,
            consider_node_labels=True,
            act_on_selection=False,
            minimum_node_distance=40
        )

        # --- Optional: rerun layout twice to further relax edges ---
        w.organic_layout(preferred_edge_length=150)

        w.show()

    #! ------------- Plot with Plotly ------------!#
    if method == 'plotly':
        import plotly.graph_objects as go
        # Create positions for nodes using spring layout
        pos = nx.spring_layout(self.G, k=1)

        # Extract the positions into separate x and y lists for Plotly
        x_vals = [pos[n][0] for n in self.G.nodes()]
        y_vals = [pos[n][1] for n in self.G.nodes()]

        # Create node trace for Plotly
        node_trace = go.Scatter(
            x=x_vals, y=y_vals,
            mode='markers',
            marker=dict(size=110, color='white',
            line=dict(width=1.7, color=['green' if self.G.nodes[n].get('isNP') == 'Yes'
                                            else 'red' for n in self.G.nodes()])),
            text=[f'Node {n}' for n in self.G.nodes()],
            hoverinfo='text')

        # Create edge trace
        edge_trace = go.Scatter(
            x=sum([[pos[e[0]][0], pos[e[1]][0], None] for e in self.G.edges()], []),
            y=sum([[pos[e[0]][1], pos[e[1]][1], None] for e in self.G.edges()], []),
            line=dict(width=1.7, color='black'),
            hoverinfo='none',
            mode='lines')

        # Create the figure with the node and edge traces
        fig = go.Figure(data=[edge_trace, node_trace])

        # Create list of images (SVG or PNG) to overlay on the graph
        image_annotations = []
        for node in self.G.nodes(data=True):
            x, y = pos[node[0]]
            image_url = node[1]['img']
            image_annotations.append(
                dict(
                    source=image_url,
                    xref="x", yref="y",
                    x=x, y=y,
                    sizex=.3, sizey=.3,
                    xanchor="center", yanchor="middle",
                    layer="above"))

        # Add the images (SVGs or PNGs) to the layout
        fig.update_layout(images=image_annotations)

        # Configure layout
        fig.update_layout(
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, visible=False),  # Remove x-axis
            yaxis=dict(showgrid=False, zeroline=False, visible=False),  # Remove y-axis
            paper_bgcolor="white",     # Set background color to white
            plot_bgcolor="white",      # Set plot background to white
            margin=dict(t=0, b=0, l=0, r=0),  # Remove padding around plot
            width=600,                 # Set width
            height=600)                # Set height to be equal to width
        
        fig.show()

    #! ----------- Plot with Cytoscape -----------!#
    elif method  == 'cytoscape':
        #* get ideas from https://giter.vip/cytoscape/ipycytoscape
        #* and https://dash.plotly.com/cytoscape/layout

        cyto = CytoscapeWidget()
        cyto.layout.height = '500px'
        cyto.layout.width = '1400px'
        cyto.graph.add_graph_from_networkx(self.G)
        cyto.set_layout(name='cose')

        # Update node labels to use the 'BU_ID' attribute
        for node in cyto.graph.nodes:
            node.data['label'] = node.data.get('BU_ID', '')

        cyto.set_style([{'selector': 'node',
                        'css': {
                            # 'content': 'data(count)',
                            'content': 'data(BU_ID)',  # Use BU_ID for labels
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
                        'css': {'line-color': '#d10808'}}
                        ])

        # display(cyto)
        return cyto
        # return self.G