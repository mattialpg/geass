import pandas as pd
from urllib import parse
from itertools import combinations
from copy import deepcopy
import threading

import networkx as nx
from ipycytoscape import CytoscapeWidget

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

#* Nomenclature
#* fragment        >  mol with dummy atom(s)
#* building unit   >  fragment with [*] replaced with real atoms 
#* building block  >  mol constituted by two or more units

#* For consistency, reaction products must be written in parentheses and separated by a dot.
#* Molecules in the resulting Mol object are then separated by Chem.GetMolFrags.
#* In the reaction smarts, the breaking bond must always be between atom 0 and 1.

reactions = {
    #!UPDATED
    'amide': "[NX3;$(N([#6])[#6]):0]-[C:1]=[O:2]>>([#7:0].[C:1](=[O:2])-O)",
    'ester': "[O;$(O([#6])([#6])):0]-[C;!$([C;r5,r6](-[O;r5,r6])):1]=[O:2]>>([O:0].[C:1](=[O:2])-O)",
    'thioester': "[SX2:0]-[C:1]=[O:2]>>([SX2:0].[C:1](=[O:2])-O)",
    'ether': "[#6;!$(C=O);!$([CH3]);!$([C;r5,r6](-[O;r5,r6])-O):0]-[OX2H0;!r;!$(O-[C;$([C;r5,r6](-[O;r5,r6])-O)]);!$(O-[CH3]):1]>>([#6:0]-O.[O:1])",
    'glycosidic': "[C;$([C;r5,r6](-[O;r5,r6])-O):0]-[OX2H0;!r;!$(O-[CH3]):1]>>([C:0]-O.[O:1])",
    'N-alk': "[#7:0][CH3:1]>>([#7H1:0].[C:1])",
    'O-alk': "[#8:0][CH3:1]>>([#8H1:0].[C:1])",
    'iminol': "[NX2H0;!R:0]=[C:1]-[O:2]>>([N:0].O[C:1]=[OX2H0:2])",
    }
reactions = {k: AllChem.ReactionFromSmarts(v) for k, v in reactions.items()}


def flatten(list):
    flat_list = [item for sublist in list for item in sublist]
    return(flat_list)

def remove_isotopes(mol):
    for atom in mol.GetAtoms():
        atom.SetIsotope(0)

def neutralize_atoms(mol):
    """
    https://baoilleach.blogspot.com/2019/12/no-charge-simple-approach-to.html
    # +1 charged atoms must have a hydrogen, or we can't remove H+
    # we must check neighbor atom charges to avoid altering groups
    # with charge-separated representations (e.g. nitro).
    """
    pattern = Chem.MolFromSmarts("[+1!H0!$([-1,-2,-3,-4]),-1!$([+1,+2,+3,+4])]")
    matches = [x[0] for x in mol.GetSubstructMatches(pattern)]
    for match in matches:
        atom = mol.GetAtomWithIdx(match)
        atom.SetNumExplicitHs(atom.GetTotalNumHs() - atom.GetFormalCharge())
        atom.SetFormalCharge(0)
        atom.UpdatePropertyCache()
    del mol.__sssAtoms
    return mol

def mol2svg(mol):
    try:
        Chem.rdmolops.Kekulize(mol)
    except: pass
    drawer = rdMolDraw2D.MolDraw2DSVG(600, 600)
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    svg_string = drawer.GetDrawingText()

    # In some cases the smiles must be modified when dealing with rings
    # smi = smi[0].upper() + smi[1:]
    # svg_string = smi2svg(smi)
    
    impath = 'data:image/svg+xml;charset=utf-8,' + parse.quote(svg_string, safe="")

    return impath


class Geass:
    _df_BUs = pd.DataFrame()

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.bonds_to_break = {}
        self.G = nx.Graph()
        self.UCL = None

        # Initialize BUL_dict and df_BUs once at the class level
        if Geass._df_BUs.empty:
            self._initialize_BUL()

    @property
    def df_BUs(self):
        """Access df_BUs through a property."""
        return Geass._df_BUs

    def _initialize_BUL(self):
        try:
            Geass._df_BUs = pd.read_pickle('data/Database_NEWBUs.pkl')
        except Exception as e:
            print(f"Error loading BU Library: {e}")


    def check_bul(self, mol, check_unknowns=False):
        # First use smiles to to search BUs (faster)
        bu_search = self.df_BUs[self.df_BUs['ISOSMILES'] == Chem.MolToSmiles(mol)]['ID']

        isNP = 'No'; bu_id = 'NA'
        if len(bu_search) > 0:
            isNP = 'Yes'; bu_id = bu_search.iloc[0]
        else:
            # Use SubstructMatch to search BUs (slower)
            bu_formula = self.df_BUs[self.df_BUs['Formula'] == CalcMolFormula(mol)]
            # display(bu_formula)
            for row in bu_formula.itertuples():
                match = mol.GetSubstructMatch(row.ROMol)
                if len(match) == mol.GetNumAtoms():
                    isNP = 'Yes'; bu_id = row.ID[0]
                    break
            if isNP == 'No' and check_unknowns and self.UCL is not None:
                unknown_mol = mol
                already_in_list = False
                new_name = ""
                if not self.UCL:
                    # Create first id
                    new_name = "UC_0001"
                else:
                    for name, structure in self.UCL.items():
                        if len(unknown_mol.GetSubstructMatch(structure, )) == unknown_mol.GetNumAtoms():
                            already_in_list = True
                            new_name = name
                            break
                    if not already_in_list:
                        # Get last UCL compound name
                        last_name = list(self.UCL)[-1]
                        # Get last id number
                        last_id = int(last_name[3:])
                        # Create string of new id with trailing zeroes
                        new_id = str(last_id + 1).zfill(4)
                        # Create new id with format UC_XXXX
                        new_name = f"UC_{new_id}"
                if new_name not in self.UCL:
                    self.UCL[new_name] = unknown_mol
                bu_id = new_name
        
        mol.SetProp('ISOSMILES', Chem.MolToSmiles(mol))
        mol.SetProp('isNP', isNP)
        mol.SetProp('BU_ID', bu_id)
         

    def get_bonds_to_break(self):
        self.bonds_to_break = {}
        for rxn_name, rxn in reactions.items():
            matches = self.query_mol.GetSubstructMatches(Chem.Mol(rxn.GetReactantTemplate(0).ToBinary()))
            for match in matches:
                self.bonds_to_break[match[:2]] = rxn_name
        return self.bonds_to_break


    def get_fragments(self, query_mol, check_unknowns=False):
        # TODO: Instead of removing ions, add them to graph with special 'ionic' edge
        self.query_mol = deepcopy(query_mol)
        remover = SaltRemover()
        self.query_mol = remover.StripMol(self.query_mol)
        self.query_mol = neutralize_atoms(self.query_mol)
        reactant_list = [self.query_mol]
        
        # Remove multi-molecule objects
        if len(Chem.MolToSmiles(self.query_mol).split('.')) > 1:
            self.frag_mols = reactant_list
            for frag in self.frag_mols:
                self.check_bul(frag, check_unknowns)
            return self.frag_mols

        # Use isotopes to keep track of atoms during fragmentation
        for atom in reactant_list[0].GetAtoms():
            atom.SetIsotope(atom.GetIdx()+1)  # Without +1, atom #0 would not get labelled

        # Get reactions involved in the fragmentation
        self.get_bonds_to_break()
        reactions_involved = {r: reactions[r] for r in set(self.bonds_to_break.values())}
        if not reactions_involved:
            self.frag_mols = [query_mol]
            for frag in self.frag_mols:
                self.check_bul(frag, check_unknowns)
            return self.frag_mols

        final_prods = self.fragment_molecule_with_timeout(reactant_list, reactions_involved, 8, check_unknowns)
        # print(final_prods)

        # Use dictionary {(idx): mol} to remove duplicated Mols
        isotope_prods = {tuple(sorted([atom.GetIsotope() for atom in prod.GetAtoms()
            if atom.GetIsotope() != 0])): prod for prod in final_prods}
        
        if len(isotope_prods) == 1:
            self.frag_mols = list(isotope_prods.values())
        else:
            # Combine products to cover the whole query mol
            self.combine_frags_with_timeout(isotope_prods, timeout=10)

        # Add BUL information
        for frag in self.frag_mols:
            remove_isotopes(frag)
            self.check_bul(frag, check_unknowns)

        return self.frag_mols


    def fragment_molecule(self, reactant_list, reactions_involved):
        final_prods = []
        for rxn_name, rxn in list(reactions_involved.items()):
            waiting_list = []
            while True:
                intermediate_prods = []
                for reactant in reactant_list:
                    prod_Mol = rxn.RunReactants((reactant,), maxProducts=1)

                    if not prod_Mol:
                        waiting_list.append(reactant)
                        continue

                    prod_Mol = prod_Mol[0][0]
                    atom_pair = [atom.GetIdx() for atom in prod_Mol.GetAtoms()
                                 if atom.HasProp('old_mapno')]

                    for idx in atom_pair:
                        prod_Mol.GetAtomWithIdx(idx).SetProp(
                            'old_mapno',
                            str(prod_Mol.GetAtomWithIdx(idx).GetIsotope() - 1))

                    # print(rxn_name, prod_Mol)
                    prod_mols = Chem.GetMolFrags(prod_Mol, asMols=True)
                    intermediate_prods.extend(prod_mols)

                if intermediate_prods:
                    reactant_list = waiting_list + intermediate_prods
                    reactant_list = [r for i, r in enumerate(reactant_list)
                                     if Chem.MolToSmiles(r) not in {Chem.MolToSmiles(reactant_list[j])
                                     for j in range(i)}]
                else:
                    final_prods = reactant_list
                    return final_prods

    def fragment_molecule_with_timeout(self, reactant_list, reactions_involved, timeout, check_unknowns=False):
        def target():
            try:
                self.fragment_molecule(reactant_list, reactions_involved)
            except: pass

        thread = threading.Thread(target=target)
        timer = threading.Timer(timeout, lambda: None)

        thread.start()
        timer.start()

        thread.join(timeout)
        timer.cancel()

        if thread.is_alive():
            self.frag_mols = [self.query_mol]
        else:
            self.frag_mols = self.fragment_molecule(reactant_list, reactions_involved)

        remove_isotopes(self.query_mol)
        self.check_bul(self.query_mol, check_unknowns)

        return self.frag_mols


    def combine_frags_with_timeout(self, isotope_prods, timeout):
        def target():
            try:
                for r in range(2, len(isotope_prods.keys()) + 1):
                    for combo in combinations(isotope_prods.keys(), r):
                        if sorted(flatten(combo)) == list(range(1, self.query_mol.GetNumAtoms() + 1)):
                            return [isotope_prods[x] for x in combo]  #<-- Arbitrarily choosing the first combo; TODO: fix
            except: pass

        timer = threading.Timer(timeout, lambda: None)
        thread = threading.Thread(target=target)

        thread.start()
        timer.start()

        thread.join(timeout)
        timer.cancel()

        if thread.is_alive():
            return []
        return target()


    def create_connections_graph(self):
        
        for i, frag_mol in enumerate(self.frag_mols):
            self.G.add_nodes_from([(i, {'SMILES': Chem.MolToSmiles(frag_mol),
                                        'isNP': frag_mol.GetProp('isNP'),
                                        'BU_ID': frag_mol.GetProp('BU_ID'),
                                        'img': mol2svg(frag_mol)
                                        })])
        # Create a dictionary mapping query_mol to frag_mols atoms
        old_to_new_dict = {}
        for i, frag_mol in enumerate(self.frag_mols):
            for atom in frag_mol.GetAtoms():
                if atom.HasProp('old_mapno'):
                    old_to_new_dict[int(atom.GetProp('old_mapno'))] = i
        # print(old_to_new_dict)

        # Create edge/attributes dictionary
        edge_dict = {(old_to_new_dict[k[0]], old_to_new_dict[k[1]]): v
                     for k, v in self.bonds_to_break.items()
                     if k[0] in old_to_new_dict and k[1] in old_to_new_dict}
                     #<-- This last condition arises because, for symmetrical envs,
                     #<-- only one of the two possible breakage contained in bonds_to_break
                     #<-- is actually carried out.
        #TODO Fantastic-biofrag should overcome this fundamental limitation by creating a branching point
        #TODO the fragmentation tree when symmetrical patterns produce asymmetrical blocks.
        # print(edge_dict)

        for edge, attribute in edge_dict.items():
            self.G.add_edge(*edge, reaction=attribute)
        
        return self.G
    

    def show_connections_graph(self):

        # self.create_connections_graph()
        # nx.write_graphml(self.G, "graph.graphml")

        # #! ------------- Plot with yEd ------------!#
        # '''
        # https://colab.research.google.com/github/yWorks/yfiles-jupyter-graphs/blob/main/examples/00_toc.ipynb
        # '''

        # def custom_images(id, node):
        #     label = list(self.G.nodes())[id]
        #     img = self.G.nodes[label]['img']
        #     return {'img': img}
        
        # # def custom_factor_mapping(node):
        # #     """choose random factor"""
        # #     return 2

        # from yfiles_jupyter_graphs import GraphWidget
        # w = GraphWidget(graph=self.G)
        # # w.hierarchic_layout()
        # w.node_label_mapping = ''  # Remove labels
        # w.set_node_styles_mapping(custom_images)
        # # w.get_edge_thickness_factor_mapping(custom_factor_mapping)
        # # w.del_node_property_mapping()
        # # w.set_node_size_mapping = 200
        # w.set_overview(enabled=False)
        # w.set_sidebar(enabled=False)
        # w.show()

        # nx.write_graphml(self.G, "your_graph.graphml")
        #! -----------------------------------------!#

        #! ------------- Plot with Plotly ------------!#
        # import plotly.graph_objects as go
        # # Create positions for nodes using spring layout
        # pos = nx.spring_layout(self.G, k=1)

        # # Extract the positions into separate x and y lists for Plotly
        # x_vals = [pos[n][0] for n in self.G.nodes()]
        # y_vals = [pos[n][1] for n in self.G.nodes()]

        # # Create node trace for Plotly
        # node_trace = go.Scatter(
        #     x=x_vals, y=y_vals,
        #     mode='markers',
        #     marker=dict(size=110, color='white',
        #       line=dict(width=1.7, color=['green' if self.G.nodes[n].get('isNP') == 'Yes'
        #                                       else 'red' for n in self.G.nodes()])),
        #     text=[f'Node {n}' for n in self.G.nodes()],
        #     hoverinfo='text'
        # )

        # # Create edge trace
        # edge_trace = go.Scatter(
        #     x=sum([[pos[e[0]][0], pos[e[1]][0], None] for e in self.G.edges()], []),
        #     y=sum([[pos[e[0]][1], pos[e[1]][1], None] for e in self.G.edges()], []),
        #     line=dict(width=1.7, color='black'),
        #     hoverinfo='none',
        #     mode='lines'
        # )

        # # Create the figure with the node and edge traces
        # fig = go.Figure(data=[edge_trace, node_trace])

        # # Create list of images (SVG or PNG) to overlay on the graph
        # image_annotations = []
        # for node in self.G.nodes(data=True):
        #     x, y = pos[node[0]]
        #     image_url = node[1]['img']
        #     image_annotations.append(
        #         dict(
        #             source=image_url,
        #             xref="x", yref="y",
        #             x=x, y=y,
        #             sizex=.3, sizey=.3,
        #             xanchor="center", yanchor="middle",
        #             layer="above",
        #         )
        #     )

        # # Add the images (SVGs or PNGs) to the layout
        # fig.update_layout(images=image_annotations,)

        # # Configure layout
        # fig.update_layout(
        #     showlegend=False,
        #     xaxis=dict(showgrid=False, zeroline=False, visible=False),  # Remove x-axis
        #     yaxis=dict(showgrid=False, zeroline=False, visible=False),  # Remove y-axis
        #     paper_bgcolor="white",     # Set background color to white
        #     plot_bgcolor="white",      # Set plot background to white
        #     margin=dict(t=0, b=0, l=0, r=0),  # Remove padding around plot
        #     width=600,                 # Set width
        #     height=600                 # Set height to be equal to width
        # )

        # # Display the graph
        # fig.show()
        #! -----------------------------------------!#

        #! ----------- Plot with Cytoscape -----------!#
        # # #* get ideas from https://giter.vip/cytoscape/ipycytoscape
        # # #* and https://dash.plotly.com/cytoscape/layout

        cyto = CytoscapeWidget()
        cyto.layout.height = '500px'
        cyto.layout.width = '1400px'
        cyto.graph.add_graph_from_networkx(self.G)
        cyto.set_layout(name='cose')

        cyto.set_style([{'selector': 'node',
                         'css': {
                            'content': 'data(count)',
                            'width': 50,
                            'height': 50,
                            'border-width': 1.5,
                            'background-color': '#FFFFFF',
                            # 'background-fit': 'contain',
                            'background-width': '90%',
                            'background-height': '90%',
                            'background-image': 'data(img)'}},

                        {'selector': "node[isNP = 'Yes']",
                         'css': {'border-color': '#009b73'}},
                        
                        {'selector': "node[isNP = 'No']",
                         'css': {'border-color': '#d10808'}},
                        
                        {'selector': 'edge',
                         'css': {
                            'curve-style': 'bezier',
                            'width': 1.5,
                            'label': 'data(reaction)',
                            'font-family': 'helvetica',
                            'font-size': '10px',
                            'line-color': '#009b73'}},
                        
                        {'selector': "edge[reaction='']",
                         'css': {'line-color': '#d10808'}}
                        ])

        return cyto