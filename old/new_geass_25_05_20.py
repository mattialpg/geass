import re, sys, os
import threading
from urllib import parse
import pandas as pd
from itertools import combinations
from copy import deepcopy

import networkx as nx
from ipycytoscape import CytoscapeWidget

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.MolStandardize.standardize import enumerate_tautomers_smiles, canonicalize_tautomer_smiles
from rdkit.DataStructs import TanimotoSimilarity
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

current_file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_file_dir)

#* Nomenclature
#* fragment        >  mol with dummy atom(s)
#* building unit   >  fragment whose dummy atoms (*) have been replaced with real atoms 
#* building block  >  molecule formed by two or more building units

#* In reaction smarts, the breaking bond must always be between atom 0 and 1.
#* Reaction products must be written in parentheses and separated by a dot.
#* They can be separated using Chem.GetMolFrags(product_Mol).

reactions = [
    ['amide', "[NX3;$(N([#6])[#6]):0]-[C:1]=[O:2]>>([#7:0].[C:1](=[O:2])-O)", [(0,1)]],
    ['ester', "[O;$(O([#6;!$([CH3])])([#6;!$([CH3])])):0]-[C;!$([C;r5,r6](-[O;r5,r6])):1]=[O:2]>>([O:0].[C:1](=[O:2])-O)", [(0,1)]],
    ['thioester', "[SX2:0]-[C:1]=[O:2]>>([SX2:0].[C:1](=[O:2])-O)", [(0,1)]],
    ['ether', "[#6;!$(C=O);!$([CH3]);!$([C;r5,r6](-[O;r5,r6])-O):0]-[OX2H0;!r;!$(O-[C;$([C;r5,r6](-[O;r5,r6])-O)]);!$(O-[CH3]):1]>>([#6:0]-O.[O:1])", [(0,1)]],
    ['glycosidic', "[C;$([C;r5,r6](-[O;r5,r6])-O):0]-[OX2H0;!r;!$(O-[CH3]):1]>>([C:0]-O.[O:1])", [(0,1)]],
    ['N-alkyl', "[#7:0][CH3:1]>>([#7:0].[C:1])", [(0,1)]],
    ['O-alkyl', "[#8H0:0][CH3:1]>>([#8H1:0].[C:1])", [(0,1)]],
    ['iminol', "[NX2H0;!R:0]=[C:1]-[O:2]>>([N:0].O[C:1]=[OX2H0:2])", [(0,1)]],
    # ['thiazole', "[#16;r5:0]~[#6;r5:1]~[#7;r5:2]~[#6;r5:3]~[#6;r5:4]>>([#16:0].[#6:1]=[#7:2].[O]=[#6:3]-[#6:4])", [(0,1), (2,3)]],
    # ['acyl', "[OX2;!R;$(O(C(=O)[*])[*]):1](-[CX4H1;R:2])(-[C;!R;$(C(=O)(O-[*])[*]):3])>>([OX2H;$(O[#6]);!$(OC(=O)):1](-[CX4H1;R:2]).[C;!R;$(C(=O)(O-[*])[*]):3])", [(1,2)]],
]

df_reactions = pd.DataFrame(reactions, columns=['ec_num', 'rule', 'broken_bonds'])
df_reactions['rxn'] = df_reactions['rule'].apply(AllChem.ReactionFromSmarts)

# # Add reaction rules
# df_reactions = pd.read_csv(R'C:\Users\Idener\Sync\CODE\fantastic-biofragmenter\reactions\EnzymeMap_rules.csv')
# df_reactions['rxn'] = df_reactions['rule'].apply(AllChem.ReactionFromSmarts)
# df_reactions['broken_bonds'] = df_reactions['broken_bonds'].apply(eval)


def flatten(list):
    flat_list = [item for sublist in list for item in sublist]
    return(flat_list)

def remove_isotopes(mol):
    for atom in mol.GetAtoms():
        atom.SetIsotope(0)

def neutralize_atoms(mol):
    """
    https://baoilleach.blogspot.com/2019/12/no-charge-simple-approach-to.html
    +1 charged atoms must have a hydrogen, or we can't remove H+
    we must check neighbor atom charges to avoid altering groups
    with charge-separated representations (e.g. nitro).
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

def reset_view(mol):
    """ Reset coordinates for display """
    mol = deepcopy(mol)
    Chem.rdDepictor.SetPreferCoordGen(True)
    Chem.rdDepictor.Compute2DCoords(mol)
    Chem.rdDepictor.StraightenDepiction(mol)
    return mol


class Geass:
    _df_BUL = pd.DataFrame()
    _df_UCL = pd.DataFrame()

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.bonds_to_break = {}
        self.reactions_involved = {}
        self.frag_mols = []
        self.G = nx.Graph()

        # Initialize libraries once at the class level
        if Geass._df_BUL.empty or Geass._df_UCL.empty:
            self._initialize_libraries()

    @property
    def df_BUL(self):
        """Access df_BUL through a property."""
        return Geass._df_BUL
    
    @property
    def df_UCL(self):
        return Geass._df_UCL

    def _initialize_libraries(self):
        try:
            Geass._df_BUL = pd.read_pickle(f'{current_file_dir}/data/Database_NEWBUL.pkl')
            Geass._df_UCL = pd.read_csv(f'{current_file_dir}/data/Database_UCL.csv', keep_default_na=False)
        except Exception as e:
            print(f"Error loading libraries: {e}")


    def check_BUs(self, mol, check_unknowns):
        #TODO: arginine is not recognised. Probably because of tautomers

        isNP = 'No'; bu_id = 'NA'; brite = 'NA'
        inchi = Chem.MolToInchi(mol)
        smiles = Chem.MolToSmiles(mol)

        # # First, use smiles to to search BUs (faster)
        # bu_from_smiles = self.df_BUL[self.df_BUL['ISOSMILES'] == Chem.MolToSmiles(mol)]
        # if len(bu_from_smiles) > 0:
        #     isNP = 'Yes'
        #     bu_id = bu_from_smiles.iloc[0]['ID']
        #     brite = bu_from_smiles.iloc[0]['BRITE']
        # else:
        #     # Second, use SubstructMatch to search BUs (slower)
        #     bu_from_formula = self.df_BUL[self.df_BUL['Formula'] == CalcMolFormula(mol)]
        #     for row in bu_from_formula.itertuples():
        #         match = mol.GetSubstructMatch(row.ROMol)
        #         if len(match) == mol.GetNumAtoms():
        #             isNP = 'Yes'
        #             bu_id = bu_from_formula.iloc[0]['ID']
        #             brite = bu_from_formula.iloc[0]['BRITE']
        #             break

        df_aux = self.df_BUL[self.df_BUL['InChI'] == inchi]
        if len(df_aux) > 0:
            isNP = 'Yes'
            bu_id = df_aux.iloc[0]['ID']
            brite = df_aux.iloc[0]['BRITE']

        # Assign U-code to non-NP BUs
        if isNP == 'No' and check_unknowns:
            df_aux = self.df_UCL[self.df_UCL['ISOSMILES'] == smiles]
            if len(df_aux) > 0:
                if df_aux.iloc[0]['Similar_KID'] != 'NA':
                    isNP = 'Similar'
                    bu_id = f"~{df_aux.iloc[0]['Similar_KID']}"
                else:
                    bu_id = df_aux.iloc[0]['UID']
            else:
                last_uid = self.df_UCL['UID'].iloc[-1]
                new_uid = re.sub(R"\d{5}", lambda x: f"{int(x.group())+1:05}", last_uid)
                bu_id = new_uid

                mw = Chem.rdMolDescriptors.CalcExactMolWt(mol)
                fp = Chem.RDKFingerprint(mol, maxPath=7, minPath=1, fpSize=2048)

                df_aux_2 = self.df_BUL[(self.df_BUL['MW'] - 70 <= mw) & (self.df_BUL['MW'] + 70 >= mw)]

                if len(df_aux_2) > 0:
                    df_aux_2['similarity'] = df_aux_2['FP'].apply(lambda x: TanimotoSimilarity(x, fp))
                    max = df_aux_2.loc[df_aux_2['similarity'].idxmax()]
                    if max['similarity'] >= 0.6:
                        isNP = 'Similar'
                        bu_id = f"~{max.ID}"
                        # display(df_aux_2)
                    # else:
                    #     # Check tautomers/charged forms
                    #     # rdMolStandardize.StandardizeSmiles(smiles)
                    #     for smiles_tau in enumerate_tautomers_smiles(smiles):
                    #         mol_tau = Chem.MolFromSmiles(smiles_tau)
                    #         fp = Chem.RDKFingerprint(mol_tau, maxPath=7, minPath=1, fpSize=2048)

                    #         df_aux_2['similarity'] = df_aux_2['FP'].apply(lambda x: TanimotoSimilarity(x, fp))
                    #         max = df_aux_2.loc[df_aux_2['similarity'].idxmax()]
                    #         if max['similarity'] >= 0.6:
                    #             isNP = 'Similar'
                    #             bu_id = f"~{max.ID}"
                    #             # display(df_aux_2)
                    #             break

        mol.SetProp('ISOSMILES', smiles)
        mol.SetProp('isNP', isNP)
        mol.SetProp('BU_ID', bu_id)
        # mol.SetProp('BRITE', brite)
         
    def get_bonds_to_break(self, query_mol):
        ''' query_mol is treated as a non-self variable to
        allow for the function to be called outside the class '''
        for row in df_reactions.itertuples():
            matches = query_mol.GetSubstructMatches(row.rxn.GetReactants()[0])
            for match in matches:
                for bond in row.broken_bonds:
                    self.bonds_to_break[(match[bond[0]], match[bond[1]])] = row.ec_num
                self.reactions_involved[row.ec_num] = row.rxn
        return self.bonds_to_break


    def get_fragments(self, query_mol, check_unknowns=True):
        self.query_mol = deepcopy(query_mol)
        try:
            # TODO: Instead of removing ions, add them to graph with special 'ionic' edge
            remover = SaltRemover()
            self.query_mol = remover.StripMol(self.query_mol)
            self.query_mol = neutralize_atoms(self.query_mol)
        except: pass
        
        # Get reactions involved in the fragmentation
        self.get_bonds_to_break(self.query_mol)
        self.reactions_involved = {r: df_reactions[df_reactions['ec_num'] == r]['rxn'].tolist()[0]
                                      for r in set(self.bonds_to_break.values())}
        # print(self.bonds_to_break)
        
        if not self.reactions_involved:
            self.check_BUs(self.query_mol, check_unknowns)
            self.frag_mols = [self.query_mol]
            return self.frag_mols

        # Use isotopes to keep track of atoms during fragmentation
        for atom in self.query_mol.GetAtoms():
            atom.SetIsotope(atom.GetIdx()+1)  # Without +1, atom #0 would not get labelled

        self.frag_mols = self.fragment_molecule_with_timeout(check_unknowns, timeout=8)
        # display(*self.frag_mols)

        # Use dictionary {(idx): mol} to remove duplicated Mols
        isotope_prods = {tuple(sorted([atom.GetIsotope() for atom in prod.GetAtoms()
            if atom.GetIsotope() != 0])): prod for prod in self.frag_mols}
        
        if len(isotope_prods) == 1:
            self.frag_mols = list(isotope_prods.values())
        else:
            # Combine products to cover the whole query mol
            self.combine_frags_with_timeout(isotope_prods, timeout=10)

        # Add BU properties
        for frag in self.frag_mols:
            for atom in frag.GetAtoms():
                atom.SetProp('old_aam', str(atom.GetIsotope()-1))
            remove_isotopes(frag)
            self.check_BUs(frag, check_unknowns)

        return self.frag_mols


    def fragment_molecule(self, mark_unknowns):
        reactant_list = [self.query_mol]

        for bond, rxn_name in self.bonds_to_break.items():
            aux_list = reactant_list
            for i, reactant in enumerate(reactant_list):
                rxn = self.reactions_involved[rxn_name]

                #     bu_bool = self.check_BUs(prod_mol, mark_unknowns)
                #     if bu_bool:
                #         final_prods.append(prod_mol)
                #     else:
                #         waiting_list.extend(prod_mols)
            
                prod_Mol = rxn.RunReactants((reactant,), maxProducts=1)

                if not prod_Mol:
                    continue

                prod_Mol = prod_Mol[0][0]
                # display(prod_Mol)

                atom_pair = [atom.GetIdx() for atom in prod_Mol.GetAtoms()
                                if atom.HasProp('old_aam')]
                for idx in atom_pair:
                    prod_Mol.GetAtomWithIdx(idx).SetProp('old_aam',
                        str(prod_Mol.GetAtomWithIdx(idx).GetIsotope() - 1))

                if rxn_name == 'thiazole':
                    Chem.Kekulize(prod_Mol, clearAromaticFlags=True)
                    Chem.SanitizeMol(prod_Mol)

                prod_mols = Chem.GetMolFrags(prod_Mol, asMols=True)
                aux_list.pop(i)
                aux_list.extend(prod_mols)

            reactant_list = aux_list
        return reactant_list


    def fragment_molecule_with_timeout(self, mark_unknowns, timeout):
        def target():
            try:
                self.fragment_molecule(mark_unknowns)
            except Exception as e:
                print(e)

        thread = threading.Thread(target=target)
        timer = threading.Timer(timeout, lambda: None)

        thread.start()
        timer.start()

        thread.join(timeout)
        timer.cancel()

        if thread.is_alive():
            self.frag_mols = [self.query_mol]
        else:
            self.frag_mols = self.fragment_molecule(mark_unknowns)

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


    def create_connection_graph(self):
        # Add fragments as nodes
        properties = ['ISOSMILES', 'isNP', 'BU_ID']
        nodes = []
        for i, frag_mol in enumerate(self.frag_mols):
            node_attrs = {prop: frag_mol.GetProp(prop) for prop in properties}
            node_attrs['img'] = mol2svg(frag_mol)
            nodes.append((i, node_attrs))
        self.G.add_nodes_from(nodes)
        
        # Map fragments to nodes and add edges
        edges = []
        for k, v in self.bonds_to_break.items():
            edge, idx = [], []
            for i, frag_mol in enumerate(self.frag_mols):
                for atom in frag_mol.GetAtoms():
                    if int(atom.GetProp('old_aam')) in k:
                        edge.append(i)
                        idx.append(atom.GetIdx())
                        break
                    if len(edge) == 2:
                        break
            if len(edge) == 2:
                edges.append((*edge, {'reaction': v}))

        self.G.add_edges_from(edges)
        return self.G
    

    def show_connection_graph(self, method='cytoscape'):

        self.G = self.create_connection_graph()
        # nx.write_graphml(self.G, "your_graph.graphml")

        #! ------------- Plot with yEd ------------!#
        if method == 'yEd':
            """
            https://colab.research.google.com/github/yWorks/yfiles-jupyter-graphs/blob/main/examples/00_toc.ipynb
            """

            def custom_images(id, node):
                label = list(self.G.nodes())[id]
                img = self.G.nodes[label]['img']
                return {'img': img}
            
            # def custom_factor_mapping(node):
            #     """choose random factor"""
            #     return 2

            from yfiles_jupyter_graphs import GraphWidget
            w = GraphWidget(graph=self.G)
            # w.hierarchic_layout()
            w.node_label_mapping = ''  # Remove labels
            w.set_node_styles_mapping(custom_images)
            # w.get_edge_thickness_factor_mapping(custom_factor_mapping)
            # w.del_node_property_mapping()
            # w.set_node_size_mapping = 200
            w.set_overview(enabled=False)
            w.set_sidebar(enabled=False)
            w.show()
            # return w
            # return self.G

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
        
    
    # def recompose_molecule(self, mol):
        #* MWE of reversing molecular fragmentation
        # # It lacks recognition of atoms forming bonds  
        # tree = []
        # for edge in g.edges:
        #     node1, node2 = edge
        #     smiles1 = g.nodes[node1]['SMILES']
        #     smiles2 = g.nodes[node2]['SMILES']
        #     tree.append((smiles1, smiles2, g.edges[edge]['reaction']))

        # reactions = {'amide': AllChem.ReactionFromSmarts("[NX3;$(N([#6])[#6]):0]-[C:1]=[O:2]>>[#7:0].[C:1](=[O:2])-O")}

        # rev_reactions = {}
        # for name, rxn in reactions.items():
        #     rxn_rev = AllChem.ChemicalReaction()
        #     for i in range(rxn.GetNumReactantTemplates()):
        #         rxn_rev.AddProductTemplate(rxn.GetReactantTemplate(i))
        #     for i in range(rxn.GetNumProductTemplates()):
        #         rxn_rev.AddReactantTemplate(rxn.GetProductTemplate(i))
        #     rxn_rev.Initialize()
        #     rev_reactions[name] = rxn_rev

        # There shuold be a way to enumerate atoms linked in the original molecule
        # So that they can be rejoined again 
        # new_prods = rxn_rev.RunReactants('.....')[0][0]


    def replace_BUs(self, query_mol, bu_to_replace, new_bu, bu_class):
        
        def prepare_BUs(bu):
            replacements = {'aas': (Chem.MolFromSmiles('CC(N)C(=O)O'), Chem.MolFromSmiles('C')),  # Working from C-beta onwards
                            'fas': (Chem.MolFromSmarts('C(=O)O'), Chem.MolFromSmiles('[14C](=O)'))}  # Labelling to keep track during replacement
            replacement_centers = {'aas': Chem.MolFromSmarts('[CX3;$(CC(N)C(=O))]'),  # C-beta
                                   'fas': Chem.MolFromSmarts('[14C;$(C(=O))]')}  # If not labelled, the first matching carbonyl group is chosen
            
            # Prepare BUs to be replaced
            bu = Chem.ReplaceSubstructs(bu, replacements[bu_class][0], replacements[bu_class][1])[0]

            # Renumber atoms such that atom 0 is the one involved in bond formation
            patt = bu.GetSubstructMatch(replacement_centers[bu_class])
            new_order = list(patt) + [x for x in range(bu.GetNumAtoms()) if x not in patt][::-1]
            bu = Chem.ReplaceSubstructs(bu, Chem.MolFromSmiles('[14C](=O)'), Chem.MolFromSmiles('C(=O)'))[0]  # Removing labelling
            Chem.SanitizeMol(bu)

            return AllChem.RenumberAtoms(bu, new_order)

        bu_to_replace, new_bu = [prepare_BUs(bu) for bu in [bu_to_replace, new_bu]]

        # Replace BU
        chimera = AllChem.ReplaceSubstructs(query_mol, bu_to_replace, new_bu, useChirality=True)[0]

        #! BUG: ReplaceSubstructs automatically invert chirality even with useChirality=True.
        #! Revert to originl chirality
        # C_beta = query_mol.GetAtomWithIdx(query_mol.GetSubstructMatch(bu_to_replace)[0])
        # C_alpha_idx = C_beta.GetNeighbors()[0].GetIdx()
        # chimera.GetAtomWithIdx(C_alpha_idx).InvertChirality()
        
        chimera = Chem.RemoveHs(chimera)
        Chem.SanitizeMol(chimera)

        return chimera
        # return reset_view(chimera)