# import necessary modules

import re
import time
import signal
import ipycytoscape
import pandas as pd
import networkx as nx
import metadisassembler5 as medi5
from collections import Counter
from urllib import parse
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdChemReactions
from IPython.display import clear_output

def get_SMILES_length_from_mol(mol):
    return len(Chem.MolToSmiles(mol))

def disassembler_timeout_handler(signum, frame):
    print("Timeout of 60s reached. Skipping the fragment.")
    raise Exception("Timeout of 60s reached. Skipping the fragment.")

# Helper functions to avoid typing the same thing over and over
def to_smiles(mol_tuple):
    return tuple(Chem.MolToSmiles(mol) for mol in mol_tuple)

def from_smiles(smiles_tuple):
    return tuple(Chem.MolFromSmiles(smiles) for smiles in smiles_tuple)

def run_fragmentation(mol_to_fragment, rxn, fragments_aux):
    '''
    Fragments a molecule recursively with the reaction rule given by rxn.

    Inputs:
        - mol_to_fragment: rdkit.Chem.mol to fragment
        - rxn: rdkit.Chem.rdChemReactions.ChemicalReaction reaction rule (use rdChemReactions.ReactionFromSmarts(smarts))
        - fragments_aux: The list that will be recursively filled with molecules

    Outputs:
        - The list of current molecule fragments
    '''
    reactants = (mol_to_fragment, )
    products = rxn.RunReactants(reactants)
    if products:
        # "Clean" mol by transforming it to SMILES and back to mol
        smiles_aux = to_smiles(products[0])
        tpl_prod = from_smiles(smiles_aux)
        for t_idx, clean_prod in enumerate(tpl_prod):
            # if clean_prod is None:
            #     prod_fragments = Chem.GetMolFrags(products[0][t_idx], asMols=True, sanitizeFrags=False)
            # else:
            #     prod_fragments = Chem.GetMolFrags(clean_prod, asMols=True)
            prod_fragments = Chem.GetMolFrags(clean_prod, asMols=True)
            for prod_fragment in prod_fragments:
                fragments_aux = run_fragmentation(prod_fragment, rxn, fragments_aux)
    else:
        fragments_aux.append(mol_to_fragment)
    return fragments_aux

def create_graphs_without_MD(mols, indexer_original_mols, brokenBondsDefs, reactants, reactions, fillingAtoms, BULnames, UCL, old_version=False):
    start_time = time.time()
    result_graphs = {}
    unknown_occurrences = {}

    for mol_idx, current_mol in enumerate(mols):
        clear_output()
        print(f'PROGRESS: [{mol_idx+1}/{len(mols)}]')
        error_geass = False
        try:
            # Initialize Geass
            fragmentator = Geass(current_mol, brokenBondsDefs, reactants, reactions, fillingAtoms, BULnames, UCL, verbose=0)

            # Get bonds to break
            fragmentator.get_bonds_to_break()
            
            # Create graph
            if old_version:
                fragmentator.create_fragmentation_graph_old()
            else:
                fragmentator.create_fragmentation_graph()

            # Update the UCL
            UCL = fragmentator.UCL
            
            # Get not NP fragments
            graph_nodes = list(fragmentator.graph.nodes(data=True))
            not_NP_fragments = {frag[0]: frag[1]['kegg_id'] for frag in graph_nodes if frag[1]['isNP'] == 'No'}
        except:
            error_geass = True

        if not error_geass:
            for j, unknown_fragment_ID in not_NP_fragments.items():
                if unknown_fragment_ID not in unknown_occurrences:
                    unknown_occurrences[unknown_fragment_ID] = []
                unknown_occurrences[unknown_fragment_ID].append(indexer_original_mols[mol_idx])
        
        # Add the graph to the result graph list
        result_graphs[indexer_original_mols[mol_idx]] = fragmentator
    
    print(f"Elapsed: {round(time.time() - start_time, 3)}s")    
    return result_graphs, UCL, unknown_occurrences

def create_graphs(mols, indexer_original_mols, PBU, BUL, brokenBondsDefs, reactants, reactions, fillingAtoms, BULnames, output_folder="./metadisassembler_v3_results", old_version=False):
    signal.signal(signal.SIGALRM, disassembler_timeout_handler)
    start_time = time.time()
    result_graphs = {}
    disassembled_ids = {}

    # Lists to analyze results
    fragmented_all_NP = set()
    fragmented_not_NP_OK = set()
    fragmented_not_NP_ERROR = set()
    not_fragmented_all_NP = set()
    not_fragmented_not_NP_OK = set()
    not_fragmented_not_NP_ERROR = set()

    for mol_idx, current_mol in enumerate(mols):
        clear_output()
        print(f'PROGRESS: [{mol_idx+1}/{len(mols)}]')
        error_geass = False
        try:
            # Initialize Geass
            fragmentator = Geass(current_mol, brokenBondsDefs, reactants, reactions, fillingAtoms, BULnames, verbose=0)

            # Get bonds to break
            fragmentator.get_bonds_to_break()
            
            # Create graph
            if old_version:
                fragmentator.create_fragmentation_graph_old()
            else:
                fragmentator.create_fragmentation_graph()
            
            # Get not NP fragments
            graph_nodes = list(fragmentator.graph.nodes(data=True))
            not_NP_fragments = {frag[0]: frag[1]['SMILES'] for frag in graph_nodes if frag[1]['isNP'] == 'No'}
            
            fragments = {}
            fragment_names = {}
        except:
            error_geass = True
        
        if not error_geass:
            # Run metabolic disassembler
            for j, molecule_smiles in not_NP_fragments.items():
                # Iterate over the molecules 

                disassembler = medi5.MetaDisassembler5() # Init the disassembler
                disassembler.output_folder = output_folder # Choose the output folder
                disassembler.useChirality = False # Whether to use Chilarity. IT HAS NOT BEEN TESTED IN 'TRUE' MODE
                disassembler.pbu = PBU # Assign the pbu
                disassembler.bul = BUL # Assign the bul
                disassembler.precomputed_qsu = False # We are not computed the qsu yet.
                disassembler.n_img = 1 
                disassembler.time_limit = 1
                name = str(j) + '_' + indexer_original_mols[mol_idx]  # The name of the molecule
                molecule = Chem.MolFromSmiles(molecule_smiles) # The molecule from the current SMILES
                disassembler.input_molecule(molecule, name) # Define the input with the molecular structure and the name

                try:
                    # Define the timeout for disassembling
                    signal.alarm(60)
                    disassembler.disassemble() # disassemble
                    # Cancel the timer if the function finished before timeout
                    signal.alarm(0)
                    fragment_names[j] = [disassembler.query_specific_bul.names[disassembler.match_bu_idx[k]] for aux, k in enumerate(disassembler.result[0]["target"])]
                    fragments[j] = [disassembler.query_specific_bul.cpds[disassembler.match_bu_idx[k]] for aux, k in enumerate(disassembler.result[0]["target"])]
                except:
                    print('Error')
                    fragments[j] = 'Error'
                    fragment_names[j] = 'Error'

            # Add the Kegg ids to the id list of current compound
            disassembled_ids[indexer_original_mols[mol_idx]] = []
            for names in fragment_names.values():
                if names != 'Error':
                    disassembled_ids[indexer_original_mols[mol_idx]] = disassembled_ids[indexer_original_mols[mol_idx]] + names
                else:
                    disassembled_ids[indexer_original_mols[mol_idx]] = disassembled_ids[indexer_original_mols[mol_idx]] + [names]
            if fragments:
                if 'Error' not in fragments.values():
                    # Add new nodes and edges
                    for not_NP_index in not_NP_fragments:
                        # Creating new nodes list
                        previous_graph_length = len(fragmentator.graph.nodes)
                        new_nodes = [(i + previous_graph_length, {'SMILES': Chem.MolToSmiles(frag_mol.mol), 'img': smi2image(Chem.MolToSmiles(frag_mol.mol)), 'isNP': 'Yes', 'kegg_id': fragment_names[not_NP_index][i][0]}) for i, frag_mol in enumerate(fragments[not_NP_index])]
                        # Adding the nodes to the Geass graph
                        fragmentator.graph.add_nodes_from(new_nodes)

                        # Creating new edges list
                        new_edges = []
                        for fragment_index in range(len(fragments[not_NP_index])):
                            new_edges.append((not_NP_index, fragment_index + previous_graph_length, {'reaction': 'unknown'}))
                        # Adding the edges to the Geass graph
                        fragmentator.graph.add_edges_from(new_edges)
                    # Add compound to corresponding no NP OK analytics list
                    if len(graph_nodes) == 1:
                        disassembled_ids[indexer_original_mols[mol_idx]] = [graph_nodes[0][1]['kegg_id']] + disassembled_ids[indexer_original_mols[mol_idx]]
                        not_fragmented_not_NP_OK.add(indexer_original_mols[mol_idx])
                    else:
                        disassembled_ids[indexer_original_mols[mol_idx]] = [gn[1]['kegg_id'] for gn in graph_nodes] + disassembled_ids[indexer_original_mols[mol_idx]]
                        fragmented_not_NP_OK.add(indexer_original_mols[mol_idx])
                else:
                    # Add compound to corresponding no NP ERROR analytics list
                    if len(graph_nodes) == 1:
                        disassembled_ids[indexer_original_mols[mol_idx]] = [graph_nodes[0][1]['kegg_id']] + disassembled_ids[indexer_original_mols[mol_idx]]
                        not_fragmented_not_NP_ERROR.add(indexer_original_mols[mol_idx])
                    else:
                        disassembled_ids[indexer_original_mols[mol_idx]] = [gn[1]['kegg_id'] for gn in graph_nodes] + disassembled_ids[indexer_original_mols[mol_idx]]
                        fragmented_not_NP_ERROR.add(indexer_original_mols[mol_idx])
            else:
                # Add compound to corresponding NP analytics list
                if len(graph_nodes) == 1:
                    disassembled_ids[indexer_original_mols[mol_idx]] = [graph_nodes[0][1]['kegg_id']]
                    not_fragmented_all_NP.add(indexer_original_mols[mol_idx])
                else:
                    disassembled_ids[indexer_original_mols[mol_idx]] = [gn[1]['kegg_id'] for gn in graph_nodes]
                    fragmented_all_NP.add(indexer_original_mols[mol_idx])
            # Add the graph to the result graph list
            result_graphs[indexer_original_mols[mol_idx]] = fragmentator
    print(f"Elapsed: {round(time.time() - start_time, 3)}s")
    analytics_result = {
        'Fragmented All NP': list(fragmented_all_NP),
        'Fragmented not NP OK': list(fragmented_not_NP_OK),
        'Fragmented not NP ERROR': list(fragmented_not_NP_ERROR),
        'Not fragmented All NP': list(not_fragmented_all_NP),
        'Not fragmented not NP OK': list(not_fragmented_not_NP_OK),
        'Not fragmented not NP ERROR': list(not_fragmented_not_NP_ERROR)
    }
    return result_graphs, disassembled_ids, analytics_result

def create_compound_edges(result_graphs):
    compound_edges = {}
    for shared_name, rg in result_graphs.items():
        current_edges = []
        #print(shared_name)
        for edge in list(rg.graph.edges(data=True)):
            current_kegg_id_0 = rg.graph.nodes[edge[0]]['kegg_id']
            current_kegg_id_1 = rg.graph.nodes[edge[1]]['kegg_id']
            if current_kegg_id_0 == 'NA':
                edge0name = rg.get_all_NA_neighbours_names(edge[0])
            else:
                edge0name = current_kegg_id_0[0] if type(current_kegg_id_0) == tuple else current_kegg_id_0
            if current_kegg_id_1 == 'NA':
                edge1name = rg.get_all_NA_neighbours_names(edge[1])
            else:
                edge1name = current_kegg_id_1[0] if type(current_kegg_id_1) == tuple else current_kegg_id_1
            current_edges.append((edge0name, edge1name, edge[2]['reaction']))
        compound_edges[shared_name] = tuple(current_edges)
    return compound_edges

def create_similarity_matrix(compound_edges):
    start = time.time()
    similarity_matrix = []
    for i, sent_i in enumerate(compound_edges.values()):
        templist = []
        for j, sent_j in enumerate(compound_edges.values()):
            if i == j:
                templist.append(1)
            else:
                templist.append(jaccard_similarity(sent_i, sent_j))
        similarity_matrix.append(templist)
    similarity_df = pd.DataFrame(similarity_matrix, index=list(compound_edges.keys()), columns=list(compound_edges.keys()))
    print(f"Elapsed: {round(time.time() - start, 3)}s")
    return similarity_df

def get_graph_intersection(G_graph, H_graph):
    G_nodes = [n for n in G_graph.graph.nodes(data=True)]
    H_nodes = [n for n in H_graph.graph.nodes(data=True)]
    G_edges = [e for e in G_graph.graph.edges(data=True)]
    H_edges = [e for e in H_graph.graph.edges(data=True)]

    G_edges_intersected = []
    H_edges_intersected = []

    for e1 in G_edges:
        source1_idx = e1[0]
        source1 = G_graph.graph.nodes[source1_idx]['kegg_id'] if G_graph.graph.nodes[source1_idx]['kegg_id'] != 'NA' else G_graph.get_all_NA_neighbours_names(source1_idx)
        target1_idx = e1[1]
        target1 = G_graph.graph.nodes[target1_idx]['kegg_id'] if G_graph.graph.nodes[target1_idx]['kegg_id'] != 'NA' else G_graph.get_all_NA_neighbours_names(target1_idx)
        reaction1 = e1[2]
        for e2 in H_edges:
            source2_idx = e2[0]
            source2 = H_graph.graph.nodes[source2_idx]['kegg_id'] if H_graph.graph.nodes[source2_idx]['kegg_id'] != 'NA' else H_graph.get_all_NA_neighbours_names(source2_idx)
            target2_idx = e2[1]
            target2 = H_graph.graph.nodes[target2_idx]['kegg_id'] if H_graph.graph.nodes[target2_idx]['kegg_id'] != 'NA' else H_graph.get_all_NA_neighbours_names(target2_idx)
            reaction2 = e2[2]
            if (source1 == source2 and target1 == target2 or source1==target2 and target1 == source2) and reaction1 == reaction2:
                G_edges_intersected.append((source1_idx, target1_idx))
                H_edges_intersected.append((source2_idx, target2_idx))
                H_edges.remove(e2)
                break
                
    G_nodes_intersected = set(item for sublist in G_edges_intersected for item in sublist)
    H_nodes_intersected = set(item for sublist in H_edges_intersected for item in sublist)
    print(G_edges_intersected)
    print(G_nodes_intersected)
    print(H_edges_intersected)
    print(H_nodes_intersected)

    G_nodes = [x for x in G_nodes if x[0] not in G_nodes_intersected]
    H_nodes = [x for x in H_nodes if x[0] not in H_nodes_intersected]
    # node = (i, {attr1: val1, attr2: val2})

    for ng in G_nodes:
        for nh in H_nodes:
            if ng[1]['kegg_id'] == nh[1]['kegg_id']:
                G_nodes_intersected.add(ng[0])
                H_nodes_intersected.add(nh[0])
                H_nodes.remove(nh)
                break

    print(G_nodes_intersected)
    print(H_nodes_intersected)

    G_nodes_aux = list(G_nodes_intersected)
    G_nodes_aux.sort()
    print(G_nodes_aux)
    
    # Creating the result graph
    intersection_graph = nx.Graph()
    intersection_graph.add_nodes_from([(new_id, G_graph.graph.nodes[n_id]) for new_id, n_id in enumerate(G_nodes_aux)])
    intersection_graph.add_edges_from([(G_nodes_aux.index(e[0]), G_nodes_aux.index(e[1]), G_graph.graph.edges[e]) for e in G_edges_intersected])
    return intersection_graph

def get_all_pairs(lst):
    pairs = []
    for i, x in enumerate(lst):
        for j, y in enumerate(lst[i + 1:]):
            pairs.append((x, y))
    return pairs


def smi2svg(smi):
    mol = Chem.MolFromSmiles(smi)
    try:
        Chem.rdmolops.Kekulize(mol)
    except:
        pass
    drawer = rdMolDraw2D.MolDraw2DSVG(350, 300)
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return svg


def smi2image(smi):
    # print(smi)
    try:
        svg_string = smi2svg(smi)
    except:
        # In some cases the smiles must be modified when dealing with rings
        smi = smi[0].upper() + smi[1:]
        # print(smi)
        svg_string = smi2svg(smi)
    impath = 'data:image/svg+xml;charset=utf-8,' + parse.quote(svg_string, safe="")
    return impath

def jaccard_similarity(g, h):
    if len(g) == 0 or len(h) == 0:
        return 0
    i = list((Counter(g) & Counter(h)).elements())
    divider = len(g) + len(h) - len(i)
    return round(len(i) / divider,3)

class Geass():
    def __init__(self, mol, bonds, reactants, reactions, filling_atoms, BUL, UCL=None, verbose=0, mol_bonds=None, breakable_mol_bonds=None):
        """

        :param mol: the mol to analyse
        :param bonds: the bonds that can be broken (rdkit bonds)
        :param reactants: the reactants of the reaction rules
        :param filling_atoms: the rules for filling the dummy atoms
        :param BUL: the natural products library/database
        :param verbose: the degree of messages to be output
        :param mol_bonds: the indices of the bonds of the molecule that are going to be broken
        :param breakable_mol_bonds:
        """
        if filling_atoms is None:
            filling_atoms = {}
        if breakable_mol_bonds is None:
            breakable_mol_bonds = {}
        if mol_bonds is None:
            mol_bonds = []

        self.mol = mol

        # Checks that all the reactants have a defined bond to break
        assert len([_ for _ in reactants.keys() if _ in bonds.keys()]) == len(reactants), 'Bonds missing in input'

        self.verbose = verbose
        self.bonds = bonds
        self.mol_bonds = mol_bonds
        self.breakable_mol_bonds = breakable_mol_bonds
        self.reactants = reactants
        self.reactions = reactions
        self.filling_atoms = filling_atoms
        self.graph = nx.Graph()
        self.UCL = UCL
        # Check if BUL is pandas data frame
        if isinstance(BUL, pd.core.frame.DataFrame):
            self.BUL = dict(zip(BUL['ID'], BUL['Molecule']))
        else:
            self.BUL = BUL

    def find_bond_in_reactant(self, reaction):
        reactant_bonds = []
        for bond in self.bonds[reaction]:
            for b in self.reactants[reaction].GetBonds():
                if b.Match(bond):
                    reactant_bonds.append(b)
                    break

        return reactant_bonds

    def bonds_from_pairs(self, pairs):
        self.mol_bonds = []
        for pair in pairs:
            bond_mol = self.mol.GetBondBetweenAtoms(pair[0], pair[1])
            if bond_mol is not None:
                self.mol_bonds.append(bond_mol)

    def find_bond_in_list(self, bond):
        """
        Returns the first match
        """
        for i, b in enumerate(self.mol_bonds):

            lstb = [b.GetBeginAtom().GetSymbol(), b.GetEndAtom().GetSymbol()]
            lsta = [bond.GetBeginAtom().GetSymbol(), bond.GetEndAtom().GetSymbol()]

            flag = True
            for x in lsta:
                if x == '*':
                    continue
                if x not in lstb:
                    flag = False

            if b.Match(bond) and flag:
                return b, i

        return None, None

    def get_bonds_to_break(self):

        start_time = time.time()

        for reaction in self.bonds.keys():
            to_break_bonds = self.find_bond_in_reactant(reaction)
            if len(to_break_bonds) == 0 and self.verbose > 0:
                print(f'WARNING: no breakable bond was found in reaction {reaction}')

            matches = self.mol.GetSubstructMatches(self.reactants[reaction])
            for match in matches:
                atomPairs = get_all_pairs(match)
                self.bonds_from_pairs(atomPairs)

                for b in to_break_bonds:
                    matchedBond, idx = self.find_bond_in_list(b)
                    self.breakable_mol_bonds[matchedBond.GetIdx()] = reaction
                    if self.verbose == 1:
                        print('Adding bond to break: ', matchedBond.GetIdx())
                    del self.mol_bonds[idx]

        if self.verbose > 0:
            print("The identification of the bonds took --- %s seconds ---" % (time.time() - start_time))

    def fill_dummies_mol(self, queryDummy, mol, reaction):
        """

        :param queryDummy: the dummy to be filled
        :param mol: the molecule (fragment) containing the dummy
        :param reaction: the name of the reaction associated to the dummy to fill
        :return: the molecule with the dummy filled and a flag of success
        """
        Chem.SanitizeMol(mol)

        try:
            # Get the atoms indices that matches the dummy
            dummyAtoms = [(atom.GetIdx(), atom) for atom in mol.GetAtoms() if atom.GetSmarts() == queryDummy]
        except:
            return mol, False

        while len(dummyAtoms) > 0:
            # We itereate until all the dummy atoms have been replaced. There may be more than one.
            idx, atom = dummyAtoms[0] # recover the atom and the index

            # IMPORTANT: the molecule must be converted into an Editable mol in order to modify its atoms
            editmol = Chem.rdchem.EditableMol(mol)

            # Get the neighbors of the atom. As it is a dummy atom, coming form a fragmented bond, we expect that
            # it only has one neighbor
            nhs = atom.GetNeighbors()

            if len(nhs) > 1:
                print('WARNING: A dummy atom with mor than 1 neighbor was found.')

            # Get the atomic symbol of the neighbor
            nhSymbol = nhs[0].GetSymbol()

            # We now replace the dummy atom. For that purpose, we query self.filling_atoms with the reaction, and them
            # with the neighbor.

            if self.filling_atoms[reaction][nhSymbol] != '':
                # If the dummy has to be replaced, replace it.
                editmol.ReplaceAtom(idx, Chem.Atom(self.filling_atoms[reaction][nhSymbol]))
                mol = editmol.GetMol()
            else:

                if nhSymbol not in ['C', 'c']:
                    # If the atom must be eliminated, it is replaced by an hydrogen
                    editmol.ReplaceAtom(idx, Chem.Atom('H'))
                else:
                    # If the atom must be eliminated and it is a carbon, remove it without replacement by hydrogen
                    editmol.RemoveAtom(idx)

                # Turn the editable mol into regular molecule
                mol = editmol.GetMol()

            Chem.SanitizeMol(mol)
            # Recompute the dummy atoms and its indices.
            dummyAtoms = [(atom.GetIdx(), atom) for atom in mol.GetAtoms() if atom.GetSmarts() == queryDummy]

        return mol, True

    def create_fragmentation_graph(self):

        if len(self.breakable_mol_bonds) == 0:

            # If no breakable bonds are founded, a graph with a single node is returned

            print('No breakable bonds were found. Impossible to create the fragmentation graph.')
            NP = 'No'
            kegg_id = 'NA'
            for name, structure in self.BUL.items(): # BUL dict
                if len(self.mol.GetSubstructMatch(structure, )) == self.mol.GetNumAtoms():
                    NP = 'Yes'
                    kegg_id = name
                    break
            self.graph.add_nodes_from(
                [(0, {'SMILES': Chem.MolToSmiles(self.mol), 'img': smi2image(Chem.MolToSmiles(self.mol)), 'isNP': NP, 'kegg_id': kegg_id})])
            return None

        start_time = time.time()

        # The next function creates the fragments, adding labels with the index of the bonds that were broken (needed to create edges later on)
        mol_f_dummies = Chem.FragmentOnBonds(self.mol, self.breakable_mol_bonds, addDummies=True,
                                     dummyLabels=[(x, x) for x in self.breakable_mol_bonds])
        # This creates the fragments without the dummies (these will be the nodes)
        mol_f = Chem.FragmentOnBonds(self.mol, self.breakable_mol_bonds, addDummies=False)

        brokenSmiles = Chem.MolToSmiles(mol_f_dummies)  # Turns the fragmented molecule into smiles
        # dummies = re.findall('[0-9]*\*', brokenSmiles)  # Finds all the dummies

        # atoms rules

        # for fragment in fragments:
        #     # Iterate over the fragments
        #     for dummy in dummies:
        #         # Iterate over all possible dummies
        #         smiles = Chem.MolToSmiles(fragment)  # Convert the fragment to smiles
        #         if dummy == '*':
        #             # Considers the special case in which the bondIdx 0 was broken
        #             queryDummy = dummy
        #             bond = 0
        #         else:
        #             queryDummy = '[' + dummy + ']'  # The dummies with a number different than 0 are included in []
        #             bond = int(dummy.split('*')[0])  # Recover the bond index

        #         if queryDummy in smiles:
        #             # If the dummy is in the smiles
        #             reactionName = self.breakable_mol_bonds[bond]  # Recover the reaction associated to this bond
        #             fragment, flag = self.fill_dummies_mol(queryDummy, fragment, reactionName)  # Fills the dummies (*)

        #             if not flag:
        #                 # Raise an error and return the invalid fragment
        #                 print('An invalid fragment appeared during graph computation')
        #                 return fragment

        #     fragments_fill.append(fragment)

        try:
            fragments_fill = list(Chem.GetMolFrags(mol_f, asMols=True))
            fragments_dummies = Chem.GetMolFrags(mol_f_dummies, asMols=True)

        except:
            # Aromatic atoms may pose problems when converting from smiles to mol
            fragments_fill = list(Chem.GetMolFrags(mol_f, asMols=True, sanitizeFrags=False))
            fragments_dummies = Chem.GetMolFrags(mol_f_dummies, asMols=True, sanitizeFrags=False)

        fragment_lib = {i: mol for i, mol in enumerate(fragments_dummies)}  # Indexed dictionary
        
        # Get the reactions involved in the fragmentation
        reactions_involved_names = {v for v in self.breakable_mol_bonds.values()}
        reactions_involved = {r: self.reactions[r] for r in self.breakable_mol_bonds.values() if r in reactions_involved_names}

        full_molecule_list = [self.mol]
        # Get the actual molecules without dummies and with the completed atoms
        for reaction in reactions_involved.values():
            current_reaction_mol_lists = []
            for i, molec in enumerate(full_molecule_list):
                current_reaction_mol_lists.append(run_fragmentation(molec, reaction, []))
            # Flatten the list of lists
            full_molecule_list = [m for mol_sublist in current_reaction_mol_lists for m in mol_sublist]
        
        unindexed_fragments = full_molecule_list.copy()
        unindexed_fragments.sort(key=get_SMILES_length_from_mol)
        # Align the fragments from FragmentOnBonds (without dummies) with the fragments from RunReactants
        for i, frag in enumerate(fragments_fill):
            for j, m in enumerate(unindexed_fragments):
                if len(m.GetSubstructMatch(frag, )) == frag.GetNumAtoms():
                    fragments_fill[i] = m
                    # Remove the matched item from the unindexed fragments list
                    unindexed_fragments.pop(j)
                    break

        self.fragments_filled = fragments_fill
        isNP = []
        for i, molec in enumerate(fragments_fill):
            isNP.append({'isNP': 'No', 'kegg_id': 'NA'})
            for name, structure in self.BUL.items(): # BUL dict
                if len(molec.GetSubstructMatch(structure, )) == molec.GetNumAtoms():
                    isNP[i] = {'isNP': 'Yes', 'kegg_id': name}
                    break
        self.graph.add_nodes_from([(i, {'SMILES': Chem.MolToSmiles(fragments_fill[i]), 'isNP': isNP[i]['isNP'], 'kegg_id': isNP[i]['kegg_id']}) for i in range(len(fragments_fill))])
        
        # The SMILES cannot be the node index because there may be identical fragments
        edges = []
        edges_attr = {}

        nh = {}  # fragment_index : nh (as dummies code)
        for i, fragment in enumerate(fragments_dummies):
            smile = Chem.MolToSmiles(fragment)
            nh_temp = re.findall('[0-9]*\*', smile)
            nh[i] = nh_temp

        for i, lst1 in nh.items():
            for j, lst2 in nh.items():
                if i == j:
                    continue
                for dummy in lst1:
                    if dummy in lst2:
                        if (j, i) not in edges:  # Undirected graph, remove duplicate bonds
                            edges.append((i, j))
                            # Saves the reaction as a bond attribute
                            if dummy == '*':
                                bond = 0
                            else:
                                bond = int(dummy.split('*')[0])

                            reactionName = self.breakable_mol_bonds[bond]
                            edges_attr[(i, j)] = {'reaction': reactionName}

        self.graph.add_edges_from(edges)
        nx.set_edge_attributes(self.graph, edges_attr)
        smiles_attr = {k: {'img': smi2image(Chem.MolToSmiles(mol))} for k, mol in enumerate(fragments_fill)}
        nx.set_node_attributes(self.graph, smiles_attr)

        # Moved this part of the code where the nodes are created for the first time
        # smiles_attr = nx.get_node_attributes(self.graph, 'SMILES')
        # isNP = {}
        # for nodeKey, smiles in smiles_attr.items():
        #     isNP[nodeKey] = {'isNP': 'No', 'kegg_id': 'NA'}
        #     fragment = Chem.MolFromSmiles(smiles)
        #     for name, structure in self.BUL.items(): # BUL dict
        #         if len(fragment.GetSubstructMatch(structure, )) == fragment.GetNumAtoms():
        #             isNP[nodeKey] = {'isNP': 'Yes', 'kegg_id': name}
        #             break

        # nx.set_node_attributes(self.graph, isNP)

        if self.verbose > 0:
            print("The generation of the fragmentation graph took --- %s seconds ---" % (time.time() - start_time))

    def create_fragmentation_graph_old(self):

        if len(self.breakable_mol_bonds) == 0:

            # If no breakable bonds are founded, a graph with a single node is returned

            print('No breakable bonds were found. Impossible to create the fragmentation graph.')
            NP = 'No'
            kegg_id = 'NA'
            for name, structure in self.BUL.items(): # BUL dict
                if len(self.mol.GetSubstructMatch(structure, )) == self.mol.GetNumAtoms():
                    NP = 'Yes'
                    kegg_id = name
                    break
            if NP == 'No' and self.UCL is not None:
                unknown_mol = self.mol
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
                kegg_id = new_name
            self.graph.add_nodes_from(
                [(0, {'SMILES': Chem.MolToSmiles(self.mol), 'img': smi2image(Chem.MolToSmiles(self.mol)), 'isNP': NP, 'kegg_id': kegg_id})])
            return None

        start_time = time.time()

        # The next function creates the fragments, adding labels with the index of the bonds that were broken (needed to create edges later on)
        mol_f_dummies = Chem.FragmentOnBonds(self.mol, self.breakable_mol_bonds, addDummies=True,
                                     dummyLabels=[(x, x) for x in self.breakable_mol_bonds])
        # This creates the fragments without the dummies (these will be the nodes)
        # mol_f = Chem.FragmentOnBonds(self.mol, self.breakable_mol_bonds, addDummies=False)

        brokenSmiles = Chem.MolToSmiles(mol_f_dummies)  # Turns the fragmented molecule into smiles
        dummies = re.findall('[0-9]*\*', brokenSmiles)  # Finds all the dummies

        # atoms rules

        fragments = Chem.GetMolFrags(mol_f_dummies, asMols=True)

        fragments_fill = []
        for fragment in fragments:
            # Iterate over the fragments
            for dummy in dummies:
                # Iterate over all possible dummies
                smiles = Chem.MolToSmiles(fragment)  # Convert the fragment to smiles
                if dummy == '*':
                    # Considers the special case in which the bondIdx 0 was broken
                    queryDummy = dummy
                    bond = 0
                else:
                    queryDummy = '[' + dummy + ']'  # The dummies with a number different than 0 are included in []
                    bond = int(dummy.split('*')[0])  # Recover the bond index

                if queryDummy in smiles:
                    # If the dummy is in the smiles
                    reactionName = self.breakable_mol_bonds[bond]  # Recover the reaction associated to this bond
                    fragment, flag = self.fill_dummies_mol(queryDummy, fragment, reactionName)  # Fills the dummies (*)

                    if not flag:
                        # Raise an error and return the invalid fragment
                        print('An invalid fragment appeared during graph computation')
                        return fragment

            fragments_fill.append(fragment)

        try:
            # fragments_fill = list(Chem.GetMolFrags(mol_f, asMols=True))
            fragments_dummies = Chem.GetMolFrags(mol_f_dummies, asMols=True)

        except:
            # Aromatic atoms may pose problems when converting from smiles to mol
            # fragments_fill = list(Chem.GetMolFrags(mol_f, asMols=True, sanitizeFrags=False))
            fragments_dummies = Chem.GetMolFrags(mol_f_dummies, asMols=True, sanitizeFrags=False)

        fragment_lib = {i: mol for i, mol in enumerate(fragments_dummies)}  # Indexed dictionary
        
        # Get the reactions involved in the fragmentation
        # reactions_involved_names = {v for v in self.breakable_mol_bonds.values()}
        # reactions_involved = {r: self.reactions[r] for r in self.breakable_mol_bonds.values() if r in reactions_involved_names}

        # full_molecule_list = [self.mol]
        # Get the actual molecules without dummies and with the completed atoms
        # for reaction in reactions_involved.values():
        #     current_reaction_mol_lists = []
        #     for i, molec in enumerate(full_molecule_list):
        #         current_reaction_mol_lists.append(run_fragmentation(molec, reaction, []))
        #     # Flatten the list of lists
        #     full_molecule_list = [m for mol_sublist in current_reaction_mol_lists for m in mol_sublist]
        
        # unindexed_fragments = full_molecule_list.copy()
        # # Align the fragments from FragmentOnBonds (without dummies) with the fragments from RunReactants
        # for i, frag in enumerate(fragments_fill):
        #     for j, m in enumerate(unindexed_fragments):
        #         if len(m.GetSubstructMatch(frag, )) == frag.GetNumAtoms():
        #             fragments_fill[i] = m
        #             # Remove the matched item from the unindexed fragments list
        #             unindexed_fragments.pop(j)
        #             break

        self.fragments_filled = fragments_fill
        # isNP = []
        # for i, molec in enumerate(fragments_fill):
        #     isNP.append({'isNP': 'No', 'kegg_id': 'NA'})
        #     for name, structure in self.BUL.items(): # BUL dict
        #         if len(molec.GetSubstructMatch(structure, )) == molec.GetNumAtoms():
        #             isNP[i] = {'isNP': 'Yes', 'kegg_id': name}
        #             break
        self.graph.add_nodes_from([(i, {'SMILES': Chem.MolToSmiles(fragments_fill[i])}) for i in range(len(fragments_fill))])
        
        # The SMILES cannot be the node index because there may be identical fragments
        edges = []
        edges_attr = {}

        nh = {}  # fragment_index : nh (as dummies code)
        for i, fragment in enumerate(fragments_dummies):
            smile = Chem.MolToSmiles(fragment)
            nh_temp = re.findall('[0-9]*\*', smile)
            nh[i] = nh_temp

        for i, lst1 in nh.items():
            for j, lst2 in nh.items():
                if i == j:
                    continue
                for dummy in lst1:
                    if dummy in lst2:
                        if (j, i) not in edges:  # Undirected graph, remove duplicate bonds
                            edges.append((i, j))
                            # Saves the reaction as a bond attribute
                            if dummy == '*':
                                bond = 0
                            else:
                                bond = int(dummy.split('*')[0])

                            reactionName = self.breakable_mol_bonds[bond]
                            edges_attr[(i, j)] = {'reaction': reactionName}

        self.graph.add_edges_from(edges)
        nx.set_edge_attributes(self.graph, edges_attr)
        smiles_attr = {k: {'img': smi2image(Chem.MolToSmiles(mol))} for k, mol in enumerate(fragments_fill)}
        nx.set_node_attributes(self.graph, smiles_attr)

        # Moved this part of the code where the nodes are created for the first time
        smiles_attr = nx.get_node_attributes(self.graph, 'SMILES')
        isNP = {}
        for nodeKey, smiles in smiles_attr.items():
            isNP[nodeKey] = {'isNP': 'No', 'kegg_id': 'NA'}
            fragment = Chem.MolFromSmiles(smiles)
            for name, structure in self.BUL.items(): # BUL dict
                if len(fragment.GetSubstructMatch(structure, )) == fragment.GetNumAtoms():
                    isNP[nodeKey] = {'isNP': 'Yes', 'kegg_id': name}
                    break

        nx.set_node_attributes(self.graph, isNP)

        if self.verbose > 0:
            print("The generation of the fragmentation graph took --- %s seconds ---" % (time.time() - start_time))

    def plot_graph(self):
        undirected = ipycytoscape.CytoscapeWidget()
        undirected.layout.height = '800px'
        undirected.layout.width = '970px'
        undirected.graph.add_graph_from_networkx(self.graph, directed=False)

        undirected.set_layout(name='circle', nodeSpacing=200, edgeLengthVal=100)
        undirected.set_style([{'selector': "node[isNP = 'Yes']", 'css': {
            'content': 'data(count)',
            'text-valign': 'top',
            'color': 'white',
            'text-outline-width': 2,
            'text-outline-color': '#FFFFFF',
            'shape': 'rectangle',
            'width': 400,
            'height': 350,
            'border-width': 20,
            'border-color': '#08d13e',
            'background-color': '#FFFFFF',
            'background-fit': 'contain',
            'background-image': 'data(img)'}},

                              {'selector': "node[isNP = 'No']", 'css': {
                                  'content': 'data(count)',
                                  'text-valign': 'top',
                                  'color': 'white',
                                  'text-outline-width': 2,
                                  'curve-style': 'straight',
                                  'text-outline-color': '#FFFFFF',
                                  'shape': 'rectangle',
                                  'width': 400,
                                  'height': 350,
                                  'border-width': 20,
                                  'border-color': '#d10808',
                                  'background-color': '#FFFFFF',
                                  'background-fit': 'contain',
                                  'background-image': 'data(img)'}},
                              {'selector': "edge", 'css': {
                                  "curve-style": "bezier",
                                  'width': 20,
                                  'label': 'data(reaction)',
                                  'line-color': '#08d13e', }},
                              {'selector': "edge[reaction='unknown']", 'css': {
                                  'line-color': '#d10808', }},
                                  ])

        return undirected

    def get_all_NA_neighbours_names(self, node_id):
        names = []
        # Iterate over the edges of the selecetd node
        for edge in list(self.graph.edges(node_id, data=True)):
            # Only include the name of the neighbour if the reaction is unknown (notNP)
            if edge[2]['reaction'] == 'unknown':
                # Check which one of the relation members is not the selected node
                if edge[0] == node_id:
                    current_neighbour = self.graph.nodes[edge[1]]
                else:
                    current_neighbour = self.graph.nodes[edge[0]]
                # Add the name to the name list
                names.append(current_neighbour['kegg_id'][0] if type(current_neighbour['kegg_id']) == tuple else current_neighbour['kegg_id'])
        return '+'.join(names)
