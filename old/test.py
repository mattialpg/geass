import time
import geass
import metadisassembler5 as medi5
from rdkit import Chem
from IPython.display import clear_output

def create_graphs(mols, indexer_original_mols, PBU, BUL, brokenBondsDefs, reactants, fillingAtoms, BULnames, output_folder="./metadisassembler_v3_results"):
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
        # Initialize Geass
        fragmentator = geass.Geass(current_mol, brokenBondsDefs, reactants, fillingAtoms, BULnames, verbose=0)

        # Get bonds to break
        fragmentator.get_bonds_to_break()
        
        # Create graph
        fragmentator.create_fragmentation_graph()
        
        # Get not NP fragments
        graph_nodes = list(fragmentator.graph.nodes(data=True))
        not_NP_fragments = {frag[0]: frag[1]['SMILES'] for frag in graph_nodes if frag[1]['isNP'] == 'No'}
        
        fragments = {}
        fragment_names = {}
        
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
                disassembler.disassemble() # disassemble
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
        if fragments:
            if 'Error' not in fragments.values():
                # Add new nodes and edges
                for not_NP_index in not_NP_fragments:
                    # Creating new nodes list
                    previous_graph_length = len(fragmentator.graph.nodes)
                    new_nodes = [(i + previous_graph_length, {'SMILES': Chem.MolToSmiles(frag_mol.mol), 'img': geass.smi2image(Chem.MolToSmiles(frag_mol.mol)), 'isNP': 'Yes', 'kegg_id': fragment_names[not_NP_index][i][0]}) for i, frag_mol in enumerate(fragments[not_NP_index])]
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
                    not_fragmented_not_NP_OK.add(indexer_original_mols[mol_idx])
                else:
                    fragmented_not_NP_OK.add(indexer_original_mols[mol_idx])
            else:
                # Add compound to corresponding no NP ERROR analytics list
                if len(graph_nodes) == 1:
                    not_fragmented_not_NP_ERROR.add(indexer_original_mols[mol_idx])
                else:
                    fragmented_not_NP_ERROR.add(indexer_original_mols[mol_idx])
        else:
            # Add compound to corresponding NP analytics list
            if len(graph_nodes) == 1:
                not_fragmented_all_NP.add(indexer_original_mols[mol_idx])
            else:
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
            #templist.append(geass.jaccard_similarity(sent_i, sent_j))
            if i == j:
                templist.append(1)
            else:
                templist.append(geass.jaccard_similarity(sent_i, sent_j))
        similarity_matrix.append(templist)
    similarity_df = pd.DataFrame(similarity_matrix, index=list(compound_edges.keys()), columns=list(compound_edges.keys()))
    print(f"Elapsed: {round(time.time() - start, 3)}s")


def get_graph_intersection(G_graph, H_graph):
    G_nodes = [n for n in G_graph.graph.nodes(data=True)]
    H_nodes = [n for n in H_graph.graph.nodes(data=True)]
    G_edges = [e for e in G_graph.graph.edges(data=True)]
    H_edges = [e for e in H_graph.graph.edges(data=True)]

    # asdf = [(G_graph.graph.nodes[e[0]], G_graph.graph.nodes[e[1]], e[2]) for e in G_graph.graph.edges(data=True)]
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
