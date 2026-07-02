
from rdkit import Chem
from urllib import parse
from copy import deepcopy
import re

def flatten(list):
    flat_list = [item for sublist in list for item in sublist]
    return(flat_list)

def remove_isotopes(mol):
    for atom in mol.GetAtoms():
        atom.SetIsotope(0)
    return mol


from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
normalizer = rdMolStandardize.Normalizer()
reionizer = rdMolStandardize.Reionizer()
uncharger = rdMolStandardize.Uncharger()

def _neutralize_atoms(mol):
    """ Based on: https://baoilleach.blogspot.com/2019/12/no-charge-simple-approach-to.html"""

    # Neutralise common +1 centres 
    if any(a.GetFormalCharge() != 0 for a in mol.GetAtoms()):
        pattern = Chem.MolFromSmarts("[+1!H0!$([-1,-2,-3,-4]),-1!$([+1,+2,+3,+4])]")
        matches = [x[0] for x in mol.GetSubstructMatches(pattern)]
        for match in matches:
            atom = mol.GetAtomWithIdx(match)
            fc = atom.GetFormalCharge()
            atom.SetNumExplicitHs(atom.GetTotalNumHs() - fc)
            atom.SetFormalCharge(0)
            atom.UpdatePropertyCache()

    # # Quench radicals
    # smiles = Chem.MolToSmiles(mol)
    # smiles = re.sub(R"\[[CcNnO]\d?\+?\]", lambda m: m.group(0)[1], smiles)
    # return Chem.MolFromSmiles(smiles)
    return mol

def standardize_mol(mol):
    """
    Prepare and sanitise a molecule for fragmentation.
    Keeps the largest fragment and neutralises charges if present.
    """
    try:
        # Keep largest fragment (removes salts / small components)
        cmol = rdMolStandardize.Cleanup(mol)
        
        if len(Chem.GetMolFrags(cmol)) > 1:
            cmol = rdMolStandardize.FragmentParent(cmol)

        # Neutralise if formal charges are present
        if any(a.GetFormalCharge() != 0 for a in cmol.GetAtoms()):
            cmol = _neutralize_atoms(cmol)

        return cmol
    except Exception as exc:
        print(exc)
        return mol


from ipycytoscape import CytoscapeWidget
def create_compound_edges(result_graphs):
    compound_edges = {}
    for shared_name, rg in result_graphs.items():
        current_edges = []
        #print(shared_name)
        for edge in list(rg.edges(data=True)):
            current_kegg_id_0 = rg.nodes[edge[0]]['MU_ID']
            current_kegg_id_1 = rg.nodes[edge[1]]['MU_ID']
            if current_kegg_id_0 == 'NA':
                edge0name = get_all_NA_neighbours_names(rg, edge[0])
            else:
                edge0name = current_kegg_id_0[0] if type(current_kegg_id_0) == tuple else current_kegg_id_0
            if current_kegg_id_1 == 'NA':
                edge1name = get_all_NA_neighbours_names(rg, edge[1])
            else:
                edge1name = current_kegg_id_1[0] if type(current_kegg_id_1) == tuple else current_kegg_id_1
            current_edges.append((edge0name, edge1name, edge[2]['reaction']))
        compound_edges[shared_name] = tuple(current_edges)
    return compound_edges

def get_all_NA_neighbours_names(geass_obj, node_id):
    names = []
    # Iterate over the edges of the selecetd node
    for edge in list(geass_obj.G.edges(node_id, data=True)):
        # Only include the name of the neighbour if the reaction is unknown (notNP)
        if edge[2]['reaction'] == 'unknown':
            # Check which one of the relation members is not the selected node
            if edge[0] == node_id:
                current_neighbour = geass_obj.G.nodes[edge[1]]
            else:
                current_neighbour = geass_obj.G.nodes[edge[0]]
            # Add the name to the name list
            names.append(current_neighbour['BU_ID'][0] if type(current_neighbour['BU_ID']) == tuple else current_neighbour['BU_ID'])
    return '+'.join(names)

def show_connection_graph(graph):
    cyto = CytoscapeWidget()
    cyto.layout.height = '500px'
    cyto.layout.width = '1400px'
    cyto.graph.add_graph_from_networkx(graph)
    
    cyto.set_layout(name='cose')

    cyto.set_style([

        # Base node style: hide all text completely
        {
            'selector': 'node',
            'css': {
                # 'label': 'data(MU_ID)',                     # prevent fallback labels
                # 'content': 'data(MU_ID)',                   # no text by default
                # 'text-opacity': 1,               # ensure invisible
                # 'font-family': 'helvetica',
                # 'font-size': '10px',
                'width': 70,
                'height': 70,
                'border-width': 1.5,
                'background-color': '#FFFFFF',
                'background-width': '90%',
                'background-height': '90%',
                # 'background-image': 'data(img)',
            }
        },

    #     # Edge style
    #     {
    #         'selector': 'edge',
    #         'css': {
    #             'curve-style': 'bezier',
    #             'width': 1.5,
    #             'label': 'data(reaction)',
    #             'font-family': 'helvetica',
    #             'font-size': '10px',
    #             'line-color': '#009b73',
    #             'padding': '50px'
    #         }
    #     },
    #     {
    #         'selector': "edge[reaction='']",
    #         'css': {'line-color': '#d10808'}
    #     },
    #     {
    #         'selector': "edge[intersection='No']",
    #         'css': {
    #             'line-style': 'dashed',
    #             'line-dash-pattern': [3, 2],
    #             'width': 0.5
    #         }
    #     },
    #     {
    #         'selector': "edge[origin_graph=1]",
    #         'css': {'line-color': '#1976d2'}
    #     },
    #     {
    #         'selector': "edge[origin_graph=2]",
    #         'css': {'line-color': '#ef6c00'} 
    #     }
    ])


    return cyto
