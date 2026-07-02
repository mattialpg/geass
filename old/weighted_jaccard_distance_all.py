import pandas as pd
import numpy as np
from itertools import product
import pickle
import utils
from rdkit import Chem
from tqdm import tqdm

# Step 1: Load Data
df_fragments = pd.read_csv('NPA_analysis/NPA_fragments.csv', header=None,
    names=['NPA_ID', 'ISOSMILES', 'Frag_ISOSMILES', 'Frag_IDs'])
df_fragments['Frag_ISOSMILES'] = (df_fragments['Frag_ISOSMILES'].str.strip('[]').str.split(','))
df_fragments['Frag_IDs'] = (df_fragments['Frag_IDs'].str.strip('[]').str.split(','))

# 1.1 Get edges data
npa_graphs = pickle.load(open('NPA_analysis/NPA_graphs.pkl', 'rb'))

npa_edges = utils.create_compound_edges(npa_graphs)
rows = [(cid, n1, n2, reaction)
        for cid, edges in npa_edges.items()
        for n1, n2, reaction in edges]
df_edges = pd.DataFrame(rows, columns=["ID", "n1", "n2", "ReactionType"])

# 1.2 Load node weights from node_molecular_weights.csv
from rdkit.Chem.Descriptors import MolWt

node_weight_dict = {}
id_weight_dict = {}
for cid, g in tqdm(npa_graphs.items()):
    smiles = df_fragments.loc[df_fragments.NPA_ID == cid, "ISOSMILES"].iat[0]
    id_weight_dict[cid] = MolWt(Chem.MolFromSmiles(smiles))

    for node_id, data in g.nodes(data=True):
        node_smi = data["SMILES"]
        node_weight_dict[f"{cid}_{node_id}"] = MolWt(Chem.MolFromSmiles(node_smi))

df_node_weights = pd.DataFrame.from_dict(node_weight_dict,
    orient='index', columns=['Weight'])
df_node_weights.index.name = 'NodeID'
df_node_weights.reset_index(inplace=True)

df_id_weights = pd.DataFrame.from_dict(id_weight_dict,
    orient='index', columns=['Weight'])
df_id_weights.index.name = 'ID'
df_id_weights.reset_index(inplace=True)

#-------- Calculate similarities ----------#
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import TanimotoSimilarity
from rdkit.Chem import MACCSkeys

# Collect fragment SMILES directly
node_ids = []
node_smiles = []

npa_graphs = pickle.load(open('NPA_analysis/NPA_graphs.pkl', 'rb'))
for cid, g in npa_graphs.items():
    for node_id, attrs in g.nodes(data=True):
        node_ids.append(f"{cid}_{node_id}")
        node_smiles.append(attrs["SMILES"])

# Compute Morgan fingerprints (radius=2, nBits=2048)
fps = [MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(smi)) for smi in node_smiles]

N = len(fps)
sim_matrix = np.zeros((N, N), float)

# Compute pairwise similarities
for i in range(N):
    fp_i = fps[i]
    for j in range(i, N):
        sim = TanimotoSimilarity(fp_i, fps[j])
        sim_matrix[i, j] = sim
        sim_matrix[j, i] = sim

# Save similarity matrix to CSV
df_node_sim_matrix = pd.DataFrame(sim_matrix, index=node_ids, columns=node_ids)


# Step 2: Prepare Data Structures
# Create dictionaries for node weights and ID weights
node_weight_dict = df_node_weights.set_index('NodeID')['Weight'].to_dict()
id_weight_dict = df_id_weights.set_index('ID')['Weight'].to_dict()

# Convert node similarity matrix DataFrame to a dictionary for quick lookup
node_similarity = df_node_sim_matrix.to_dict()

# Function to get node similarity
def get_node_similarity(n1, n2):
    try:
        return node_similarity[n1][n2]
    except KeyError:
        return 0.0  # Default similarity if nodes not found

# Step 3: Build Graphs per ID with Adjusted Weights

# Function to get adjusted node weight
def get_adjusted_node_weight(node_id, graph_id):
    node_weight = node_weight_dict.get(node_id, 1.0)  # Default to 1.0 if not found
    id_weight = id_weight_dict.get(graph_id, 1.0)     # Default to 1.0 if not found
    adjusted_weight = node_weight / id_weight if id_weight != 0 else 0.0
    return adjusted_weight

# Create a dictionary to hold the edges for each graph with adjusted weights
graph_dict = {}

for idx, row in df_edges.iterrows():
    graph_id = row['ID']
    n1 = str(row['n1'])
    n2 = str(row['n2'])
    
    # Get adjusted node weights
    adjusted_node_weight_n1 = get_adjusted_node_weight(n1, graph_id)
    adjusted_node_weight_n2 = get_adjusted_node_weight(n2, graph_id)
    
    # Compute adjusted edge weight
    edge_weight = 1.0  # Base edge weight
    adjusted_edge_weight = edge_weight * adjusted_node_weight_n1 * adjusted_node_weight_n2
    
    # Initialize graph if not present
    if graph_id not in graph_dict:
        graph_dict[graph_id] = {}
    
    # Represent edge as a sorted tuple to avoid duplicates
    edge = tuple(sorted((n1, n2)))
    
    # If edge already exists, sum the weights
    if edge in graph_dict[graph_id]:
        graph_dict[graph_id][edge] += adjusted_edge_weight
    else:
        graph_dict[graph_id][edge] = adjusted_edge_weight

# Step 4: Compute Adjusted Weighted Jaccard Distance Between Graphs

# Get a list of all graph IDs
graph_ids = list(graph_dict.keys())

# Initialize a DataFrame to hold the distances
distance_matrix = pd.DataFrame(index=graph_ids, columns=graph_ids, dtype=float)

# Function to compute adjusted weighted Jaccard distance between two graphs
def adjusted_weighted_jaccard(graph_a, graph_b, threshold=0.8):
    numerator = 0.0
    denominator = 0.0

    # Build sets of edges
    edges_a = set(graph_a.keys())
    edges_b = set(graph_b.keys())
    all_edges = edges_a.union(edges_b)

    # For efficient lookup
    graph_a_weights = graph_a
    graph_b_weights = graph_b

    for edge in all_edges:
        w_a = graph_a_weights.get(edge, 0.0)
        w_b = graph_b_weights.get(edge, 0.0)
        # Edge similarity
        if edge in edges_a and edge in edges_b:
            S_e = 1.0  # Edge exists in both graphs with identical nodes
        else:
            # Check for similar edges based on node similarities
            n1_a, n2_a = edge
            similar_edges_in_b = []
            for edge_b in edges_b:
                n1_b, n2_b = edge_b
                # Compute node similarities
                S_n1 = get_node_similarity(n1_a, n1_b)
                S_n2 = get_node_similarity(n2_a, n2_b)
                S_e_candidate = (S_n1 + S_n2) / 2.0
                if S_e_candidate >= threshold:
                    similar_edges_in_b.append((edge_b, S_e_candidate))
            if similar_edges_in_b:
                # Take the maximum similarity
                max_similar_edge, max_S_e = max(similar_edges_in_b, key=lambda x: x[1])
                S_e = max_S_e
                # Update w_b to use the weight of the most similar edge
                w_b = graph_b_weights.get(max_similar_edge, 0.0)
            else:
                S_e = 0.0
        numerator += min(w_a, w_b) * S_e
        denominator += max(w_a, w_b)
    if denominator == 0.0:
        return 1.0  # Define distance as maximum when denominator is zero
    similarity = numerator / denominator
    distance = 1.0 - similarity
    return distance

# Compute the adjusted weighted Jaccard distance for each pair of graphs
for i in range(len(graph_ids)):
    for j in range(i, len(graph_ids)):
        id_a = graph_ids[i]
        id_b = graph_ids[j]
        graph_a = graph_dict[id_a]
        graph_b = graph_dict[id_b]
        distance = adjusted_weighted_jaccard(graph_a, graph_b)
        distance_matrix.loc[id_a, id_b] = distance
        distance_matrix.loc[id_b, id_a] = distance  # Symmetric matrix

# Step 5: Output the Distance Matrix

# Display the distance matrix
print(distance_matrix)

# Save the distance matrix to a CSV file
distance_matrix.to_csv('NPA_analysis/adj_jaccard_distance_matrix_combined.csv')
