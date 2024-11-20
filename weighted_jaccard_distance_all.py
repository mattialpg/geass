import pandas as pd
import numpy as np
from itertools import product

# Step 1: Load Data

# 1.1 Load the edges data from SECRETed_edges_all.csv
edges_df = pd.read_csv('SECRETed_edges_all.csv', sep=',', dtype={'ID': str, 'n1': str, 'n2': str, 'ReactionType': str})

# 1.2 Load node weights from node_molecular_weights.csv
node_weights_df = pd.read_csv('node_molecular_weights.csv', sep=',', dtype={'NodeID': str, 'Weight': float})

# 1.3 Load ID weights from SECRETed_molecular_weights.csv
id_weights_df = pd.read_csv('SECRETed_molecular_weights.csv', sep=',', dtype={'ID': str, 'Weight': float})

# 1.4 Load node similarities from UCL_similarity_matrix_morgan.csv (square matrix format)
node_sim_matrix_df = pd.read_csv('UCL_similarity_matrix_morgan.csv', sep=',', index_col=0)

# Step 2: Prepare Data Structures

# Create dictionaries for node weights and ID weights
node_weight_dict = node_weights_df.set_index('NodeID')['Weight'].to_dict()
id_weight_dict = id_weights_df.set_index('ID')['Weight'].to_dict()

# Convert node similarity matrix DataFrame to a dictionary for quick lookup
node_similarity = node_sim_matrix_df.to_dict()

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

for idx, row in edges_df.iterrows():
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
distance_matrix.to_csv('adjusted_weighted_jaccard_distance_matrix_combined.csv')
