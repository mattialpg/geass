# LOAD THE DIFFERENT MODULES

import numpy as np
import pandas as pd
import pickle
import joblib
import warnings
import geass
#import tmap as tm
from rdkit import Chem
from rdkit.Chem import rdFMCS
from sklearn.cluster import AgglomerativeClustering, OPTICS
from sklearn.metrics import pairwise_distances
from scipy.cluster.hierarchy import dendrogram
#from map4 import MAP4Calculator
warnings.filterwarnings('ignore')

def compute_distance_matrix(df, path = None, metric='jaccard'):
    
    '''
    Computes and optionally saves in the provided path
    the similarity/distance matrix of a given binary dataFrame.
    The metric by default is jaccard.
    '''
    
    matrix = pairwise_distances(np.array(df), metric=metric)
    
    if path != None:
        with open(path, 'wb') as fp:
            pickle.dump(matrix, fp)
    
    return matrix

def structure_clustering(threshold, linkage, metric='jaccard', distance_matrix = None, df=None):
    
    '''
    Implements aglomerative hierarchical clustering.
    
    - df: the training dataset
    - threshold: The linkage distance threshold above which, clusters will not be merged.
    - linkage: 'simple', 'average', 'complete' or 'ward'. If ward is employed, the metric
    must be changed to 'euclidean'.
    - distance_matrix: a precomputed distance matrix is prefered to decrease times of computation.
    
    Returns the sklearn model
    '''
    if df is None and distance_matrix is None:
        print('Nor distance matrix either dataset was passed as argument')
        return False
    
    
    if distance_matrix is None:
        distance_matrix = compute_distance_matrix(df)
    
    if linkage == 'ward' and metric != 'euclidean':
        print('The linkage ward can be only employed with euclidean similarity')
    elif linkage == 'ward' and metric == 'euclidean':
        model = AgglomerativeClustering(n_clusters = None, linkage = linkage, distance_threshold = threshold, affinity = 'euclidean').fit(df)
    else:       
        model = AgglomerativeClustering(n_clusters = None,linkage = linkage, distance_threshold = threshold, affinity = 'precomputed').fit(distance_matrix)
    
    return model

def save(trained_model, path):
    '''
    Saves the hierarchical clustering model.
    
    - trained_model: is an output of structure_clustering function
    - path: string containing the directory to be saved. The filename
    must be in format .joblib.
    '''
    
    joblib.dump(trained_model, path)

def get_clusters(df_smiles, trained_model):
    
    '''
    From a trained model and the dataset it obstains a 
    dictionary where the keys are the cluster number
    and the values are the lists of the ID belonging to
    that cluster
    '''
    
    clusters = {}
    n_clusters = get_nclusters(trained_model)
    labels = trained_model.labels_
    df_aux = df_smiles.copy()
    df_aux['label'] = labels
    for cluster in list(range(0, n_clusters)):
        
        clusters[cluster] = list(df_aux[df_aux['label'] == cluster].index)
        
    return clusters

def get_clusters_v2(df, trained_model):
    
    '''
    From a trained model and the dataset it obstains a 
    dictionary where the keys are the cluster number
    and the values are the lists of the indexes belonging to
    that cluster
    '''
    
    clusters = {}
    n_clusters = get_nclusters(trained_model)
    labels = trained_model.labels_
    df_new = df.copy()
    df_new = df_new.reset_index()
    df_new['label'] = labels
    for cluster in list(range(0, n_clusters)):
        
        clusters[cluster] = list(df_new[df_new['label'] == cluster].index)
        
    return clusters

def get_substructures_similarities_map4(df_smiles, distances_sub_lst_complete, leaves, children_new, distance_matrix, distances, linkage='complete'):
    
        '''
        Obtains a list of list in which echa least correspond to a dendrogram_distance
        elements and contains the substructures of all the clusters obtained with this distance.
        '''
        sim_lst_lst_lst = []
        n_clusters_lst = []
        n_samples = len(distances) + 1
        j=0
        for distance in distances:
            print('PROGRESS: ' + str(j/len(distances) * 100))
            trained_model = structure_clustering(distance, linkage, distance_matrix = distance_matrix)
            clusters = get_clusters(df_smiles, trained_model)
            n_clusters = get_nclusters(trained_model)
            
            n_clusters_lst.append(n_clusters)
            
            sim_lst_lst = []
        
            for key, value in clusters.items():
                flag = True
                
                sim_lst = []
                if len(value) == 1:
                    sim_lst.append(0.)
                    flag=False
                else:
                    for key_2, value_2 in leaves.items():
                        if set(value) == set(value_2):
                            flag = False
                            key_objective = key_2
                            i = np.argwhere(children_new[:, 2] == key_objective)[0][0]
                            sim_lst = distances_sub_lst_complete[int(key_objective) - n_samples]
                            #arr = np.array([df_MCS.loc[int(key_objective)].values, df.iloc[index].values])
                            #sim_lst.append(pairwise_distances(arr, metric = metric)[0, 1])
                            
                            
                if flag:
                    print('ERROR: Cluster ' + str(key) + ' not identified')
                sim_lst_lst.append(sim_lst)
                
            sim_lst_lst_lst.append(sim_lst_lst)

            j+=1
                     
        return sim_lst_lst_lst, n_clusters_lst

def get_cluster_size(trained_model):
    
    '''
    Returns the percentage of the total number of
    examples that is contained in each cluster
    '''
    
    labels = trained_model.labels_
    total = len(labels)
    clusters, count = np.unique(labels, return_counts = True)
    
    size = {}
    for i, cluster in enumerate(clusters):
        size[cluster] = count[i]/total*100
        
    return size

def get_nclusters(trained_model):
    
    '''
    Returns the number of clusters
    '''
    
    return trained_model.n_clusters_

def draw_cluster(df_smiles, trained_model, cluster, clusters):
    
    '''
    Returns the fraction of the sdf dataset that is contained 
    in the querying cluster
    '''
    
    lst = clusters[cluster]
    cluster_df = pd.DataFrame()
    df_aux = df_smiles.copy()
    for element in lst:
        cluster_df = cluster_df.append(df_aux.iloc[element])
    
    return cluster_df

def leaves_search(third, children_new, leaves_lst, n_samples):
    
    first = children_new[np.argwhere(children_new[:, 2] == third)[0][0], 0]
    second = children_new[np.argwhere(children_new[:, 2] == third)[0][0], 1]
    if first >= n_samples:
        leaves_search(first, children_new, leaves_lst, n_samples)
    else:
        leaves_lst.append(int(first))
        
    if second >= n_samples:
        leaves_search(second, children_new, leaves_lst, n_samples)
    else:
        leaves_lst.append(int(second))
        
    return leaves_lst

def get_leaves(children):
    
    n_samples = children.shape[0] + 1
    children_new = np.zeros((children.shape[0],children.shape[1]+1))
    children_new[:,:-1] = children
    
    for index, row in enumerate(children_new):
        row[2] =  n_samples + index
        children_new[index] = row
    
    leaves = {}
    third = children_new[0]
    first = children_new[0][0]
    second = children_new[0][1]
       
    for index, row in enumerate(children_new):
        leaves[int(row[2])] = []
        third = int(row[2])
        
        leaves_lst = []
        leaves_lst = leaves_search(third, children_new, leaves_lst, n_samples)
        leaves[int(row[2])] = leaves_lst
        
                
    return leaves, children_new

def get_substructures(df, threshold, distance_matrix, dendrogram_distances, linkage='complete'):
    model = structure_clustering(df, threshold, linkage, distance_matrix = distance_matrix)
    n_clusters = get_nclusters(model)
    for i, distance in enumerate(dendrogram_distances):
        if threshold <= distance:
            index = i
            break
    
    return n_clusters, index

def get_distances(trained_model):
    '''
    Returns the distances at which a change in the number of clusters
    is found
    '''
    
    return trained_model.distances_

def get_substructure_list(df, leaves, rings = False, seeds = None):
    
    substructures_smarts = []
    substructures_molecules = []
    substructures_cancelations = []
    df_aux = df.copy()

    i=0
    for key, value in leaves.items():
        print(f'Progress:  [{i}/{len(leaves.keys())}], leave key: {key}', end='\r')
        
        # TODO: INSTEAD OF MOLS, GRAPHS
        graphs = []
        for element in value:
            graph = df_aux.iloc[element]['Molecule']
            graphs.append(graph)

        # SEEDS NOT NEEDED
        if seeds is not None:
            seed = seeds[i]
            # CHANGE get_substructure_v3 TO STACKOVERFLOW GRAPH ALGORITHM
            smarts, molecule, cancelation = get_substructure_v3(graphs, rings=rings, seed = seed)
        else:
            smarts, molecule, cancelation = get_substructure_v3(graphs, rings=rings)
        
        # RESULT IS COMMON GRAPH OR CANCELATION (NO COMMON GRAPH)
        substructures_smarts.append(smarts)
        substructures_molecules.append(molecule)
        substructures_cancelations.append(cancelation)
        i+=1
    
    return substructures_smarts, substructures_molecules, substructures_cancelations

def get_max_common_graph(graphs):
    pass

def get_substructure_v3(cluster_mols, rings = False, seed = None):
    
    if len(cluster_mols) == 1:
        cluster_mols.append(cluster_mols[0])
    
    valences = True
    #if seed is None or len(seed)>250:
    if seed is None:
        substructure = rdFMCS.FindMCS(cluster_mols, timeout=1, ringMatchesRingOnly = rings, completeRingsOnly = rings, matchValences = valences)
        #substructure = rdFMCS.FindMCS(cluster_molecules, timeout=1, seedSmarts = substructure.smartsString, ringMatchesRingOnly = rings, completeRingsOnly = rings)
    else:
        #print([Chem.MolToSmarts(x) for x in cluster_molecules])
        substructure = rdFMCS.FindMCS(cluster_mols, timeout=1, seedSmarts = seed, ringMatchesRingOnly = rings, completeRingsOnly = rings, matchValences = valences)
    return substructure.smartsString, Chem.MolFromSmarts(substructure.smartsString), substructure.canceled

# Function courtesy of Scikit Learn

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)
    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

def get_brothers(df_smiles, leaves, distance_matrix, distances, linkage='complete'):
    brothers = {}
    n_samples = len(distances) +1
    for node, distance in enumerate(distances):
        
        node  = node + n_samples
        print("Computing node " + str(node), end="\r")
        
        brothers[node] = []
        trained_model = structure_clustering(distance, linkage, distance_matrix = distance_matrix)
        clusters = get_clusters(df_smiles, trained_model)
        
        for key, value in clusters.items():
            flag= True
            
            if len(value) == 1:
                brothers[node].append(value[0])
                flag=False
            else:
                for key_2, value_2 in leaves.items():
                    if set(value) == set(value_2):
                        brothers[node].append(key_2)
                        flag=False
            if flag:
                print('ERROR: Cluster ' + str(key) + ' not identified.')
                
    return brothers

def mcs_silhouette(distance_index, fps_molecules, fps_substructures, leaves, brothers, ENC, config='mol2MCS'):
    
    '''
    n_samples-1 possible clustering models, distance index is the index of the distance in dendrogam
    defining the corresponding cluster. It is as well the highest node of the model.
    '''
    a = []
    b = []
    s = []
    isol = 0
    n_samples = len(fps_molecules)
    brothers_i = brothers[distance_index + n_samples]
    n_clusters = len(brothers_i)
    for i in range(n_samples):
        #print(f'Calculating the molecule [{i+1}/{n_samples}]', end='\r')
        flag = True
        for brother in brothers_i:
            try:
                if i in leaves[brother]:
                    node = brother # The node at which i belongs
                    flag=False
                    is_isolated=False
                    break
            except:
                if i == brother:
                    isol += 1
                    node = brother
                    is_isolated = True
                    flag=False
                    break
        
        if flag:
            print("A molecule was not identified in any cluster")
        
        if is_isolated:
            #s.append(0)
            a.append(0)
            b.append(0)
            s.append(0)
            continue
        else:
            # TODO: THIS WILL BE JACCARD BETWEEN ORIGINAL GRAPH AND INTERSECTION GRAPH
            a.append(ENC.get_distance(fps_molecules[i], fps_substructures[node-n_samples]))
            #b.append(0)
            #continue
            
        #a.append(ENC.get_distance(fps_molecules[i], fps_substructures[node-n_samples]))
        
        b_temp=[]
        for brother in brothers_i:
            if brother == node:
                continue
            if brother >= n_samples:
                if config == 'mol2MCS':
                    b_temp.append(ENC.get_distance(fps_molecules[i], fps_substructures[brother-n_samples]))
                else:
                    b_temp.append(ENC.get_distance(fps_substructures[node-n_samples], fps_substructures[brother-n_samples]))
            else:
                if config=='mol2MCS':
                    b_temp.append(ENC.get_distance(fps_substructures[node-n_samples], fps_molecules[brother]))
                else:
                    b_temp.append(ENC.get_distance(fps_molecules[i], fps_molecules[brother]))
        
        b.append(np.min(b_temp))
        s_temp = (b[i]-a[i])/np.max([a[i], b[i]])
        if s_temp == s_temp:
            s.append(s_temp)
        else:
            s.append(0)
    
    results = {
        'mean_silhouette':np.mean(s),
        'isolated_mols': isol,
        'n_clusters': n_clusters,
        'mean_a': np.mean(a),
        'mean_b': np.mean(b),
    }
    
    return results

def mcs_silhouette_graph(distance_index, fps_molecules, fps_substructures, leaves, brothers, ENC, config='mol2MCS'):
    
    '''
    n_samples-1 possible clustering models, distance index is the index of the distance in dendrogam
    defining the corresponding cluster. It is as well the highest node of the model.
    '''
    a = []
    b = []
    s = []
    isol = 0
    n_samples = len(fps_molecules)
    brothers_i = brothers[distance_index + n_samples]
    n_clusters = len(brothers_i)
    for i in range(n_samples):
        #print(f'Calculating the molecule [{i+1}/{n_samples}]', end='\r')
        flag = True
        for brother in brothers_i:
            try:
                if i in leaves[brother]:
                    node = brother # The node at which i belongs
                    flag=False
                    is_isolated=False
                    break
            except:
                if i == brother:
                    isol += 1
                    node = brother
                    is_isolated = True
                    flag=False
                    break
        
        if flag:
            print("A molecule was not identified in any cluster")
        
        if is_isolated:
            #s.append(0)
            a.append(0)
            b.append(0)
            s.append(0)
            continue
        else:
            # TODO: THIS WILL BE JACCARD BETWEEN ORIGINAL GRAPH AND INTERSECTION GRAPH
            a.append(ENC.get_distance(fps_molecules[i], fps_substructures[node-n_samples]))
            #b.append(0)
            #continue
            
        #a.append(ENC.get_distance(fps_molecules[i], fps_substructures[node-n_samples]))
        
        b_temp=[]
        for brother in brothers_i:
            if brother == node:
                continue
            if brother >= n_samples:
                if config == 'mol2MCS':
                    b_temp.append(ENC.get_distance(fps_molecules[i], fps_substructures[brother-n_samples]))
                else:
                    b_temp.append(ENC.get_distance(fps_substructures[node-n_samples], fps_substructures[brother-n_samples]))
            else:
                if config=='mol2MCS':
                    b_temp.append(ENC.get_distance(fps_substructures[node-n_samples], fps_molecules[brother]))
                else:
                    b_temp.append(ENC.get_distance(fps_molecules[i], fps_molecules[brother]))
        
        b.append(np.min(b_temp))
        s_temp = (b[i]-a[i])/np.max([a[i], b[i]])
        if s_temp == s_temp:
            s.append(s_temp)
        else:
            s.append(0)
    
    results = {
        'mean_silhouette':np.mean(s),
        'isolated_mols': isol,
        'n_clusters': n_clusters,
        'mean_a': np.mean(a),
        'mean_b': np.mean(b),
    }
    
    return results