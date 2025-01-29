import networkx as nx
from utils import get_all_NA_neighbours_names

# v3
def get_graph_intersections(G_geass, H_geass, add_not_common=True):
    G_nodes = [n for n in G_geass.G.nodes(data=True)]
    H_nodes = [n for n in H_geass.G.nodes(data=True)]
    G_edges_original = [e for e in G_geass.G.edges(data=True)]
    H_edges_original = [e for e in H_geass.G.edges(data=True)]

    def backtrack(G_edges, H_edges, G_H_nodes_equivalences, G_edges_intersected, H_edges_intersected):
        if not G_edges:
            G_nodes_intersected = set(item for sublist in G_edges_intersected for item in sublist)
            H_nodes_intersected = set(item for sublist in H_edges_intersected for item in sublist)
            G_nodes_aux = list(G_nodes_intersected)
            G_nodes_aux.sort()
            # Get G and H nodes that are not in the intersection
            G_nodes_not_intersected = [n[0] for n in G_nodes if n[0] not in G_nodes_intersected]
            H_nodes_not_intersected = [n[0] for n in H_nodes if n[0] not in H_nodes_intersected]
            # Get G and H edges that are not in the intersection
            G_edges_not_intersected = [e for e in G_edges_original if (e[0], e[1]) not in G_edges_intersected]
            H_edges_not_intersected = [e for e in H_edges_original if (e[0], e[1]) not in H_edges_intersected]
            intersection_graph = nx.Graph()
            intersection_graph.add_nodes_from([(new_id, G_geass.G.nodes[n_id]) for new_id, n_id in enumerate(G_nodes_aux)])
            # Add the property 'intersection': True to the nodes that are in the intersection
            for n in intersection_graph.nodes:
                intersection_graph.nodes[n]['intersection'] = 'Yes' 
            intersection_graph.add_edges_from([(G_nodes_aux.index(e[0]), G_nodes_aux.index(e[1]), G_geass.G.edges[e]) for e in G_edges_intersected])
            # Add the property 'intersection': True to the edges that are in the intersection
            for e in intersection_graph.edges:
                intersection_graph.edges[e]['intersection'] = 'Yes'
            if add_not_common:
                # Add the nodes from G that are not in the intersection
                intersection_graph.add_nodes_from([(new_id+len(G_nodes_aux), G_geass.G.nodes[n_id]) for new_id, n_id in enumerate(G_nodes_not_intersected)])
                # Add the nodes from H that are not in the intersection
                intersection_graph.add_nodes_from([(new_id+len(G_nodes_aux)+len(G_nodes_not_intersected), H_geass.G.nodes[n_id]) for new_id, n_id in enumerate(H_nodes_not_intersected)])
                # Add the property 'intersection': False to the nodes that are not in the intersection
                for n in range(len(G_nodes_aux), len(G_nodes_aux)+len(G_nodes_not_intersected)):
                    intersection_graph.nodes[n]['intersection'] = 'No'
                    intersection_graph.nodes[n]['origin_graph'] = 1
                for n in range(len(G_nodes_aux)+len(G_nodes_not_intersected), len(G_nodes_aux)+len(G_nodes_not_intersected)+len(H_nodes_not_intersected)):
                    intersection_graph.nodes[n]['intersection'] = 'No'
                    intersection_graph.nodes[n]['origin_graph'] = 2
                # Create the edges that are not in the intersection with the new ids
                edges_not_intersected = []
                for e in G_edges_not_intersected:
                    source = G_nodes_aux.index(e[0]) if e[0] in G_nodes_aux else len(G_nodes_aux)+G_nodes_not_intersected.index(e[0])
                    target = G_nodes_aux.index(e[1]) if e[1] in G_nodes_aux else len(G_nodes_aux)+G_nodes_not_intersected.index(e[1])
                    new_data = e[2].copy()
                    new_data['intersection'] = 'No'
                    new_data['origin_graph'] = 1
                    edges_not_intersected.append((source, target, new_data))
                for e in H_edges_not_intersected:
                    source_G_equivalence = next((k for k in G_H_nodes_equivalences if G_H_nodes_equivalences[k]==e[0]), None)
                    target_G_equivalence = next((k for k in G_H_nodes_equivalences if G_H_nodes_equivalences[k]==e[1]), None)
                    source = G_nodes_aux.index(source_G_equivalence) if source_G_equivalence is not None else len(G_nodes_aux)+len(G_nodes_not_intersected)+H_nodes_not_intersected.index(e[0])
                    target = G_nodes_aux.index(target_G_equivalence) if target_G_equivalence is not None else len(G_nodes_aux)+len(G_nodes_not_intersected)+H_nodes_not_intersected.index(e[1])
                    new_data = e[2].copy()
                    new_data['intersection'] = 'No'
                    new_data['origin_graph'] = 2
                    edges_not_intersected.append((source, target, new_data))
                # Add the edges not in intersection to intersection_graph
                intersection_graph.add_edges_from(edges_not_intersected)
            # results.append(intersection_graph)
            print("End condition reached (no edges left)")
            return [intersection_graph]

        current_level_results = []
        
        for e1 in G_edges:
            source1_idx = e1[0]
            source1 = G_geass.G.nodes[source1_idx]['BU_ID'] if G_geass.G.nodes[source1_idx]['BU_ID'] != 'NA' else get_all_NA_neighbours_names(G_geass, source1_idx)
            target1_idx = e1[1]
            target1 = G_geass.G.nodes[target1_idx]['BU_ID'] if G_geass.G.nodes[target1_idx]['BU_ID'] != 'NA' else get_all_NA_neighbours_names(G_geass, target1_idx)
            reaction1 = e1[2]

            for e2 in H_edges:
                source2_idx = e2[0]
                source2 = H_geass.G.nodes[source2_idx]['BU_ID'] if H_geass.G.nodes[source2_idx]['BU_ID'] != 'NA' else get_all_NA_neighbours_names(H_geass, source2_idx)
                target2_idx = e2[1]
                target2 = H_geass.G.nodes[target2_idx]['BU_ID'] if H_geass.G.nodes[target2_idx]['BU_ID'] != 'NA' else get_all_NA_neighbours_names(H_geass, target2_idx)
                reaction2 = e2[2]

                if (source1 == source2 and target1 == target2 
                    and (source1_idx in G_H_nodes_equivalences and G_H_nodes_equivalences[source1_idx] == source2_idx 
                        or source1_idx not in G_H_nodes_equivalences and source2_idx not in G_H_nodes_equivalences.values())
                    and (target1_idx in G_H_nodes_equivalences and G_H_nodes_equivalences[target1_idx] == target2_idx
                        or target1_idx not in G_H_nodes_equivalences and target2_idx not in list(G_H_nodes_equivalences.values()))
                    or 
                    source1==target2 and target1 == source2 
                    and (source1_idx in G_H_nodes_equivalences and G_H_nodes_equivalences[source1_idx] == target2_idx
                        or source1_idx not in G_H_nodes_equivalences and target2_idx not in list(G_H_nodes_equivalences.values()))
                    and (target1_idx in G_H_nodes_equivalences and G_H_nodes_equivalences[target1_idx] == source2_idx
                        or target1_idx not in G_H_nodes_equivalences and source2_idx not in list(G_H_nodes_equivalences.values()))) and reaction1['reaction'] == reaction2['reaction']:
                    new_G_H_nodes_equivalences = G_H_nodes_equivalences.copy()
                    new_G_edges_intersected = G_edges_intersected.copy()
                    new_H_edges_intersected = H_edges_intersected.copy()

                    if source1 == source2:
                        if source1_idx not in new_G_H_nodes_equivalences and source2_idx not in new_G_H_nodes_equivalences.values():
                            new_G_H_nodes_equivalences[source1_idx] = source2_idx
                        if target1_idx not in new_G_H_nodes_equivalences and target2_idx not in new_G_H_nodes_equivalences.values():
                            new_G_H_nodes_equivalences[target1_idx] = target2_idx
                    else:
                        if source1_idx not in new_G_H_nodes_equivalences and target2_idx not in new_G_H_nodes_equivalences.values():
                            new_G_H_nodes_equivalences[source1_idx] = target2_idx
                        if target1_idx not in new_G_H_nodes_equivalences and source2_idx not in new_G_H_nodes_equivalences.values():
                            new_G_H_nodes_equivalences[target1_idx] = source2_idx

                    new_G_edges_intersected.append((source1_idx, target1_idx))
                    new_H_edges_intersected.append((source2_idx, target2_idx))

                    new_results = backtrack(G_edges[1:], [e for e in H_edges if e != e2], new_G_H_nodes_equivalences, new_G_edges_intersected, new_H_edges_intersected)
                    current_level_results = current_level_results + new_results
        # If there are no results yet, add the current result before returning
        if not current_level_results:
            G_nodes_intersected = set(item for sublist in G_edges_intersected for item in sublist)
            H_nodes_intersected = set(item for sublist in H_edges_intersected for item in sublist)
            G_nodes_aux = list(G_nodes_intersected)
            G_nodes_aux.sort()
            # Get G and H nodes that are not in the intersection
            G_nodes_not_intersected = [n[0] for n in G_nodes if n[0] not in G_nodes_intersected]
            H_nodes_not_intersected = [n[0] for n in H_nodes if n[0] not in H_nodes_intersected]
            # Get G and H edges that are not in the intersection
            G_edges_not_intersected = [e for e in G_edges_original if (e[0], e[1]) not in G_edges_intersected]
            H_edges_not_intersected = [e for e in H_edges_original if (e[0], e[1]) not in H_edges_intersected]
            intersection_graph = nx.Graph()
            intersection_graph.add_nodes_from([(new_id, G_geass.G.nodes[n_id]) for new_id, n_id in enumerate(G_nodes_aux)])
            # Add the property 'intersection': True to the nodes that are in the intersection
            for n in intersection_graph.nodes:
                intersection_graph.nodes[n]['intersection'] = 'Yes' 
            intersection_graph.add_edges_from([(G_nodes_aux.index(e[0]), G_nodes_aux.index(e[1]), G_geass.G.edges[e]) for e in G_edges_intersected])
            # Add the property 'intersection': True to the edges that are in the intersection
            for e in intersection_graph.edges:
                intersection_graph.edges[e]['intersection'] = 'Yes'
            if add_not_common:
                # Add the nodes from G that are not in the intersection
                intersection_graph.add_nodes_from([(new_id+len(G_nodes_aux), G_geass.G.nodes[n_id]) for new_id, n_id in enumerate(G_nodes_not_intersected)])
                # Add the nodes from H that are not in the intersection
                intersection_graph.add_nodes_from([(new_id+len(G_nodes_aux)+len(G_nodes_not_intersected), H_geass.G.nodes[n_id]) for new_id, n_id in enumerate(H_nodes_not_intersected)])
                # Add the property 'intersection': False to the nodes that are not in the intersection
                for n in range(len(G_nodes_aux), len(G_nodes_aux)+len(G_nodes_not_intersected)):
                    intersection_graph.nodes[n]['intersection'] = 'No'
                    intersection_graph.nodes[n]['origin_graph'] = 1
                for n in range(len(G_nodes_aux)+len(G_nodes_not_intersected), len(G_nodes_aux)+len(G_nodes_not_intersected)+len(H_nodes_not_intersected)):
                    intersection_graph.nodes[n]['intersection'] = 'No'
                    intersection_graph.nodes[n]['origin_graph'] = 2
                # Create the edges that are not in the intersection with the new ids
                edges_not_intersected = []
                for e in G_edges_not_intersected:
                    source = G_nodes_aux.index(e[0]) if e[0] in G_nodes_aux else len(G_nodes_aux)+G_nodes_not_intersected.index(e[0])
                    target = G_nodes_aux.index(e[1]) if e[1] in G_nodes_aux else len(G_nodes_aux)+G_nodes_not_intersected.index(e[1])
                    new_data = e[2].copy()
                    new_data['intersection'] = 'No'
                    new_data['origin_graph'] = 1
                    edges_not_intersected.append((source, target, new_data))
                for e in H_edges_not_intersected:
                    source_G_equivalence = next((k for k in G_H_nodes_equivalences if G_H_nodes_equivalences[k]==e[0]), None)
                    target_G_equivalence = next((k for k in G_H_nodes_equivalences if G_H_nodes_equivalences[k]==e[1]), None)
                    source = G_nodes_aux.index(source_G_equivalence) if source_G_equivalence is not None else len(G_nodes_aux)+len(G_nodes_not_intersected)+H_nodes_not_intersected.index(e[0])
                    target = G_nodes_aux.index(target_G_equivalence) if target_G_equivalence is not None else len(G_nodes_aux)+len(G_nodes_not_intersected)+H_nodes_not_intersected.index(e[1])
                    new_data = e[2].copy()
                    new_data['intersection'] = 'No'
                    new_data['origin_graph'] = 2
                    edges_not_intersected.append((source, target, new_data))
                # Add the edges not in intersection to intersection_graph
                intersection_graph.add_edges_from(edges_not_intersected)
            current_level_results.append(intersection_graph)
            print("End condition reached (end of loop)")
        return current_level_results

    results = []
    results = backtrack(G_edges_original, H_edges_original, {}, [], [])
    return results

def get_best_intersection(intersected_graphs):
    best_intersection = None
    for graph in intersected_graphs:
        n_edges = graph.number_of_edges()
        n_nodes = graph.number_of_nodes()
        n_edges_with_intersection = sum(1 for _, _, data in graph.edges(data=True) if data.get('intersection') == 'Yes')
        n_nodes_with_intersection = sum(1 for _, data in graph.nodes(data=True) if data.get('intersection') == 'Yes')
        if best_intersection:
            best_intersection_n_edges_with_intersection = sum(1 for _, _, data in best_intersection.edges(data=True) if data.get('intersection') == 'Yes')
            best_intersection_n_nodes_with_intersection = sum(1 for _, data in best_intersection.nodes(data=True) if data.get('intersection') == 'Yes')
        if not best_intersection or n_edges_with_intersection > best_intersection_n_edges_with_intersection and n_nodes_with_intersection > best_intersection_n_nodes_with_intersection:
            best_intersection = graph
    return best_intersection