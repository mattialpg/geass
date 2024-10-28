import networkx as nx

# v3
def get_graph_intersections(G_graph, H_graph):
    G_nodes = [n for n in G_graph.graph.nodes(data=True)]
    H_nodes = [n for n in H_graph.graph.nodes(data=True)]
    G_edges = [e for e in G_graph.graph.edges(data=True)]
    H_edges = [e for e in H_graph.graph.edges(data=True)]

    def backtrack(G_edges, H_edges, G_H_nodes_equivalences, G_edges_intersected, H_edges_intersected):
        if not G_edges:
            G_nodes_intersected = set(item for sublist in G_edges_intersected for item in sublist)
            H_nodes_intersected = set(item for sublist in H_edges_intersected for item in sublist)
            G_nodes_aux = list(G_nodes_intersected)
            G_nodes_aux.sort()
            intersection_graph = nx.Graph()
            intersection_graph.add_nodes_from([(new_id, G_graph.graph.nodes[n_id]) for new_id, n_id in enumerate(G_nodes_aux)])
            intersection_graph.add_edges_from([(G_nodes_aux.index(e[0]), G_nodes_aux.index(e[1]), G_graph.graph.edges[e]) for e in G_edges_intersected])
            # results.append(intersection_graph)
            print("End condition reached (no edges left)")
            return [intersection_graph]

        current_level_results = []
        
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
                        or target1_idx not in G_H_nodes_equivalences and source2_idx not in list(G_H_nodes_equivalences.values()))) and reaction1 == reaction2:
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
            intersection_graph = nx.Graph()
            intersection_graph.add_nodes_from([(new_id, G_graph.graph.nodes[n_id]) for new_id, n_id in enumerate(G_nodes_aux)])
            intersection_graph.add_edges_from([(G_nodes_aux.index(e[0]), G_nodes_aux.index(e[1]), G_graph.graph.edges[e]) for e in G_edges_intersected])
            current_level_results.append(intersection_graph)
            print("End condition reached (end of loop)")
        return current_level_results

    results = []
    results = backtrack(G_edges, H_edges, {}, [], [])
    return results

def get_best_intersection(intersected_graphs):
    best_intersection = None
    for graph in intersected_graphs:
        n_edges = graph.number_of_edges()
        n_nodes = graph.number_of_nodes()
        if not best_intersection or n_edges > best_intersection.number_of_edges() and n_nodes > best_intersection.number_of_nodes():
            best_intersection = graph
    return best_intersection