def create_compound_edges(result_graphs):
    compound_edges = {}
    for shared_name, rg in result_graphs.items():
        current_edges = []
        #print(shared_name)
        for edge in list(rg.G.edges(data=True)):
            current_kegg_id_0 = rg.G.nodes[edge[0]]['kegg_id']
            current_kegg_id_1 = rg.G.nodes[edge[1]]['kegg_id']
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