
# N # number of vertices aka number of nodes
# k # number of clusters
# cluster = []*N # for i in range(N): cluster[i] = clusterID where node i belongs to
# graph = [[] for i in range(N)] # adjacent list to store graph
# all_weight = [[] for i in range(k)] # all_weight[i][j] = sum of weight from node j
# 								# to all the nodes within cluster i

from graph import *


def growCluster(graph, cluster, cluster_id, seed_id, growSize): # seed_id: belonging to the list of nodes which is sorted based on density(desc) 
    # nodeToCluWeight = [0]*N
    # nodeToCluWeight[seed] = 1 # is it supposed seed_id instead of seed?

    # for i in range(growSize):
    #     nodeToAdd = max_weight(nodeToCluWeight) # External node with highest weight to current cluster (cluster_id)

    #     if nodeToCluWeight[nodeToAdd] <= 0: # No more Max Node with weight > 0
    #         return cluster
        
    #     cluster[nodeToAdd] = cluster_id
    #     nodeToCluWeight[node_id] = 0
        
    #     for j in range(size(graph[nodeToAdd][j])):
    #         (node_id, weight) = graph[nodeToAdd][j]
    #         if node_id not in cluster:
    #             nodeToCluWeight[node_id] += weight
    
    # return cluster
    if not graph.weights:
        graph_weights = graph.get_dict_of_weights()
    else: 
        graph_weights = graph.weight

    graph.vertex_in_cluster[seed_id] = cluster_id # assign the seed to the current cluster

    i = 0
    while True:
        max_weight = 0
        max_vertex = Vertex('-1')
        '''
        Calculate every single external vertex(vertices that not in current cluster) 
        weight with vertices in current cluster (Wij)
        '''
        for v in graph.vertices: 
            if v not in cluster.vertices and graph.vertex_in_cluster[v.id] <= 0:
                v_weight = 0
                for cluster_v in cluster.vertices:    
                    if (v.id, cluster_v.id) in graph_weights:
                        v_weight += graph_weights[(v.id, cluster_v.id)]
                if v_weight > max_weight:
                    max_vertex = v
                    max_weight = v_weight

        if max_weight == 0:
            return cluster
        i += 1
        if i >= growSize:
            graph.vertex_in_cluster[max_vertex.id] = cluster_id
            cluster.add_vertex(max_vertex)
            break

    return cluster    

        