from graph import *


def growCluster(graph, cluster, seed, growSize): 
    graph.vertex_in_cluster[seed.id] = cluster # assign the seed to the current cluster
    cluster.add(seed, graph)
    i = 0
    # Create a do-while loop with i
    while True: 
        
        if i >= growSize:
            break

        max_weight = -1
        max_vertex = Vertex('-1')
        '''
        Calculate every single external vertex's weight(vertices that not in current cluster) 
        with vertices in current cluster (Wij)
        ''' 
        for v in graph.vertices: 
            if v not in cluster.vertices and (v.id not in graph.vertex_in_cluster.keys()):
                v_weight = 0
                for cluster_v in cluster.vertices:    
                    if (v.id, cluster_v.id) in graph.weights:
                        v_weight += graph.weights[(v.id, cluster_v.id)]
                if v_weight > max_weight:
                    max_vertex = v
                    max_weight = v_weight

        if max_weight == -1:
            return cluster
        # Add the max_vertex to the cluster 
        graph.vertex_in_cluster[max_vertex.id] = cluster
        cluster.add(max_vertex, graph)

        i += 1 
        
    return cluster      

        