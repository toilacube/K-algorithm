from Algorithm_3 import *
import random

def sorted_base_on_density(graph) -> list: 

    # Equation: dens(i) = multyplying the neighbor nod's total weight (Mj) by the edge weight(Wij) connecting to that neighbor +
    dens = []

    # Calculate density for each vertex
    for vertex in graph.vertices: 
        for neighbor in vertex.neighbors:
            vertex.density += graph.weights[(vertex.id, neighbor.id)] * neighbor.weight
            dens.append(vertex)

    # Sort the vertex base on density
    return sorted(dens, key = lambda vertex: vertex.density, reverse = True)


def initialPartion(graph):

    N = len(graph.vertices)

    dens = sorted_base_on_density(graph) # a list of vertex sorted base on density (Desc = True)
    seed_index = 0

    for i in range(graph.K):

        # select the seed that not in a cluster yet
        while graph.vertex_in_cluster[dens[seed_index].id] is not None: 
            seed_index += 1

        if seed_index >= N:
            break

        if graph.cluster[i] is None:
            graph.cluster[i].id = i
            graph.cluster[i] = growCluster( graph,
                                            graph.cluster[i],
                                            dens[seed_index], 
                                            growSize = 0.8 * N /graph.K)
        
    for i in range(N, 0, -1):
        if graph.vertex_in_cluster[dens[i].id] is None:
            j = random.randrange(0, graph.K)
            graph.cluster[j].add_vertex(dens[i]) 
            graph.vertex_in_cluster[dens[i].id] = j  

    return graph.cluster
 

