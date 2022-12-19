# https://github.com/toilacube/K-algorithm.git

import numpy as np
import csv
import random



def read_data(filename):
    # Read data from file.
 
    data = []
    with open(filename, 'rt') as csvfile:
        spamreader = csv.reader(csvfile, delimiter = ' ')
        for row in spamreader:
            data.append(row)
    return (np.array(data, dtype = object))


def load_graphs(filename):
    # Loads graphs from file

    data = read_data(filename)
    graph = Graph()
    for line in data:
            if line[0] == 'v':
                v = Vertex(id = line[1])
                graph.add_vertex(vertex = v)  
            elif line[0] == 'e':
                e = Edge(
                         from_vertex = graph.get_vertex(id = line[1]), 
                         to_vertex = graph.get_vertex(id = line[2]),
                         weight = int(line[3]))
                graph.add_edge(edge = e) 
    return graph

class Vertex:
    def __init__(self, id: str):
        self.id = id # id của 1 đỉnh
        self.neighbors = []
        self.weight = 0 # sum of weight of all edges contain this vertex
        self.density = 0 # vertex density

    def add_neighbor(self, vertex):
        self.neighbors.append(vertex)
    
class Edge:
    def __init__(self, from_vertex, to_vertex, weight = 0):
        self.weight = weight
        self.from_vertex = from_vertex
        self.to_vertex = to_vertex

    def connected_to(self, vertex):
        return (vertex.id == self.from_vertex.id or
        vertex.id == self.to_vertex.id)
    
class Graph: 
    # edges, vertices = [], []
    def __init__(self, K = 1) -> None:
        self.edges = []
        self.vertices = []

        self.weights = {} # a dictionary to store weight(value) correspond to its edge(key_v - (from, to_v))
                        
        self.vertex_in_cluster = {}		# a dictionary with key is a vertex_id and value is its cluster it belongs to

        self.K = K # number of clusters
        self.cluster = [Cluster] * K

    def add_vertex(self, vertex):
        self.vertices.append(vertex)

    def add_edge(self, edge):
        self.edges.append(edge)

        edge.from_vertex.add_neighbor(edge.to_vertex)
        edge.to_vertex.add_neighbor(edge.from_vertex)

        edge.from_vertex.weight += edge.weight
        edge.to_vertex.weight += edge.weight

        self.weights[(edge.from_vertex.id, edge.to_vertex.id)] = edge.weight
        self.weights[(edge.to_vertex.id, edge.from_vertex.id)] = edge.weight

    def get_vertex(self, id):
        for v in self.vertices:
            if v.id == id:
                return v
        raise KeyError('No vertex with the id was found in graph')
    
    def adjacent_edges(self, vertex):
        adj_edges = []
        for e in self.edges:
            if e.connected_to(vertex):
                adj_edges.append(e)	
        return adj_edges
    
    def adjacent_vertices(self, vertex):
        adj_vertices = []
        adj_e = self.adjacent_edges(vertex)
        for e in adj_e:
            if e.from_vertex.id == vertex.id:
                adj_vertices.append(e.to_vertex)
            else: 
                adj_vertices.append(e.from_vertex)
        return adj_vertices

class Cluster(Graph):
    def __init__(self, id = -1):
        Graph().__init__()
        self.id = id

    def get_cluster_weight(self):
        weight = 0
    #	self.weights = self.get_dict_of_weights()
        for i in range(len(self.vertices)):
            for j in range(i + 1, len(self.vertices)):
                if (self.vertices[i].id, self.vertices[j].id) in self.weights:
                    weight += self.weights[(self.vertices[i].id, self.vertices[j].id)]
        return weight*2

    def get_vertex_weight(self, vertex):
        weight = 0
        for v in self.vertices:
            if v.id == vertex.id:
                for v1 in self.vertices:
                    if (v.id, v1.id) in self.weights:
                        weight += self.weights[(v.id, v1.id)]
        return weight
    
    def remove(self, vertex):
        # remove vertex
        self.vertices.remove(vertex)
        # remove edge and weights
        for i in range(len(self.edges)):
            if self.edges[i].from_vertex.id == vertex.id or \
                self.edges[i].to_vertex.id == vertex.id: 
                    self.weights.pop((self.edges[i].from_vertex.id, self.edges[i].to_vertex.id))
                    self.weights.pop((self.edges[i].to_vertex.id, self.edges[i].from_vertex.id))
                    self.edges.remove(self.edges[i])

    def add(self, vertex, graph):
        # add to vertices
        self.vertices.append(vertex)
        # add all possible edges and weights
        keys = graph.weights.keys()
        for v in self.vertices:
            if (v.id, vertex.id) in keys:
                self.edges.append(Edge(v, vertex, graph.weights[(v.id, vertex.id)])) # add edge
                self.weights[(v.id, vertex.id)] = graph.weights[(v.id, vertex.id)] # add weight



# for IIW(j, x, y): 1/(Wx - Wxj) + 1/(Wy + Wyj) - 1/Wx - 1/Wy
'''
    notation:
    Wx = 2*(sum of all possible weights from cluster node); status: ok, can calculate with cluster.get_cluster_weight()
    Wxi = sum of weights from node i to all possible node in cluster x; status: ok, use cost_weight {}
'''
def cost_function(clus1, clus2, cost_weight):
    weight1 = clus1.get_cluster_weight()
    weight2 = clus2.get_cluster_weight()
    cost1 = cost_weight[clus1.id]
    cost2 = cost_weight[clus2.id]
    return 1/(weight1 - cost1) + 1/(weight2 + cost2) - 1/weight1 - 1/weight2

def K_Algorithm(graph):

    '''
        Arg: Graph, type: Graph object
        Return: Cluster, type: list of Cluster object
    '''

# Line 1	
    N = len(graph.vertices)
    print(N)
    cluster = initialPartion(graph) # list of Clusters, but still only use graph.clusters
    changed = 0
    cost_weight = {} # a dict with keys are K cluster_id, values are weights from a node to cluster i  

    zero = {}
    for i in range(graph.K):
        zero[str(i)] = 0

    # Create a list of N random unique values from 0 to N-1
    sample = random.sample(range(0, N), N) # why we have to do randomly? maybe about the probability

    while True:

        for i in sample:
            cost_weight = zero
            bestdelta = float('inf')
            new_clus_id = 1
            v = graph.vertices[i]
            old_clus_id = graph.vertex_in_cluster[v.id].id

            for neighbor in v.neighbors: # loop all connections of i to calculate Wxi for all cluster x
                clus_id = graph.vertex_in_cluster[neighbor.id].id
                cost_weight[clus_id] += 2*graph.weights[(neighbor.id, v.id)]

            for j in range(graph.K):
                cost = cost_function (graph.vertex_in_cluster[v.id], graph.cluster[j], cost_weight)
                if cost < bestdelta:
                    bestdelta = cost
                    new_clus_id = graph.cluster[j].id

            if new_clus_id != old_clus_id: # if node v change to better cluster
                changed += 1
                graph.vertex_in_cluster[v.id] = graph.cluster[new_clus_id] 
                # delete node v from previous cluster 
                graph.cluster[old_clus_id].remove(v)
                # add node v to current best cluster
                graph.cluster[new_clus_id].add(v, graph)

        if changed <= 0: # if all nodes dont change its cluster then finish
            break

    return cluster 


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
        while dens[seed_index].id in graph.vertex_in_cluster.keys():
        #while graph.vertex_in_cluster[dens[seed_index].id] is not None: 
            seed_index += 1

        if seed_index >= N:
            break
        print(type(graph.cluster[i]))
        if graph.cluster[i].id == -1:
            graph.cluster[i].id = i
            graph.cluster[i] = growCluster( graph,
                                            graph.cluster[i],
                                            dens[seed_index], 
                                            growSize = 0.8 * N /graph.K)
        
    for i in range(N, 0, -1):
        if dens[i].id not in graph.vertex_in_cluster.keys():
            j = random.randrange(0, graph.K)
          #  print(type(graph.cluster[j]),"\n" ,type(dens[i]))
            graph.cluster[j].add_vertex(dens[i]) 
            graph.vertex_in_cluster[dens[i].id] = graph.cluster[j] 
    return graph.cluster

def growCluster(graph, cluster, seed, growSize): 

    graph.vertex_in_cluster[seed.id] = cluster # assign the seed to the current cluster
    cluster.add_vertex(seed)
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

        if max_weight == 0:
            return cluster
        # Add the max_vertex to the cluster 
        graph.vertex_in_cluster[max_vertex.id].id = cluster.id
        cluster.add_vertex(max_vertex)

        i += 1 

    return cluster    

def main():
    graph = Graph()
    graph = load_graphs('cube_data.txt')
    cluster = K_Algorithm(graph)
    print(cluster)

if __name__ == "__main__":
    main()
