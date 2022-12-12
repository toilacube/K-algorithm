
# N # number of vertices aka number of nodes
# k # number of clusters
# cluster = []*N # for i in range(N): cluster[i] = clusterID where node i belongs to
# graph = [[] for i in range(N)] # adjacent list to store graph
# all_weight = [[] for i in range(k)] # all_weight[i][j] = sum of weight from node j
# 								# to all the nodes within cluster i

# weight = True
# directed = False

# does dictionary in this project can cause memory problem?

import numpy as np
import csv

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
						
		self.vertex_in_cluster = {}		# a dictionary with key is a vertex_id and value is its cluster_id it belongs to

		self.K = K # number of clusters
		self.cluster = [] * K

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

	def get_weight_with_from_to(self, from_vertex_id, to_vertex_id):
		for e in self.edges:
			if e.from_vertex.id == from_vertex_id and e.to_vertex.id == to_vertex_id:
				return e.get_weight()

class Cluster(Graph):
	def __init__(self, id) -> None:
		Graph.__init__(self)
		self.id = id

	# dict_wegihts: a dictionary with key is an edge(from_vertex.id, to_vertex.id) and value is its weight

	def get_cluster_weight(self, dict_weights):
		weight = 0
	#	self.weights = self.get_dict_of_weights()
		for i in range(len(self.vertices)):
			for j in range(i + 1, len(self.vertices)):
				if (self.vertices[i].id, self.vertices[j].id) in dict_weights:
					weight += dict_weights[(self.vertices[i].id, self.vertices[j].id)]
		return weight*2

	def get_vertex_weight(self, vertex, dict_weights):
		weight = 0
		for v in self.vertices:
			if v.id == vertex.id:
				for v1 in self.vertices:
					if (v.id, v1.id) in dict_weights:
						weight += dict_weights[(v.id, v1.id)]
		return weight
			


def main():
	graph = load_graphs('cube_data.txt')
	cluster = Cluster(Graph)
	cluster.add_vertex(Vertex('2'))
	cluster.add_vertex(Vertex('4'))
	cluster.add_vertex(Vertex('6'))
	cluster.add_edge(Edge(Vertex('2'), Vertex('4'), 3))
	cluster.add_edge(Edge(Vertex('4'), Vertex('6'), 3))
	# graph.get_dict_of_weights()
	# cluster.get_dict_of_weights()
	
	print(graph.weights)

	 # Equation: dens(i) = multyplying the neighbor nod's total weight (Mj) by the edge weight(Wij) connecting to that neighbor +	
	dens = []
    # Calculate density for each vertex
	for vertex in graph.vertices: 
		for neighbor in vertex.neighbors:
			vertex.density += graph.weights[(vertex.id, neighbor.id)] * neighbor.weight
		dens.append(vertex)
    # Sort the vertex base on density
	dens = sorted(dens, key = lambda vertex: vertex.density, reverse = True)

	print(len(dens), 'ok\n')
	for v in dens:
		print(v.id, v.density)
	print(graph.weights[('1', '2')])
	print(graph.vertices[0].neighbors[0].id)

if __name__ == "__main__":
	main()
