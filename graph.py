
# N # number of vertices aka number of nodes
# k # number of clusters
# cluster = []*N # for i in range(N): cluster[i] = clusterID where node i belongs to
# graph = [[] for i in range(N)] # adjacent list to store graph
# all_weight = [[] for i in range(k)] # all_weight[i][j] = sum of weight from node j
# 								# to all the nodes within cluster i

# weight = True
# directed = False

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
	def get_vertex_info(self):
		return self.id


class Edge:
	def __init__(self, from_vertex, to_vertex, weight = 0):
		self.weight = weight
		self.from_vertex = from_vertex
		self.to_vertex = to_vertex

	def connected_to(self, vertex):
		return (vertex.id == self.from_vertex.id or
		vertex.id == self.to_vertex.id)
	
	def get_edge_info(self):
		info = [self.from_vertex.id,
					self.to_vertex.id,
					self.weight]
		return info


class Graph:
	# edges, vertices = [], []
	def __init__(self):
		self.edges = []
		self.vertices = []
		self.weights = {} # a dictionary to store weight(value), edge(key_v - (from, to_v))
		for e in self.edges:
			self.weights[(e.from_vertex, e.to_vertex)] = e.weight
			self.weights[(e.to_vertex, e.from_vertex)] = e.weight

	def add_vertex(self, vertex):
		self.vertices.append(vertex)

	def add_edge(self, edge):
		self.edges.append(edge)

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

	def get_dict_of_weights(self): 

	# Create a dictionary with key is vertices of an edge and value is edge's weight

		for e in self.edges:
			self.weights[(e.from_vertex.id, e.to_vertex.id)] = e.weight
			self.weights[(e.to_vertex.id, e.from_vertex.id)] = e.weight
		return self.weights

class Cluster(Graph):
	def __init__(self, id) -> None:
		Graph.__init__(self)
		self.id = id

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
	print(len(cluster.vertices))
	dict_weights = cluster.get_dict_of_weights()
	print(cluster.get_cluster_weight(dict_weights))
	print(cluster.weights)
	cube = Vertex('4')
	print(cluster.get_vertex_weight(cube, dict_weights))

if __name__ == "__main__":
	main()
