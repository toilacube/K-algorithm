
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
    """
        Read data from file.
        Will also return header if header=True
    """
    data = []
    with open(filename, 'rt') as csvfile:
        spamreader = csv.reader(csvfile, delimiter = ' ')
        for row in spamreader:
            data.append(row)
    return (np.array(data, dtype = object))


def load_graphs(filename):
    """
        Loads graphs from file
    """
    data = read_data(filename)
    graph = Graph()
    for line in data:
            if line[0] == 'v':
                v = Vertex(id = int(line[1]))
                graph.add_vertex(vertex = v)  
            elif line[0] == 'e':
                e = Edge(
                         from_vertex = graph.get_vertex(id = int(line[1])), 
                         to_vertex = graph.get_vertex(id = int(line[2])),
                         weight = int(line[3]))
                graph.add_edge(edge = e) # type: ignore
    return graph

class Cluster:
	pass


class Vertex:
	def __init__(self, id):
		self.id = id # id của 1 đỉnh
	def get_vertex_info(self):
		return self.id


class Edge:
	def __init__(self, from_vertex, to_vertex, weight = 0):
		self.weight = weight
		self.from_vertex = from_vertex
		self.to_vertex = to_vertex

	def connected_to(self, vertex):
		return vertex.id == self.from_vertex.id or \
	    	vertex.id == self.to_vertex.id

	def get_edge_info(self):
		info = [self.from_vertex.get_vertex_info(),
					self.to_vertex.get_vertex_info(),
					self.weight]
		return info

class Graph:
	edges, vertices = [], []
	def __init__(self):
		self.edges = []
		self.vertices = []

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

	def get_num_of_vertex(self):
		return len(self.vertices)

	def get_weight(self, from_vertex, to_vertex):
		for e in self.edges:
			if e.from_vertex.id == from_vertex and \
				e.to_vertex.id == to_vertex:
				return e.weight

def main():
	graph = load_graphs('cube_data.txt')
	vertex = Vertex(1)
	test = graph.adjacent_edges(vertex)  # type: ignore
	for edge in test:
		print(edge.get_edge_info())
	print(graph.get_num_of_vertex())
	print(graph.get_weight(0, 1))


if __name__ == "__main__":
	main()
