import random

from Algorithm_2 import *


# for IIW(j, x, y): 1/(Wx - Wxj) + 1/(Wy + Wyj) - 1/Wx - 1/Wy
'''
	notation:
	Wx = 2*(sum of all possible weights from cluster node)
	Wxi = sum of weights from node i to all possible node in cluster x
'''
def cost_function(vertex, clus1, clus2):

	return 0

def K_Algorithm(graph):
# Line 1	
	N = len(graph.vertices)
	cluster = initialPartion(graph) # list of Clusters
	changed = 0
	cost_weight = {} # a dict with keys are K cluster_id, values are weights from a node to cluster i  

	zero = {}
	for i in range(graph.K):
		zero[str(i)] = 0

	# Create a list of N random unique values from 0 to N-1
	sample = random.sample(range(0, N), N)

	while True:

		for i in sample:
			bestdelta = float('inf')
			new_clus_id = 1
			v = graph.vertices[i]
			old_clus_id = graph.vertex_in_cluster[v.id]

			for neighbor in v.neighbors: # loop all connections of i
				clus_id = graph.vertex_in_cluster[neighbor.id]
				cost_weight[clus_id] += 2*graph.weights[(neighbor.id, v.id)]

			for j in range(graph.K):
				cost = cost_function(v,graph.vertex_in_cluster[v.id], j)
				if cost < bestdelta:
					bestdelta = cost
					new_clus_id = j

			if new_clus_id != old_clus_id: # if node i change its cluster
				changed += 1
				graph.vertex_in_cluster[v.id] = new_clus_id

		if changed <= 0: # if all nodes dont change its cluster then finish
			break

	return cluster 

