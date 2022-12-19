import random
from Algorithm_2 import *

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

