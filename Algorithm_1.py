# N # number of vertices aka number of nodes
# k # number of clusters
# cluster = []*N # for i in range(N): cluster[i] = clusterID where node i belongs to
# graph = [[] for i in range(N)] # adjacent list to store graph
# W_ij = [[] for i in range(N)]*k # W_ij[i][j] = sum of weight from node j
# 								# to all the nodes within cluster i; j <= k && i <= N
import random

from Algorithm_2 import *

def cost_function(a, b, c) -> float:
	return 0

def func_to_create_cluster(graph, k): # có nên Giả sử  k = 2 để hoàn thành algorithm 1 trước ?????
	# clusters = []
	# for i in range(k):
	# 	clusters.append(Cluster(id = i))
	# for i in range(len(graph.vertices)):
	# 	if i 
	# return clusters
	pass

def K_Algorithm(graph, k, cluster = [[]]):
# Line 1	
	N = graph.get_num_of_vertex()
	if cluster == None: # Null
		cluster = func_to_create_cluster(N, k)
	changed = 0

	while True:
	# Create N random unique values from 1 to N
		sample = random.sample(range(0, N), N)
# Line 5
		for i in sample:
			old = cluster[i] # cluster tiện tại của node i 
			newpart = 1
			bestdelta = float('inf')
# Line 9	
			Wi = [0] * k # Sum of internal weights in cluster i
			Wij = [[0] * N] * k # Sum of weights from node j to nodes within cluster i: Wij[i][j]

			for clus in range(k):
				pass

				
# Line 10
			for j in range(size(graph[i])):
				(nodeId, weight) = graph[i][j]
				x = cluster[nodeId]
				
				# Tính tổng số weight của node i tới từng các cluster
				for one_weight in W:
					one_weight[i] += 2*weight 
# Line 14
				for y in range(k):
					d = cost_function(i, cluster[i], j)
					if d < bestdelta:
						bestdelta = d
						newpart = j
				if newpart != old:
					changed +=1
					cluster[i] = new # ?? Unknown variable new, maybe it is newpart
		if changed <= 0:
			break
	return cluster 

