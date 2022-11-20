
N # number of vertices aka number of nodes
k # number of clusters
cluster = []*N # for i in range(N): cluster[i] = clusterID where node i belongs to
graph = [[] for i in range(N)] # adjacent list to store graph
all_weight = [[] for i in range(k)] # all_weight[i][j] = sum of weight from node j
								# to all the nodes within cluster i

def growCluster(graph = [[]], cluster = [], cluster_id, seed_id, growSize):
    nodeToCluWeight = [0]*N
    nodeToCluWeight[seed] = 1 # is it supposed seed_id instead of seed?

    for i in range(growSize):
        nodeToAdd = max_weight(nodeToCluWeight) # External node with highest weight to current cluster (cluster_id)

        if nodeToCluWeight[nodeToAdd] <= 0: # when nodeToAdd == seed (nodeToCluWeight[seed] = 1)
            return cluster
        
        cluster[nodeToAdd] = cluster_id
        nodeToCluWeight[node_id] = 0
        
        for j in range(size(graph[nodeToAdd][j])):
            (node_id, weight) = graph[nodeToAdd][j]
            if node_id not in cluster:
                nodeToCluWeight[node_id] += weight
    
    return cluster