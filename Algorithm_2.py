from asyncio.windows_events import NULL


N # number of vertices aka number of nodes
k # number of clusters
cluster = []*N # for i in range(N): cluster[i] = clusterID where node i belongs to
graph = [[] for i in range(N)] # adjacent list to store graph
all_weight = [[] for i in range(k)] # all_weight[i][j] = sum of weight from node j
								# to all the nodes within cluster i

def initialPartion(graph = [[]], k):
# Line 1
    for i in range(N):
        cluster[i] = NULL
    graph = sorted_based_on_density(graph)
    seed = cluster_id = 1
# Line 5
    while cluster_id <= k and seed <= N:
        while cluster[seed] not NULL:
            seed += 1

        cluster = growCluster(graph, cluster, cluster_id,seed, 0.8*(N/k)) # 80% node will be assign to a cluster
        cluster_id += 1

    for i in range(N): # the remaining 20% node will be chosen randomly
        if cluster[i] == NULL:
            cluster[i] = rand(1, k)
    return cluster
    


