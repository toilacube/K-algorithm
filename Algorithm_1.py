import random
from Algorithm_2 import *
# for IIW(j, x, y): 1/(Wx - Wxj) + 1/(Wy + Wyj) - 1/Wx - 1/Wy
'''
    notation:
    Wx = 2*(sum of all possible weights from cluster node); status: ok, can calculate with cluster.get_cluster_weight()
    Wxi = sum of weights from node i to all possible node in cluster x; status: ok, use cost_weight {}
'''
def cost_function(clus1, clus2, cost_weight):
    '''
    clus1: current cluster
    clus2: consider cluster
    cost_weight: list of weight from consider node to each ith cluster
    '''
    weight1 = clus1.get_cluster_weight()
    weight2 = clus2.get_cluster_weight()
    weight1 = weight1 if weight1 > 0 else 1 # when weight = 0 (cluster have only 1 node)
    weight2 = weight2 if weight2 > 0 else 1 # when weight = 0 (cluster have only 1 node)
    cost1 = cost_weight[clus1.id]
    cost2 = cost_weight[clus2.id]
    print(f'cost1: {cost1}, cost2: {cost2}, weight1: {weight1}, weight2: {weight2} ')
    #return 1/(weight1 - cost1) + 1/(weight2 + cost2) - 1/weight1 - 1/weight2
    return (1/(weight2 + cost2) - 1/weight2) - (1/weight1 - 1/(weight1 - cost1))

def second_cost(clus1, clus2, cost_weight, vertex):
    conduct1 = (vertex.weight - cost_weight[clus1.id])/ (vertex.weight)
    conduct2 = (vertex.weight - cost_weight[clus2.id])/ (vertex.weight)
    return conduct1 > conduct2
    pass

def K_Algorithm(graph):


    '''
        Arg: Graph, type: Graph object
        Return: Cluster, type: list of Cluster object
    '''

    N = len(graph.vertices)
    cluster = initialPartion(graph) # list of Clusters, but still only use graph.clusters
    changed = 0
    cost_weight = {} # a dict with keys are K cluster_id, values are weights from a node to cluster i  

    zero = {}
    for i in range(graph.K):
        zero[i] = 0
        cost_weight[i] = 0

    # Create a list of N random unique values from 0 to N-1
    sample = random.sample(range(0, N), N) # why we have to do randomly? maybe about the probability

    while True:
        changed = 0
        for i in sample:
            # changed = 0
            cost_weight.update(zero)
            bestdelta = float('inf')
            v = graph.vertices[i]
            new_clus_id = graph.vertex_in_cluster[v.id].id
            old_clus_id = graph.vertex_in_cluster[v.id].id

            for neighbor in v.neighbors: # loop all connections of i to calculate Wxi for all cluster x
                clus_id = graph.vertex_in_cluster[neighbor.id].id
                #cost_weight[clus_id] += 2*graph.weights[(neighbor.id, v.id)]
                cost_weight[clus_id] += graph.weights[(neighbor.id, v.id)]

            print(f'\nChoose vertex: {v.id}')
            print('Current cluster: ', graph.vertex_in_cluster[v.id].id)
            for cube in graph.vertex_in_cluster[v.id].vertices:
                print(cube.id)
            for j in range(graph.K):
                if graph.vertex_in_cluster[v.id].id != graph.cluster[j].id:  
                    print(f'\nconsidering clulster {graph.cluster[j].id} for vertex {v.id}: ')
                    for cube in graph.cluster[j].vertices:
                        print(cube.id)

                    cost = second_cost(graph.vertex_in_cluster[v.id], graph.cluster[j], cost_weight, v)
                    if cost:
                        print(f'vertex {v.id} changed to cluster {j}: ')
                        changed += 1
                        new_clus_id = j
                        # for cube in graph.cluster[old_clus_id].vertices:
                        #     print(cube.id)
                        graph.vertex_in_cluster[v.id] = graph.cluster[new_clus_id] 
                        # delete node v from previous cluster 
                        graph.cluster[old_clus_id].remove(v)
                        # add node v to current best cluster
                        graph.cluster[new_clus_id].add(v, graph)
                        old_clus_id = j
                        for cube in graph.cluster[j].vertices:
                            print(cube.id)
                    else:
                        print('vertex stay')

            #         cost = cost_function (graph.vertex_in_cluster[v.id], graph.cluster[j], cost_weight)
            #         if cost < bestdelta:
            #             print(f'vertex {v.id} changed to cluster {j} ')
            #             print(f'previouse cluster cost: {bestdelta} \nchanged cluster cost: {cost}\n')
            #             bestdelta = cost
            #             new_clus_id = j #graph.cluster[j].id
            #         else:
            #           print('vertex stay')
            # if new_clus_id != old_clus_id: # if node v change to better cluster
            #     changed += 1
            #     graph.vertex_in_cluster[v.id] = graph.cluster[new_clus_id] 
            #     # delete node v from previous cluster 
            #     graph.cluster[old_clus_id].remove(v)
            #     # add node v to current best cluster
            #     graph.cluster[new_clus_id].add(v, graph)

        if changed <= 0: # if all nodes dont change its cluster then finish
            break

    return cluster 