from __future__ import division
from __future__ import print_function

import time
import os
import networkx as nx
import matplotlib.pyplot as plt

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""

# import tensorflow as tf
import numpy as np
import scipy.sparse as sp

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from input_data_club import load_data

from igraph import *
import igraph
from sklearn import metrics
from community_detection import community_detection
from attack_edges import get_sorted_edges
import random
import itertools
import time



#args

datasets = "karate" # data
adj, features = load_data(datasets)
cd_alg = 'multilevel' # community detection algorithms

chrom_size = 1 # custom

group_size = 1 # custom
n = 0
pc = 0.7 # custom
pm = 0.01 # custom
generation = 1 # custom




G = nx.from_scipy_sparse_matrix(adj) 
adj = nx.to_numpy_array(G)
adj = adj - np.diag(np.diag(adj))
adj = nx.to_numpy_array(G) 
g = Graph.Adjacency(adj.astype(bool).tolist(),mode='undirected')

community1 = community_detection(g,cd_alg)





A = []
for i in community1:
    A.append(i)
for i in range(len(community1)):
    for j in A[i]:
        G.nodes[j]["commA"] = i
commA = nx.get_node_attributes(G,'commA')

nmiA = []
for i in range(nx.number_of_nodes(G)):
    nmiA.append(commA[i])






def edgesets(community1,n,G):
	edges = []
	edgesout = []
	for k in range(len(community1)):
		nodein = set(community1[k])
		nodeout = set(G.nodes) - nodein


		
		for i in itertools.product(nodein, nodeout):
	
			i = sorted(list(i))
			i = tuple(i)
			
			edges.append(i)

		edgesout = []
		for i in G.edges(list(nodeout)):
			i = sorted(list(i))
			i = tuple(i)
			edgesout.append(i)
		
		edgesin = list(G.edges - edgesout) 

		outedges = list(set(edges) - set(G.edges)) #

    return edgesin, outedges




edgesin, outedges = edgesets(community1,n,G)
def creatchrom(chrom_size):

    deln = random.choice(range(chrom_size))
    addn = chrom_size - deln

    popd = []
    popa = []
    pop = []
    for i in random.sample(edgesin,deln):
        popd.append([0,i])
    for i in random.sample(outedges,addn):
        popa.append([1,i])
    pop.extend(popa)
    pop.extend(popd)


    random.shuffle(pop)
    return pop

def getFistGroup(group_size):
    group = []
    for i in range(group_size):
        chrom = creatchrom(chrom_size)
        group.append(chrom)

    print(group)

    return group

def fitness(g,cd_alg):
    g.simplify(multiple=True, loops=True)  
    
    community2 = community_detection(g,cd_alg)

    B = []
    for i in community2:
        B.append(i)
    for i in range(len(community2)):
        for j in B[i]:
            G.nodes[j]["commB"] = i
    commB = nx.get_node_attributes(G, 'commB')

    nmiB = []
    for i in range(nx.number_of_nodes(G)):
        nmiB.append(commB[i])

    NMI = metrics.normalized_mutual_info_score(nmiA,nmiB)


    f = 1 - NMI





    return f,community2



def group_fit(group,cd_alg):

    gf = []
    for i in group:
        g = Graph.Adjacency(adj.astype(bool).tolist(),mode='undirected')

        for j in i:

            if j[0] > 0 :


                g.add_edges([j[1]])
            else:

                try:
                    g.delete_edges([j[1]])
                except ValueError:
                    continue

        temp,_ = fitness(g,cd_alg)
        gf.append(temp)

    return gf,g



def selection(group,gf):
    new_fit = []
    new_group = []
    total_fit = sum(gf)
    t = 0
    max_gf = max(gf)
    for i in range(len(group)):
        t += gf[i]/total_fit

        new_fit.append([group[i], t])

    for i in range(len(new_fit)):

        r = random.random()
        for j in range(len(new_fit)):

            if new_fit[j][1] > r:
                parents = j
                break
        new_group.append(new_fit[parents][0])

    return new_group


def crossover(group, gf, pc):
    parents_group = selection(group, gf) 

    group_len = len(parents_group)
    for i in range(0, group_len, 2):
        if(random.random() < pc): 
            cpoint = random.randint(0, len(parents_group[0])) 
            temp1 = []
            temp2 = []
            temp1.extend(parents_group[i][0:cpoint])
            temp1.extend(parents_group[i+1][cpoint:len(parents_group[i])])
            temp2.extend(parents_group[i+1][0:cpoint])
            temp2.extend(parents_group[i][cpoint:len(parents_group[i])])
            group[i] = temp1
            group[i+1] = temp2


def mutation(group, pm):
    px = len(group)
    py = len(group[0])

    for i in range(px): 
        if(random.random() < pm):
            mpoint = random.randint(0, py-1) 
            popd = []
            popa = []
            pop = []
            for j in random.sample(edgesin, 1):
                popd.append([0, j])
            for k in random.sample(outedges, 1):
                popa.append([1, k])
            pop = popa + popd

            rpop = random.sample(pop,1)

            group[i][mpoint] = rpop[0]


def best(group, gf):
    px = len(group)
    best_in = group[0]
    best_fit = gf[0]
    for i in range(1, px):
        if(gf[i] > best_fit):
            best_fit = gf[i]
            best_in = group[i]

    return [best_in, best_fit]


group = getFistGroup(group_size)

gf,_ = group_fit(group,cd_alg)
results = []

g1 = g

X = []
Y = []
NMIs = []
ENs = []
strat = time.clock()
for i in range(generation):
    if i >= 
        pm = 0.01
    if i > :
        pm = 0.01
    gf,g1 = group_fit(group,cd_alg)

    best_individual, best_fit = best(group, gf)
    results.append([i, best_fit, best_individual])
    crossover(group,gf,pc)
    mutation(group,pm)
    X.append(i)
    Y.append(1-results[i][1])
    NMIs.append(1 - results[i][1])
    ENs.append(chrom_size * (1-results[i][1]))
    min_nmi = min(NMIs)
    if min_nmi <= 0.9:
        break
end = time.clock()
rank = sorted(results, key=lambda x:x[1])







plt.plot(X, Y)

plt.show()


_,community2 = fitness(g1,cd_alg)






