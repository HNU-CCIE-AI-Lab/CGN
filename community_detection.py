from __future__ import division
from __future__ import print_function

import time
import os
import networkx as nx
import matplotlib.pyplot as plt

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""

import tensorflow as tf
import numpy as np
import scipy.sparse as sp

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from gae.optimizer import OptimizerAE, OptimizerVAE
from gae.input_data_club import load_data
from gae.model import GCNModelAE, GCNModelVAE
from gae.preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple, mask_test_edges
from igraph import *
import igraph
from sklearn import metrics

def community_detection(g,algorithm):
    if algorithm == "multilevel":
        community = g.community_multilevel()
        return community

    if algorithm == "eigenvector":
        community = g.community_leading_eigenvector(clusters=50, arpack_options=None)
        # community = g.community_leading_eigenvector()
        return community
    if algorithm == "fastgreedy":
        community = g.community_fastgreedy()
        community = community.as_clustering()
        return community
    if algorithm == "infomap":
        community = g.community_infomap()
        return community
    if algorithm == "label_propagation":
        community = g.community_label_propagation()
        return community
    if algorithm == "edge_betweenness":
        # community = g.community_edge_betweenness(clusters=200,directed=False)
        community = g.community_edge_betweenness()
        community = community.as_clustering()
        return community
    if algorithm == "spinglass":
        community = g.community_spinglass(weights=None, spins=100, parupdate=False, start_temp=1, stop_temp=0.01, cool_fact=0.99, update_rule="config", gamma=1, implementation="orig", lambda_=1)
        # community = g.community_spinglass()
        return community
    if algorithm == "walktrap":
        community = g.community_walktrap()
        community = community.as_clustering()
        return community

# adj, features = load_data('dblp') # pubmed,cora,citeseer,karate,miserables,words,nets,power,dblp
# # print(adj)
#
# G = nx.from_scipy_sparse_matrix(adj) #nx的图
# adj = nx.to_numpy_array(G)
# adj = adj - np.diag(np.diag(adj))
#
#
# adj = nx.to_numpy_array(G) # igraph的图
# g = Graph.Adjacency(adj.astype(bool).tolist(),mode='undirected')
#
# community = community_detection(g,'infomap')
# print(community)