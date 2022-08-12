# encoding: utf-8

import numpy as np
import sys
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from igraph import *
from sklearn.preprocessing import normalize
import importlib
import matplotlib.pyplot as plt

importlib.reload(sys)

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_data(dataset):
    # load the data: x, tx, allx, graph
    path = './datasets/'
    if dataset == 'cora' or dataset == 'pubmed' or dataset == 'citeseer':
        names = ['x', 'tx', 'allx', 'graph']
        objects = []
        for i in range(len(names)):
            with open("data/ind.{}.{}".format(dataset, names[i]), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))
        x, tx, allx, graph = tuple(objects)
        test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset))
        test_idx_range = np.sort(test_idx_reorder)


        if dataset == 'citeseer':
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range-min(test_idx_range), :] = tx
            tx = tx_extended



        # features = sp.vstack((allx, tx)).tolil()
        # # features[test_idx_reorder, :] = features[test_idx_range, :]

        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
        g = nx.from_scipy_sparse_matrix(adj)
        features = sp.vstack((nx.to_scipy_sparse_matrix(g))).tolil()
    if dataset == 'karate':
        # # 换成俱乐部数据 0.998
        g = nx.karate_club_graph()
        features = sp.vstack((nx.to_scipy_sparse_matrix(g))).tolil()
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(nx.to_dict_of_lists(g)))
    if dataset == 'miserables':
        # 《悲惨世界》中人物的共现网络 0.992
        g = nx.les_miserables_graph()
        features = sp.vstack((nx.to_scipy_sparse_matrix(g))).tolil()
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(nx.to_dict_of_lists(g)))

    # # 佛罗伦萨族图
    if dataset == 'family':
        g = nx.florentine_families_graph()
        features = sp.vstack((nx.to_scipy_sparse_matrix(g))).tolil()
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(nx.to_dict_of_lists(g)))

    # 戴维斯南方妇女社交网络
    if dataset == 'women':
        g = nx.davis_southern_women_graph()
        features = sp.vstack((nx.to_scipy_sparse_matrix(g))).tolil()
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(nx.to_dict_of_lists(g)))
    if dataset == 'words':
        # words数据集论文里的 0.93
        g = nx.readwrite.read_gml(path=path+'adjnoun.gml')
        features = sp.vstack((nx.to_scipy_sparse_matrix(g))).tolil()
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(nx.to_dict_of_lists(g)))
    if dataset == 'nets':

        # nets数据集论文里的 0.93
        g = nx.readwrite.read_gml(path=path+'netscience.gml')
        features = sp.vstack((nx.to_scipy_sparse_matrix(g))).tolil()
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(nx.to_dict_of_lists(g)))
    if dataset == 'power':

        # power
        g = nx.readwrite.read_gml(path=path+'power.gml', label='id')
        # g = nx.readwrite.read_gml(path=path + 'power.gml', label='id')
        features = sp.vstack((nx.to_scipy_sparse_matrix(g))).tolil()
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(nx.to_dict_of_lists(g)))

    # # DBLP
    # g = nx.readwrite.read_adjlist('./datasets/com-dblp.top5000.cmty.txt.gz',delimiter='\t')
    # features = sp.vstack((nx.to_scipy_sparse_matrix(g))).tolil()
    # adj = nx.adjacency_matrix(nx.from_dict_of_lists(nx.to_dict_of_lists(g)))

    # # facebook
    # g = nx.readwrite.read_edgelist(path=path+'musae_facebook_edges.csv', delimiter=',')
    # features = sp.vstack((nx.to_scipy_sparse_matrix(g))).tolil()
    # adj = nx.adjacency_matrix(nx.from_dict_of_lists(nx.to_dict_of_lists(g)))
    if dataset == 'dblp':
        dataset = "dblp"
        # Load data
        if dataset == "dblp":
            adj = sp.load_npz("../../gae/data/dblp/dblp_medium_adj.npz")
            features = np.load("../../gae/data/dblp/dblp_medium_features.npy")
            features_normlize = normalize(features, axis=0, norm='max')
            features = sp.csr_matrix(features_normlize)
            target_list = np.load("../../gae/data/dblp/dblp_medium_label.npy")
        elif dataset == "finance":
            adj = sp.load_npz('./data/finance/Finance_large_adj.npz')
            features = np.load("data/finance/Finance_large_features.npy")
            features_normlize = normalize(features, axis=0, norm='max')
            features = sp.csr_matrix(features_normlize)
            target_list =  np.load("data/finance/Finance_large_label.npy")



    return adj, features



# adj, features = load_data('cora')
# #
# g = nx.from_scipy_sparse_matrix(adj)
# adj = nx.to_numpy_array(g)
# adj = adj - np.diag(np.diag(adj))
# print(adj)
#
# ax = plt.gca()
# cax = plt.imshow(adj, cmap='viridis')
# cbar = plt.colorbar(cax, extend='both', drawedges = False)
# cbar.set_label('Intensity',size=36, weight =  'bold')
# cbar.ax.tick_params( labelsize=18 )
# cbar.minorticks_on()
# ticks=np.arange(0,adj.shape[0],1)
# plt.xticks(ticks, fontsize=8, fontweight = 'bold')
# ax.set_xticklabels(ticks)
# plt.yticks(ticks, fontsize=8, fontweight = 'bold')
# ax.set_yticklabels(ticks)
# # plt.matshow(adj)
# plt.savefig('karate_club_adj.png', dpi = 300)
# plt.close()
#
#
# plt.show()

# adj, features = load_data('citt')
#
# print(adj)
# print(features)



# adj = nx.to_numpy_array(g)
# g = Graph.Adjacency(adj.astype(bool).tolist(),mode='undirected')
#
# community = g.community_multilevel()
# communitys = g.community_multilevel()
#
# summary(g)
#
# plt.subplot(211)
# # layout = g.layout('kk') # 使用常见的Kamada-Kawai布局
# visual_style = {}
# # visual_style["edge_color"] = "black" # 设置边的颜色
# visual_style["vertex_size"] = 10 # 节点大小设置
# # visual_style["vertex_color"] = 'rgb(218.89, 232.93, 245.96)'# 节点颜色设置
# # visual_style["layout"] = layout # 设置布局模板
# # visual_style["bbox"] = (300, 300) # 设置大小
# # visual_style["margin"] = 20 # 设置图形离边缘的距离
# visual_style["edge_curved"] = False # 指定边为弯曲或者直边
# # visual_style["vertex_label"] = range(g.vcount())
# plot(community,labels='id',**visual_style)



