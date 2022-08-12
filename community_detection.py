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


from input_data_club import load_data

from igraph import *
import igraph
from sklearn import metrics

def community_detection(g,algorithm):
    if algorithm == "multilevel":
        community = g.community_multilevel()
        return community

    
