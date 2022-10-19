import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'narratives/clustering'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'narratives/textrank'))

from topN_clustering import cluster as topn_cluster
from textrank_clustering import cluster as tr_cluster

# config file : cluster_config.py
if __name__ == '__main__':
    topn_cluster()
    tr_cluster()