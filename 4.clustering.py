import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'narratives'))

from clustering import topn_cluster
from clustering import tr_cluster

# config file : cluster_config.py
if __name__ == '__main__':
    topn_cluster()
    tr_cluster()