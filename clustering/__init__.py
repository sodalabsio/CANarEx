import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'textrank'))

from .topN_clustering import cluster as topn_cluster
from .textrank_clustering import cluster as tr_cluster