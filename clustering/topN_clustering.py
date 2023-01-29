from .sentence_clustering import get_embeddings, get_best_model_K, get_best_model
from .cluster_config import cluster_config
from narratives import read_jsonl_chunks, create_folder


import pandas as pd
import numpy as np
from functools import partial

def _get_main_doc_cluster(narratives_top2):
    # get size of the 2 clusters within each document
    topn = narratives_top2.groupby(['id', 'labels_2']).size().reset_index(name='cl')
    topn.loc[:, 'maxl'] = topn.groupby('id')['cl'].transform(max)
    # filter for sentences associated with the largest clusters
    # assumption main theme => larger cluster, secondary theme => smaller cluster
    topn = topn[topn.cl == topn.maxl].drop(columns='maxl')
    topn = pd.merge(narratives_top2, topn.drop(columns='cl'))

    return topn

def _get_top_n_narratives(topn, n):
    if n == 1:
        # pick top 1 sentence from the chosen cluster
        topn = topn.sort_values(['id', 'labels_2', 'dist_2'], ascending=True).groupby('id').head(1)
    else:
        # filter out duplicate narratives within the same document
        topn = topn.sort_values(['id', 'labels_2', 'dist_2'], ascending=True).groupby(['id', 'sent']).head(1)
        # pick top N sentences from the chosen cluster
        topn = topn.sort_values(['id', 'labels_2', 'dist_2'], ascending=True).groupby('id').head(n)

    topn = topn.reset_index(drop=True)
    return topn

def doc_level_clustering(doc, k):
    """
    @param doc: Document containing column embeddings that belong to single ID
    @type doc: DataFrame
    @param k: Number of clusters
    @type k: Int
    @return: Document with additional clustering results appended
    @rtype: DataFrame
    """
    if doc.shape[0] <= 2:
        doc.loc[:, 'labels_{}'.format(k)] = None
        doc.loc[:, 'dist_{}'.format(k)] = None
        return doc

    embeddings = doc.embeddings.tolist()
    _, labels, dist, score = get_best_model_K(embeddings, k)

    doc.loc[:, 'labels_{}'.format(k)] = labels
    doc.loc[:, 'dist_{}'.format(k)] = dist
    doc.loc[:, 'score_{}'.format(k)] = score
    return doc

def cluster_topN(narratives_top2):
    """
    Clusters using top N (from topn_range) results within each document.
    @param narratives_top2: Results from document level clustering
    @type narratives_top2: DataFrame
    @return: None
    @rtype: None
    """
    all_result = pd.DataFrame(columns=['topN', 'k', 'score', 'dist', 'labels'])

    for n in cluster_config['topn_range']:
        # filter for main theme clusters and n sentences within them
        topn = _get_main_doc_cluster(narratives_top2)
        topn = _get_top_n_narratives(topn, n)

        print('Processing embeddings for top{}'.format(n))
        # top_embed = get_embeddings(topn.sent.tolist())
        top_embed = topn.embeddings.tolist()

        print('Processing clustering for top{}'.format(n))
        result, models = get_best_model(top_embed,
                                             krange=cluster_config['topn_k'])

        result['topN'] = n

        all_result = pd.concat([all_result, result], axis=0)
    return all_result

def get_best_score(topn_df):
    """
    Returns the best N number of sentences for clustering based on silhouette score
    @param topn_df: Results of topn clustering
    @type topn_df: DataFrame (with columns score, topN, k)
    @return: Best top N number of sentences, number of k clusters, and labels of clustering
    @rtype: tuple
    """
    N = topn_df[topn_df.score == topn_df.score.max()].iloc[0].topN
    k = topn_df[topn_df.score == topn_df.score.max()].iloc[0].k

    topn_df = topn_df[topn_df.score == topn_df.score.max()]
    topn_df = topn_df.explode(['dist', 'labels']).reset_index(drop=True)

    print('Best model, topn: {}, k: {}'.format(N, k))
    return N, k, topn_df

def _merge_labels_narratives(k2, N, topN):
    # filter for main theme clusters and N sentences within them
    topn = _get_main_doc_cluster(k2)
    # get the number of narratives b/w 1-5 based on max value
    topn = _get_top_n_narratives(topn, N)

    topn = topn.reset_index(drop=True)
    topn = pd.concat([topn, topN], axis=1)
    return topn

def cluster():
    print('Running subset {} ...'.format(cluster_config['subset_prefix']))

    input_narrative = cluster_config['input_narrative']
    output_folder = cluster_config['output']
    subset = cluster_config['subset_prefix']

    # paths to save
    embed_path = '{}/{}_embeddings_topN_all.pk'.format(output_folder, subset)
    doc_cluster_path = '{}/{}_kmeans2_doc_level.pk'.format(output_folder, subset)
    topn_cluster_path = '{}/{}_topn_range.pk'.format(output_folder, subset)
    topn_cluster_result = partial('{output_folder}/{subset}_kmeans2_topn_{N}_final_k_{k}'.format,
                                 output_folder=output_folder, subset=subset)

    # create output directory
    create_folder(output_folder)

    # read narratives
    narratives = read_jsonl_chunks(input_narrative)
    narratives['seq'] = narratives.index

    # create embeddings of all sentences
    narratives.loc[:, 'embeddings'] = list(get_embeddings(narratives.sent.tolist(), embed_path))

    print('Clustering document level ...')
    # At document level run k=2 clusters
    narratives_top2 = narratives.groupby('id').apply(
        lambda x: doc_level_clustering(x, k=cluster_config['doc_k']))
    narratives_top2.to_pickle(doc_cluster_path)

    print('Clustering subset level ...')
    narratives_top2 = pd.read_pickle(doc_cluster_path)
    # cluster top N results from each document by varying N
    topn_result = cluster_topN(narratives_top2)
    topn_result.to_pickle(topn_cluster_path)

    # merge narratives with labels
    topn_result = pd.read_pickle(topn_cluster_path)
    N, k, topN = get_best_score(topn_result)
    topn = _merge_labels_narratives(narratives_top2, N, topN)

    # save
    topn_cluster_result = topn_cluster_result(N=N, k=k)
    topn.to_csv('{}.csv'.format(topn_cluster_result),
                index=False)
    topn.to_pickle('{}.pk'.format(topn_cluster_result))


if __name__ == '__main__':
    cluster()