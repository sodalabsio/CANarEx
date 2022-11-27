import os
import joblib
import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics import silhouette_score

import faiss
from .cluster_config import cluster_config


def get_embeddings(sentences_list, path):
    sent_model = SentenceTransformer(cluster_config['s_name'],
                                     device=cluster_config['device'])

    if path and os.path.isfile(path):
        print('Loading embeddings from file path ..')
        embeddings = joblib.load(path)
    else:
        print('Creating embeddings ...')
        embeddings = sent_model.encode(sentences_list,
                                       batch_size=cluster_config['batch_size'],
                                       show_progress_bar=True,
                                       convert_to_numpy=True)

    if path:
        joblib.dump(embeddings, path)
    return embeddings


def fit_model(k, embeddings):
    print('Running Kmeans ...')
    clust_model = faiss.Kmeans(d=cluster_config['s_dimension'],
                               k=k,
                               niter=1000,
                               nredo=100,
                               verbose=False,
                               gpu=True,
                               seed=cluster_config['cluster_seed'])
    clust_model.train(embeddings.astype(np.float32))

    return (clust_model)


def _check_embeddings(embeddings, k):
    if issubclass(type(embeddings), list) or embeddings.shape[1] != cluster_config['s_dimension']:
        embeddings = _check_embeddings(embeddings, k)
    if embeddings.shape[0] < k:
        raise ValueError("Data points cannot be less than cluster count: {}, {}".format(embeddings.shape[0], k))
    return embeddings


def get_best_model_K(embeddings, k):
    """
    Fits the k-means model for given k
    @param embeddings: sentence embeddings
    @param k: cluster count
    @return: kmeans model, labels, distance from centroid, silhouette score
    """
    print('Clustering k {}'.format(k))
    embeddings = np.array(embeddings).reshape(-1, cluster_config['s_dimension'])
    model = fit_model(k, embeddings)

    D, I = model.index.search(embeddings, 1)
    labels = I.reshape(len(embeddings))
    dist = D.reshape(len(embeddings))
    score = -1

    if len(np.unique(labels)) > 1:
        score = silhouette_score(embeddings, labels)

    return model, labels, dist, score


def get_best_model(embeddings, krange):
    """
    Fits the k-means model for given range of k
    @param embeddings: sentence embeddings
    @param krange: list of cluster counts [k]
    @return: tuple(
        result: dataframe with columns [k,  silhouette score, distance from centroid, labels],
        models: kmeans models for different [k])
    """
    result = pd.DataFrame(columns=['k', 'score', 'dist', 'labels'])
    models = []

    for k in krange:
        _, labels, dist, score = get_best_model_K(embeddings, k)

        result = pd.concat([result, pd.DataFrame([[k, score, dist, labels]],
                                                 columns=['k', 'score', 'dist', 'labels'])], axis=0)
        # models.append(model)
    # TBD: returns null models
    return result, models


def write_clust_results(df, path, top=40):
    print('Writing cluster results...')

    df = df.groupby(['id', 'narrative']).head(1)
    with open(path, 'w') as f:
        for i in df.labels.unique():
            f.write('Labels {}\n'.format(i))
            f.write('\n')
            for l in df[df.labels == i].sort_values('dist').head(top)['narrative']:
                f.write(l)
                f.write('\n')
            f.write('*' * 20)
            f.write('\n')
            f.write('Last 20: \n')

            for l in df[df.labels == i].sort_values('dist').tail(top)['narrative']:
                f.write(l)
                f.write('\n')
            f.write('*' * 20)
            f.write('\n')
            f.write('\n')
