import pandas as pd

from narratives import read_jsonl_chunks, create_folder
from .cluster_config import cluster_config

from .sentence_clustering import get_embeddings, get_best_model, write_clust_results
from .textrank_summarizer import create_textrank_summary, create_narrative_summary



def _merge_labels_narratives(ratio, k):
    # Read best cluster
    fn = pd.read_pickle('{}/{}_kmeans_tr_{}.pk'.format(output, subset, ratio))
    fn = fn[fn.k == k].explode(['dist', 'labels']).reset_index(drop=True)

    # Read summarised narratives
    narrative_tr_output = '{}/{}_tr_{}_final_narratives.jsonl'.format(output,
                                                                      subset,
                                                                      ratio)
    tr = pd.read_json(narrative_tr_output, lines=True).reset_index(drop=True)

    # Merge labels with narratives
    tr = pd.concat([tr, fn], axis=1)

    # Save
    tr_cluster_result = '{}/{}_kmeans_tr_{}_final_{}.csv'.format(output, subset, ratio, k)
    tr.to_csv('{}.csv'.format(tr_cluster_result),index=False)
    tr.to_pickle('{}.pk'.format(tr_cluster_result))
    return tr


def get_best_cluster():
    fn = None
    for ratio in cluster_config['tr_ratios']:
        temp = pd.read_pickle('{}/{}_kmeans_tr_{}.pk'.format(output, subset, ratio))
        temp = temp[['k', 'score']].reset_index(drop=True)
        temp.loc[:, 'ratio'] = ratio

        if fn is None:
            fn = temp
        else:
            fn = pd.concat([fn, temp], axis=0).reset_index(drop=True)

    # Pick the ratio and K according to 'best ratio'
    ratio = fn[fn.score == fn.score.max()].iloc[0].ratio
    k = fn[fn.score == fn.score.max()].iloc[0].k
    print('Best model, ratio: {}, k: {}'.format(ratio, k))

    return ratio, k


def textrank_clustering():
    for ratio in cluster_config['tr_ratios']:
        print('Running cluustering for ratio {} ...'.format(ratio))
        narrative_tr_output = '{}/{}_tr_{}_final_narratives.jsonl'.format(output,
                                                                          subset,
                                                                          ratio)

        narratives = read_jsonl_chunks(narrative_tr_output)
        narratives['seq'] = narratives.index
        embeddings = get_embeddings(narratives.sent.tolist(), path=None)

        print('Clustering document level ...')
        result, _ = get_best_model(embeddings, krange=cluster_config['tr_k'])
        result.to_pickle('{}/{}_kmeans_tr_{}.pk'.format(output, subset, ratio))


def _summarise_narratives():
    # read input data and generate textrank summaries
    for ratio in cluster_config['tr_ratios']:
        coref_df = pd.read_json(cluster_config['data'], lines=True)
        tr_output = '{}/{}_tr_summary_{}.jsonl'.format(output,
                                                       subset,
                                                       ratio)
        create_textrank_summary(coref_df, ratio, tr_output)

        narratives = read_jsonl_chunks(cluster_config['input_narrative'])
        narrative_tr_output = '{}/{}_tr_{}_final_narratives.jsonl'.format(output,
                                                      subset,
                                                      ratio)
        create_narrative_summary(narratives, tr_output, narrative_tr_output)


def cluster():
    # create output directory
    create_folder(output)

    # Filter narratives by textrank
    _summarise_narratives()

    textrank_clustering()
    ratio, k = get_best_cluster()

    # Merge labels with narratives
    df = _merge_labels_narratives(ratio, k)

    # write_clust_results(df, '{}/{}_kmeans_tr_final_{}_{}.txt'.format(output, subset, ratio, k),
    #                     top=100)


output = cluster_config['output']
subset = cluster_config['subset_prefix']

if __name__ == '__main__':
    cluster()
