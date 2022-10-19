import pandas as pd
import json
import pickle
import joblib

import numpy as np
from collections import deque
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def get_rows(time_rows):
    missing = set(range(0, 500)) - set(time_rows)
    return pd.DataFrame({'time': list(missing), 'l': [0]* len(missing)})


def get_best_cluster():
    print('loading scores from {}'.format(cpath))
    
    df = pd.read_pickle(cpath)
    df = df[df.score == df.score.max()]
    df = df.explode(['dist', 'labels'])
    df = df.reset_index(drop = True)

    return df


def get_best_cluster_textrank(path):
    fn = None
    for ratio in [0.05, 0.1, 0.2, 0.4]:
        cpath='{}/kmeans_{}.pkl'.format(path, ratio)
        df = pd.read_pickle(cpath)
        df['ratio'] = ratio
        if fn is not None:
            fn = pd.concat([fn, df], axis = 0).reset_index(drop = True)
        else:
            fn = df

    ratio = fn[fn.score == fn.score.max()].iloc[0].ratio
    k = fn[fn.score == fn.score.max()].iloc[0].k
    
    return ratio, k



def get_narratives():
    chunks = pd.read_json(narratives_path, 
                      lines=True, chunksize=10000)
    narratives = None
    for chunk in chunks:
        if narratives is None:
            narratives = chunk
        else:
            narratives = pd.concat([narratives, chunk], axis=0)
            
    return narratives
          
    
def create_narrative_clusters():
    df = get_best_cluster()
    narratives = get_narratives()
    final = pd.concat([narratives, df.reset_index(drop = True)],
                  axis = 1)
    
    final.to_json(result_path, lines = True, orient='records')
    
    
def get_original_synthetic_data():
    df = pd.read_json('{}/synthetic_test_data_{}.jsonl'.format('synthetic_data', ctype), 
                  lines = True, orient='records')
    
    return df


def get_mapped_narrative_clusters():
    df = pd.read_json(result_path, lines = True)
    return df


def get_mse(X, y):
    X = np.reshape(X, (500, 1))
    regr = LinearRegression().fit(X,y)
    y_pred = regr.predict(X)

    return {'intercept': regr.intercept_,
            'coef': regr.coef_,
            'mse': mean_squared_error(y_pred, y),
            'r_sq': r2_score(y_pred, y)}
    

def calculate_mse(synthetic_data, recovered_data):
    eval_df = pd.DataFrame(columns=['orig_clust', 'pred_clust', 'intercept', 'coef', 'mse', 'r_sq'])
    
    for orig_clust in [1, 2, 3]:
        # get original synthetic cluster sentence counts by time and noise
        orig_clust_count = synthetic_data[(synthetic_data.synthetic_label==orig_clust)]
        orig_clust_count = orig_clust_count[['time','id','noise', 'text', 'sentence_id']].drop_duplicates().\
            groupby(['time', 'noise']).size().reset_index(name='l')
        # remove noise
        orig_clust_count.loc[orig_clust_count.noise == True, 'l'] = 0

        # get original synthetic cluster sentence counts by time
        orig_clust_count = orig_clust_count.groupby(['time'])['l'].apply(sum).reset_index(name='l')
        orig_clust_count = orig_clust_count.sort_values('time')
        
        for new_label in recovered_data.labels.unique():
            test_clust_count = recovered_data[(recovered_data.labels==new_label)].copy()
            test_clust_count = test_clust_count[['time','id', 'noise', 'text', 'sentence_id']].\
                drop_duplicates().\
                groupby(['time', 'noise']).size().reset_index(name='l')
            test_clust_count = test_clust_count.groupby(['time'])['l'].apply(sum).reset_index(name='l')
            test_clust_count = pd.concat([test_clust_count.reset_index(drop = True), 
                                   get_rows(test_clust_count.time.tolist())], axis=0)
            test_clust_count = test_clust_count.reset_index(drop = True).sort_values('time')
            
            result = get_mse(orig_clust_count.l.to_numpy(), 
                             test_clust_count.l.to_numpy())
            result['orig_clust']= orig_clust
            result['pred_clust'] = new_label
            
            eval_df = pd.concat([eval_df, 
                                 pd.DataFrame(result)],
                                 axis = 0).reset_index(drop = True)
        
    return eval_df


def evaluate():
    print('Running evaluation ...')
    df = get_original_synthetic_data()
    final = get_mapped_narrative_clusters()
    
    eval_df = calculate_mse(df, final)
        
    print(eval_df[eval_df.mse == eval_df.groupby(['orig_clust'])['mse'].transform('min')])
    eval_df.to_csv(eval_path, index = False)


cpath = narratives_path = result_path = eval_path = None
ctype = None

def set_params(path):
    global cpath, narratives_path, result_path, eval_path
    
    cpath = '{}/kmeans_all.pkl'.format(path)
    narratives_path='{}/final_narratives.jsonl'.format(path)
    result_path = '{}/results.jsonl'.format(path)
    eval_path = '{}/eval_output.csv'.format(path)
    

def set_tr_params(path, ratio):
    global cpath, narratives_path, result_path, eval_path
    
    cpath='{}/kmeans_{}.pkl'.format(path, ratio)
    narratives_path='{}/narratives_summary_{}.jsonl'.format(path, ratio)
    result_path = '{}/results_{}.jsonl'.format(path, ratio)
    eval_path = '{}/eval_output_{}.csv'.format(path, ratio)
    

def evaluate_canarex():
    print('Evaluating CANarEx...')
    path='canarex_coref_text/'
    
    global ctype
    ctype='coref_text'
    
    set_params(path)
    
    print(cpath)
    print(narratives_path)
    print(result_path)
    print(eval_path)
    
    create_narrative_clusters()
    evaluate()
    
# no micro-narrative generation
def evaluate_canarex_with_no_splits():
    print('Evaluating CANarEx without micro-narrative generation...')
    path='canarex_coref_text_no_split_sentences/'
    
    global ctype
    ctype='coref_text'
    
    set_params(path)
    
    print(cpath)
    print(narratives_path)
    print(result_path)
    print(eval_path)
    
    create_narrative_clusters()
    evaluate()
    
# no coreference resolution
def evaluate_canarex_with_no_coreference():
    print('Evaluating CANarEx without co-reference resolution...')
    path='canarex_true_text/'
    
    global ctype
    ctype='true_text'
    
    set_params(path)
    
    print(cpath)
    print(narratives_path)
    print(result_path)
    print(eval_path)
    
    create_narrative_clusters()
    evaluate()
    
    
# no micro-narrative generation
# no co-reference
def evaluate_canarex_with_no_coreference_no_splits():
    print('Evaluating CANarEx without co-reference resolution and micro-narrative generation...')
    path='canarex_true_text_no_split_sentences/'
    
    global ctype
    ctype='true_text'
    
    set_params(path)
    
    print(cpath)
    print(narratives_path)
    print(result_path)
    print(eval_path)
    
    create_narrative_clusters()
    evaluate()
    
    
def eval_textrank():
    print('Evaluating CANarEx textrank...')
    path='canarex_coref_text/subset/'
    
    global ctype
    ctype='coref_text'
    
    ratio, k = get_best_cluster_textrank(path)
    print('Best ratio {}, K {}...'.format(ratio, k))
    
    set_tr_params(path, ratio)
    
    print(cpath)
    print(narratives_path)
    print(result_path)
    print(eval_path)
    
    create_narrative_clusters()
    evaluate()
    
    
if __name__ == '__main__':
    evaluate_canarex()
    # evaluate_canarex_with_no_splits()
    # evaluate_canarex_with_no_coreference()
    # evaluate_canarex_with_no_coreference_no_splits()
    # eval_textrank()
    