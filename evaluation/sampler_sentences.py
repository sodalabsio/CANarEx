import pandas as pd
import numpy as np
from scipy import stats

import random

import uuid

from itertools import count

ids = count(start=0, step=1)

from pathlib import Path
path = Path('synthetic_data')
path.mkdir(parents=True, exist_ok=True)

sent_dist = pd.read_json('sentences_dist.jsonl', lines = True)
approx_dist = stats.gaussian_kde(sent_dist.l) 
rng = np.random.default_rng(3005)

def get_no_sentences():
    no_sentences = 0
    while(no_sentences <= 0):
        no_sentences = int(approx_dist.resample(1, seed=rng)[0][0])
    return no_sentences

def create_document(sentences, noise):
    global ids
    
    doc_df = pd.DataFrame(columns=['id', 'sent_id', 'text', 'noise'])
    new_len = len(sentences)
    index_sentences = set(range(0, len(sentences)))

    while new_len > 0:
        no_sentences = get_no_sentences()
        if no_sentences > new_len:
            no_sentences = new_len
            
        noise_ss = []
        noise_indicator = []
        if noise is not None and len(noise) > 0:
            # max 0.1-0.3
            min_len = int(0.1 * no_sentences)
            max_len = int(0.3 * no_sentences)
            
            noise_ss = rng.choice(noise, 
                                  size=min_len if max_len == min_len
                                  else rng.integers(min_len, max_len))
            noise_indicator.extend([True] * len(noise_ss))
            no_sentences -= len(noise_ss)
            
        
              
        # ss_indexes = random.sample([*index_sentences], random.randint(1, new_len))
        # print('iter: {}, no_sent: {}, index: {}'.format(next(ids), no_sentences,
                                                       # len(index_sentences)))
        
        ss_indexes = rng.choice([*index_sentences], size=no_sentences)
        ss = [sentences[i] for i in ss_indexes]
        ss_indicator = [False] * len(ss)
        
        ss.extend(noise_ss)
        ss_indicator.extend(noise_indicator)
        
        df = pd.DataFrame({'text': ss, 'noise': ss_indicator})
        df['sent_id'] = list(range(0, df.shape[0]))
        # df['id'] = next(ids)
        df['id'] = str(uuid.uuid4())
        doc_df = pd.concat([doc_df.reset_index(drop=True), df], axis=0)

        index_sentences = index_sentences - set(ss_indexes)
        new_len = len(index_sentences)
    return doc_df


def sample_documents(sentences, noise=None, max_prop_noise=10):
    if len(sentences) == 0 and noise is not None:
        # Generate only noise documents
        ss = rng.choice(noise, size=1 if 1 == max_prop_noise
                                  else rng.integers(1, max_prop_noise))
        doc_df = create_document(ss, None)
        doc_df.loc[:, 'noise'] = True
    else:
        doc_df = create_document(sentences, noise)

    return (doc_df)


def create_synthetic_test_samples(clust_sentences, sample_prop, noise=None):
    # K is number of sentences
    synthetic_documents = None
    max_samples_prop = np.max(sample_prop)
    # max_prop_noise = 10#np.round(0.05 * max_samples_prop)
    # print(max_prop_noise)

    for idx, prop in enumerate(sample_prop):
        noise_level = rng.choice([0.1, 0.2, 0.3])
        max_prop_noise = np.round(noise_level * max_samples_prop)

        # create sentence samples
        clust_samples = list(rng.choice(clust_sentences, size=int(prop)))
        # create document from sentences
        documents = sample_documents(clust_samples, noise, max_prop_noise)
        documents.loc[:, 'time'] = idx

        if synthetic_documents is None:
            synthetic_documents = documents
        else:
            synthetic_documents = pd.concat([synthetic_documents.reset_index(drop=True), documents], axis=0)

    return (synthetic_documents)



if __name__ == '__main__':
    for col in ['true_text', 'coref_text']:
        # GPT-3 synthesised column [text]
        clust = pd.read_json('gpt3_train/synthetic_clusters_{}.jsonl'.format(col), lines=True)
        clust1 = clust[clust.true_labels == 18][['true_labels', 'text']].drop_duplicates().text.tolist()
        clust2 = clust[clust.true_labels == 24][['true_labels', 'text']].drop_duplicates().text.tolist()
        clust3 = clust[clust.true_labels == 63][['true_labels', 'text']].drop_duplicates().text.tolist()
    
        # GPT-3 synthesised column for noise [text]
        noise_sentences = pd.read_json('gpt3_train/synthetic_clusters_noise_{}.jsonl'.format(col), lines=True). \
            text.tolist()

        rng.shuffle(clust1)
        rng.shuffle(clust2)
        rng.shuffle(clust3)
        rng.shuffle(noise_sentences)

        # reduce re-sampling proportion
        # proportion height of the wave
        max_samples_prop = 6
        # sample_proportion = np.array(np.round(50 * max_samples_prop))

        # read signals: y_tri 	y_squ 	y_rnd
        # sample signal (traingle pulse, sq pulse, and random series)
        test = pd.read_csv('synth_series1_w0.3_m5.csv')

        # triangle
        print('Creating triange wave sample...')
        sample_proportion = np.array(np.round(test.y_tri * len(clust1) / max_samples_prop))
        # sample_proportion = np.array(np.round(test.y_tri * max_samples_prop))
        sample1 = create_synthetic_test_samples(clust1, sample_proportion, noise_sentences)
        sample1['synthetic_label'] = 1
        sample1.to_json('synthetic_data/tri_samples_{}.jsonl'.format(col),
                        lines=True, orient='records')

        # square
        print('Creating square wave sample...')
        sample_proportion = np.array(np.round(test.y_squ * len(clust2) / max_samples_prop))
        # sample_proportion = np.array(np.round(test.y_squ * max_samples_prop))
        sample2 = create_synthetic_test_samples(clust2, sample_proportion, noise_sentences)
        sample2['synthetic_label'] = 2
        sample2.to_json('synthetic_data/sq_samples_{}.jsonl'.format(col),
                        lines=True, orient='records')

        # round
        print('Creating random wave sample...')
        sample_proportion = np.array(np.round(test.y_rnd * len(clust3) / max_samples_prop))
        # sample_proportion = np.array(np.round(test.y_rnd * max_samples_prop))
        sample3 = create_synthetic_test_samples(clust3, sample_proportion, noise_sentences)
        sample3['synthetic_label'] = 3
        sample3.to_json('synthetic_data/rnd_samples_{}.jsonl'.format(col),
                        lines=True, orient='records')

        docs = pd.concat([sample1, sample2], axis=0)
        docs = pd.concat([docs, sample3], axis=0)
        docs = docs.reset_index(drop=True)
        docs['sentence_id'] = docs['id'] + '_' + docs.sent_id.astype(str)

        docs.to_json('synthetic_data/synthetic_test_data_{}.jsonl'.format(col),
                     lines=True, orient='records')
