import os
import pandas as pd
import json

from narratives import Narratives


def create_sentences_json(config):
    """
    Extracts co-referenced sentences into list of sentence dictionary
    sentences_json, [<sentence_id> : <sentence>]

    Saves the sentences_json in json format in the specified location
        @param:
            data: Co-referenced data. 

        @return:
            None
    """

    df = pd.read_json(config['data'], lines=True)
    sentences = df[['sentences', 'sentences_idx', 'id']]
    sentences = sentences.explode(['sentences', 'sentences_idx'])
    sentences['sentence_id'] = sentences.id + '_' + sentences.sentences_idx.astype(str)
    sentences = [sentences.sentence_id.tolist(), sentences.sentences.tolist()]

    sentences_json = [{"sentence": sent, 'id': id} for sent, id in zip(sentences[1], sentences[0])]

    with open(config['sentences_path'], 'w') as f:
        json.dump(sentences_json, f)

    config['sentences'] = sentences_json


def extract_narratives(config):
    """
    Creates narratives (final_narratives.jsonl) from the co-referenced sentences
    sentences_json, [<sentence_id> : <sentence>]

    Saves the narratives (final_narratives.jsonl) in the the specified location
        @param:
        config:  sentences_json[<sentence_id> : <sentence>] and output_folder

        @return:
            None
    """
    nr = Narratives(output_folder=config['output_folder'])
    nr.create_narratives(sentences_json=config['sentences'], save=True)


def merge_with_input(config):
    """
    Merge the narratives with original data
    
        @param:
        config:  data and output_folder

        @return:
            None
    """
    df = pd.read_json((config['data']), lines=True)
    df = df.drop(columns=['content', 'clusters', 'sentences', 'sentences_idx'],
                 errors='ignore')

    narratives = pd.read_json('{}/final_narratives.jsonl'.format(config['output_folder']), lines=True)
    # Merge keys from the original data
    narratives = pd.merge(narratives, df)

    output_narrative_path = '{}/{}_final_narratives.jsonl'.format(config['output_folder'],
                                                             os.path.basename(config['data']))
    narratives.to_json(output_narrative_path,
                       orient='records',
                       lines=True)


def run_factiva():
    config = {'data': 'data/factiva/_first_nations.jsonl',
              'sentences_path': 'data/factiva/coref_sentences.json',
              'sentences': None,
              'output_folder': 'data/factiva'}
    create_sentences_json(config)
    extract_narratives(config)
    merge_with_input(config)


def run_hansard():
    config = {'data': 'data/hansard/_first_nations.jsonl',
              'sentences_path': 'data/hansard/coref_sentences.json',
              'sentences': None,
              'output_folder': 'data/hansard'}
    create_sentences_json(config)
    extract_narratives(config)
    merge_with_input(config)


if __name__ == '__main__':
    # run_factiva()  # data not shared
    run_hansard()
