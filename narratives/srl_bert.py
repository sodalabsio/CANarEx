import os
import pandas as pd
from allennlp_models import pretrained
from allennlp.common.util import lazy_groups_of

import spacy

spacy.prefer_gpu()

from tokenizers import BertWordPieceTokenizer
from tqdm import tqdm
from allennlp.common.tqdm import Tqdm

import json


class SRL_Bert:
    """
    SRL class using BERT
    """
    def __init__(self, model, 
                 selected_tags,
                 cuda_device, 
                 batch_size=100):
        self._model = model
        self._cuda_device = cuda_device
        self._batch_size = batch_size
        self._selected_tags = selected_tags

    def _truncate_sentences_bert(sentences, max_token_length=512):
        """
        Truncates sentences greater than 512
        @param:
            sentences: List of sentence dictionary [<sentence_id> : <sentence>]
            max_token_length: Truncate sentence length to max_token_length.

        @return:
            Truncated sentences.

        """

        tokenizer = BertWordPieceTokenizer("bert-base-uncased-vocab.txt", lowercase=True)
        print('Checking sentence length...')
        
        for idx in tqdm(range(0, len(sentences))):
            sent = sentences[idx]['sentence']
            tokenized_sequence = tokenizer.encode(sent)
            if len(tokenized_sequence.tokens) > max_token_length:
                tokenizer.enable_truncation(max_length=max_token_length)
                tokenized_sequence = tokenizer.encode(sent)
                sentences[idx]['sentence'] = tokenizer.decode(tokenized_sequence.ids)
                tokenizer.no_truncation()

        return sentences

    def __call__(self, sentences_json, output_folder, *args, **kwargs):
        """
        Runs allnlp SRL extraction
        @param:
            sentences: List of sentence dictionary [<sentence_id> : <sentence>]
            output_folder: Save the SRL results as 'srl_resul.json' in the folder location

        @return:
            SRL result from allennlp model

        """
        sentences_json = SRL_Bert._truncate_sentences_bert(sentences_json)

        print('Running SRL...')
        predictor = pretrained.load_predictor(self._model, cuda_device=self._cuda_device)

        srl_result = []
        for batch in Tqdm.tqdm(lazy_groups_of(sentences_json, self._batch_size)):
            try:
                res_batch = predictor.predict_batch_json(batch)
                for sentence, res in zip(batch, res_batch):
                    res['sentence_map'] = sentence
                    
                srl_result.extend(res_batch)
            except ValueError:
                raise

        if os.path.isdir(output_folder):
            with open('{}/srl_result.json'.format(output_folder), "w") as f:
                json.dump(srl_result, f)
        return srl_result

    def extract_arguments(self, srl_result, output_folder):
        """
        Extracts the selected list of argumants {ex: ARG1, ARG2, V, etc.} 
        @param:
            srl_result: Allennlp extracted SRL result with sentence mapping
            output_folder: Save the SRL extracted results as 'srl_extracted.jsonl' in 
            the folder location

        @return:
            SRL result from allennlp model

        """
        arg_values = []
        print('Running SRL argument extraction...')
        for res in tqdm(srl_result):
            word_list = res['words']
            sentence_map = res['sentence_map']
            for item in res['verbs']:
                arg_indices = {k: [] for k in self._selected_tags}
                arg_val = {k: [] for k in self._selected_tags}
                arg_val['verb'] = item['verb']
                for i, tag in enumerate(item['tags']):
                    if not tag.startswith('R-'):
                        for s in self._selected_tags:
                            if s in tag:
                                # roles[s] = roles[s] + ' ' + word_list[i]
                                arg_indices[s].append(i)
                                break

                for key, idx_list in arg_indices.items():
                    arg_val[key] = ' '.join([word_list[i] for i in idx_list])
                    arg_val[key + '_'] = (idx_list[0], idx_list[len(idx_list) -1]) \
                        if len(idx_list) > 0 else () 
                    # arg_val[key + '_'] = idx_list
                arg_val.update(sentence_map)
                arg_values.append(arg_val)

        arg_values = pd.DataFrame(arg_values)
        
        if os.path.isdir(output_folder):
            arg_values.to_json('{}/srl_extracted.jsonl'.format(output_folder),
                               lines = True, 
                               orient='records')
        # with open('{}/srl_extracted.jsonl'.format(output_folder), 'w', encoding='utf-8') as f:
        #             json.dump(arg_val, f)
        #             json.dump("\n", f)

        return arg_values