import os
# Change current working directory
os.chdir('coref')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
sys.path.append('.')

from pathlib import Path
from tqdm import tqdm


import uuid
import json
import re
import copy
import numpy as np
from operator import itemgetter

# import spacy

import tensorflow as tf
from bert import tokenization
import util
import pyhocon

import sacremoses
 

# References:
# https://github.com/mandarjoshi90/coref
# https://colab.research.google.com/drive/1SlERO9Uc9541qv6yH26LJz5IM9j7YVra#scrollTo=q0jLV2_sHC7e

_genre = "nw"
# The Ontonotesf data sources
# ["bc": broadcast conversation, "bn": broadcast news, "mz": magazine
# "nw": newswire, "pt": Bible text, "tc": telephone conversation, "wb": web data]

# The fine-tuned model: [bert_base, spanbert_base, bert_large spanbert_large]
_model_name = "spanbert_base"

os.environ['data_dir'] = "."



# The fine-tuned model: [bert_base, spanbert_base, bert_large spanbert_large]
def _get_max_segment():
    max_segment = None
    for line in open('experiments.conf'):
        if line.startswith(_model_name):
            max_segment = True
        elif line.strip().startswith("max_segment_len"):
            if max_segment:
                max_segment = int(line.strip().split()[-1])
                break
    return max_segment

# reads a jsonl file as a list
def _read_jsonl_list(path):
    with open(path, 'r', encoding='utf-8') as f:
        contents = f.read()
    corpus_df = [json.loads(str(item)) for item in contents.strip().split('\n')]

    return corpus_df

def _get_batch(text_record, batch_size):
    text_id = text_record['id']
    text_content = text_record['content']

    if batch_size == 1:
        print('dummy')
        yield from [{'id': text_id,
                     'sentences': text_content['sentences'],
                     'sentences_idx': text_content['sentences_idx']}]
    contents = []
    total_len = len(text_content['sentences'])

    batch_len = 0
    remaining = total_len
    while batch_len < total_len:
        max_len = remaining if remaining <= batch_size else batch_size
        contents.append({'id': text_id,
                         'sentences': text_content['sentences'][batch_len:(batch_len + max_len)],
                         'sentences_idx': text_content['sentences_idx'][batch_len:(batch_len + max_len)],
                         })
        batch_len += max_len
        remaining = total_len - batch_len
    yield from contents


def _merge_dictionary_lists(dict1, dict2, sort_key):
    # Merge keys from the original data
    dict1.sort(key=itemgetter(sort_key))
    dict2.sort(key=itemgetter(sort_key))

    final_dict = []
    # Keep key/values from dict1
    for u, v in zip(dict1, dict2):
        if u[sort_key] != v[sort_key]:
            raise('Invalid mapping of keys from input to final data')
        final_dict.append({**u, **v})

    return final_dict

def _convert_mention(comb_text, output, mention):
    start = output['subtoken_map'][mention[0]]
    end = output['subtoken_map'][mention[1]] + 1
    nmention = (start, end)
    mtext = ''.join(' '.join(comb_text[mention[0]:mention[1] + 1]).split(" ##"))
    return (mtext)


def _replace_mention(comb_text, replace_text, output, mention):
    start = output['subtoken_map'][mention[0]]
    end = output['subtoken_map'][mention[1]] + 1
    # nmention = (start, end)
    for i in range(mention[0], mention[1] + 1):
        comb_text[i] = ''
    comb_text[mention[0]] = replace_text
    return (comb_text)


def _clean_text(text):
    # remove extra spaces b/w punctuations
    text = re.sub(r"(\d+[\,\.])(?:\s+?)(\d+)", r"\1\2", text)
    text = re.sub(r"\s*'\s*", "'", text)
    text = re.sub(r"\s*-\s*", "-", text)
    return (text)


def _remove_separators(text, split=False):
    """
    Removes separators from text and returns as list based on split param 

    @param text: Detokenized BERT document, 
           split: If true, return sentences else return text as document
    @return: Cleaned text (or as sentences with ids)
    """

    ids = None
    if split:
        text = re.sub('\s?\[CLS\]\s?', '', text)
        text = [t.strip() for t in text.split("[SEP]") if t.strip() and bool(re.search('\w', text))]
        ids = list(range(1, len(text) + 1))
    else:
        text = re.sub('\s?\[CLS\]\s?', '', text)
        text = re.sub('\s?\[SEP\]\s?', '', text)

    return (text, ids)



# todo: pending merge with tokenize_data
def tokenize_data_batches(input_file, output_file, batch_size=500, trunc=True):
    """
    Tokennizes the data for BERT sentences input

    @param input: jsonl file with split sentences per article
    @param output: BERT tokenized data
    @return:
    """
    print('Tokenizing data ...')
    max_segment = _get_max_segment()

    with open(input_file, 'r', encoding='utf-8') as f:
        contents = f.read()
    corpus_df = [json.loads(str(item)) for item in contents.strip().split('\n')]

    tokenizer = tokenization.FullTokenizer(vocab_file="cased_config_vocab/vocab.txt", do_lower_case=False)
    sentences_json_list = []
    for text_record in tqdm(corpus_df):
        contents = _get_batch(text_record, batch_size)

        # run batches
        for content in contents:
            for sent_num, sentence in zip(content['sentences_idx'], content['sentences']):
                data = {'doc_key': 'nw', 'id': content['id'], 'sentences': [["[CLS]"]], 'speakers': [["[SPL]"]],
                        'clusters': [], 'sentence_map': [], 'subtoken_map': [0]}
                subtoken_num = 0

                raw_tokens = [i for i in sentence.split() if i.strip()]
                if len(raw_tokens) > 1:
                    tokens = tokenizer.tokenize(sentence)

                    # Truncate to max token length given by model
                    if trunc and len(tokens) > (max_segment - 1):
                        tokens = tokens[0:(max_segment - 2)]

                    # Create new batch if total length of sentences > max token allowed (
                    if len(tokens) + len(data['sentences'][-1]) >= (max_segment - 1):
                        data['sentences'][-1].append("[SEP]")
                        data['sentence_map'].append(sent_num - 1)

                        data['sentences'].append(["[CLS]"])
                        data['sentence_map'].append(sent_num)

                        data['speakers'][-1].append("[SPL]")
                        data['subtoken_map'].append(subtoken_num - 1)

                        data['speakers'].append(["[SPL]"])
                        data['subtoken_map'].append(subtoken_num)

                    ctoken = raw_tokens[0]
                    cpos = 0
                    for token in tokens:
                        # assign [CLS] with sent_num
                        if len(data['sentence_map']) == 0:
                            data['sentence_map'].append(sent_num)

                        if not trunc and len(data['sentences'][-1]) >= (max_segment - 1):
                            data['sentences'][-1].append("[SEP]")
                            data['sentence_map'].append(sent_num)

                            data['sentences'].append(["[CLS]"])
                            data['sentence_map'].append(sent_num)

                            data['speakers'][-1].append("[SPL]")
                            data['subtoken_map'].append(subtoken_num)

                            data['speakers'].append(["[SPL]"])
                            data['subtoken_map'].append(subtoken_num)

                        data['sentences'][-1].append(token)
                        if 'speaker' in text_record.keys():
                            data['speakers'][-1].append(text_record['speaker'])
                        else:
                            data['speakers'][-1].append('-')

                        data['sentence_map'].append(sent_num)
                        data['subtoken_map'].append(subtoken_num)

                        if token.startswith("##"):
                            token = token[2:]
                        if len(ctoken) == len(token):
                            subtoken_num += 1
                            cpos += 1
                            if cpos < len(raw_tokens):
                                ctoken = raw_tokens[cpos]
                        else:
                            ctoken = ctoken[len(token):]

            data['sentences'][-1].append("[SEP]")
            data['speakers'][-1].append("[SPL]")
            data['sentence_map'].append(sent_num)
            data['subtoken_map'].append(subtoken_num - 1)

            sentences_json_list.append(data)

    with open(output_file, 'w') as f:
        json.dump(sentences_json_list, f, sort_keys=True)


def tokenize_data(input_file, output_file, trunc=True):
    """
    Tokenises the data for BERT sentences input

    @param input_file: jsonl file with split sentences per article
    @param output: BERT tokenized data
    @return: None

    Writes the tokens generated to the output folder
    """
    print('Tokenizing data ...')
    max_segment = _get_max_segment()
    corpus_list = _read_jsonl_list(input_file)

    tokenizer = tokenization.FullTokenizer(vocab_file="cased_config_vocab/vocab.txt", do_lower_case=False)
    tokens_json_list = []

    # for debugging
    # truncated_ids = []
    for text_record in tqdm(corpus_list):
        if 'id' not in text_record.keys():
            text_record['id'] = str(uuid.uuid4())

        data = {'doc_key': _genre, 'id': text_record['id'], 'sentences': [["[CLS]"]], 'speakers': [["[SPL]"]],
                'clusters': [], 'sentence_map': [], 'subtoken_map': [0]}
        subtoken_num = 0

        content = text_record['content']
        # to do: batch this: currently 2 records being truncated
        # memory issues
        # if len(content['sentences']) > 500:
        #     print('Record {} is being truncated'.format(text_record['id']))
        #     content['sentences'] = content['sentences'][0:500]
        #     content['sentences_idx'] = content['sentences_idx'][0:500]
        for sent_num, sentence in zip(content['sentences_idx'], content['sentences']):
            raw_tokens = [i for i in sentence.split() if i.strip()]
            if len(raw_tokens) > 1:
                tokens = tokenizer.tokenize(sentence)

                # Truncate to max token length given by model
                if trunc and len(tokens) > (max_segment - 1):
                    # debugging
                    # truncated_ids.append({'id': text_record['id'], 'sent_id':sent_num})
                    tokens = tokens[0:(max_segment - 2)]

                # Create new batch if total length of sentences > max token allowed
                if len(tokens) + len(data['sentences'][-1]) >= (max_segment - 1):
                    data['sentences'][-1].append("[SEP]")
                    data['sentence_map'].append(sent_num - 1)

                    data['sentences'].append(["[CLS]"])
                    data['sentence_map'].append(sent_num)

                    data['speakers'][-1].append("[SPL]")
                    data['subtoken_map'].append(subtoken_num - 1)

                    data['speakers'].append(["[SPL]"])
                    data['subtoken_map'].append(subtoken_num)

                ctoken = raw_tokens[0]
                cpos = 0
                for token in tokens:
                    # assign [CLS] with sent_num
                    if len(data['sentence_map']) == 0:
                        data['sentence_map'].append(sent_num)

                    if not trunc and len(data['sentences'][-1]) >= (max_segment - 1):
                        data['sentences'][-1].append("[SEP]")
                        data['sentence_map'].append(sent_num)

                        data['sentences'].append(["[CLS]"])
                        data['sentence_map'].append(sent_num)

                        data['speakers'][-1].append("[SPL]")
                        data['subtoken_map'].append(subtoken_num)

                        data['speakers'].append(["[SPL]"])
                        data['subtoken_map'].append(subtoken_num)

                    data['sentences'][-1].append(token)
                    if 'speaker' in text_record.keys():
                        data['speakers'][-1].append(text_record['speaker'])
                    else:
                        data['speakers'][-1].append('-')

                    data['sentence_map'].append(sent_num)
                    data['subtoken_map'].append(subtoken_num)

                    if token.startswith("##"):
                        token = token[2:]
                    if len(ctoken) == len(token):
                        subtoken_num += 1
                        cpos += 1
                        if cpos < len(raw_tokens):
                            ctoken = raw_tokens[cpos]
                    else:
                        ctoken = ctoken[len(token):]

        data['sentences'][-1].append("[SEP]")
        data['speakers'][-1].append("[SPL]")
        data['sentence_map'].append(sent_num)
        data['subtoken_map'].append(subtoken_num - 1)

        tokens_json_list.append(data)

    with open(output_file, 'w') as f:
        json.dump(tokens_json_list, f, sort_keys=True)

    # with open('truncated_ids.txt', 'w') as f:
    #     for tid in truncated_ids:
    #         f.write(str(tid))
    #         f.write("\n")


def run_coref(input_file, output_file):
    """
    Runs coreference resolution using BERT
    @param input_file: Tokenised BERT document,
    output_file: File path for co-references extracted
    
    @return: None
    Writes co-references to output file
    """

    print('Running BERT coref on data ...')
    config = pyhocon.ConfigFactory.parse_file("experiments.conf")[_model_name]
    config["log_dir"] = util.mkdirs(os.path.join(config["log_root"], _model_name))

    model = util.get_model(config)
    saver = tf.train.Saver()

    with tf.Session() as session:
        model.restore(session)

        with open(output_file, "a") as output_f:
            with open(input_file) as f:
                for line in f.readlines():
                    examples = json.loads(line)
                    for example in tqdm(examples):
                        # try:
                        tensorized_example = model.tensorize_example(example, is_training=False)
                        feed_dict = {i: t for i, t in zip(model.input_tensors, tensorized_example)}
                        _, _, _, top_span_starts, top_span_ends, top_antecedents, top_antecedent_scores = session.run(
                            model.predictions, feed_dict=feed_dict)
                        predicted_antecedents = model.get_predicted_antecedents(top_antecedents, top_antecedent_scores)
                        new_example = copy.deepcopy(example)
                        new_example["predicted_clusters"], _ = model.get_predicted_clusters(top_span_starts,
                                                                                            top_span_ends,
                                                                                            predicted_antecedents)
                        new_example["top_spans"] = list(
                            zip((int(i) for i in top_span_starts), (int(i) for i in top_span_ends)))
                        # new_example['head_scores'] = []

                        output_f.write(json.dumps(new_example))
                        output_f.write("\n")
                        # except BaseException as err:
                        #     print('Unexpected {}, {}'.format(err, type(err)))
                        #     print('Error {}'.format(err))
                        #     raise


def resolve_coref(corpus_list, coref_path, output_path):
    """
    Resolves the coreferences and merges with input data

    @param corpus_list: input data as list (original corpus),
           coref_path: co-referenced input data,
           output_path: final resolved coreference output file path
    @return: None

    Writes resovled co-references to output file path
    """

    print('Cleaning references ...')
    data = _read_jsonl_list(coref_path)

    formatted_data = []
    # merge the tokens created using BERT
    detok = sacremoses.MosesDetokenizer('en')

    for text_record in tqdm(data):
        dat = {'id': text_record['id'], 'content': None, 'sentences': [], 'sentences_idx': []}
        comb_text = [word for sentence in text_record['sentences'] for word in sentence]
        all_mapped = []
        for cluster in text_record['predicted_clusters']:
            replace_text = ''
            mapped = []
            for idx, mention in enumerate(cluster):
                if idx == 0:
                    # Use first mention as cluster head, i.e. replacement text
                    replace_text = _convert_mention(comb_text, text_record, mention)
                    mapped.append(replace_text)
                else:
                    # Merge all subsequent mentions with the cluster head
                    rt = _convert_mention(comb_text, text_record, mention)
                    mapped.append(rt)
                    comb_text = _replace_mention(comb_text, replace_text, text_record, mention)
            all_mapped.append(mapped)

        dat['clusters'] = all_mapped

        sents_id = np.array(text_record['sentence_map'])
        for sid in np.unique(sents_id):
            final_text = itemgetter(*np.where(sents_id == sid)[0])(comb_text)
            final_text = ''.join(' '.join(final_text).split(" ##"))
            # if len(text_record['speakers'][0]) > 1:
            #     # TODO: speaker replace?
            #     final_text = re.sub(r"\b(I|My|my|me)\b", text_record['speakers'][0][1], final_text)

            # Get fully formed sentences from BERT tokens
            final_text = detok.detokenize(final_text.split(" "))
            # Remove spaces between numericals separated by comma
            final_text = _clean_text(final_text)
            # Remove [SEP], [CLS] tokens
            final_text, _ = _remove_separators(final_text)
            # Get coreferenced content as sentences
            dat['sentences'].append(final_text)
            dat['sentences_idx'].append(int(sid))

            
        # Get coreferenced content as a paragraph
        dat['content'] = ' '.join(dat['sentences'])
        formatted_data.append(dat)

    # Merge keys from the original data
    print('Merging keys...')
    formatted_data = _merge_dictionary_lists(corpus_list, formatted_data, 'id')

    with open(output_path, "w") as f:
        for record in formatted_data:
            json.dump(record, f)
            f.write('\n')


def execute(config):
    """
    Reads the input file, does coref-resolution and writes the
    co-referenced results to a given output file in the given output directory

    @param config: Input/output and model information
    @return: None
    """

    output = config['output']
    # create temp folder for intermittent data
    path = Path('{}/temp'.format(output))
    path.mkdir(parents=True, exist_ok=True)

    # read split sentences of original text
    corpus_list = _read_jsonl_list(path='{}/temp/orig_sentences.jsonl'.format(output))
    
    # Tokenize data
    tokenize_data('{}/temp/orig_sentences.jsonl'.format(output), '{}/temp/sentences_bert_tokens.json'.format(output))

    # if running in batch mode (very large documents)
    # tokenize_data_batches('{}/temp/sentences_bert.jsonl'.format(output),
    #                       '{}/temp/sentences_bert_tokens.json'.format(output))

    # run coref 
    run_coref('{}/temp/sentences_bert_tokens.json'.format(output), '{}/temp/sentences_bert_coref.json'.format(output))

    # add _ to original file name
    output_path = '{}/_{}'.format(output, os.path.basename(config['data']))

    # resolve co-reference
    resolve_coref(corpus_list, '{}/temp/sentences_bert_coref.json'.format(output), output_path)


def run_co_reference_factiva():
    config = {'data': '../data/factiva/first_nations.jsonl', 
              'output': '../data/factiva/'}
    
    execute(config)
    
    
def run_co_reference_hansard():
    config = {'data': '../data/hansard/first_nations.jsonl',
              'output': '../data/hansard/'}
    execute(config)
    
    
if __name__ == '__main__':
    # run_co_reference_factiva() # data not shared
    run_co_reference_hansard()
