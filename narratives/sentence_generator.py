import json
from tqdm import tqdm
from copy import deepcopy
import uuid
import unidecode

import spacy
spacy.prefer_gpu()

# @Language.component("custom_sentencizer")
# def custom_sentencizer(doc):
#     for i, token in enumerate(doc[:-2]):
#         if "\n\n" in token.text:
#             doc[i - 1].is_sent_start = True
#
#     return doc


class SentenceGenerator:
    """
    Parses document corpus and generates sentences and entities
    """
    def __init__(self, record_entity=True):
        self._model = "en_core_web_trf"
        self._disable_tags = ["tagger", "lemmatizer", "ner"]
        self._record_entity = record_entity
        
        if record_entity:
            self._disable_tags = ["tagger", "lemmatizer"]

        self._nlp = spacy.load(self._model, disable=self._disable_tags)


    def _get_sentences(self, content):
        paras = [item.strip() for item in content.split('\n') if item.strip()]
        sentences = []
        sentences_idx = []
        entities = []
        sent_idx = 1
        for para in paras:
            doc = self._nlp(para)
            for sent in doc.sents:
                sent_str = unidecode.unidecode(str(sent)).strip()
                if sent_str:
                    sentences.append(sent_str)
                    sentences_idx.append(sent_idx)

                    sent_idx = sent_idx + 1
            # Save entities
            if self._record_entity:
                for e in doc.ents:
                    entities.append({'text': e.text, 'type': e.label_})
        return entities, sentences, sentences_idx


    def _create_sentence_record(self, item):
        entities, sentences, sentences_idx = self._get_sentences(item['content'])

        record = deepcopy(item)
        record['content'] = {}
        record['content']['sentences'] = sentences
        record['content']['sentences_idx'] = sentences_idx
        record['entities'] = entities
        if 'id' not in record.keys():
            record['id'] = str(uuid.uuid4())
        return record

                    
    def __call__(self, corpus_list, output_path):
        """
        Reads the content from the article and separates text into individual sentences.
        If record entity is set, records the named entities.


        @param corpus_list: List of dictionary articles (must contain 'content' key)
        @param record_entity: True for recording NER
        @return: corpus_list: List of dictionary articles with new keys containing sentences, sentence ids and entities

        """
        print('Splitting into sentences ...')


        # nlp.add_pipe("custom_sentencizer", before="parser")

        with open(output_path, 'a') as f:
            for item in tqdm(corpus_list):
                record = self._create_sentence_record(item)

                json.dump(record, f)
                f.write('\n')
