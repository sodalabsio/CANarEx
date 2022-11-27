import os
import numpy as np
import pandas as pd
import pickle
import json

from tqdm import tqdm

import re
from string import ascii_letters

from .srl_bert import SRL_Bert

import spacy
spacy.load("en_core_web_trf")

class Narratives:
    """
    Creates micro-narratives of form E-V-E (from documents)
    Uses SRL
    """
    def __init__(self,
                cuda_device=0,
                micro_narratives=True,
                output_folder='.'):
        self._run_split = micro_narratives
        self._selected_tags = ["ARG0", "ARG1", "ARG2", "ARG3", "ARG4",
                               'ARGM-TMP', 'ARGM-LOC',
                               "B-ARGM-MOD", 'B-ARGM-NEG', 'B-V']
        self._srl = SRL_Bert('structured-prediction-srl-bert',
                             selected_tags = self._selected_tags,
                             cuda_device=cuda_device,
                             batch_size=100)
        self._merge = micro_narratives
        self._output_folder = output_folder
        
        # debug
        self._save_all = True
        

    def _load_json(fpath):
        json_file = None
        if os.path.isfile(fpath):
            with open(fpath, 'r', encoding='utf-8') as f:
                json_file = json.load(f)
        return json_file

    def _clean_text(text):
        if text:
            text = text.strip()
            text = re.sub(r"\s*'\s*s", '', text)
            text = text.replace(r'(\(|\))', '')
            # Hard coding parliament (hansard)
            # text = re.sub(r'\b(we|We)\b', 'Parliament', text)
            if all(item.strip().lower() in spacy.lang.en.stop_words.STOP_WORDS for item in text.split()):
                return None
        return (text)

    
    def create_micro_narratives(self, df):
        """
        Merges arguments (E-V-Es) to subsequent arguments (E-V-Es) if present
        ex: ARG0: [Uriah] V:(think) ARG1:[he could beat the game in under three hours]
            ARG0: [he] V:(could beat) ARG1:[the game] ARGM-TMP:[in under three hours]
            
        result:
            [Uriah] V:(think) [he] 
            [he] (could beat) [the game] [in under three hours]
        @param:
            df: DataFrame containing SRL extracted tags (ARGM , verbs, modifiers)

        @return:
            Merged arguments

        """
            
        print("Creating micro-narratives...")
        df = pd.DataFrame(df)
        df = df.groupby(['id', 'sentence'])[
            ['ARG0', 'ARG1', 'ARG2', 'ARG3', 'ARG4', 'ARGM-LOC', 'ARGM-TMP',
             'ARG0_', 'ARG1_', 'ARG2_', 'ARG3_', 'ARG4_', 'ARGM-LOC_', 'ARGM-TMP_',
             'B-ARGM-MOD', 'B-ARGM-NEG','B-V', 'verb']] \
            .apply(lambda x: x.to_dict(orient='records')) \
            .to_dict()

        if self._save_all:
            with open("{}/srl_dic.pk".format(self._output_folder), 'wb') as f:
                pickle.dump(df, f)

        args = ['ARG0', 'ARG1', 'ARG2', 'ARG3', 'ARG4', 'ARGM-LOC']
        args_idx = ['ARG0_', 'ARG1_', 'ARG2_', 'ARG3_', 'ARG4_', 'ARGM-LOC_']

        # new
        for key, values in tqdm(df.items()):
            roles = []
            roles_idx = []
            verbs = []
            for role in values:
                for k, v in role.items():
                    # 5 arguments
                    if k in args:
                        roles.append(v)
                    # indices in the sentences
                    elif k in args_idx:
                        roles_idx.append(v)
                    else:
                        verbs.append(v)
            # We have 6 arguments (roles), 12 arguments in total if we consider only the Es in E-V-E
            # We traverse through the roles, and stop comparison when we are in the last {E set}
            # Forward pass
            for i in range(0, len(roles) - (len(args) - 1)):
                if len(roles_idx[i]) == 0:
                    continue
                for j in range((i + len(args) - (i % len(args))), len(roles)):
                    # stree = SuffixTree(roles[i])
                    # sub_s = roles[j]
                    if len(roles_idx[j]) == 0:
                        continue
                    stree_min, stree_max = roles_idx[i]  # 8, 12
                    sub_s_min, sub_s_max = roles_idx[j]  # 9, 11
                    # if (stree.find_substring(sub_s) >= 0) & (len(roles[j]) < len(roles[i])):
                    # Verify if the word is the subset of the previous E
                    if (sub_s_min >= stree_min) & (sub_s_max <= stree_max):
                        roles[i] = roles[j]
                        roles_idx[i] = roles_idx[j]

            # Reverse pass
            for i in reversed(range(len(roles))):
                if len(roles_idx[i]) == 0:
                    continue
                for j in reversed(range(i - (i % len(args)))):
                    if len(roles_idx[j]) == 0:
                        continue
                    stree_min, stree_max = roles_idx[i]
                    sub_s_min, sub_s_max = roles_idx[j]
                    if (sub_s_min >= stree_min) & (sub_s_max <= stree_max):
                        roles[i] = roles[j]
                        roles_idx[i] = roles_idx[j]

            idx = 0
            # Copy the changes from list back to the dictionary
            for role in values:
                for k in role.keys():
                    if k in args:
                        role[k] = roles[idx]
                        idx = idx + 1

        if self._save_all:
            with open('{}/srl_res_processed.pk'.format(self._output_folder), 'wb') as f:
                pickle.dump(df, f)
        return df

    def create_entities(self, final):
        """
        Create E-V-E connections [ARGM] [V] [ARGM]
        @param:
            df: DataFrame containing merged SRL extracted tags (ARGM , verbs, modifiers)

        @return:
            Dataframe of E-V-Es

        """
        
        print('Creating narratives...')
        if final is None:
            with open('{}/srl_res_processed.pk'.format(self._output_folder), 'rb') as f:
                final = pickle.load(f)
                
        final = pd.DataFrame.from_dict(final, orient='index')
        final = final.reset_index()
        final = final.melt(id_vars='index')

        final = final.drop(columns=['variable']).reset_index(drop=True)
        final = final.rename(columns={'index': 'id'})

        df = pd.DataFrame(final['id'].tolist(), index=final.index)
        df.columns = ['id', 'text']
        final = pd.concat([final.drop(columns='id'), df], axis=1)
        final = final[['id', 'text', 'value']].sort_values('id').reset_index(drop=True)

        final = final.dropna()
        df = pd.DataFrame(final['value'].tolist(), index=final.index)
        final = pd.concat([final.drop('value', axis=1), df], axis=1)

        final = final.replace('', np.nan)
        final.loc[:, 'min_args'] = final[['ARG0', 'ARG1', 'ARG2', 'ARG3', 'ARG4', 'ARGM-LOC', 'ARGM-TMP']].count(axis=1)
        final = final.replace(np.nan, '')
        final = final[final.min_args >= 2]  # filter atleast 2 arguments
        final = final.drop(columns=['min_args'])

        if self._save_all:
            with open('{}/processed_narratives.pk'.format(self._output_folder), 'wb') as f:
                pickle.dump(final, f)
        return final

    def clean_narratives(self, final, save=False):
        print('Cleaning narratives')
        
        if final is None:
            final = pd.read_pickle('{}/processed_narratives.pk'.format(self._output_folder))
            
        # add negation to verb
        final['B-ARGM-NEG'] = np.where(final['B-ARGM-NEG'] != '', 'not', '')
        final.loc[:, 'verb'] = (final['B-ARGM-MOD'] + ' ' + final['B-ARGM-NEG'] + ' ' + final['B-V']).str.strip()
        
        # merge all argm into a list so as to create E-V-E
        final.loc[:, 'combined'] = final[['ARG0', 'ARG1', 'ARG2', 'ARG3', 'ARG4']].values.tolist()
        final.loc[:, 'combined'] = final.combined.apply(lambda x: [i for i in x if i])
        split_df = pd.DataFrame(final['combined'].tolist())
        split_df.columns = ['entity' + str(i + 1) for i in split_df.columns]
        split_df = pd.concat([final.reset_index(drop=True), split_df.reset_index(drop=True)], axis=1)
        final = split_df[['id', 'text', 'verb', 'entity1', 'entity2', 'combined', 'ARGM-LOC', 'ARGM-TMP']].copy()
        
        # clean entity references
        final.loc[:, 'entity1'] = final.entity1.apply(Narratives._clean_text)
        final.loc[:, 'entity2'] = final.entity2.apply(Narratives._clean_text)
        final = final.dropna()
        final = final[final.entity1 != final.entity2]

        # clean verbs
        allowed = set(ascii_letters + ' ')
        final.loc[:, 'verb'] = final.verb.apply(lambda x: ''.join(l for l in x if l in allowed))
        final.loc[:, 'verb'] = final.verb.str.strip()
        final = final[final.verb.str.len() > 1]
        final = final[~final.verb.isin(['is', 'are'])]

        # create narratives
        final.loc[:, 'narrative'] = '[' + final['entity1'] + '] (' + \
                                    final['verb'] + ') [' + \
                                    final['entity2'] + ']'
        final = final.groupby(['id', 'narrative']).head(1)
        
        # create narrative sentences
        final.loc[:, 'sent'] = final['entity1'] + ' ' + \
                                final['verb'] + ' ' + \
                                final['entity2']
        final['sentence_id'] = final['id']
        final['id'] = final.id.str.split('_').str[0]

        print('Saving cleaned narratives ..')
        if save:
            final.to_json('{}/final_narratives.jsonl'.format(self._output_folder), orient='records', lines=True)


    def create_narratives(self, sentences_json, save = True):
        srl_result_path = '{}/srl_result.json'.format(self._output_folder)
        srl_extracted_path = '{}/srl_extracted.jsonl'.format(self._output_folder)

        srl_result = self._srl(sentences_json, self._output_folder)
        srl_result = self._srl.extract_arguments(srl_result, self._output_folder)

        if self._merge:
            srl_result = self.create_micro_narratives(srl_result)

        srl_result = self.create_entities(srl_result)

        self.clean_narratives(srl_result, save=save)

if __name__ == '__main__':
    pass