import pandas as pd

from narratives import create_folder
from narratives.sentence_generator import SentenceGenerator
    
            
if __name__ == '__main__':
    # Hansard data
    # output = 'data/hansard'
    # input_file ='first_nations.jsonl'
    
    # Sample data
    fpath = 'data/hansard_sample'
    input_file ='{}/first_nations_sample.jsonl'.format(fpath)
    
    # get the list of articles
    corpus_df = pd.read_json(input_file, lines=True)
    corpus_df = corpus_df.to_dict('records')
    
    create_folder('{}/temp'.format(fpath))
    
    sgenerator = SentenceGenerator()
    sgenerator(corpus_df, output_path='{}/temp/orig_sentences.jsonl'.format(fpath))
