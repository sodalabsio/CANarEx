import pandas as pd

from narratives import create_folder
from narratives.sentence_generator import SentenceGenerator

            
if __name__ == '__main__':
    output = 'data/hansard'
    # get the list of articles
    corpus_df = pd.read_json('{}/first_nations.jsonl'.format(output), lines=True)
    corpus_df = corpus_df.to_dict('records')
    
    create_folder('{}/temp'.format(output))
    
    sgenerator = SentenceGenerator()
    sgenerator(corpus_df, output_path='{}/temp/orig_sentences.jsonl'.format(output))
