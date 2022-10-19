import pandas as pd
from tqdm import tqdm

from summa import summarizer


def get_summary_data(dat, ratio):
    sentences = []
    for text in tqdm(dat):
        sentences.append(summarizer.summarize(text, ratio=ratio))
    return (sentences)


def create_narrative_summary(narratives, tr_summary_fpath, output):
    corpus_df_sum = pd.read_json(tr_summary_fpath, lines=True, orient='records')
    corpus_df_sum.loc[:, 'summary'] = corpus_df_sum['summary'].apply(lambda x: x.split('\n'))
    corpus_df_sum = corpus_df_sum.explode('summary')
    corpus_df_sum = corpus_df_sum.rename(columns={'sent': 'text'})

    # Merge narratives with filtered statements
    fn = pd.merge(narratives, corpus_df_sum)

    # sent = pd.read_json('{}/_{}.jsonl'.format(output, subset), lines=True)
    # sent = sent[['id', 'sentences_idx', 'sentences']]
    # sent = sent.apply(pd.Series.explode)
    # sent.columns = ['id', 'idx', 'sent']
    #
    # sent = pd.merge(sent, corpus_df_sum, left_on=['id', 'sent'],  right_on=['id', 'summary'])
    # sent = sent.drop_duplicates()
    # sent['sentence_id'] = sent.id.astype(str) + '_' + sent.idx.astype(str)
    # sents_id = sent['sentence_id'].unique().tolist()
    #
    # del sent, corpus_df_sum
    # fn = narratives[narratives.sentence_id.isin(sents_id)]

    fn.to_json(output, lines=True, orient='records')


def create_textrank_summary(coref_df, ratio, output):
    coref_df.loc[:, 'text'] = coref_df['sentences'].str.join('[SEP]')

    print('Creating textrank summary for ratio {}'.format(ratio))
    sentences = get_summary_data(coref_df.text.tolist(), ratio)

    corpus_df_sum = pd.DataFrame({'id': coref_df.id.tolist(), 'summary': sentences})
    corpus_df_sum.to_json(output, lines=True, orient='records')
   

if __name__=='__main__':
    pass
