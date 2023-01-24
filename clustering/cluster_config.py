cluster_config = dict(
    data='dummy_result/_merged_text.jsonl', # co-referenced data (for textrank summary)
    input_narrative='dummy_result/_merged_text_final_narratives.jsonl', # final narratives path
    output='dummy_result/cluster', # output folder
    subset_prefix='merged_text',
    s_name='all-MiniLM-L6-v2', # sentence embedding model
    s_dimension=384, # sentence embedding dimension
    device='cuda',
    cluster_seed=505,

    # embedding batch size
    batch_size=64,

    # topn clustering parameters
    doc_k=2,  # number of clusters per document
    topn_range=list(range(1, 6)),  # number of sentences to select within each cluster
    topn_k=list(range(2, 10)) + list(range(10, 50, 5)),  # number of clusters to run on selected N sentences after
    # document level clustering

    # textrank parameters
    tr_ratios=[0.05, 0.1, 0.2, 0.4]  # textrank ratio of sentences to select
    tr_k= list(range(2, 10)) + list(range(10, 50, 5)) # number of clusters to run on selected N sentences after 
    # textrank clustering
)
