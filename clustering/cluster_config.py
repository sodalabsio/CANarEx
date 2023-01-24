cluster_config = dict(
	# sample data
    data='data/hansard_sample/_first_nations_sample.jsonl', # co-referenced data (for textrank summary)
    input_narrative='data/hansard_sample/_first_nations_sample_final_narratives.jsonl', # final narratives path
    output='data/hansard_sample/cluster', # output folder
    subset_prefix='first_nations',
    s_name='all-MiniLM-L6-v2', # sentence embedding model
    s_dimension=384, # sentence embedding dimension
    device='cuda',
    cluster_seed=505,

    # embedding batch size
    batch_size=64,
    
    # k-means faiss
    use_gpu=False,
    verbose=False,

    # topn clustering parameters
    doc_k=2,  # number of clusters per document
    topn_range=list(range(1, 6)),  # number of sentences to select within each cluster
    topn_k=list(range(2, 6)) # for sample data
    #topn_k=list(range(2, 10)) + list(range(10, 50, 5)),  # number of clusters to run on selected N sentences after
    # document level clustering

    # textrank parameters
    tr_ratios=[0.05, 0.1, 0.2, 0.4],  # textrank ratio of sentences to select
    tr_k=list(range(2, 6)) # for sample data
    # tr_k=list(range(2, 10)) + list(range(10, 50, 5)) # number of clusters to run on selected N sentences after 
    # textrank clustering
)
