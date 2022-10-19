# CANarEx pipeline

Runs on Linux and macOS using Python 3.9.5

## Factiva and Hansard 'First Nations' dataset

- CaNarEx environment

   ````python
    pip install -r requirements.txt
   ````

### Step1: Split data into sentences
- Use CaNarEx environment
- Run split_sentences_trf.py (data already provided)
    ````python
        python 1.split_sentences_trf.py
    ````


### Step2: Coreference resolution
     
Using SpanBERT 

- Download https://github.com/mandarjoshi90/coref and follow installation instructions from https://github.com/mandarjoshi90/coref ('spanbert_base') into `coref_env` environment
- Install following packages into `coref_env`: 
    ````python
        pip install tokenization
        pip install sacremoses
    ````
- Run coreference resolution
      ````python
      python 2.coref_bert.py
      ````

### Step3: SRL extraction
- Use CaNarEx environment
- Run run_canarex.py
    ````python
    python 3.run_canarex.py
   ````

### Step4: Filtering narratives

#### TopN clustering (document level clustering) and Textrank clustering

   ````python
    python 4.clustering.py
   ````

## Evaluation
The evaluation folder contains generation of synthetic test data for narrative time-series clustering using jupyter notebook.

## Reference (Baseline: Relatio)
- Environment: Follow setup steps from relatio: https://github.com/relatio-nlp/relatio
- Relatio folder provided: changed to add document ids to output generated.

````python
    python 5.run_relatio.py
   ````


### References

- [Text Semantics Capture Political and Economic Narratives](https://arxiv.org/abs/2108.01720) 

- [SpanBERT: Improving pre-training by representing and predicting spans.](https://arxiv.org/pdf/1907.10529)

- [BERT for coreference resolution: Base-lines and analysis.](https://aclanthology.org/D19-1588.pdf)

- [Simple BERT Models for Relation Extraction and Semantic Role Labeling](https://arxiv.org/pdf/1904.05255)

- [Making monolingual sentence embeddings multilingual using knowledge distillation.](https://aclanthology.org/2020.emnlp-main.365.pdf)

- [Fast interpolation- 184 based t-SNE for improved visualization of single-cell 185 RNA-seq data.](https://www.nature.com/articles/s41592-018-0308-4)

- [Textrank.](https://github.com/summanlp/textrank)
