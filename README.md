# Open Information Extraction via Chunks

## Introduction
About Chunk-OIE: existing OIE systems split a sentence into tokens and recognize token spans as tuple relations and arguments. We instead propose Sentence as Chunk sequence (SaC) and recognize chunk spans as tuple relations and arguments. We argue that SaC has better properties for OIE than sentence as token sequence. We propose a simple end-to-end BERT-based model, Chunk-OIE, for sentence chunking and tuple extraction on top of SaC. The details of this work are elaborated in our paper published in Main Conference of [EMNLP 2023](https://aclanthology.org/2023.emnlp-main.951/).

## Chunk-OIE Model
### Installation Instructions

Use a python-3.7 environment and install the dependencies using,
```
pip install -r requirements.txt
```

### Preparing traing dataset
The training dataset is compressed in the path
```data\oie```.
Please unzip it in the same path before training.

### Running the code

1. For end-to-end Chunk-OIE
```
python allennlp_e2e_run.py --config config/wiki_multi_view.json  --epoch 1 --batch 16  --model trained_model/e2e_chunk_oie
```

2. For 2-stage Chunk-OIE
```
python allennlp_run.py --config config/oie_wiki_oia.json --epoch 1 --batch 32 --model trained_model/2stage_chunk_oie
```

3. For training a chunker and using it to infer chunks (optional)
```
python allennlp_chunk_run.py --config config/chunk_oia.json  --epoch 1 --batch 16  --model trained_model/2stage_chunker
python allennlp_chunk_predict.py
```

Arguments:
- config: configuration file containing all the parameters for the model
- model:  path of the directory where the model will be saved
- epoch:  number of epoch for training
- batch:  number of instances per batch
- 



## Citing
If you use this code in your research, please cite:

```
@inproceedings{dong-etal-2023-open,
    title = "Open Information Extraction via Chunks",
    author = "Dong, Kuicai  and
      Sun, Aixin  and
      Kim, Jung-jae  and
      Li, Xiaoli",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.951",
    doi = "10.18653/v1/2023.emnlp-main.951",
    pages = "15390--15404",
    abstract = "Open Information Extraction (OIE) aims to extract relational tuples from open-domain sentences. Existing OIE systems split a sentence into tokens and recognize token spans as tuple relations and arguments. We instead propose Sentence as Chunk sequence (SaC) and recognize chunk spans as tuple relations and arguments. We argue that SaC has better properties for OIE than sentence as token sequence, and evaluate four choices of chunks (i.e., CoNLL chunks, OIA simple phrases, noun phrases, and spans from SpanOIE). Also, we propose a simple end-to-end BERT-based model, Chunk-OIE, for sentence chunking and tuple extraction on top of SaC. Chunk-OIE achieves state-of-the-art results on multiple OIE datasets, showing that SaC benefits the OIE task.",
}

```

## Contact
In case of any issues, please send a mail to
```kuicai001 (at) e (dot) ntu (dot) edu (dot) sg```