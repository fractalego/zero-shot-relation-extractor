## Introduction
This is a zero-shot relation extractor based on the paper  [Exploring the zero-shot limit of FewRel](https://www.aclweb.org/anthology/2020.coling-main.124).

## Installation
```bash
$ pip install zero-shot-re
```

## Run the Extractor
```python
from transformers import AutoModel, AutoTokenizer
from zero-shot-re import RelationExtractor

model = AutoModel.from_pretrained("fractalego/fewrel-zero-shot")
tokenizer = AutoTokenizer.from_pretrained("fractalego/fewrel-zero-shot")

relations = ['noble title', 'founding date', 'occupation of a person']
extractor = RelationExtractor(model, tokenizer, relations)
ranked_rels = extractor.rank(text='John Smith received an OBE', head='John Smith', tail='OBE')
print(ranked_rels)
```
with results
```python3
[('noble title', 0.9690611883997917),
 ('occupation of a person', 0.0012609362602233887),
 ('founding date', 0.00024014711380004883)]
```

## Accuracy
The results as in the paper are

| Model                  | 0-shot 5-ways | 0-shot 10-ways |
|------------------------|--------------|----------------|
|(1) Distillbert         |70.1±0.5      | 55.9±0.6       |
|(2) Bert Large          |80.8±0.4      | 69.6±0.5       |
|(3) Distillbert + SQUAD |81.3±0.4      | 70.0±0.2       |
|(4) Bert Large + SQUAD  |86.0±0.6      | 76.2±0.4       |

This version uses the (4) Bert Large + SQUAD model

## Cite as
```bibtex
@inproceedings{cetoli-2020-exploring,
    title = "Exploring the zero-shot limit of {F}ew{R}el",
    author = "Cetoli, Alberto",
    booktitle = "Proceedings of the 28th International Conference on Computational Linguistics",
    month = dec,
    year = "2020",
    address = "Barcelona, Spain (Online)",
    publisher = "International Committee on Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.coling-main.124",
    doi = "10.18653/v1/2020.coling-main.124",
    pages = "1447--1451",
    abstract = "This paper proposes a general purpose relation extractor that uses Wikidata descriptions to represent the relation{'}s surface form. The results are tested on the FewRel 1.0 dataset, which provides an excellent framework for training and evaluating the proposed zero-shot learning system in English. This relation extractor architecture exploits the implicit knowledge of a language model through a question-answering approach.",
}
```

