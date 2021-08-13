from transformers import AutoTokenizer
from zero_shot_re import RelTaggerModel, RelationExtractor

model = RelTaggerModel.from_pretrained("fractalego/fewrel-zero-shot")
tokenizer = AutoTokenizer.from_pretrained("fractalego/fewrel-zero-shot")

relations = ['noble title', 'founding date', 'occupation of a person']
extractor = RelationExtractor(model, tokenizer, relations)
ranked_rels = extractor.rank(text='John Smith received an OBE', head='John Smith', tail='OBE')
print(ranked_rels)
