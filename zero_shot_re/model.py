import copy
import json

import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers import BertModel

_pretrained_model = 'bert-large-uncased-whole-word-masking-finetuned-squad'
_bert = BertModel.from_pretrained(_pretrained_model)


class RelTaggerModel(PreTrainedModel):
    _bert_hidden_size = 1024

    base_model_prefix = 'rel_tagger'

    def __init__(self, config, ninp=200, dropout=0.2):
        super().__init__(config)
        self.language_model = _bert
        self.dropout = dropout

        self.input_linear = nn.Linear(self._bert_hidden_size, ninp)

        nout = 2
        self.linear_out1 = nn.Linear(ninp, nout)
        self.linear_out2 = nn.Linear(ninp, nout)


    def forward(self, src, position_ids):
        output = self.language_model(src)[0]

        output = self.input_linear(output)
        output = F.relu(output)

        out1 = self.linear_out1(output)
        out2 = self.linear_out2(output)

        subj_start, subj_end = [F.softmax(item[position_ids[0]:].transpose(0, 1), dim=-1)
                                for item in out1.transpose(0, 2)]

        obj_start, obj_end = [F.softmax(item[position_ids[0]:].transpose(0, 1), dim=-1)
                              for item in out2.transpose(0, 2)]

        return subj_start, subj_end, obj_start, obj_end

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)

        return output

    def to_json_string(self):
        return json.dumps(list(self.to_dict()), indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path):
        with open(json_file_path, "w", encoding='utf-8') as writer:
            writer.write(self.to_json_string())

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        kwargs['config'] = _bert.config
        return super().from_pretrained(*args, **kwargs)
