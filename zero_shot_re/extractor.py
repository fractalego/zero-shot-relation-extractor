import torch


class RelationExtractor:
    '''
    This repository contains a notebook with some examples of how to use the extractor
    notebooks/extractor_examples.ipynb
    Please refer to that notebook.
    '''

    def __init__(self, model, tokenizer, relations):
        '''

        :param model: One of the models in this repository
        :param tokenizer: The appropriate tokenizer
        :param relations: The list of surface forms, one for each relation
                          Example: ['noble title', 'founding date', 'occupation of a person']
        '''
        self._model = model
        self._tokenizer = tokenizer
        self._relations = relations

    def rank(self, text, head, tail):
        '''

        :param text: The text from which to extract the relation
        :param head: The entity that is the subject of the relation
        :param tail: The entity that is the object of the relation

        Example: (text='John Smith works as a carpenter', head='John Smith', tail='carpenter')

        :return: A sorted list of pairs [(surface_form1, probability1), (surface_form2, probability2), ...]
        '''

        text_tokens = text.split()
        head_tokens = head.split()
        tail_tokens = tail.split()

        start_head = _find_sub_list(text_tokens, head_tokens)
        start_tail = _find_sub_list(text_tokens, tail_tokens)
        end_head = start_head + len(head_tokens)
        end_tail = start_tail + len(tail_tokens)

        text_tokens = _double_tokens(text_tokens, start_head, end_head, start_tail, end_tail)
        text_tokens = self._tokenizer.encode(' '.join(text_tokens), add_special_tokens=False)

        scores = []
        for relation_text in self._relations:
            relation_tokens = self._tokenizer.encode(relation_text, add_special_tokens=False)
            adversarial_score = _get_adversarial_score(self._model, text_tokens, relation_tokens)
            scores.append(1 - float(adversarial_score))

        to_return = list(zip(self._relations.copy(), scores))
        to_return = sorted(to_return, key=lambda x: -x[1])

        return to_return


def _run_model(model, text_tokens, relation_tokens):
    inputs = torch.tensor([[101] + relation_tokens
                           + [102] + text_tokens
                           + [102]
                           ])
    length = torch.tensor([len(relation_tokens) + 1])
    subj_starts, subj_ends, obj_starts, obj_ends = model(inputs.cpu(), length)
    return subj_starts[0][0], subj_ends[0][0], obj_starts[0][0], obj_ends[0][0]


def _get_adversarial_score(model, text_tokens, relation_tokens):
    adversarial_score = min(_run_model(model, text_tokens, relation_tokens))
    return adversarial_score


def _find_sub_list(lst, sublist):
    results = []
    sll = len(sublist)
    for ind in (i for i, e in enumerate(lst) if e == sublist[0]):
        if lst[ind:ind + sll] == sublist:
            results.append(ind)

    if not results:
        raise RuntimeError('The entity "' + ' '.join(sublist) + '" is not in the text.')

    return results[0]


def _double_tokens(lst, start_head, end_head, start_tail, end_tail):
    new_lst = []
    for index, item in enumerate(lst):
        new_lst.append(item)
        if start_head <= index < end_head:
            new_lst.append(item)

        if start_tail <= index < end_tail:
            new_lst.append(item)

    return new_lst
