import pyconll
import pickle
import numpy as np
import torch
# https://github.com/UniversalDependencies/docs/blob/pages-source/format.md

pre_defined = {'PAD': 0, 'root': 1, 'other': 2}
def get_metadata(sentences):
# first pass:
# collect token.lemma into a dict with counts, make indexes for upos and deprel
    token_counts = {}
    upos = {'PAD': 0}
    deprel = {'PAD': 0}
    maxlen = 0
    for sentence in train:
        maxlen = max(maxlen, len(sentence)+1) # we prepend the root token to each sentence
        for token in sentence:
            if token.lemma in token_counts:
                token_counts[token.lemma] += 1
            else:
                token_counts[token.lemma] = 1

            if token.upos not in upos:
                upos[token.upos] = len(upos)

            if token.deprel not in deprel:
                deprel[token.deprel] = len(deprel)
    # plot frequencies, determine cutoff
    cutoff = 10
    # create dict from lemma to int
    emb_ind = {}
    next_index = len(pre_defined)
    for token, count in token_counts.items():
        if count < cutoff:
            emb_ind[token] = pre_defined['other']
        else:
            emb_ind[token] = next_index
            next_index += 1


    print(len(token_counts), len(emb_ind))

    return {'emb_index': emb_ind, 'upos': upos, 'deprel': deprel, 'maxlen': maxlen, 'counts': token_counts}


def pad(lst, tgt_len, pad_ind = pre_defined['PAD']):
    while len(lst) < tgt_len:
        lst.append(pad_ind)
    return lst


def preprocess_data(sentences, meta):
    # second pass:
    embeds = []
    maxlen=meta['maxlen']
    for sentence in sentences:
        this_sentence = {'token': [pre_defined['root']],
                         'head': [pre_defined['root']],
                         'upos': [pre_defined['root']],
                         'deprel': [pre_defined['root']]}
        for t,token in enumerate(sentence):
            assert(int(token.id) == t+1)
            assert(int(token.head) <=len(sentence))
            this_sentence['token'].append(meta['emb_index'][token.lemma])
            this_sentence['head'].append(int(token.head))
            this_sentence['upos'].append(meta['upos'][token.upos])
            this_sentence['deprel'].append(meta['deprel'][token.deprel])

        this_sentence = {key: torch.tensor(pad(val, maxlen)) for key, val in this_sentence.items()}
        embeds.append(this_sentence)
    return embeds

if __name__=='__main__':
    data_root = '../../../../dependency_trees/data'
    UD_ENGLISH_TRAIN = data_root + '/UD_English-LinES/en_lines-ud-train.conllu'

    train = pyconll.load_from_file(UD_ENGLISH_TRAIN)
    print(len(train))
    meta = get_metadata(train)
    embeds = preprocess_data(train, meta)

    with open('data.pickle','wb') as f:
        pickle.dump(embeds, f)

    with open('meta.pickle','wb') as f:
        pickle.dump(meta, f)


print('done!')
# save the list of dicts