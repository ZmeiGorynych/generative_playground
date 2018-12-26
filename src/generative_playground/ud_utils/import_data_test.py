import pyconll
import pickle
import numpy as np
import torch
# https://github.com/UniversalDependencies/docs/blob/pages-source/format.md
pre_defined = {'PAD': 0, 'root': 1, 'other': 2}

def get_metadata(sentence_lists, max_len=float('inf')):
# first pass:
# collect token.lemma into a dict with counts, make indexes for upos and deprel
    token_counts = {}
    upos = {'PAD': 0}
    deprel = {'PAD': 0}
    maxlen = 0
    for sentences in sentence_lists:
        for sentence in sentences:
            if len(sentence) > max_len:
                [print(token.lemma) for token in sentence]
                continue
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


def preprocess_data(sentence_lists, meta, max_len=float('inf')):
    # second pass:
    embeds = []
    maxlen=meta['maxlen']
    for sentences in sentence_lists:
        for sentence in sentences:
            if len(sentence) > max_len:
                print(sentence)
                continue
            try:
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
            except:
                continue
    return embeds

def read_string(fn):
    with open(fn,'r') as f:
        out = f.read()
    return out

if __name__=='__main__':
    data_root = '../../../../dependency_trees/data/ud-treebanks-v2.3/'
    #UD_ENGLISH_TRAIN = data_root + -ud-train.conllu'

    datasets=['UD_English-LinES/en_lines',
              'UD_English-ESL/en_esl',
              'UD_English-EWT/en_ewt',
              'UD_English-GUM/en_gum',
              'UD_English-ParTUT/en_partut'
              ]

    endings = ['-ud-dev.conllu','-ud-test.conllu','-ud-train.conllu']
    valid = []
    train = []
    test = []
    for d in datasets:
        fn = data_root + d
        try:
            valid.append(pyconll.load_from_file(fn + endings[0]))
            train.append(pyconll.load_from_file(fn + endings[2]))
            test.append(pyconll.load_from_file(fn + endings[1]))
        except:
            pass
        print(d, len(train[-1]), len(valid[-1]), len(test[-1]))

    max_len = 10
    meta = get_metadata(train + valid + test, max_len)
    train_embeds = preprocess_data(train, meta, max_len)
    valid_embeds = preprocess_data(valid, meta, max_len)
    test_embeds = preprocess_data(test, meta, max_len)

    with open('train_data.pickle','wb') as f:
        pickle.dump(train_embeds, f)
    with open('test_data.pickle','wb') as f:
        pickle.dump(test_embeds, f)
    with open('valid_data.pickle','wb') as f:
        pickle.dump(valid_embeds, f)

    with open('meta.pickle','wb') as f:
        pickle.dump(meta, f)


print('done!')
# save the list of dicts