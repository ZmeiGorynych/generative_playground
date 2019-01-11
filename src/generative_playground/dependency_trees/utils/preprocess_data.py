import pyconll
import os
import pickle
import numpy as np
from collections import OrderedDict
import torch
from polyglot.text import Word
import codecs
# https://github.com/UniversalDependencies/docs/blob/pages-source/format.md
pre_defined = {'PAD': 0, 'en': 1, 'de': 2, 'fr': 3}
pre_defined['other'] = len(pre_defined)


def get_metadata(data, cutoff=10):
    '''

    :param data: OrderedDict[lang: {'train':train,
                                    'valid': valid, 'test': test}]
    :param max_len: int, maximal sequence length incl first (language) token
    :return: dict with metadata
    '''
# first pass:
# collect token.lemma into a dict with counts, make indexes for upos and deprel
# upos and deprel are shared, thanks Universal Dependencies!, and token_count is lang-specific
    token_counts = OrderedDict()
    upos = {'PAD': 0}
    deprel = {'PAD': 0}
    for lang, data_dict in data.items():
        token_counts[lang] = OrderedDict()
        data_list = []
        for _, this_data in data_dict.items():
            data_list += this_data

        for sentences in data_list:
            for sentence in sentences:
                for token in sentence:
                    if token.lemma in token_counts[lang]:
                        token_counts[lang][token.lemma] += 1
                    else:
                        token_counts[lang][token.lemma] = 1

                    if token.upos not in upos:
                        upos[token.upos] = len(upos)

                    if token.deprel not in deprel:
                        deprel[token.deprel] = len(deprel)

    # plot frequencies, determine cutoff
    en_length = len([token for token, cnt in token_counts['en'].items() if cnt > cutoff])

    # create dict from lemma to int, with cutoff
    emb_ind = OrderedDict()
    for lang, token_cnt in token_counts.items():
        # sort by frequency and cut to same length as English
        count_list = sorted([(token, count) for token, count in token_cnt.items()],
                            key=lambda x: x[1],
                            reverse=True)
        if len(count_list) > en_length:
            count_list_short = count_list[:en_length]
            print(len(count_list_short)/len(count_list), 'tokens get their own index in ', lang)
        nice_tokens = [token for token, count in count_list_short]

        # now define the mapping from frequent tokens to indices
        emb_ind[lang] = OrderedDict()
        next_index = pre_defined['other'] + 1  # all earlier indices are used to encode other stuff
        for token, count in count_list:
            # TODO: what is the None token about?
            if token in nice_tokens:
                emb_ind[lang][token] = next_index
                next_index += 1
            else:
                emb_ind[lang][token] = pre_defined['other']

        assert next_index == pre_defined['other'] + en_length + 1

    print(len(token_counts), len(emb_ind))

    return {'emb_index': emb_ind,
            'upos': upos,
            'deprel': deprel,
            'counts': token_counts,
            'maxlen': max_len}


def pad(lst, tgt_len, pad_ind = pre_defined['PAD']):
    while len(lst) < tgt_len:
        lst.append(pad_ind)
    return lst


def preprocess_data(sentence_lists, meta, lang):
    # second pass:
    embeds = []
    #maxlen=meta['maxlen']
    count = 0
    for sentences in sentence_lists:
        for sentence in sentences:
            count += 1
            if len(sentence) > max_len-1: # first token will denote language
                # print(sentence)
                continue
            try:
                this_sentence = {'token': [pre_defined[lang]],
                                 'head': [pre_defined[lang]],
                                 'upos': [pre_defined[lang]],
                                 'deprel': [pre_defined[lang]]}
                for t,token in enumerate(sentence):
                    try:
                        assert int(token.id) == t+1, "token.id must equal t+1, instead got " +token.id+ ", t=" + t
                        assert int(token.head) <= len(sentence)
                    except:
                        raise ValueError

                    word = Word(token.form, language=lang)
                    try:
                        word_vector = word.vector
                    except:
                        word_vector = np.ones(256, dtype=np.float32)/256

                    if 'embed' not in this_sentence:
                        this_sentence['embed'] = [np.zeros_like(word_vector)]
                    this_sentence['embed'].append(word_vector)
                    this_sentence['token'].append(meta['emb_index'][lang][token.lemma])
                    this_sentence['head'].append(int(token.head))
                    this_sentence['upos'].append(meta['upos'][token.upos])
                    this_sentence['deprel'].append(meta['deprel'][token.deprel])

                this_sentence_nice = {key: torch.tensor(pad(val, max_len))
                                 for key, val in this_sentence.items() if key != 'embed'}
                pad_embed = pad(this_sentence['embed'], max_len, np.zeros_like(this_sentence['embed'][0]))
                pad_embed_nice = torch.from_numpy(np.array(pad_embed))
                this_sentence_nice['embed'] = pad_embed_nice
                embeds.append(this_sentence_nice)

            except ValueError:
                continue

    print('kept ', len(embeds)/count, ' of all sentences')
    return embeds

# def read_string(fn):
#     with open(fn,'r') as f:
#         out = f.read()
#     return out

if __name__=='__main__':
    data_root = '../data/ud-treebanks-v2.3/'

    datasets=OrderedDict()
    datasets['en'] = ['UD_English-EWT/en_ewt',
              'UD_English-LinES/en_lines',
              'UD_English-ESL/en_esl',
              'UD_English-GUM/en_gum',
              'UD_English-ParTUT/en_partut'
              ]
    datasets['de'] = ['UD_German-GSD/de_gsd',
                      'UD_German-PUD/de_pud']
    datasets['fr'] = ['UD_French-FTB/fr_ftb',
                      'UD_French-GSD/fr_gsd',
                      'UD_French-ParTUT/fr_partut',
                      'UD_French-PUD/fr_pud',
                      'UD_French-Sequoia/fr_sequoia']

    endings = {'valid': '-ud-dev.conllu',
               'test': '-ud-test.conllu',
               'train': '-ud-train.conllu'}
    valid = []
    train = []
    test = []
    max_len = 15
    cutoff = 3
    data = OrderedDict()

    for lang, names in datasets.items():
        data[lang] = OrderedDict([(key, []) for key in endings.keys()])
        for d in names:
            print(d,'...')
            fn_root = data_root + d
            for etype, ending in endings.items():
                fn = fn_root + ending
                if os.path.isfile(fn):
                    f = codecs.open(fn, encoding='utf-8')
                    data_string = f.read()
                    tmp = pyconll.load_from_string(data_string)
                    #tmp = pyconll.load_from_file(fn)

                    data[lang][etype].append(tmp)

            print(OrderedDict(
                [(key, 0 if len(value)==0 else len(value[-1])) for key,value in data[lang].items()]
                    ))

    meta = get_metadata(data, cutoff)
    meta['maxlen'] = max_len

    # TODO: store the files zipped!
    with open('../data/processed/meta.pickle','wb') as f:
        pickle.dump(meta, f)

    for lang, datasets in data.items():
        for dataset_type, dataset in datasets.items():
            embeds = preprocess_data(dataset, meta, lang)
            with open('../data/processed/' + lang + '_' + dataset_type + '_data.pickle','wb') as f:
                pickle.dump(embeds, f)

print('done!')
