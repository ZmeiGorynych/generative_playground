import pyconll
import os
import pickle
import gzip
import numpy as np
from collections import OrderedDict
import torch
from polyglot.text import Word
from polyglot.downloader import downloader
import json
import pandas as pd
from generative_playground.dependency_trees.utils.prepare_for_processing import prepare_for_ingestion
import codecs
# https://github.com/UniversalDependencies/docs/blob/pages-source/format.md

PAD_INDEX = 0

def get_metadata(data_dict, max_len, cutoff):
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
    # for lang, data_dict in data.items():
    lang = 'en';
    token_counts[lang] = OrderedDict()
    data_list = []
    for _, this_data in data_dict.items():
        data_list += this_data

    for sentences in data_list:
        for sentence in sentences:
            if len(sentence) > max_len - 1: # no point in tokenizing stuff we'll drop later anyway
                continue
            for token in sentence:
                if token.lemma in token_counts[lang]:
                    token_counts[lang][token.lemma] += 1
                else:
                    token_counts[lang][token.lemma] = 1
    # plot frequencies, determine cutoff
    en_length = len([token for token, cnt in token_counts['en'].items() if cnt > cutoff])

    return {'upos': upos,
            'deprel': deprel,
            'maxlen': max_len,
            'en_length': en_length,
            'emb_index': OrderedDict()
            }


def update_meta_with_another_language(data_dict, meta, pre_defined):
    en_length = meta['en_length']
    max_len = meta['maxlen']
    data_list = []
    # combine train, valid and test datasets so we can tokenize together
    for _, this_data in data_dict.items():
        data_list += this_data

    # collect token frequencies and all possible tags
    token_counts = OrderedDict()
    for sentences in data_list:
        for sentence in sentences:
            if len(sentence) > max_len - 1: # no point in tokenizing stuff we'll drop later anyway
                continue
            for token in sentence:
                if token.lemma in token_counts:
                    token_counts[token.lemma] += 1
                else:
                    token_counts[token.lemma] = 1

                if token.upos not in meta['upos']:
                    meta['upos'][token.upos] = len(meta['upos'])

                if token.deprel not in meta['deprel']:
                    meta['deprel'][token.deprel] = len(meta['deprel'])


    # sort by frequency and trim to same length as English
    count_list = sorted([(token, count) for token, count in token_counts.items()],
                        key=lambda x: x[1],
                        reverse=True)
    if len(count_list) > en_length:
        count_list_short = count_list[:en_length]
        print(len(count_list_short)/len(count_list), 'tokens get their own index in ', lang)
        trimmed = True
    else:
        count_list_short = count_list
        print('All', len(count_list), 'tokens get their own indices in', lang)
        trimmed = False
    nice_tokens = [token for token, count in count_list_short]

    # now define the mapping from frequent tokens to indices
    meta['emb_index'][lang] = OrderedDict()
    next_index = pre_defined['other'] + 1  # all earlier indices are used to encode other stuff
    for token, count in count_list:
        # TODO: what is the None token about?
        if token in nice_tokens:
            meta['emb_index'][lang][token] = next_index
            next_index += 1
        else:
            meta['emb_index'][lang][token] = pre_defined['other']
    if trimmed:
        assert next_index == pre_defined['other'] + en_length + 1
    else:
        assert next_index <= pre_defined['other'] + en_length + 1
    print('total tokens:', len(token_counts), len(meta['emb_index'][lang]))

    return meta


def pad(lst, tgt_len, pad_ind=PAD_INDEX):
    while len(lst) < tgt_len:
        lst.append(pad_ind)
    return lst


def preprocess_data(sentence_lists, pre_defined, meta, lang, dataset_type):
    # second pass:
    embeds = []
    max_len = meta['maxlen']
    count = 0
    too_long = 0
    errors = 0
    no_embedding = 0
    strange_id = 0
    for sentences in sentence_lists:
        for sentence in sentences:
            count += 1
            if len(sentence) > max_len-1: # first token will denote language
                # print(sentence)
                too_long += 1
                continue
            try:
                # first token is the root token for the dependency tree, also encodes the language
                this_sentence = {'token': [pre_defined[lang]],
                                 'head': [pre_defined[lang]],
                                 'upos': [pre_defined[lang]],
                                 'deprel': [pre_defined[lang]]}

                #go through the sentence, discarding all the tokens with composite id
                tokens =[]
                for tok in sentence:
                    if tok.head is None:
                        continue
                    try:
                        int(tok.id) # check if that fails
                        tokens.append(tok)
                    except:
                        continue

                for t,token in enumerate(tokens):
                    try:
                        assert int(token.id) == t+1, "token.id must equal t+1, instead got " +token.id+ ", t=" + t
                        assert int(token.head) <= len(sentence)
                    except:
                        strange_id +=1
                        raise ValueError("strange id")

                    word = Word(token.form, language=lang)
                    try:
                        word_vector = word.vector
                    except:
                        no_embedding +=1
                        word_vector = np.zeros(256, dtype=np.float32)
                        #raise ValueError("no embedding") #

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

            except ValueError as e:
                errors +=1
                continue
    if count>0:
        print('kept ', len(embeds)/count, ' of all sentences')
    else:
        print("no valid sentences at all - what's going on here?")

    print('total', count, ', too long', too_long,
          ', no_embedding', no_embedding,
          ', strange ids', strange_id,
          ', total errors', errors)
    meta['stats'][lang][dataset_type] = OrderedDict()
    meta['stats'][lang][dataset_type]['orig_size'] = count
    meta['stats'][lang][dataset_type]['size'] = len(embeds)
    return embeds

# def read_string(fn):
#     with open(fn,'r') as f:
#         out = f.read()
def dowload_languages(language_table):
    for row in language_table.iterrows():
        if isinstance(row[1]['language'], str):#not np.isnan(row[1]['language']):
            code = row[1]['code']
            print('downloading', code)
            downloader.download("LANG:" + code)
            print('downloaded', code)

if __name__=='__main__':
    datasets, language_table = prepare_for_ingestion('../data/')
    pre_defined = {'PAD': PAD_INDEX}

    for lang in datasets.keys():
        pre_defined[lang] = len(pre_defined)
    pre_defined['other'] = len(pre_defined)

    #datasets = {key:value for key, value in datasets.items() if key in ['en','ca','fr','gl']}
    # new_data ={}
    # new_data['zh'] = datasets['zh']
    # new_data['en'] = datasets['en']
    # datasets = new_data

    if False:
        # download polyglot language packages
        dowload_languages(language_table)

    endings = {'train': '-ud-train.conllu',
               'valid': '-ud-dev.conllu',
               'test': '-ud-test.conllu'
               }
    valid = []
    train = []
    test = []
    max_len = 46
    cutoff = 5 # so at least cutoff+1 occurrences
    data = OrderedDict()

    #for lang, names in datasets.items():

    def get_data_for_language(lang):
        root_names = datasets[lang]
        out = OrderedDict([(key, []) for key in endings.keys()])
        for d in root_names:
            print(d,'...')
            fn_root = d
            for etype, ending in endings.items():
                fn = fn_root + ending
                if os.path.isfile(fn):
                    tmp = pyconll.load_from_file(fn)
                    out[etype].append(tmp)

            print(lang, etype, OrderedDict(
                [(key, 0 if len(value)==0 else len(value[-1])) for key,value in out.items()]
                    ))
        return out

    meta = get_metadata(get_data_for_language('en'),
                        max_len,
                        cutoff)

    meta['num_lang'] = len(pre_defined) - 2 # also has 'PAD' and 'other'
    meta['cutoff'] = cutoff
    meta['maxlen'] = max_len
    meta['files'] = {}
    meta['num_tokens'] = meta['en_length'] + len(pre_defined) + 1
    meta['stats'] = OrderedDict()
    meta['predefined'] = pre_defined

    languages = ['en'] + [lang for lang in datasets.keys()]
    for lang in datasets.keys():
        meta['files'][lang] = {}
        meta['stats'][lang] = OrderedDict()
        this_data = get_data_for_language(lang)
        # make sure we have the token indices for that language, and have included any new tags from that language's datasets
        meta = update_meta_with_another_language(this_data, meta, pre_defined)
        meta['stats'][lang]['tokens'] = len(meta['emb_index'][lang])
        for dataset_type, dataset in this_data.items():
            embeds = preprocess_data(dataset, pre_defined, meta, lang, dataset_type)
            print(lang, dataset_type, len(embeds))
            my_path = os.path.dirname(os.path.realpath(__file__))
            this_filename =  lang + '_' + dataset_type + \
                      '_' + str(max_len) + '_' + str(cutoff) + '_data.pkz'
            target_path = os.path.join(my_path, '../data/processed/', this_filename)
            meta['files'][lang][dataset_type] = target_path
            with gzip.open(target_path,'wb') as f:
                pickle.dump(embeds, f)

    with open('../data/processed/meta.pickle','wb') as f:
        pickle.dump(meta, f)
    print('tokens:', len(meta['emb_index']['en']))

print('done!')