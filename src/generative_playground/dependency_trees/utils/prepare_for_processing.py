import glob
import pandas as pd
from collections import OrderedDict


def split_lang(x):
    if ', ' in x:
        return x.split(', ')
    else:
        return x, ''

def fix_backslash(data):
    # convert windows-style directory delimiters to Unix-style
    return [d.replace('\\','/') for d in data]


def prepare_for_ingestion(data_root):
    print("Checking which languages are available in UD...")
    datasets = OrderedDict()
    dirs = fix_backslash(glob.glob(data_root + 'ud-treebanks-v2.3/*'))
    long_to_short = {}
    for dir_ in dirs:
        files = fix_backslash(glob.glob(dir_ + '/*.conllu'))
        dir_root = files[0].replace(dir_, '').split('-')[0]
        lang = dir_root.replace('/', '').split('_')[0]
        long_language = dir_.replace(data_root+'ud-treebanks-v2.3/', '').replace('UD_', '').split('-')[0].replace('_', ' ')
        long_to_short[long_language] = lang
        if lang not in datasets:
            datasets[lang] = []
        datasets[lang].append(dir_ + dir_root)

    def long_to_short_fun(x):
        if x in long_to_short:
            return long_to_short[x]
        else:
            return ''
    print('Looking which languages are available in polyglot...')
    table = pd.read_csv(data_root + 'languages.csv') # entered by hand from universaldependencies.org
    table['code'] = table['LanguageName'].apply(long_to_short_fun)
    table['group1'] = table['Classification'].apply(lambda x: split_lang(x)[0])
    table['group2'] = table['Classification'].apply(lambda x: split_lang(x)[1])
    table = table.drop(columns=['Classification'])
    polyglot = pd.read_csv(data_root + 'polyglot_codes.csv') # generated by hand from polyglot-printed output
    table = pd.merge(table, polyglot, how='left', on='code')
    # table.to_csv(data_root + 'languages_ext.csv')
    print('Discarding languages with no polyglot embeddings...')
    datasets ={lang: value for lang, value in datasets.items()
                if type(table[table['code']==lang]['language'].iloc[0]) == str}

    return datasets, table
