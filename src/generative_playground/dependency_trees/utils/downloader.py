from polyglot.downloader import downloader
import pandas as pd
import numpy as np
# downloader.list(show_packages=False)
# downloader.supported_languages(task="ner2")
languages = pd.read_csv('../data/languages_ext.csv')
for row in languages.iterrows():
    if isinstance(row[1]['language'], str):#not np.isnan(row[1]['language']):
        code = row[1]['code']
        print('downloading', code)
        downloader.download("LANG:" + code)
        print('downloaded', code)