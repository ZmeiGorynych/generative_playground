try:
    import generative_playground
except:
    import sys, os, inspect
    my_location = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    sys.path.append('../../..')

import pickle
from generative_playground.dependency_trees.train.main_train_dependencies import train_dependencies

with open('../data/processed/meta.pickle','rb') as f:
    meta = pickle.load(f)

batch_size = 10
drop_rate = 0.05
max_steps = meta['maxlen']
model, fitter1 = train_dependencies(EPOCHS=1000,
                                    BATCH_SIZE=batch_size,
                                    max_steps=max_steps,
                                    lr=3e-5,
                                    drop_rate=drop_rate,
                                    decoder_type='attention',
                                    plot_prefix='lr 3e-5 both roman',
                                    dashboard ='dependencies_novae',
                                    #save_file='dependencies_test.h5',
                                    include_predefined_embedding=True,
                                    use_self_attention='both', # None, True, False or Both
                                    vae=False,
                                    plot_ignore_initial=300,
                                    target_names=['head', 'upos', 'deprel'],#'token' ,
                                    meta=meta,
                                    languages=['ca','fr','gl'],#,'ro','es','pt','it'],#['pt'],##['en', 'de', 'fr'],
                                    ignore_padding=True)
                                                #preload_file='policy_gradient_run.h5')

while True:
    next(fitter1)