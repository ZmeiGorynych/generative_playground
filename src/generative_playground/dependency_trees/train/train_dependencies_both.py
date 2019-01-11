try:
    import generative_playground
except:
    import sys, os, inspect
    my_location = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    sys.path.append('../../..')
    #sys.path.append('../../../../DeepRL')
    sys.path.append('../../../../../transformer_pytorch')

import pickle
from generative_playground.dependency_trees.train.main_train_dependencies import train_dependencies
use_old = False
if use_old:
    with open('../data/processed/meta_old.pickle','rb') as f:
        meta = pickle.load(f)
    language = None
else:
    with open('../data/processed/meta.pickle','rb') as f:
        meta = pickle.load(f)
    language = 'en'


batch_size = 100
drop_rate = 0.15
max_steps = meta['maxlen']
model, fitter1 = train_dependencies(EPOCHS=1000,
                                    BATCH_SIZE=batch_size,
                                    max_steps=max_steps,
                                    lr=3e-5,
                                    drop_rate=drop_rate,
                                    decoder_type='attention',
                                    plot_prefix='lr 3e-5 both',
                                    dashboard ='dependencies_novae',
                                    #save_file='dependencies_test.h5',
                                    use_self_attention='both', # None, True, False or Both
                                    vae=False,
                                    target_names=['head', 'upos', 'deprel'],#'token' ,
                                    language=language,
                                    meta=meta)
                                                #preload_file='policy_gradient_run.h5')

while True:
    next(fitter1)