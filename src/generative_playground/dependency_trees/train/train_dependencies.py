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

with open('../../ud_utils/meta.pickle','rb') as f:
    meta = pickle.load(f)

batch_size = 100
drop_rate = 0.4
max_steps = meta['maxlen']
model, fitter1 = train_dependencies(EPOCHS=1000,
                                    BATCH_SIZE=batch_size,
                                    max_steps=max_steps,
                                    lr=3e-5,
                                    drop_rate=drop_rate,
                                    decoder_type='attention',
                                    plot_prefix='lr 3e-5 ',
                                    dashboard ='dependencies_novae',
                                    #save_file='dependencies_test.h5',
                                    use_self_attention=False, # None, True, False or Both
                                    vae=False,
                                    target_names=['token' ,'head', 'upos', 'deprel'],
                                    meta=meta)
                                                #preload_file='policy_gradient_run.h5')

while True:
    next(fitter1)