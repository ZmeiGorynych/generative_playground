try:
    import generative_playground
except:
    import sys, os, inspect
    my_location = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    sys.path.append('../../..')
    #sys.path.append('../../../../DeepRL')
    sys.path.append('../../../../../transformer_pytorch')

import pickle
from .main_train_dependencies import train_dependencies

with open('../../ud_utils/meta.pickle','rb') as f:
    meta = pickle.load(f)

batch_size = 40
drop_rate = 0.3
max_steps = meta['maxlen']
model, fitter1 = train_dependencies(EPOCHS=100,
                                                BATCH_SIZE=batch_size,
                                                max_steps=max_steps,
                                                lr=1e-4,
                                                drop_rate=drop_rate,
                                                decoder_type='attention',
                                                plot_prefix='test ',
                                                dashboard = 'dependencies',
                                                save_file='dependencies_test.h5')
                                                #preload_file='policy_gradient_run.h5')

while True:
    next(fitter1)