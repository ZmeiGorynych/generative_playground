import sys
# that path is the location of the audio generator
sys.path.append("../aind/AIND-VUI-Capstone")

import torch
from torch.optim import lr_scheduler
from warpctc_pytorch import CTCLoss
from gpu_utils import to_gpu
from data_utils.audio2pytorch import inputs2pytorch
from models.audio_models import LSTMModel
from fit import fit
# TODO: do we reset hidden state for each batch?

# loss
batch_size = 16
def my_ctc_loss(ext_probs, ext_labels):
    ctc_loss = to_gpu(CTCLoss())
    return ctc_loss(ext_probs[0], ext_labels[0], ext_probs[1], ext_labels[1])/batch_size

# data
from data_utils.audio_data_generator import AudioGenerator2
spectrogram = False

output_dim = 29
# get the data, from a pre-existing generator
audio_gen_train = AudioGenerator2(spectrogram=spectrogram,
                                  pad_sequences = False,
                                  minibatch_size = batch_size,
                                  audio_location="../aind/AIND-VUI-Capstone")

audio_gen_train.load_data(fit_params=True)

audio_gen_valid = AudioGenerator2(spectrogram=spectrogram,
                                  pad_sequences = False,
                                  minibatch_size = batch_size,
                                  audio_location="../aind/AIND-VUI-Capstone")

audio_gen_valid.load_data(desc_file='valid_corpus.json')
audio_gen_valid.feats_mean, audio_gen_valid.feats_std = audio_gen_train.norm_params()

train_gen = lambda: audio_gen_train.gen(pytorch_format=True)
valid_gen = lambda: audio_gen_valid.gen(pytorch_format=True)



# create and call the model
# grab one input batch just to get the input dimension
ext_inputs, ext_labels = next(train_gen())
# and define the model using that dim
model = LSTMModel(input_dim=ext_inputs[0].shape[2],
                  hidden_dim=200,
                  output_dim=output_dim,
                  batch_size=batch_size)
ext_outputs = model(ext_inputs)

# evaluate the cost
cost = my_ctc_loss(ext_outputs, ext_labels)
#cost.backward()

print('about to start fitting...')
# true_w = torch.ones((20, 1))
#

optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
loss = my_ctc_loss

epochs = 10
save_path = 'test.mdl'


fit(train_gen = train_gen,
    valid_gen = valid_gen,
    model = model,
    optimizer = optimizer,
    scheduler = scheduler,
    epochs = epochs,
    loss_fn = loss,
    save_path=save_path)
#
# # now try creating a new model and loading the old weights
# model_2 = to_gpu(Net(true_w.shape))
# model_2.load_state_dict(torch.load(save_path))
# print(model_2.w)
#
#
