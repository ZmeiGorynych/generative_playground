<<<<<<< Updated upstream
# basic_pytorch
Some simple demos/wrappers showing how to make things work in pytorch
run `run.py` to see how it all fits together.
=======
# Introduction

In this repo I intend to provide a complete Pytorch port of https://github.com/mkusner/grammarVAE, 
which is the official implementation of he <a href="https://arxiv.org/abs/1703.01925">Grammar Variational Autoencoder</a> paper.

Thanks to https://github.com/episodeyang/grammar_variational_autoencoder for an
initial pytorch implementation, which I extended to add proper masking to the loss function,
generalize the code to work with both equations and molecules, and fix a minor bug in the decoder - and intend to extend further.

# How to install/run

## Requirements

* `pip install -r requirements.txt`
* pytorch: [0.4 packaged version](https://pytorch.org/#pip-install-pytorch) appears to have a bug loading saved weights
for modules with batch_norm in them, so I recommend you build from source instead
* rdkit: `conda install -c rdkit rdkit`
* To run the Transformer modules, you'll also need to check out https://github.com/ZmeiGorynych/transformer_pytorch.git
and add the transformer_pytorch (top) directory to your path

## How to run
* `data_utils/make_dataset.py` creates the hd5 datasets necessary to train the models. 
Set the `molecules` boolean at its start to decide whether to generate the dataset for molecules or equations, 
and the `grammar` boolean to decide whether to encode the grammar production sequences or character sequences.
* `train/train_baseline.py` trains the model - again set the `molecules` and `grammar` booleans to choose dataset and model (am in the process of tuning the calibration)
* `back_and_forth.py` goes the full cycle from a SMILES string to a latent space vector and back. As it's using initialized but untrained weights for now, expect the generated strings to be garbage :)
* `notebooks/PrettyPic.ipynb` draws a pretty picture of a molecule from a SMILES string
* To run the Bayesian Optimization, first run `data_utils/generate_latent_features_and_targets.py` to generate the 
necessary datasets for the optimizer, then run `bayesian_opt/run_bo.py` (that one's still under development)

## Changes made in comparison to mkusner/grammarVAE:
* Port to Python 3
* Port the neural model from Keras to PyTorch
* Refactor code to eliminate some repetition
    * Consolidate one-hot dataset creation code, turn on compression in hdf5 and process incrementally to save RAM
    * Merge all neural models into one
    * Move (almost) all settings specific to a particular model to `models/model_settings.py`
* Add extra masking to guarantee sequences are complete by max_len
* Port Bayesian optimization to use GPyOpt (work in progress)

## Known issues:
* I didn't yet try terribly hard to make sure all the settings for string models (kernel sizes etc) are exactly as in
the original code. If you want to amend that, please edit `models/model_settings.py`

## Major things left to do:
* Tune the training of the model(s)
* Add pre-trained weights for each of the four models to the repo
* Make the non-grammar version use tokens rather than raw characters - would be a fairer comparison 
* Add some unit tests, eg for encode-decode roundtrip from string to one-hot and back for each of the four models.

## Extensions implemented so far:
* Add (optional) dropout to all layers of the model
* Provide alternative encoder using RNN + attention
* Version of the model with no sampling of latent variable, and alternative regularization instead of KL
>>>>>>> Stashed changes
