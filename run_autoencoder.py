# Kai Chang - Caltech CMS-CERN 2016
#
# Program runs the denoising autoencoder developed.
# Backbone from
# - https://github.com/blackecho/Deep-Learning-TensorFlow/blob/master/command_line/run_autoencoder.py
# - https://gist.github.com/blackecho/3a6e4d512d3aa8aa6cf9
#
#
# Needs to have TensorFlow, numpy, and scikit installed on computer.
# =============================================================================


import tensorflow as tf 
import numpy as np
import os

import sys
sys.path.insert(0, '/Users/kaichang/Documents/summer_2016/deep-learning/models')
sys.path.insert(0, '/Users/kaichang/Documents/summer_2016/deep-learning/utils')

import autoencoder

from yadlt.utils import datasets


# tf.apps.flags module is a thin wapper around argparse, implementing a subset
# of the functionality in python-gflags. It is used to configure a network.
flags = tf.app.flags
FLAGS = flags.FLAGS


# Global configuration
flags.DEFINE_string('model_name', 'dae', 'Model name.')
flags.DEFINE_string('dataset', 'mnist', 'Which dataset to use. ["mnist", "cifar10", "neutrino", "custom"]')
flags.DEFINE_string('data_path', '', 'Path to the neutrino dataset directory.')
flags.DEFINE_string('train_dataset', '', 'Path to train set .npy file.')
flags.DEFINE_string('valid_dataset', '', 'Path to valid set .npy file.')
flags.DEFINE_string('test_dataset', '', 'Path to test set .npy file.')
flags.DEFINE_string('cifar_dir', '', 'Path to the cifar 10 dataset directory.')
flags.DEFINE_integer('seed', -1, 'Seed for the random generators (>=0). Useful for testing hyperparameters.')
flags.DEFINE_boolean('restore_previous_model', False, 'If true, restore previous model corresponding to model name.')
flags.DEFINE_boolean('encode_train', False, 'Whether to encode and store the training set.')
flags.DEFINE_boolean('encode_valid', False, 'Whether to encode and store the validation set.')
flags.DEFINE_boolean('encode_test', False, 'Whether to encode and store the test set.')
flags.DEFINE_string('save_reconstructions', '', 'Path to a .npy file to save the reconstructions of the model.')
flags.DEFINE_string('save_parameters', '', 'Path to save the parameters of the model.')
flags.DEFINE_string('weights', None, 'Path to a numpy array containing the weights of the autoencoder.')
flags.DEFINE_string('h_bias', None, 'Path to a numpy array containing the encoder bias vector.')
flags.DEFINE_string('v_bias', None, 'Path to a numpy array containing the decoder bias vector.')


# Stacked Denoising Autoencoder specific parameters
flags.DEFINE_integer('n_components', 256, 'Number of hidden units in the dae.')
flags.DEFINE_float('l2reg', 5e-4, 'Regularization parameter. If 0, no regularization.')
flags.DEFINE_string('corr_type', 'masking', 'Type of input corruption. ["none", "masking", "salt_and_pepper"]')
flags.DEFINE_float('corr_frac', 0, 'Fraction of the input to corrupt.')
flags.DEFINE_integer('xavier_init', 1, 'Value for the constant in xavier weights initialization.')
flags.DEFINE_string('enc_act_func', 'tanh', 'Activation function for the encoder. ["sigmoid", "tanh"]')
flags.DEFINE_string('dec_act_func', 'none', 'Activation function for the decoder. ["sigmoid", "tanh", "none"]')
flags.DEFINE_string('main_dir', 'dae/', 'Directory to store data relative to the algorithm.')
flags.DEFINE_string('loss_func', 'mean_squared', 'Loss function. ["mean_squared" or "cross_entropy"]')
flags.DEFINE_string('calc_acc', True, 'Display accuracy of learning.')
flags.DEFINE_integer('verbose', 1, 'Level of verbosity. 0 - silent, 1 - print accuracy.')
flags.DEFINE_integer('weight_images', 10, 'Number of weight images to generate.')
flags.DEFINE_string('opt', 'gradient_descent', '["gradient_descent", "ada_grad", "momentum", "adam"]')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_float('momentum', 0.4, 'Momentum parameter.')
flags.DEFINE_integer('num_epochs', 1000, 'Number of epochs.')
flags.DEFINE_integer('batch_size', 1000, 'Size of each mini-batch.')


assert FLAGS.dataset in ['mnist','cifar10','neutrinos','custom']
assert FLAGS.train_dataset != '' if FLAGS.dataset == 'custom' else True
assert FLAGS.enc_act_func in ['sigmoid', 'tanh']
assert FLAGS.dec_act_func in ['sigmoid', 'tanh', 'none']
assert FLAGS.corr_type in ['masking', 'salt_and_pepper', 'none']
assert 0. <= FLAGS.corr_frac <= 1.
assert FLAGS.loss_func in ['cross_entropy', 'mean_squared']
assert FLAGS.opt in ['gradient_descent', 'ada_grad', 'momentum', 'adam']


def load_from_np(dataset_path):
	if dataset_path != '':
		return np.load(dataset_path)
	else:
		return None


if __name__ == '__main__':
#	Some defining terms
#	-------------------
#	trX : traning set
#	vlX : validation set
#	teX : test set
	
#	utils.random_seed_np_tf(FLAGS.seed) # not useful at the moment

	# MNIST
	if FLAGS.dataset == 'mnist':
		trX, vlX, teX = datasets.load_mnist_dataset(mode='unsupervised')

	# cifar10
	elif FLAGS.dataset == 'cifar10': 
		trX, teX = datasets.load_cifar10_dataset(FLAGS.cifar_dir, mode='unsupervised')
		vlX = teX[:5000] # Validation set is the first half of the test set

	# neutrinos
	elif FLAGS.dataset == 'neutrinos':
		if FLAGS.data_path:
			trX = load_from_np(FLAGS.data_path)
		else:
			trX = load_from_np(FLAGS.train_dataset)
	
		half = trX.shape[0] / 2
		vlX = trX[:half] # Validation set is the first half of the test set
		teX = load_from_np(FLAGS.test_dataset)

	# custom
	elif FLAGS.dataset == 'custom':
		trX = load_from_np(FLAGS.train_dataset)
		vlX = load_from_np(FLAGS.valid_dataset)
		teX = load_from_np(FLAGS.test_dataset)

	# cannot be reached
	else:	
		trX = None
		vlX = None
		teX = None

	# create the object (not sure if needed)
    # enc_act_func = utilities.str2actfunc(FLAGS.enc_act_func)
	# dec_act_func = utilities.str2actfunc(FLAGS.dec_act_func)

	# models_dir = os.path.join(config.models_dir, FLAGS.main_dir)
	# data_dir = os.path.join(config.data_dir, FLAGS.main_dir)
	# summary_dir = os.path.join(config.summary_dir, FLAGS.main_dir)

	# create the object
	dae = autoencoder.DenoisingAutoencoder(
		seed=FLAGS.seed,
		model_name=FLAGS.model_name,
		n_components=FLAGS.n_components,
		enc_act_func=FLAGS.enc_act_func,
		dec_act_func=FLAGS.dec_act_func,
		xavier_init=FLAGS.xavier_init,
		corr_type=FLAGS.corr_type,
		corr_frac=FLAGS.corr_frac, 
		dataset=FLAGS.dataset,
		loss_func=FLAGS.loss_func,
		calc_acc=FLAGS.calc_acc,
		main_dir=FLAGS.main_dir, 
		opt=FLAGS.opt,
		learning_rate=FLAGS.learning_rate, 
		momentum=FLAGS.momentum,
		verbose=FLAGS.verbose, 
		num_epochs=FLAGS.num_epochs, 
		batch_size=FLAGS.batch_size,
		)

	# fit the model
	W = None
	if FLAGS.weights:
		W = np.load(FLAGS.weights)

	bh = None
	if FLAGS.h_bias:
		bh = np.load(FLAGS.h_bias)

	bv = None
	if FLAGS.v_bias:
		bv = np.load(FLAGS.v_bias)

	dae.fit(trX, teX, restore_previous_model=FLAGS.restore_previous_model)

	# Save the model paramenters
	if FLAGS.save_parameters:
		print('Saving the parameters of the model...')
		params = dae.get_model_parameters()
		for p in params:
			np.save(FLAGS.save_parameters + '-' + p, params[p])

	# Save the reconstructions of the model
	if FLAGS.save_reconstructions:
		print('Saving the reconstructions for the test set...')
		np.save(FLAGS.save_reconstructions, dae.reconstruct(teX))

	# Encode the training data and store it
	dae.transform(trX, name='train', save=FLAGS.encode_train)
	dae.transform(vlX, name='validation', save=FLAGS.encode_valid)
	dae.transform(teX, name='test', save=FLAGS.encode_test)

	# save images
	dae.get_weights_as_images(28, 28, max_images=FLAGS.weight_images)
