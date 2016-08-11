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
import matplotlib.pyplot as plt
import os, math

import sys
sys.path.insert(0, '/Users/kaichang/Documents/summer_2016/deep-learning/models')
sys.path.insert(0, '/Users/kaichang/Documents/summer_2016/deep-learning/utils')

import autoencoder
import config

from yadlt.utils import datasets


# tf.apps.flags module is a thin wapper around argparse, implementing a subset
# of the functionality in python-gflags. It is used to configure a network.
flags = tf.app.flags
FLAGS = flags.FLAGS


# Global configuration
flags.DEFINE_string('model_name', 'dae', 'Model name.')
flags.DEFINE_string('dataset', 'mnist', 'Which dataset to use. ["mnist", "cifar10", "neutrino", "custom"]')
flags.DEFINE_string('neutrino_path', '', 'Path to the neutrino dataset directory.')
flags.DEFINE_string('train_dataset', '', 'Path to train set .npy file.')
flags.DEFINE_string('valid_dataset', '', 'Path to valid set .npy file.')
flags.DEFINE_string('test_dataset', '', 'Path to test set .npy file.')
flags.DEFINE_string('cifar_dir', '', 'Path to the cifar 10 dataset directory.')
flags.DEFINE_boolean('restore_previous_model', False, 'If true, restore previous model corresponding to model name.')
flags.DEFINE_boolean('encode_train', False, 'Whether to encode and store the training set.')
flags.DEFINE_boolean('encode_valid', False, 'Whether to encode and store the validation set.')
flags.DEFINE_boolean('encode_test', False, 'Whether to encode and store the test set.')
flags.DEFINE_string('save_reconstructions', '', 'Path to a .npy file to save the reconstructions of the model.')
flags.DEFINE_string('save_parameters', '', 'Path to save the parameters of the model.')
flags.DEFINE_string('weights', None, 'Path to a numpy array containing the weights of the autoencoder.')
flags.DEFINE_string('h_bias', None, 'Path to a numpy array containing the encoder bias vector.')
flags.DEFINE_string('v_bias', None, 'Path to a numpy array containing the decoder bias vector.')
flags.DEFINE_integer('seed', -1, 'Seed for the random generators (>=0). Useful for testing hyperparameters.')


# Stacked Denoising Autoencoder specific parameters
flags.DEFINE_string('main_dir', 'dae/', 'Directory to store data relative to the algorithm.')
flags.DEFINE_integer('n_components', 256, 'Number of hidden units/features in the dae.')
flags.DEFINE_float('l2reg', 5e-4, 'Regularization parameter. If 0, no regularization.')
flags.DEFINE_string('corr_type', 'masking', 'Type of input corruption. ["none", "masking", "salt_and_pepper"]')
flags.DEFINE_float('corr_frac', 0, 'Fraction of the input to corrupt.')
flags.DEFINE_integer('xavier_init', 1, 'Value for the constant in xavier weights initialization.')
flags.DEFINE_string('enc_act_func', 'tanh', 'Activation function for the encoder. ["sigmoid", "tanh"]')
flags.DEFINE_string('dec_act_func', 'none', 'Activation function for the decoder. ["sigmoid", "tanh", "none"]')
flags.DEFINE_string('loss_func', 'mean_squared', 'Loss function. ["mean_squared" or "cross_entropy"]')
flags.DEFINE_string('calc_acc', True, 'Display accuracy of learning.')
flags.DEFINE_integer('verbose', 1, 'Level of verbosity. 0 - silent, 1 - print accuracy.')
flags.DEFINE_string('opt', 'gradient_descent', '["gradient_descent", "ada_grad", "momentum", "adam"]')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_float('momentum', 0.4, 'Momentum parameter.')
flags.DEFINE_integer('num_epochs', 100, 'Number of epochs.')
flags.DEFINE_integer('batch_size', 128, 'Size of each mini-batch.')

flags.DEFINE_integer('weight_images', 10, 'Number of weight images to generate.')
flags.DEFINE_integer('encdec_images', 10, 'Number of encoded and decoded images to generate (ie. 10 means each using the same 10 data).')
flags.DEFINE_boolean('custom_dimension', False, 'If image is not a square, then True.')
flags.DEFINE_integer('encoded_width', 16, 'Encoded image width')
flags.DEFINE_integer('encoded_height', 16, 'Encoded image height')
flags.DEFINE_integer('decoded_width', 28, 'Decoded image width')
flags.DEFINE_integer('decoded_height', 28, 'Decoded image height')
flags.DEFINE_string('image_type', 'grey', 'Color image produced ["grey", "RGB", "CMYK"]')


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

def is_square(p_int):
	x = p_int // 2
	seen = set([x])
	while x * x != p_int:
		x = (x + (p_int // x)) // 2
		if x in seen: return False
		seen.add(x)
	return True

def factor_pairs(val):
    return [(i, val / i) for i in range(1, int(val**0.5)+1) if val % i == 0]


if __name__ == '__main__':
#	Some defining terms
#	-------------------
#	trX : traning set
#	vlX : validation set
#	teX : test set
	
#	utils.random_seed_np_tf(FLAGS.seed) # not useful at the moment

	# Writing datapaths if nonexistent
	main_dir = FLAGS.main_dir + '/' if FLAGS.main_dir[-1] != '/' else FLAGS.main_dir

	models_dir = os.path.join(main_dir, config.models_dir)
	data_dir = os.path.join(main_dir, config.data_dir)
	summary_dir = os.path.join(main_dir, config.summary_dir)

	for d in [models_dir, data_dir, summary_dir]:
		if not os.path.exists(d):
			os.makedirs(d)

	
	# MNIST
	if FLAGS.dataset == 'mnist':
		trX, vlX, teX = datasets.load_mnist_dataset(mode='unsupervised')		# already normalized
		d_width = d_height = 28

	# cifar10
	elif FLAGS.dataset == 'cifar10': 
		trX, teX = datasets.load_cifar10_dataset(FLAGS.cifar_dir, mode='unsupervised')
		vlX = teX[:5000] # Validation set is the first half of the test set
		d_width = d_height = 32

	# neutrinos
	elif FLAGS.dataset == 'neutrinos':
		if FLAGS.data_path:
			data = load_from_np(FLAGS.neutrino_path)
		else:
			data = load_from_np(FLAGS.train_dataset)

		trX = data[:-data.shape[0]/5]
		# vlX = trX[-trX.shape[0]/5:]   # Validation set is the last 1/5 of the training set
		vlX = None
		teX = data[-data.shape[0]/5:]

		d_width = None 		# FIX LATER
		d_height = None

	# custom
	elif FLAGS.dataset == 'custom':
		trX = load_from_np(FLAGS.train_dataset)
		vlX = load_from_np(FLAGS.valid_dataset)
		teX = load_from_np(FLAGS.test_dataset)

		d_width = None 		# FIX LATER
		d_height = None

	# cannot be reached
	else:	
		trX = None
		vlX = None
		teX = None

		d_width = None 		# FIX LATER
		d_height = None

	# custom weight and bias settings
	W_ = None
	if FLAGS.weights:
		W_ = np.load(FLAGS.weights)

	bh_ = None
	if FLAGS.h_bias:
		bh = np.load(FLAGS.h_bias)

	bv_ = None
	if FLAGS.v_bias:
		bv_ = np.load(FLAGS.v_bias)

	# print shape of sample
	print 'shape: ', trX.shape

	# create the object
	dae = autoencoder.DenoisingAutoencoder(
		model_name=FLAGS.model_name,
		n_components=FLAGS.n_components,
		models_dir=models_dir, 
		data_dir=data_dir,
		summary_dir=summary_dir,
		enc_act_func=FLAGS.enc_act_func,
		dec_act_func=FLAGS.dec_act_func,
		loss_func=FLAGS.loss_func,
		calc_acc=FLAGS.calc_acc,
		num_epochs=FLAGS.num_epochs, 
		batch_size=FLAGS.batch_size,
		dataset=FLAGS.dataset,
		xavier_init=FLAGS.xavier_init,
		opt=FLAGS.opt,		
		learning_rate=FLAGS.learning_rate, 
		momentum=FLAGS.momentum,
		corr_type=FLAGS.corr_type,
		corr_frac=FLAGS.corr_frac, 
		verbose=FLAGS.verbose, 
		seed=FLAGS.seed,
		l2reg=FLAGS.l2reg,
		W_=W_,
		bh_=bh_,
		bv_=bv_
		)


	# fit the model
	dae.fit(trX, validation_set=teX, restore_previous_model=FLAGS.restore_previous_model)

	# computes reconstructed models
	teX_encoded = dae.transform(teX, save=False)		# (10000, n_component for MNIST)
	teX_decoded = dae.reconstruct(teX, save=False)			# (10000, 784 for MNIST)

	# shape of our transformed and encoded data
	print 'transform/encoded shape: ', teX_encoded.shape
	print 'reconstruct/decoded shape: ', teX_decoded.shape


	# displays decoded models
	# first checks if u want to print images, then
	# checks if your dimension are alirght by: if standard dataset -> if custom dimensions matches each sample length -> if square
	# fails if doesn't meat the three criteria
	if FLAGS.encdec_images > 0:
		n = FLAGS.encdec_images
		pr_dec = True
		plt.figure(figsize=(20, 2)) 	# figsize -> (w,h) in inches

		if d_width is not None:
			pass

		elif FLAGS.custom_dimension:
			if FLAGS.decoded_height * FLAGS.decoded_width != teX_decoded.shape[1]:
				print 'Not correct dimensions --> %i != %i' %(FLAGS.decoded_height * FLAGS.decoded_width, teX_decoded.shape[1])
				pr_dec = False
			else:
				d_width = FLAGS.decoded_width
				d_height = FLAGS.decoded_height 
		else:
			if is_square(teX_decoded.shape[1]):
				d_width = d_height = int(math.sqrt(teX_decoded.shape[1]))
			else:
				print 'Not a square image'
				pr_dec = False

		if pr_dec == True:
			for i in range(n):
				ax = plt.subplot(2, n, i + n) 	# (row, col, plot number -- identify particular subplot, starts at 1, ends at max ie. row * col)
				plt.imshow(teX_decoded[i].reshape(d_width, d_height))
				plt.gray()
				ax.get_xaxis().set_visible(False)
				ax.get_yaxis().set_visible(False)

			plt.show()

	# displays encoded models
	# The decoder first checks if your custom dimensions fit the n_components
	# If not, then it will optimize for the best results
	# - first by square
	# - second by scale
	# - third by increment
	# will fail and not print if does not pass the two trials
	if FLAGS.encdec_images > 0:
		pr_enc = False
		plt.figure(figsize=(20, 2)) 	# figsize -> (w,h) in inches

		if FLAGS.encoded_height * FLAGS.encoded_width != teX_encoded.shape[1]:
			print 'Not correct dimensions --> %i != %i' %(FLAGS.encoded_height * FLAGS.encoded_width, teX_encoded.shape[1])
		else:
			if is_square(teX_encoded.shape[1]):
				pr_enc = True
				e_width = e_height = int(math.sqrt(teX_encoded.shape[1]))

			elif is_square(teX_encoded.shape[1] * float(d_width) / d_height):
				pr_enc = True
				e_width = int(math.sqrt(teX_encoded.shape[1] * float(d_width) / d_height))
				e_height = int(math.sqrt(teX_encoded.shape[1] * d_height / float(d_width)))

			else:
				(e_width, e_height) = 1, teX_encoded.shape[1]
				pairs = factor_pairs(e_height)
				for (w,h) in pairs:
					if w > e_width and h < e_height and w < teX_encoded.shape[1]/2:
						(e_width, e_height) = (w, h)

				if e_width != 1:
					pr_enc = True

		if pr_enc == True:
			for i in range(n):
				ax = plt.subplot(2, n, i + n)
				plt.imshow(teX_encoded[i].reshape(e_width, e_height))
				plt.gray()
				ax.get_xaxis().set_visible(False)
				ax.get_yaxis().set_visible(False)

			plt.show()

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

	# save what the autoencoder has learned <-- currently not working (has to do with Graph(), will fix by adding as_default() like prior.)
	# dae.get_weights_as_images(28, 28, n_images=FLAGS.weight_images, img_type=FLAGS.image_type)
