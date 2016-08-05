# Kai Chang - Caltech CMS-CERN 2016
#
# Program takes the filtered .npy files and then feeds the rechit information
# into a denoising autoencoder to retrieve any unaccounted noise.
#
#
# Needs to have TensorFlow installed on computer.
# =============================================================================

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import os
import math
from libs.utils import corrupt


def autoencoder(dimensions=[239,300,150,50,24]):
	'''Build a deep denoising autoencoder with tied weights

	Parameters
	----------
	dimensions : list, optional
		The number of neurons for each layer of the autoencoder.


	Returns
	-------
	x : Tensor
		Input placeholder to the network
	z : Tensor
		Inner-most latent representation
	y : Tensor
		Output reconstruction of the input
	cost : Tensor
		Overall cost to use for training
	'''

	# input into the network
	cell = tf.placeholder(tf.float32, [None, dimensions[0]], name='cell')


	### Probability we will corrupt input.
	# This is the essence of the denoising autoencoder -- We feed forward
	# a noisy input, allowing our network to generalize better to occlusions
	# of what we're really interested in (features of jets and physics, NOT
	# instrumentation or physics noise.) But to measure accuracy, we'll enforce
	# a training signal which measures the original image's reconstruction cost.

	# We'll change this to 1 during training but when ready for 
	# testing/production ready environments, we'll put it back to 0. 
	corrupt_prob = tf.placeholder(tf.float32, [1])
	current_input = corrupt(cell) * corrupt_prob + cell * (1 - corrupt_prob)


	# Build encoder
	encoder = []
	for layer_i, n_output in enumerate(dimensions[1:]):
		n_input = int(current_input.get_shape()[1])					# size of input
		W = tf.Variable(											# weight
			tf.random_uniform([n_input, n_output],
								-1.0 / math.sqrt(n_input),
								1.0 / math.sqrt(n_input)))
		b = tf.Variable(tf.zeros([n_output]))						# linear offset
		encoder.append(W)
		output = tf.nn.tanh(tf.matmul(current_input, W) + b)		# sigmoid function -- http://deeplearning.net/tutorial/dA.html#da
		current_input = output

	# latent representation
	z = current_input
	encoder.reverse()

	# build the decoder using the same weights
	for layer_i, n_output in enumerate(dimensions[:-1][::-1]):
		W = tf.transpose(encoder[layer_i])
		b = tf.Variable(tf.zeros([n_output]))
		ouput = tf.nn.tanh(tf.matmul(current_input, W) + b)
		current_input = output

	# reconstruction through the network
	y = current_input

	# cost function measures pixel-wise difference for good measure
	cost = tf.sqrt(tf.reduce_mean(tf.square(y - x)))
	return {'x': x, 'z': z, 'y': y,
			'corrupt_prob': corrupt_prob,
			'cost': cost}


def test_neutrinos(tf_sample):
	''' Tests the dAe with our neutrino sample, 
		which theoretically should have 
		~0 energy deposition.
	'''

	# data setup
	mean_samp = np.mean(tf_sample, axis=0)
	ae = autoencoder(dimension=[239, 150, 50, 24])

	# alpha (learning rate) and cost function
	learning rate = 0.001
	optimizer = trf.train.AdamOptimizer(learning_rate).minimize(ae['cost'])

	# Create a session to use the graph
	sess = tf.Session()
	sess.run(tf.initialize_all_variables())

	# fit all training data
	batch_size = 50
	n_epochs = 10
	for epoch_i in range(n_epochs):
		for batch_i in range(tf_sample // batch_size)
			batch_xs, _ = 





if __name__ == '__main__':

	# input external data into TensorFlow
	gun_sample = np.load("data/rechit_formatted.npy")
	tf_gsample = tf.convert_to_tensor(gun_sample, dtype=tf.float32)

	input = tf.placeholder(tf.float32) # defines tf.placeholder objects for data entry

	# see data structure
	with tf.Session() as sess:
		print(sess.run(tf_gsample))

