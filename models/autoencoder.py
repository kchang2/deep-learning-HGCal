# Kai Chang - Caltech CMS-CERN 2016
#
# Program takes the filtered .npy files and then feeds the rechit information
# into a denoising autoencoder to retrieve any unaccounted noise.
#
#
# Modified from https://gist.github.com/blackecho/3a6e4d512d3aa8aa6cf9
#				https://github.com/blackecho/Deep-Learning-TensorFlow/
#
# Needs to have TensorFlow installed on computer.
# =============================================================================


import tensorflow as tf
import numpy as np
import os
import sys
sys.path.insert(0, '/Users/kaichang/Documents/summer_2016/deep-learning/utils')

import config
import utils

class DenoisingAutoencoder(object):
	''' Implementation of Denoising Autoencoder using TensorFlow
	The interface of the class is similar to sklearn
	'''

	def __init__(self, 
				model_name='dae', 
				n_components=256, 
				main_dir='dae/', 
				enc_act_func='tanh',
				dec_act_func='none', 
				loss_func='mean_squared',
				calc_acc=False,
				num_epochs=10,
				batch_size=10,
				dataset='neutrinos',
				xavier_init=1,
				opt='gradient_descent',
				learning_rate=0.01,
				momentum=0.5,
				corr_type='none',
				corr_frac=0,
				verbose=1,
				seed=-1
				):
		'''
		Parameters
		----------
		model_name 			: name of model to use, used to save data
		n_components 		: number of hidden units (number of components to keep)
		main_dir 			: main directory to put the modles, data, and summary directories
		enc_act_func 		: activation function for the encoder (ie. tanh, sigmoid)
		dec_act_function 	: activation function for the decoder (ie. tanh, sigmoid, none)
		loss_func 			: loss function (ie. mean_squared, cross_entropy) used to measure degree of fit
		accuracy 			: accuracy of our reconstructed results
		num_epochs 			: number of epoch or how many revolutions / cycles
		batch_size 			: size of each mini-batch or samples inputed at any given time
		dataset 			: optional name for the dataset
		xavier_init 		: Value of the constant for xavier weights initialization
		opt 				: which tensorflow optimization method to use (ie. gradient_descent, momentum, ada_grad)
		learning_rate 		: initial learning rate (alpha term, fixed)
		momentum 			: momentum parameter (adds a fraction m of the previous weight update to the current one, NOT fixed)
		corr_type 			: type of input corruption (ie. none, masking, salt_and_pepper)
		corr_frac 			: fraction of the input to corrupt
		verbose 			: Level of verbosity or frequency of information regarding learning process printed (0 - silent, 1 - print accuracy)
		seed 				: positive integer for seeding random generators. Ignored if < 0.
	
		Note these are optional
		-----------------------
		batch_size - default 10
		learning rate - default 0.01
		num_epochs - default 10
		verbose - default 0
		'''

		# initialization (all set from input or creation of value)
		self.model_name = model_name
		self.n_components = n_components
		self.main_dir = main_dir
		self.enc_act_func = enc_act_func
		self.dec_act_func = dec_act_func
		self.loss_func = loss_func
		self.calc_acc = calc_acc
		self.num_epochs = num_epochs
		self.batch_size = batch_size
		self.dataset = dataset
		self.xavier_init = xavier_init
		self.opt = opt
		self.learning_rate = learning_rate
		self.momentum = momentum
		self.corr_type = corr_type
		self.corr_frac = corr_frac
		self.verbose = verbose
		self.seed = seed

		if self.seed >= 0:
			np.random.seed(self.seed)
			tf.set_random_seed(self.seed)

		self.models_dir, self.data_dir, self.tf_summary_dir = self._create_data_directories()
		self.model_path = self.models_dir + self.model_name

		self.input_data = None
		self.input_data_corr = None

		self.W_ = None
		self.bh_ = None
		self.bv_ = None

		self.encode = None
		self.decode = None

		self.train_step = None
		self.cost = None
		self.accuracy = None

		self.tf_session = None
		self.tf_merged_summaries = None
		self.tf_summary_writer = None
		self.tf_saver = None


	def fit(self, train_set, validation_set=None, restore_previous_model=False):
		''' Fit the model to the data.

		Parameters
		----------
		train_set 				: training data
		validation_set 			: validation data - optional, default None.
		restore_previous_value 	: if true, a previous trained model with the same name of
								  with the same name of this model is restored from disk
								  to continue trading.

		Returns
		-------
		self
		'''

		n_features = train_set.shape[1]

		self._build_model(n_features)

		with tf.Session() as self.tf_session:
			self._initialize_tf_utilities_and_ops(restore_previous_model)
			self._train_model(train_set, validation_set)
			self.tf_saver.save(self.tf_session, self.models_dir + self.model_name)


	def _initialize_tf_utilities_and_ops(self, restore_previous_model):
		''' Initialize TensorFlow operations: summaries, init operations, saver, summar_writer.
		Restore a previously trained model if the flag restore_previous_model is true.
		'''

		self.tf_merged_summaries = tf.merge_all_summaries()
		init_op = tf.initialize_all_variables()
		self.tf_saver = tf.train.Saver()

		self.tf_session.run(init_op)

		if restore_previous_model:
			self.tf_saver.restore(self.tf_session, self.model_path)

		self.tf_summary_writer = tf.train.SummaryWriter(self.tf_summary_dir, self.tf_session.graph)
		# train_writer = tf.train.SummaryWriter(self.tf_summary_dir + '/train', self.tf_session.graph)
		# test_writer = tf.train.SummaryWriter(self.tf_summary_dir + '/test', self.tf_session.graph)

	def _train_model(self, train_set, validation_set):
		''' Trains the model

		Parameters
		----------
		train_set 		: training set
		validation set 	: validation set -- optional, default None

		Returns
		-------
		self
		'''

		corruption_ratio = np.round(self.corr_frac * train_set.shape[1]).astype(np.int)

		for i in range(self.num_epochs):
			self._run_train_step(train_set, corruption_ratio)

			if validation_set is not None:
				self._run_validation_error_and_summaries(i, validation_set)


	def _run_train_step(self, train_set, corruption_ratio):
		''' Run a training step. A training step is made by randomly corrupting
		the training set, randomly shuffling it, and dividing it into batches and 
		running the optimizer for each batch.

		Parameters
		----------
		train_set 			: training set
		corruption_ratio	: fraction of elements to corrupt

		Returns
		-------
		self
		'''

		x_corrupted = self._corrupt_input(train_set, corruption_ratio)

		shuff = zip(train_set, x_corrupted)
		np.random.shuffle(shuff)

		batches = [_ for _ in utils.gen_batches(shuff, self.batch_size)]

		for batch in batches:
			x_batch, x_corr_batch = zip(*batch)
			tr_feed = {self.input_data: x_batch, self.input_data_corr: x_corr_batch}
			self.tf_session.run(self.train_step, feed_dict=tr_feed)


	def _corrupt_input(self, data, v):
		''' Corrupt a fraction 'v' of 'data' according to the noise method of this autoencoder.

		Returns
		-------
		corrupted data
		'''

		if self.corr_type == 'none':
			return np.copy(data)

		if v > 0.0:
			if self.corr_type == 'masking':
				return utils.masking_noise(data, v)
			elif self.corr_type == 'salt_and_pepper':
				return utils.salt_and_pepper_noise(data, v)
		else:
			return np.copy(data)

	def _run_validation_error_and_summaries(self, epoch, validation_set):
		''' Run the summaries and error computation on the validation set.

		Parameters
		----------
		epoch 			: current epoch
		validation_set	: validation data

		Returns
		-------
		self
		'''

		vl_feed = {self.input_data: validation_set, self.input_data_corr: validation_set}
		result = self.tf_session.run([self.tf_merged_summaries, self.cost], feed_dict=vl_feed)
		summary_str = result[0]
		err = result[1]

		self.tf_summary_writer.add_summary(summary_str, epoch) # records all info from summary at point epoch
		self.tf_summary_writer.flush()
		
		if self.verbose == 1:
			print('validation cost at step %s: %s' %(epoch, err))


	def _build_model(self, n_features):
		''' Creates the computational graph.

		Parameters
		----------
		n_features	: number of features, int
		regtype 	: regularization type
		W_ 			: weight of matrix np array
		bh_ 		: hidden bias np array
		bv_ 		: visible bias np array

		Returns
		-------
		self
		'''

		self.input_data, self.input_data_corr = self._create_placeholders(n_features)
		self.W_, self.bh_, self.bv_ = self._create_variables(n_features)

		self._create_encode_layer()
		self._create_decode_layer()

		self._create_cost_function_node()
		self._create_train_step_node()
		self._create_accuracy_node()
		self._create_variable_node(self.W_, 'weight')
		self._create_variable_node(self.bh_, 'hidden bias')
		self._create_variable_node(self.bv_, 'visible bias')


	def _create_placeholders(self, n_features):
		''' Creates the TensorFlow placeholders for the model.

		Returns
		-------
		tuple ( input_data( shape(None, n_features)),
				input_data_corr( shape(None, n_feature)))
		'''

		input_data = tf.placeholder('float', [None, n_features], name='x-input')
		input_data_corr = tf.placeholder('float', [None, n_features], name='x-corr-input')

		return input_data, input_data_corr


	def _create_variables(self, n_features):
		''' Create the TensorFlow variables for the model.

		Returns 
		-------
		tuple ( weights( shape(n_features, n_components)),
				hidden bias( shape(n_components)),
				visible bias( shape(n_features)))
		'''

		W_ = tf.Variable(utils.xavier_init(n_features, self.n_components, self.xavier_init), name='enc-w')
		bh_ = tf.Variable(tf.zeros([self.n_components]), name='hidden-bias')
		bv_ = tf.Variable(tf.zeros([n_features]), name='visible-bias')

		return W_, bh_, bv_


	def _create_encode_layer(self):
		''' Create teh encoding alyer of the network

		Returns
		-------
		self
		'''

		with tf.name_scope('W_x_bh'):
			if self.enc_act_func == 'sigmoid':
				self.encode = tf.nn.sigmoid(tf.matmul(self.input_data_corr, self.W_) + self.bh_)

			elif self.enc_act_func == 'tanh':
				self.encode = tf.nn.tanh(tf.matmul(self.input_data_corr, self.W_) + self.bh_)

			else:
				self.encode = None


	def _create_decode_layer(self):
		''' Create the decoding layer of the network.

		Returns
		-------
		self
		'''

		with tf.name_scope('Wg_y_bv'):
			if self.dec_act_func == 'sigmoid':
				self.decode = tf.nn.sigmoid(tf.matmul(self.encode, tf.transpose(self.W_)) + self.bv_)
				_ = tf.histogram_summary('decoding layer -- matmul sigmoid', self.decode)


			elif self.dec_act_func == 'tanh':
				self.decode = tf.nn.tanh(tf.matmul(self.encode, tf.transpose(self.W_)) + self.bv_)
				_ = tf.histogram_summary('decoding layer -- matmul tanh', self.decode)


			elif self.dec_act_func == 'none':
				self.decode = tf.matmul(self.encode, tf.transpose(self.W_)) + self.bv_
				_ = tf.histogram_summary('decoding layer -- matmul', self.decode)


			else:
				self.decode = None


	def _create_cost_function_node(self):
		''' create the cost function node of the network

		Returns
		-------
		self
		'''

		with tf.name_scope('cost'):
			if self.loss_func == 'cross_entropy':
				self.cost = -tf.reduce_sum(self.input_data * tf.log(self.decode))
				_ = tf.scalar_summary("cross_entropy", self.cost)

			elif self.loss_func == 'mean_squared':
				self.cost = tf.sqrt(tf.reduce_mean(tf.square(self.input_data - self.decode)))
				_ = tf.scalar_summary("mean_squared", self.cost)

			else:
				self.cost = None


	def _create_accuracy_node(self):
		''' create the accuracy node of the network

		Returns
		-------
		self
		'''
		with tf.name_scope('accuracy'):
			if self.accuracy:
				correct_prediction = tf.equal(tf.argmax(self.decode, 1), tf.argmax(self.input_data,1))
				accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
				_ =  tf.scalar_summary('accuracy', accuracy)
			else:
				self.accuracy = None


	def _create_variable_node(self, var, name):
		''' creates the summary variables to a Tensor.
		'''
		with tf.name_scope('summaries'):
			mean = tf.reduce_mean(var)
			_ = tf.scalar_summary('mean/' + name, mean)
			with tf.name_scope('stddev'):
				stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))

			_ = tf.scalar_summary('stddev/' + name, stddev)
			_ = tf.scalar_summary('max/' + name, tf.reduce_max(var))
			_ = tf.scalar_summary('min/' + name, tf.reduce_min(var))
			_ = tf.histogram_summary(name, var)


	def _create_train_step_node(self):
		''' Create the training step node of the network.

		Returns
		-------
		self
		'''

		with tf.name_scope("train"):
			if self.opt == 'gradient_descent':
				self.train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)

			elif self.opt == 'ada_grad':
				self.train_step = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.cost)

			elif self.opt == 'momentum':
				self.train_step = tf.train_MomentumOptimizer(self.learning_rate, self.momentum).minimize(self.cost)

			else:
				self.train_step = None


	def transform(self, data, name='train', save=False):
		''' Transforms data according to the model.

		Parameters
		----------
		data : data to transform
		name : identifier for the data that is being encoded
		save : if true, save data to disk

		Returns
		-------
		transformed data
		'''

		with tf.Session() as self.tf_session:
			self.tf_saver.restore(self.tf_session, self.models_dir + self.model_name)
			encoded_data = self.encode.eval({self.input_data_corr: data})

			if save:
				np.save(self.data_dir + self.model_name + '-' + name, encoded_data)

			return encoded_data


	def load_model(self, shape, model_path):
		''' Restore a previously trained model from disk.

		Parameters
		----------
		shape 		: tuple (n_features, n_components)
		model_path	: path to the trained model

		Returns
		-------
		self
		the trained model
		'''

		self.n_components = shape[1]
		self._build_model(shape[0])
		init_op = tf.initalize_all_variables()
		self.tf_saver = tf.train.Saver()

		with tf.Session() as self.tf_session:
			self.tf_session.run(init_op)
			self.tf_saver.restore(self.tf_session, model_path)


	def get_model_parameters(self):
		''' Return the model parameters in the form of numpy arrays.

		Returns
		-------
		model parameters
		'''
		with tf.Session() as self.tf_session:
			self.tf_saver.restore(self.tf_session, self.models_dir + self.model_name)

			return{
				'enc_w': self.W_.eval(),
				'enc_b': self.bh_.eval(),
				'dec_b': self.bv_.eval(),
			}

	def _create_data_directories(self):
		''' Create the three directories for storing respectively the models,
		the data generated by training and the TensorFlow's summaries.

		Returns
		-------
		tuple of strings (models_dir, data_dir, summary_dir)
		'''

		self.main_dir = self.main_dir + '/' if self.main_dir[-1] != '/' else self.main_dir

		models_dir = config.models_dir + self.main_dir
		data_dir = config.data_dir + self.main_dir
		summary_dir = config.summary_dir + self.main_dir

		for d in [models_dir, data_dir, summary_dir]:
			if not os.path.isdir(d):
				os.mkdir(d)

		return models_dir, data_dir, summary_dir


	def get_weights_as_images(self, width, height, outdir='img/', max_images=10, model_path=None):
		''' Saves the weights of this autoencoder as images, one image per hidden unit. 
		This is useful to visualize what the autoencoder has learned

		Parameters
		----------
		width 		: width of the images, int
		height 		: height of the images, int
		outdir 		: output directory for the images -- this path is appended to self.data_dir, string (default is 'data/sdae/img')
		max_images	: number of images to return, int (default is 10)
		'''

		assert max_images <= self.n_components

		outdir = self.data_dir + outdir

		if not os.path.isdir(outdir):
			os.mkdir(outdir)

		with tf.Session() as self.tf_session:
			if model_path is not None:
				self.tf_saver.restore(self.tf_session, model_path)
			else:
				self.tf_saver.restore(self.tf_session, self.models_dir + self.model_name)

			enc_weights = self.W_.eval()

			perm = np.random.permutation(self.n_components)[:max_images]

			for p in perm:
				enc_w = np.array([i[p] for i in enc_weights])
				image_path = outdir + self.model_name + '-enc_weights_{}.png'.format(p)
				utils.gen_image(enc_w, width, height, image_path)




