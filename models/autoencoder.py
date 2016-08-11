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

import utils

class DenoisingAutoencoder(object):
	''' Implementation of Denoising Autoencoder using TensorFlow
	The interface of the class is similar to sklearn
	'''

	def __init__(self, 
				model_name='dae', 
				n_components=256, 
				models_dir='dae/models/',
				data_dir='dae/data/',
				summary_dir='dae/summary/', 
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
				seed=-1,
				l2reg=0,
				W_=None,
				bh_=None,
				bv_=None
				):
		'''
		Parameters
		----------
		model_name 			: name of model to use, used to save data
		n_components 		: number of hidden units (number of components to keep)
		models_dir 			: directory to store models
		data_dir 			: directory to store data
		summary_dir 		: directory to store summary
		enc_act_func 		: activation function for the encoder (ie. tanh, sigmoid)
		dec_act_function 	: activation function for the decoder (ie. tanh, sigmoid, none)
		loss_func 			: loss function (ie. mean_squared, cross_entropy) used to measure degree of fit
		calc_acc			: whether or not to print the accuracy of our reconstructed results
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
		l2reg				: regularization parameter. If 0, then no regularization.
		W_ 					: weight of the autoencoder
		bh_ 				: bias of the encoder vector
		bv_ 				: bias of the decoder vector
	
	
		Note these are optional
		-----------------------
		batch_size - default 10
		learning rate - default 0.01
		num_epochs - default 10
		verbose - default 0
		l2reg - default 0
		'''

		# initialization (all set from input or creation of value)
		self.model_name = model_name
		self.n_components = n_components
		self.models_dir = models_dir
		self.data_dir = data_dir
		self.summary_dir = summary_dir
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
		self.W_ = W_
		self.bh_ = bh_
		self.bv_ = bv_

		# random seeder for hyperparameters
		if self.seed >= 0:
			np.random.seed(self.seed)
			tf.set_random_seed(self.seed)

		# model set path directory
		self.model_path = self.models_dir + self.model_name

		# storage objects for tensorflow variables
		self.input_data = None
		self.input_data_corr = None

		self.encode = None
		self.decode = None

		self.train_step = None
		self.cost = None
		self.accuracy = None

		# tensorflow objects
		self.tf_graph = tf.Graph()
		self.tf_session = None
		self.tf_merged_summaries = None
		self.tf_summary_writer = None
		self.tf_saver = None


	def fit(self, train_set, validation_set=None, restore_previous_model=False, graph=None):
		''' Fit the model to the data.
			See keras fit model: https://github.com/fchollet/keras/blob/master/keras/models.py

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
		g = graph if graph is not None else self.tf_graph

		with g.as_default():
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

		self.tf_summary_writer = tf.train.SummaryWriter(self.summary_dir, self.tf_session.graph)
		# train_writer = tf.train.SummaryWriter(self.summary_dir + '/train', self.tf_session.graph)
		# test_writer = tf.train.SummaryWriter(self.summary_dir + '/test', self.tf_session.graph)

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
			# print("Reconstruction loss at step %s: %s" % (epoch, err))


	def _build_model(self, n_features):
		''' Creates the computational graph.

		Parameters
		----------
		n_features	: number of features, int
		regtype 	: regularization type
		W_ 			: weight of matrix, np array
		bh_ 		: hidden bias, np array
		bv_ 		: visible bias, np array

		Returns
		-------
		self
		'''

		self.input_data, self.input_data_corr = self._create_placeholders(n_features)
		self.W_, self.bh_, self.bv_ = self._create_variables(n_features)

		self._create_encode_layer()
		self._create_decode_layer()

		self._create_cost_function_node()
		# self._create_accuracy_node()
		self._create_variable_node(self.W_, 'weight')
		self._create_variable_node(self.bh_, 'hidden bias')
		self._create_variable_node(self.bv_, 'visible bias')

		self._create_train_step_node()



	def _create_placeholders(self, n_features):
		''' Creates the TensorFlow placeholders for the model.
		Parameters
		----------
		n_features	: number of features, int

		Returns
		-------
		tuple ( input_data( shape(None, n_features)),
				input_data_corr( shape(None, n_feature)))
		'''

		input_data = tf.placeholder(tf.float32, [None, n_features], name='x-input')
		input_data_corr = tf.placeholder(tf.float32, [None, n_features], name='x-corr-input')

		# unsupervised mode
		#input_labels = tf.placeholder(tf.float32)
		#keep_prob = tf.placeholder(tf.float32, name='keep-probs')
		return input_data, input_data_corr #, input_labels, keep_prob


	def _create_variables(self, n_features):
		''' Create the TensorFlow variables for the model.
		Parameters
		----------
		n_features	: number of features, int

		Returns 
		-------
		tuple ( weights( shape(n_features, n_components)),
				hidden bias( shape(n_components)),
				visible bias( shape(n_features)))
		'''
		if self.W_:
			W_ = tf.Variable(self.W_, name='enc-w')
		else:
			W_ = tf.Variable(utils.xavier_init(n_features, self.n_components, self.xavier_init), name='enc-w')
			#self.W_ = tf.Variable(tf.trucated_normal(shape=[n_features, self.n_components], stdddev=0.1), name='enc-w')

		if self.bh_:
			bh_ = tf.Variable(self.bh_, name='hidden-bias')
		else:
			bh_ = tf.Variable(tf.constant(0.1, shape=[self.n_components]), name='hidden-bias')

		if self.bv_:
			bv_ = tf.Variable(self.bv_, name='visible-bias')
		else:
			bv_ = tf.Variable(tf.constant(0.1, shape=[n_features]), name='visible-bias')

		return W_, bh_, bv_


	def _create_encode_layer(self):
		''' Create the encoding layer of the network. 
			The encoded layer is the encoded representation of the input

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
				self.encode = tf.matmul(self.input_data, self.W_) + self.bh_


	def _create_decode_layer(self):
		''' Create the decoding layer of the network.
			The decoded is the lossy reconstruction of the input

		Returns
		-------
		self
		'''

		with tf.name_scope('Wg_y_bv'):
			if self.dec_act_func == 'sigmoid':
				self.decode = tf.nn.sigmoid(tf.matmul(self.encode, tf.transpose(self.W_)) + self.bv_)
				_ = tf.histogram_summary('decoding layer -- sigmoid', self.decode)

			elif self.dec_act_func == 'tanh':
				self.decode = tf.nn.tanh(tf.matmul(self.encode, tf.transpose(self.W_)) + self.bv_)
				_ = tf.histogram_summary('decoding layer -- tanh', self.decode)

			else:
				self.decode = tf.matmul(self.encode, tf.transpose(self.W_)) + self.bv_
				_ = tf.histogram_summary('decoding layer -- None', self.decode)


	def _create_cost_function_node(self, regterm=None):
		''' create the cost function node of the network
		Important variables
		-------------------
		self.decode 		: reconstructed image, or the model output node
		self.input_data 	: original image, or reference input placeholder node
		self.regterm 		: regularization term 			# NOT NEEDED FOR NOW

		Returns
		-------
		self
		'''

		with tf.name_scope('cost'):
			if self.loss_func == 'cross_entropy':
				cost = -tf.reduce_sum(self.input_data * tf.log(self.decode))

			elif self.loss_func == 'softmax_cross_entropy':
				softmax = tf.nn.softmax(self.decode)
				cost = - tf.reduce_mean(self.input_data * tf.loag(softmax) + (1 - self.input_data) * tf.log(1 - softmax))

			elif self.loss_func == 'mean_squared':
				cost = tf.sqrt(tf.reduce_mean(tf.square(self.input_data - self.decode)))

			else:
				cost = None

		if cost is not None:
			self.cost = cost + regterm if regterm is not None else cost
			_ = tf.scalar_summary(self.loss_func, self.cost)
		else:
			self.cost = None


	def _create_accuracy_node(self):
		''' create the accuracy node of the network.
		** This is not needed for the dAe **

		Returns
		-------
		self
		'''
		with tf.name_scope('accuracy'):
			if self.calc_acc:
				with tf.name_scope('correct_prediction'):
					correct_prediction = tf.equal(tf.argmax(self.decode, 1), tf.argmax(self.input_data, 1))
				with tf.name_scope('accuracy'):
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

			elif self.opt == 'adam':
				self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

			else:
				self.train_step = None


	def transform(self, data, name='train-transform', graph=None, save=False):
		''' Transforms data (corrupted) according to the model. [Encode]
			It should return whatever end encoder node/layer (with final minimally-optimized n-components) image it
			converts your images to. This layer should have n-components << first layer n-components

		Parameters
		----------
		data 	: data to transform
		name 	: identifier for the data that is being encoded
		graph 	: tf graph objects
		save 	: if true, save data to disk

		Important Variables
		-------------------
		input_data_corr 	: the corrupted info put into the transform method, want to see outcome from our model on corrupted set

		Returns
		-------
		transformed data
		'''

		g = graph if graph is not None else self.tf_graph

		with g.as_default():
			with tf.Session() as self.tf_session:
				self.tf_saver.restore(self.tf_session, self.model_path)
				encoded_data = self.encode.eval({self.input_data_corr: data})
				if save:
					np.save(self.data_dir + self.model_name + '-' + name, encoded_data)

				return encoded_data


	def reconstruct(self, data, name='train-reconstruct', graph=None, save=False):
		''' Reconstruct the data (corrupted) using the learned model. [Decode]
			This should return the final result from the entire process of encoding to decoding.
			In this layer the n-components == starting n-components. 

		Parameters
		----------
		data 		: Data to reconstruct
		graph		: tf graph objects
		name 		: identifier for the data that is being reconstructed
		save 		: if true, saves data to disk

		Returns
		-------
		labels
		'''

		g = graph if graph is not None else self.tf_graph

		with g.as_default():
			with tf.Session() as self.tf_session:
				self.tf_saver.restore(self.tf_session, self.model_path)
				decoded_data = self.decode.eval({self.input_data_corr: data})
				if save:
					np.save(self.data_dir + self.model_name + '-' + name, decoded_data)

				return decoded_data


	def compute_reconstruction_loss(self, data, data_ref, graph=None):
		''' Computes the reconstruction loss over the chosen dataset (test). [Decoded vs. original]

		Parameters
		----------
		data 		: corrupted data to reconstruct
		data_ref	: original data to check
		graph 		: tf graph objects

		Returns
		-------
		labels
		'''

		g = graph if graph is not None else self.tf_graph

		with g.as_default():
			with tf.Session() as self.tf_session:
				self.tf_saver.restore(self.tf.session, self.model_path)
				return self.cost.eval({self.input_data_corr: data, self.input_data: data_ref})


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


	def get_model_parameters(self, graph=None):
		''' Return the model parameters in the form of numpy arrays.

		Returns
		-------
		model parameters
		'''
		g = graph if graph is not None else self.tf_graph

		with g.as_default():
			with tf.Session() as self.tf_session:
				self.tf_saver.restore(self.tf_session, self.models_dir + self.model_name)

				return{
					'enc_w': self.W_.eval(),
					'enc_b': self.bh_.eval(),
					'dec_b': self.bv_.eval(),
				}


	def get_weights_as_images(self, width, height, main_dir='dae/', outdir='img/', n_images=10, model_path=None, img_type='grey'):
		''' Saves the weights of this autoencoder as images, one image per hidden unit. 
		This is useful to visualize what the autoencoder has learned

		Parameters
		----------
		width 		: width of the images, int
		height 		: height of the images, int
		main_dir	: path where all your encoding information is placed
		outdir 		: output directory for the images -- this path is appended to self.data_dir, string (default is 'data/sdae/img')
		n_images	: number of images to return, int (default is 10)
		'''

		assert n_images <= self.n_components

		main_dir = main_dir + '/' if main_dir[-1] != '/' else main_dir
		outdir = outdir + '/' if outdir[-1] != '/' else outdir
		outdir = main_dir + outdir
		
		if not os.path.isdir(outdir):
			os.mkdir(outdir)


		with tf.Session() as self.tf_session:
			if model_path is not None:
				self.tf_saver.restore(self.tf_session, model_path)
			else:
				self.tf_saver.restore(self.tf_session, self.model_path)

			enc_weights = self.W_.eval()

			perm = np.random.permutation(enc_weights.shape[1])[:n_images]
			for p in perm:
				enc_w = np.array([i[p] for i in enc_weights])
				image_path = outdir + self.model_name + '-enc_weights_{}.png'.format(p)
				utils.gen_image(enc_w, width, height, image_path, img_type)




