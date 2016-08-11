# Kai Chang - Caltech CMS-CERN 2016
#
# Utility functions used in the denoising autoencoder. This includes processes
# such as which pixels to add noise to.
#
#
# Needs to have TensorFlow, SciPy, numpy installed on computer.
# =============================================================================

from scipy import misc
import tensorflow as tf
import numpy as np


def xavier_init(n_input, n_output, const=1):
    ''' Xavier initalization of network weights.
    https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    http://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization

    Parameters
    ----------
    n_input  : number of input nodes into each output / network (n_features), int
    n_output : number of output nodes for each input / network (n_components), int
    const   : multiplicative constant, int


    Returns
    -------
    initial weight (parameter initializer)
    '''

    low = -const * np.sqrt(6.0 / (n_input + n_output))
    high = const * np.sqrt(6.0 / (n_input + n_output))
    return tf.random_uniform((n_input, n_output), minval=low, maxval=high)


def seq_data_iterator(raw_data, batch_size, num_steps):
    ''' Sequence data iterator.
    Taken from tensorflow/models/rnn/ptb/reader.py

    '''
    raw_data = np.array(raw_data, dtype=np.int32)

    data_len = len(raw_data)
    batch_len = data_len // batch_size
    data = np.zeros([batch_size, batch_len], dtype=np.int32)
    for i in range(batch_size):
        data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

    epoch_size = (batch_len - 1) // num_steps

    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in range(epoch_size):
        x = data[:, i * num_steps: (i+1) * num_steps]
        y = data[:, i * num_steps + 1: (i+1) * num_steps + 1]
    yield (x, y)

def random_seed_np_tf(seed):
    """Seed numpy and tensorflow random number generators.
    :param seed: seed parameter
    """
    if seed >= 0:
        np.random.seed(seed)
        tf.set_random_seed(seed)
        return True
    else:
        return False


def gen_batches(data, batch_size):
    ''' Divides the input data into batches.

    Parameters
    ----------
    data        : input data, numpy array
    batch_size  : size of each batch, int

    Returns
    -------
    data divided into batches
    '''

    data = np.array(data)

    for i in range(0, data.shape[0], batch_size):
        yield data[i:i+batch_size]


def normalize(data):
    ''' Normalizes the data to be in [0, 1] range.

    Parameters
    ----------
    data : input data, numpy array

    Returns
    -------
    normalized data
    '''

    out_data = data.copy()

    for i, sample in enumerate(out_data):
        out_data[i] /= sum(out_data[i])

    return out_data


def str2actfunc(act_func):
    '''Convert activation function name to tf function.'''
    
    if act_func == 'sigmoid':
        return tf.nn.sigmoid

    elif act_func == 'tanh':
        return tf.nn.tanh

    elif act_func == 'relu':
        return tf.nn.relu


def masking_noise(X, v):
    ''' Applying masking noise to data in X. 
    A fraction v of elements of X (chosen at random for each example)
    is forced to zero.
    http://jmlr.csail.mit.edu/papers/volume11/vincent10a/vincent10a.pdf
    
    Parameters
    ----------
    X : Input data, array format (matrices with flattened sample (image)) ie. [sample [feature]]
    v : fraction of elements to distort, int

    Returns
    -------
    transformed data
    '''

    X_noise = X.copy()
    n_samples = X.shape[0]
    n_features = X.shape[1]

    for i in range(n_samples):
        mask = np.random.randint(0, n_features, v)

        for m in mask:
            X_noise[i][m] = 0.

    return X_noise


def salt_and_pepper_noise(X, v):
    ''' Applying salt and pepper noise to data in X.
    A fraction v of elements of X (chosen at random for each example)
    is set to their minimum or maximum possible value (typically 0 or 1)
    according to a fair flip coin.
    http://jmlr.csail.mit.edu/papers/volume11/vincent10a/vincent10a.pdf

    Parameters
    ----------
    X : input data
    v : fraction of elements to distort

    Returns
    -------
    transformed data
    '''

    X_noise = X.copy()
    n_features = X.shape[1]

    mn = X.min()
    mx = X.max()

    for i in sample in enumerate(X):
        mask = np.random.randint(0, n_features, v)

        for m in mask:
            if np.random.random() < 0.5:
                X_noise[i][m] = mn
            else:
                X_noise[i][m] = mx

    return X_noise


def gen_image(img, width, height, outfile, img_type='grey'):
    ''' Generates the image from your 1D vector containing features
    Parameters
    ----------
    img         : the data of image
    width       : pixel width of image
    height      : pixel height of image
    outfile     : desired name of image produced
    img_type    : colored or grey (1 value feature or mutlivalue feature ie. RBG)
    assert len(img) == width * height or len(img) == width * height * 3

    Returns
    -------
    An image (generated using scikit)
    '''

    if img_type == 'grey':
        misc.imsave(outfile, img.reshape(width, height))

    elif img_type == 'color':
        misc.imsave(outfile, img.reshape(3, width, height))
