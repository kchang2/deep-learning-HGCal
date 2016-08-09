from keras.datasets import mnist

from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model

import matplotlib.pyplot as plt
import numpy as np

# load dataset
(x_train, _), (x_test, _) = mnist.load_data()						# 1 long array


# format dataset
x_train = x_train.astype('float32') / 255. 							# normalize values between 0 and 1
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 1, 28, 28)) 			# reshape dataset to be images of 28 x 28 pixels
x_test = np.reshape(x_test, (len(x_test), 1, 28, 28))


# apply corruption or noise factor
noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape) 		# gaussian distribution noise matrix
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape) 

x_train_noisy = np.clip(x_train_noisy, 0., 1.)						# clip images between 0 and 1
x_test_noisy = np.clip(x_test_noisy, 0., 1.)


# display noisy image
n = 10
plt.figure(figsize=(20, 2))
for i in range(n):
	# display corrupted image
    ax = plt.subplot(1, n, i)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


# take a single image -- this is our input placeholder
input_img = Input(shape=(1, 28, 28))

# the encoded reprentation of the input
# Stacking with filtering layers
x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(input_img)			#relu = rectifier for NN
x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(x)
encoded = MaxPooling2D((2, 2), border_mode='same')(x)


# at this point the representation is (32, 7, 7) ie. 1568-dimensional
# the loss reconstruction of the input
x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Convolution2D(1, 3, 3, activation='sigmoid', border_mode='same')(x)

# we map the model an input to its reconstruction
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# train for 100 epoch
autoencoder.fit(x_train_noisy, x_train,
                nb_epoch=100,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test_noisy, x_test))



# displays reconstructed models:
x_test_encoded = autoencoder.predict(x_test, batch_size=batch_size)

for i in range(n):
	# display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(x_test_encoded[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
