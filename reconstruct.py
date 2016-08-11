# Kai Chang - Caltech CMS-CERN 2016
#
# Program reconstructs the HGCal data accordingly
# Used for pre and post (optional) processing.
#
#
# Needs to have scikit, numpy installed on computer.
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np


n = 10
n_array = []

# import unprocessed neutrino dataset
n_data = np.load('./data/neutrino/rechit_formatted.npy')

# add extra zeros
extra_0 = [0 for i in range(17)]

for i in xrange(n):
	arr = np.concatenate((n_data[i], extra_0), axis=0).flatten()
	n_array.append(arr)

n_array = np.array(n_array)

print 'original np shape: ', n_data.shape
print 'ready np shape: ', n_array.shape


plt.figure(figsize=(20, 4)) 	# w,h tuple in inches
for i in range(n):

	# display original
	ax = plt.subplot(2, n, i + 1) 				# nrows, ncolumns, plot_number
	plt.imshow(n_array[i].reshape(16,16))		# displays image on axis
	plt.gray()									# grayscale, WILL CONVERT TO COLOR SCALE w/ colorscale as energy density
	ax.get_xaxis().set_visible(False)			
	ax.get_yaxis().set_visible(False)

plt.show()