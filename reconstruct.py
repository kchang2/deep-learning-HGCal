# Kai Chang - Caltech CMS-CERN 2016
#
# Takes the resulted reconstructed data with energy and plots the HGCal structure.
#
#
# Needs to have ROOT, NumPy installed on computer.
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import os

import sys
sys.path.insert(0, './utils')
sys.path.insert(0, './data/ae_fmt')
sys.path.insert(0, './dae/data/result')

import plothgc as hgc


# Global Parameters
wafer_positions		= 'wafer_positions.npy'		# every cell position in every wafer
weird_wafer_numbers	= 'weird_wafer.npy'			# all wafers that don't match theoretical thickness to cell count

# Special Parameters
train_cost			= 'pdg12_pt35_t200_feat100_tr_cost.npy'	# training cost per epoch
valid_cost			= 'pdg12_pt35_t200_feat100_vl_cost.npy'	# validation cost per epoch
log					= 0										# logscale for cost. 1 = True

original_src		= 'pdg12_pt35_t200.npy'					# original data source (in autoencoder format)
decoded_src			= 'pdg12_pt35_t200_feat100_dec.npy'		# decoded data source (after fed through autoencoder)
wafer_src			= 'pdg12_pt35_t200_src.npy'				# source to which wafer each set of data is taken from
layer_src			= 'pdg12_pt35_layer10.npy'				# source where layer data for each event is kept
index				= 2										# image in sequence. 0 = first image


def graph_epoch_err(train_err, val_err, log=0):
	x_epoch = range(len(train_err))
	y_train = train_err
	y_valid = val_err

	if log == 1:
		plt.semilogy(x_epoch, y_train,'r--', x_epoch, y_valid, 'k')
	else:
		plt.plot(x_epoch, y_train,'r--', x_epoch, y_valid, 'k')
	plt.xlabel('epoch')
	plt.ylabel('cost')




####### Information gathering ########
# xy position information
os.chdir('./data')
try:
	xy = np.load(wafer_positions)
except:
	pass

# pdg information (original data)
os.chdir('./ae_fmt')
try:
	origX = np.load(original_src)
except:
	pass

# necessary wafer reference information
try:
	wafer_ref = np.load(wafer_src)
	weird_w = np.load(weird_wafer_numbers)
except:
	pass

# pdg information (decoded data)
os.chdir('../../dae/data/result')
try:
	decodedX = np.load(decoded_src)
except:
	pass

# obtain cost information
try:
	train_i = np.load(train_cost)
	valid_i = np.load(valid_cost)
except:
	pass

# necessary layer information
try:
	layer_ref = np.load(layer_src)
except:
	pass

# get to result folder to get ready to plot and save images
os.chdir('../../data/result')

# uncomment for knowing what wafer in layer 10
# print wafer_ref[index]

####### Plotting information ########
# Attempts to plot cost vs. epoch
try:
	graph_epoch_err(train_i, valid_i, log)
	plt.savefig(decoded_src[:-7] + 'err_epoch.png', bbox_inches='tight')
	print 'Saved Cost vs. Epoch plot ...'
	plt.clf()
	plt.close()
except:
	print 'Unable to plot cost vs. epoch. Maybe check filename or location?'

# Attempts to plot the original plot in hexbin
try:
	hgc.plot_hgc_wafer_hexbin(index, origX, wafer_ref, xy, weird_w)
	figname = ('wafer_%i_' + original_src[:-4] + '_orig.png') %wafer_ref[index]
	plt.savefig(figname, bbox_inches='tight') 
	plt.clf()
	plt.close()

	# hgc.plot_hgc_wafer_histo(index, origX, wafer_ref, xy, weird_w)
	# figname = ('wafer_%i_' + original_src[:-4] + '_orig_histo.png') %wafer_ref[index]
	# plt.savefig(figname) 
	# plt.clf()
	# plt.close()

	print 'Saved original rechits hexbin ...'
except:
	print 'Unable to plot original image. Maybe check filename or location?'

# Attempts to plot the decoded plot in hexbin
try:
	hgc.plot_hgc_wafer_hexbin(index, decodedX, wafer_ref, xy, weird_w)
	figname = ('wafer_%i_' + decoded_src[:-4] + '.png') %wafer_ref[index]
	plt.savefig(figname, bbox_inches='tight')
	plt.clf()
	plt.close()

	# hgc.plot_hgc_wafer_histo(index, decodedX, wafer_ref, xy, weird_w)
	# figname = ('wafer_%i_' + decoded_src[:-4] + '_histo.png') %wafer_ref[index]
	# plt.savefig(figname) 
	# plt.clf()
	# plt.close()

	print 'Saved decoded rechits hexbin ...'
except:
	print 'Unable to plot decoded image. Maybe check filename or location?'

# Attempts to plot the original plot in histogram
try:
	hgc.plot_hgc_wafer_histo(index, origX, wafer_ref, xy, weird_w)
	figname = ('wafer_%i_' + original_src[:-4] + '_orig_histo.png') %wafer_ref[index]
	plt.savefig(figname, bbox_inches='tight') 
	plt.clf()
	plt.close()

	print 'Saved original rechits histogram ...'
except:
	print 'Unable to plot original image. Maybe check filename or location?'

# Attempts to plot the decoded plot in histogram
try:
	hgc.plot_hgc_wafer_histo(index, decodedX, wafer_ref, xy, weird_w)
	figname = ('wafer_%i_' + decoded_src[:-4] + '_histo.png') %wafer_ref[index]
	plt.savefig(figname, bbox_inches='tight') 
	plt.clf()
	plt.close()

	print 'Saved decoded rechits histogram ...'
except:
	print 'Unable to plot decoded image. Maybe check filename or location?'


# Attempts to plot the entire layer 10
try:
	hgc.plot_hgc_layer_hexbin(index, layer_ref)
	('layer_%i_' + layer_src[:-4] + '_dec.png') %10
	plt.savefig(figname, bbox_inches='tight')
	print 'Saved entire layer 10 plot ...'
except:
	print 'Unable to plot layer. Maybe check filename or location?'



