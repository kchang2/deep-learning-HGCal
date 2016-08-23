# Kai Chang - Caltech CMS-CERN 2016
#
# Program takes the root file created from the reco-ntuplizer for the HGC 
# and produces appropriate numpy arrays for the trees inside. 
# This is the hardcore version of the former root_processor.py
# Has to run on cmsenv.
#
#
# Blocks (BLOCKS) of code taken from RootProcessor.py
# https://github.com/bencbartlett/tVertexing/blob/master/RootProcessor.py
# =============================================================================

'''
Usage (in cmsenv): RootProcessor.py <[file or directory]> <[options]>

See parser for more details.
'''

import argparse
import ROOT as rt
import numpy as np
import os, sys, platform
import glob
import sys

sys.path.insert(0, '/Users/kaichang/Documents/summer_2016/deep-learning/utils')
import wafer

from FastProgressBar import progressbar


# Argument parser
parser = argparse.ArgumentParser(description = 
'''Converts HGCROIAnalysis-format .root files into RecHit numpy arrays.
	If converting all files in a directory using -f, make sure all files are HGCROI-formatted.''',\
	epilog = '''>> Example: python RootProcessor.py -f ~/public/testfile.root -o ~/public/data
	NOTE: This program must be run while cmsenv is active.

''')

parser.add_argument("-d", "--directory", \
	help="Process all files in this directory. Default is current path.", action="store")
parser.add_argument("-o", "--outdir", \
	help="Directory to save output to. Defaults to Data/.", action="store")
parser.add_argument("-t", "--thickdir", \
	help="location of thickness separation array.", action="store")
args = parser.parse_args()


def process(f, outdir):
	'''
	Usage: process(fileName, writeDirectory)

	Takes a HGCROI-formatted .root file and converts it to a list of structured numpy arrays.

	The format is as follows:
		List of events
			For each event, array for various data structures (RecHits, Clusters, etc.).
			Unfortunately, there is no way to do recursive recarrays, so we use the following
			organisation method:
			eventArray[0]=event
			eventArray[1]=rechits
				For each data structure array, recarray of properties

	For example:
		(Index 0 = Event 0)
			(Index 2 = RecHits)
				'x' -> [1.23, 4.25, ...]
				'y' -> [5.24, 6.42, ...]
				...
				'clusterID' -> [1, 2, 1, 1, ...]
			(Index 1 = Clusters)
				'centerX' -> [3.21, 2.56, ...]
				...
			...
		(Index 1 = Event 1)
			(Index 0 = RecHits)
				...
			(Index 1 = Clusters)
				...
			...
		...

	So, for example, to access the array of x positions in the RecHits of the 7th event,
	you would use:
		xpos = Data[7][2]['x']
	'''
	# initialization
	print "Processing file: " + str(f) + "..."
	outArray = []								# produced array
	fIn = rt.TFile.Open(f)						# open .root file
	tree = fIn.Get('ana/hgc')					# find tree
	n_entries = tree.GetEntries()				# get total # of entries
	print 'number of events: ' n_entries

	# storing entries
	for i in xrange(0, n_entries):
		tree.GetEntry(i)
		eventArray	= []
		names		= ""

		# rechits
		rechits 		= True
		r_layer			= []
		r_wafer			= []
		r_cell			= []
		r_x				= []
		r_y				= []
		r_z				= []
		r_eta			= []
		r_phi			= []
		r_energy		= []
		r_time			= []
		r_thickness		= []
		r_isHalf		= []
		r_flags			= []
		r_cluster2d		= []

		if rechits:
			for hit in tree.rechits:
				r_layer.append(hit.layer)
				r_wafer.append(hit.wafer)
				r_cell.append(hit.cell)
				r_x.append(hit.x)
				r_y.append(hit.y)
				r_z.append(hit.z)
				r_eta.append(hit.eta)
				r_phi.append(hit.phi)
				r_energy.append(hit.energy)
				r_time.append(hit.time)
				r_thickness.append(hit.thickness)
				r_isHalf.append(hit.isHalf)
				r_flags.append(hit.flags)
				r_cluster2d.append(hit.cluster2d)
			rechits_array = np.core.records.fromarrays([r_layer, r_wafer, r_cell, r_x, r_y, r_z, \
				r_eta, r_phi, r_energy, r_time, r_thickness, r_isHalf, r_flags, r_cluster2d], \
				names='layer, wafer, cell, x, y, z, eta, phi, energy, time, thickness, isHalf, flags, cluster2d')
			eventArray.append(rechits_array)
			names += 'rechits'
		else:
			eventArray.append([])

		# combine arrays into single event and append to outArray
		outArray.append(eventArray)


	# save final result to file
	filename = str(f[:-5]) + '.npy'				# replaces .root with .npy
	print filename
	filename = filename.split("/")[-1]			# removes directory prefixes
	filepath = outdir+filename
	print 'writing file ' + os.path.abspath(filepath) + '...'
	np.save(filepath, outArray)					# saves array into .npy file
	print 'Processing complete. \n'



def create_feeding_arrays(files, t_list, outdir):
	''' Merge data into three numpy datasets sorted by wafer thickness.
		Currently only using layer 10 data.

	Parameters
	----------
	t_list	:	Thickness list
	outdir	:	Location where dae-ready npy file will be located
	fname	:	name of dae npy file

	'''

	# creating thickness arrays formatted for autoencoding
	t100 = []
	t200 = []
	t300 = []

	# creating organized array for picture reconstruction
	outArray = []

	# creating wafers with appropriate cell count
	# l_array = [[] for i in range(0, 28)]			# 28 layers in HGCal
	w_array = [[] for i in range(0, 461)]			# 461 wafers in layer 10 of HGCal
	c1_array = [0 for i in range(240)]				# thinner wafers, nominally 256
	c23_array = [0 for i in range(133)]				# thicker wafers, nominally 128, energy in GeV


	for idx, t in np.ndenumerate(t_list):
		print t
		if idx == 0:
			c_array = c1_array
			empty_array = [0 for i in xrange(240)]
		else:
			c_array = c23_array
			empty_array = [0 for i in xrange(133)]

		for i in t:
			w_array[i] = deepcopy(c_array)


	for fi in files:
		f = np.load(fi)

		for event in f:
			layer = deepcopy(w_array)			# create a clean layer 10 for each event

			for hit in event[0]:
				if hit['layer'] == 10: 
					try:
						layer[hit['wafer']][hit['cell']] = hit['energy']		# both index at 0
					except IndexError:
						print 'Index (%i,%i) out of range' %(hit['wafer'], [hit['cell'])

			outArray.append(layer)				#append layer 10 instance to outArray

			for indx, w in enumerate(layer):
				if w != empty_array:
					if indx in t_list[0]:
						t100.append(w)
					if indx in t_list[1]:
						t200.append(w)
					if indx in t_list[2]:
						t300.append(w)
					else:
						pass


	print 'writing autoencoding formatted file'

	# save 100um dataset to file
	filename = str(f[0][:-5]) + 't100.npy'		# replaces .root with .npy
	print filename
	filename = filename.split("/")[-1]			# removes directory prefixes
	filepath = outdir+filename
	print 'writing file ' + os.path.abspath(filepath) + '...'
	np.save(filepath, t100)					# saves array into .npy file

	# save 200um dataset to file
	filename = str(f[0][:-5]) + 't200.npy'		# replaces .root with .npy
	print filename
	filename = filename.split("/")[-1]			# removes directory prefixes
	filepath = outdir+filename
	print 'writing file ' + os.path.abspath(filepath) + '...'
	np.save(filepath, t200)					# saves array into .npy file

	# save 300um dataset to file
	filename = str(f[0][:-5]) + 't300.npy'		# replaces .root with .npy
	print filename
	filename = filename.split("/")[-1]			# removes directory prefixes
	filepath = outdir+filename
	print 'writing file ' + os.path.abspath(filepath) + '...'
	np.save(filepath, t300)					# saves array into .npy file


	print 'Process complete.'
	    




if __name__ == "__main__":
	if args.outdir == None:
		if not os.path.exists("data"):
			os.makedirs("data")
		outdir = os.getcwd() + '/data/'
	else:
		outdir = args.outdir

	if args.thickdir == None:
		t_list = wafer.genThickness()
	else:
		t_list = np.load(thickdir)

	if args.directory == None:
		directory = os.getcwd()
	else:
		directory = args.directory
	os.chdir(directory)

	files = [f for f in os.listdir(directory) if f.endswith('.root')]

	for file in files:
		try:
			process(file, outdir)
		except TypeError:
			raise

	# this will store each simulation (RECO) as a .npy where the np to tensor program will loop through each .npy
	# and save them into an array for that .npy, and then merge all these arrays so we have independent data for
	# the same layer/wafer but different simulation.

	# move back to appropriate folder
	os.chdir(outdir)

	if not os.path.exists("ae_fmt")
		os.makedirs("ae_fmt")
		outdir = os.getcwd() + '/ae_fmt/'

	files = [f for f in os.listdir('.') if f.endswith('.npy')]
	# files = [f for f in files if '12' not in f]


	try:
		create_feeding_arrays(files, t_list, outdir)
	except TypeError:
		raise

	raw_input("Operation completed\nPress enter to return to the terminal.")
