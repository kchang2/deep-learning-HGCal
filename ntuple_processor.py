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
	print n_entries

	# storing entries
	for i in xrange(0, n_entries):
		tree.GetEntry(i)
		eventArray	= []
		names		= ""

		# event
		event 		= False
		e_run		= []
		e_evn		= []
		e_ngen		= []
		e_nhit		= []
		e_nclus2d	= []
		e_nclus3d	= []
		e_vx 		= []
		e_vy		= []
		e_vz		= []

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

		if event:
			e_run.append(tree.event.run)
			e_evn.append(tree.event.evn)
			e_ngen.append(tree.event.ngen)
			e_nhit.append(tree.event.nhit)
			e_nclus2d.append(tree.event.nclus2d)
			e_nclus3d.append(tree.event.nclus3d)
			e_vx.append(tree.event.vx)
			e_vy.append(tree.event.vy)
			e_vz.append(tree.event.vz)
			events_array = np.core.records.fromarrays([e_run, e_evn, e_ngen, \
				e_nhit, e_nclus2d, e_nclus3d, e_vx, e_vy, e_vz], \
				names = 'run, evn, ngen, nhit, nclus2d, nclus3d, vx, vy, vz')
			eventArray.append(events_array)
			names += 'events'
		else:
			eventArray.append([])

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
				names='layer, x, y, z, eta, phi, energy, time, thickness, isHalf, flags, cluster2d')
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
 

if __name__ == "__main__":
	if args.outdir == None:
		if not os.path.exists("data"):
			os.makedirs("data")
		outdir = os.getcwd() + '/data/'
	else:
		outdir = args.outdir

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


	# Format files for dAe (rechit only)
	# dirname = os.path.basename(os.path.normpath(directory)) + '_formatted'
	# os.mkdir(outdir + dirname + '/')

	# # get list of .npy files created prior
	# files = [f for f in os.listdir(outdir) if f.endswith('.npy')]

	# thickness = []
	# # go through the first file to get characteristics of wafers in the layer
	# for event in file[0]:



	# # organized array (by layer selection, wafer characteristics, cell, energy)
	# outArray_o = []

	# # loops through each file and extract layer 10 info
	# for f in files:
	# 	arr = np.load(f)


	# 	for event in arr:






















 #    mu=np.mean(result)
 #    sav=np.save('%s' %(task),mu)



	raw_input("Operation completed\nPress enter to return to the terminal.")
