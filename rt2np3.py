# Kai Chang - Caltech CMS-CERN 2016
#
# Program takes any root file and produces appropriate numpy arrays
# for the trees inside. This is the hardcore version of it.
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

from FastProgressBar import progressbar


# Argument parser
parser = argparse.ArgumentParser(description = 
'''Converts HGCROIAnalysis-format .root files into RecHit numpy arrays.
	If converting all files in a directory using -f, make sure all files are HGCROI-formatted.''',\
	epilog = '''>> Example: python RootProcessor.py -f ~/public/testfile.root -o ~/public/data
	NOTE: This program must be run while cmsenv is active.

''')

parser.add_argument("-d", "--directory", \
	help="Process all files in this directory.", action="store_true")
parser.add_argument("input_d", \
	help="File to process. If -f is used, process only a single file.")
parser.add_argument("-f", "--file", \
	help="Process all files in this directory.", action="store_true")
parser.add_argument("input_f", \
	help="File to process.")
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
			eventArray[1]=particles
			eventArray[2]=rechits
			eventArray[2]=cluster2d
			eventArray[3]=multicluster
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
		event 		= True
		e_run		= []
		e_evn		= []
		e_ngen		= []
		e_nhit		= []
		e_nclus2d	= []
		e_nclus3d	= []
		e_vx 		= []
		e_vy		= []
		e_vz		= []

		# particles
		particles	= True
		p_eta 		= []
		p_phi 		= []
		p_pt 		= []
		p_dvx 		= []
		p_dvy 		= []
		p_dvz 		= []
		p_pid 		= []

		# rechits
		rechits 	= True
		r_layer	 		= []
		r_x 			= []
		r_y 			= []
		r_z 			= []
		r_eta 			= []
		r_phi 			= []
		r_energy	  	= []
		r_time 			= []
		r_thickness 	= []
		r_isHalf 		= []
		r_flags 		= []
		r_cluster2d  	= []

		# cluster2d
		cluster2d 	= True
		c_x 			= []
		c_y 			= []
		c_z 			= []
		c_eta 			= []
		c_phi	 		= []
		c_energy 		= []
		c_layer 		= []
		c_nhitCore 		= []
		c_nhitAll 		= []
		c_multicluster 	= []

		# multicluster
		multicluster	= True
		m_eta 			= []
		m_phi	 		= []
		m_energy 		= []
		m_nclus 		= []

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

		if particles:
			for hit in tree.particles:
				p_eta.append(hit.eta)
				p_phi.append(hit.phi)
				p_pt.append(hit.pt)
				p_dvx.append(hit.dvx)
				p_dvy.append(hit.dvy)
				p_dvz.append(hit.dvz)
				p_pid.append(hit.pid)
			particles_array = np.core.records.fromarrays([p_eta, p_phi, p_pt, \
				p_dvx, p_dvy, p_dvz, p_pid], names='eta, phi, pt, dvx, dvy, dvz, pid')
			eventArray.append(particles_array)
			names += 'particles'
		else:
			eventArray.append([])

		if rechits:
			for hit in tree.rechits:
				r_layer.append(hit.layer)
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
			rechits_array = np.core.records.fromarrays([r_layer, r_x, r_y, r_z, \
				r_eta, r_phi, r_energy, r_time, r_thickness, r_isHalf, r_flags, r_cluster2d], \
				names='x, y, z, eta, phi, energy, time, thickness, isHalf, flags, cluster2d')
			eventArray.append(rechits_array)
			names += 'rechits'
		else:
			eventArray.append([])

		if cluster2d:
			for hit in tree.cluster2d:
				c_x.append(hit.x)
				c_y.append(hit.y)
				c_z.append(hit.z)
				c_eta.append(hit.eta)
				c_phi.append(hit.phi)
				c_energy.append(hit.energy)
				c_layer.append(hit.layer)
				c_nhitCore.append(hit.nhitCore)
				c_nhitAll.append(hit.nhitAll)
				c_multicluster.append(hit.multicluster)
			cluster2d_array = np.core.records.fromarrays([c_x, c_y, c_z, \
				c_eta, c_phi, c_energy, c_layer, c_nhitCore, c_nhitAll, c_multicluster], \
				names = 'x, y, z, eta, phi, energy, layer, nhitCore, nhitAll, multicluster')
			eventArray.append(cluster2d_array)
			names += 'cluster2d'
		else:
			eventArray.append([])
		
		if multicluster:
			for hit in tree.multicluster:
				m_eta.append(hit.eta)
				m_phi.append(hit.phi)
				m_energy.append(hit.energy)
				m_nclus.append(hit.nclus)
			multicluster_array = np.core.records.fromarrays([m_eta, m_phi, m_energy,\
				m_nclus], names = 'eta, phi, energy, numofclus')
			eventArray.append(multicluster_array)
			names += 'multicluster'
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
		outdir = 'data/'
	else:
		outdir = args.outdir

	if args.directory:
		directory = args.input_d
	else:
		directory = os.getcwd()
	os.chdir(directory)

	if args.file:
		f = args.input_f
	else:
		f = [files for files in os.listdir(directory) if files.endswith(".root")] 



	# f = 'twogamma_pt5_eta2_nosmear_calib_ntuple.root'
	# outdir = '/afs/cern.ch/user/k/kachang/work/public/CMSSW_8_1_0_pre8/src/RecoNtuples/HGCalAnalysis/test/'
	for file in f:
		try:
			process(file, outdir)
		except TypeError:
			raise

	# this will store each simulation (RECO) as a .npy where the np to tensor program will loop through each .npy
	# and save them into an array for that .npy, and then merge all these arrays so we have independent data for
	# the same layer/wafer but different simulation.

	raw_input("Operation completed\nPress enter to return to the terminal.")



# directory for dataset (Where ntuple tree is)
# dir = '/Users/kaichang/Documents/summer_2016/data/twogamma_pt5_eta2_nosmear_calib_ntuple.root'

#opens up file and looks for hgc tree
