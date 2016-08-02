# Kai Chang - Caltech CMS-CERN 2016
#
# Program takes any root file and produces appropriate numpy arrays
# for the trees inside. This is the hardcore version of it
#
# lots of blocks of code taken from RootProcessor.py
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

parser.add_argument("-f", "--folder", \
	help="Process all files in this directory.", action="store_true")
parser.add_argument("input", \
	help="File to process. If -f is used, full directory to process.")
parser.add_argument("-o", "--outdir", \
	help="Directory to save output to. Defaults to Data/.", action="store")
args = parser.parse_args()


# helper class to make multiprocessing progress bars
class Writer(object):
    def __init__(self, location):
        # Input: location - tuple of ints (x, y), the position of the bar in the terminal
        self.location = location
    def write(self, string):
        with term.location(*self.location):
			print(string)


def process((f, outdir, showProgress)):
	'''
	Usage: process(fileName, writeDirectory)

	Takes a HGCROI-formatted .root file and converts it to a list of structured numpy arrays.

	The format is as follows:
		List of events
			For each event, array for various data structures (RecHits, Clusters, etc.).
			Unfortunately, there is no way to do recursive recarrays, so we use the following
			organisation method:
			eventArray[0]=event
			eventArray[1]=AGenPart
			eventArray[2]=ARecHit
			eventArray[2]=ACluster2d
			eventArray[3]=AMultiCluster
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

	# progress bar
	pbar = progressbar("{0:22s} &count&: ".format(str(f)), n_entries + 1)
	pbar.start()

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

		# AGenPart
		AGenPart	= True
		AG_eta 		= []
		AG_phi 		= []
		AG_pt 		= []
		AG_dvx 		= []
		AG_dvy 		= []
		AG_dvz 		= []
		AG_pid 		= []

		# ARecHit
		ARecHit 	= True
		AR_layer	 	= []
		AR_x 			= []
		AR_y 			= []
		AR_z 			= []
		AR_eta 			= []
		AR_phi 			= []
		AR_energy	  	= []
		AR_time 		= []
		AR_thickness 	= []
		AR_isHalf 		= []
		AR_flags 		= []
		AR_cluster2d 	= []

		# ACluster2d
		ACluster2d 	= True
		AC_x 			= []
		AC_y 			= []
		AC_z 			= []
		AC_eta 			= []
		AC_phi	 		= []
		AC_energy 		= []
		AC_layer 		= []
		AC_nhitCore 	= []
		AC_nhitAll 		= []
		AC_multicluster = []

		# AMultiCluster
		AMultiCluster	= True
		AM_eta 			= []
		AM_phi	 		= []
		AM_energy 		= []
		AM_nclus 		= []

		if event:
			for hit in tree.event:
				e_run.append(hit.run)
				e_evn.append(hit.evn)
				e_ngen.append(hit.ngen)
				e_nhit.append(hit.nhit)
				e_nclus2d.append(hit.nclus2d)
				e_nclus3d.append(hit.nclus3d)
				e_vx.append(hit.vx)
				e_vy.append(hit.vy)
				e_vz.append(hit.vz)
			AGenPart_array = np.core.records.fromarrays([e_run, e_evn, e_ngen, \
				e_nhit, e_nclus2d, e_nclus3d, e_vx, e_vy, e_vz], names='run, evn, \
				ngen, nhit, nclus2d, nclus3d, vx, vy, vz')
			eventArray.append(AGenPart_array)
			names += 'AGenPart'
		else:
			eventArray.append([])

		if AGenPart:
			for hit in tree.AGenPart:
				AG_eta.append(hit.eta)
				AG_phi.append(hit.phi)
				AG_pt.append(hit.pt)
				AG_dvx.append(hit.dvx)
				AG_dvy.append(hit.dvy)
				AG_dvz.append(hit.dvz)
				AG_pid.append(hit.pid)
			AGenPart_array = np.core.records.fromarrays([AG_eta, AG_phi, AG_pt, \
				AG_dvx, AG_dvy, AG_dvz, AG_pid], names='eta, phi, pt, dvx, dvy, dvz, pid')
			eventArray.append(AGenPart_array)
			names += 'AGenPart'
		else:
			eventArray.append([])

		if ARecHits:
			for hit in tree.RecHits:
				AR_layer.append(hit.layer)
				AR_x.append(hit.x)
				AR_y.append(hit.y)
				AR_z.append(hit.z)
				AR_eta.append(hit.eta)
				AR_phi.append(hit.phi)
				AR_energy.append(hit.energy)
				AR_time.append(hit.time)
				AR_thickness.append(hit.thickness)
				AR_isHalf.append(hit.isHalf)
				AR_flags.append(hit.flags)
				AR_cluster2d.append(hit.cluster2d)
			ARecHits_array = np.core.records.fromarrays([AR_layer, AR_x, AR_y, AR_z, \
				AR_eta, AR_phi, AR_energy, AR_time, AR_thickness, AR_isHalf, AR_flags, AR_cluster2d], \
				names='x, y, z, eta, phi, energy, time, thickness, isHalf, flags, cluster2d')
			eventArray.append(ARecHits_array)
			names += 'ARecHit'
		else:
			eventArray.append([])

		if ACluster2d:
			for hit in tree.ACluster2d:
				AC_x.append(hit.x)
				AC_y.append(hit.y)
				AC_z.append(hit.z)
				AC_eta.append(hit.eta)
				AC_phi.append(hit.phi)
				AC_energy.append(hit.energy)
				AC_layer.append(hit.layer)
				AC_nhitCore.append(hit.nhitCore)
				AC_nhitAll.append(hit.nhitAll)
				AC_multicluster.ppend(hit.multicluster)
			ACluster2d_array = np.core.records.fromarrays([AC_x, AC_y, AC_z, \
				AC_eta, AC_phi, AC_energy, AC_layer, AC_nhitCore, AC_nhitAll, AC_multicluster], \
				names = 'x, y, z, eta, phi, energy, layer, nhitCore, nhitAll, multicluster')
			eventArray.append(ACluster2d_array)
			names += 'ACluster2d'
		else:
			eventArray.append([])
		
		if AMultiCluster:
			for hit in tree.AMultiCluster:
				AM_eta.append(hit.eta)
				AM_phi.append(hit.phi)
				AM_energy.append(hit.energy)
				AM_nclus.append(hit.nclus)
			AMultiCluster_array = np.core.records.fromarrays([AM_eta, AM_phi, AM_energy,\
				AM_nclus], name= 'eta, phi, energy, numofclus')
			eventArray.append(AMultiCluster_array)
			names += 'AMultiCluster'
		else:
			eventArray.append([])

		# combine arrays into single event and append to outArray
		outArray.append(eventArray)
		pbar.update(i)

		# save final result to file
		pbar.finish()
		filename = str(f[:-5]) + '.npy'				# replaces .root with .npy
		filename = filename.split("/")[-1]			# removes directory prefixes
		filename = outdir+filename
		print 'writing file ' + os.path.abspath(filepath) + '...'
		np.save(filepath, outArray)					# saves array into .npy file
		print 'Processing complete. \n'


if __name__ == "__main__":
	if args.outdir == None:
		outdir = 'Data/'
	else:
		outdir = args.outdir

	showProgress = True

	if args.folder:
		os.system('clear')
		dirs = args.input
		files = [f for f in os.listdir(dirs) if f.endswith('.root')]
		os.chdir(dirs)

		print 'Processing %i files from %s ...' % (len(files), dirs)
		for f in files:
			process((f, outdir, showProgress))

	else:
		dirs, filename = args.input.rsplit("/",1)
		os.system("clear")
		os.chdir(dirs)
		process((filename, outdir, showProgress))

	raw_input("Operation completed\nPress enter to return to the terminal.")



# directory for dataset (Where ntuple tree is)
# dir = '/Users/kaichang/Documents/summer_2016/data/twogamma_pt5_eta2_nosmear_calib_ntuple.root'

#opens up file and looks for hgc tree
f = rt.TFile(dirs)
tree = f.Get('ana/hgc')
