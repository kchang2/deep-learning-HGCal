# Kai Chang - Caltech CMS-CERN 2016
#
# Program takes any numpy file and produces appropriate numpy arrays
# for the TensorFlow application. The output should look very similar
# to the MNIST dataset, except now with more than 1 feature.
#
# 
# =============================================================================


import numpy as np
import os, sys, platform

from FastProgressBar import progressbar


# Argument parser
parser = argparse.ArgumentParser(description=
	'''Selects all layers to parse unless specified''')

parser.add_argument("-f", "--file", \
	help="Process all files in this directory.", action="store_true")

parser.add_argument("-i", "--include", \
	help="Include the specific layer you are interested in", action="store_const")

parser.add_argument("-o", "--outdir", \
 	help="Directory to save output to. Defaults to Data/.", action="store")

args = parser.parse_args()


def translate(f, outdir, options):
	'''
	Usage: process(fileName, writeDirectory)

	Takes a dictionary .npy file (list of structured numpy arrays) and converts it
	int a tensor ready .npy.

	The format is as follows:
		List of layers
			For each layer, array for each hexagonal wafer are made containing
			specific traits from RecHits.
			eventArray[0]=first event recorded with wafer j
			eventArray[1]=second event recorded on wafer k 			k != || == j
			eventArray[2]=third event recorded on wafer l 			l != || == k
			.
			.
			.
			eventArray[n]=event n recorded on wafer N

	For example:
		(Index 0 = Layer 0)
			(Index 2 = event 2 RecHit features)
				'x' -> [1.23, 4.25, ...]
				'y' -> [5.24, 6.42, ...]
				...
				'clusterID' -> [1, 2, 1, 1, ...]
			(Index 1 = event 1)
				'centerX' -> [3.21, 2.56, ...]
				...
			...
		(Index 1 = Layer 1)
			(Index 0 = event 0 RecHit features)
				...
			(Index 1 = event 1)
				...
			...
		...
	
	So, for example, to access the feature 'x' in the 2nd event of the 7th layer,
	you would use:
		xpos = Data[7][2]['x']
	'''


	# initialization
	print "Processing file: " + str(f) + "..."
	outArray = []								# produced array
	entryArray = np.load(f)						# open .root file
	t_events = len(entryArray[0])			# total events in the numpy file

	# progressbar
	pbar = progressbar("Optimizing for tensor shaping", total_data)
	pbar.start()


	# there are 5 events or collisions --> a bunch of energy deposits captured (data points)
	# these data points are binned by layer --> binned by wafer
	# each data point has a unique layer, wafer, cell_id
	# but they are binned [layer [wafer [id1 + info1, id2 + info2, id3 + info3, ..., id_n + info_n]]] --> as of now, no layer section
	if options != False:
		layer = options 		# specified layer
		num_of_wafers = max()	# total wafer count in layer
		event = [[] for i in range(0,num_of_wafers)]			# array for each event (should be 5 events/collisions per file)

		for e in xrange(0, t_events):

			for i in xrange(0, len(entryArray[e][2])): 		# rechits total data points captured
				if entryArray[e][2]['layer'][i] != layer:
					continue
				else:
					if event[entryArray[e][2]['detectorID (wafer)'][i]] != []:
						event[entryArray[e][2]['detectorID (wafer)'][i]].append([cell, thickness, energy, isHalf]]
					else:
						event[entryArray[e][2]['detectorID (wafer)'][i]] = [[cell, thickness, energy, isHalf]]

			outArray.append(event)
	else:
		layer = 10
		event = [[] for i in range(0, num_of_wafers)]

	filename = str([:-5]) + '_tensorReady.npy'
	print filename
	filename = filename.split("/")[-1]			# removes directory prefixes
	filepath = outdir+filename
	print 'writing file ' + os.path.abspath(filepath) + '...'
	np.save(filepath, outArray)
	print 'Processing complete. \n'

def testTensor(array):


	# test to see if structure is correct
	if structure:
		return "Passed"
	else:
		return "Failed"



if __name__ == "__main__":

	if args.file == None: # assume in directory
		f = 'twogamma_pt5_eta2_nosmear_calib_ntuple.npy'
	else:
		f = args.file

	if args.outdir == None: # assume in data folder
		outdir = '/Users/kaichang/Documents/summer_2016/data/'
	else:
		outdir = args.outdir

	if args.include == None: # assume you want layer 10
		options = False
	else:
		options = args.options


	process(f, outdir, options)

	raw_input("Operation completed\nPress enter to return to the terminal.")

