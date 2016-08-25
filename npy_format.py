# Kai Chang - Caltech CMS-CERN 2016
#
# Program takes the created .npy files and formats them into autoencoder shape.
# Used if you successfully created ntuple .npy but didn't succesfully run
# the autoencoding format section.
#
#
# Blocks (BLOCKS) of code taken from RootProcessor.py
# https://github.com/bencbartlett/tVertexing/blob/master/RootProcessor.py
# =============================================================================


import argparse
import ROOT as rt
import numpy as np
import os, sys, platform
import glob
import sys
import ntuple_processor as ntp

sys.path.insert(0, './utils')
import wafer

from FastProgressBar import progressbar
from copy import deepcopy


# Argument parser
parser = argparse.ArgumentParser(description = 
'''Converts HGCROIAnalysis-format .root files into RecHit numpy arrays.
	If converting all files in a directory using -f, make sure all files are HGCROI-formatted.''',\
	epilog = '''>> Example: python RootProcessor.py -f ~/public/testfile.root -o ~/public/data
	NOTE: This program must be run while cmsenv is active.

''')

parser.add_argument("-o", "--outdir", \
	help="Directory to save output to. Defaults to Data/.", action="store")
parser.add_argument("-t", "--thickdir", \
	help="location of thickness separation array.", action="store")
args = parser.parse_args()



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


	# this will store each simulation (RECO) as a .npy where the np to tensor program will loop through each .npy
	# and save them into an array for that .npy, and then merge all these arrays so we have independent data for
	# the same layer/wafer but different simulation.

	# move back to appropriate folder
	os.chdir(outdir)

	if not os.path.exists("ae_fmt"):
		os.makedirs("ae_fmt")
	outdir = os.getcwd() + '/ae_fmt/'		

	files = [f for f in os.listdir('./pdg/') if f.endswith('.npy')]
	files = [f for f in files if 'pdg12_pt35' in f]
	os.chdir('./pdg/')

	try:
		ntp.create_feeding_arrays(files, t_list, outdir)
	except TypeError:
		raise

	raw_input("Operation completed\nPress enter to return to the terminal.")

