# Kai Chang - Caltech CMS-CERN 2016
#
# Program runs root_processor on the long-run scale. Will eventually incorporate
# other parts of the package, including dAe.
#
# Needs to have root_processor events line blank,
# Has to run on CMS environment (cmsenv)
# =============================================================================

import sys, os
import fileinput

# open neutrino list file
nfile = open('neutrinos.list')
lines = nfile.readlines()
nfile.close()

for line in lines:
	f = open('root_processor.py', 'r')
	oldfile = f.read()
	f.close()

	newfile = oldfile.replace("loc/to/reco/root", line)

	if line[-9] != '_': #single digit 
		newfile = newfile.replace("rechit_unformatted.npy", "rechit_unformatted_" + str(line[-7]) + ".npy")
		newfile = newfile.replace("rechit_organized.npy", "rechit_organized_" + str(line[-7]) + ".npy")
		newfile = newfile.replace("rechit_formatted.npy", "rechit_formatted_" + str(line[-7]) + ".npy")
	else:
		newfile = newfile.replace("rechit_unformatted.npy", "rechit_unformatted_" + str(line[-8:-6]) + ".npy")
		newfile = newfile.replace("rechit_organized.npy", "rechit_organized_" + str(line[-8:-6]) + ".npy")
		newfile = newfile.replace("rechit_formatted.npy", "rechit_formatted_" + str(line[-8:-6]) + ".npy")

	f = open('root_processor.py', 'w')
	f.write(newfile)
	f.close()

	# runs the root processor
	os.system('python root_processor.py')

	# return to original format
	f = open('root_processor.py', 'w')
	f.write(oldfile)
	f.close()
