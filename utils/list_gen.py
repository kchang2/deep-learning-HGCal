# Kai Chang - Caltech CMS-CERN 2016
#
# Program funnels all the desired .root files in EOS into a useful list
# The precursor to running the script and alternatively root_processor.py.
# =============================================================================

import sys, os
import argparse

parser = argparse.ArgumentParser(description = 
'''Creates a list of directory locations in EOS based on your pdg request.''',\
	epilog = '''>> Example: python list_gen.py -i 12 -p 35		#neutrinos, pt35.

''')

parser.add_argument("-i", "--pdg", \
	help="Particle Data Group ID number.", action="store_const")
parser.add_argument("-p", "--pt", \
	help="Particle transverse momentum.", action="store_const")
parse.add_argument("-n", "--npart" \
	help="Adding nPart1 to request.", action="store_false")
args = parser.parse_args()


if __name__ == "__main__":

	# checks if particle and pt are valid
	if args.pdg == None:
		raise ValueError('Aint got no particles to reference to.')

	if args.pdg not in [111, 11, 12, 13, 211, 22]:
		raise ValueError('Aint no reasonable particle.')

	if args.npart == True and int(pt) not in [5, 35, 2, 10]:
		raise ValueError('Wrong pT')
	
	if args.npart == False and int(pt) not in [10, 35]:
		raise ValueError('Wrong pT')

	if args.pt == None:
		pt = ['Pt35']

	# classifying and formatting the particle and momentum
	pdg = 'PDGid' + str(args.pdg)
	pt = 'Pt' + str(args.pt)
	pdg_list = {'11':'electron', '12': 'e_neutrino', '13': 'muon' , '22': 'photon', '111': 'n_pion', '211': 'c_pion'}
	pname = pdg_list[str(args.pdg)]
	# matching = [s for s in some_list if "abc" in s]


	# gather directory locations into list
	cmd = 'eos find root://eoscms.cern.ch//eos/cms/store/cmst3/group/hgcal/CMG_studies/Production/ name "*.root" grep ".root" | grep "%s" | grep "RECO" | grep "%s" > %s.list' %(pdg, pt, pname)
	os.system(cmd)

	f = open('%s.list', 'r') %pname
	new_f = f.read()
	f.close()

	f = open('%s.list', 'w') %pname
	for line in new_f:
		if 'hydra' not in line and '.root' in line:
			f.write(line)

	f.close()
