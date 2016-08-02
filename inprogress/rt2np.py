# Kai Chang - Caltech CMS-CERN 2016
#
# Program takes any root file and produces appropriate numpy arrays
# for the trees inside.
#
# =============================================================================

import os

import ROOT as rt
import numpy as np
import root_numpy as rnp

rt.gSystem.Load('hgcal_dict.so')

# Will use to update later (searches for root files and makes)
# cdir = os.getcwd()
# files = os.listdir(cdir)
# rfiles =[]
# for f in files:
	# if f.endswith(".root"):
	# 	rfiles.append(f)
# rfiles.sort()

rt.gROOT.Class(ARecHit)
exit()

#dir = '/afs/cern.ch/user/k/kachang/work/public/CMSSW_8_1_0_pre8/src/RecoNtuples/HGCalAnalysis/test/DL-HGCal/twogamma_pt5_eta2_nosmear_calib_ntuple.root'

#dir = '2015C_EcalNtp_10.root'
dir = 'twogamma_pt5_eta2_nosmear_calib_ntuple.root'
f = rt.TFile(dir)
tree = f.Get('ana/hgc')
#tree = f.Get('Tree_Optim')
print tree.Print()
nentries = tree.GetEntries()
print nentries

for i in range(0, 5):
	#tree.GetEntry(i) #tells how many events are in that entry (interesting info)
	print tree.GetEntry(i)
	tree.Show(i)

#arr = rnp.tree2array(tree)


# Convert a TTree in a ROOT file into a NumPy structured array
# f = rt.TFile(dir)
# tree = f.Get('ana/hgc')

# print tree.Print()
# nentries=tree.GetEntries()
# for i in range(0,nentries):
# 	print tree.GetEntry(i)

#arr = rnp.tree2array(tree)

# Or first get the TTree from the ROOT file
# rfile = ROOT.TFile(filename)
# tree = rfile.Get('tree')

# and convert the TTree into an array
# array = tree2array(tree)
# array = tree2array(tree,
#     branches=['x', 'y', 'sqrt(y)', 'TMath::Landau(x)', 'cos(x)*sin(y)'],
#     selection='z > 0',
#     start=0, stop=10, step=2)


