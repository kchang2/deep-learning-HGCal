# Kai Chang - Caltech CMS-CERN 2016
#
# Program takes the RECO ROOT files in the eos server, the files generated
# from the GEANT4 simulation, and then extracts only the rechit information
# and saves them in a root file.
#
# Replaces the .py file created from the RecoNtuple directory
# Has to run on CMS environment (cmsenv)
#
# Backbone written by Lindsey Gray
# =============================================================================



#! /usr/bin/env python


# import ROOT in batch mode
import sys, os
import numpy as np
oldargv = sys.argv[:]
sys.argv = [ '-b-' ]
import ROOT
ROOT.gROOT.SetBatch(True)
sys.argv = oldargv

# load FWLite C++ libraries
ROOT.gSystem.Load("libFWCoreFWLite.so");
ROOT.gSystem.Load("libDataFormatsFWLite.so");
ROOT.FWLiteEnabler.enable()

#cms python data types
import FWCore.ParameterSet.Config as cms

# load FWlite python libraries
from DataFormats.FWLite import Handle, Events

from ROOT import TH1F, TFile

# open file (you can use 'edmFileUtil -d /store/whatever.root' to get the physical file name)
events = Events("root://eoscms.cern.ch//eos/cms/store/cmst3/group/hgcal/CMG_studies/Production/partGun_clange_PDGid12_Pt35_20160729/RECO/partGun_PDGid12_x40_Pt35.0To35.0_RECO_1.root")

hgcalHits, hgcalHitsLabel = [Handle('edm::SortedCollection<HGCRecHit,edm::StrictWeakOrdering<HGCRecHit> >'),
                             Handle('edm::SortedCollection<HGCRecHit,edm::StrictWeakOrdering<HGCRecHit> >')], ["HGCalRecHit:HGCEERecHits:".split(":"),
                                                                                                               "HGCalRecHit:HGCHEFRecHits:".split(":")]

# unfilted array format will be as such:
# [event][(hit_info1), (hit_info2), ...]
# [event][(layer1, wafer1, cell1, energy1), (layer2, wafer2, cell2, energy2)]
outArray_u = []


# clear the terminal
os.system('clear')



#### STORE UNFILTERED ARRAY ####

for iev,event in enumerate(events): # iev = index of event, event = specific event (xNN -> ie. x40 means 40 events)

    # https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideEDMGetDataFromEvent#get_ByLabel
    # get the data, The InputTag can hold the module label and the product instance name. 
    # basically checks to see if data and label are there -- the labels should be unique.
    for i in xrange(2):
        event.getByLabel(hgcalHitsLabel[i][0],hgcalHitsLabel[i][1],hgcalHitsLabel[i][2],hgcalHits[i])

    hgcalRh  = [hgcalHits[0].product(),hgcalHits[1].product()]


    # rechit info
    # eventArray = []
    # names = ""
    r_layer         = [] # can sort later using recursive approach
    r_wafer         = []
    r_cell          = []
    r_energy        = []

    max_cell = 100
    for hits in hgcalRh:         # event in list of events
        for hit in hits:         # hits in each event
            # getting cell information
            hid = ROOT.HGCalDetId(hit.id())     #print hid.subdetId(), hid.layer(), hid.wafer(), hid.cell()
            
            # numpy
            r_layer.append(hid.layer())
            r_wafer.append(hid.wafer())
            r_cell.append(hid.cell())
            if hid.cell() < max_cell and hid.layer() == 10:
                max_cell = hid.cell()
            r_energy.append(hit.energy())
    print max_cell       
    rechits_array = np.core.records.fromarrays([r_layer, r_wafer, r_cell, r_energy], \
        names='layer, wafer, cell, energy')
    
    # only needed if adding other attributes outside of rechits
    # eventArray.append(rechits_array)
    # names += 'rechits'
    # outArray_u.append(eventArray)

    outArray_u.append(rechits_array)

print 'writing unfiltered file rechit_unformatted.npy'
np.save('rechit_unformatted.npy', outArray_u)



#### STORE FILTED ARRAY ####
# For now, we only extract layer 10

outArray_f = []

for event in outArray_u:
    # layer_array = [[] for i in range(0, 28)] # 28 layers is total number of layers in HGCal, 239 wafers is total number of cell in full wafer
    cell_array =[[i,0] for i in range(0,236)] #236 wafers in layer 10 full wafer
    
    for hit in event:
        layer_array[hit['layer']-1].append( [hit['cell'], hit['energy']] ) # layers index at 1, numpy index at 0

    outArray_f.append(layer_array)
    for 

print 'writing filtered file rechit_formatted.npy'
np.save('rechit_formatted.npy', outArray_f)

print 'Process complete.'
    
            
