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

# max wafer: 462, max cell: 236 (for all layers)
# max wafer: 461, max cell: 235 (for layer 10 recorded data)


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
# max_wafer = 0
# max_cell = 0

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

    for hits in hgcalRh:         # event in list of events
        for hit in hits:         # hits in each event
            # getting cell information
            hid = ROOT.HGCalDetId(hit.id())     #print hid.subdetId(), hid.layer(), hid.wafer(), hid.cell()
            
            # numpy
            r_layer.append(hid.layer())
            r_wafer.append(hid.wafer())
            r_cell.append(hid.cell())
            r_energy.append(hit.energy())

            # if hid.layer() == 10 and hid.wafer() > max_wafer:
            #     max_wafer = hid.wafer()
            # if hid.layer() == 10 and hid.cell() > max_cell:
            #     max_cell = hid.cell()

    rechits_array = np.core.records.fromarrays([r_layer, r_wafer, r_cell, r_energy], \
        names='layer, wafer, cell, energy')
    
    # only needed if adding other attributes outside of rechits
    # eventArray.append(rechits_array)
    # names += 'rechits'
    # outArray_u.append(eventArray)

    outArray_u.append(rechits_array)

# print "max wafer: %i, max cell: %i" %(max_wafer, max_cell)

print 'writing unfiltered file rechit_unformatted'
np.save('rechit_unformatted.npy', outArray_u)



#### STORE FILTED ARRAY ####
# For now, we only extract layer 10
from copy import deepcopy

outArray_f = []
outArray_o = []

for event in outArray_u:
    # layer_array = [[] for i in range(0, 28)]  # 28 layers is total number of layers in HGCal, 
    wafer_array = [[] for i in range(0,461)]    # 461 wafers in layer 10 of HGCal
    cell_array =[0 for i in range(0,239)]       # 239 cells in layer 10 full wafer

    for i in xrange(0,461):
        wafer_array[i] = deepcopy(cell_array)
    
    for hit in event:
        if hit['layer'] == 10:
            wafer_array[hit['wafer']-1][hit['cell']-1] = hit['energy'] # wafer/cells index at 1, numpy index at 0

    outArray_o.append(wafer_array)

print 'writing structurally organized file rechit_organized'
np.save('rechit_organized.npy', outArray_o)

# removes event distinction
for event_i in xrange(0,len(outArray_o)):
    for wafer_i in xrange(0,len(outArray_o[0])):

        outArray_f.append(outArray_o[event_i][wafer_i])
        # # if no noise -- remove from set (NOT WORKING OR VALID)
        # if all(cell == 0 for cell in outArray_o[event_i][wafer_i]):
        #     continue
        # else:
            # outArray_f.append(outArray_o[event_i][wafer_i])


print 'writing filtered file rechit_formatted'
np.save('rechit_formatted.npy', outArray_f)

print 'Process complete.'
    
            
