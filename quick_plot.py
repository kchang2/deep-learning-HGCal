# Kai Chang - Caltech CMS-CERN 2016
#
# Program takes the RECO ROOT files in the eos server, the files generated
# from the GEANT4 simulation, and then extracts only the rechit information
# and saves them in a root file.
#
# Replaces the .py file created from the RecoNtuple directory
#
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

# create root file
out = TFile.Open('info.root','recreate')
out.cd()
rechit_energy = TH1F('rechit_energy','; RecHit Energy (GeV)',500,0,10)

# cretes output array
outArray = []

# clear the terminal
os.system('clear')

for iev,event in enumerate(events):
    # print iev, event  iev = index of event, event = specific event (xNN -> ie. x40 means 40 events)

    # https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideEDMGetDataFromEvent#get_ByLabel
    # get the data, The InputTag can hold the module label and the product instance name. 
    # basically checks to see if data and label are there -- the labels should be unique.
    for i in xrange(2):
        event.getByLabel(hgcalHitsLabel[i][0],hgcalHitsLabel[i][1],hgcalHitsLabel[i][2],hgcalHits[i])
        # print hgcalHitsLabel[i][0]
        # print hgcalHitsLabel[i][1]
        # print hgcalHitsLabel[i][2]
        # print hgcalHits[i]

    hgcalRh  = [hgcalHits[0].product(),hgcalHits[1].product()]


    # event in list of events
    for hits in hgcalRh:
        # rechit info
        eventArray = []
        names = ""
        r_layer         = []
        r_wafer         = []
        r_cell          = []
        r_energy        = []
        # r_time          = []
        r_thickness     = []
        r_isHalf        = []


        # hits in each event
        for hit in hits:
            # getting cell information
            hid = ROOT.HGCalDetId(hit.id())     #print hid.subdetId(), hid.layer(), hid.wafer(), hid.cell()
            # w = ROOT.HGCalDDDConstants.waferTypeL(hid.wafer())

            print hid.layer(), hid.wafer(), hid.cell(), hit.energy()
            exit()
            # numpy
            r_layer.append(hid.layer())
            r_wafer.append(hid.wafer())
            r_cell.append(hid.cell())

            r_energy.append(hit.energy())
            # r_time.append(hit.time())
            # r_thickness.append(hid.thickness())
            # r_isHalf.append(hit.isHalf())

            rechits_array = np.core.records.fromarrays([r_layer, r_wafer, r_cell, \
                r_energy], \
                names='layer, wafer, cell, energy')

            eventArray.append(rechits_array)
            names += 'rechits'

            # histogram for ROOT file
            rechit_energy.Fill(hit.energy())

        outArray.append(eventArray)

print 'writing files'
np.save('rechit.npy', outArray)

out.Write()
out.Close()

print 'Process complete.'
    
            
