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
import sys
import numpy
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

out = TFile.Open('energies.root','recreate')
out.cd()

rechit_energy = TH1F('rechit_energy','; RecHit Energy (GeV)',500,0,10)

for iev,event in enumerate(events):
    # print iev, event  iev = index of event, event = specific event (xNN -> ie. x40 means 40 events)

    print hgcalHitsLabel[i][0],hgcalHitsLabel[i][1],hgcalHitsLabel[i][2],hgcalHits[i]
    print event.getByLabel(hgcalHitsLabel[i][0],hgcalHitsLabel[i][1],hgcalHitsLabel[i][2],hgcalHits[i])
    exit()
    for i in xrange(2):
        event.getByLabel(hgcalHitsLabel[i][0],hgcalHitsLabel[i][1],hgcalHitsLabel[i][2],hgcalHits[i])

    hgcalRh  = [hgcalHits[0].product(),hgcalHits[1].product()]

    for hits in hgcalRh:
        for hit in hits:
            #### example of getting cell information
            #hid = ROOT.HGCalDetId(hit.id())
            #print hid.subdetId(), hid.layer(), hid.wafer(), hid.cell()
            rechit_energy.Fill(hit.energy())


out.Write()
out.Close()
    
            
