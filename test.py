import argparse
import ROOT as rt
import numpy as np
import os, sys, platform
import glob


# get list of .npy files created prior
files = [f for f in os.listdir(outdir) if f.endswith('.npy')]

thickness = []
# go through the first file to get characteristics of wafers in the layer
for event in file[0]: