import os
from os.path import exists
import glob
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd

rootdir = "/mnt/cluster/data/"

t1fns = glob.glob( rootdir + "SRPBS_multidisorder_MRI/traveling_subjects/SRPBTravel/sub-*/anat/*_T1w.nii.gz" )
t1fns = t1fns + glob.glob( rootdir + "anatomicalLabels/Mindboggle101_volumes/*volumes/*/t1weighted.nii.gz" )
t1fns = t1fns + glob.glob( rootdir + "PPMI2/PPMI/*/*/T1w/*/dcm2niix/V0/*-dcm2niix-V0.nii.gz" )
t1fns = t1fns + glob.glob( rootdir + "PPMI1/*/*/*/*/*nii.gz" )
import sys
import re
fileindex=137
fileindex=222
myoffset=1000

dosr = True
if len( sys.argv ) > 1:
    fileindex = int(sys.argv[1])
if len( sys.argv ) > 2:
    dosr = eval(sys.argv[2])
if len( sys.argv ) > 3:
    myoffset = int(sys.argv[3])
if (fileindex + myoffset) > len( t1fns):
    sys.exit(0)
t1fn = t1fns[ fileindex + myoffset]
print( "target: " + t1fn + "  " + str(myoffset) + " " + str( fileindex ) )
myrw = str( fileindex )+"_"+str(myoffset)


outpre = "/mnt/cluster/data/T1wJoin/PPMI_RBP/zz" + myrw + "_" + \
    re.sub( ".nii.gz", "_RBP", os.path.basename( t1fn ) )
print( "Begin: " + outpre )
import ants
import antspynet
import antspymm
import tensorflow as tf
import tensorflow.keras.backend as K
K.set_floatx("float32")
import antspyt1w
import numpy as np

# whole head outlierness
t1 = ants.image_read( t1fn )
clusoe = antspyt1w.inspect_raw_t1( t1, outpre )
