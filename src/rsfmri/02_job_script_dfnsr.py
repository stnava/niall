import os
import glob
import sys
import re
from os.path import exists
import ants
import antspymm
import antspynet
import tensorflow as tf
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def pastetoid( x, n = 10 ):
    xsplit = x.split("/")
    if n > len(xsplit):
        n = len(xsplit)
    newoutdir=''
    newprefix=''
    for k in range(n):
        newoutdir = newoutdir + '/' + xsplit[k]
    return newoutdir

def directory2prefix( x, lo, hi, sep='-' ):
    xsplit = x.split("/")
    newprefix=xsplit[lo]
    if hi > len(xsplit):
        hi = len(xsplit)
    for k in range(lo+1,hi):
        newprefix = newprefix + sep + xsplit[k]
    return x + "/" + newprefix

rootdir = "/mnt/cluster/data/PPMI2/PPMI/"
if not exists( rootdir ):
    rootdir = "/Users/stnava/data/PPMI2/"
    print("rootdir " + rootdir )
modality = "restingStatefMRI"
targetfns = glob.glob( rootdir + "*/*/*/*/dcm2niix/V0/*" +
    modality + "*dcm2niix-V0-SR.nii.gz" )
if len(targetfns) == 0:
    targetfns = glob.glob( rootdir + "*/*/*/*/*/dcm2niix/V0/*" +
    modality + "*dcm2niix-V0-SR.nii.gz" )
import sys
if 'fileindex' not in globals():
    fileindex = 0

if len( sys.argv ) > 1:
    fileindex = int(sys.argv[1])
targetfn = targetfns[ fileindex ]
targetsplit = targetfn.split("/")

istest=False
if istest:
    targetfn = "/Users/stnava/data/PPMI2/temp/PPMI-53925-20210609-restingStatefMRI-I1490468-dcm2niix-V0.nii.gz"


newoutdir = re.sub( modality, 'restingNetworksSR', pastetoid( targetfn ) )
os.makedirs( newoutdir, exist_ok=True )
newprefix = directory2prefix( newoutdir, 6, 11 )
img1 = ants.image_read( targetfn )

ptidIndex=6
dateIndex=ptidIndex+1
print("begin: " + newprefix )
t1fns = glob.glob( rootdir + "*/*/T1wHierarchical/*/*"+targetsplit[ptidIndex]+"*" + targetsplit[dateIndex] + "*T1*brain_n4_dnz-SR.nii.gz" )
t1sfns = glob.glob( rootdir + "*/*/T1wHierarchical/*/*"+targetsplit[ptidIndex]+"*" + targetsplit[dateIndex] + "*T1*tissuesegmentationSR.nii.gz" )

if len( t1fns ) == 0:
    print("Missing T1 for "+targetsplit[ptidIndex]+ " " + targetsplit[dateIndex] )
else:
    t1fn = t1fns[ len(t1fns) - 1 ] # take the last one

if len( t1sfns ) == 0:
    print("Missing T1 seg for "+targetsplit[ptidIndex]+ " " + targetsplit[dateIndex] )
else:
    t1sfn = t1sfns[ len(t1sfns) - 1 ] # take the last one

t1 = ants.image_read( t1fn )
t1seg = ants.image_read( t1sfn )

myrsf = antspymm.resting_state_fmri_networks( img1, t1, t1seg )

outkeys = ['meanBold', 'CinguloopercularTaskControl', 'DefaultMode', 'MemoryRetrieval', 'VentralAttention', 'Visual', 'FrontoparietalTaskControl', 'Salience', 'Subcortical', 'DorsalAttention']

for k in outkeys:
    ants.image_write( myrsf[k], newprefix + "-" + k + ".nii.gz" )

import pandas as pd
pd.DataFrame( myrsf['FD' ] ).to_csv( newprefix + "-" + "FD.csv" )
myrsf['corr'].to_csv( newprefix + "-" + "crossnetworkcorrelations.csv" )
