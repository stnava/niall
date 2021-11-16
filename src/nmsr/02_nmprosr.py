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
modality = "Neuromelanin"
targetfns = glob.glob( rootdir + "*/*/" + modality )
import sys
if 'fileindex' not in globals():
    fileindex = 0

if len( sys.argv ) > 1:
    fileindex = int(sys.argv[1])
targetfn = targetfns[ fileindex ]

targetsplit = targetfn.split("/")

newoutdir = re.sub( modality, 'NMAverage', pastetoid( targetfn ) )
os.makedirs( newoutdir, exist_ok=True )
newprefix = directory2prefix( newoutdir, 6, 11 )
nmext = "brainSR"
nmext = "SSRS"
fnlist = glob.glob( targetfn + "/*/*/*/*" + nmext + ".nii.gz" )

ilist = []
for x in fnlist:
    ilist.append( ants.image_read( x ) )

ptidIndex=6
dateIndex=ptidIndex+1
print("begin: " + newprefix )
t1fns = glob.glob( rootdir + "*/*/T1wHierarchical/*/*"+targetsplit[ptidIndex]+"*" + targetsplit[dateIndex] + "*T1*brain_n4_dnz-SR.nii.gz" )
t1sfns = glob.glob( rootdir + "*/*/T1wHierarchical/*/*"+targetsplit[ptidIndex]+"*" + targetsplit[dateIndex] + "*T1*SYNCC2CIT168Labels.nii.gz" )
t1slfns = glob.glob( rootdir + "*/*/T1wHierarchical/*/*"+targetsplit[ptidIndex]+"*" + targetsplit[dateIndex] + "*T1*SYNCC2CIT168MTSlabLabel.nii.gz" )

if len( t1fns ) == 0:
    print("Missing T1 for "+targetsplit[ptidIndex]+ " " + targetsplit[dateIndex] )
else:
    t1fn = t1fns[ len(t1fns) - 1 ] # take the last one

if len( t1sfns ) == 0:
    print("Missing T1 seg for "+targetsplit[ptidIndex]+ " " + targetsplit[dateIndex] )
else:
    t1sfn = t1sfns[ len(t1sfns) - 1 ] # take the last one

if len( t1slfns ) == 0:
    print("Missing T1 slab for "+targetsplit[ptidIndex]+ " " + targetsplit[dateIndex] )
else:
    t1slfn = t1slfns[ len(t1slfns) - 1 ] # take the last one


t1 = ants.image_read( t1fn )
t1seg = ants.image_read( t1sfn )
t1slseg = ants.image_read( t1slfn )
# neuromelanin(list_nm_images, t1, t1slab, t1lab)
mynm = antspymm.neuromelanin( ilist, t1, t1slseg, t1seg )

ants.image_write(mynm['NM_avg'], newprefix + ".nii.gz" )
ants.image_write(mynm['NM_labels'], newprefix + "CIT168.nii.gz" )

import antspyt1w
summarydf = antspyt1w.map_segmentation_to_dataframe( 'CIT168_Reinf_Learn_v1_label_descriptions_pad', mynm['NM_labels'] )
import pandas as pd
summarydf.to_csv( newprefix + "CIT168.csv" )

