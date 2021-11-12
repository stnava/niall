import os
nth="24"
os.environ["TF_NUM_INTEROP_THREADS"] = nth
os.environ["TF_NUM_INTRAOP_THREADS"] = nth
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = nth
from os.path import exists
import glob

rootdir = "/mnt/cluster/data/PPMI2/PPMI/"
if not exists( rootdir ):
    rootdir = "/Users/stnava/data/PPMI2/"
    print("rootdir " + rootdir )
modality = "Flair"
targetfns = glob.glob( rootdir + "*/*/*/*/dcm2niix/V0/*" + 
    modality + "*dcm2niix-V0.nii.gz")
if len( targetfns ) == 0:
    targetfns = glob.glob( rootdir + "*/*/*/*/*/dcm2niix/V0/*" + 
    modality + "*dcm2niix-V0.nii.gz" )
import sys
fileindex = 0
if len( sys.argv ) > 1:
    fileindex = int(sys.argv[1])
targetfn = targetfns[ fileindex ]
print( targetfn )
import re

newmod='FlairWMH'
mysubbed = re.sub( modality, newmod, targetfn )
mysubbedsplit = mysubbed.split("/")

# Define the directories and create them
newoutdir = ''
newprefix = ''
keyindex = 10 # change for each case
for k in range( keyindex ):
    newoutdir = newoutdir + '/' + mysubbedsplit[k]
    if k > 5:

        newprefix = newprefix + mysubbedsplit[k] + '-'
newprefix = newoutdir + '/' + newprefix

# Create the directory
myx = os.path.isdir( newoutdir )
print( "make " +  newoutdir + " " + str( myx ) )
if not myx:
    os.makedirs( newoutdir, exist_ok=True  )

print( "made " +  newoutdir + " successfully " )
outfn = newprefix + newmod + '.nii.gz'
myoutfnexists = exists( outfn )
if myoutfnexists:
    print( outfn + "exists already" )
    sys.exit()
print( "continue to computation of: " +  outfn + " run " )

import ants
import antspymm
import tensorflow as tf
import antspyt1w

import pandas as pd

targetImage = ants.image_read( targetfn )
print("begin: " + newprefix )

ptidIndex = 6
dateIndex = ptidIndex + 1

# Get T1 tissue segmentation image from antspyt1w output
t1segfn = glob.glob(rootdir + "*/*/*/*/" +  mysubbedsplit[ptidIndex] + "*" +  mysubbedsplit[dateIndex] + "*tissue_segmentation.nii.gz")

# Get T1 image
t1fns = glob.glob(rootdir + "*/*/" + "T1w" + "/*/dcm2niix/V0/" + "*" +  mysubbedsplit[ptidIndex] + "*" + mysubbedsplit[dateIndex] + "*T1w*dcm2niix-V0.nii.gz")

if len( t1fns ) == 0:
    print("Missing T1 for" + mysubbedsplit[ptidIndex] + " " + mysubbedsplit[dateIndex] )
else:
    t1fn = t1fns[ len(t1fns) - 1] # take the last one 

t1seg = ants.image_read(t1segfn[0])
t1 = ants.image_read(t1fn)

WMH_output =  antspymm.wmh( targetImage, t1, t1seg )

print("complete: " + newprefix )

# Write out WMH image and WMH mass 
ants.image_write( WMH_output['WMH_probability_map'], outfn )
output2df = {'Value' : [WMH_output['wmh_mass']], 'Description' : 'Mass_of_WMH'}
flairdf = pd.DataFrame(output2df)
flairdf.to_csv( newprefix + newmod + ".csv")
