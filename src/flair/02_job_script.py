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
    modality + "*dcm2niix-V0.nii.gz" )
if len(targetfns) == 0:
    targetfns = glob.glob( rootdir + "*/*/*/*/*/dcm2niix/V0/*" +
    modality + "*dcm2niix-V0.nii.gz" )
import sys
fileindex = 0
if len( sys.argv ) > 1:
    fileindex = int(sys.argv[1])
targetfn = targetfns[ fileindex ]
import re
# FIXME - more specific stuff below for this modality
mysubbed = re.sub('T1w', 'T1wHierarchical', targetfn )
mysubbedsplit = mysubbed.split("/")
# define the directories and create them
newoutdir = ''
newprefix = ''
keyindex = 10 # change for each case
for k in range(keyindex):
    newoutdir = newoutdir + '/' + mysubbedsplit[k]
    if k > 5:
        newprefix = newprefix + mysubbedsplit[k] + '-'
newprefix = newoutdir + '/' + newprefix
# create the directory
myx = os.path.isdir( newoutdir )
print( "make " +  newoutdir + " " + str( myx ) )

if not myx:
    os.mkdir( newoutdir )

print( "made " +  newoutdir + " successfully " )
outfn = newprefix + "hippR" + '.nii.gz'
myoutfnexists = exists( outfn )
if myoutfnexists:
    print( outfn + "exists already" )
    sys.exit()

print( "continue " +  outfn + " run " )
import ants
import antspymm
import tensorflow as tf
import antspyt1w
targetImage = ants.image_read( targetfn )
print("begin: " + newprefix )
# some work on this modality here
print("complete: " + newprefix )

# write stuff out - this is just an example that needs to be changed
# for a specific modality
mtlfn = os.path.expanduser( "~/.antspyt1w/mtl_description.csv" )
if not exists( mtlfn ):
    t1h['medial_temporal_lobe'][ 'mtl_description'].to_csv( mtlfn )
mtldf = antspyt1w.map_segmentation_to_dataframe(
    'mtl_description', t1h['medial_temporal_lobe'][ 'mtl_segmentation' ] )
ants.image_write( t1h['medial_temporal_lobe'][ 'mtl_segmentation' ],
    newprefix + "mtl.nii.gz" )
(mtldf).to_csv( newprefix + "mtl.csv" )
