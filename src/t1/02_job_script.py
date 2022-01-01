import os
nth="96"
os.environ["TF_NUM_INTEROP_THREADS"] = nth
os.environ["TF_NUM_INTRAOP_THREADS"] = nth
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = nth
from os.path import exists
import glob
rootdir = "/mnt/cluster/data/"
if not exists( rootdir ):
    rootdir = "/Users/stnava/data/PPMI2/"
    print("rootdir " + rootdir )

t1fns = glob.glob( rootdir + "PPMI2/PPMI/*/*/T1w/*/dcm2niix/V0/*-dcm2niix-V0.nii.gz" )
t1fns = t1fns + glob.glob( rootdir + "PPMI1/*/*/*/*/*nii.gz" )
import sys
fileindex = 0
dosr = True
if len( sys.argv ) > 1:
    fileindex = int(sys.argv[1])
if len( sys.argv ) > 2:
    dosr = eval(sys.argv[2])
t1fn = t1fns[ fileindex ]
import re
mysubbed = re.sub('T1w', 'T1wHierarchical', t1fn )
mysubbed = re.sub('MRI_T1', 'T1wHierarchical', mysubbed )
mysubbed = re.sub('/mnt/cluster/data/PPMI2/PPMI/', '/mnt/cluster/data/PPMIPostJoin/', mysubbed )
mysubbed = re.sub('/mnt/cluster/data/PPMI1/', '/mnt/cluster/data/PPMIPostJoin/', mysubbed )
mysubbed = re.sub('dcm2niix/V0', '', mysubbed )
mysubbed = re.sub('-dcm2niix-V0', '', mysubbed )
newprefix = re.sub('.nii.gz','',mysubbed)
mysubbedsplit = mysubbed.split("/")
# define the directories and create them
newoutdir = ''
for k in range(len(mysubbedsplit)-1):
    newoutdir = newoutdir + '/' + mysubbedsplit[k]

# create the directory
myx = os.path.isdir( newoutdir )
print( "make " +  newoutdir + " " + str( myx ) )

if not myx:
    os.makedirs( newoutdir, exist_ok=True  )

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
import superiq
t1 = ants.image_read( t1fn )
if dosr:
    print("begin: " + newprefix +  " dosr " + str( dosr ) )
    newprefix = newprefix + "-SR"
    print("first a bxt ")
    t1 = ants.iMath( t1, "TruncateIntensity", 1e-4, 0.999 ).iMath( "Normalize" )
    t1bxt = antspyt1w.brain_extraction( t1 )
    t1 = ants.denoise_image( t1, t1bxt, noise_model='Gaussian')
    t1 = ants.n4_bias_field_correction( t1, mask=t1bxt, rescale_intensities=True, ).iMath("Normalize")
    print("second is SR")
    mdlfn = "/home/ubuntu/models/SEGSR_32_ANINN222_3.h5"
    mdl = tf.keras.models.load_model( mdlfn )
    mysr = superiq.super_resolution_segmentation_per_label(
        t1, t1bxt, [2,2,2], mdl, [1], dilation_amount=6, probability_images=None,
        probability_labels=None, max_lab_plus_one=True, verbose=True )
    t1 = mysr['super_resolution']
    t1bxt = ants.resample_image_to_target( t1bxt, t1, interp_type='nearestNeighbor' )
    ants.image_write( t1, newprefix + ".nii.gz" )
    print("begin hier: " + newprefix )
    derka
    t1h = antspyt1w.hierarchical( t1, output_prefix=newprefix, imgbxt=t1bxt, cit168=True )
else:
    print("begin hier: " + newprefix )
    t1h = antspyt1w.hierarchical( t1, output_prefix=newprefix, cit168=True )

print("complete: " + newprefix )

# write extant dataframes
for myvar in t1h['dataframes'].keys():
    if t1h['dataframes'][myvar] is not None:
        t1h['dataframes'][myvar].dropna(0).to_csv(newprefix + myvar + ".csv")

(t1h['rbp']).to_csv( newprefix + "rbp.csv" )

myvarlist = [
    'brain_n4_dnz',
    'brain_extraction',
    'wm_tractsL',
    'wm_tractsR',
    'bf',
    'mtl',
    'snseg',
    'deep_cit168lab',
    'cit168lab',
    'left_right' ]
for myvar in myvarlist:
    if t1h[myvar] is not None:
        ants.image_write( t1h[myvar], newprefix + myvar + '.nii.gz' )

myvarlist = [
    'tissue_segmentation',
    'dkt_parcellation',
    'dkt_lobes',
    'dkt_cortex',
    'hemisphere_labels' ]
for myvar in myvarlist:
    ants.image_write( t1h['dkt_parc'][myvar], newprefix + myvar + '.nii.gz' )

ants.image_write( t1h['hippLR']['HLBin'], newprefix + "hippL" + '.nii.gz' )
ants.image_write( t1h['hippLR']['HRBin'], newprefix + "hippR" + '.nii.gz' )

# SR
