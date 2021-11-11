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

t1fns = glob.glob( rootdir + "*/*/*/*/*/dcm2niix/V0/*T1*dcm2niix-V0.nii.gz" )
import sys
fileindex = 0 # int(sys.argv[1])
t1fn = t1fns[ fileindex ]
import re
outfn = re.sub('V0.nii.gz', 'V0-SR.nii.gz', t1fn )
mysubbed = re.sub('T1w', 'T1wHierarchical', t1fn )
mysubbedsplit = mysubbed.split("/")
# define the directories and create them
newoutdir = ''
newprefix = ''
keyindex = 9 # change for each case
for k in range(keyindex):
    newoutdir = newoutdir + '/' + mysubbedsplit[k]
    if k > 5:
        newprefix = newprefix + mysubbedsplit[k] + '-'
newprefix = newoutdir + '/' + newprefix + mysubbedsplit[keyindex] + '-'
# create the directory
if not exists( newoutdir ):
    os.mkdir( newoutdir )
myoutfnexists = exists( outfn )
if myoutfnexists:
    print( outfn + "exists already" )
    sys.exit()
import ants
import antspymm
import tensorflow as tf
import antspyt1w
# mdlfn = antspymm.get_data( "brainSR", target_extension=".h5")
# mdl = tf.keras.models.load_model( mdlfn )
t1 = ants.image_read( t1fn )
print("begin: " + newprefix )
t1h = antspyt1w.hierarchical( t1, output_prefix=newprefix )
print("complete: " + newprefix )
# hierarchical(x, output_prefix, labels_to_register=[2, 3, 4, 5], is_test=False, verbose=True)
#    Default processing for a T1-weighted image.  See README.
# map_segmentation_to_dataframe(segmentation_type, segmentation_image)
