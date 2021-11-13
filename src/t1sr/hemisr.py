import os
import glob
modality = "T1wHierarchical"
dtifns = glob.glob( "/mnt/cluster/data/PPMI2/PPMI/*/*/" + modality + "/*/*brain_n4_dnz.nii.gz" )
import sys

if len( sys.argv ) > 1:
    fileindex = int(sys.argv[1])
else:
    fileindex=0

dtifn = dtifns[ fileindex ]
import re
outfn = re.sub('dnz.nii.gz', 'dnz-SR.nii.gz', dtifn )
hemifn = re.sub('brain_n4_dnz.nii.gz', 'left_right.nii.gz', dtifn )
from os.path import exists
myoutfnexists = exists( outfn )
if not exists( hemifn ):
    print( "hemifn " + hemfin + " does not exist")
    sys.exit(1)

if myoutfnexists:
    print( outfn + "exists already" )
    sys.exit()
else:
    print( "BeginT1SR " + outfn )
import ants
import antspymm
import tensorflow as tf
import superiq
mdlfn = "/home/ubuntu/models/SEGSR_32_ANINN222_3.h5"
# mdl = tf.keras.models.load_model( mdlfn )
dti = ants.image_read( dtifn )
hemi = ants.image_read( hemifn )
mysr = superiq.super_resolution_segmentation_per_label( dti, hemi, [2,2,2], mdlfn, [1,2], dilation_amount=6, probability_images=None, probability_labels=None, max_lab_plus_one=True, verbose=True )

ants.image_write( mysr['super_resolution'], outfn )

