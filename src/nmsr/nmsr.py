import os
import glob
modality = "Neuromelanin"
dtifns = glob.glob( "/mnt/cluster/data/PPMI2/PPMI/*/*/Neuromelanin/*/*/*/*gz" )
import sys
# 735 of these ! print( len( dtifns ) )
if len( sys.argv ) > 1:
    fileindex = int(sys.argv[1])
else:
    fileindex=0

dtifn = dtifns[ fileindex ]
import re
outfn = re.sub('dcm2niix-V0.nii.gz', 'SSRS.nii.gz', dtifn )
from os.path import exists
myoutfnexists = exists( outfn )

if myoutfnexists:
    print( outfn + "exists already" )
    sys.exit()
else:
    print( "BeginNMSR " + outfn )
import ants
import antspymm
import tensorflow as tf
import superiq

mdlfn = "/home/ubuntu/models/SEGSR_32_ANINN222_3.h5"
# mdl = tf.keras.models.load_model( mdlfn )
dti = ants.image_read( dtifn ).denoise_image( noise_model='Gaussian')
hemi = ants.get_mask(img).iMath("ME",2).iMath("GetLargestComponent")
mysr = superiq.super_resolution_segmentation_per_label( dti, hemi, [2,2,2], mdlfn, [1], dilation_amount=0, probability_images=None, probability_labels=None, max_lab_plus_one=True, verbose=True )

ants.image_write( mysr['super_resolution'], outfn )

