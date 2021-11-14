import os
import glob
modality = "Neuromelanin"
dtifns = glob.glob( "/mnt/cluster/data/PPMI2/PPMI/*/*/Neuromelanin/*/*/*/*V0.nii.gz" )
import sys
# 735 of these ! print( len( dtifns ) )
if len( sys.argv ) > 1:
    fileindex = int(sys.argv[1])
else:
    fileindex=0

dtifn = dtifns[ fileindex ]
import re
outfn = re.sub('dcm2niix-V0.nii.gz', 'SSRS.nii.gz', str( dtifn ) )
outfn2 = re.sub('dcm2niix-V0.nii.gz', 'brainSR.nii.gz', str( dtifn ) )

from os.path import exists
myoutfnexists = ( exists( outfn ) and exists(outfn2) )
if myoutfnexists:
    print( outfn + "exists already" )
else:
    print( "BeginNMSR " + outfn )
import ants
import antspymm
import tensorflow as tf
import superiq
import antspynet
mdlfn = "/home/ubuntu/models/SEGSR_32_ANINN222_3.h5"
dti = ants.image_read( dtifn ).denoise_image( noise_model='Gaussian',p="2x2x1",r="3x3x1").iMath("TruncateIntensity",0.01,0.995)
hemi = ants.get_mask(dti).iMath("ME",2).iMath("GetLargestComponent")
mysr = superiq.super_resolution_segmentation_per_label( dti, hemi, [2,2,2], mdlfn, [1], dilation_amount=0, probability_images=None, probability_labels=None, max_lab_plus_one=True, verbose=True )

ants.image_write( mysr['super_resolution'], outfn )

mdlfn = antspymm.get_data( "brainSR", target_extension=".h5")
mdl = tf.keras.models.load_model( mdlfn )
mysr2 = antspynet.apply_super_resolution_model_to_image( dti, mdl )
ants.image_write( mysr2, outfn2 )

