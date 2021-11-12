import os
nth="24"
os.environ["TF_NUM_INTEROP_THREADS"] = nth
os.environ["TF_NUM_INTRAOP_THREADS"] = nth
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = nth
import glob
modality = "restingStatefMRI"
dtifns = glob.glob( "/mnt/cluster/data/PPMI2/PPMI/*/*/*/*/dcm2niix/V0/*" +
    modality + "*dcm2niix-V0.nii.gz" )
import sys
fileindex = int(sys.argv[1])
dtifn = dtifns[ fileindex ]
import re
outfn = re.sub('V0.nii.gz', 'V0-SR.nii.gz', dtifns[ fileindex ] )

from os.path import exists
myoutfnexists = exists( outfn )
if myoutfnexists:
    print( outfn + "exists already" )
    sys.exit()

import ants
import antspymm
import tensorflow as tf
mdlfn = antspymm.get_data( "brainSR", target_extension=".h5")
mdl = tf.keras.models.load_model( mdlfn )
dti = ants.image_read( dtifn )
print("begin SR for" + outfn )
srimg = antspymm.super_res_mcimage( dti, mdl, verbose=True )
ants.image_write( srimg, outfn )
