import os
nth="24"
os.environ["TF_NUM_INTEROP_THREADS"] = nth
os.environ["TF_NUM_INTRAOP_THREADS"] = nth
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = nth
import glob
dtifns = glob.glob( "/mnt/cluster/data/PPMI2/PPMI/*/*/*/*/dcm2niix/V0/*DTI*dcm2niix-V0.nii.gz" )
bvefns = glob.glob( "/mnt/cluster/data/PPMI2/PPMI/*/*/*/*/dcm2niix/V0/*DTI*dcm2niix-V0.bvec" )
bvafns = glob.glob( "/mnt/cluster/data/PPMI2/PPMI/*/*/*/*/dcm2niix/V0/*DTI*dcm2niix-V0.bval" )

import sys
fileindex = int(sys.argv[1])
dtifn = dtifns[ fileindex ]
bvefn = bvefns[ fileindex ]
bvafn = bvafns[ fileindex ]
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
# mm = antspymm.dipy_dti_recon( dti, "PPMI-107099-20210914-DTI_LR-I1498906-dcm2niix-V0.bval", "PPMI-107099-20210914-DTI_LR-I1498906-dcm2niix-V0.bvec")
print("begin SR for" + outfn )
srimg = antspymm.super_res_mcimage( dti, mdl, verbose=True )
ants.image_write( srimg, outfn )
