import os
import glob
modality = "T1wHierarchical"
dtifns = glob.glob( "/mnt/cluster/data/SRPBS_multidisorder_MRI/traveling_subjects/SRPBTravel/sub-*/T1wHierarchical/*brain_n4_dnz-SR.nii.gz" )
print( len( dtifns ) )
import sys

if len( sys.argv ) > 1:
    fileindex = int(sys.argv[1])
else:
    fileindex=0

dtifn = dtifns[ fileindex ]
import re
outfn = re.sub('brain_n4_dnz-SR.nii.gz', 'SRHIER', dtifn )
bxtfn = re.sub('brain_n4_dnz-SR.nii.gz', 'brain_extraction.nii.gz', dtifn )
from os.path import exists

import ants
import antspyt1w
import antspymm
import tensorflow as tf
import superiq
newprefix = outfn
print( "Begin NEW HIER " + outfn )

dti = ants.image_read( dtifn )
bxt = ants.image_read( bxtfn )
srbxt = ants.resample_image_to_target( bxt, dti, interp_type='genericLabel' )


import ants
import antspymm
import tensorflow as tf
import antspyt1w
# t1h = antspyt1w.hierarchical( dti, output_prefix=outfn, imgbxt=srbxt, cit168=True )
ch13_weights="/home/ubuntu/.antspyt1w/ch13_weights.h5"
nbm_weights="/home/ubuntu/.antspyt1w/nbm3_weights.h5"

refintensityimage = ants.image_read( "/home/ubuntu/SRPBTravel-sub-094-T1wHierarchical-SRHIERbrain_n4_dnz.nii.gz" )
refintensityimage = ants.iMath( refintensityimage, "Normalize" )
refintensityimage = ants.iMath( refintensityimage, "TruncateIntensity",0.0001,0.999).iMath( "Normalize" )

dti = ants.iMath( dti, "Normalize" )
# dti = ants.histogram_match_image( dti, refintensityimage ).iMath("Normalize")
#  number_of_histogram_bins = 48, number_of_match_points=12 ).iMath("Normalize")

dnbm = antspyt1w.deep_nbm( dti, ch13_weights, nbm_weights, registration=True, csfquantile=0.15, verbose=False)


# write extant dataframe
dnbm['description'].to_csv(newprefix + "newdnbm.csv")
ants.image_write( dnbm['segmentation'], newprefix + "newdnbm.nii.gz" )


print( "new nbm complete" )
