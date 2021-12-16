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
t1h = antspyt1w.hierarchical( dti, output_prefix=outfn, imgbxt=srbxt, cit168=True )

# write extant dataframes
for myvar in t1h['dataframes'].keys():
    t1h['dataframes'][myvar].to_csv(newprefix + myvar + ".csv")

(t1h['rbp']).to_csv( newprefix + "rbp.csv" )
myvarlist = [
    'brain_n4_dnz',
    'brain_extraction',
    'wm_tractsL',
    'wm_tractsR',
    'bf',
    'mtl',
    'cit168lab',
    'left_right' ]
for myvar in myvarlist:
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

print( outfn + " complete" )
