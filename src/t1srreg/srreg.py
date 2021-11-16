import os
import glob
modality = "T1wHierarchical"
dtifns = glob.glob( "/mnt/cluster/data/PPMI2/PPMI/*/*/" + modality + "/*/*brain_n4_dnz-SR.nii.gz" )
print( len( dtifns ) )
import sys

if len( sys.argv ) > 1:
    fileindex = int(sys.argv[1])
else:
    fileindex=0

dtifn = dtifns[ fileindex ]
import re
outfn = re.sub('brain_n4_dnz-SR.nii.gz', 'SYNCC2CIT168', dtifn )
from os.path import exists
outfns = [ outfn + "1Warp.nii.gz" , outfn + '0GenericAffine.mat' ]
myoutfnexists = exists( outfns[0] ) and exists( outfns[1] )

import ants
import antspyt1w
import antspymm
import tensorflow as tf
import superiq

if myoutfnexists:
    print( outfn + "exists already" )
else:
    print( "BeginT1CITREG " + outfn )

dti = ants.image_read( dtifn )

# get CIT image
citatlas = ants.image_read( antspyt1w.get_data( "CIT168_T1w_700um_pad_adni", target_extension='.nii.gz') )
detlabels = ants.image_read( antspyt1w.get_data( "det_atlas_25_pad_LR_adni", target_extension=".nii.gz") )
bstlabels = ants.image_read( antspyt1w.get_data( "CIT168_T1w_700um_pad_adni_brainstem", target_extension=".nii.gz") )
# syncc2 t1
mytx = 'SyNCC'
# mytx = 'AffineFast'

if myoutfnexists:
    reg = { 'fwdtransforms': outfns }
else:
    reg = ants.registration( dti, citatlas, mytx, outprefix=outfn )

myinterp='genericLabel'
# map det atlas to t1
detlab = ants.apply_transforms( dti, detlabels, reg['fwdtransforms'], interpolator=myinterp )
ants.image_write( detlab,  outfn + "Labels.nii.gz" )
bstlabel = ants.apply_transforms( dti, bstlabels, reg['fwdtransforms'], interpolator=myinterp )
ants.image_write( bstlabel,  outfn + "BrainStemLabels.nii.gz" )
mtslab = ants.image_read(  antspyt1w.get_data( "CIT168_MT_Slab_adni", target_extension=".nii.gz") )
mtslab = ants.apply_transforms( dti, mtslab,  reg['fwdtransforms'], interpolator=myinterp )
ants.image_write( mtslab,  outfn + "MTSlabLabel.nii.gz" )

bflab = ants.image_read( antspyt1w.get_data( "CIT168_basal_forebrain_adni", target_extension=".nii.gz") )
bflab = ants.apply_transforms( dti, bflab, reg['fwdtransforms'], interpolator=myinterp )
ants.image_write( bflab,  outfn + "basalforebrainbasic.nii.gz" )
antspyt1w.map_segmentation_to_dataframe( "basal_forebrain", bflab ).to_csv( outfn + "basalforebrainbasic.csv" )

# import pandas as pd
# detdf=pd.read_csv( antspyt1w.get_data( "CIT168_Reinf_Learn_v1_label_descriptions_pad",  target_extension='.csv' ) )
antspyt1w.map_segmentation_to_dataframe( "CIT168_Reinf_Learn_v1_label_descriptions_pad", detlab ).to_csv( outfn + "LabelStats.csv" )
antspyt1w.map_segmentation_to_dataframe( "CIT168_T1w_700um_pad_adni_brainstem", bstlabel ).to_csv( outfn + "BrainStemLabelStats.csv" )

ants.plot( dti, detlab, axis=2, overlay_alpha = 0.85, cbar=False, nslices = 24, ncol=8, cbar_length=0.3, cbar_vertical=True, crop=True, filename=outfn+'slices.png' )
dticrop = ants.crop_image( dti, ants.threshold_image( bstlabel, 1, 99999 ) )
bstlabelcrop = ants.crop_image( bstlabel, ants.threshold_image( bstlabel, 1, 99999 ) )
ants.plot( dticrop, bstlabelcrop, axis=2, overlay_alpha = 0.85, cbar=False, nslices = 24, ncol=8, cbar_length=0.3, cbar_vertical=True, crop=True, filename=outfn+'brainstemslices.png' )

sys.exit(0)
# throw in some dkt for completeness
import antspynet
bfn = antspynet.get_antsxnet_data( "croppedMni152" )
templateb = ants.image_read( bfn )
templateb = ( templateb * antspynet.brain_extraction( templateb, 't1' ) ).iMath( "Normalize" )
myparc = antspyt1w.deep_brain_parcellation( dti, templateb, do_cortical_propagation=True )
dktfn = re.sub('brain_n4_dnz-SR.nii.gz', '', dtifn )
ants.image_write( myparc['tissue_segmentation'] ,  dktfn + "tissuesegmentationSR.nii.gz" )
antspyt1w.map_segmentation_to_dataframe( "tissues", myparc['tissue_segmentation'] ).to_csv( outfn + "tissuesegmentationSR.csv" )
dktfn = re.sub('brain_n4_dnz-SR.nii.gz', 'DKT', dtifn )
ants.image_write( myparc['dkt_parcellation'] ,  outfn + "DKTSR.nii.gz" )
ants.image_write( myparc['dkt_cortex'] ,  outfn + "DKTcortexSR.nii.gz" )
antspyt1w.map_segmentation_to_dataframe( "dkt", myparc['dkt_parcellation'] ).to_csv( outfn + "DKTSR.csv" )
antspyt1w.map_segmentation_to_dataframe( "dkt", myparc['dkt_cortex'] ).to_csv( outfn + "DKTcortexSR.csv" )
print( outfn + " complete" )

