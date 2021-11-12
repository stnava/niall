import os
nth="12"
os.environ["TF_NUM_INTEROP_THREADS"] = nth
os.environ["TF_NUM_INTRAOP_THREADS"] = nth
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = nth
import glob
import sys
import re
from os.path import exists
import ants
import antspymm
import antspynet
import tensorflow as tf
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
rootdir = "/mnt/cluster/data/PPMI2/PPMI/"
if not exists( rootdir ):
    rootdir = "/Users/stnava/data/PPMI2/"
    print("rootdir " + rootdir )
modality = "restingStatefMRI"
targetfns = glob.glob( rootdir + "*/*/*/*/dcm2niix/V0/*" +
    modality + "*dcm2niix-V0.nii.gz" )
if len(targetfns) == 0:
    targetfns = glob.glob( rootdir + "*/*/*/*/*/dcm2niix/V0/*" +
    modality + "*dcm2niix-V0.nii.gz" )
import sys
fileindex = 1
if len( sys.argv ) > 1:
    fileindex = int(sys.argv[1])
targetfn = targetfns[ fileindex ]
targetsplit = targetfn.split("/")
print( targetfn )
mysubbed = re.sub( modality, 'restingNetworks', targetfn )
mysubbedsplit = mysubbed.split("/")
newoutdir = ''
newprefix = ''
keyindex = 10 # change for each case
for k in range(keyindex):
    newoutdir = newoutdir + '/' + mysubbedsplit[k]

newprefix = os.path.splitext(mysubbed)[0]
newprefix = os.path.splitext(newprefix)[0]

# create the directory
myx = os.path.isdir( newoutdir )
print( "make " +  newoutdir + " " + str( myx ) )
if not myx:
    os.makedirs( newoutdir, exist_ok=True )
print( "made " +  newoutdir + " successfully " )
print("newprefix: " + newprefix )


img1 = ants.image_read( targetfn )
print("begin: " + newprefix )
if 'dwp' not in globals():
    dwp = antspymm.dewarp_imageset( [img1], iterations=1, padding=8,
        target_idx = [8,9,10],
        syn_sampling = 20, syn_metric='mattes',
        type_of_transform = 'SyN',
        total_sigma = 0.0, random_seed=1,
        reg_iterations = [50,20] )


und = dwp['dewarpedmean']
bmask = antspynet.brain_extraction( und, 'bold' ).threshold_image( 0.3, 1.0 )
powers_areal_mni_itk = pd.read_csv(antspymm.get_data('powers_mni_itk', target_extension=".csv")) # power coordinates
ptidIndex = 6
dateIndex = ptidIndex + 1
t1fns = glob.glob( rootdir + "*/*/*/*/*/dcm2niix/V0/*"+mysubbedsplit[ptidIndex]+"*" + mysubbedsplit[dateIndex] + "*T1*dcm2niix-V0.nii.gz" )
if len( t1fns ) == 0:
    print("Missing T1 for "+mysubbedsplit[ptidIndex]+ " " + mysubbedsplit[dateIndex] )
else:
    t1fn = t1fns[ len(t1fns) - 1 ] # take the last one
mysubbedt1 = re.sub('T1w', 'T1wHierarchical', t1fn )
derka
t1fn = antspymm.get_data( 'LS2001_3T_T1w_MPR1_gdc' , target_extension='.nii.gz' )
t1 = ants.image_read( t1fn ).n3_bias_field_correction( 8 ).n3_bias_field_correction( 4 )
t1bxt = antspynet.brain_extraction( t1, 't1' ).threshold_image( 0.3, 1.0 )
t1seg = antspynet.deep_atropos( t1 )
t1reg = ants.registration( und * bmask, t1 * t1bxt, "SyN" ) # in practice use something different
# ants.plot( t1*t1bxt, t1reg['warpedfixout'] , axis=2, overlay_alpha=0.25, ncol=8, nslices=24 )
# ants.plot( und, t1reg['warpedmovout'], overlay_alpha = 0.25, axis=2, nslices=24, ncol=6 )
boldseg = ants.apply_transforms( und, t1seg['segmentation_image'],
  t1reg['fwdtransforms'], interpolator = 'nearestNeighbor' )
# ants.plot( und, boldseg, overlay_alpha = 0.25, axis=2, nslices=24, ncol=6 )
csfAndWM = ( ants.threshold_image( boldseg, 1, 1 ) +
             ants.threshold_image( boldseg, 3, 3 ) ).morphology("erode",1)
dwpind = 1
mycompcor = ants.compcor( dwp['dewarped'][dwpind],
  ncompcor=6, quantile=0.80, mask = csfAndWM,
  filter_type='polynomial', degree=4 )

nt = dwp['dewarped'][dwpind].shape[3]
import matplotlib.pyplot as plt
plt.plot(  range( nt ), mycompcor['components'][:,0] )
# plt.show()
plt.plot(  range( nt ), mycompcor['components'][:,1] )
# plt.show()

myvoxes = range(powers_areal_mni_itk.shape[0])
anat = powers_areal_mni_itk['Anatomy']
syst = powers_areal_mni_itk['SystemName']
Brod = powers_areal_mni_itk['Brodmann']
xAAL  = powers_areal_mni_itk['AAL']
ch2 = ants.image_read( ants.get_ants_data( "ch2" ) )
if 'treg' not in globals():
    treg = ants.registration( t1 * t1bxt, ch2, 'SyN' )
concatx2 = treg['invtransforms'] + t1reg['invtransforms']
pts2bold = ants.apply_transforms_to_points( 3, powers_areal_mni_itk, concatx2,whichtoinvert = ( True, False, True, False ) )
locations = pts2bold.iloc[:,:3].values
ptImg = ants.make_points_image( locations, bmask, radius = 2 )
# ants.plot( und, ptImg, axis=2, nslices=24, ncol=8 )

tr = ants.get_spacing( dwp['dewarped'][dwpind] )[3]
highMotionTimes = np.where( dwp['FD'][dwpind] >= 0.5 )
print( "highMotionTimes: " + str(highMotionTimes) )
goodtimes = np.where( dwp['FD'][dwpind] < 0.5 )
gmseg = ants.threshold_image( boldseg, 2, 2 )
spa, spt = 1.5, 0.0 # spatial, temporal - which we ignore b/c of frequency filtering
smth = ( spa, spa, spa, spt ) # this is for sigmaInPhysicalCoordinates = F
simg = ants.smooth_image(dwp['dewarped'][dwpind], smth, sigma_in_physical_coordinates = False )

nuisance = mycompcor['components']
nuisance = np.c_[ nuisance, mycompcor['basis'] ]
nuisance = np.c_[ nuisance, dwp['FD'][dwpind] ]

gmmat = ants.timeseries_to_matrix( simg, gmseg )
gmmat = ants.bandpass_filter_matrix( gmmat, tr = tr, lowf=0.03, highf=0.08 ) # some would argue against this
gmmat = ants.regress_components( gmmat, nuisance )

postCing = powers_areal_mni_itk['AAL'].unique()[9]
networks = powers_areal_mni_itk['SystemName'].unique()
ww = np.where( powers_areal_mni_itk['SystemName'] == networks[5] )[0]
dfnImg = ants.make_points_image(pts2bold.iloc[ww,:3].values, bmask, radius=1).threshold_image( 1, 400 )
# ants.plot( und, dfnImg, axis=2, nslices=24, ncol=8 )

dfnmat = ants.timeseries_to_matrix( simg, ants.threshold_image( dfnImg * gmseg, 1, dfnImg.max() ) )
dfnmat = ants.bandpass_filter_matrix( dfnmat, tr = tr, lowf=0.01, highf=0.09  )
dfnmat = ants.regress_components( dfnmat, nuisance )
dfnsignal = dfnmat.mean( axis = 1 )

from scipy.stats.stats import pearsonr
gmmatDFNCorr = np.zeros( gmmat.shape[1] )
for k in range( gmmat.shape[1] ):
    gmmatDFNCorr[ k ] = pearsonr( dfnsignal, gmmat[:,k] )[0]

corrImg = ants.make_image( gmseg, gmmatDFNCorr  )

corrImgPos = corrImg * ants.threshold_image( corrImg, 0.25, 1 )
ants.plot( und, corrImgPos, axis=2, overlay_alpha = 0.6, cbar=False, nslices = 24, ncol=8, cbar_length=0.3, cbar_vertical=True )
