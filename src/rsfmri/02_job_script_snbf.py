import os
import glob
import sys
import re
from os.path import exists
import ants
import antspyt1w
import antspymm
import antspynet
import tensorflow as tf
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def pastetoid( x, n = 10 ):
    xsplit = x.split("/")
    newoutdir=''
    newprefix=''
    for k in range(n):
        newoutdir = newoutdir + '/' + xsplit[k]
    return newoutdir

def directory2prefix( x, lo, hi, sep='-' ):
    xsplit = x.split("/")
    newprefix=xsplit[lo]
    for k in range(lo+1,hi):
        newprefix = newprefix + sep + xsplit[k]
    return x + "/" + newprefix

rootdir = "/mnt/cluster/data/PPMI2/PPMI/"
if not exists( rootdir ):
    rootdir = "/Users/stnava/data/PPMI2/"
    print("rootdir " + rootdir )
modality = "restingStatefMRI"
targetfns = glob.glob( rootdir + "*/*/*/*/dcm2niix/V0/*" +
    modality + "*dcm2niix-V0-SR.nii.gz" )
if len(targetfns) == 0:
    targetfns = glob.glob( rootdir + "*/*/*/*/*/dcm2niix/V0/*" +
    modality + "*dcm2niix-V0-SR.nii.gz" )
import sys

if 'fileindex' not in globals():
    fileindex = 22

if len( sys.argv ) > 1:
    fileindex = int(sys.argv[1])
targetfn = targetfns[ fileindex ]
targetsplit = targetfn.split("/")

newoutdir = re.sub( modality, 'restingNetworks', pastetoid( targetfn ) )
os.makedirs( newoutdir, exist_ok=True )
newprefix = directory2prefix( newoutdir, 6, 10 )
img1 = ants.image_read( targetfn )
print("begin: " + newprefix )
if 'dwp' not in globals():
    dwp = antspymm.dewarp_imageset( [img1], iterations=1, padding=8,
        target_idx = [7,8,9],
        syn_sampling = 20, syn_metric='mattes',
        type_of_transform = 'SyN',
        total_sigma = 0.0, random_seed=1,
        reg_iterations = [50,20] )


und = dwp['dewarpedmean']
bmask = antspynet.brain_extraction( und, 'bold' ).threshold_image( 0.3, 1.0 )
powers_areal_mni_itk = pd.read_csv(antspymm.get_data('powers_mni_itk', target_extension=".csv")) # power coordinates
ptidIndex = 6
dateIndex = ptidIndex + 1
t1fns = glob.glob( rootdir + targetsplit[ptidIndex] + "/" + targetsplit[dateIndex] + "/" + "T1wHierarchical/*/*brain_n4_dnz-SR.nii.gz" )
if len( t1fns ) == 0:
    print("Missing T1 for "+targetsplit[ptidIndex]+ " " + targetsplit[dateIndex] )
else:
    t1fn = t1fns[ len(t1fns) - 1 ] # take the last one
t1atx = re.sub('brain_n4_dnz-SR.nii.gz', 'SYNCC2CIT1680GenericAffine.mat', t1fn )
t1wtx = re.sub('brain_n4_dnz-SR.nii.gz', 'SYNCC2CIT1681Warp.nii.gz', t1fn )
t1seg = ants.image_read( re.sub('brain_n4_dnz-SR', 'tissuesegmentationSR',t1fn) )
t1 = ants.image_read( t1fn )

t1reg = ants.registration( und * bmask, t1, "SyNCC" ) # in practice use something different

# get BF labels
concattx =[]
concattx.append( t1reg['fwdtransforms'][0] )
concattx.append( t1reg['fwdtransforms'][1] )
concattx.append( t1wtx )
concattx.append( t1atx )
mybf = ants.image_read( antspyt1w.get_data( "CIT168_basal_forebrain_adni", target_extension='.nii.gz' ) )
mybf2t1 = ants.apply_transforms( t1, mybf, concattx[2:4], interpolator='nearestNeighbor')
ants.image_write( mybf2t1, re.sub( "brain_n4_dnz-SR" , "basalforebrain-SR" , t1fn)  )
mybf = ants.apply_transforms( und, mybf, concattx, interpolator='nearestNeighbor')

# ants.plot( t1*t1bxt, t1reg['warpedfixout'] , axis=2, overlay_alpha=0.25, ncol=8, nslices=24 )
# ants.plot( und, t1reg['warpedmovout'], overlay_alpha = 0.25, axis=2, nslices=24, ncol=6 )
boldseg = ants.apply_transforms( und, t1seg,
  t1reg['fwdtransforms'], interpolator = 'nearestNeighbor' ) * bmask
gmseg = ants.threshold_image( t1seg, 2, 2 ).iMath("MD",1)
gmseg = gmseg + ants.threshold_image( mybf2t1, 1, 4 )
gmseg = ants.threshold_image( gmseg, 1, 10 )
gmseg = ants.apply_transforms( und, gmseg,
  t1reg['fwdtransforms'], interpolator = 'nearestNeighbor' )  * bmask
csfAndWM = ( ants.threshold_image( t1seg, 1, 1 ) +
             ants.threshold_image( t1seg, 3, 3 ) ).morphology("erode",2)
csfAndWM = ants.apply_transforms( und, csfAndWM,
  t1reg['fwdtransforms'], interpolator = 'nearestNeighbor' )  * bmask
ants.image_write( gmseg, newprefix + 'gmseg.nii.gz' )
dwpind = 0
nc = 6
mycompcor = ants.compcor( dwp['dewarped'][dwpind],
  ncompcor=nc, quantile=0.90, mask = csfAndWM,
  filter_type='polynomial', degree=2 )

nt = dwp['dewarped'][dwpind].shape[3]
tr = ants.get_spacing( dwp['dewarped'][dwpind] )[3]
highMotionTimes = np.where( dwp['FD'][dwpind] >= 1.0 )
print( "highMotionTimes: " + str(highMotionTimes) )
goodtimes = np.where( dwp['FD'][dwpind] < 0.5 )
spa, spt = 1.5, 0.5 # spatial, temporal - which we ignore b/c of frequency filtering
smth = ( spa, spa, spa, spt ) # this is for sigmaInPhysicalCoordinates = F
simg = ants.smooth_image(dwp['dewarped'][dwpind], smth, sigma_in_physical_coordinates = False )

nuisance = mycompcor['components']
nuisance = np.c_[ nuisance, mycompcor['basis'] ]
nuisance = np.c_[ nuisance, dwp['FD'][dwpind] ]

gmmat = ants.timeseries_to_matrix( simg, gmseg )
f=[0.01,0.08]
gmmat = ants.bandpass_filter_matrix( gmmat, tr = tr, lowf=f[0], highf=f[1] ) # some would argue against this
gmsignal = gmmat.mean( axis = 1 )
nuisance = np.c_[ nuisance, gmsignal ]
gmmat = ants.regress_components( gmmat, nuisance )

# get SNc labels
sncfn = re.sub('brain_n4_dnz-SR', 'SYNCC2CIT168Labels', t1fn)
if not exists( sncfn ) :
    print( sncfn + ' does not exist' )
mysnc = ants.image_read( sncfn )
mysncalone = ants.threshold_image( mysnc, 7, 7 ) + ants.threshold_image( mysnc, 23, 23 ) + ants.threshold_image( mysnc, 9, 9 ) + ants.threshold_image( mysnc, 25, 25 )
mysncalone = ants.apply_transforms( und, mysncalone, t1reg['fwdtransforms'], interpolator='nearestNeighbor')
# mycsv = pd.read_csv( antspyt1w.get_data( "CIT168_Reinf_Learn_v1_label_descriptions_pad" , target_extension='.csv' ) )

# process signal from snc bold
dfnmat = ants.timeseries_to_matrix( simg, mysncalone )
dfnmat = ants.bandpass_filter_matrix( dfnmat, tr = tr, lowf=f[0], highf=f[1]  )
dfnmat = ants.regress_components( dfnmat, nuisance )
dfnsignal = dfnmat.mean( axis = 1 )
from scipy.stats.stats import pearsonr
gmmatDFNCorr = np.zeros( gmmat.shape[1] )
for k in range( gmmat.shape[1] ):
    gmmatDFNCorr[ k ] = pearsonr( dfnsignal, gmmat[:,k] )[0]
corrImg = ants.make_image( gmseg, gmmatDFNCorr  )
ants.image_write( corrImg, newprefix + "SNCConnectivity.nii.gz" )
ants.image_write( mysncalone, newprefix + "SNC.nii.gz" )
mybfstats = ants.label_stats( corrImg, mybf )
mybfstats.to_csv(  newprefix + 'SNCbasalforebrainCorr.csv' )
ants.image_write( mybf, newprefix + 'basalforebrain.nii.gz' )
