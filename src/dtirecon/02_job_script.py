import os
nth="24"
os.environ["TF_NUM_INTEROP_THREADS"] = nth
os.environ["TF_NUM_INTRAOP_THREADS"] = nth
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = nth
from os.path import exists
import glob
import re
from matplotlib import pyplot as plt
rootdir = "/mnt/cluster/data/PPMI2/PPMI/"
if not exists( rootdir ):
    rootdir = "/Users/stnava/data/PPMI2/"
    print("rootdir " + rootdir )
modality = "DTI_LR"
modality2 = "DTI_RL"
targetfns = glob.glob( rootdir + "*/*/*/*/dcm2niix/V0/*" +
    modality + "*dcm2niix-V0-SR.nii.gz" )
targetfns2 = glob.glob( rootdir + "*/*/*/*/dcm2niix/V0/*" +
    modality2 + "*dcm2niix-V0-SR.nii.gz" )
if len(targetfns) == 0:
    targetfns = glob.glob( rootdir + "*/*/*/*/*/dcm2niix/V0/*" +
    modality + "*dcm2niix-V0-SR.nii.gz" )
import sys
fileindex = 0
if len( sys.argv ) > 1:
    fileindex = int(sys.argv[1])
targetfn = targetfns[ fileindex ]
targetfn2 = re.sub(modality, modality2, targetfn )
targetsplit = targetfn.split("/")
print( targetfn )
targetfn2 = glob.glob( re.sub( targetsplit[9], "*", targetfn2 ))[0]
if not exists( targetfn ) or not exists( targetfn2 ) :
    print( targetfn2 + "does not exist")
    sys.exit(1)


# FIXME - more specific stuff below for this modality
mysubbed = re.sub( modality, 'DTIJoinL', targetfn )
mysubbed2 = re.sub( modality2, 'DTIJoinR', targetfn2 )
newprefixList = []
newoutdirList = []
for newout in [mysubbed, mysubbed2]:
    mysubbedsplit = newout.split("/")
    # define the directories and create them
    newoutdir = ''
    newprefix = ''
    keyindex = 10 # change for each case
    for k in range(keyindex):
        newoutdir = newoutdir + '/' + mysubbedsplit[k]
        if k > 4:
            newprefix = newprefix + mysubbedsplit[k] + '-'
    newprefix = newoutdir + '/' + newprefix
    # create the directory
    myx = os.path.isdir( newoutdir )
    print( "make " +  newoutdir + " " + str( myx ) )
    if not myx:
        os.makedirs( newoutdir, exist_ok=True )
    print( "made " +  newoutdir + " successfully " )
    print("newprefix: " + newprefix )
    newprefixList.append( newprefix )
    newoutdirList.append( newoutdir )


myoutfnexists = exists( mysubbed )
myoutfnexists2 = exists( newprefixList[1] + 'SRRGB.nii.gz' )
if myoutfnexists2:
    print( newprefixList[1] + 'SRRGB.nii.gz' + "exists already" )
    sys.exit()

print( "continue " +  mysubbed + " run " )

import ants
import antspymm
import tensorflow as tf
import antspyt1w
img1 = ants.image_read( targetfn )
img2 = ants.image_read( targetfn2 )
print("begin: " + newprefix )
if True:
    b0indices = antspymm.segment_timeseries_by_meanvalue( img1, 0.995 )['highermeans']
    dwp = antspymm.dewarp_imageset( [img1,img2], iterations=2, padding=6,
        target_idx = b0indices,
        syn_sampling = 20, syn_metric='mattes',
        type_of_transform = 'SyN',
        total_sigma = 0.0, random_seed=1,
        reg_iterations = [5,0,0] )

# ants.image_write( dwp['dewarped'][0], newprefixList[0] + 'SRdewarped.nii.gz' )
# ants.image_write( dwp['dewarped'][1], newprefixList[1] + 'SRdewarped.nii.gz' )

# now reconstruct DTI
import pandas as pd
outfn1 = newprefixList[0] + 'SRRGB.nii.gz'
zz=82
if not exists( outfn1 ):
    print("Begin Recon 1: " + outfn1 )
    bvec = re.sub( "-SR.nii.gz", ".bvec", targetfn )
    bval = re.sub( "-SR.nii.gz", ".bval", targetfn )
    b0indices = antspymm.segment_timeseries_by_meanvalue( img1, 0.995 )['highermeans']
    dd = antspymm.dipy_dti_recon( dwp['dewarped'][0], bval, bvec, median_radius=8, dilate=1,
        vol_idx = b0indices )
    plt.imshow(dd['RGB'].numpy()[zz,:,:,:])
    plt.savefig( newprefixList[0] + 'SRRGBsliceX.png' )
    plt.imshow(dd['RGB'].numpy()[:,zz,:,:])
    plt.savefig( newprefixList[0] + 'SRRGBsliceY.png' )
    plt.imshow(dd['RGB'].numpy()[:,:,zz,:])
    plt.savefig( newprefixList[0] + 'SRRGBsliceZ.png' )
    pd.DataFrame(data=dwp['FD'][0],columns=['FD'] ).to_csv(  newprefixList[0] + 'SR' + 'FD.csv')
    for mykey in ['MD','FA','RGB']:
        ants.image_write( dd[mykey],  newprefixList[0] + 'SR' + mykey + '.nii.gz' )

outfn1 = newprefixList[1] + 'SRRGB.nii.gz'
if not exists( outfn1 ):
    print("Begin Recon 2: " + outfn1 )
    b0indices = antspymm.segment_timeseries_by_meanvalue( dwp['dewarped'][1], 0.995 )['highermeans']
    bvec = re.sub( "-SR.nii.gz", ".bvec", targetfn2 )
    bval = re.sub( "-SR.nii.gz", ".bval", targetfn2 )
    ee = antspymm.dipy_dti_recon( img2, bval, bvec, median_radius=8, dilate=1,
        vol_idx = b0indices )
    plt.imshow(ee['RGB'].numpy()[zz,:,:,:])
    plt.savefig( newprefixList[1] + 'SRRGBsliceX.png' )
    plt.imshow(ee['RGB'].numpy()[:,zz,:,:])
    plt.savefig( newprefixList[1] + 'SRRGBsliceY.png' )
    plt.imshow(ee['RGB'].numpy()[:,:,zz,:])
    plt.savefig( newprefixList[1] + 'SRRGBsliceZ.png' )
    pd.DataFrame(data=dwp['FD'][1],columns=['FD'] ).to_csv(  newprefixList[1] + 'SR' + 'FD.csv')
    for mykey in ['MD','FA','RGB']:
        ants.image_write( ee[mykey],  newprefixList[1] + 'SR' + mykey + '.nii.gz' )


print("complete: " + newprefixList[0] + " & " + newprefixList[1] )
