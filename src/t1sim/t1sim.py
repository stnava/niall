import os
import glob
import sys
from os.path import exists
import ants
import antspynet
import antspyt1w
import numpy as np
import pandas as pd
import re

def special_crop( x, pt, domainer ):
        pti = np.round( ants.transform_physical_point_to_index( x, pt ) )
        xdim = x.shape
        for k in range(len(xdim)):
            if pti[k] < 0:
                pti[k]=0
            if pti[k] > (xdim[k]-1):
                pti[k]=(xdim[k]-1)
        mim = ants.make_image( domainer )
        ptioff = pti.copy()
        for k in range(len(xdim)):
            ptioff[k] = ptioff[k] - np.round( domainer[k] / 2 )
        domainerlo = []
        domainerhi = []
        for k in range(len(xdim)):
            domainerlo.append( int(ptioff[k] - 1) )
            domainerhi.append( int(ptioff[k] + 1) )
        loi = ants.crop_indices( x, tuple(domainerlo), tuple(domainerhi) )
        mim = ants.copy_image_info(loi,mim)
        return ants.resample_image_to_target( x, mim )

dtifns = glob.glob( "/mnt/cluster/data/anatomicalLabels/Mindboggle101_volumes/*/*/T1wHierarchical/*T1wHierarchical_brain_n4_dnz-SR.nii.gz" )

if len( sys.argv ) > 1:
    fileindex = int(sys.argv[1])
else:
    fileindex=0

imgfn = dtifns[ fileindex ]
print( imgfn )
refimg = ants.image_read( antspyt1w.get_data( "CIT168_T1w_700um_pad_adni", target_extension='.nii.gz' ))
refimg = ants.resample_image( refimg, [0.5,0.5,0.5] )
refimgseg = ants.image_read( antspyt1w.get_data( "det_atlas_25_pad_LR_adni", target_extension='.nii.gz' ))
refimgsmall = ants.resample_image( refimg, [2.5,2.5,2.5] )
segfn = re.sub( "_brain_n4_dnz-SR.nii.gz", "_SRHIERcit168lab.nii.gz", imgfn)
outipre = re.sub( ".nii.gz", "", imgfn)
outspre = re.sub( ".nii.gz", "", segfn)
ifnbase = re.sub( ".nii.gz", "", os.path.basename( imgfn ) )
sfnbase = re.sub( ".nii.gz", "",  os.path.basename( segfn ) )
spre = "/mnt/cluster/data/anatomicalLabels/Mindboggle101_volumes/simulated/"
outipre = spre + ifnbase
outspre = spre + sfnbase
outppre = re.sub( "T1wHierarchical_SRHIERcit168lab", "T1wHierarchical_cit168priors", outspre )
print(outipre)
print(outspre)
print(outppre)
img = ants.image_read( imgfn )
seg = ants.image_read( segfn )
reg = ants.registration( refimgsmall, img, 'SyN', verbose=False )
img = ants.apply_transforms( refimg, img, reg['fwdtransforms'][1] )
seg = ants.apply_transforms( refimg, seg, reg['fwdtransforms'][1], interpolator='genericLabel' )

ilist = list()
ilist.append( [img] )
slist = [seg]
nsim = 64
uu = antspynet.randomly_transform_image_data( refimg, ilist, slist,
    number_of_simulations = nsim,
    transform_type='scaleShear', sd_affine=0.05 )
deftx  = ants.transform_from_displacement_field( ants.image_read( reg['fwdtransforms'][0] ) )
deftxi = ants.transform_from_displacement_field( ants.image_read( reg['invtransforms'][1] ) )

for k in range( nsim ):
    print( "k: " + str(k) )
    bmask = ants.threshold_image( uu['simulated_segmentation_images'][k], 1, 999 )
    pt = list( ants.get_center_of_mass( bmask ) )
    pt[1] = pt[1] + 10.0 
    temp = uu['simulated_images'][k][0]
    bias_field = antspynet.simulate_bias_field( temp, number_of_points=10,
        sd_bias_field=0.10, number_of_fitting_levels=4, mesh_size=1)
    temp2 = temp * (bias_field + 1)
    cmptx = ants.compose_ants_transforms( [( uu['simulated_transforms'][k]), deftxi ] ) # good
    priors = ants.apply_ants_transform_to_image( cmptx, refimgseg, temp2, interpolation='nearestneighbor')
    # ants.label_overlap_measures( zz, uu['simulated_segmentation_images'][k] )
    dmn = [160,160,112]
    cimg = special_crop( temp2, pt, dmn )
    cimgs = special_crop(  uu['simulated_segmentation_images'][k],  pt, dmn )
    pimg = special_crop( priors,  pt, dmn )
    # ants.plot( cimg, cimgs, axis=2, crop=False, nslices=21, ncol=7  )
    centroids = ants.label_image_centroids( cimgs, cimgs )
    mydf = pd.DataFrame({"Label":centroids['labels'],
        "x":centroids['vertices'][:,0],
        "y":centroids['vertices'][:,1],
        "z":centroids['vertices'][:,2]} )
    mydf.to_csv( outspre + "_sim_" + str(k) + "points.csv" )
    ants.image_write( cimgs, outspre + "_sim_" + str(k) + ".nii.gz"  )
    ants.image_write( cimg, outipre + "_sim_" + str(k) + ".nii.gz"  )
    ants.image_write( pimg, outppre + "_sim_" + str(k) + ".nii.gz"  )
    print( outspre + "_sim_" + str(k) + ".nii.gz" )
    print( outipre + "_sim_" + str(k) + ".nii.gz"  )
    print( outppre + "_sim_" + str(k) + ".nii.gz"  )

print("done")

