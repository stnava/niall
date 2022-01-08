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
import random

import random, string

istest=True
dtifns = glob.glob( "/mnt/cluster/data/anatomicalLabels/Mindboggle101_volumes/*/*/t1weighted.nii.gz" )
if istest:
    dtifns = glob.glob( "/Users/stnava/data/anatomicalLabels/Mindboggle101_volumes/Extra-18_volumes/Twins-*/t1weighted.nii.gz" )

spre = "/mnt/cluster/data/anatomicalLabels/Mindboggle101_volumes/simulated_whole_brain/"
if istest:
    spre="/tmp/testit/"

if len( sys.argv ) > 1:
    fileindex = int(sys.argv[1])
else:
    fileindex=0

random.seed( fileindex )
def randword(length):
   letters = string.ascii_lowercase
   return ''.join(random.choice(letters) for i in range(length))
randstring = randword( 8 )

imgfn = dtifns[ fileindex ]
print( imgfn )
refimg = ants.image_read( antspyt1w.get_data( "CIT168_T1w_700um_pad", target_extension='.nii.gz' ))
refimg = ants.resample_image( refimg, [0.5,0.5,0.5] )
refimgseg = ants.image_read( antspyt1w.get_data( "det_atlas_25_pad_LR", target_extension='.nii.gz' ))
refimgsmall = ants.resample_image( refimg, [2.5,2.5,2.5] )
ifnbase = randstring + "_img"
sfnbase = randstring + "_seg"
pfnbase = randstring + "_pri"
outipre = spre + ifnbase
outspre = spre + sfnbase
outppre = spre + pfnbase
print(outipre)
print(outspre)
print(outppre)

img = ants.image_read( imgfn )
imgbxt = antspyt1w.brain_extraction( img, method='v1' )
img = antspyt1w.preprocess_intensity( img, imgbxt )
reg = ants.registration( refimgsmall, img, 'SyN', verbose=False )
if istest:
    reggd = reg
else:
    reggd = ants.registration( refimg, img, 'SyN', verbose=False )
seg2sub = ants.apply_transforms( img, refimgseg, reg['invtransforms'], interpolator='nearestNeighbor' )

# we build maps s.t. we have a composed tx that takes the template seg to the subject via:
# sim-map => inv1 => inv2

ilist = list()
ilist.append( [refimg] )
nsim = 64
if istest:
    nsim=4
uu = antspynet.randomly_transform_image_data( refimg, ilist,
    number_of_simulations = nsim,
    transform_type='scaleShear', sd_affine=0.05 )
deftx  = ants.transform_from_displacement_field( ants.image_read( reg['fwdtransforms'][0] ) )
deftxi = ants.transform_from_displacement_field( ants.image_read( reg['invtransforms'][1] ) )
deftxgood  = ants.transform_from_displacement_field( ants.image_read( reggd['fwdtransforms'][0] ) )
deftxigood = ants.transform_from_displacement_field( ants.image_read( reggd['invtransforms'][1] ) )
fwdaff = ants.read_transform( reggd['fwdtransforms'][1])

for k in range( nsim ):
    print( "k: " + str(k) )
    # the map to simulate the subject deformation is reg-fwd + sim-tx
    simtx = uu['simulated_transforms'][k]
    simtxinv = ants.invert_ants_transform( simtx )
    cmptx = ants.compose_ants_transforms( [fwdaff, simtx] ) # good
    subjectsim = ants.apply_ants_transform_to_image( cmptx, img, refimg, interpolation='linear' )
    # now generate the mapping for the template segmentation to the sim subject
    cmptxseg = ants.compose_ants_transforms( [simtx,deftxigood] ) # good
    segsim = ants.apply_ants_transform_to_image( cmptxseg, refimgseg, refimg, interpolation='nearestneighbor' )
    # now generate the mapping for the template segmentation to the sim subject
    cmptxprior = ants.compose_ants_transforms( [simtx,deftxi] ) # good
    priorsim = ants.apply_ants_transform_to_image( cmptxprior, refimgseg, refimg, interpolation='nearestneighbor' )
    bias_field = antspynet.simulate_bias_field( subjectsim, number_of_points=10,
        sd_bias_field=0.10, number_of_fitting_levels=4, mesh_size=1)
    subjectsim = subjectsim * (bias_field + 1)
    subjectsim = ants.rank_intensity( subjectsim )
    centroids = ants.label_image_centroids( segsim, segsim )
    mydf = pd.DataFrame({"Label":centroids['labels'],
        "x":centroids['vertices'][:,0],
        "y":centroids['vertices'][:,1],
        "z":centroids['vertices'][:,2]} )
    mydf.to_csv( outspre + "_sim_" + str(k) + "points.csv" )
    ants.image_write( segsim, outspre + "_sim_" + str(k) + ".nii.gz"  )
    ants.image_write( subjectsim, outipre + "_sim_" + str(k) + ".nii.gz"  )
    ants.image_write( priorsim, outppre + "_sim_" + str(k) + ".nii.gz"  )
    print( outspre + "_sim_" + str(k) + ".nii.gz" )
    print( outipre + "_sim_" + str(k) + ".nii.gz"  )
    print( outppre + "_sim_" + str(k) + ".nii.gz"  )

print("done")
