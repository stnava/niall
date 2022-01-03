import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import os.path

import glob
import numpy as np
import sys
import random
if len( sys.argv ) == 1:
    rseed=0
else:
    rseed = int(sys.argv[1])
random.seed( rseed )
import ants
import antspynet
import antspyt1w
import re
import pandas as pd
import tensorflow as tf
tf.random.set_seed( rseed )
import tensorflow.keras as keras
import tensorflow.keras.backend as K
K.set_floatx("float32")

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

def coordinate_images( mask ):
  idim = mask.dimension
  myr = list()
  for k in range(idim):
      myr.append(0)
  temp = ants.get_neighborhood_in_mask( mask, mask, myr,
    boundary_condition = "image", spatial_info = True,
    physical_coordinates = True, get_gradient = False)
  ilist = []
  for i in range(idim):
      ilist.append( ants.make_image( mask, temp['indices'][:,i] ) )
  return ilist

nbmtemplate = ants.image_read(antspyt1w.get_data("CIT168_T1w_700um_pad_adni", target_extension=".nii.gz"))
nbmtemplate = ants.resample_image( nbmtemplate, [0.5,0.5,0.5] )
templateSmall = ants.resample_image( nbmtemplate, [2.0,2.0,2.0] )
myprior = ants.image_read( antspyt1w.get_data( "det_atlas_25_pad_LR_adni",  target_extension=".nii.gz"))
patchSize = [ 160,160,112 ]

def batch_generator(
    image_filenames,
    segmentation_filenames,
    image_size,
    group_labels_in,
    batch_size=32,
    ):
    X = np.zeros( (batch_size, *(patchSize), 1) )
    Xcc = np.zeros( (batch_size, *(patchSize), 3) )
    Y = np.zeros( (batch_size, *(patchSize) ) )
    Ypr = np.zeros( (batch_size, *(patchSize) ) )
    print( X.shape )
    npts = len( group_labels_in ) - 1
    npts = 34
    Ypts = np.zeros( ( batch_size,  npts, 3 ) )
    batch_count = 0
    print("BeginBatch")
    while batch_count < batch_size:
        i = random.sample(list(range(20,len(image_filenames))), 1)[0]
        print( str( batch_count ) + " " + image_filenames[i] )
        t1 = ants.image_read(image_filenames[i])
        t1 = ants.iMath( t1, "TruncateIntensity", 0.0001, 0.999 ).iMath("Normalize")
        orireg = ants.registration(
            fixed = templateSmall,
            moving = t1,
            type_of_transform="SyN", verbose=False )
        t1 = ants.apply_transforms( nbmtemplate, t1, orireg['fwdtransforms'][1] )
        mypr =  ants.apply_transforms( t1, myprior,
            orireg['invtransforms'][1], interpolator='nearestNeighbor'  )
        bmask = ants.threshold_image( mypr, 1, 999 )
        pt = list( ants.get_center_of_mass( bmask ) )
        pt[1] = pt[1] + 10.0
        t1 = special_crop( t1, pt, patchSize)
        mycc = coordinate_images( t1 * 0 + 1 )
        mypr = special_crop( mypr, pt, patchSize)
        # ants.image_read( pr_fns[i] )
        # mypr = ants.mask_image( mypr, mypr, group_labels_in, binarize=False )
        seg = ants.image_read(segmentation_filenames[i])
        seg = ants.apply_transforms( nbmtemplate, seg, orireg['fwdtransforms'][1], interpolator='nearestNeighbor' )
        centroids = ants.label_image_centroids( seg, seg )
        mydf = pd.DataFrame({"Label":centroids['labels'],
          "x":centroids['vertices'][:,0],
          "y":centroids['vertices'][:,1],
          "z":centroids['vertices'][:,2]} )
        seg = special_crop( seg, pt, patchSize )
        # seg = ants.mask_image( seg, seg, group_labels_in, binarize=False )
        print("T1")
        print(t1)
        X[batch_count,:,:,:,0] = t1.numpy()
        print("cc")
        print( mycc[0] )
        Xcc[batch_count,:,:,:,0] = mycc[0].numpy()
        Xcc[batch_count,:,:,:,1] = mycc[1].numpy()
        Xcc[batch_count,:,:,:,2] = mycc[2].numpy()
        print("seg")
        print(seg)
        Y[batch_count,:,:,:] = seg.numpy()
        print("pr")
        print(mypr)
        Ypr[batch_count,:,:,:] = mypr.numpy()
        Ypts[batch_count,:,0]=mydf['x']
        Ypts[batch_count,:,1]=mydf['y']
        Ypts[batch_count,:,2]=mydf['z']
        batch_count = batch_count + 1
        if batch_count >= batch_size:
                break

    # encY = antspynet.encode_unet(Y.astype('int'), group_labels_in )
    # encYpr = antspynet.encode_unet(Ypr.astype('int'), group_labels_in[1:len(group_labels_in)] )
    return X, Xcc, Y, Ypts, Ypr

data_directory = "/mnt/cluster/data/anatomicalLabels/Mindboggle101_volumes/simulated/"
exfn="/mnt/cluster/data/SRPBS_multidisorder_MRI/traveling_subjects/SRPBTravel/sub-041/T1wHierarchical/SRPBTravel-sub-041-T1wHierarchical-brain_n4_dnz-SR.nii.gz"
exsegfn = re.sub( "brain_n4_dnz-SR.nii.gz", "SRHIERcit168lab.nii.gz", exfn )
data_directory = "/mnt/cluster/data/SRPBS_multidisorder_MRI/traveling_subjects/SRPBTravel/"
eximg = ants.image_read( exfn )
group_1_labels = [0,1,2,5,6,17,18,21,22]
group_2_labels = [0,7,8,9,23,24,25,33,34]
group_labels = group_1_labels

image_size = [160,160,112]


print("Loading brain data.")

t1_fns = glob.glob( data_directory + "*/*/*brain_n4_dnz-SR.nii.gz" )
seg_fns = t1_fns.copy()
for k in range(len(t1_fns)):
    seg_fns[k] = re.sub( "brain_n4_dnz-SR", "SRHIERcit168lab", seg_fns[k] )
pt_fns = t1_fns.copy()
pr_fns = t1_fns.copy()
for k in range(len(t1_fns)):
    pr_fns[k] = re.sub( "brain_n4_dnz-SR", "cit168priors", pr_fns[k] )
    pt_fns[k] = re.sub( "brain_n4_dnz-SR", "SRHIERcit168lab", seg_fns[k] )
    pt_fns[k] = re.sub( ".nii.gz", "points.csv", pt_fns[k] )

print("Total training image files: ", len(t1_fns))
import random, string
def randword(length):
   letters = string.ascii_lowercase
   return ''.join(random.choice(letters) for i in range(length))
randstring = randword( 8 )
outpre = "/mnt/cluster/data/SRPBS_multidisorder_MRI/traveling_subjects/numpy/TR_" + randstring
print( outpre )
seg = ants.image_read( seg_fns[0] )
group_labels = np.unique(seg.numpy()).astype(int)
###
#
# Set up the training generator
#

batch_size = 32
generator = batch_generator( t1_fns,
        seg_fns,
        image_size=image_size,
        batch_size = batch_size,
        group_labels_in=group_labels )

np.save( outpre + "_Ximages.npy", generator[0] )
np.save( outpre + "_Xcc.npy", generator[1] )
np.save( outpre + "_Y.npy", generator[2] )
# np.save( outpre + "_Y1hot.npy", generator[3] )
np.save( outpre + "_Ypts.npy", generator[3] )
np.save( outpre + "_Xprior.npy", generator[4])
