import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import glob
import numpy as np
import random
import ants
import antspynet
import re
import pandas as pd
from tensorflow.keras.layers import Conv3D
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
K.set_floatx("float32")

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


def batch_generator(
    image_filenames,
    segmentation_filenames,
    image_size, 
    group_labels_in,
    batch_size=32,
    ):
    X = np.zeros( (batch_size, *(image_size), 1) )
    Xcc = np.zeros( (batch_size, *(image_size), 3) )
    Y = np.zeros( (batch_size, *(image_size) ) )
    Ypr = np.zeros( (batch_size, *(image_size) ) )
    npts = len( group_labels_in ) - 1
    Ypts = np.zeros( ( batch_size,  npts, 3 ) )
    batch_count = 0
    print("BeginBatch")
    while batch_count < batch_size:
        i = random.sample(list(range(len(image_filenames))), 1)[0]
        t1 = ants.image_read(image_filenames[i]).iMath("Normalize")
        mycc = coordinate_images( t1 * 0 + 1 )
        zz=pd.read_csv( pt_fns[i] )
        mypr = ants.image_read( pr_fns[i] )
        seg = ants.image_read(segmentation_filenames[i])
        X[batch_count,:,:,:,0] = t1.numpy()
        Xcc[batch_count,:,:,:,0] = mycc[0].numpy()
        Xcc[batch_count,:,:,:,1] = mycc[1].numpy()
        Xcc[batch_count,:,:,:,2] = mycc[2].numpy()
        Y[batch_count,:,:,:] = seg.numpy()
        Ypr[batch_count,:,:,:] = mypr.numpy()
        Ypts[batch_count,:,0]=zz['x']
        Ypts[batch_count,:,1]=zz['y']
        Ypts[batch_count,:,2]=zz['z']
        batch_count = batch_count + 1
        if batch_count >= batch_size:
                break

    encY = antspynet.encode_unet(Y.astype('int'), group_labels_in )
    encYpr = antspynet.encode_unet(Ypr.astype('int'), group_labels_in[1:len(group_labels_in)] )
    return X, Xcc, Y, encY, Ypts, encYpr

data_directory = "/mnt/cluster/data/anatomicalLabels/Mindboggle101_volumes/simulated/"

exfn = "Mindboggle_Afterthought-1_T1wHierarchical_brain_n4_dnz-SR_sim_11.nii.gz"
eximg = ants.image_read( data_directory + exfn )
group_1_labels = [0,1,2,5,6,17,18,21,22]
group_2_labels = [0,7,8,9,23,24,25,33,34]
group_labels = group_1_labels

image_size = [160,160,112]


print("Loading brain data.")

t1_fns = glob.glob( data_directory + "*brain_n4_dnz-SR_sim_*.nii.gz" )
seg_fns = t1_fns.copy()
for k in range(len(t1_fns)):
    seg_fns[k] = re.sub( "brain_n4_dnz-SR_sim", "SRHIERcit168lab_sim", seg_fns[k] )
pt_fns = t1_fns.copy()
pr_fns = t1_fns.copy()
for k in range(len(t1_fns)):
    pr_fns[k] = re.sub( "brain_n4_dnz-SR_sim", "cit168priors_sim", pr_fns[k] )
    pt_fns[k] = re.sub( "brain_n4_dnz-SR_sim", "SRHIERcit168lab_sim", seg_fns[k] )
    pt_fns[k] = re.sub( ".nii.gz", "points.csv", pt_fns[k] )

print("Total training image files: ", len(t1_fns))

print( "Training")
group_labels = np.unique(seg.numpy()).astype(int)
###
#
# Set up the training generator
#

batch_size = 64
generator = batch_generator( t1_fns, seg_fns, image_size=image_size,
    batch_size = batch_size, group_labels_in=group_labels)

import random, string

def randword(length):
   letters = string.ascii_lowercase
   return ''.join(random.choice(letters) for i in range(length))

randstring = randword( 8 )
outpre = "/mnt/cluster/data/anatomicalLabels/Mindboggle101_volumes/numpy/MB_" + randstring
print( outpre )
np.save( outpre + "_Ximages.npy", generator[0] )
np.save( outpre + "_Xcc.npy", generator[1] )
np.save( outpre + "_Y.npy", generator[2] )
np.save( outpre + "_Y1hot.npy", generator[3] )
np.save( outpre + "_Ypts.npy", generator[4] )
np.save( outpre + "_Xprior.npy", generator[5])


