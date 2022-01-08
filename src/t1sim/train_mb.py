import os
import glob
import numpy as np
import random
import ants
import antspynet
import re

from os.path import exists

from tensorflow.keras.layers import Conv3D
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
K.set_floatx("float32")

def tfsubset( x, indices ):
    with tf.device('/CPU:0'):
        outlist = []
        for k in indices:
            outlist.append( x[:,:,:,:,int(k)] )
        return tf.stack( outlist, axis=4 )


def tfsubsetbatch( x, indices ):
    with tf.device('/CPU:0'):
        outlist2 = []
        for j in range( len( x ) ):
            outlist = []
            for k in indices:
                if len( x[j].shape ) == 5:
                    outlist.append( x[j][k,:,:,:,:] )
                if len( x[j].shape ) == 4:
                    outlist.append( x[j][k,:,:,:] )
            outlist2.append( tf.stack( outlist, axis=0 ) )
    return outlist2

def batch_generator( verbose=False ):
    with tf.device('/CPU:0'):
        allexist=False
        while not allexist:
            i = random.sample(list(range(len(t1_fns))), 1)[0]
            if exists(t1_fns[i]) and exists(pr_fns[i]) and exists(seg_fns[i]):
                if verbose:
                    print(i)
                x = np.load(t1_fns[i])
                xp = np.load(pr_fns[i])
                xp = tf.one_hot( xp, nLabels )
                xp = tfsubset( xp, group_labels[1:len(group_labels)] )
                y = np.load(seg_fns[i])
                y = tfsubset( tf.one_hot( y, nLabels ), group_labels )
                allexist=True
        y2 = y[:,:,:,:,1] + y[:,:,:,:,5] + y[:,:,:,:,2] + y[:,:,:,:,6] + y[:,:,:,:,3] + y[:,:,:,:,4] + y[:,:,:,:,7] + y[:,:,:,:,8]
        return tf.concat( [x, xp], axis=4), [y, y2]

data_directory = "/Users/stnava/Downloads/temp/testtrain/numpy/"
if not os.path.isdir( data_directory ):
    data_directory = "/raid/data_BA/cit168training/numpy/"

# load an example image
exfn = "../simulated/Mindboggle_Afterthought-1_T1wHierarchical_brain_n4_dnz-SR_sim_11.nii.gz"
eximg = ants.image_read( data_directory + exfn )
exfn = "../simulated/Mindboggle_Afterthought-1_T1wHierarchical_SRHIERcit168lab_sim_11.nii.gz"
exseg = ants.image_read( data_directory + exfn )
ulabels = np.unique( exseg.numpy() )
nLabels = len( ulabels )
group_1_labels = [0,1,2,5, 6,17,18,21,22]
group_2_labels = [0,7,8,9,23,24,25,33,34]
group_labels = group_2_labels
image_size = list( eximg.shape )
number_of_classification_labels = len(group_labels)
number_of_channels = len(group_labels) # image + coord conv
################################################
unet0 = antspynet.create_unet_model_3d(
         [ None, None, None, number_of_channels ],
         number_of_outputs = 1, # number of landmarks must be known
         number_of_layers = 4, # should optimize this wrt criterion
         number_of_filters_at_base_layer = 32, # should optimize this wrt criterion
         convolution_kernel_size = 3, # maybe should optimize this wrt criterion
         deconvolution_kernel_size = 2,
         pool_size = 2,
         strides = 2,
         dropout_rate = 0.0,
         weight_decay = 0,
         additional_options = "nnUnetActivationStyle",
         mode =  "sigmoid" )

unet1 = antspynet.create_unet_model_3d(
    [None,None,None,2],
    number_of_outputs=number_of_classification_labels,
    mode="classification",
    number_of_filters=(32, 64, 96, 128, 256),
    convolution_kernel_size=(3, 3, 3),
    deconvolution_kernel_size=(2, 2, 2),
    dropout_rate=0.2,
    weight_decay=0,
    additional_options = "nnUnetActivationStyle")

temp = tf.split( unet0.inputs[0], 9, axis=4 )
temp[1] = unet0.outputs[0]
newmult = tf.concat( temp[0:2], axis=4 )
unetonnet = unet1( newmult )
unet_model = tf.keras.models.Model(
        unet0.inputs,
        [ unetonnet,  unet0.outputs[0] ] )

wts = [0.001]
for k in range(1,number_of_classification_labels):
    wts.append(0.01)
weighted_loss = antspynet.weighted_categorical_crossentropy(weights=wts)
dice_loss = antspynet.multilabel_dice_coefficient(dimensionality=3, smoothing_factor=1e-5)
binary_dice_loss = antspynet.binary_dice_coefficient(smoothing_factor=1e-5)


weights_filename = "deepCIT168_sn_rank.h5"
csv_filename = re.sub("h5", "csv", weights_filename)

if os.path.exists(weights_filename):
    print("Load prior weights")
    unet_model.load_weights(weights_filename)

################################################
#  Load the data
################################################
t1_fns = glob.glob( data_directory + "*Ximages.npy" )
seg_fns = t1_fns.copy()
pr_fns = t1_fns.copy()
for k in range(len(t1_fns)):
    seg_fns[k] = re.sub( "Ximages.npy", "Y.npy", seg_fns[k] )
    pr_fns[k] = re.sub( "Ximages.npy", "Xprior.npy", pr_fns[k] )

print("Total training image files: ", len(t1_fns))

import pandas as pd
mydf = None

epoch = 1
num_epochs = 20000
optimizerE = tf.keras.optimizers.Adam(1.e-4)
batchsize = 2

# load the testing data
with tf.device('/CPU:0'):
    testX = np.load( "/raid/data_BA/cit168training/numpy_test/TR_mynbiqpm_Ximages.npy" )
    testXp = np.load( "/raid/data_BA/cit168training/numpy_test/TR_mynbiqpm_Xprior.npy" )
    testXp = tf.one_hot( testXp, nLabels )
    testXp = tfsubset( testXp, group_labels[1:len(group_labels)] )
    testY = np.load( "/raid/data_BA/cit168training/numpy_test/TR_mynbiqpm_Y.npy" )
    testY = tfsubset( tf.one_hot( testY, nLabels ), group_labels )
    testX = tf.concat( [testX, testXp], axis=4 )

for epoch in range(epoch, num_epochs):
    if epoch == 1 or epoch % int(np.round(64/batchsize)) == 0:
        XtrBig = batch_generator( verbose = False )
    inds = random.sample(list(range(XtrBig[0].shape[0])), batchsize)
    print( inds )
    Xtr1 = tfsubsetbatch( [XtrBig[0]], inds )[0]
    Xtr2 = tfsubsetbatch( XtrBig[1], inds )
    with tf.GradientTape(persistent = False) as tape:
          preds = unet_model( Xtr1 )
          mloss = dice_loss( tf.cast(Xtr2[0],'float32'), preds[0] ) * 1.0
          binloss = binary_dice_loss(tf.cast(Xtr2[1],'float32'), tf.squeeze(preds[1]))
          cceloss = tf.reduce_sum( weighted_loss( tf.cast(Xtr2[0],'float32'), preds[0]  ) ) * 1e-4
          loss = mloss + binloss + cceloss
          unet_gradients = tape.gradient(loss, unet_model.trainable_variables)
    optimizerE.apply_gradients(  zip( unet_gradients, unet_model.trainable_variables ) )
    # report per label dice scores
    # for j in range(1,9):
    #    temp = tf.cast(Xtr2[0][:,:,:,:,j],'float32')
    #    print( binary_dice_loss( temp, preds[0][:,:,:,:,j] ) )
    testloss=tf.cast( np.math.inf, 'float32')
    if epoch == 1 or epoch % int(20) == 0:
        preds = unet_model.predict( testX, batch_size=batchsize )
        with tf.device('/CPU:0'):
            predput = ants.from_numpy(preds[0][0,:,:,:,1])
            predcaud = ants.from_numpy(preds[0][0,:,:,:,2])
            truecaud = ants.from_numpy(testY.numpy()[0,:,:,:,2])
            predput[ predput < 0.1 ] = 0
            predcaud[ predcaud < 0.1 ] = 0
            #ants.plot( ants.from_numpy( testX.numpy()[0,:,:,:,0 ]), predput, axis=2 )
            #ants.plot( ants.from_numpy( testX.numpy()[0,:,:,:,0 ]), predcaud, axis=2 )
            #ants.plot( ants.from_numpy( testX.numpy()[0,:,:,:,0 ]), truecaud, axis=2 )
            print("Testing")
            testloss = tf.cast( 0.0, 'float32' )
            for j in range(1,9):
                temp = binary_dice_loss(
                    tf.cast(testY[:,:,:,:,j],'float32'), preds[0][:,:,:,:,j] )
                print( temp )
                testloss = testloss + temp/8.0
    ismin = False
    if epoch > 1:
        if testloss.numpy() <= mydf['test_loss'].min():
            unet_model.save_weights(weights_filename)
            ismin=True
    temp = pd.DataFrame({'train_loss': [loss.numpy()],
            'test_loss': [testloss.numpy()],
            'cce_loss':[cceloss.numpy()],
            'md_loss': [mloss.numpy()],
            'bin_loss': [binloss.numpy()],"epoch": [epoch], "ismin":ismin } )
    if mydf is None:
        mydf = temp
    else:
        mydf = mydf.append(temp, ignore_index = True)
    mydf.to_csv( csv_filename )
    print( temp )

