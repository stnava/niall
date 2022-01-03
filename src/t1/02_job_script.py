import os
from os.path import exists
import glob
rootdir = "/mnt/cluster/data/anatomicalLabels/Mindboggle101_volumes/"
t1fns = glob.glob( "/mnt/cluster/data/anatomicalLabels/Mindboggle101_volumes/*/*/t1weighted.nii.gz" )
import sys
dosr=True
fileindex = 0
if len( sys.argv ) > 1:
    fileindex = int(sys.argv[1])
t1fn = t1fns[ fileindex ]
import re
if dosr:
    mysubbed = re.sub('t1weighted.nii.gz', 'T1wSRHierarchical', t1fn )
else:
    mysubbed = re.sub('t1weighted.nii.gz', 'T1wHierarchical', t1fn )
mysubbedsplit = mysubbed.split("/")
# define the directories and create them
newoutdir = ''
mysep="_"
newprefix = 'Mindboggle'+mysep
keyindex = 9 # change for each case
for k in range(keyindex):
    newoutdir = newoutdir + '/' + mysubbedsplit[k]
    if k > 6:
        newprefix = newprefix + mysubbedsplit[k] + mysep
newprefix = newoutdir + '/' + newprefix
# create the directory
myx = os.path.isdir( newoutdir )
print( "make " +  newoutdir + " " + str( myx ) )
if not myx:
    os.makedirs( newoutdir, exist_ok=True  )

print( "made " +  newoutdir + " successfully " )
if not dosr:
    outfn = newprefix + "hippLR" + '.nii.gz'
else:
    outfn = newprefix + "SR" + "hippLR" + '.nii.gz'

myoutfnexists = exists( outfn )
if myoutfnexists:
    print( outfn + "exists already" )
    sys.exit()

print( "continue " +  outfn + " run " )
import ants
import antspynet
import antspymm
import tensorflow as tf
import tensorflow.keras.backend as K
K.set_floatx("float32")
import antspyt1w
import superiq
t1 = ants.image_read( t1fn )
if dosr:
    srfnout = newprefix + "SR.nii.gz"
    if not exists( srfnout ) :
        print("begin: " + newprefix +  " dosr " + str( dosr ) )
        print("first a bxt ")
        t1bxt = antspyt1w.brain_extraction( t1, method='v0' )
        t1 = t1 * t1bxt
        t1 = antspyt1w.preprocess_intensity( t1, t1bxt )
        tfn = antspyt1w.get_data('T_template0', target_extension='.nii.gz' )
        tlrfn = antspyt1w.get_data('T_template0_LR', target_extension='.nii.gz' )
        templatea = ants.image_read( tfn )
        templatea = ( templatea * antspynet.brain_extraction( templatea, 't1' ) ).iMath( "Normalize" )
        templatealr = ants.image_read( tlrfn )
        t1crop = ants.crop_image( t1, ants.iMath(  t1bxt, "MD", 6 ) )
        t1crop = ants.iMath( t1crop, "TruncateIntensity", 1e-4, 0.999 ).iMath( "Normalize" )
        ants.image_write( t1crop, newprefix + "brain_n4_dnz.nii.gz" )
        newprefix = newprefix + "SR"
        print( "t1crop" )
        print( t1crop )
        mylr = antspyt1w.label_hemispheres( t1crop, templatea, templatealr )
        print("second is SR")
        mdlfn = "/home/ubuntu/models/SEGSR_32_ANINN222_3.h5"
        mdl = tf.keras.models.load_model( mdlfn )
        mysr = superiq.super_resolution_segmentation_per_label(
            t1crop, mylr, [2,2,2], mdl, [1,2], dilation_amount=0, probability_images=None,
            probability_labels=None, max_lab_plus_one=False, verbose=True )
        t1 = mysr['super_resolution']
        t1bxt = ants.resample_image_to_target( t1bxt, t1, interp_type='nearestNeighbor' )
        ants.image_write( t1, newprefix + ".nii.gz" )
        ants.image_write( t1bxt, newprefix + "brain_extraction.nii.gz" )
        sys.exit()
    else:
        newprefix = newprefix + "SR"
        t1=ants.image_read( newprefix + ".nii.gz" )
        t1bxt=ants.image_read( newprefix + "brain_extraction.nii.gz" )
    print("begin hier: " + newprefix )
    t1h = antspyt1w.hierarchical( t1, output_prefix=newprefix, imgbxt=t1bxt, cit168=True )
else:
    print("begin hier: " + newprefix )
    derka
    t1h = antspyt1w.hierarchical( t1, output_prefix=newprefix, cit168=True )

print("complete: " + newprefix )

antspyt1w.write_hierarchical( t1h, output_prefix=newprefix )
