import os
from os.path import exists
import glob
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from random_word import RandomWords
r = RandomWords()

rootdir = "/mnt/cluster/data/"

t1fns = glob.glob( rootdir + "SRPBS_multidisorder_MRI/traveling_subjects/SRPBTravel/sub-*/anat/*_T1w.nii.gz" )
t1fns = t1fns + glob.glob( rootdir + "anatomicalLabels/Mindboggle101_volumes/*volumes/*/t1weighted.nii.gz" )
t1fns = t1fns + glob.glob( rootdir + "PPMI2/PPMI/*/*/T1w/*/dcm2niix/V0/*-dcm2niix-V0.nii.gz" )
t1fns = t1fns + glob.glob( rootdir + "PPMI1/*/*/*/*/*nii.gz" )
import sys
import re
fileindex=137
fileindex=222
myoffset=1000

dosr = True
if len( sys.argv ) > 1:
    fileindex = int(sys.argv[1])
if len( sys.argv ) > 2:
    dosr = eval(sys.argv[2])
if len( sys.argv ) > 3:
    myoffset = int(sys.argv[3])
if (fileindex + myoffset) > len( t1fns):
    sys.exit(0)
t1fn = t1fns[ fileindex + myoffset]
print( "target: " + t1fn + "  " + str(myoffset) + " " + str( fileindex ) )
myrw = str( fileindex )+"_"+str(myoffset)


outfn = "/mnt/cluster/data/T1wJoin/PPMI_RBP/zz" + myrw + "_" + re.sub( ".nii.gz", "_RBP.csv", os.path.basename( t1fn ) )
pngfn=re.sub( "csv", "png", outfn )

outfnb = re.sub( "_RBP", "_RBPb", outfn )
pngfnb=re.sub( "csv", "png", outfnb )

if exists( pngfn ):
    print( outfn + " done " )
    # sys.exit(0)

import ants
import antspynet
import antspymm
import tensorflow as tf
import tensorflow.keras.backend as K
K.set_floatx("float32")
import antspyt1w
import numpy as np

# whole head outlierness
t1 = ants.image_read( t1fn ).iMath("TruncateIntensity",0.05, 0.99).iMath("Normalize")
lomask = ants.get_mask( t1, low_thresh=t1.mean()*0.25 ) #  threshold_image( t1, 0.05, np.math.inf )
t1 = ants.rank_intensity( t1, mask=lomask, get_mask=False ) # t1 = t1 * lomask
ants.plot( t1, axis=2, nslices=21, ncol=7, filename=pngfn, crop=True )
bfn = antspynet.get_antsxnet_data( "S_template3" )
templateb = ants.image_read( bfn ).iMath("Normalize")
templatesmall = ants.resample_image( templateb, (2,2,2), use_voxels=False )
rbp = antspyt1w.random_basis_projection( t1, templatesmall, type_of_transform='Translation',
        refbases=pd.read_csv( "~/.antspyt1w/refbasis_head.csv") )
rbp.to_csv( outfn )

looper=float(rbp['loop_outlier_probability'])
ttl="LOOP: " + "{:0.4f}".format(looper) + " MD: " + "{:0.4f}".format(float(rbp['mhdist']))
img = Image.open( pngfn ).copy()
plt.figure(dpi=300)
plt.imshow(img)
plt.text(20, 0, ttl, color="red", fontsize=12 )
plt.axis("off")
plt.subplots_adjust(0,0,1,1)
plt.savefig( pngfn, bbox_inches='tight',pad_inches = 0)
plt.close()

print(outfn +  " DONE!" )


t1 = ants.image_read( t1fn ).iMath("TruncateIntensity",0.001, 0.999).iMath("Normalize")
lomask = antspynet.brain_extraction( t1, "t1" )
t1 = ants.rank_intensity( t1 * lomask, mask=lomask, get_mask=False ) # t1 = t1 * lomask
ants.plot( t1, axis=2, nslices=21, ncol=7, filename=pngfnb, crop=True )
templatemask = antspynet.brain_extraction( templateb, "t1" )
templateb = ants.rank_intensity( templateb * templatemask, mask=templatemask, get_mask=False )
templatesmall = ants.resample_image( templateb, (2,2,2), use_voxels=False )
rbp = antspyt1w.random_basis_projection( t1, templatesmall, type_of_transform='Rigid',
        refbases=pd.read_csv( "~/.antspyt1w/refbasis_brain.csv") )
rbp.to_csv( outfnb )

looper=float(rbp['loop_outlier_probability'])
ttl="LOOP: " + "{:0.4f}".format(looper) + " MD: " + "{:0.4f}".format(float(rbp['mhdist']))
img = Image.open( pngfnb ).copy()
plt.figure(dpi=300)
plt.imshow(img)
plt.text(20, 0, ttl, color="red", fontsize=12 )
plt.axis("off")
plt.subplots_adjust(0,0,1,1)
plt.savefig( pngfnb, bbox_inches='tight',pad_inches = 0)
plt.close()


print(outfnb +  " DONE!" )
