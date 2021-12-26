import os
from os.path import exists
import glob
rootdir = "/mnt/cluster/data/anatomicalLabels/Mindboggle101_volumes/"
t1fns = glob.glob( "/mnt/cluster/data/anatomicalLabels/Mindboggle101_volumes/*/*/t1weighted.nii.gz" )
import sys
fileindex = 0
if len( sys.argv ) > 1:
    fileindex = int(sys.argv[1])
t1fn = t1fns[ fileindex ]
import re
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
outfn = newprefix + "hippR" + '.nii.gz'
myoutfnexists = exists( outfn )
if myoutfnexists:
    print( outfn + "exists already" )
    sys.exit()

print( "continue " +  outfn + " run " )
import ants
import antspymm
import tensorflow as tf
import antspyt1w
# mdlfn = antspymm.get_data( "brainSR", target_extension=".h5")
# mdl = tf.keras.models.load_model( mdlfn )
t1 = ants.image_read( t1fn )
print("begin: " + newprefix )
t1h = antspyt1w.hierarchical( t1, output_prefix=newprefix, cit168=True )
print("complete: " + newprefix )

# write extant dataframes
for myvar in t1h['dataframes'].keys():
    t1h['dataframes'][myvar].to_csv(newprefix + myvar + ".csv")

(t1h['rbp']).to_csv( newprefix + "rbp.csv" )
myvarlist = [
    'brain_n4_dnz',
    'brain_extraction',
    'wm_tractsL',
    'wm_tractsR',
    'bf',
    'mtl',
    'cit168lab',
    'left_right' ]
for myvar in myvarlist:
    ants.image_write( t1h[myvar], newprefix + myvar + '.nii.gz' )

myvarlist = [
    'tissue_segmentation',
    'dkt_parcellation',
    'dkt_lobes',
    'dkt_cortex',
    'hemisphere_labels' ]
for myvar in myvarlist:
    ants.image_write( t1h['dkt_parc'][myvar], newprefix + myvar + '.nii.gz' )

ants.image_write( t1h['hippLR']['HLBin'], newprefix + "hippL" + '.nii.gz' )
ants.image_write( t1h['hippLR']['HRBin'], newprefix + "hippR" + '.nii.gz' )
