import antspymm
mdlfn = antspymm.get_data( "brainSR", target_extension=".h5")
import tensorflow as tf
import ants
fn='/mnt/cluster/data/PPMI2/PPMI/40543/20210819/restingStatefMRI/I1499272/dcm2niix/V0/PPMI-40543-20210819-restingStatefMRI-I1499272-dcm2niix-V0.nii.gz'
img=ants.image_read( fn )
mdl = tf.keras.models.load_model( mdlfn )
imgsr=antspymm.super_res_mcimage( img, mdl )
