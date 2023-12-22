import os
from os.path import exists
import glob

####################################################################################

base_directory = "/mnt/cluster/data/SRPBS_multidisorder_MRI/"
rootdir = base_directory + "traveling_subjects/SRPBTravel/"
t1fns = glob.glob( rootdir + "sub-*" )
t1fns.sort()
import sys
fileindex = 39
if len( sys.argv ) > 1:
    fileindex = int(sys.argv[1])

if fileindex > len( t1fns ):
    sys.exit(0)

t1fn = t1fns[ fileindex ]
import re
newoutdir = base_directory + 'traveling_subjects_antspymm/'
os.makedirs( newoutdir, exist_ok=True  )
os.makedirs( base_directory + 'studycsvs', exist_ok=True  )

subject_id = os.path.basename( t1fn )

print( "RUN " + subject_id + " --- " + newoutdir + " " )
import antspymm

import os
nth='12'
os.environ["TF_NUM_INTEROP_THREADS"] = nth
os.environ["TF_NUM_INTRAOP_THREADS"] = nth
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = nth
import sys
import pandas as pd
import glob
import ants

import antspymm
from antspymm import generate_mm_dataframe


template = ants.image_read("~/.antspymm/PPMI_template0.nii.gz")
bxt = ants.image_read("~/.antspymm/PPMI_template0_brainmask.nii.gz")
template = template * bxt
template = ants.crop_image( template, ants.iMath( bxt, "MD", 12 ) )

anatfn = t1fn + '/anat/' + subject_id + "_T1w.nii.gz"
rsfn = t1fn + '/func/' + subject_id + '_task-rest_run-01_bold.nii.gz'
if not os.path.exists( anatfn ):
    print( anatfn + " does not exist : exiting ")
    sys.exit(0)

# generate_mm_dataframe(projectID, subjectID, date, imageUniqueID, modality, source_image_directory, output_image_directory, t1_filename, flair_filename=[], rsf_filenames=[], dti_filenames=[], nm_filenames=[], perf_filename=[])

studycsv = antspymm.generate_mm_dataframe(
        'SRPBS',
        subject_id,
        'ses-1',
        '000',
        'T1w',
        rootdir,
        newoutdir,
        t1_filename = anatfn,
        rsf_filenames = [rsfn],
    )
studycsv.to_csv(base_directory + "studycsvs/" + subject_id + ".csv")
studycsv2 = studycsv.dropna(axis=1)
mmrun = antspymm.mm_csv(studycsv2,
                        dti_motion_correct='SyN',
                        dti_denoise=True,
                        normalization_template=template,
                        normalization_template_output='ppmi',
                        normalization_template_transform_type='antsRegistrationSyNQuickRepro[s]',
                        normalization_template_spacing=[1,1,1],
                        mysep='_')  # should be this





