from dotenv import load_dotenv
load_dotenv()
import numpy as np
import os
from os.path import join
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import os
import argparse
import nibabel as nib

#Goal: get timeseries trial estimates for 2s chunks of continuous video

"""
Each experiment included multiple sessions of 8 min and 24 s long. 
During each session, an 8-min single movie segment was presented; 
before the movie presentation, the first movie frame was displayed
as a static picture for 12 s; after the movie, the last movie 
frame was also displayed as a static picture for 12 s.

Take caution:
1. During the experiments for all the training movie segments and the first testing movie 
segment (i.e. test1.mp4), the fMRI data recording started right after the movie were 
presented for 2 seconds, but ended at the same time as movie ended (see the figure bellow).
For all the other four testing movie segments (test2, test3, test4 and test5), the fMRI 
recording and stimuli presenting started and ended at the same time. Please pay attention 
to the number of volumes when processing the fMRI data.
2. For Subject3, during one fMRI scanning session (i.e. session "test2_9"), the fMRI recording
was around 10s behind the movie "test2.mp4" playing. Therefore, take caution when using 
"test2_9.nii.gz", "test2_9_mni.nii.gz", and "test2_9_Atlas.dtseries.nii"

Ben's comments:
The picture in the README does not match the text description and number of volumes acquired in the
dtseries files. I think fmri scanner began acquisition at 14s after the researcher pressed play (after
the 12s of static first frame and 2s of movie) and ended appropriately, after the 12s of static last frame.

Due to the late start of the fMRI scanner (2s after the movie already started),
we have no data for clips 0s-2s.

Each segment and repetition is analyzed separately.
"""

def main(args):
    subject = f"subject{args.subject}"
    if args.verbose:
        print(f"Running single trial estimates on CC2017 Wen dataset main task for subject {subject}")

    TR = 2 #acquisition TR
    videoChunk = 2 #what snippet length you want to chunk the continuous video into
    stim_length = 480 #in seconds, how long is the stimuli segment. does not account for when fmri acquisition started, just the length of the mp4 file
    dummy_offset = 2 #in seconds. for most runs, the fmri acquisision started 2s after the movie started
    BOLDoffset = 4 #in seconds, what is the offset of the peak BOLD signal

    sub_save_root = os.path.join(args.root, "video_fmri_dataset", "TSTrialEstimates", subject)
    if not os.path.exists(sub_save_root):
        os.makedirs(sub_save_root)

    if args.task == 'train':
        segments = [f"seg{rep}" for rep in range(1,19)]
        numrepeats = 2
    elif args.task == 'test':
        segments = [f"test{rep}" for rep in range(1,6)]
        numrepeats = 10

    for segment in segments:
        for rep in range(1,numrepeats+1):
            stim_order = []
            print(f"{subject} segment {segment} repetition {rep}")
            #overwrite defaults if special task, run, or rep. See the README Take Caution section
            if segment in ['test2', 'test3', 'test4', 'test5']:
                dummy_offset=0 #this overwrites the above dummy_offset default

            #load data
            cifti = nib.load(os.path.join(args.root, "video_fmri_dataset", subject, "fmri", segment, "cifti", f"{segment}_{rep}_Atlas.dtseries.nii"))
            cifti_data = cifti.get_fdata(dtype=np.float32)

            fmri = cifti_data.copy().T

            num_vertices = fmri.shape[0]  # get shape of data for convenience
            if (subject == 'subject3') and (segment == 'test2') and (rep == 9): #in this specific run, the first scans correspond to the static image fixation. check the README
                num_regressors_of_interest = int((stim_length-10)/videoChunk)
            else:
                num_regressors_of_interest = int((stim_length-dummy_offset)/videoChunk)

            estimates = np.zeros((num_regressors_of_interest, num_vertices))

            seg_chunk_order = []
            for r in range(0, num_regressors_of_interest):
                if (subject == 'subject3') and (segment == 'test2') and (rep == 9):
                    estimates[r,:] = fmri[:,int(r*(videoChunk/TR)+BOLDoffset/TR+10/TR)]
                else:
                    estimates[r,:] = fmri[:,int(r*(videoChunk/TR) + BOLDoffset/TR)]
                
                t_onset = int(r*videoChunk) #by definition the time of regressor r onsets at r*videoChunk
                trim_begin = dummy_offset+t_onset #readme says the fmri acquisition started 2s after the 8min movie started, unless an exception
                trim_end = trim_begin + videoChunk
                seg_chunk_order.append(f"{segment}_begin-{trim_begin}_end-{trim_end}")
            stim_order.append(seg_chunk_order)

            outputdir = os.path.join(sub_save_root, f"{args.task}", f"{segment}_{rep}")
            if not os.path.exists(outputdir):
                os.makedirs(outputdir)

            #save design matrix and single trial estimates
            if args.verbose:
                print("saving single trial estimates")
            np.save(join(outputdir, f"{subject}_{segment}_rep-{rep}_stimOrder.npy"), stim_order, allow_pickle=True)
            np.save(join(outputdir, f"{subject}_{segment}_rep-{rep}_estimates.npy"), estimates, allow_pickle=True)

if __name__=='__main__':
    dataset_root_default = os.path.join(os.getenv("DATASETS_ROOT", "/default/path/to/datasets"),"CC2017") #use default if DATASETS_ROOT env variable is not set.
    
    #arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--subject", type=int, required=True, help="The subject from 1-10 that you wish to process")
    parser.add_argument("-c", "--root", type=str, default=dataset_root_default, help="Root path to scratch datasets folder.")
    parser.add_argument("-t", "--task", default='train', required=True, help="test or train task that you are processing")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print verbose")

    args = parser.parse_args()

    main(args)
