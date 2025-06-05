import numpy as np
import scipy
import os
from os.path import join
import time
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import os
import scipy.io as sio
import argparse
import nibabel as nib
from glmsingle.glmsingle import GLM_single

#Goal: get beta estimates for 2s chunks of continuous video
#perform GLM on each segment separately, as provided in the data release, no on the entire run
#as acquired


#compute glm using glmsingle.
# https://github.com/cvnlab/GLMsingle
#helpful nibabel + cifti tutorial: https://nbviewer.org/github/neurohackademy/nh2020-curriculum/blob/master/we-nibabel-markiewicz/NiBabel.ipynb
#other cifti links:
# http://www.nitrc.org/projects/cifti/
# https://www.humanconnectome.org/software/workbench-command/-cifti-help

"""
    Parcels: each index refers to a named subset of the brainordinates (i.e.
        'V1', and the surface vertices in V1)
    Scalars: each index is simply given a name (i.e. 'Myelin')
    Series: each index is assigned a quantity in a linear series (i.e., a
        timeseries of 0 sec, 0.7 sec, 1.4 sec, ...)
    Labels: each index is assigned a name (i.e., 'Visual Areas'), but also a
        list of labels that maps integer data values to names and colors (i.e.
        {(5, 'V1', #ff0000), (7, 'V2', #00ff00), ...}

   The common types of cifti files and the mapping types they use are:
      dconn: ROW is dense, COLUMN is dense
      dscalar: ROW is scalars, COLUMN is dense
      dtseries: ROW is series, COLUMN is dense
      dlabel: ROW is labels, COLUMN is dense
      pconn: ROW is parcels, COLUMN is parcels
      pdconn: ROW is dense, COLUMN is parcels
      dpconn: ROW is parcels, COLUMN is dense
      pscalar: ROW is scalars, COLUMN is parcels
      ptseries: ROW is series, COLUMN is parcels

"""

"""
load data
cifti = nb.load(fname)
cifti_data = cifti.get_fdata(dtype=np.float32)
cifti_hdr = cifti.header
nifti_hdr = cifti.nifti_header

save the data
img = cifti2.Cifti2Image(cifti_data, header=cifti_hdr)
img.to_filename(’/path/test.dtseries.nii’)

"""

def interpolate_ts(fmri, tr_acq, tr_resamp):
    #interopolate the fmri time series. Can be either 2D (surface x time) or 4D (volume x time) array.
    #number of scans (time) has to be the last dimension
    numscans_acq = fmri.shape[-1]
    secsperrun = numscans_acq*tr_acq #time in seconds of the run
    numscans_resamp = int(secsperrun/tr_resamp)

    x = np.linspace(0, numscans_acq, num=numscans_acq, endpoint=True)
    f = scipy.interpolate.interp1d(x, fmri)
    x_new = np.linspace(0, numscans_acq, num=numscans_resamp, endpoint=True)

    fmri_interp = f(x_new)
    return fmri_interp

#arguments
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--subject", type=int, required=True, help="The subject from 1-10 that you wish to process")
parser.add_argument("-c", "--root", default="/data/vision/oliva/scratch/datasets/CC2017/video_fmri_dataset", help="The root path to the cifti files")
parser.add_argument("-t", "--task", default='train', required=True, help="test or train task that you are processing")
parser.add_argument("-v", "--verbose", action="store_true", help="Print verbose")

args = parser.parse_args()
"""
class arguments():
    def __init__(self) -> None:
        self.subject = "3"
        self.root = "/data/vision/oliva/scratch/datasets/CC2017/video_fmri_dataset"
        self.task = "test"
        self.verbose = True
args = arguments()
"""
subject = f"subject{args.subject}"
if args.verbose:
    print(f"Running GLMsingle on CC2017 Wen dataset main task for subject {subject}")

#look at wen's README for some GLM parameters and warnings
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

I'm assuming "10 seconds behind the movie" for subject 3 test2 rep 9 means the fMRI started during second 2 of
the static frame, then 10 seconds later (at t=12) the actual movie began. So this would correspond to a 
dumm_offset of -10. Double check this with reliability results.

Due to the late start of the fMRI scanner (2s after the movie already started),
we have no data for clips 0s-2s.

Each segment and repetition is analyzed separately. Ideally I could do GLM on both reps and use
GLMsingle CD, but combining runs across sessions can get noisy.
"""

TR = 2 #acquisition TR
TR_resamp = 2 # resample time series to be locked to stimulus onset
videoChunk = 2 #what snippet length you want to chunk the continuous video into
stim_length = 480 #in seconds, how long is the stimuli segment. does not account for when fmri acquisition started, just the length of the mp4 file
stimDur = 2 #in scans. Parameter for GLM. could be impulse (0)
dummy_offset = 2 #in seconds. for most runs, the fmri acquisision started 2s after the movie started

sub_save_root = os.path.join(args.root, "GLMsingle", subject)
if not os.path.exists(sub_save_root):
    os.makedirs(sub_save_root)

if args.task == 'train':
    segments = [f"seg{rep}" for rep in range(1,19)]
    numrepeats = 2
elif args.task == 'test':
    segments = [f"test{rep}" for rep in range(1,6)]
    numrepeats = 10

for count, segment in enumerate(segments):
    for rep in range(1,numrepeats+1):
        data = []
        design = []
        stim_order = []
        print(f"{subject} segment {segment} repetition {rep}")
        #overwrite defaults if special task, run, or rep. See the README Take Caution section
        if segment in ['test2', 'test3', 'test4', 'test5']: #246 volumes instead of the usual 245
            dummy_offset=0 #this overwrites the above dummy_offset default

        #load data
        cifti = nib.load(os.path.join(args.root, subject, "fmri", segment, "cifti", f"{segment}_{rep}_Atlas.dtseries.nii"))
        cifti_data = cifti.get_fdata(dtype=np.float32)

        #interpolate time series
        if TR_resamp == TR:
            fmri_interp = cifti_data.copy().T
        else:
            fmri_interp = interpolate_ts(cifti_data.T, TR, TR_resamp)
            
        numscans_interp = fmri_interp.shape[1]
        data.append(fmri_interp)

        #load events
        if (subject == 'subject3') and (segment == 'test2') and (rep == 9): #in this specific run, the first scans correspond to the static image fixation. check the README
            num_regressors_of_interest = int((stim_length-10)/videoChunk)
        else:
            num_regressors_of_interest = int((stim_length-dummy_offset)/videoChunk)
        
        run_design = np.zeros((numscans_interp, num_regressors_of_interest))
        seg_chunk_order = []
        for count, r in enumerate(range(0, num_regressors_of_interest)):
            if (subject == 'subject3') and (segment == 'test2') and (rep == 9):
                run_design[int(r*(videoChunk/TR_resamp)+(10/TR_resamp)), r] = 1 
            else:
                run_design[int(r*(videoChunk/TR_resamp)), r] = 1
            
            t_onset = int(r*videoChunk) #by definition the time of regressor r onsets at r*videoChunk
            trim_begin = dummy_offset+t_onset #readme says the fmri acquisition started 2s after the 8min movie started, unless an exception
            trim_end = trim_begin + videoChunk
            seg_chunk_order.append(f"{segment}_begin-{trim_begin}_end-{trim_end}")
        design.append(run_design)
        stim_order.append(seg_chunk_order)

        #glmsingle params
        opt = dict()
        # set important fields for completeness (but these would be enabled by default)
        opt['wantlibrary'] = 1
        opt['wantglmdenoise'] = 0
        opt['wantfracridge'] = 0

        opt['wantfileoutputs'] = [0,1,0,0]
        opt['wantmemoryoutputs'] = [0,1,0,0]
        outputdir_glmsingle = os.path.join(sub_save_root, f"{args.task}", f"{segment}_{rep}")

        if not os.path.exists(outputdir_glmsingle):
            os.makedirs(outputdir_glmsingle)

        start_time = time.time()
        results_glmsingle = dict()

        if args.verbose:
            print(f"running GLMsingle...")
        #sometimes the default 50,000 chunk length doesn't chunk into equal lengths, throwing an error when converting to array
        numvertices = data[0].shape[0]  # get shape of data for convenience
        numchunks = 1
        while (numvertices % numchunks) != 0:
            numchunks += 1
        opt['chunklen'] = int(numvertices / numchunks)

        glmsingle_obj = GLM_single(opt)
        # run GLMsingle
        results_glmsingle = glmsingle_obj.fit(
            design,
            data,
            stimDur,
            TR_resamp,
            outputdir=outputdir_glmsingle)

        #save design matrix. save to outputdir_glmsingle after fitting glm, otherwise contents will be overwritten
        if args.verbose:
            print("saving design matrix")
        np.save(join(outputdir_glmsingle, f"{subject}_{segment}_rep-{rep}_stimOrder.npy"), stim_order, allow_pickle=True)

        #TODO figure out cifti files
        """
        #additionally save betas as cifti file with cifti header
        betas = np.load(os.path.join(outputdir_glmsingle, "TYPED_FITHRF_GLMDENOISE_RR.npy"), allow_pickle=True).item()
        bm = list(cifti.header.get_index_map(1).brain_models)[0:2] #uses the header information from the last loaded run
        scalar_axis = nib.cifti2.ScalarAxis(['betas_raw'])  # Takes a list of names, one per row
        new_header = nib.Cifti2Header.from_axes([scalar_axis, bm])
        save2cifti(file_path=os.path.join(outputdir_glmsingle, f"{subject}_betas-typed.dscalar.nii") , data=betas['betasmd'].T, brain_models=bm, map_names=ses_conds) #data must be in (maps, values)
        """
