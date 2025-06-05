from dotenv import load_dotenv
load_dotenv()
import os
import numpy as np
import argparse
import pickle
import glob as glob
import pandas as pd

def main(args):
    #betas are zscored across session groups and compiled into a cross-session matrix for easy use
    subject = f"sub-{int(args.subject):02}"
    fmri_path = os.path.join(args.dataset_root,"derivatives", "GLM")
    session_groups = ['perceptionArtificialImage','perceptionLetterImage']
    
    shape_map = pd.read_csv(os.path.join(args.dataset_root, "derivatives", "stimuli_metadata", "images","ArtificialImage.tsv"), delimiter='\t',usecols=[0, 1], names=['filename','stimulus_id'], header=None)
    letter_map = pd.read_csv(os.path.join(args.dataset_root, "derivatives", "stimuli_metadata","images", "LetterImage.tsv"), delimiter='\t',usecols=[0, 1], names=['filename','stimulus_id'], header=None)

    #divide the stimuli into test and train
    tmp = pd.read_table(os.path.join(args.dataset_root, "derivatives", "stimuli_metadata", "deeprecon_stiminfo.tsv"),index_col=False)
    artificial_stimuli = tmp[(tmp['source'] == 'Artificial') & (tmp[f"{subject}_reps"] > 0)]['filename'].tolist()

    print(f"Length of artificial stim: {len(artificial_stimuli)}")
    assert(len(artificial_stimuli) == 50)

    betas_tmp_artificial = {stim: [] for stim in artificial_stimuli}
    
    #load fMRI data from that subject and run
    for sg in session_groups:
        session_group = f"sessiongroup-{sg}"
        print(f"loading sessiongroup {session_group}")
        fmri_data_wb = np.load(os.path.join(fmri_path, subject, session_group, "TYPED_FITHRF_GLMDENOISE_RR.npy"), allow_pickle=True).item()
        fmri_data_wb = fmri_data_wb['betasmd'].squeeze() #squeezed to shape numvertices x numtrials
        fmri_data_wb = fmri_data_wb.T #transpose to shape numtrials x numvertices, more representative of the samples x features format    
        
        with open(os.path.join(fmri_path, subject, session_group, f"{subject}_{session_group}_task-perception_conditionOrderDM.pkl"), 'rb') as f:
            events_run, _ = pickle.load(f)
        ses_conds = []
        for event in events_run:
            for stim_id in event['trial_type']:
                if 'perceptionArtificialImage' in session_group:
                    stim = f"{shape_map.loc[shape_map['stimulus_id'] == stim_id, 'filename'].to_list()[0]}.tiff"
                elif 'perceptionLetterImage' in session_group:
                    stim = f"{letter_map.loc[letter_map['stimulus_id'] == stim_id, 'filename'].to_list()[0]}.tif"
                else:
                    raise ValueError(f"invalid stimulus ID {stim_id}")
                ses_conds.append(stim)
                
        session_mean = np.mean(fmri_data_wb, axis=0)
        session_std = np.std(fmri_data_wb, axis=0, ddof=1)
        fmri_data_wb_normalized = (fmri_data_wb - session_mean) / session_std
        if sg in ['perceptionArtificialImage','perceptionLetterImage']:
            #all test images
            for count, stim in enumerate(ses_conds):
                betas_tmp_artificial[stim].append(fmri_data_wb_normalized[count,:])

    numvertices = 91282
    numreps_artificial = 20 #repeats are 20 and 12 shapes and letters, respectively

    betas_artificial = np.zeros((len(betas_tmp_artificial), numreps_artificial, numvertices))
    betas_artificial.fill(np.nan)
    stimorder_artificial = []
    for stimcount, b in enumerate(betas_tmp_artificial.keys()):
        value = betas_tmp_artificial[b]
        stimorder_artificial.append(b)
        for repcount, v in enumerate(value): #loop over reps
            betas_artificial[stimcount, repcount, :] = np.array(v)

    #save betas
    save_root = os.path.join(fmri_path, f"{subject}", "prepared_betas")
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    print(f"saving {subject} artificial betas")
    with open(os.path.join(save_root, f"{subject}_organized_betas_task-artificial_normalized.pkl"), 'wb') as f:
        pickle.dump((betas_artificial,stimorder_artificial), f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    dataset_root_default = os.path.join(os.getenv("DATASETS_ROOT", "/default/path/to/datasets"),"deeprecon") #use default if DATASETS_ROOT env variable is not set.

    #arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--subject", type=int, required=True, help="The subject from 1-3 that you wish to process")
    parser.add_argument("-d", "--dataset_root", default=dataset_root_default, help="The root path to the dataset directory. Specifying path here overwrites environment variable.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print verbose")
    args = parser.parse_args()

    main(args)