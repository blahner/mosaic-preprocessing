from dotenv import load_dotenv
load_dotenv()
import os
import pandas as pd
"""
Create events.tsv files in BIDS format for NSDsynthetic as they are not immediately available. Files created from 
the "design_nsdsynthetic_runXX.tsv"
"""
def ceildiv(a, b):
    return -(a // -b)

dataset_root = os.path.join(os.getenv('DATASETS_ROOT'), "NaturalScenesDataset")

events_cols = ["onset",	"duration", "stim_idx", "trial_type"]
numruns = 8 #each subject has 8 nsd synthetic runs.
numsubs = 8
tr = 1 #seconds. func1mm design.tsv files assume a 1 second volume. 1.8mm assumes 1.333 second volume
duration = 2 #seconds
for sub in range(1,numsubs+1):
    for run in range(1,numruns+1):
        events_data = {col: [] for col in events_cols} #reset the events_data variable for each run
        #load design tsv file
        design = pd.read_table(os.path.join(dataset_root, "derivatives", "design", f"sub-{sub:02}", f"design_nsdsynthetic_run{run:02}.tsv"), header=None)
        for idx, row in design.iterrows():
            stim = row.values[0]
            if stim !=0: #just enter info for volumes that presented a stimulus
                events_data['onset'].append(tr*idx)
                events_data['duration'].append(duration)
                events_data['stim_idx'].append(stim) #1-indexed
                events_data['trial_type'].append('synthetic')
        #save events.tsv file
        df = pd.DataFrame(events_data)
        if run%2 == 0: #if run is even
            task = 'memory' #or one-back
        else:
            task = 'fixation'
        task_run = ceildiv(run, 2)
        df.to_csv(os.path.join(dataset_root, "Nifti", f"sub-{sub:02}", "ses-nsdsynthetic", "func", f"sub-{sub:02}_ses-nsdsynthetic_task-{task}_run-{task_run:02}_events.tsv"), index=False, sep='\t')
