from dotenv import load_dotenv
load_dotenv()
import os
import numpy as np
from pathlib import Path
from PIL import Image
import random
import glob
import re
from torchvision.transforms import v2
from collections import defaultdict
import copy
import h5py
import threading
from typing import Optional
import pandas as pd
import random
from tqdm import tqdm

#local
from src.utils.transforms import SelectROIs

dataset_root = os.path.join(os.getenv("DATASETS_ROOT", "/default/path/to/datasets"), "MOSAIC") #use default if DATASETS_ROOT env variable is not set.

class FMRIDataset(object):
    def __init__(self,
                 data: Optional[list[dict]] | None,
                 ROI_selection: Optional[SelectROIs] | None,
                 use_noiseceiling: Optional[bool] | None, 
                 trial_selection: Optional[str] = 'average', 
                 stimuli_phase: Optional[dict] = None,
                 subjectID_mapping: dict=None, 
                 img_transforms: v2=None, 
                 fmri_transforms: v2=None, 
                 subjectID_transforms: v2=None,
                 angleID_transforms: v2=None,
                 ):
        """
        Initialize the dataset instance. The 
        dataset class retrieves and transforms images and labels. 
        It will be input to a dataloader class that will more specifically
        define batch sizes, shuffling etc. 
        INPUTS:
        data: list of dictionaries that map the stimulus filenames to the brain response filenames. Loaded from a 'train.json' or 'test.json' file.
        ROI_selection: SelectROIs class with information on with rois/vertices to process.
        use_noiseceiling: bool, if true, use the noiseceiling for the subjectID (and apply the appropriate transforms to it) to scale the loss. If false,
            don't use the noiseceiling and just keep the stimulus and fmri label.
        trial_selection: str, must be one of 'first', 'random', or 'average'. For a given stimulus, we identify the list of possible single trial
            brain repsonses. 'first' means the selected brain response label is simply the first single trial response in the list. 
            'random' means you will randomly (uniform) select a single trial response from the list. 'average' means you will randomly (uniform)
            select a subjectID from the list of possible single trial brain responses (a given stimulus can have brain respnoses from mulitple subjects
            and/or datasets), and average the single trial reponses over that subjectID. trial_selection also affects the noiseceiling used 
            (if specified) - 'first' and 'random' use single trial respnoses so the noiseceiling for that subject is n=1. 'average' uses an average of 
            single trial response so the noiseceiling is n=X, where X is the number of averaged responses.
        img_transforms: v2.Compose(), collection of transforms to apply to the image stimulus, such as resizing and cropping and mapping to tensor. Be
            careful of transforms that may significantly alter the associated brain response, such flipping, rotating, recolor etc.
        fmri_transforms: v2.Compose(), collection of transfroms to apply to the fmri and noiseceiling (if specified) labels, such as ROI selection, 
            filing nans, and mapping to tensor.
        subjectID_transforms: v2.Compose(), collection of transfroms to apply to the subjectID, such as converting to torch tensor
        """
        assert trial_selection in ['average', 'random', 'first', 'all'], f"trial selection {trial_selection} not valid. Must be one of ['average', 'random', 'first', 'all']"
        self.root_dir = dataset_root
        self.data = data
        self.ROI_selection = ROI_selection
        self.subjectID_mapping = subjectID_mapping
        self.use_noiseceiling = use_noiseceiling
        self.img_transforms = img_transforms
        self.fmri_transforms = fmri_transforms
        self.subjectID_transforms = subjectID_transforms
        self.angleID_transforms = angleID_transforms
        self.trial_selection = trial_selection
        self.stimuli_phase = stimuli_phase
        if self.data:
            self.subject_indices = self._get_subject_specific_indices(self.data)
        self.local = threading.local()
        self.hdf5_file_path = os.path.join(dataset_root,"mosaic_version-1_0_0_renamed.hdf5")
        self.hdf5_chunks_file_path = os.path.join(dataset_root,"mosaic_version-1_0_0_chunks_renamed.hdf5")
        self.selected_roi_indices = ROI_selection.selected_roi_indices
        self.subject_noiseceilings = {}
        #if self.stimuli_phase:
        self.subject_noiseceilings = self._get_all_noiseceilings()
        self.NSDsynthetic_stimulus_filenames = set(pd.read_csv(os.path.join(dataset_root, "stimuli", "datasets_stiminfo", "nsd_artificial_list.tsv"), sep="\t")['filename'].tolist())

    def _get_file_handle(self) -> h5py.File:
        """Get thread-local file handle, creating it if necessary."""
        if not hasattr(self.local, 'handle1'):
            self.local.handle1 = h5py.File(self.hdf5_file_path, 'r', swmr=True)
        return self.local.handle1

    def _get_file_handle_chunks(self) -> h5py.File:
        """Get thread-local file handle, creating it if necessary."""
        if not hasattr(self.local, 'handle2'):
            self.local.handle2 = h5py.File(self.hdf5_chunks_file_path, 'r', swmr=True)
        return self.local.handle2

    def close_handle(self):
        """Close the thread-local file handle if it exists."""
        if hasattr(self.local, 'handle1'):
            self.local.handle1.close()
            delattr(self.local, 'handle1')
        
        if hasattr(self.local, 'handle2'):
            self.local.handle2.close()
            delattr(self.local, 'handle2')
    
    def _get_subject_specific_indices(self, data):
        subject_indices = defaultdict(list) #maps a subjectID to a list of all indices in the dataset
        for idx, item_dict in enumerate(data):
            subjectID = list(item_dict.keys())[0].split('_stimulus-')[0]
            subject_indices[subjectID].append(idx)
        return subject_indices

    def __getitem__(self, idx):
        """
        Method to tell the dataset how to retrieve an image and label.
        This method mimics torch's "__getitem__" method that is usually inherited when you
        use the pytorch framework. Each index (idx) into the dataset is a key-value pair where
        the key contains subject and stimlus filename info in the format "sub-XX_DATASET_stimulus-STIMULUSFILENAME"
        and the value is a list of corresponding brain response filenames. Note that if multiple subjects saw 
        the same stimulus with the same STIMULUSFILENAME, there will be multiple entries in the dataset
        differentiated by the "sub-XX_DATASET" info. In other words, a sample of the training or testing set is 
        defined by the subject+stimulus, not just the stimulus. Thus in one epoch of training, the same stimulus
        may be input mulitple times but they will be paired with a subject-specific fmri label.
        Parameters
        ----------
        idx : int
            The index into the dataset of the image and label you want to retrieve

        Returns
        -------
        sample: dictionary
            A dictionary with keys "stimulus" and "fmri" corresponding to transformed stimulus
            and its associated fmri response.

        """

        try:
            hdf5 = self._get_file_handle()
        except (OSError, IOError) as e:
            # If file handle is invalid, clear it and retry once
            if hasattr(self.local, 'handle'):
                del self.local.handle
            hdf5 = self._get_file_handle()

        #load stimulus
        filename_key = list(self.data[idx].keys())[0]
        subjectID, stim_filename = filename_key.split('_stimulus-')
        loaded_filename, stim = self.load_stimulus(stim_filename)

        #get the filenames of possible brain response corresponding to the stimulus
        possible_responses = self.data[idx][filename_key]

        #select and load from from the list of possible brain responses
        label = self.load_fMRI_response_hdf5(possible_responses, self.trial_selection, subjectID, hdf5)

        if self.use_noiseceiling:
            for key_phase, stimset in self.stimuli_phase.items():
                if stim_filename in stimset:
                    phase = key_phase
                    dataset = subjectID.split('_')[-1] #TODO better handle this train/test noiseceiling issue
                    if dataset in ['GOD', 'BOLD5000', 'THINGS'] and phase == 'train':
                        phase='test' #these datasets do not have a noisceiling for the train set so use their test set.
                    break
            if self.trial_selection == 'average': #fmri transforms (if applicable) are already applied to noiseceilings here
                noiseceiling = self.subject_noiseceilings[subjectID][f"{subjectID}_phase-{phase}_n-avg"]
            else:
                noiseceiling = self.subject_noiseceilings[subjectID][f"{subjectID}_phase-{phase}_n-1"]
        else:
            noiseceiling = []

        if self.subjectID_mapping:
            if subjectID in self.subjectID_mapping:
                subjectID_value = self.subjectID_mapping[subjectID]
            else:
                subjectID_value = -1 #out of training set subject
        else:
            subjectID_value = -1 #no mapping defined
        #this inputs the viewing angle to the sample no matter what
        viewing_angle = hdf5[subjectID].attrs['visual_angle']/360 #self.viewing_angle_mapping[datasetID]/360 #normalize the viewing angle by dividing by 360

        #create the sample
        sample = {"stimulus": stim,
                  "fmri": label,
                  "subjectID": subjectID_value,
                  "subjectID_str": subjectID,
                  "viewing_angle": viewing_angle,
                  "noiseceiling": noiseceiling,
                  "stimulus_filename": loaded_filename}
        
        #apply transforms
        if self.img_transforms:
            sample['stimulus'] = self.img_transforms(sample['stimulus'])
        if self.fmri_transforms:
            sample['fmri'] = self.fmri_transforms(sample['fmri'])
        if self.subjectID_transforms:
            sample['subjectID'] = self.subjectID_transforms(sample['subjectID'])
        if self.angleID_transforms:
            sample['viewing_angle'] = self.angleID_transforms(sample['viewing_angle'])
        return sample
                        
    def __len__(self):
        """
        We tell python to return the length of the datafile (number of stimuli) as the length of the dataset.
        """
        return len(self.data)

    def _get_noiseceiling(self,subjectID):
        """
        Given a subjectID, returns a dictionary of that subjectID's noiseceilings
        """
        try:
            hdf5 = self._get_file_handle_chunks()
        except (OSError, IOError) as e:
            # If file handle is invalid, clear it and retry once
            if hasattr(self.local, 'handle'):
                del self.local.handle
            hdf5 = self._get_file_handle_chunks()
        noiseceilings = {}
        for nc_key in hdf5[subjectID]['noiseceilings'].keys():
            phase = hdf5[subjectID]['noiseceilings'][nc_key].attrs['phase']
            n = hdf5[subjectID]['noiseceilings'][nc_key].attrs['n']
            nc_array = hdf5[subjectID]['noiseceilings'][nc_key][self.selected_roi_indices]/100
            if self.fmri_transforms:
                nc_array = self.fmri_transforms(nc_array)
            noiseceilings[f"{subjectID}_phase-{phase}_n-{n}"] = nc_array

        self.close_handle()
        return noiseceilings

    def _get_all_noiseceilings(self):
        """
        Return a dictionary mapping all subjectIDs in the hdf5 file to their noiseceiling vectors, if it exists
        """
        try:
            hdf5 = self._get_file_handle_chunks()
        except (OSError, IOError) as e:
            # If file handle is invalid, clear it and retry once
            if hasattr(self.local, 'handle'):
                del self.local.handle
            hdf5 = self._get_file_handle_chunks()
        noiseceilings = {}
        for subjectID in hdf5.keys():
            if 'noiseceilings' not in hdf5[subjectID].keys():
                continue #some subjects like HAD and NOD 10-30 do not have noiseceilings
            noiseceilings[subjectID] = {}
            for nc_key in hdf5[subjectID]['noiseceilings'].keys():
                phase = hdf5[subjectID]['noiseceilings'][nc_key].attrs['phase']
                n = hdf5[subjectID]['noiseceilings'][nc_key].attrs['n']
                nc_array = hdf5[subjectID]['noiseceilings'][nc_key][self.selected_roi_indices]/100
                if self.fmri_transforms:
                    nc_array = self.fmri_transforms(nc_array)
                noiseceilings[subjectID][f"{subjectID}_phase-{phase}_n-{n}"] = nc_array

        return noiseceilings
    
    def load_stimulus(self, stimulus_filename):
        """
        Given a stimulus filename this method loads it. Handles loading of video frames and images.
        INPUT:
        stimulus_filename: str, should be a valide stimulus filename including extension.
        RETURNS:
        loaded_filename: str, the filename of the stimulus that was actually loaded. In the case of videos,
            the middle frame is loaded hence why this variable is returned.
        stim: Image object, the Image object of the loaded image or frame.
        """
        if Path(stimulus_filename).suffix in ['.mp4']:
            stim_filename_middleframe = glob.glob(os.path.join(self.root_dir, "stimuli", "frames_middle", f"{Path(stimulus_filename).stem}*.jpg"))
            assert(len(stim_filename_middleframe) == 1)
            #converting to RGB usually not necessary but some images are CMYK, so the extra dimension messes up the subsequent transforms.
            with Image.open(stim_filename_middleframe[0]).convert("RGB") as img:
                stim = img.copy()  #get middle frame for videos
            loaded_filename = Path(stim_filename_middleframe[0]).name
        elif stimulus_filename in self.NSDsynthetic_stimulus_filenames:
            with Image.open(os.path.join(self.root_dir, "stimuli", "raw", stimulus_filename)) as img:
                img_array = np.array(img)
                img_array = (np.sqrt(img_array/255)*255).astype(np.uint8)
                img = Image.fromarray(img_array).convert('RGB')
                stim = img.copy()
            loaded_filename = stimulus_filename

        else:
            #converting to RGB usually not necessary but some images are CMYK, so the extra dimension messes up the subsequent transforms.
            with Image.open(os.path.join(self.root_dir, "stimuli", "raw", stimulus_filename)).convert("RGB") as img:
                stim = img.copy() 
            loaded_filename = stimulus_filename

        return loaded_filename, stim
    
    def load_fMRI_response_hdf5(self, possible_responses, trial_selection, subjectID, hdf5):
        """
        This method loads the desired fMRI response from a given stimulus based on 'trial_selection' argument.
        INPUT:
        possible_responses: list of str, list of filenames that correspond to a single trial fMRI response to the INPUT stimulus_filename
        trial_selection: str, must be one of 'random', 'first', 'average'. 'random' selects one response from the list of 'possible_responses' with 
        uniform probability. 'first' just chooses the first response in the list of 'possible_responses'. 'average' selects one subject (if more than one)
        of subjectIDs with uniform probability and averages the fMRI responses of that subject.
        RETURNS:
        label: numpy array, the whole brain fMRI response of shape (91282,) according to the desired 'trial_selection' procedure of the stimulus responses
        included in 'possible_responses'.
        """

        if trial_selection == 'random':
            random_idx = np.random.randint(0,high=len(possible_responses), size=1)[0] #[low,high) range
            label_filename = possible_responses[random_idx]
            label = hdf5[subjectID]['betas'][Path(label_filename).stem][self.selected_roi_indices]
        elif trial_selection == 'first':
            label_filename = possible_responses[0]
            label = hdf5[subjectID]['betas'][Path(label_filename).stem][self.selected_roi_indices]
        elif trial_selection == 'average':
            label = np.zeros((len(self.selected_roi_indices),))
            count = 0
            for label_filename in possible_responses:
                label += hdf5[subjectID]['betas'][Path(label_filename).stem][self.selected_roi_indices]
                count+=1
            label = label/count #average over the responses from that subject

        return label

    def load_responses_block_hdf5(self, subjectID, load_stimulus: bool=False, verbose: bool=False):
        """
        loads a block of beta values for a given sub-XX_DATASET. Much faster if you know you need to access all data for a subject
        """
        try:
            hdf5 = self._get_file_handle_chunks()
        except (OSError, IOError) as e:
            # If file handle is invalid, clear it and retry once
            if hasattr(self.local, 'handle'):
                del self.local.handle
            hdf5 = self._get_file_handle_chunks()

        print("loading betas...")
        betas_tmp = hdf5[subjectID]['betas'][:,self.ROI_selection.selected_roi_indices]
        stimulus_order_bytes = hdf5[subjectID]['presented_stimulus_filename'][:]
        stimulus_order_tmp = [s.decode('utf-8') for s in stimulus_order_bytes]
        #stimulus_order_noext_tmp = ['.'.join(s.split('.')[:-1]) for s in stimulus_order_tmp] #remove filename extension
        viewing_angle = hdf5[subjectID].attrs['visual_angle'] / 360

        if self.angleID_transforms:
            viewing_angle = self.angleID_transforms(viewing_angle)
        if self.subjectID_mapping:
            subjectID_value = self.subjectID_mapping[subjectID]
        else:
            subjectID_value = subjectID 
        if self.subjectID_transforms:
            subjectID_value = self.subjectID_transforms(subjectID_value)

        """
        if self.use_noiseceiling:
            for key_phase, stimset in self.stimuli_phase.items():
                if stim_filename in stimset:
                    phase = key_phase
                    break
            if self.trial_selection == 'average': #fmri transforms (if applicable) are already applied to noiseceilings here
                noiseceiling = self.subject_noiseceilings[subjectID][f"{subjectID}_phase-{phase}_n-max"]
            else:
                noiseceiling = self.subject_noiseceilings[subjectID][f"{subjectID}_phase-{phase}_n-1"]
        else:
            noiseceiling = []
        """

        if self.use_noiseceiling:
            noiseceiling = self.subject_noiseceilings[subjectID] #applies fmri transforms, if applicable, inside this method
        else:
            noiseceiling = [] #some subjects, like HAD sub-01 to sub-30 and NOD sub-10 to sub-30, have no noiseceiling due to no repeated stimuli

        if self.trial_selection == 'all':
            betas = betas_tmp
            stimulus_order = ['.'.join(s.split('.')[:-1]) for s in stimulus_order_tmp] #stimulus_order_noext_tmp #['.'.join(s.split('.')[:-1]) for s in stimulus_order_tmp] #remove filename extension
            stimulus_order_ext = stimulus_order_tmp
        else:
            betas = []
            stimulus_order = []
            stimulus_order_ext = []
            stimuli_unique = list(dict.fromkeys(stimulus_order_tmp)) #list(dict.fromkeys(stimulus_order_noext_tmp)) #remove duplicates while preserving order
            iterator = tqdm(stimuli_unique, total=len(stimuli_unique), desc=f"loading betas for subject {subjectID}") if verbose else stimuli_unique
            for stim in iterator:
                if self.trial_selection == 'first':
                    indices = stimuli_unique.index(stim)
                    fmri = betas_tmp[indices,:]
                elif self.trial_selection == 'random':
                    indices = [i for i, x in enumerate(stimulus_order_tmp) if x == stim] #get the indices into the betas fmri block that match the stimulus
                    fmri = np.mean(betas_tmp[random.choice(indices),:],axis=0)
                elif self.trial_selection == 'average':
                    indices = [i for i, x in enumerate(stimulus_order_tmp) if x == stim] #get the indices into the betas fmri block that match the stimulus
                    fmri = np.mean(betas_tmp[indices,:],axis=0)
                
                if self.fmri_transforms:
                    fmri = self.fmri_transforms(fmri)
                betas.append(fmri)
                stimulus_order.append('.'.join(stim.split('.')[:-1])) #remove filename extension  #stimulus_order.append(stim)
                stimulus_order_ext.append(stim) #useful for loading the filenames
            betas = np.vstack(betas)

        if load_stimulus:
            stimuli_arrays = []
            iterator = tqdm(stimulus_order_ext, total=len(stimulus_order_ext), desc=f"loading stimuli for subject {subjectID}") if verbose else stimulus_order_ext
            for stim_filename in iterator:
                _, stim = self.load_stimulus(stim_filename) 
                if self.img_transforms: #apply img transforms
                    stimuli_arrays.append(self.img_transforms(stim))  
                else:
                    stimuli_arrays.append(stim) 
        else:
            stimuli_arrays = [] 

        sample = {"stimulus": stimuli_arrays,
                  "fmri": betas,
                  "subjectID": [subjectID_value] * betas.shape[0],
                  "subjectID_str": [f"{subjectID}"] * betas.shape[0],
                  "viewing_angle": viewing_angle,
                  "noiseceiling": noiseceiling, #dict
                  "stimulus_filename": stimulus_order}

        #apply fmri and other transforms
        if self.fmri_transforms:
            sample['fmri'] = self.fmri_transforms(sample['fmri']) #noiseceilings already have tsfms applied
        if self.subjectID_transforms:
            sample['subjectID'] = self.subjectID_transforms(sample['subjectID'])
        if self.angleID_transforms:
            sample['viewing_angle'] = self.angleID_transforms(sample['viewing_angle'])

        return sample

    def get_indices_by_subjectID(self):
        """
        Returns indices for each subjectID available. They can be used to make a custome batch sampler
        that batches the data by subjectID
        """
        subject_idx = {sub: [] for sub in self.subject_list}
        for idx, stimulus_dict in enumerate(self.data):
            for _, dataset_dict in stimulus_dict.items():
                for _, subject_dict in dataset_dict.items():
                    for subjectID in subject_dict.keys():
                        subject_idx[subjectID].append(idx) #the same index can be assigned to multiple subjectsIDs if multiple subjects saw that stimulus
        return subject_idx


class SubjectSampler():
    def __init__(self, subject_indices, batch_size: int=16, subject_ordering: str='random', confidence_scores: Optional[dict] | None = None, shuffle=False):
        """
        dl = DataLoader(ds, batch_sampler=sampler(ds.classes()))
        stack overflow: https://stackoverflow.com/questions/66065272/customizing-the-batch-with-specific-elements
        """
        assert subject_ordering in ['random', 'ascending', 'descending'], f"invalid subject_ordering argument. You input {subject_ordering}, must be one of ['random', 'ascending', 'descending']"
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_duplicate_map = {}
        self.subject_ordering = subject_ordering
        if confidence_scores and self.subject_ordering != 'random':
            assert confidence_scores.keys() == subject_indices.keys(), f"confidence scores keys {confidence_scores.keys()} are different than the provided subject_indices keys {subject_indices.keys()}"
            if self.subject_ordering == 'ascending':
                reverse = False
            elif self.subject_ordering == 'descending':
                reverse = True #high confidence to low confidence order
            else:
                raise ValueError(f"self.subject_ordering {self.subject_ordering} not recognized. Must be one of 'random', 'ascending', 'descending'")
            self.confidence_scores_sorted = {k: v for k, v in sorted(confidence_scores.items(), key=lambda item: item[1], reverse=reverse)} 
            self.subject_indices = {subjectID: [] for subjectID in self.confidence_scores_sorted.keys()}
            for subjectID, indices in subject_indices.items(): #'subject_indices' input is a random order of subjectID: [indices] mapping of the data samples
                self.subject_indices[subjectID] = indices
        else:
            self.confidence_scores_sorted = None
            self.subject_indices = subject_indices

    def get_duplicate_mask(self, batch_idx):
        """
        Returns boolean mask indicating which samples in the batch are duplicates.
        
        Args:
            batch_idx: Index of the batch
            
        Returns:
            List[bool]: Mask where True indicates a duplicate sample
        """
        return self.batch_duplicate_map.get(batch_idx, [])

    def __iter__(self):
        """
        Returns an iterable list of batches identifying indices in the dataset. These indices are grouped
        in batches based on the same subject.
        """
        batches = []
        self.batch_duplicate_map = {}  #reset tracking for new iteration
        batch_idx = 0

        for indices in self.subject_indices.values(): #loop over each subject and their indices
            num_samples = len(indices)
            num_full_batches = num_samples // self.batch_size
            remainder = num_samples % self.batch_size
            
            #fill in full batches
            for i in range(num_full_batches):
                start = i * self.batch_size
                batch_indices = indices[start:start + self.batch_size]
                if self.shuffle:
                    #shuffle order of indices within a subject
                    random.shuffle(batch_indices)
                self.batch_duplicate_map[batch_idx] = [False] * self.batch_size
                batches.append(batch_indices)
                batch_idx += 1
                
            #handle remainder with repetition. That that this may bias some samples
            if remainder > 0:
                batch_indices = indices[-remainder:]  #get remaining samples
                #this while loop handles the (hopefully) rare case where a subject does not have enough samples to fill even one batch
                duplicate_mask = [False] * remainder
                while len(batch_indices) < self.batch_size:
                    batch_indices.append(random.choice(indices))
                    duplicate_mask.append(True)
                if self.shuffle:
                    #shuffle order of indices within a subject
                    combined = list(zip(batch_indices, duplicate_mask))
                    random.shuffle(combined)
                    batch_indices, duplicate_mask = zip(*combined)
                    batch_indices = list(batch_indices)
                    duplicate_mask = list(duplicate_mask)

                self.batch_duplicate_map[batch_idx] = duplicate_mask
                batches.append(batch_indices)
                batch_idx += 1
        if self.subject_ordering == 'random' and self.shuffle:
            #now shuffle the order the subjects may appear in
            combined = list(enumerate(batches))
            random.shuffle(combined)
            
            # Reorder batch_duplicate_map to match shuffled batches
            new_duplicate_map = {}
            new_batches = []
            for new_idx, (old_idx, batch) in enumerate(combined):
                new_duplicate_map[new_idx] = self.batch_duplicate_map[old_idx]
                new_batches.append(batch)
            
            self.batch_duplicate_map = new_duplicate_map
            batches = new_batches

        return iter(batches)

    def __len__(self):
        total_batches = 0
        for indices in self.subject_indices.values():
            total_batches += -(-len(indices) // self.batch_size) 
        return total_batches
    
class StimulusDataset(FMRIDataset):
    def __init__(self,
                 data: Optional[list[dict]] | None,
                 img_transforms: v2=None, 
                 ):
        """
        Initialize the dataset instance. The 
        dataset class retrieves and transforms images. 
        It will be input to a dataloader class that will more specifically
        define batch sizes, shuffling etc. 
        INPUTS:
        data: 
        img_transforms: v2.Compose(), collection of transforms to apply to the image stimulus, such as resizing and cropping and mapping to tensor. Be
            careful of transforms that may significantly alter the associated brain response, such flipping, rotating, recolor etc.
        """
        self.root_dir = dataset_root
        self.data = data
        self.img_transforms = img_transforms
        self.NSDsynthetic_stimulus_filenames = set(pd.read_csv(os.path.join(dataset_root, "stimuli", "datasets_stiminfo", "nsd_artificial_list.tsv"), sep="\t")['filename'].tolist())

    def __getitem__(self, idx):
        """
        Method to tell the dataset how to retrieve an image.
        Parameters
        ----------
        idx : int
            The index into the dataset of the image and label you want to retrieve

        Returns
        -------
        sample: dictionary
            A dictionary with keys "stimulus" corresponding to transformed stimulus
        """

        #load stimulus
        loaded_filename, stim = self.load_stimulus(self.data[idx])

        #create the sample
        sample = {"stimulus": stim,
                  "stimulus_filename": loaded_filename}
        
        #apply transforms
        if self.img_transforms:
            sample['stimulus'] = self.img_transforms(sample['stimulus'])
        return sample
                        
    def __len__(self):
        """
        We tell python to return the length of the datafile (number of stimuli) as the length of the dataset.
        """
        return len(self.data)

