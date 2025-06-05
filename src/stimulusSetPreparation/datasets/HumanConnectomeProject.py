import numpy as np
import os
from torch.utils.data import Dataset
import nibabel as nib

"""
HCP acquisition information: https://www.humanconnectome.org/hcp-protocols-ya-3t-imaging
resting state info: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3720828/
TODO: split resting state runs into smaller chunks
TODO: Add task fMRI runs
TODO: add support for loading stimuli annotations (mainly for task fMRI)
"""

class HCP_rest(Dataset):
    def __init__(self, fmri_root, project_root, phase='train', transforms=None):
        """Initialize the dataset class for the human connectome project. Flexible for both the 
        resting state and task-based fMRI.
        rfMRI TR=0.72 sec. Slice time correction was not applied"""
        self.fmri_root = fmri_root #root path to fmri data
        self.project_root = project_root #root path to the project
        self.stimulus = 0 #np.load(os.path.join(self.root_dir, "mnist_" + phase + ".csv")) #load the stimulus
        #self.subjects = np.loadtxt(os.path.join(self.root_dir, "mnist_" + phase + ".csv"), delimiter=",") #list of subjects
        with open(os.path.join(project_root, "utils", "traintest_splits", "hcp_resting_" + phase + ".txt"), 'r') as f:
            self.data_file = f.read().splitlines()
        self.transforms = transforms

    def __len__(self):
        """
        We tell python to return the length of the datafile as the length of the dataset.
        """
        return len(self.data_file)

    def __getitem__(self, idx):
        """
        Retrieves the fMRI data (entire run) and associated stimuli.
        Parameters
        ----------
        idx : int
            The index into the dataset of the image and label you want to retrieve

        Returns
        -------
        sample: dictionary
            A dictionary with keys "image" and "label" corresponding to transformed images
            and one hot labels.

        """
        fmri = nib.load(os.path.join(self.fmri_root, self.data_file[idx])).get_fdata()
        label = 'rest'
        sample = {'fmri': fmri, 'label': label, 'stimulus': self.stimulus}
        if self.transforms:
            for transform in self.transforms:
                sample = transform(sample)

        return sample
    
class BMD_rest(Dataset):
    def __init__(self, fmri_root, project_root, functional_type='task', phase='train', transforms=None):
        """Initialize the dataset class for BOLD Moments Dataset. Flexible for both the 
        resting state and task-based fMRI.
        rfMRI TR=1.75 sec. Slice time correction was done to beginning of TR"""
        self.fmri_root = fmri_root #root path to fmri data
        self.functional_type = functional_type
        self.project_root = project_root #root path to the project (not the dataset)
        self.stimulus = 0 #np.load(os.path.join(self.root_dir, "mnist_" + phase + ".csv")) #load the stimulus
        #self.subjects = np.loadtxt(os.path.join(self.root_dir, "mnist_" + phase + ".csv"), delimiter=",") #list of subjects
        with open(os.path.join(project_root, "utils", "traintest_splits", "hcp_resting_" + phase + ".txt"), 'r') as f:
            self.data_file = f.read().splitlines()
        self.transforms = transforms

    def __len__(self):
        """
        We tell python to return the length of the datafile as the length of the dataset.
        """
        return len(self.data_file)

    def __getitem__(self, idx):
        """
        Retrieves the fMRI data (entire run) and associated stimuli.
        Parameters
        ----------
        idx : int
            The index into the dataset of the image and label you want to retrieve

        Returns
        -------
        sample: dictionary
            A dictionary with keys "image" and "label" corresponding to transformed images
            and one hot labels.

        """
        fmri = nib.load(os.path.join(self.fmri_root, self.data_file[idx])).get_fdata()
        label = 'rest'
        sample = {'fmri': fmri, 'label': label, 'stimulus': self.stimulus}
        if self.transforms:
            for transform in self.transforms:
                sample = transform(sample)

        return sample