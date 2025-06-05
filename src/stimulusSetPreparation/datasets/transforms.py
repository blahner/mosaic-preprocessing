import os

import numpy as np
import torch
import hcp_utils
import pandas as pd
import scipy

project_root = os.getenv('PROJECT_ROOT')

class NormalizeTimeSeries(object):
    """
    Normalize fMRI data at each voxel across time
    """
    def __init__(self):
        pass

    def __call__(self, sample):
        ts_normalized = hcp_utils.normalize(sample['fmri'])
        return {'fmri': ts_normalized, 'label': sample['label'], 'stimulus': sample['stimulus']}
    
class SelectGlasserROIs(object):
    """
    select the ROIs to use from the glasser atlas
    """
    def __init__(self, root: str=project_root, roi_selection: list[str]=['all'], average: bool=False):
        """
        roi_selection: list of strings. string must correspond to an ROI name in the Glasser Atlas or be 'all' (default)
            to include all ROIs in cortex. left and right hemispheres are combined (TODO: allow user to specify left, right, or both hemispheres)
        average: bool. Whether to average the voxels within the ROI time series or not (default)
        """
        self.root=root #project root
        
        groups = list(np.arange(1,23)) #[1,2,3,4,5,6,7,8,9,15,16,17,18] #groups from the glasser parcellation I want. see table 1 in glasser supplementary for details
        tmp = pd.read_table(os.path.join(self.root, "assets","hcp_glasser_roilist.txt"), sep=',')
        roi_idx_running = {} #get the cifti indices for all rois
        for count in range(tmp.shape[0]):
            line = tmp.iloc[count,:]
            ROI = line['ROI']
            GROUP = line['GROUP']
            ID = line['ID']
            if GROUP in groups: #if the roi is in a group we want, include that roi
                roi_idx_running[ROI] = np.where(((hcp_utils.mmp.map_all == ID)) | (hcp_utils.mmp.map_all == ID+180))[0]
        
        if roi_selection[0] == 'all':
            rois = list(roi_idx_running.keys())
        else:
            rois = roi_selection
            for roi in rois:
                if roi not in roi_idx_running:
                    raise ValueError(f"ROI {roi} you specified is not one of the Glasser Atlas ROIs")
                
        roi_selected_indices = []
        roi_selected_indices.extend([roi_idx_running[roi] for roi in rois])

        self.roi_selection_indices=roi_selected_indices[0] #the rois you want
        self.average=average
    
    def __call__(self, sample):
        #compile all ROIs in the fmri time series
        fmri_rois = sample['fmri'][:,self.roi_selection_indices]
        if self.average:
            fmri_rois = np.mean(fmri_rois, axis=1) #average over the vertices
        return {'fmri': fmri_rois, 'label': sample['label'], 'stimulus': sample['stimulus']}

def interpolate_ts(fmri, tr_acq, tr_resamp):
    """
    interopolate the fmri time series. Can be either 2D (surface x time) or 4D (volume x time) array.
    number of scans (time) has to be the last dimension
    TODO: make time the first dimension to fit with how the .nii files are ordered
    """
    numscans_acq = fmri.shape[-1]
    secsperrun = numscans_acq*tr_acq #time in seconds of the run
    numscans_resamp = int(secsperrun/tr_resamp)

    x = np.linspace(0, numscans_acq, num=numscans_acq, endpoint=True)
    f = scipy.interpolate.interp1d(x, fmri)
    x_new = np.linspace(0, numscans_acq, num=numscans_resamp, endpoint=True)

    fmri_interp = f(x_new)
    return fmri_interp

class NormalizeVideo(object):
    """
    normalize images within a range, mean centered at 0 and 
    standard deviation of 1. Max_val, mean, and std values specific to MNIST.
    """
    def __init__(self, max_val = 255, mean = 0.1307, std = 0.3081):
        """
        Initialize the instance parameters. Specific to MNIST
        """
        self.max_val = max_val
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        """
        Defines how inputs are processed in this class.

        Parameters
        ----------
        sample : dictionary
            contains "image" and "label" keys containing the image and label

        Returns
        ----------
        dictionary
            keys of "image" and "label" of the transformed image and label
        """
        img = sample["image"]
        img = img/self.max_val #fix range between 0 and 1
        img = (img - self.mean)/self.std
        return {"image": img, "label": sample["label"]}

class ToTensor3D(object): #works for 1D arrays
    """"
    Inputs to networks should be tensors for pytorch to keep track of the gradients
    """
    def __init__(self):
        return None
    
    def __call__(self, sample):
        img = sample["image"]
        label = sample["label"]
        img_tensor = torch.Tensor(img)
        label_tensor = torch.Tensor(label)
        return {"image": img_tensor, "label": label_tensor}

class InverseNormalize(object):
    """
    Inverse normalization is useful for visualization. Simply undos
    the normalization procedure.
    """
    def __init__(self, mean = 0.1307, std = 0.3081):
        """
        Initialize the instance parameters. Specific to MNIST
        """
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        """
        Defines how inputs are processed in this class.

        Parameters
        ----------
        sample : dictionary
            contains "image" and "label" keys containing the image and label

        Returns
        ----------
        dictionary
            keys of "image" and "label" of the transformed image and label
        """
        img = sample["image"]
        img = img*self.std + self.mean
        return {"image": img, "label": sample["label"]}