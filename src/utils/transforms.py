from dotenv import load_dotenv
load_dotenv()
import numpy as np
import pandas as pd
import os
import torch
import hcp_utils as hcp

"""
Define custom transforms here
"""
project_root = os.path.join(os.getenv("PROJECT_ROOT", "/default/path/to/project"))
dataset_root = os.path.join(os.getenv("DATASETS_ROOT", "/default/path/to/datasets"), "MOSAIC")

class SelectROIs:
    def __init__(self, selected_rois: list[str], remove_nan_vertices: bool=True):
        """
        Specify which rois you want to include in your analysis. Can either specify the exact roi or glasser group. 
        Stores the roi indices to index into the whole brain betas to extract the brain activity of the desired ROIs.
        This class is additionally initialized with dictionaries to map between all ROIs and their indices and groups.
        This class just focuses on the 360 (180 each hemisphere) cortical ROIs, not subcortical. This class relies heavily 
        on hcp_utils package, available here: https://rmldj.github.io/hcp-utils/
        Inputs:
        selected_rois: list, which rois in the glasser atlas you want to include. If an ROI is "GlassGroup_X" were X is between 1-22 inclusive,
            it returns all ROIs within that group. See table 1 in glasser supplementary for details. If None, we use all 180 ROIs. Specifying
            an ROI with "L_" or "R_" before it will include only that ROI's left or right hemisphere, respectively. Specifying the ROI without
            the "L_" or "R_" will include both left and right hemispheres.
        Returns:
        None

        Example:
        roi_transform = SelectROIs(rois=['GlasserGroup_1', 'GlasserGroup_2', 'PPA2']) #uses the indices of all rois in glasser groups
          1 and 2 and both hemispheres of the roi PPA2. It removes vertices that from this group that have been NaN somewhere in the dataset
        """
        core_rois_all = self._initialize_roi_mappings() #initalizes the six mappings of self.index_to_roi, index_to_group, roi_to_index, index_to_group, group_to_index, group_to_roi
        lh_rois_all = [f"L_{roi}" for roi in core_rois_all]
        rh_rois_all = [f"R_{roi}" for roi in core_rois_all]
        rois_all = lh_rois_all + rh_rois_all

        groups_all = [f"GlasserGroup_{x}" for x in range(1,23)]
        assert isinstance(selected_rois, (list, str)), f"selected_rois must be a list of strings or the string 'all'. You entered {selected_rois}"
        if isinstance(selected_rois, list):
            assert all(isinstance(roi, str) for roi in selected_rois), "All elements in the list must be strings."
            for roi in selected_rois:
                assert roi in rois_all or roi in core_rois_all or roi in groups_all, f"Invalid ROI. ROI must be a valid Glasser ROI or Glasser group. You input {roi}."
        elif selected_rois != 'all':
            raise ValueError(f"Input 'selected_rois' must be 'all' if not a list of strings. You entered {selected_rois}")
        
        #useful parameters
        self.numvertices = len(hcp.mmp.map_all)
        #a numpy array of indices in fsLR32k space that are undefined for at least one trial across all datasets
        self.nan_indices_dataset = np.load(os.path.join(dataset_root, "nan_indices_dataset.npy")) 
        
        #now get the indices for the ROIs you want
        if selected_rois == 'all':
            self.selected_rois = rois_all.copy()
        else:
            self.selected_rois = []
            for roi in selected_rois: #loop over user specified rois
                if "GlasserGroup_" in roi:
                    rois_in_group = self.group_to_roi[roi] 
                    self.selected_rois.extend(rois_in_group)
                elif roi in rois_all: #in left or right hemisphere roi list
                    self.selected_rois.append(roi)
                elif roi in core_rois_all: #no L_ or R_ tag
                    self.selected_rois.append(f"L_{roi}")
                    self.selected_rois.append(f"R_{roi}")
                else:
                    raise ValueError(f"Invalid ROI {roi}. ROI must be one of the 360 valid Glasser Atlas ROIs or 'GlasserGroup_X', \
                                     where X is one of the 22 valid Glasser Atlas groups.")
        
        self.selected_rois = list(dict.fromkeys(self.selected_rois)) #remove duplicates and preserve order of the list
        self.selected_roi_indices = [] #collect the corresponding indices to the rois that you want
        for roi in self.selected_rois:
            self.selected_roi_indices.extend(self.roi_to_index[roi])
        if not self.selected_roi_indices:
            raise ValueError("No valid ROI indices found for the provided selected_rois.")
        
        #optionally remove the vertices that were found to be NaN at some trial throughout the dataset
        if remove_nan_vertices:
            self.selected_roi_indices = [idx for idx in self.selected_roi_indices if idx not in self.nan_indices_dataset]
        self.selected_roi_indices = sorted(self.selected_roi_indices) #sort the indices. sometimes indexing (like with the hdf5 file) needs it to be sorted.
    
    def __call__(self, sample, hdf5=None) -> np.ndarray:
        """
        Given a whole brain response in fsLR32k space, it returns a numpy array of only the desired vertices defined by
        your previously selected ROIs.
        """
        if hdf5:
            return hdf5[self.selected_roi_indices]
        else:
            assert sample.shape == (91282,), f"Must index a whole brain fsLR32k sample for the ROI indices to be accurate. Your sample was shape {sample.shape}."
            return sample[self.selected_roi_indices]

    def __len__(self) -> int:
        """
        len() returns the number of vertices that you are indexing, combined across all ROIs
        """
        return(len(self.selected_roi_indices))
    
    def __str__(self) -> str:
        """
        print returns the list of ROIs
        """
        return f"List of ROIS used: {self.selected_rois}"

    def sample2wb(self, fmri, fill_value=np.nan) -> np.ndarray:
        """
        Given a fmri sample of a short vector (maybe it was indexed from Whole Brain vertices for a specific set of ROIs)
        map the values back into the whole brain. Useful for visualization or to re-index the whole brain for RSA analysis.
        if vertices were excluded for being NaN, this remapping will account for it if fmri and self.selected_roi_indices are the same shape. 
        INPUT:
        fmri, numpy array of fmri values of shape self.selected_roi_indices.
        fill_value, float or int, the default value you want to fill all vertices not specified by your fmri data in input 'fmri' for 
            their respective self.selected_roi_indices indices.
        """
        if len(fmri) != len(self.selected_roi_indices):
            raise ValueError(f"input fmri and the selected roi indices must be the same length. \
          Your fmri input length is {fmri.shape} and self.selected_roi_indices length is {len(self.selected_roi_indices)}")
        reconstructed_sample = np.full(self.numvertices, fill_value)
        reconstructed_sample[self.selected_roi_indices] = fmri 
        return reconstructed_sample
    
    def _initialize_roi_mappings(self) -> list[str]:
        """
        This initialization function initializes mappings between indices, rois, and groups from the Glasser atlas 
        in the fsLR32k mesh.
        INPUT:
        self
        RETURNS
        core_rois_all, list of core rois (meaning no left or right hemisphere prefix)
        """
        tmp = pd.read_table(os.path.join(project_root, "src", "utils", "roi_list","hcp_glasser_roilist.txt"), sep=',')
        core_rois_all = tmp['ROI'].values
        lh_rois_all = [f"L_{roi}" for roi in core_rois_all]
        rh_rois_all = [f"R_{roi}" for roi in core_rois_all]

        rois_all = lh_rois_all + rh_rois_all
        self.numcorticalvertices = len(hcp.vertex_info.grayl) + len(hcp.vertex_info.grayr) 

        self.roi_to_group = {roi: '' for roi in rois_all}
        for roi in rois_all:
            core_roi = roi.split('_')[-1] #discard the left/right hemisphere. overwriting is ok
            x = int(tmp.loc[tmp['ROI'] == core_roi, 'GROUP'].values[0])
            self.roi_to_group[roi] = f"GlasserGroup_{x}"

        self.group_to_roi = {f"GlasserGroup_{x}": [] for x in range(1,23)}
        for x in range(1,23):
            core_rois = list(tmp.loc[tmp['GROUP'] == x, 'ROI'].values)
            lr_rois = [f'L_{roi}' for roi in core_rois] + [f'R_{roi}' for roi in core_rois] 
            self.group_to_roi[f"GlasserGroup_{x}"] = lr_rois
        
        #map each ROI to a numpy array of its fsLR32k indices 
        self.roi_to_index = {str(roi): np.where(hcp.mmp.map_all == int(roi_id))[0] for roi_id, roi in hcp.mmp.labels.items() if roi in rois_all} 
        self.index_to_roi = {index: '' for index in range(self.numcorticalvertices)}
        self.index_to_group = {index: '' for index in range(self.numcorticalvertices)}
        for roi, indices in self.roi_to_index.items():
            for idx in indices:
                self.index_to_roi[idx] = roi
                self.index_to_group[idx] = self.roi_to_group[roi]
        
        self.group_to_index = {f"GlasserGroup_{x}": [] for x in range(1,23)}
        for x in range(1,23):
            rois_in_group = self.group_to_roi[f"GlasserGroup_{x}"]
            for roi in rois_in_group:
                self.group_to_index[f"GlasserGroup_{x}"].extend(self.roi_to_index[roi])

        return core_rois_all

class ToTensorfMRI:
    def __init__(self, dtype: str=None):
        if not dtype:
            dtype = 'float32' #default dtype if not specified
        self.dtype = dtype
        if self.dtype.lower() not in ['float64', 'float32', 'float16']:
            raise ValueError(f"specified dtype {self.dtype} is not supported. Must be one of 'float64', 'float32', 'float16'")
        
    def __call__(self, numpy_array):
        if self.dtype.lower() == 'float64':
            torch_array = torch.from_numpy(numpy_array).double()
        if self.dtype.lower() == 'float32':
            torch_array = torch.from_numpy(numpy_array).float()
        elif self.dtype.lower() == 'float16':
            torch_array = torch.from_numpy(numpy_array).half()
        return torch_array

class ToTensorSubjectID:
    def __init__(self, dtype=None):
        if not dtype:
            dtype = 'long' #default dtype if not specified
        self.dtype = dtype
        if self.dtype.lower() not in ['long', 'int']:
            raise ValueError(f"specified dtype {self.dtype} is not supported. Must be one of 'long' or 'int'")
        
    def __call__(self, subject_idx):
        if self.dtype == 'long':
            torch_int = torch.tensor(subject_idx, dtype=torch.long)
        elif self.dtype == 'int':
            torch_int = torch.tensor(subject_idx, dtype=torch.int)
        return torch_int

class ToTensorAngleID:
    def __init__(self, dtype: str=None):
        if not dtype:
            dtype = 'float32' #default dtype if not specified
        self.dtype = dtype
        if self.dtype.lower() not in ['float64', 'float32', 'float16']:
            raise ValueError(f"specified dtype {self.dtype} is not supported. Must be one of 'float64', 'float32', 'float16'")
        
    def __call__(self, viewing_angle):
        if self.dtype.lower() == 'float64':
            torch_array = torch.tensor(viewing_angle).double()
        if self.dtype.lower() == 'float32':
            torch_array = torch.tensor(viewing_angle).float()
        elif self.dtype.lower() == 'float16':
            torch_array = torch.tensor(viewing_angle).half()
        return torch_array
    
class InverseNormalize:
    def __init__(self):
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    def __call__(self, image, *args, **kwds):
        unnormalized_img = image * self.std + self.mean  # Assuming inputs[0] is in [C, H, W] and normalized
        unnormalized_img = torch.clip(unnormalized_img, 0, 1)  # Clip values to [0, 1] for visualization
        return unnormalized_img
