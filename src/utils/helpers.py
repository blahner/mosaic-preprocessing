from dotenv import load_dotenv
load_dotenv()
import os
from typing import Optional
import numpy as np
import hcp_utils as hcp
import pickle
from tqdm import tqdm
import torch
import scipy
import random
from typing import Union, List, Tuple
from numpy.typing import NDArray
import warnings


dataset_root = os.path.join(os.getenv("DATASETS_ROOT"), "MOSAIC")

class FilterDataset:
    """
    Class to perform various functions on the dataset based on the specified subjects to include.
    For example, when specifying certain subjects the train/val/test splits should be filtered
    accordingly and the subjectID mappings should be appropriately defined. 
    """
    def __init__(self, subject_include: Optional[str] = None, dataset_include: Optional[str] = None, use_noiseceiling: bool=False):
        """
        INPUT:
        config: config dictionary that defines the subjects and datasets to be excluded using the config's 'include_datasets' and 'include_subjects' fields.
        """
        self.datasets_subjects = {"NSD": [f"sub-{s:02}_NSD" for s in range(1,9)],
                            "BMD": [f"sub-{s:02}_BMD" for s in range(1,11)],
                            "BOLD5000": [f"sub-{s:02}_BOLD5000" for s in range(1,5)],
                            "HAD": [f"sub-{s:02}_HAD" for s in range(1,31)],
                            "THINGS": [f"sub-{s:02}_THINGS" for s in range(1,4)],
                            "NOD": [f"sub-{s:02}_NOD" for s in range(1,31)],
                            "deeprecon": [f"sub-{s:02}_deeprecon" for s in range(1,4)],
                            "GOD": [f"sub-{s:02}_GOD" for s in range(1,6)]}
        self.subjects_nonoiseceiling = []
        for s in range(1,31):
            self.subjects_nonoiseceiling.append(f"sub-{s:02}_HAD") #all HAD subjects had no repeats so no noiseceiling is available
        for s in range(10,31):
            self.subjects_nonoiseceiling.append(f"sub-{s:02}_NOD") #NOD subjects 10-30 did not have any repeats so no noiseceiling is available
        
        self.subjects_to_include = []
        if dataset_include:
            for dset in dataset_include:
                self.subjects_to_include.extend(self.datasets_subjects[dset])

        if subject_include:
            for sub in subject_include:
                self.subjects_to_include.append(sub)
        
        if len(self.subjects_to_include) == 0 and (subject_include or dataset_include):
            raise ValueError(f"No subjects were included based on your 'subject_include' input of {subject_include} \
                             and 'dataset_include' input of {dataset_include}. Must include at least one subject.")

        if use_noiseceiling:
            subjects_to_include_copy = self.subjects_to_include.copy()
            for excluded_sub in self.subjects_nonoiseceiling:
                if excluded_sub in subjects_to_include_copy:
                    self.subjects_to_include.remove(excluded_sub)

    def filter_splits(self, json_dataset):
        """
        Given a train or test json and a config containing parameters to exclude or include certain datasets or subjects,
        this function returns a train or test json that only includes the desired subjects. Filtering down to the
        individual trial level is not supported. This filtering is on top of the previous dataset filtering during the compilation
        process, such as removal of test/train conflicts and perceptually similar images.
        INPUT:
        json_dataset: List of dictionaries in the format [{'sub-XX_DATASET_stimulus-STIMULUSNAME': [response01.npy, response02.npy]}]
            this is the dataset file output from make_testtrain.py step.
        RETURNS:
        filtered_dataset: Same design as json dataset, just only including the appropriate datasets and subjects defined by config.
        """

        filtered_dataset = []
        self.included_subjects = set() #keeps track of the subjects that are actually included with the dataset you input here and the desired subjects you initialized with
        for index_dict in json_dataset:
            filename_key = list(index_dict.keys())[0]
            subject_dataset, _ = filename_key.split('_stimulus-')
            if subject_dataset in self.subjects_to_include:
                filtered_dataset.append({filename_key: index_dict[filename_key]})
                self.included_subjects.add(subject_dataset)
        
        if len(filtered_dataset) == 0:
            raise ValueError(f"There is no data overlap between the subjects you want ({self.subjects_to_include}) and the data you input.")        
        mapping = self._subjectID_map()

        return filtered_dataset, mapping
    
    def _subjectID_map(self):
        """
        Given an 0-based index, map it to a real subject ID in the format 'sub-XX_DATASET'.
        This mapping will get passed to the dataset classes so the samples can be 
        appropriately loaded and input into the model.
        Must run 'filter_splits()' first since it is possible to initialize this class
        with subjects you want to include, but the dataset doesn't actually contain any of the subjects
        desired. For example, you can specify a subject with only naturalistic images but give it a json dataset
        of only artificial images. That combination should throw an error.
        """
        mapping = {} #subjectID: index
        for idx, sub in enumerate(self.included_subjects):
            mapping[sub] = idx
        return mapping

    def get_stimulus_filenames(self, json_dataset):
        """
        Returns a list of all stimulus filenames in a given json dataset. Does not
        account for any desired subjects you might want to include/exclude. 
        """
        stimulus_filenames = []
        for index_dict in json_dataset:
            filename_key = list(index_dict.keys())[0]
            subject_dataset, stimulus_filename = filename_key.split('_stimulus-')
            if stimulus_filename not in stimulus_filenames:
                stimulus_filenames.append(stimulus_filename)
        return stimulus_filenames
    
    def __len__(self):
        """
        The length of this class is the number of subjects included
        """
        return len(self.included_subjects)

def get_fsLR32k_adjacency_matrix(radius: int, save_flag: bool=True) -> dict[list]:
    """
    Define an adjacency matrix for the fsLR32k mesh for a given radius. For each center vertex, you will get
     the vertices that are 'radius' number of vertices away from the center vertex. For example,
    radius=0 is no adjacent vertices, same as univaritate. radius=1 means immediately adjacent vertices.
    radius=2 means the vertices both adjacent to the center 'vertex' and their adjacent vertices. Note that this
    is agnostic to any ROI selection or nan indices - it computes the adjacency for every cortical vertex.
    INPUTS:
    radius: int. Postive integer or zero that dictates how far away we return adjacent vertices for a given vertex.
    save_flag: bool. If true, this function saves the output adjacency matrix.
    RETURNS:
    selected_roi_adjacency: dictionary of lists. Each key is an index in 'desired_indices'. Each value is a list of integers
    that are adjacent vertices to the key of at most radius 'radius' away.
    """
    if not isinstance(radius, int) or radius < 0:
        raise ValueError(f"Radius input must be a positive number or zero type int. You entered {radius}.")
    
    adjacency_matrix = hcp.cortical_adjacency
    numvertices = adjacency_matrix.shape[0]
    #precompute adjacency lists for all `roi_indices`
    all_roi_adjacency = {idx: adjacency_matrix[idx].indices for idx in range(numvertices)} #define the adjacency matrix for all cortical vertices and their immediate neighbors
    all_roi_adjacency_radius = {idx: [idx] for idx in range(numvertices)} #initialize the selection at r=0
    for vertex in tqdm(range(numvertices), total=numvertices, desc=f"Creating adjacency matrix of fsLR32k vertices of radius 4 vertices"):
        vertex_list = all_roi_adjacency_radius[vertex].copy() #initial condition for the center vertex
        for _ in range(1, radius+1):
            for v in vertex_list:
                neighbors = all_roi_adjacency[v] 
                for neighbor in neighbors:
                    if neighbor not in all_roi_adjacency_radius[vertex]:
                        all_roi_adjacency_radius[vertex].append(neighbor) #this updates the list every time
            vertex_list = all_roi_adjacency_radius[vertex].copy() #update the vertex list as you increase the radius
        assert(len(all_roi_adjacency_radius[vertex]) == len(set(all_roi_adjacency_radius[vertex])))
    
    if save_flag:
        save_path = os.path.join(dataset_root, "decoding_artifacts", f"fsLR32k_adjacency_matrix_radius-{radius}.pkl")
        print(f"Saving adjacency matrix to: {save_path}")
        with open(save_path, 'wb') as f:
            pickle.dump(all_roi_adjacency_radius, f)
    return all_roi_adjacency_radius

def vectorized_correlation(x,y,axis=0, ddof=1, is_torch=False):
    """
    Compute vectorized pearson correlation. x and y inputs must be equal and be 2D.
    Suitable for numpy and torch vectors.
    """
    if x.shape != y.shape:
        raise ValueError(f"x (shape {x.shape}) and y (shape {y.shape}) must match.")
    if len(x.shape) != 2:
        raise ValueError(f"x and y shapes must be 2D. Your shapes are length {len(x.shape)}.")
    
    centered_x = x - x.mean(axis=axis, keepdims=True)
    centered_y = y - y.mean(axis=axis, keepdims=True)

    covariance = (centered_x * centered_y).sum(axis=axis, keepdims=True) / (x.shape[axis] - ddof)

    if is_torch:
        x_std = torch.std(x,dim=axis,correction=ddof,keepdims=True)
        y_std = torch.std(y,dim=axis,correction=ddof,keepdims=True)
        corr = covariance / (x_std * y_std + torch.finfo(float).eps)
        return corr.flatten()

    else:
        x_std = np.std(x,axis=axis,ddof=ddof,keepdims=True)
        y_std = np.std(y,axis=axis,ddof=ddof,keepdims=True)
        corr = covariance / (x_std * y_std + np.finfo(float).eps)
        return corr.ravel()

def get_lowertriangular(rdm, is_torch: bool=False):
    num_conditions = rdm.shape[0]
    if is_torch:
        return rdm[torch.triu_indices(num_conditions,1)]
    else:
        return rdm[np.triu_indices(num_conditions,1)]

def interpolate_ts(fmri, tr_acq, tr_resamp):
    #interopolate the fmri time series. Can be either 2D (surface x time) or 4D (volume x time) array.
    #number of scans (time) has to be the last dimension

    if fmri.shape[-1] <= 1:
        raise ValueError("Cannot interpolate with single timepoint")

    numscans_acq = fmri.shape[-1]
    secsperrun = numscans_acq*tr_acq #time in seconds of the run
    numscans_resamp = int(secsperrun/tr_resamp)

    x = np.linspace(0, numscans_acq, num=numscans_acq, endpoint=True)
    f = scipy.interpolate.interp1d(x, fmri)
    x_new = np.linspace(0, numscans_acq, num=numscans_resamp, endpoint=True)

    fmri_interp = f(x_new)
    return fmri_interp

class ComputeNoiseceiling:
    def __init__(self, betas: NDArray, n: Union[int, str] = 'avg'):
        """
        Class to compute the noise ceiling for beta estimates with varying repetitions across stimuli.
        
        Briefly, this function estimates the noise ceiling by:
        1. Compute noise variance (across trials)
        2. Compute total variance (across all stimulus repetitions)
        3. Signal variance (total variance - noise variance)
        4. NCSNR is ratio of signal to noise 
        5. noiseceiling is NCNSR^2/(NCSNR^2+avg_term) where the avergage term modulates
            the noiseceiling by how many trials would be averaged for an estimation.

        The input betas matrix can contain stimuli of differing number of repetitions. In other words,
        some stimuli may have 3 repetitions, some may have 10, others 1 etc. In this case,
        the matrix should be of shape (vertices, max_reps, num_stimuli) where stimuli
        not repeated max_reps number of times will be vectors of nans.
        In order to calculate a noiseceiling for a vertex using stimuli of varying number of repetitions,
        the noise variance is pooled across subsets of the data of differing number of reps.
        The final noiseceiling calculation accounts for the varying number of averaged trials using the term
        [(A/a + B/b + C/c)/(A + B + C)] etc. where A, B, and C are the number of stimuli and a, b and c are the 
        number of repetitions for those stimuli. 
        Parameters:
        -----------
        betas : NDArray
            Beta estimates with shape (vertices, num_reps, num_stimuli)
            Stimuli with fewer than max_reps repetitions should have NaN values
        n : Union[int, str], default='avg'
            Number of trials for noise ceiling computation:
            - 'avg': Estimates assuming responses are averaged over all available reps
            - int > 0: Requires exactly n repetitions for all stimuli

        Code adapted from GLMsingle example: https://github.com/cvnlab/GLMsingle/blob/main/examples/example9_noiseceiling.ipynb
        See NSD data manual page for derivation: https://cvnlab.slite.page/p/6CusMRYfk0/Functional-data-NSD 
        """
        # Input validation
        if not isinstance(betas, np.ndarray) or len(betas.shape) != 3:
            raise ValueError("betas must be a 3D numpy array")
        if betas.shape[0] != 91282:
            warnings.warn(f"Your number of vertices is {betas.shape[0]}. If you are in fsLR32k space using all vertices, the value should be 91282. Double check the shape is correct.")
        if isinstance(n, int) and n < 1:
            raise ValueError("n must be >= 1")
        if isinstance(n, str) and n != 'avg':
            raise ValueError(f"If n is a string, it must be 'avg', not {n}")
        
        self.betas = betas
        self.n = n
        self.num_vertices, _, self.num_stimuli = self.betas.shape
    
    def compute_ncsnr_noiseceiling(self):
        """
        Compute signal variance by subtracting the noise variance from the total variance.
        Modulate the noiseceiling by the avg_term, or how many trials you plan to average.
        For example, n=1 estimates a noiseceiling for a single trial. If you plan to average over
        trials, which will increase the signal variance, then the noiseceiling should be adjusted accordingly
        to reflect the higher ceiling.

        Returns:
        --------
        Tuple[NDArray, NDArray]
            - ncsnr: Noise-ceiling SNR at each vertex (ratio of signal std to noise std)
            - noiseceiling: Noise ceiling at each vertex (% of explainable variance)
        """
        betas_subsets = self._compute_betas_subsets()
        totalvar = self._compute_totalvar(betas_subsets)
        noisevar = self._compute_noisevar(betas_subsets)
        
        #compute signal variance and SNR
        signal_var = np.maximum(totalvar - noisevar, 0)
        ncsnr = np.sqrt(signal_var) / np.sqrt(noisevar)
        
        #calculate averaging term
        if self.n == 'avg':
            avg_term = sum(ns/(i+1) for i, ns in enumerate(self.n_avg)) / sum(self.n_avg)
        else:
            avg_term = 1/self.n
            
        #calculate final noise ceiling
        noiseceiling = 100 * (ncsnr**2 / (ncsnr**2 + avg_term))
        
        return ncsnr, noiseceiling

    def _compute_variance_randomBetaSubset_TESTCASE(self):
        """
        Only use this method for a test case to confirm that pooling across beta matrix subsets
        is not affecting the final result. If you attempt to follow this path towards a complete
        noise ceiling estimate, you will (intentionally) run into errors. This returns just the total
        and noise variance for comparison with other methods to estimate the variance.
        """
        betas_subsets = self._compute_betas_subsets(split_method='random')
        totalvar = self._compute_totalvar(betas_subsets)
        noisevar = self._compute_noisevar(betas_subsets)

        return totalvar, noisevar

    def _compute_betas_subsets(self, split_method: str='natural'):
        """
        Divide the betas matrix into subsets based on the number of repetitions per stimuli.
        For example, if your betas matrix has n stimuli with 3 reps each and m stimuli with 5 reps each,
        it will be divided into two subsets of [(nvertices, 3, n), (nvertices, 5, m)]. If all stimuli have
        the same number of reps, the original beta matrix inside a list will be returned (i.e., one subset found).
        
        n_avg is an important variable that tracks the number of stimuli that have been repeated X times. This is 
        used to modulate the noiseceiling if averaging is desired.

        split_method should only be random if using "_compute_variance_randomBetaSubset_TESTCASE()". 'random' refers
        to randomly splitting the betas into subsets (e.g., not using number of reps to split). 'natural' just refers
        to the way the implementation should be done (described above) if you actually want to estimate the noise ceiling.
        """
        if split_method == 'random':
            #should only be used for a test case!

            nsplits = np.random.choice(np.arange(2, self.num_stimuli), size=1)
            split_points = np.random.choice(self.num_stimuli-1, nsplits-1, replace=False)
            split_points = np.sort(split_points)
            
            #create list of start and end indices
            splits = []
            start_idx = 0
            
            for split_point in split_points:
                splits.append((start_idx, split_point + 1))
                start_idx = split_point + 1
            
            #add the final split
            splits.append((start_idx, self.num_stimuli))

            betas_subsets = []
            for split in splits:
                betas_subsets.append(self.betas[:, :, split[0]:split[1]])

        elif split_method == 'natural':
            betas_t = self.betas.transpose(2, 1, 0)  # Transform to match extract_repeating_stimuli input format

            # Collect statistics for each repetition count
            self.n_avg = []  # Number of stimuli with i+1 repetitions
            betas_subsets = []  # Beta matrices for each repetition count
            reps = 1
            
            while sum(self.n_avg) < self.num_stimuli:
                # Extract stimuli with exactly 'reps' repetitions
                betas_nonans, full_cond = self._extract_repeating_stimuli(betas_t, reps, cutoff='exact')
                n_stims = len(full_cond)
                self.n_avg.append(n_stims)
                
                if n_stims > 0:
                    betas_subsets.append(betas_nonans.transpose(2, 1, 0))  # Transform back to original format
                reps += 1

            if len(self.n_avg) == 1:
                raise ValueError("Cannot estimate noiseceiling with only one repetition")
            
            if isinstance(self.n, int) and self.n > 1:
                if self.n_avg[self.n-1] != self.num_stimuli:
                    raise ValueError(
                        f"Specified n={self.n} requires all stimuli to have {self.n} repetitions. "
                        f"Current repetition counts: {self.n_avg}. Use n='avg' for varying reps."
                    )
        return betas_subsets 

    def _compute_totalvar(self, betas_subsets):
        """
        Compute the total variance of the original betas matrix by concatenating subsets and
         reshaping to (nvertices, ntrials) where there is no longer the num_reps dimension. 
         The total variance at each vertex is simply the variance across every individual trial,
         no further distinctions made.
        """
        # Compute total variance
        max_reps = max(b.shape[1] for b in betas_subsets)
        padded_betas = []
        
        for subset in betas_subsets:
            if subset.shape[1] < max_reps:
                pad_width = ((0, 0), (0, max_reps - subset.shape[1]), (0, 0))
                padded = np.pad(subset, pad_width, constant_values=np.nan)
                padded_betas.append(padded)
            else:
                padded_betas.append(subset)
        
        combined_betas = np.concatenate(padded_betas, axis=2)
        total_var = np.power(np.nanstd(np.reshape(combined_betas, (self.num_vertices , -1)), axis=1),2) 

        return total_var
    
    def _compute_noisevar(self, betas_subsets):
        """
        Computes the noise variance by pooling (weighted variance by DOF) the noise variance across
        each subset of betas. Each subset of betas contains all stimuli that 
        have X number of repetitions. This pooling is necessary because noise variance
        is computed across trials repetitions and thus must be weighted properly by how many stimuli had X number of repetitions.
        """
        # Pool noise variance across subsets
        total_df = 0
        pooled_var_num = 0
        
        for subset in betas_subsets:
            if subset.shape[1] > 1:  # Skip single-rep subsets
                num_reps = subset.shape[1]
                num_stims = subset.shape[2]
                df = num_stims * (num_reps - 1)
                
                noise_sd = np.sqrt(np.mean(np.power(np.std(subset,axis=1,keepdims=1,ddof=1),2),axis=2)).reshape((self.num_vertices,))
                noisevar = np.power(noise_sd,2)
                pooled_var_num += noisevar * df
                total_df += df

        noisevar = pooled_var_num / total_df if total_df > 0 else np.zeros(self.num_vertices)

        return noisevar

    def _calculate_noiseceiling_original(self, betas, n: Union[int, List[int]]=1):
        """
        Implementation close to GLMsingle example for computing a noise ceiling with a fixed n. In other words,
        if your beta matrix has some stimuli with more reps than others, do not use this function.
        Parameters:
        betas: beta estimates in shape (vertices, num_reps, num_stimuli)
        n: number of trials you want to compute noiseceiling for. n=1 (default) is single trial noise ceiling. otherwise it should be
        equal to the number of betas you average over in your computations.
        Returns:
        ncsnr: noise-ceiling SNR at each voxel in shape (voxel_x, voxel_y, voxel_z) as ratio between signal std and noise std
        noiseceiling: noise ceiling at each voxel in shape (voxel_x, voxel_y, voxel_z) as % of explainable variance 
        Code adapted from GLMsingle example: https://github.com/cvnlab/GLMsingle/blob/main/examples/example9_noiseceiling.ipynb
        """
        assert(len(betas.shape) == 3)
        numvertices = betas.shape[0]
        num_reps = betas.shape[1]
        num_vids = betas.shape[2]
        noisesd = np.sqrt(np.mean(np.power(np.std(betas,axis=1,keepdims=1,ddof=1),2),axis=2)).reshape((numvertices,))
        noisevar = np.power(noisesd,2)
        # Calculate the total variance of the single-trial betas.
        totalvar = np.power(np.std(np.reshape(betas, (numvertices , num_reps*num_vids)), axis=1),2)

        # Estimate the signal variance and positively rectify.
        signalvar = totalvar - noisevar

        signalvar[signalvar < 0] = 0
        # Compute ncsnr as the ratio between signal standard deviation and noise standard deviation.
        ncsnr = np.sqrt(signalvar) / noisesd

        # Compute noise ceiling in units of percentage of explainable variance
        noiseceiling = 100 * (np.power(ncsnr,2) / (np.power(ncsnr,2) + 1/n))
        return ncsnr, noiseceiling, totalvar, noisevar

    def _limit_true_values(self, arr: NDArray, reps: int) -> NDArray:
        """
        Limit the number of True values in a boolean array to specified repetitions.
        
        Parameters:
        -----------
        arr : NDArray
            Boolean array
        reps : int
            Maximum number of True values to keep
            
        Returns:
        --------
        NDArray
            Copy of input array with excess True values set to False
        """
        arr_copy = arr.copy()
        count = 0
        for i in range(len(arr)):
            if arr[i]:
                if count < reps:
                    count += 1
                else:
                    arr_copy[i] = False
        return arr_copy

    def _extract_repeating_stimuli(self,
        betas: NDArray, 
        reps: int, 
        cutoff: str = 'exact'
    ) -> Tuple[NDArray, List[int]]:
        """
        Extract stimuli with specified number of repetitions.
        
        Parameters:
        -----------
        betas : NDArray
            Beta values with shape (conditions, reps, vertices)
        reps : int
            Number of repetitions to extract
        cutoff : str, default='exact'
            'exact': Extract stimuli with exactly reps repetitions
            'at_least': Extract stimuli with at least reps repetitions
            
        Returns:
        --------
        Tuple[NDArray, List[int]]
            - Beta values for selected stimuli
            - Indices of selected stimuli
        """
        if cutoff not in ['exact', 'at_least']:
            raise ValueError("cutoff must be 'exact' or 'at_least'")
            
        # Find valid repetitions for each condition
        full_cond = []
        for cond in range(betas.shape[0]):
            valid_reps = (~np.isnan(betas[cond,:,:]).all(axis=1)).sum()
            if (cutoff == 'exact' and valid_reps == reps) or \
            (cutoff == 'at_least' and valid_reps >= reps):
                full_cond.append(cond)
                
        # Extract and format valid betas
        betas_nonans = np.zeros((len(full_cond), reps, betas.shape[-1]))
        for idx, cond in enumerate(full_cond):
            nonan_reps = ~np.isnan(betas[cond,:,:]).all(axis=1)
            if cutoff == 'at_least':
                nonan_reps = self._limit_true_values(nonan_reps, reps)
            betas_nonans[idx, :, :] = betas[cond, nonan_reps, :]
            
        return betas_nonans, full_cond

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  #for multi-GPU