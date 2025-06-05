#standard
from dotenv import load_dotenv
load_dotenv()
import os
import sys
sys.path.append(os.getenv('PYTHONPATH')) 
import warnings
warnings.filterwarnings('ignore')
import pickle

#third party
import numpy as np
import hcp_utils as hcp
import matplotlib.pyplot as plt
from nilearn import plotting
from typing import Union, List

from src.utils.helpers import calculate_noiseceiling #, calculate_noiseceiling_vary_reps, extract_repeating_stimuli

from typing import Union, Tuple, List
import numpy as np
from numpy.typing import NDArray

def limit_true_values(arr: NDArray, reps: int) -> NDArray:
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

def extract_repeating_stimuli(
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
            nonan_reps = limit_true_values(nonan_reps, reps)
        betas_nonans[idx, :, :] = betas[cond, nonan_reps, :]
        
    return betas_nonans, full_cond

def calculate_noiseceiling_vary_reps(
    betas: NDArray, 
    n: Union[int, str] = 'avg'
) -> Tuple[NDArray, NDArray]:
    """
    Calculate noise ceiling for beta estimates with varying repetitions across stimuli.
    
    This function estimates the noise ceiling by:
    1. Computing standard deviation across trials
    2. Squaring and averaging across images
    3. Taking the square root to get noise standard deviation
    
    Parameters:
    -----------
    betas : NDArray
        Beta estimates with shape (vertices, num_reps, num_stimuli)
        Stimuli with fewer than max_reps repetitions should have NaN values
    n : Union[int, str], default='avg'
        Number of trials for noise ceiling computation:
        - 'avg': Estimates assuming responses are averaged over all available reps
        - int > 0: Requires exactly n repetitions for all stimuli
    
    Returns:
    --------
    Tuple[NDArray, NDArray]
        - ncsnr: Noise-ceiling SNR at each vertex (ratio of signal std to noise std)
        - noiseceiling: Noise ceiling at each vertex (% of explainable variance)
    
    Raises:
    -------
    ValueError
        - If n < 1 when n is int
        - If n is str but not 'avg'
        - If betas has only one repetition for all stimuli
        - If n > 1 but not all stimuli have n repetitions
    """
    # Input validation
    if not isinstance(betas, np.ndarray) or len(betas.shape) != 3:
        raise ValueError("betas must be a 3D numpy array")
    if isinstance(n, int) and n < 1:
        raise ValueError("n must be >= 1")
    if isinstance(n, str) and n != 'avg':
        raise ValueError(f"If n is a string, it must be 'avg', not {n}")

    num_vertices, _, num_stimuli = betas.shape
    betas_t = betas.transpose(2, 1, 0)  # Transform to match extract_repeating_stimuli input format

    # Collect statistics for each repetition count
    n_avg = []  # Number of stimuli with i+1 repetitions
    betas_subsets = []  # Beta matrices for each repetition count
    reps = 1
    
    while sum(n_avg) < num_stimuli:
        # Extract stimuli with exactly 'reps' repetitions
        betas_nonans, full_cond = extract_repeating_stimuli(betas_t, reps, cutoff='exact')
        n_stims = len(full_cond)
        n_avg.append(n_stims)
        
        if n_stims > 0:
            betas_subsets.append(betas_nonans.transpose(2, 1, 0))  # Transform back to original format
        reps += 1

    if len(n_avg) == 1:
        raise ValueError("Cannot estimate noiseceiling with only one repetition")
    
    if isinstance(n, int) and n > 1:
        if n_avg[n-1] != num_stimuli:
            raise ValueError(
                f"Specified n={n} requires all stimuli to have {n} repetitions. "
                f"Current repetition counts: {n_avg}. Use n='avg' for varying reps."
            )

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
    total_var = np.nanvar(combined_betas.reshape(num_vertices, -1), axis=1)

    # Pool noise variance across subsets
    total_df = 0
    pooled_var_num = 0
    
    for subset in betas_subsets:
        if subset.shape[1] > 1:  # Skip single-rep subsets
            num_reps = subset.shape[1]
            num_stims = subset.shape[2]
            df = num_stims * (num_reps - 1)
            
            noise_var = np.mean(np.var(subset, axis=1, ddof=1), axis=1)
            pooled_var_num += noise_var * df
            total_df += df

    noise_var = pooled_var_num / total_df if total_df > 0 else np.zeros(num_vertices)
    
    # Compute signal variance and SNR
    signal_var = np.maximum(total_var - noise_var, 0)
    ncsnr = np.sqrt(signal_var / noise_var)
    
    # Calculate averaging term
    if n == 'avg':
        avg_term = sum(ns/(i+1) for i, ns in enumerate(n_avg)) / sum(n_avg)
    else:
        avg_term = 1/n
        
    # Calculate final noise ceiling
    noiseceiling = 100 * (ncsnr**2 / (ncsnr**2 + avg_term))
    
    return ncsnr, noiseceiling


#setup paths
datasets_root = os.path.join(os.getenv("DATASETS_ROOT", "/default/path/to/datasets")) #use default if DATASETS_ROOT env variable is not set.
dataset_root = os.path.join(datasets_root, "NaturalScenesDataset")
meta_dataset_root = os.path.join(datasets_root, "MOSAIC")
project_root = os.getenv("PROJECT_ROOT", "/default/path/to/datasets")
print(f"dataset_root: {dataset_root}")
print(f"project_root: {project_root}")
fmri_path = os.path.join(dataset_root,"derivatives", "GLM")

n_task = {'test': [1,'avg', 3]} #{'artificial': [1,'avg']} #{'train': [1,'avg'], 'test': [1,'avg'], 'artificial': [1,'avg']}
ext_list = ['png']
save_flag = False
for subject in range(3,4):
    save_root = os.path.join(project_root, "src", "fmriDatasetPreparation", "datasets", "NaturalScenesDataset", "validation", "output", "noiseceiling", f"sub-{subject:02}")
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    for task in n_task.keys():
        with open(os.path.join(dataset_root, "derivatives", "GLM", f"sub-{subject:02}", "prepared_betas", f"sub-{subject:02}_organized_betas_task-{task}_normalized.pkl"), 'rb') as f: 
            betas, stimorder = pickle.load(f) #betas is shape numstim, numreps, numvertices
        
        #betas, _ = extract_repeating_stimuli(betas,3, cutoff='exact')
        betas = betas.T
        print(betas.shape)
        for n in n_task[task]:
            #betas_padded = np.pad(betas, 
            #          pad_width=((0,0), (0,2), (0,0)),  #pad remaining columns in middle dimension with nans
            #          mode='constant',
            #          constant_values=np.nan)
            #betas_padded[:,1:,:200] = np.nan
            betas_modified = betas.copy()
            print("Calculating noiseceiling...")
            if n == 'avg':
                ncsnr, noiseceiling = calculate_noiseceiling(betas, n=3)
            else:
                ncsnr, noiseceiling = calculate_noiseceiling(betas, n=n)
            print('starting subset test')
            ncsnr_test, noiseceiling_test = calculate_noiseceiling_vary_reps(betas_modified, n=n)

            #print(f"sub-{subject:02} {task} n={n} original max noiseceiling: {np.nanmax(noiseceiling)}")
            print(f"sub-{subject:02} {task} n={n} test max noiseceiling: {np.nanmax(noiseceiling_test)}")

            assert np.allclose(ncsnr, ncsnr_test, rtol=1e-5, atol=1e-8, equal_nan=True)
            assert np.allclose(noiseceiling, noiseceiling_test, rtol=1e-5, atol=1e-8, equal_nan=True)
