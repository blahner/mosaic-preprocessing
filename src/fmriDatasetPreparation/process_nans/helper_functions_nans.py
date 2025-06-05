from dotenv import load_dotenv
load_dotenv()
import numpy as np

def fillnans(sample, roi_indices, roi_adjacency=None, fillvalue=None):
    """
    Fill non-real fmri values (nan or +/- inf) with specified value.
    Inputs:
    sample: numpy array, shape (91282,). The full 91282 fsLR32k sample of some fmri measurement or derivative (e.g., noiseceiling, stat etc).
    roi_indices: list of int or bool defining which indices of the fsLR32k sample you want to check for nan indices.
    roi_adjacency: dict, adjacency matrix for the fsLR32k mesh where each key is an index and the value are those index's neighbors. Derive it from hcp_utils.cortical_adjacency. 
    fillvalue: real value of type int or float, or string 'normal_fmri', 'normal_zero', or 'adjacency', or None. If a real value (e.g., 0), 
    all nan values within sample[roi_indices] will be filled to that value. If 'normal_zero', all nan values will be filled by drawing a random number from a normal 
    distribution with 0 mean 1 std. If 'normal_fmri', all nan values will be filled by drawing a random number from a normal 
    distribution with a mean and std computed over the real-valued fmri data from the sample[roi_indices]. If 'adjacency', the nan values 
    are recursively filled by identifying the nan-vertex with the greatest number of real-valued neighbors, filling in the vertex with the mean of 
    those neighbors, and repeating until all NaN vertices are filled.
    RETURNS:
    sample, numpy array, shape (91282,) of the result of the nan filled process.
    nan_indices, numpy array, an array containing the indices relative to fsLR32k space of the NaNs identified in that sample indexed by roi_indices.
    """

    assert sample.shape == (91282,), f"Must index a whole brain fsLR32k sample for the ROI indices to be accurate. Your sample was shape {sample.shape}."
    if fillvalue is not None:
        if isinstance(fillvalue, str):
            valid_strings = ['normal_fmri', 'normal_0', 'adjacency']
            assert fillvalue in valid_strings, f"Value must be one of {valid_strings}, but got '{fillvalue}'."
        elif isinstance(fillvalue, (int, float)):
            assert np.isfinite(fillvalue), f"Value must be a finite number, but got {fillvalue}."
        else:
            raise TypeError(f"Value must be a string or a finite number, but got type {type(fillvalue).__name__}.")    

    fmri = sample[roi_indices]
    not_real_mask = np.isnan(fmri) | np.isinf(fmri)

    if not not_real_mask.any():
        nan_indices = np.array([])  #no undefined values exist
        return fmri, nan_indices
    else:
        nan_indices = np.array(roi_indices)[not_real_mask]

    if fillvalue is None:
        return sample, nan_indices
    elif fillvalue == 'normal_fmri':
        fmri[not_real_mask] = np.random.normal(loc=np.nanmean(fmri), scale=np.nanstd(fmri), size=not_real_mask.sum())               
    elif fillvalue == 'normal_zero':
        fmri[not_real_mask] = np.random.normal(loc=0, scale=1, size=not_real_mask.sum())               
    elif isinstance(fillvalue, (int, float)):
        fmri[not_real_mask] = fillvalue
    elif fillvalue == 'adjacency':
        if roi_adjacency is None:
            raise ValueError(f"An adjacency matrix for the fsLR32 mesh must be defined if you are filling NaN values based on vertex adjacency.")
        
        while not_real_mask.any():
            #identify the NaN indices in the ROI
            not_real_indices = np.array(roi_indices)[not_real_mask]

            #precompute finite percentages for all NaN indices
            percent_finite = [
                np.sum(np.isfinite(sample[roi_adjacency[idx]])) / len(roi_adjacency[idx])
                for idx in not_real_indices
            ]

            #fill the index with the highest percentage of finite values
            idx_to_fill = np.argmax(percent_finite)
            index_to_fill = not_real_indices[idx_to_fill]
            neighbors = roi_adjacency[index_to_fill]

            #compute the mean of finite values from neighbors
            finite_neighbors = sample[neighbors]
            sample[index_to_fill] = np.nanmean(finite_neighbors)

            #update NaN mask
            fmri = sample[roi_indices]
            not_real_mask = np.isnan(fmri) | np.isinf(fmri)
    else:
        raise ValueError(f"fillvalue {fillvalue} not recognized. Must be one of ['None', 'normal_fmri', 'normal_zero', 'adjacency'] or a value of type 'int' or 'float'.")

    sample[roi_indices] = fmri  #update the original sample with filled values
    return sample, nan_indices