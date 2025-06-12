import os
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import matplotlib
from matplotlib import pyplot as plt
from nilearn import datasets
from sklearn.linear_model import Ridge, Lasso, MultiTaskLassoCV
from nilearn import plotting
import pickle
from himalaya.ridge import RidgeCV
import torch
from typing import Optional
from torch.utils.data import DataLoader, Dataset
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from torchvision import transforms
from sklearn.decomposition import IncrementalPCA
from himalaya.ridge import RidgeCV
from sklearn.linear_model import LinearRegression, Lasso
#from scipy.stats import pearsonr as corr
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity
from himalaya.ridge import RidgeCV
from sklearn.decomposition import PCA
from typing import Dict
import numpy.typing as npt
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

#local
from src.utils.helpers import vectorized_correlation
from src.utils.helpers import get_fsLR32k_adjacency_matrix, get_lowertriangular
from src.utils.transforms import SelectROIs

#class and functions to perform regression for encoding and decoding and evaluate results
dataset_root = os.path.join(os.getenv("DATASETS_ROOT"), "MOSAIC")

class Evaluate:
    def __init__(self, Y_pred, Y, ROI_selection: SelectROIs):
        """
        Y_pred and Y are to be the same shape and size (nsamples, nfeatures)
        """
        if Y_pred.shape != Y.shape:
            raise ValueError(f"Y_pred (shape {Y_pred.shape}) and Y (shape {Y.shape}) are not the same size")
        self.Y_pred = Y_pred
        self.Y = Y
        self.ROI_selection = ROI_selection

    def voxelwise_encoding(self):
        """
        Y_pred and Y are to be the same shape and size (nsamples, nfeatures).
        Voxelwise correlation correlates (pearson) over the samples at each feature (vertex).
        Returns a vector of correlations of length nfeatures.
        """
        return vectorized_correlation(self.Y_pred, self.Y)
    
    def veRSA(self):
        """
        Computes voxelwise RSA for each ROI and subject. 
        Returns a dictionary of spearman correlation between the predicted ROI RDM and observd ROI RDM for each ROI
        """
        representational_similarity = {roi: np.nan for roi in self.ROI_selection.selected_rois}
        for selected_roi in tqdm(self.ROI_selection.selected_rois, total=len(self.ROI_selection.selected_rois), desc=f"Computing RSA over ROIs"):
            roi_indices = self.ROI_selection.roi_to_index[selected_roi]
            vertex_matrix_indices = [idx in roi_indices for idx in self.ROI_selection.selected_roi_indices] #iterate over all selected indices and put True if the index is in our desired ROI, False otherwise
            if sum(vertex_matrix_indices) <= 1:
                #can't do a RDM with one feature
                representational_similarity[selected_roi] = np.nan
            else:
                fmri_roi = self.Y[:,vertex_matrix_indices]
                fmri_pred_roi = self.Y_pred[:, vertex_matrix_indices]

                rdm_roi = 1 - np.corrcoef(fmri_roi)
                rdm_pred_roi = 1 - np.corrcoef(fmri_pred_roi)
                representational_similarity[selected_roi] = spearmanr(get_lowertriangular(rdm_roi), get_lowertriangular(rdm_pred_roi))[0]
            
        return representational_similarity
