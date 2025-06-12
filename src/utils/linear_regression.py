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
from sklearn.linear_model import LinearRegression, Lasso, LassoCV
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

class RegressionModels:
    def __init__(self, roi_selection: SelectROIs, adjacency_dict: Optional[dict]=None, n_jobs: int=1):
        self.roi_selection = roi_selection
        self.wb_adjacency_dict = adjacency_dict
        self.n_jobs = n_jobs #only used if a certain regression model is invoked

    def feature2fmri_encoding_regression(self, X: dict[np.array], Y: dict[np.array], regression_model: str='OLS'):
        """
        X are model features, Y are fmri data
        """
        predictions = {eval_set: [] for eval_set in X.keys()}
        if regression_model == 'OLS':
            model = LinearRegression(n_jobs=self.n_jobs)
        elif regression_model == 'Ridge':
            model = Ridge(alpha=10)
        elif regression_model == 'RidgeCV':
            model = RidgeCV(alphas=[0.1, 1, 10, 100, 1000, 10000, 100000])
        elif regression_model == 'Lasso':
            model = Lasso(alpha=10)
        elif regression_model == 'MultiTaskLassoCV':
            model = MultiTaskLassoCV(cv=5, random_state=42)
        model.fit(X['train_naturalistic'], Y['train_naturalistic'])
        if regression_model == 'LassoCV':
            print("Optimal alpha:", model.alpha_)

        for eval_set in predictions.keys():
            predictions[eval_set] = model.predict(X[eval_set])
        return model, predictions

    def fmri2feature_regression(self, 
                                vertices: npt.NDArray[np.bool_],
                                X: Dict[str, npt.NDArray],
                                Y: Dict[str, npt.NDArray],       
                                backend: str | None = None,       
                                regression_model: str = 'OLS'
                                ):
        """
        X are fmri data, Y are model features
        """
        
        predictions = {eval_set: [] for eval_set in X.keys()}
        if regression_model == 'OLS':
            model = LinearRegression(n_jobs=self.n_jobs)
        elif regression_model == 'Ridge':
            model = Ridge(alpha=10)
        elif regression_model == 'RidgeCV':
            model = RidgeCV(alphas=[0.1, 1, 10, 100, 1000, 10000, 100000])
        elif regression_model == 'Lasso':
            model = Lasso(alpha=10)
        elif regression_model == 'LassoCV':
            model = LassoCV(positive=False, cv=5, n_jobs=self.n_jobs)
        elif regression_model == 'MultiTaskLassoCV':
            model = MultiTaskLassoCV(cv=5, random_state=42)
        elif regression_model == 'RidgeCV':
            model = RidgeCV(alphas=[0.1, 1, 10, 100, 1000, 10000, 100000])
        model.fit(X['train_naturalistic'][:, vertices], Y['train_naturalistic'])
        
        for eval_set in predictions.keys():
            predictions[eval_set] = model.predict(X[eval_set][:, vertices])
        return model, predictions

    def veRSA_roi(self, Y: dict[np.array], Y_pred: dict[np.array], stimulus_dict):
        """
        Computes RSA for each ROI over all stimuli and divided by subject
        """
        #housekeeping
        subjects_naturalistic = []
        subjects_artificial = []
        for eval_set in stimulus_dict.keys():
            if 'naturalistic' in eval_set:
                subjects_naturalistic.extend(stimulus_dict[eval_set]['subject_id'])
            elif 'artificial' in eval_set:
                subjects_artificial.extend(stimulus_dict[eval_set]['subject_id'])
        all_subjectID_naturalistic = np.unique(subjects_naturalistic)
        all_subjectID_artificial = np.unique(subjects_artificial)

        representational_similarity = {} 
        for eval_set in stimulus_dict.keys():
            representational_similarity[eval_set] = {} 
            if 'naturalistic' in eval_set:
                for subjectID in all_subjectID_naturalistic:
                    representational_similarity[eval_set][subjectID] = {roi: 0 for roi in self.roi_selection.selected_rois}
            elif 'artificial' in eval_set:
                for subjectID in all_subjectID_artificial:
                    representational_similarity[eval_set][subjectID] = {roi: 0 for roi in self.roi_selection.selected_rois}
            representational_similarity[eval_set]['sub-all'] = {roi: 0 for roi in self.roi_selection.selected_rois}

        #loop over stimulus set and subjects for RSA
        for eval_set, subjectID_dict in representational_similarity.items():
            for subjectID in subjectID_dict.keys():
                if subjectID == 'sub-all':
                    bool_index = [True] * Y[eval_set].shape[0]
                    continue
                else:
                    bool_index = [subid == subjectID for subid in stimulus_dict[eval_set]['subject_id']]
                for selected_roi in tqdm(self.roi_selection.selected_rois, total=len(self.roi_selection.selected_rois), desc=f"Computing RSA over ROIs for {eval_set} {subjectID}"):
                    roi_indices = self.roi_selection.roi_to_index[selected_roi]
                    vertex_matrix_indices = [idx in roi_indices for idx in self.roi_selection.selected_roi_indices] #iterate over all selected indices and put True if the index is in our desired ROI, False otherwise
                    if sum(vertex_matrix_indices) <= 1:
                        #can't do a RDM with one feature
                        representational_similarity[eval_set][subjectID][selected_roi] = np.nan
                    else:
                        fmri_roi = Y[eval_set][bool_index][:,vertex_matrix_indices]
                        fmri_pred_roi = Y_pred[eval_set][bool_index][:, vertex_matrix_indices]

                        rdm_roi = 1 - np.corrcoef(fmri_roi)
                        rdm_pred_roi = 1 - np.corrcoef(fmri_pred_roi)
                        representational_similarity[eval_set][subjectID][selected_roi] = spearmanr(get_lowertriangular(rdm_roi),
                                                                                                    get_lowertriangular(rdm_pred_roi))[0]
        return representational_similarity

    def rsa_similarity(self,
                       fmri_train_naturalistic,
                       fmri_train_naturalistic_pred,
                       fmri_val_naturalistic,
                       fmri_val_naturalistic_pred,
                       fmri_test_naturalistic,
                       fmri_test_naturalistic_pred,
                       fmri_test_artificial,
                       fmri_test_artificial_pred,
                       is_torch: bool=False):
        #first check if the adjacency matrix was made with a radius of 0 or not
        if self.wb_adjacency_dict or len(self.wb_adjacency_dict[self.roi_selection.selected_roi_indices[0]]) <= 1:
            raise ValueError("For RSA similarity metric the adjacency matrix must be specified with a radius of 1 or more.")

        #for each index in your selected roi indices, extract the ground truth and predicted fmri responses, create an rdm, and correlate
        rkeys = ['train_naturalistic','val_naturalistic','test_naturalistic', 'test_artificial']
        representational_similarity = {key: {idx: 0 for idx in self.roi_selection.selected_roi_indices} for key in rkeys}
        for vertex in self.roi_selection.selected_roi_indices:
            neighbors = self.wb_adjacency_dict[vertex]
            vertex_matrix_indices = [idx in neighbors for idx in self.roi_selection.selected_roi_indices] #find how these adjacent vertices map to the X and Y train/val/test matrices.
            
            #train
            fmri_train_naturalistic_searchlight = fmri_train_naturalistic[:, vertex_matrix_indices]
            fmri_train_naturalistic_searchlight_pred = fmri_train_naturalistic_pred[:, vertex_matrix_indices]
            if is_torch:
                rdm_train_naturalistic_searchlight = 1 - torch.corrcoef(fmri_train_naturalistic_searchlight)
                rdm_train_naturalistic_searchlight_pred = 1 - torch.corrcoef(fmri_train_naturalistic_searchlight_pred)
            else:
                rdm_train_naturalistic_searchlight = 1 - np.corrcoef(fmri_train_naturalistic_searchlight)
                rdm_train_naturalistic_searchlight_pred = 1 - np.corrcoef(fmri_train_naturalistic_searchlight_pred)

            representational_similarity['train_naturalistic'][vertex] = spearmanr(get_lowertriangular(rdm_train_naturalistic_searchlight, is_torch=True), get_lowertriangular(rdm_train_naturalistic_searchlight_pred, is_torch=True))

            #val
            fmri_val_naturalistic_searchlight = fmri_val_naturalistic[:, vertex_matrix_indices]
            fmri_val_naturalistic_searchlight_pred = fmri_val_naturalistic_pred[:, vertex_matrix_indices]
            if is_torch:
                rdm_val_naturalistic_searchlight = 1 - torch.corrcoef(fmri_val_naturalistic_searchlight)
                rdm_val_naturalistic_searchlight_pred = 1 - torch.corrcoef(fmri_val_naturalistic_searchlight_pred)
            else:
                rdm_val_naturalistic_searchlight = 1 - np.corrcoef(fmri_val_naturalistic_searchlight)
                rdm_val_naturalistic_searchlight_pred = 1 - np.corrcoef(fmri_val_naturalistic_searchlight_pred)

            representational_similarity['val_naturalistic'][vertex] = spearmanr(get_lowertriangular(rdm_val_naturalistic_searchlight, is_torch=True), get_lowertriangular(rdm_val_naturalistic_searchlight_pred, is_torch=True))

            #test naturalistic
            fmri_test_naturalistic_searchlight = fmri_test_naturalistic[:, vertex_matrix_indices]
            fmri_test_naturalistic_searchlight_pred = fmri_test_naturalistic_pred[:, vertex_matrix_indices]
            if is_torch:
                rdm_test_naturalistic_searchlight = 1 - torch.corrcoef(fmri_test_naturalistic_searchlight)
                rdm_test_naturalistic_searchlight_pred = 1 - torch.corrcoef(fmri_test_naturalistic_searchlight_pred)
            else:
                rdm_test_naturalistic_searchlight = 1 - np.corrcoef(fmri_test_naturalistic_searchlight)
                rdm_test_naturalistic_searchlight_pred = 1 - np.corrcoef(fmri_test_naturalistic_searchlight_pred)

            representational_similarity['test_naturalistic'][vertex] = spearmanr(get_lowertriangular(rdm_test_naturalistic_searchlight, is_torch=True), get_lowertriangular(rdm_test_naturalistic_searchlight_pred, is_torch=True))

            #test artificial
            fmri_test_artificial_searchlight = fmri_test_artificial[:, vertex_matrix_indices]
            fmri_test_artificial_searchlight_pred = fmri_test_artificial_pred[:, vertex_matrix_indices]
            if is_torch:
                rdm_test_artificial_searchlight = 1 - torch.corrcoef(fmri_test_artificial_searchlight)
                rdm_test_artificial_searchlight_pred = 1 - torch.corrcoef(fmri_test_artificial_searchlight_pred)
            else:
                rdm_test_artificial_searchlight = 1 - np.corrcoef(fmri_test_artificial_searchlight)
                rdm_test_artificial_searchlight_pred = 1 - np.corrcoef(fmri_test_artificial_searchlight_pred)

            representational_similarity['test_artificial'][vertex] = spearmanr(get_lowertriangular(rdm_test_artificial_searchlight, is_torch=True), get_lowertriangular(rdm_test_artificial_searchlight_pred, is_torch=True))

        return representational_similarity