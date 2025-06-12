from dotenv import load_dotenv
load_dotenv()
import numpy as np
import torch
from src.utils.helpers import vectorized_correlation, get_fsLR32k_adjacency_matrix
import hcp_utils as hcp
import torch.nn as nn

class MAELossNC:
    """
    Scale loss by the noisceiling
    """
    def __init__(self):
        pass 
    def __call__(self, predicted, target, noiseceiling):
        l = noiseceiling * np.abs(target-predicted)
        return l.mean()
    
class MAELoss:
    """
    Typical MAE loss, not scaled by a noiseceiling
    """
    def __init__(self):
        pass 
    def __call__(self, predicted, target):
        l = np.abs(target-predicted)
        return l.mean()

class MSELossNC:
    """
    Scale loss by the noiseceiling
    """
    def __init__(self, eps=1e-3):
        self.eps = eps #dont divide by zero
    def __call__(self, predicted, target, noiseceiling):
        # Calculate squared error
        mse_loss = (target - predicted) ** 2
        
        # Add epsilon to noiseceiling for numerical stability
        weighted_nc = torch.clamp(noiseceiling, min=self.eps)
        
        # Weighted sum of losses
        weighted_loss = torch.sum(weighted_nc * mse_loss, dim=1)
        
        # Sum of weights with stability
        sum_weights = torch.sum(weighted_nc, dim=1)
        
        # Normalize with more stability
        l = weighted_loss / torch.clamp(sum_weights, min=self.eps)

        #mse_loss =(target-predicted) ** 2
        #l = torch.sum((noiseceiling + self.eps) * mse_loss, dim=1) / (torch.sum(noiseceiling, dim=1) + self.eps)
        return l.mean()
    
class L2weightedNC:
    """
    Loss scaled by noiseceiling squared.
    """
    def __init__(self):
        pass 
    def __call__(self, predicted, target, noiseceiling, floor=0.05):
        nc = torch.clamp(noiseceiling, min=floor)
        numerator = torch.sum(nc * ((target-predicted) ** 2))
        denominator = torch.sum(nc)
        return numerator/denominator
    
class VoxelwiseEncodingLoss:
    """
    voxelwise encoding correlation between predicted and target.
    Correlation is computed at each vertex between the predicted vector (batch_size,)
    and target vector (batch_size,)
    """
    def __init__(self):
        pass 
    def __call__(self, predicted, target, noiseceiling, floor=1e-9, min_batch_size=6):
        nbatch = target.shape[0]
        if nbatch < min_batch_size:
            raise ValueError(f"Batch size must be a minimum of {min_batch_size} for stable correlations. Your batch size was {nbatch}.")
        nc = torch.clamp(noiseceiling, min=floor)
        correlation_noise_normalized = (vectorized_correlation(predicted, target, ddof=1, is_torch=True)**2)/nc #correlation squared is variance, then divided by variance to be explained
        batch_distance = 1 - torch.clamp(correlation_noise_normalized, max=1)
        return torch.mean(batch_distance)

class L2weightedBatchNC:
    """
    Loss scaled by batch's noiseceiling squared. Follow 
    https://www.nature.com/articles/s41467-023-38674-4 equation 5
    """
    def __init__(self):
        pass 
    def __call__(self, predicted, target, floor=0.05, min_batch_size=6):
        nbatch = target.shape[0]

        #within batch correlation should be between at least vectors of length 3.
        #should only apply to final batch of validation or inference epochs, since training
        #epochs using this loss function should use a higher batch size and drop last batch.
        if nbatch < min_batch_size:
            batch_nc = torch.ones_like(target) * floor
        else:
            half_batch = nbatch//2
            #batch sizes may not be even
            batch_nc = vectorized_correlation(target[:half_batch,:], target[half_batch:2*half_batch, :], ddof=1, is_torch=True)**2
            batch_nc = torch.clamp(batch_nc, min=floor)
        numerator = torch.sum(batch_nc * ((target-predicted) ** 2))
        denominator = torch.sum(batch_nc)

        return numerator/denominator

class MSELoss(object):
    def __init__(self):
        """
        Initialize the Mean Squared Error cost function. The cost function
        is a measure of how good (or bad) our prediction was, telling 
        us how to update our network.
        """
        return None

    def __call__(self, predicted, target):
        """
        Defines what happens when we give input to an instance of this class.

        Parameters
        ----------
        predicted : numpy array
            the output activations of each of the 10 classes (digits 0-9) 
            of the model for each image. shape (num_classes, num_images)
        target : numpy array
            one-hot matrix of target labels for each image. shape (num_images, num_classes)

        Returns
        -------
        float
            The mean squared error loss of the model predictions

        """
        l = (target-predicted) ** 2
        return l.mean()

    def derivative(self, predicted, target):
        """
        Derivative of the Mean Squared Error Loss. The derivative is used for 
        the calculation of gradients for backpropogation.

        Parameters
        ----------
        predicted : numpy array
            the output activations of each of the 10 classes (digits 0-9) 
            of the model for each image. shape (num_classes, num_images)
        target : numpy array
            one-hot matrix of target labels for each image. shape (num_images, num_classes)

        Returns
        -------
        float
            The derivative of the mean squared error loss of the model predictions
        """

        m = target.shape[1]
        dc_da_curr = -2*(target.T - predicted)/m #MSE for the batch
        return dc_da_curr

class TopographicLoss(torch.nn.Module):
    def __init__(self, ROI_selection, vertex_loss_fcn="L2weightedNC", lambda_smooth=0.1):
        """
        Parameters:
        adjacency_matrix: sparse tensor of shape (n_vertices, n_vertices) 
                         where 1 indicates neighboring vertices
        lambda_smooth: weight for the smoothness term
        """
        super().__init__()
        self.ROI_selection = ROI_selection
        self.desired_vertex_indices = self.ROI_selection.selected_roi_indices
        self.vertex_loss_fcn = vertex_loss_fcn
        if not self.vertex_loss_fcn:
            self.vertex_loss = None
        elif self.vertex_loss_fcn == "L2weightedNC":
            self.vertex_loss = L2weightedNC()
        elif self.vertex_loss_fcn == "L2weightedBatchNC":
            self.vertex_loss = L2weightedBatchNC()
        elif self.vertex_loss_fcn == 'mse':
            self.vertex_loss = nn.MSELoss()
        elif self.vertex_loss_fcn =='VoxelwiseEncodingLoss':
            self.vertex_loss = VoxelwiseEncodingLoss()
        am = hcp.cortical_adjacency
        am = am[self.desired_vertex_indices,:]
        adjacency_matrix = am[:,self.desired_vertex_indices]
        self.adjacency_sum = adjacency_matrix.sum()
        coo = adjacency_matrix.tocoo()
        indices = torch.LongTensor([coo.row, coo.col])
        values = torch.FloatTensor(coo.data)
        sparse_tensor = torch.sparse_coo_tensor(
            indices, values, size=adjacency_matrix.shape
        )
        self.register_buffer('adjacency', sparse_tensor.unsqueeze(0).coalesce()) #expand it one dimension for broadcasting with batch
        self.lambda_smooth = lambda_smooth
        self.indices = self.adjacency.indices()  # Get row, col indices of nonzero elements
        
    def forward(self, pred, target, noiseceiling=None):
        #loss on vertex results
        if not self.vertex_loss:
            vertex_loss = 0
        if self.vertex_loss_fcn in ["L2weightedNC",'VoxelwiseEncodingLoss']: #require noiseceilings
            vertex_loss = self.vertex_loss(pred, target, noiseceiling) 
        elif self.vertex_loss_fcn in ["mse", "L2weightedBatchNC"]:
            vertex_loss = self.vertex_loss(pred,target)

        diffs = pred[:, self.indices[0]] - pred[:, self.indices[1]]  # Only compute diffs for neighbors
        smooth_loss = diffs.pow(2).mean() # No need to multiply with adjacency since we only got neighbor diffs
        
        return vertex_loss + self.lambda_smooth * smooth_loss
    
class MSETOPOLoss(torch.nn.Module):
    def __init__(self, ROI_selection, lambda_smooth=0.1, device='cpu'):
        """
        Parameters:
        adjacency_matrix: sparse tensor of shape (n_vertices, n_vertices) 
                         where 1 indicates neighboring vertices
        lambda_smooth: weight for the smoothness term
        """
        super().__init__()
        self.ROI_selection = ROI_selection
        self.desired_vertex_indices = self.ROI_selection.selected_roi_indices
        self.device = device
        am = hcp.cortical_adjacency
        am = am[self.desired_vertex_indices,:]
        am = am[:,self.desired_vertex_indices]
        rows, cols = am.nonzero()  # Sparse matrix indices
        self.neighbors = torch.tensor(cols, dtype=torch.long, device=self.device)  # Convert to tensor
        self.src_vertices = torch.tensor(rows, dtype=torch.long, device=self.device)  # Each row i has neighbors in cols
    
        self.lambda_smooth = lambda_smooth
        
    def forward(self, predicted, target, floor=0, min_batch_size=6, noiseceiling=None):
        """
        Vertex Loss scaled by batch's noiseceiling squared. Follow 
        https://www.nature.com/articles/s41467-023-38674-4 equation 5
        """
        ### compute batch-wise reliability
        nbatch = target.shape[0]

        if noiseceiling is None or len(noiseceiling) == 0:
            #within batch correlation should be between at least vectors of length 3.
            #should only apply to final batch of validation or inference epochs, since training
            #epochs using this loss function should use a higher batch size and drop last batch.
            if nbatch < min_batch_size:
                batch_nc = (torch.ones(target.shape[1]) * floor).to(self.device)
            else:
                half_batch = nbatch//2
                #batch sizes may not be even
                batch_nc = vectorized_correlation(target[:half_batch,:], target[half_batch:2*half_batch, :], ddof=1, is_torch=True)**2
                batch_nc = torch.clamp(batch_nc, min=floor)
        else:
            batch_nc = noiseceiling[0,:]

        #vertex loss - enforce predicted vertices to be same value as target
        mse_loss = (target-predicted) ** 2
        if noiseceiling is None:
            vertex_loss = torch.mean(mse_loss)
        else:
            vertex_loss = torch.sum(batch_nc * (mse_loss)) / torch.sum(batch_nc) #weight the vertex loss by the batch-wise reliability of the vertex

        # compute topo loss for predictions
        diff_pred = predicted[:, self.neighbors] - predicted[:, self.src_vertices]
        diff_pred_squared = diff_pred.pow(2)
        weighted_diff_pred = batch_nc[self.src_vertices] * torch.sum(diff_pred_squared, dim=0)
        topo_loss_pred = torch.sum(weighted_diff_pred) / torch.sum(batch_nc)

        # compute topo loss for ground truth
        diff_truth = target[:, self.neighbors] - target[:, self.src_vertices]
        diff_truth_squared = diff_truth.pow(2)
        weighted_diff_truth = batch_nc[self.src_vertices] * torch.sum(diff_truth_squared, dim=0)
        topo_loss_truth = torch.sum(weighted_diff_truth) / torch.sum(batch_nc)

        # relative topographic loss with floor at 0
        topo_loss = torch.clamp(topo_loss_pred - topo_loss_truth, min=0)

        return vertex_loss + self.lambda_smooth * topo_loss


def masked_MSEloss(output, target):
    vec = (output - target)**2
    mask = target > -900.0
    loss = torch.sum(vec[mask])/torch.sum(mask)
    #print(f"predicted min/mean/max: {torch.min(output), torch.mean(output), torch.max(output)}")
    #print(f"target min/mean/max: {torch.min(target), torch.mean(target), torch.max(target)}")

    return loss