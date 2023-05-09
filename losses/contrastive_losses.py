import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.reshapings_rescalings import SwapDims

# BS: batch_size
# T: time axis
# Z: features in latent space
# k: future prediction_steps
# n: negative_samples
# AR: ARlength


class InfoNCELoss(nn.Module):
    def __init__(self, pos_samples, neg_samples, reduction='sum', apply_on_depths=[], name='infoNCE', **kwargs):
        """ Implements InfoNCEloss from "Representation Learning with Contrastive Predictive Coding, van den Oord et al. (2018).
        Adapted from: 

        Args:
            pos_samples (int): Number of positive samples
            neg_samples (int): Number of negative samles per positive sample.
            reduction (str): Indicates how the loss should be reduced at the end.
            apply_on_depths (list, optional): List with integers indicating to which depth level to apply this loss function. If empty list, it is automatically applied to all depths.
            name (str, optional): [description]. Defaults to 'infoNCE'.
        """
        super().__init__()
        self.__class__.__name__ = name
        self.pos_samples = pos_samples
        self.neg_samples = neg_samples
        self.reduction = reduction
        self.apply_on_depths = apply_on_depths           

        self.CEloss = nn.LogSoftmax(dim=-1)
        self.permute_0_1 = SwapDims(dims=[0,1])
        self.permute_1_2 = SwapDims(dims=[1,2])
        
        self.weighing_prediction_losses = torch.as_tensor(np.ones((self.pos_samples,))/self.pos_samples, dtype=torch.float32)
        
    def get_pos_f(self, z_pred, z_pos):
        """ Computes the projection of the prediction onto the latent space of the positive samples.

        Args:
            z_pred (torch.tensor): Shape: [[batch_size, pos_samples, features],...] of length #depth_levels
            z_pos ((torch.tensor): Shape: [[batch_size,, pos_samples, features],...] of length #depth_levels,

        Returns:
            torch.tensor: A list (of length #depth_levels) with projections of the prediction on the negative samples. Shape of each element in the list: [batch_size, pos_samples,1]
        """

        bs = z_pred.shape[0]

        # First combine the batch size, and the pos_samples dimension before computing the matrix product to prevent a for loop over the positive samples.
        z_pos_r = torch.reshape(z_pos, (bs*self.pos_samples, -1))
        z_pred_r = torch.reshape(z_pred, (bs*self.pos_samples, -1))

        pos_projections = torch.matmul(z_pos_r.unsqueeze(-2), z_pred_r.unsqueeze(-1)).squeeze(-1) #[bs*pos_samples]
        pos_f = torch.reshape(pos_projections, (bs, self.pos_samples,1))
        return pos_f

    def get_neg_f(self, z_pred, z_neg):
        """ Computes the projection of the prediction onto the latent space of the negative samples.

        Args:
            z_pred (torch.tensor): Shape: [[batch_size, pos_samples, features],...] of length #depth_levels
            z_neg (torch.tensor): Shape: [[batch_size,, total_nr_neg_samples, features],...] of length #depth_levels, where total_nr_neg_samples corresponds to pos_samples * nr_of_neg_smaples per future prediction.

        Returns
            torch.tensor: A list (of length #depth_levels) with projections of the prediction on the negative samples. Shape of each element in the list: [batch_size, pos_samples, negative_samples]
        """

        bs = z_pred.shape[0]


        # First combine the batch size, and the pos_samples dimension before computing the matrix product to prevent a for loop over the positive samples.
        z_neg_r = torch.reshape(z_neg, (bs, self.pos_samples, self.neg_samples, -1)) #[BS, pos_samples, neg_samples, feat]
        z_neg_r = torch.transpose(torch.reshape(z_neg_r, (bs*self.pos_samples, self.neg_samples, -1)), -1, -2) #[BS*pos_samples, feat, neg_samples]
        z_pred_r = torch.reshape(z_pred, (bs*self.pos_samples, -1)).unsqueeze(1) #[BS*pos_samples, 1, feat]

        neg_projections = torch.matmul(z_pred_r, z_neg_r).squeeze(1) #[BS*pos_samples, neg_samples]
        neg_f = torch.reshape(neg_projections, (bs, self.pos_samples, self.neg_samples))
        return neg_f

    def compute_loss(self, z_pred, z_neg, z_pos):
        """ Compute the cross-entropy loss for all future predictions and sum these to return the total InfoNCE loss.

        Args:
            z_pred (list): List (of length #depth_levels) containing torch.tensors with predictions of the current data in future steps. Shape of each element in the list: [batch_size, pos_samples, features]
            z_neg (list): Latent spaces of negative samples. Shape: [[batch_size,, total_nr_neg_samples, features],...] of length #depth_levels, where total_nr_neg_samples corresponds to pos_samples * nr_of_neg_smaples per future prediction.
            z_pos (list): Latent spaces of positive samples. Shape: [[batch_size,, pos_samples, features],...] of length #depth_levels,

        Returns:
            torch.tensor, torch.tensor: infoNCE loss, infoNCE loss for every future k predictions separately.
        """
       
        proj_pred_pos =  self.get_pos_f(z_pred, z_pos)  # [BS, k, 1]
        proj_pred_neg = self.get_neg_f(z_pred, z_neg)  #  [BS, k, n]

        results = torch.cat((proj_pred_pos, proj_pred_neg), dim=-1)  # [BS, k, 1+n]

        # Calculate the binary cross-entropy loss for all depths, every step in the future and take only the cross-entropy for the positive prediction
        loss_per_future_step = -self.CEloss(results)[...,0] #[BS, k]

        # Multiply each loss corresponding to a future prediction with its corresponding multiplier and sum all losses.
        total_loss = torch.sum(self.weighing_prediction_losses.to(loss_per_future_step.device) * loss_per_future_step, dim=-1) #[BS]


        if self.reduction == 'sum':
            return torch.sum(total_loss), torch.sum(loss_per_future_step,0) 
        else:
            raise ValueError
        
    def interpret_pred(self, pred, **kwargs):
        z_pred, z_neg, z_pos = pred['z_pred'], pred['z_neg'], pred['z_pos']
        return z_pred, z_neg, z_pos 

    def return_final_loss(self, total_loss, loss_per_future_step):
        return total_loss, {'loss_per_future_step':loss_per_future_step}

    def forward(self, pred, **kwargs):
        """[summary]

        Args:
            pred (dict): Dict where each key is an output variable, that contains aggregated predictions at different depth levels. 

        Returns:
            torch.tensor: [description]
        """
        z_pred, z_neg, z_pos  = self.interpret_pred(pred, **kwargs)
  
        assert z_pred.shape[0] == z_neg.shape[0] == z_pos.shape[0], 'The batch size is not equal for z_pred, negative and positive samples.'
        assert z_pred.shape[-1] == z_neg.shape[-1] == z_pos.shape[-1], f'The number of features is not equal for z_pred, negative and positive samples {z_pred.shape} {z_neg.shape} {z_pos.shape}.'

        assert z_neg.shape[-2] == self.neg_samples * self.pos_samples
        assert z_pos.shape[-2] == self.pos_samples

        total_loss, loss_per_future_step = self.compute_loss(z_pred, z_neg, z_pos) 
        return self.return_final_loss(total_loss, loss_per_future_step) 

