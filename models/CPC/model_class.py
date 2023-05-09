from pathlib import Path

import torch
import torch.nn as nn
from my_utils.pytorch import tensor2array

from modules.AR_modules import *
from modules.encoders import *
from modules.decoders import *
from modules.classifiers import *
from modules.reshapings_rescalings import *

class CPCModel(nn.Module):
    def __init__(self, config, pos_samples, neg_samples, device):
        super().__init__()

        self.config = config
        self.pos_samples = pos_samples
        self.neg_samples = neg_samples * self.pos_samples # For each positive sample we want a different set of negative samples.
        self.device = device
        self.encode_windows_separately = config.encoder.get('encode_windows_separately',True)

        self.encoder = eval(self.config.encoder['class'])(**self.config.encoder)
        if self.config.ARmodel.get('class', None):
            self.ARmodel =  eval(self.config.ARmodel['class'])(**self.config.ARmodel)

        self.classifier = nn.ModuleList()
        for _ in range(self.pos_samples):
            self.classifier.append(eval(self.config.classifier['class'])(**self.config.classifier))

        # Create a module that combines the batch_size dimension with the number of positive or negative samples dimension.
        self.combine_dims = CombineDims(dims=[0,1])
        self.swap_axis = SwapDims(dims=[0,1])

        # Which of the predictions should be aggregated for callbacks?  Only store the z_pred.
        self.aggregate_pred_names = ['z_pred']

    def forward(self, inp, **kwargs):
        x, neg_samples, pos_samples = inp

        assert x.shape[0] == neg_samples.shape[0] == pos_samples.shape[0], 'The batch_size dimension is different for x, neg_samples and pos_samples.'
        assert x.shape[-2] == neg_samples.shape[-2] == pos_samples.shape[-2], 'The channels dimension is different for x, neg_samples and pos_samples.'

        if self.encode_windows_separately or (x.shape[-1] / self.encoder.input_dim[-1] == 1):

            # If the input signal length is longer than what the encoder expects, it contains multiple windows for the AR module.
            if x.shape[-1] > self.encoder.input_dim[-1]:
                assert self.config.get('ARmodel', None)
                assert x.shape[-1] % self.encoder.input_dim[-1] == 0 #The signal length must be an integer number the expected input size of the encoder.
                nr_windows = int(x.shape[-1] // self.encoder.input_dim[-1])

                all_latents = [] 
                for i in range(nr_windows):
                    # x is of shape: [bs, ch, signal_length*nr_windows]
                    all_latents.append(self.encoder(x[..., i * self.encoder.input_dim[-1]:(i + 1) * self.encoder.input_dim[-1]]).squeeze(-1)) #[bs, ch, F]
                z_t = torch.stack(all_latents, dim=0)  #[nr_windows, bs, F]
            else:
                assert x.shape[-1] == neg_samples.shape[-1] == pos_samples.shape[-1], 'The signal length dimension is different for x, neg_samples and pos_samples.'
                z_t = torch.stack([self.encoder(x).squeeze(-1)],dim=0)
        
        else:
            assert x.shape[-1] % self.encoder.input_dim[-1] == 0 #The signal length must be an integer number the expected input size of the encoder.
            assert neg_samples.shape[-1] == pos_samples.shape[-1], 'The signal length dimension is different for neg_samples and pos_samples.'
            nr_separate_windows = int(x.shape[-1] // self.encoder.input_dim[-1])
            self.encoder.input_dim[-1] = x.shape[-1] #Overwrite the input dim of the encoder here to be able to encode all windows together.
            
            z_t = self.encoder(x).squeeze(-1) #[bs, F, windows]
            z_t = z_t.permute((2,0,1)) #[nr_windows, bs, F]
            assert z_t.shape[0] == nr_separate_windows, 'The encoding does not have a downsampling factor that is equal to the size of 1 window.'

        neg_samples_comb = self.combine_dims(neg_samples) #shape: [BS*neg_samples, ch, signal length]
        pos_samples_comb = self.combine_dims(pos_samples)

        # Reset the encoder input dim again to be able to encode the positive and negative samples separately.
        if not self.encode_windows_separately or not x.shape[-1] / self.encoder.input_dim[-1] == 1:
            self.encoder.input_dim[-1] = neg_samples_comb.shape[-1]
        z_neg = self.encoder(neg_samples_comb).squeeze(-1)
        z_pos = self.encoder(pos_samples_comb).squeeze(-1)

        if hasattr(self, 'ARmodel'):
            c_t = self.ARmodel(z_t)[-1]
        else:
            c_t = z_t[-1]

        z_pred = []
        for classif in self.classifier:
            z_pred.append(classif(c_t)) 
            
        z_pred = torch.stack(z_pred) #shape: [pos_samples, BS, features]
        z_pred = self.swap_axis(z_pred) #shape: [BS, pos_samples, features]

        # Reshape the positive and negative samples to shape [batch_size, nr_samples, features]
        z_neg = z_neg.view((list(neg_samples.shape)[0], self.neg_samples, -1))
        z_pos = z_pos.view((list(pos_samples.shape)[0], self.pos_samples, -1))

        return dict(zip(self.return_output_var_names(), [z_pred, z_neg, z_pos]))


    def return_output_var_names(self):
        return ['z_pred', 'z_neg', 'z_pos']

    def aggregate_pred(self, y_pred_all_steps, pred, **kwargs):
        """
        Appends the new predictions to the lists in each key o the y_pred_all_steps dictionary.
        """

        if not y_pred_all_steps:
            y_pred_all_steps = {k: [] for k in self.aggregate_pred_names}

        for out_var in self.aggregate_pred_names:
            # Only save the last window's prediction in case of an AR model
            if isinstance(pred[out_var], tuple) or isinstance(pred[out_var], list):
                y_pred_all_steps[out_var].append(pred[out_var][-1]) ###Add it as a list to indicate that we have only 1 depth level. 
            # without AR model
            else:
                y_pred_all_steps[out_var].append(pred[out_var])

        return y_pred_all_steps

    def convert_pred(self, y_pred_all_steps):
        """
        y_pred_all_steps (list of dict): The list contains the model predictions for each element in the data set. Each dict contains the multiple predictions/outputs of the model. 

        Returns:
            all_preds (dict): Dict that contains the different outputs of the model as keys, and the items as a numpy array concatenated for all elements in the data set.
        """
     
        # Loop over the multiple predictions (i.e. model outputs) that have been aggregated throughout the epoch.
        for out_var, value in y_pred_all_steps.items():
            # value is a list of one output variable over all batches.
            y_pred_all_steps[out_var] = tensor2array(torch.cat(value))
        return y_pred_all_steps 

        