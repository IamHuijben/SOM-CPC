from pathlib import Path

import torch
import torch.nn as nn
from my_utils.pytorch import tensor2array
from modules.AR_modules import *
from modules.encoders import *
from modules.decoders import *
from modules.classifiers import *
from modules.reshapings_rescalings import *
from modules.SOM import *

class SOM_CPC(nn.Module):
    def __init__(self, config, pos_samples, neg_samples, device):
        super().__init__()
        """ 
        config (Config): config.data with all settings
        pos_samples (int): Number of positive samples
        neg_samples (int): Number of negative samples
        device (str): Device to push the model to.
        encode_windows_separately (bool): Whether or not each window that will be used for the AR model should be encoded separately. Only set to False if the encoder has a perfect downsampling resulting in exactly 1 feature vector per window (e.g. for Librispeech)
        """

        self.config = config
        self.pos_samples = pos_samples
        self.neg_samples = neg_samples * self.pos_samples # For each positive sample we want a different set of negative samples.
        self.device = device
        self.encode_windows_separately = config.encoder.get('encode_windows_separately',True)
        self.SOMontopof_ARmodule = config.quantizer.get('ontopof_ARmodule', False)

        # Initialize quantizer
        self.quantizer = eval(self.config.quantizer['class'])(**self.config.quantizer)

        # Initialize encoder and AR modules
        self.encoder = eval(self.config.encoder['class'])(**self.config.encoder)
        if self.config.ARmodel.get('class', None):
            self.ARmodel =  eval(self.config.ARmodel['class'])(**self.config.ARmodel)

        # Initialize predictive classifiers
        if self.config.classifier.get('class', None): 
            self.classifier = nn.ModuleList()
            for _ in range(self.pos_samples):
                self.classifier.append(eval(self.config.classifier['class'])(**self.config.classifier))

        # Create a module that combines the batch_size dimension with the number of positive or negative samples dimension.
        self.combine_dims = CombineDims(dims=[0,1])
        self.swap_axis = SwapDims(dims=[0,1])

    def run_quantizer(self,input, epoch):
        quant_output = self.quantizer(input,epoch=epoch) 
        codebook_idx, z, codebook_idxs_neighbours, neighbour_weighing, distance_matrix = quant_output['all_codebook_idxs'], quant_output['all_z_cont'], quant_output['all_codebook_idxs_neighbours'], quant_output.get('neighbour_weighing'), quant_output['distance_matrix']
        
        # Select last window only
        distance_matrix = distance_matrix #.reshape((bs, nr_separate_windows,-1))[:,-1] #[bs, nr_nodes]
        all_codebook_idxs = codebook_idx #.reshape((bs, nr_separate_windows))[:,-1] #[bs] (last window)
        if codebook_idxs_neighbours is not None:
            all_codebook_idxs_neighbours = codebook_idxs_neighbours #.reshape((bs, nr_separate_windows, -1))[:,-1] #[bs, nr_neighbours] (last window)
        else:
            assert neighbour_weighing is not None
            neighbour_weighing = neighbour_weighing #.reshape((bs, nr_separate_windows,*som_dims))[:,-1]
            all_codebook_idxs_neighbours = codebook_idxs_neighbours

        return all_codebook_idxs, all_codebook_idxs_neighbours, neighbour_weighing, z, distance_matrix

    def forward(self, inp, epoch=None, **kwargs):
        """
        The model as input a tuple with the following elements:
        x: torch.Tensor of size: [batch_size, channels, window_size_in_samples]
        neg_samples: torch.Tensor of size: [batch_size, total_num_negative_samples, channels, window_size_in_samples]
        pos_samples: torch.Tensor of size: [batch_size, pos_samples, channels, window_size_in_samples]
        """
        x, neg_samples, pos_samples = inp
                    
        assert x.shape[0] == neg_samples.shape[0] == pos_samples.shape[0], 'The batch_size dimension is different for x, neg_samples and pos_samples.'
        assert x.shape[-2] == neg_samples.shape[-2] == pos_samples.shape[-2], 'The channels dimension is different for x, neg_samples and pos_samples.'

        ### ENCODING ###
        if self.encode_windows_separately or (x.shape[-1] / self.encoder.input_dim[-1] == 1):
            all_z, all_codebook_idxs, all_codebook_idxs_neighbours = [] , [], []
            # If the input signal length is longer than what the encoder expects, it contains multiple windows for the AR module that should be encoded separately.
            if x.shape[-1] > self.encoder.input_dim[-1]:
                assert self.config.get('ARmodel', None) or self.quantizer.transitions
                assert x.shape[-1] % self.encoder.input_dim[-1] == 0 #The signal length must be an integer number the expected input size of the encoder.
                nr_separate_windows = int(x.shape[-1] // self.encoder.input_dim[-1])
            else:
                nr_separate_windows = 1
                assert x.shape[-1] == neg_samples.shape[-1] == pos_samples.shape[-1], 'The signal length dimension is different for x, neg_samples and pos_samples.'

            for i in range(nr_separate_windows):
                x_window = x[..., i * self.encoder.input_dim[-1]:(i + 1) * self.encoder.input_dim[-1]] # x is of shape: [bs, ch, signal_length*nr_separate_windows]
                z = self.encoder(x_window).squeeze(-1) #[bs, F]  
                all_z.append(z) 

            # Store the continuous embeddings of all windows.                  
            all_z = torch.stack(all_z, dim=0) #[windows, bs, F]

        else:
            assert x.shape[-1] % self.encoder.input_dim[-1] == 0 #The signal length must be an integer number the expected input size of the encoder.
            nr_separate_windows = int(x.shape[-1] // self.encoder.input_dim[-1])
            self.encoder.input_dim[-1] = x.shape[-1] #Overwrite the input dim of the encoder here to be able to encode all windows together.
            
            z = self.encoder(x).squeeze(-1) #[bs, F, windows]
            assert z.shape[-1] == nr_separate_windows, 'The dimensions of the data in combination with the downsampling of this encoder is not suitable for encoding multiple windows together.'
            all_z = z.permute((2,0,1)) #[windows, bs, F]

        ### QUANTIZATION ###
        # Perform SOM on last window, and before AR module
        if not self.SOMontopof_ARmodule:
            all_codebook_idxs, all_codebook_idxs_neighbours, neighbour_weighing, all_z, distance_matrix = self.run_quantizer(input=all_z[-1], epoch=epoch)
                
        ### AR MODULE ###
        if hasattr(self, 'ARmodel'):
            assert nr_separate_windows > 1
            c = self.ARmodel(all_z)[-1] #[bs, F]
            if self.SOMontopof_ARmodule:
                all_codebook_idxs, all_codebook_idxs_neighbours, neighbour_weighing, all_z, distance_matrix = self.run_quantizer(input=c, epoch=epoch)
        else:
            if not self.SOMontopof_ARmodule:
                c = all_z
            else: #it still contains the windows dimension
                c = all_z[-1] #[bs, F]

        
        if isinstance(all_z, list):
            all_z = all_z[-1]

        ### Predict the future
        preds = {'z_pred':[]}

        for classif in self.classifier:
            preds[f'z_pred'].append(classif(eval(f'c'))) 
        
        preds[f'z_pred'] = torch.stack(preds[f'z_pred']) #shape: [pos_samples, BS, features]
        preds[f'z_pred'] = self.swap_axis(preds[f'z_pred']) #shape: [BS, pos_samples, features]
        
        ### Compute positive and negative samples:
        contr_res = {'neg_samples_comb':0, 'pos_samples_comb':0, 'z_neg':0, 'z_pos':0, 'codebook_idxs_neg':0, 'codebook_idxs_pos':0} 
        

        # For the contrastive samples, only return the index of the codebook, rather than the actual discrete latents, to save memory. 
        for contr_type in ['pos', 'neg']:
            contr_res[f'{contr_type}_samples_comb'] = self.combine_dims(eval(f'{contr_type}_samples')) #shape: [BS*neg_samples, ch, signal length]
            
            # Encoding     
            # Reset the encoder dimension again to be able to encoder the positive and negative samples separately again.
            if not self.encode_windows_separately or not x.shape[-1] / self.encoder.input_dim[-1] == 1:
                self.encoder.input_dim[-1] = contr_res[f'{contr_type}_samples_comb'].shape[-1]
            contr_res[f'z_{contr_type}'] = self.encoder(contr_res[f'{contr_type}_samples_comb']).squeeze(-1) #shape: [BS*contr_samples, features]

            # Quantizing (not possible for positive and negative samples when applied on top of AR)
            if not self.SOMontopof_ARmodule:    
                quant_output = self.quantizer(contr_res[f'z_{contr_type}'], epoch=epoch, deterministic_mode=True) 
                contr_res[f'codebook_idxs_{contr_type}']  = quant_output['all_codebook_idxs']
                
                # Reshape to shape [batch_size, contr_samples, **dim]
                contr_res[f'codebook_idxs_{contr_type}'] = contr_res[f'codebook_idxs_{contr_type}'].view((neg_samples.shape[0], eval(f'self.{contr_type}_samples'), -1))    
            else:
                contr_res[f'codebook_idxs_{contr_type}'] = None

            # Reshape to shape [batch_size, contr_samples, **dim]
            contr_res[f'z_{contr_type}'] = contr_res[f'z_{contr_type}'].view((neg_samples.shape[0], eval(f'self.{contr_type}_samples'), -1))

        return dict(zip(self.return_output_var_names(), [all_z, preds['z_pred'], contr_res['z_neg'], contr_res['z_pos'], all_codebook_idxs, contr_res['codebook_idxs_neg'], contr_res['codebook_idxs_pos'], all_codebook_idxs_neighbours,neighbour_weighing, distance_matrix]))

    def return_output_var_names(self):
        #  Return both the continuous and discrete variables
        return ['all_z_cont', 'z_pred', 'z_neg', 'z_pos'] + ['all_codebook_idxs', 'codebook_idxs_neg', 'codebook_idxs_pos'] + ['all_codebook_idxs_neighbours','neighbour_weighing']+['distance_matrix']

    def aggregate_pred(self, y_pred_all_steps, pred, aggregate_pred_names, **kwargs):
        """
        Appends the new predictions to the lists in each key of the y_pred_all_steps dictionary.
        """

        if isinstance(aggregate_pred_names, list) and len(aggregate_pred_names) > 0:
            if not y_pred_all_steps:
                y_pred_all_steps = {k: [] for k in aggregate_pred_names}

            for out_var in aggregate_pred_names:
                # Only save the last window's prediction in case of an AR model
                if hasattr(self, 'ARmodel'):
                    # pred[out_var] can be a list of windows, or a tensor in which's case already only the last window is passed.
                    if isinstance(pred[out_var], list): 
                        assert self.encode_windows_separately
                        pred[out_var] = pred[out_var][-1]
                    y_pred_all_steps[out_var].append(pred[out_var])
                    
                # without AR model
                else:
                    y_pred_all_steps[out_var].append(pred[out_var])
        else:
            y_pred_all_steps = {}
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

        