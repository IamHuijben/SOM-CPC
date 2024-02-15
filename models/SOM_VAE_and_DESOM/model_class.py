import torch
import torch.nn as nn
from modules.AR_modules import *
from modules.encoders import *
from modules.decoders import *
from modules.classifiers import *
from modules.reshapings_rescalings import *
from modules.SOM import *
from my_utils.pytorch import tensor2array



class SOM_VAE(nn.Module):
    def __init__(self, config, device, **kwargs):
        """
        Implements the SOM-VAE model of Fortuin et al. (2019). 
        Alterations can be made via the parameteres to turn it into SOM-VAE-prob (Fortuin et al.) or DESOM (Forest et al., 2021)
        """
        super().__init__()

        self.config = config
        self.device = device
     
        self.encoder = eval(self.config.encoder['class'])(**self.config.encoder)
            
        if self.config.ARmodel.get('class', None):
            self.ARmodel =  eval(self.config.ARmodel['class'])(**self.config.ARmodel)

        self.quantizer = eval(self.config.quantizer['class'])(**self.config.quantizer)

        if self.config.decoder_cont['class'] is not None:
            self.decoder_cont = eval(self.config.decoder_cont['class'])(**self.config.decoder_cont)
            self.decode_past_windows = self.config.decoder_cont.get('decode_past_windows', False)

        if self.config.decoder_disc['class'] is not None:
            self.decoder_disc = eval(self.config.decoder_disc['class'])(**self.config.decoder_disc)
            assert self.decode_past_windows == self.config.decoder_disc.get('decode_past_windows', False)

    def forward(self, inp, epoch,**kwargs):
        """[summary]

        Args:
            inp (list): List of inputs. First entry is the data array of size bs x channels x window_length*nr_windows
            epoch (int): Epoch index during training

        Returns:
            codebook_idxs (tensor): Index of the closest embedding in the  dictionary. Shape: bs x nr_neighbours
            z_cont (tensor): Continuous embedding of size bs x z_dim
            z_disc (tensor): Quantized embedding of size bs x z_dim
            z_disc_neighbours (torch.tensor): Shape: bs x nr_neighbours x z_dim
            codebook_idxs_neighbours (tensor): Index of the closest embedding in the  dictionary for all neighbours. Shape: bs x nr_neighbours
            neighbour_weighing (torch.tensor): Contains epoch-dependent weights for all nodes in the map when using gaussian kernel neighbourhood. 
            distance_matrix (tensor): MSE between last windows continuous latent space z_cont, and all vectors in the codebook. Shape: bs x som_nodes
            x_hat_cont (torch.tensor): Reconstruction from the continuous latent space.
            x_hat_disc (torch.tensor): Reconstruction from the discrete latent space.

        """

        x = inp[0] # [bs, ch, window_length*nr_windows]

        assert x.shape[-1] % self.encoder.input_dim[-1] == 0
        nr_windows = x.shape[-1] //self.encoder.input_dim[-1]
        orig_encoder_input_dim = self.encoder.input_dim.copy() #This setting is overwritten in the encoder whenever it receives multiple windows, so we need to set it back after every encoder call.

        all_z_cont = self.encoder(x).squeeze(-1) #[BS, channels, F*windows]
        all_z_cont_for_decoder = all_z_cont
        self.encoder.input_dim = orig_encoder_input_dim

        if nr_windows > 1: 
            if self.encoder.encode_windows_separately:
                assert all_z_cont.shape[0] == nr_windows
                if len(all_z_cont.shape) == 3:
                    encoding_to_1D = True
                elif len(all_z_cont.shape) == 4:
                    encoding_to_1D = False
                    channels, features = all_z_cont.shape[-2], all_z_cont.shape[-1]
                all_z_cont = torch.flatten(all_z_cont,  start_dim=2, end_dim=-1) #[windows, bs, F'])
            else:
                assert all_z_cont.shape[-1] % nr_windows == 0
                assert len(all_z_cont.shape) == 3, f'shape of all_z_cont is {all_z_cont.shape}'
                if all_z_cont.shape[-1] == nr_windows: #latent space is 1D
                    all_z_cont = all_z_cont.permute((2,0,1)) #[windows, bs, F]
                    encoding_to_1D = True
                else: #[bs, channels, windows*F]
                    all_z_cont = torch.stack(torch.chunk(all_z_cont,nr_windows, -1),0) #[windows, bs, channels, F]
                    channels, features = all_z_cont.shape[-2], all_z_cont.shape[-1]
                    all_z_cont = torch.flatten(all_z_cont, start_dim=2, end_dim=-1) #[windows, bs, F'] with F'= channels*F
                    encoding_to_1D = False


        if hasattr(self, 'ARmodel'):
            assert nr_windows > 1
            # all_z_cont shape: [windows, bs, F']
            c_cont = self.ARmodel(all_z_cont) #[windows, bs, F']

            if encoding_to_1D:
                if self.decode_past_windows:
                    all_z_cont_for_decoder = c_cont.permute((1,2,0)) #[bs,F',windows] 
                else:
                    all_z_cont_for_decoder = c_cont[-1] 
            else:
                if self.decode_past_windows:
                    all_z_cont_for_decoder_r = torch.nn.Unflatten(-1, (channels, features))(c_cont).permute((1,2,0,3)) #[windows, bs, channels, F] --> [bs, channels, windows, F]
                    all_z_cont_for_decoder = torch.flatten(all_z_cont_for_decoder_r, start_dim=1, end_dim=-1) #[bs, channels*windows*F]
                else:
                    all_z_cont_for_decoder = c_cont[-1] 
        else:
            if nr_windows == 1:
                c_cont = [all_z_cont]
                all_z_cont_for_decoder =  all_z_cont
            else:
                c_cont = all_z_cont
                
                if encoding_to_1D:
                    if self.decode_past_windows:
                        all_z_cont_for_decoder = all_z_cont.permute((1,2,0)) #[bs,F',windows] 
                    else:
                        all_z_cont_for_decoder = all_z_cont[-1].unsqueeze(-1) 
                else:
                    if self.decode_past_windows:
                        all_z_cont_for_decoder_r = torch.nn.Unflatten(-1, (channels, features))(all_z_cont).permute((1,2,0,3)) #[windows, bs, channels, F] --> [bs, channels, windows, F]
                        all_z_cont_for_decoder = torch.flatten(all_z_cont_for_decoder_r, start_dim=1, end_dim=-1) #[bs, channels*windows*F]
                    else:
                        all_z_cont_for_decoder = all_z_cont[-1] 
        
        # Only quantize the last one
        quant_output = self.quantizer(c_cont[-1], epoch=epoch)
        codebook_idxs, z_cont, z_disc, codebook_idxs_neighbours, neighbour_weighing, distance_matrix = quant_output['all_codebook_idxs'], quant_output['all_z_cont'], quant_output['all_z_disc'], quant_output['all_codebook_idxs_neighbours'], quant_output.get('neighbour_weighing'), quant_output['distance_matrix']
        
        if self.quantizer.transitions: #also quantize the pen-ultimate window in case of transitions
            quant_output_penultimate = self.quantizer(c_cont[-2], epoch=epoch)
            codebook_idxs_penultimate = quant_output_penultimate['all_codebook_idxs']
            codebook_idxs = torch.stack([codebook_idxs_penultimate, codebook_idxs],0)

        # In case of having multiple latents over time, for the continuous decoder its possible to decode all. This is not possible for the decoder_disc currently.
        if nr_windows == 1 and len(all_z_cont_for_decoder.shape) == 2:
            all_z_cont_for_decoder = all_z_cont_for_decoder.unsqueeze(-1) #in case of having a 1D feature vector after encoding. #[bs, F, 1]
        
        if self.quantizer.transitions:
            all_z_cont = all_z_cont[...,-1].unsqueeze(-1)

        x_hat_cont =  self.decoder_cont(all_z_cont_for_decoder) #[bs, F, input.dim[-1]]

        if hasattr(self, 'decoder_disc'):
            x_hat_disc = self.decoder_disc(all_z_cont_for_decoder)
        else:
            x_hat_disc = None
   
        return dict(zip(self.return_output_var_names(), [codebook_idxs, z_cont, z_disc, codebook_idxs_neighbours, neighbour_weighing, distance_matrix, x_hat_cont, x_hat_disc]))

    def return_output_var_names(self):
        return ['all_codebook_idxs', 'all_z_cont', 'all_z_disc','all_codebook_idxs_neighbours', 'neighbour_weighing', 'distance_matrix', 'x_hat_cont', 'x_hat_disc']

    def aggregate_pred(self, y_pred_all_steps, pred, aggregate_pred_names):
        self.aggregate_pred_names = aggregate_pred_names

        if not y_pred_all_steps:
            y_pred_all_steps = {k: [] for k in self.aggregate_pred_names}

        for out_var in self.aggregate_pred_names:
            # Only save the last window's prediction in case of an AR model
            if isinstance(pred[out_var], tuple) or isinstance(pred[out_var], list):
                y_pred_all_steps[out_var].append(tensor2array(pred[out_var][-1]))
            elif isinstance(pred[out_var], dict): # without AR model
                y_pred_all_steps[out_var].append(tensor2array(pred[out_var]['out']))
            else:
                y_pred_all_steps[out_var].append(tensor2array(pred[out_var]))

        return y_pred_all_steps


    def convert_pred(self, y_pred_all_steps):
        """
        y_pred_all_steps (list): List of lists, each sub-list contains one of the model predictions for each element in the data set. 

        Returns:
            all_preds (list): List that contains the different outputs of the model, as a numpy array concantenated for all elements in the data set.
        """

        # Loop over the multiple predictions (i.e. model outputs) that have been aggregated throughout the epoch.
        for out_var, value in y_pred_all_steps.items():
            if out_var == 'all_codebook_idxs' and self.quantizer.transitions:
                y_pred_all_steps[out_var] = np.concatenate(value,-1)[-1] #only the last window's nodes
            else:
                # Concatenate the predictions of all batches.
                y_pred_all_steps[out_var] = np.concatenate(value) 
        return y_pred_all_steps 
