import torch.nn as nn
import numpy as np
from my_utils.config import Config
from modules.utils import return_full_layers_settings, return_layer_setting
from modules.reshapings_rescalings import *

class ConvDecoder(nn.Module):
    def __init__(self, dimension, input_dim, channels, stacked_convs, kernel_sizes, strides, paddings, dilations, bias, dropouts, upsamplings, activation, out_activation, output_cropping, output_paddings=0, freeze=False, first_linear_layer=None, name='encoder', **kwargs):
        """ Create a convolutional encoder.

        Args:
            dimension (str): Encoder dimension. One from {'1d', '2d', '3d'}.
            input_dim (list or tuple): Contains the input dimensions of the data in the format: [channels, *] (ommitting the batch size). The convolution is applied on the last axis/axes.
            channels (list): List of integers denoting the number of channels at the end of each block.
            stacked_convs (int): Number of stacked convolutional layers in one block.

            Convolutional parameters:
            If a dict is provided, the key should denote the layer index string, and each value corresponds to the setting of that layer.

            kernel_sizes (dict, or single tuple or int): Denotes the sizes of the kernels to be applied. If only one tuple is provided, the same size is used for all convoltuions.
            strides (dict, or single tuple or int):  Denotes the strides to be applied. If only one tuple or int is provided, the same stride is used for all convoltuions.
            paddings (dict, or single tuple or int):  Denotes the padding to be applied. If only one tuple or int is provided, the same padding is used for all convoltuions.
            output_paddings (dict, or single tuple or int):  Denotes the padding to be applied. If only one tuple or int is provided, the same padding is used for all convoltuions.
            dilations (dist, or single tuple or int): Denotes the dilation to be applied. If only one tuple or int is provided, the sample is used for all convolutions.
            bias (dict, or bool): If set to True the convolutional layer include a bias. If one bool is given all layer do or don't receive a bias.
            
            dropouts (dict): Dict with keys denoting indices of the blocks after which dropout should be applied, and items denote the dropout probability.
            upsamplings (dict): Dict with keys denoting indices of the blocks after which upsampling should be applied, and items denote the settings.
            activation (dict, or single dict): Dictionary containing the activation types, and possible additional settings for this activation function. If provided a dict each layer can have its own activation. Default to None.
            out_activation (dict, optional) Dictionary containing the output activation type, and possible additional settings for this activation function. Default to None.
            input_padding (list, optional): List containing the number of samples to crop (from left and right, respectively) at the end of the decoder.
            first_linear_layer (None or Dict): Dict contains information about the number of output nodes, and whether unflatten is needed.
            freeze (bool): If set to True, the model parameters are frozen (in the optimizer), and here the behavior is set to deterministic behavior (eval mode).
            name (str, optional): Name of this module. Defaults to 'encoder'.
        """
        super().__init__()

        assert dimension.lower() in ['1d', '2d', '3d']
        self.dimension = dimension.lower()
        self.input_dim = input_dim
        
        if first_linear_layer is not None:
            self.input_channels = first_linear_layer.unflatten_shape[0]
        else:
            if len(input_dim)==2:
                self.input_channels = input_dim[0]     
            elif len(input_dim) == 1:
                self.input_channels = 1

        self.channels = channels
        self.n_blocks = len(self.channels)
        self.stacked_convs = stacked_convs
        settings = []
        for setting in [kernel_sizes, strides, paddings, dilations, bias, output_paddings]:
            if isinstance(setting, Config):
                settings.append(setting.serialize())
            else:
                settings.append(setting)
        self.conv_dict_settings = dict(zip(['kernel_size', 'stride', 'padding', 'dilation', 'bias', 'output_padding'], settings))
        if kwargs.get('upconvs') is None:
            self.conv_fnc = eval('nn.Conv'+dimension)
            self.conv_dict_settings.pop('output_padding')
        elif kwargs.get('upconvs') == 'single':
            self.conv_fnc = eval('ConvUp'+dimension)
        else:
            self.conv_fnc = eval('DoubleConvUp'+dimension)
        #self.conv_fnc = eval('nn.ConvTranspose'+dimension)
        
        self.name = name
        self.freeze = freeze

        self.dropouts = dropouts
        self.upsamplings = upsamplings
        self.first_linear_layer = first_linear_layer
        self.activation = activation
        self.out_activation = out_activation

        self.build_blocks()

        self.output_cropping = output_cropping
        if self.output_cropping:
            self.add_module('output_crop_fnc', Crop1d(self.output_cropping))
        

    def train(self, mode=True):
        """Sets the module in training mode, except when the module is frozen."""      
        if self.freeze:
            return super().train(mode=False)
        else:
            return super().train(mode)

    def build_blocks(self):

        input_channels = self.input_channels

        if self.first_linear_layer is not None:
            self.add_module(f'squeeze', Squeeze())
            if self.first_linear_layer.linear_layer:
                self.add_module(f'linear', nn.Linear(in_features=self.first_linear_layer.input_size , out_features=self.first_linear_layer.output_size))
            if self.first_linear_layer.activation is not None:
                self.add_module(f'start_activation', self.add_activation(-1, act=self.first_linear_layer.activation))
            if self.first_linear_layer.unflatten:
                self.add_module(f'unflatten', nn.Unflatten(-1, tuple(self.first_linear_layer.unflatten_shape)))
        else:
            self.first_linear_layer = {}


        for block_idx in range(self.n_blocks):
            block_conv_dict_setting = return_full_layers_settings(
                self.conv_dict_settings, block_idx, self.n_blocks)

            self.add_upsampling(idx=block_idx)

            self.add_conv_block(
                idx=block_idx,
                in_dim=input_channels,
                out_dim=self.channels[block_idx],
                **block_conv_dict_setting,
            )
            input_channels = self.channels[block_idx]

            self.add_dropout(block_idx)


    def add_conv_block(self, idx, in_dim, out_dim, **kwargs):


        for conv_iter in range(self.stacked_convs):
            self.add_module(
                f'block_{idx}_{conv_iter}_conv{self.dimension}',  self.conv_fnc(in_dim, out_dim, **kwargs))

            if (idx < (self.n_blocks-1)):
                activation_fnc = self.add_activation(idx)
                if activation_fnc is not None:
                    self.add_module(
                        f'block_{idx}_{conv_iter}_activation', activation_fnc)

            elif (idx == self.n_blocks-1) and (conv_iter == self.stacked_convs-1):
                out_activation_fnc = self.add_activation(idx, out=True)
                if out_activation_fnc is not None:
                    self.add_module(
                        f'block_{idx}_{conv_iter}_activation', out_activation_fnc)

            # The input dimension of the second and further stackings is equal to the output dimension
            in_dim = out_dim


    def add_activation(self, idx, out=False, act=None):
        """Adds an attribute to the module including the activation module to be used after every convolution. 

        Args:
            idx (int): Index of current block.
            out (bool, optional): If set to true, the activation is applied on the output, and it's named accordingly. Defaults to False.
            act (string, optional): If provided, it overwrites the ids of out variable to find an activation.
        """
        if act is None:            
            if out:
                act = self.out_activation
            else:
                act = self.activation
            if (act is None) or (act.get('type', None) is None):
                return None

        activation_fncs = {
            'relu': nn.ReLU(),
            'leakyrelu': nn.LeakyReLU(negative_slope=act.get('negative_slope', 0.01)),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'softmax': nn.Softmax(dim=act.get('dim', -1)),
            'logsoftmax': nn.LogSoftmax(dim=act.get('dim', -1))
        }

        
        if out:
            activation_fnc = activation_fncs.get(
                act['type'].lower(), None)
            assert activation_fnc is not None, 'The requested output activation function ' + \
                act['type'] + ' is not implemented.'
        else:
            activation = return_layer_setting(act, idx, self.n_blocks)
            if activation is not None:
                activation_fnc = activation_fncs.get(
                    activation['type'].lower(), None)
                assert activation_fnc is not None, 'The requested activation function ' + \
                    act['type'] + ' is not implemented.'
            else:
                activation_fnc = None
        return activation_fnc

    def add_upsampling(self, idx):
        """Creates an attribute of a list of dicts or one dict called upsampling_fnc, that includes the upsampling layer to be applied at the end of each block.
        """
        if self.upsamplings is None:
            return

        upsamplings = return_layer_setting(self.upsamplings, idx, self.n_blocks)
        if upsamplings is not None:
            upsamplings_copy = upsamplings.deep_copy()
            upsamplings_type = upsamplings_copy.pop('type')
            if upsamplings_type is not None:
                upsampling_fnc = eval('nn.'+upsamplings_type)

                self.add_module(
                    f'block_{idx}_upsample{self.dimension}', upsampling_fnc(**upsamplings_copy))

    def add_dropout(self, idx):
        if self.dropouts is None:
            return 

        dropout_fncs = {
                '1d': nn.Dropout,
                '2d': nn.Dropout2d,
                '3d': nn.Dropout3d
            }

        dropout = return_layer_setting(self.dropouts, idx, self.n_blocks)
        if dropout is not None:
            self.add_module(
                f'block_{idx}_dropout{self.dimension}', dropout_fncs[self.dimension](dropout))



    def forward(self, x, **kwargs):

        if (len(x.shape) == 2 and len(self.input_dim)==2) and not self.first_linear_layer.unflatten: # x is of shape [BS,conv-axis] and misses a channel axis. 
            if x.shape[1] == self.input_dim[0]: #misses the last axis
                x = x.unsqueeze(-1) #[BS,ch, 1-axis]
            elif x.shape[-1] == self.input_dim[1]: #misses the channel axis
                x = x.unsqueeze(1) #[BS,1, conv-axis]

        elif self.first_linear_layer.get('unflatten'):
            if self.first_linear_layer.get('linear_layer'):
                assert x.shape[1] == self.first_linear_layer.get('input_size')
                assert self.first_linear_layer.get('output_size') == np.prod(self.first_linear_layer.get('unflatten_shape'))
            else:
                assert x.shape[1] == np.prod(self.first_linear_layer.get('unflatten_shape'))
        else:
            assert x.shape[1] == self.input_channels

        # input_dim does not contain batch size.
        assert len(x.shape) == (len(self.input_dim) + 1)
        assert list(x.shape[1:]) == self.input_dim 

        for module in self.children():
            x = module(x)


        return x


class DoubleConvUp1d(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()

        if kwargs.get('stride',1) > 1:
            self.upsample = nn.ConvTranspose1d(in_dim, in_dim, kernel_size=kwargs['stride'], stride=kwargs['stride'], output_padding=kwargs.get('output_padding'))
        else:
            self.upsample = nn.Identity()

        kwargs['stride'] = 1
        kwargs.pop('output_padding', None) #Remove output_padding settings from kwargs if present.
        self.conv_fnc = nn.Conv1d(in_dim, out_dim, **kwargs)

    def forward(self,x,**kwargs):
        x_up = self.upsample(x)
        x_up_conv = self.conv_fnc(x_up)
        return x_up_conv


class ConvUp1d(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()

        if kwargs.get('stride',1) == 1:
            self.conv_fnc = nn.Conv1d(in_dim, out_dim, kernel_size=kwargs['kernel_size'], stride=kwargs['stride'], padding=kwargs['padding'])
        else:
            self.conv_fnc = nn.ConvTranspose1d(in_dim, out_dim, kernel_size=kwargs['kernel_size'], stride=kwargs['stride'], padding=kwargs['padding'], output_padding=kwargs['output_padding'])

    def forward(self,x,**kwargs):
        x_up_conv = self.conv_fnc(x)
        return x_up_conv


class ConvDecoder1D(ConvDecoder):
    def __init__(self, input_dim, channels, stacked_convs, kernel_sizes, strides, paddings, output_paddings=0, dilations=[(1,)], bias=True, dropouts={}, upsamplings={}, activation={'type':'relu'},  out_activation=None, output_cropping = [0,0], freeze=False, first_linear_layer=None, **kwargs):
        super().__init__('1d', input_dim, channels, stacked_convs, kernel_sizes, strides,
                         paddings, dilations, bias, dropouts, upsamplings, activation, out_activation, output_cropping, output_paddings, freeze, first_linear_layer, **kwargs)
        """ 
        Return a fully parameterized 1D convolutional decoder module.
        """
