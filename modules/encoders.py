import torch
import torch.nn as nn
from my_utils.config import Config
from modules.reshapings_rescalings import *


from modules.utils import return_full_layers_settings, return_layer_setting


class ConvEncoder(nn.Module):
    def __init__(self, dimension, input_dim, channels, stacked_convs, kernel_sizes, strides, paddings, poolings, dilations, bias, batch_norm, dropouts, activation, out_activation, input_padding = {}, freeze=False, last_linear_layer = None, name='encoder', **kwargs):
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
            paddings (dict):  Dictionary containing one dict key per block_idx in which zero-padding in the convolutions should take place. 
            poolings (dict, or single tuple): Dictionary containing one dict key per block_idx in which pooling should take place, indicating the type of pooling, and possibly additional settings for the pooling layer. Defaults to None.
            dilations (dist, or single tuple or int): Denotes the dilation to be applied. If only one tuple or int is provided, the sample is used for all convolutions.
            bias (dict, or bool): If set to True the convolutional layer include a bias. If one bool is given all layer do or don't receive a bias.
            batch_norm (bool): Whether or not to apply batch normalization after each convolutional layer.

            dropouts (dict): Dict with keys denoting indices of the blocks after which dropout should be applied, and items denote the dropout probability.
            activation (dict, or single dict): Dictionary containing the activation types, and possible additional settings for this activation function. If provided a dict each layer can have its own activation. Default to None.
            out_activation (dict, optional): Dictionary containing the output activation type, and possible additional settings for this activation function. Default to None.
            input_padding (dict, optional): Dict containing the type of padding to be applied, and the number of samples to add to the left and right before encoding the data.
            last_linear_layer (None or dict): Dict contains information about the number of output nodes, and whether flatten is needed.
            freeze (bool): If set to True, the model parameters are frozen (in the optimizer), and here the behavior is set to deterministic behavior (eval mode).
            name (str, optional): Name of this module. Defaults to 'encoder'.
        """
        super().__init__()

        assert dimension.lower() in ['1d', '2d', '3d']
        self.dimension = dimension.lower()
        self.input_dim = input_dim
        self.input_channels = input_dim[0] if len(input_dim)==2 else 1
        self.channels = channels
        self.n_blocks = len(self.channels)
        self.stacked_convs = stacked_convs
        self.batchnorm = batch_norm
        self.poolings = poolings
        self.last_linear_layer = last_linear_layer
        self.encode_windows_separately = kwargs.get('encode_windows_separately',False)

        settings = []
        for setting in [kernel_sizes, strides, paddings, dilations, bias]:
            if isinstance(setting, Config):
                settings.append(setting.serialize())
            else:
                settings.append(setting)
        self.conv_dict_settings = dict(zip(['kernel_size', 'stride', 'padding', 'dilation', 'bias'], settings))
        self.conv_fnc = eval('nn.Conv'+dimension)
        if self.batchnorm:
            self.batchnorm_fnc = eval('nn.BatchNorm' + dimension)
        
        self.name = name
        self.freeze = freeze
        
        self.input_padding = input_padding
        if self.input_padding and self.input_padding['class'] is not None:
            self.add_module('input_padding_fnc', eval(self.input_padding['class'])(**self.input_padding))
        
        self.dropouts = dropouts
        self.activation = activation
        self.out_activation = out_activation

        self.normalize_per_window = kwargs.get('normalize_per_window', False)


        self.build_blocks()

    def train(self, mode=True):
        """Sets the module in training mode, except when the module is frozen."""      
        if self.freeze:
            return super().train(mode=False)
        else:
            return super().train(mode)

    def build_blocks(self):

        input_channels = self.input_channels

        for block_idx in range(self.n_blocks):
            block_conv_dict_setting = return_full_layers_settings(
                self.conv_dict_settings, block_idx, self.n_blocks)

            self.add_conv_block(
                idx=block_idx,
                in_dim=input_channels,
                out_dim=self.channels[block_idx],
                **block_conv_dict_setting,
            )
            input_channels = self.channels[block_idx]

            self.add_pooling(block_idx)

            self.add_dropout(block_idx)
        
        if self.last_linear_layer is not None:
            if self.last_linear_layer.first_flatten:
                self.add_module(f'flatten', nn.Flatten()) 
            if self.last_linear_layer.linear_layer: 
                self.add_module(f'linear', nn.Linear(in_features=self.last_linear_layer.input_size , out_features=self.last_linear_layer.output_size))
            if self.last_linear_layer.activation is not None:
                self.add_module(f'last_activation', self.add_activation(-1, act=self.last_linear_layer.activation))


    def add_conv_block(self, idx, in_dim, out_dim, **kwargs):


        for conv_iter in range(self.stacked_convs):
            self.add_module(
                f'block_{idx}_{conv_iter}_conv{self.dimension}',  self.conv_fnc(in_dim, out_dim, **kwargs))

            if self.batchnorm:
                self.add_module(f'block_{idx}_{conv_iter}_batchnorm{self.dimension}', self.batchnorm_fnc(out_dim))  

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
        
    def add_pooling(self, idx):
        """Creates an attribute of a list of dicts or one dict called pooling_fnc, that includes the pooling layer to be applied at the end of each block.
        """
        if self.poolings is None:
            return

        pooling = return_layer_setting(self.poolings, idx, self.n_blocks)
        if pooling is not None:
            pooling_copy = pooling.deep_copy()
            pooling_type = pooling_copy.pop('type')
            if pooling_type is not None:
                pooling_fnc = eval('nn.'+pooling_type+'Pool'+self.dimension)

                self.add_module(
                    f'block_{idx}_pool{self.dimension}', pooling_fnc(**pooling_copy))

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
        # input_dim does not contain batch size.
        assert len(x.shape) == (len(self.input_dim) + 1), f'{x.shape} {self.input_dim}'

        if len(x.shape) == 2: # x is of shape [BS,conv-axis] and misses a channel axis. Happens for hierarchical CPC
            x = x.unsqueeze(1) #[BS,ch, conv-axis]

        # The nr of input samples and channels should be equal to the provided shape 
        if self.encode_windows_separately and (x.shape[-1] > self.input_dim[-1]):
                assert x.shape[-1] % self.input_dim[-1] == 0 #The signal length must be an integer number the expected input size of the encoder.
                nr_windows = int(x.shape[-1] // self.input_dim[-1])

                all_latents = []
                for i in range(nr_windows):
                    out = x[..., i * self.input_dim[-1]:(i + 1) * self.input_dim[-1]]
                
                    if self.normalize_per_window: #Make each window zero-mean and standard deviation=1
                        x_unbiased = out - torch.mean(out, -1).unsqueeze(-1)
                        out = x_unbiased / torch.std(x_unbiased, -1).unsqueeze(-1)
                        
                    for module in self.children():
                        out = module(out)

                    all_latents.append(out)
                all_latents = torch.stack(all_latents, dim=0)   #[nr windows, BS, F]

        else:

            assert x.shape[-1] % self.input_dim[-1] == 0 #The signal length must be an integer number the expected input size of the encoder.
            nr_separate_windows = int(x.shape[-1] // self.input_dim[-1])

            if nr_separate_windows > 1:
                self.input_dim[-1] = x.shape[-1] #Overwrite the input dim of the encoder here to be able to encode all windows together.
            
                if self.normalize_per_window: #Make each window zero-mean and standard deviation=1
                    x_unbiased = x - torch.mean(x, -1).unsqueeze(-1)
                    x = x_unbiased / torch.std(x_unbiased, -1).unsqueeze(-1)

                for module in self.children():
                    x = module(x)
                x = x.squeeze()
                assert x.shape[-1] % nr_separate_windows == 0
            else:
                assert x.shape[-1] == self.input_dim[-1], f'{x.shape} {self.input_dim}'
                if len(self.input_dim) == 2:
                    assert x.shape[1] == self.input_channels == self.input_dim[-2] , f'{x.shape} {self.input_channels} {self.input_dim}'
                elif len(self.input_dim) == 3:
                    assert x.shape[1] == self.input_channels == self.input_dim[-3] , f'{x.shape} {self.input_channels} {self.input_dim}'
                else:
                    raise NotImplementedError

                if self.normalize_per_window: #Make each window zero-mean and standard deviation=1
                    x_unbiased = x - torch.mean(x, -1).unsqueeze(-1)
                    x = x_unbiased / torch.std(x_unbiased, -1).unsqueeze(-1)

                for module in self.children():
                    x = module(x)
            all_latents = x
        return all_latents
        
        

 
class ConvEncoder1D(ConvEncoder):
    def __init__(self, input_dim, channels, stacked_convs, kernel_sizes, strides, paddings, batch_norm=False, poolings=None, dilations=[(1,)], bias=True, dropouts={}, activation={'type':'relu'},  out_activation=None, input_padding = {}, freeze=False, last_linear_layer=None, **kwargs):
        super().__init__('1d', input_dim, channels, stacked_convs, kernel_sizes, strides,
                         paddings, poolings, dilations, bias, batch_norm, dropouts, activation, out_activation, input_padding, freeze, last_linear_layer, **kwargs)
        """ 
        Return a fully parameterized 1D convolutional encoder module.
        """
        