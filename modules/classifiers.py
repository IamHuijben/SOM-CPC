import torch
import torch.nn as nn
from modules.utils import return_layer_setting


class Classifier(nn.Module):
    def __init__(self, input_channels, output_channels, dropouts = {}, activation = None, out_activation = None, batch_norm=False, bias=True, weight_normalization=False, freeze=False, name='classifier', **kwargs):
        """ Create a (non-)linear classifier. Expected input is of shape [batch_size, channels, data dim]

        Args:
            input_channels (int): Number of input channels. Should coincide with the second dimension of the input.
            output_channels (int or list): Denoting the number of output channels. In case of a list, the length determines the numer of layers with the respective number of output channels for each layer.
            dropouts (dict, optional): Dict with keys denoting indices of the blocks after which dropout should be applied, and items denote the dropout probability. If not provided, no dropout is applied.
            activation (list of dicts, or single dict):  Dictionary containing the activation types, and possible additional settings for this activation function. If provided a dict each layer can have its own activation. Default to None.
            out_activation (dict): Dictionary containing the output activation type, and possible additional settings for this activation function. Default to None.
            batch_norm (bool): Whether to apply batch normalization before the non-linearity.
            bias (list of bools or single bool, optional): If set to True the layers include a bias. If one bool is given all layer do or don't receive a bias.
            weight_normalization (bool, optional): If set to True, the weights of the classifier layer are normalized classw-wise.
            freeze (bool): If set to True, the model parameters are frozen (in the optimizer), and here the behavior is set to deterministic behavior (eval mode).
            name (str, optional): Name of this module. Defaults to 'classifier'.
        """
        super().__init__()
        self.input_channels = input_channels
        
        if isinstance(bias, list) and isinstance(output_channels, list):
            assert len(bias) == len(output_channels)

        if not isinstance(output_channels, list):
            self.output_channels = [output_channels]
        else:
            self.output_channels = output_channels
              

        self.activation = activation
        self.dropouts = dropouts
        self.out_activation = out_activation
        self.bias = bias
        self.weight_normalization = weight_normalization
        self.n_layers = len(self.output_channels)
        self.batchnorm = batch_norm
        if self.batchnorm:
            self.batchnorm_fnc = eval('nn.BatchNorm1d')

        self.name = name
        self.freeze = freeze

        self.build_classifier()

        if self.weight_normalization:
            assert len(self.output_channels) == 1 #Only allow weight normalization if one dense layer is used.
            # Fix the gain per class to be 1.
            self.dense_0.weight_g = torch.nn.Parameter(torch.ones((self.output_channels[0],), device=self.dense_0.weight_v.device), requires_grad=False) 
        

    def train(self, mode=True):
        """Sets the module in training mode, except when the module is frozen."""      
        if self.freeze:
            return super().train(mode=False)
        else:
            return super().train(mode)

    def build_classifier(self):
        in_features = self.input_channels

        for layer_idx, out_features in enumerate(self.output_channels):
            if self.weight_normalization:
                self.add_module(f'dense_{layer_idx}', torch.nn.utils.weight_norm(nn.Linear(
                in_features=in_features, out_features=out_features, bias=return_layer_setting(self.bias, layer_idx, len(self.output_channels))), name='weight', dim=0))
            else:
                self.add_module(f'dense_{layer_idx}', nn.Linear(
                in_features=in_features, out_features=out_features, bias=return_layer_setting(self.bias, layer_idx, len(self.output_channels))))
            in_features = out_features

            
            if self.batchnorm:
                self.add_module(f'batchnorm_{layer_idx}', self.batchnorm_fnc(out_features))  

       

            if layer_idx < (len(self.output_channels)-1):
                activation_fnc = self.add_activation(layer_idx)
                if activation_fnc is not None:
                    self.add_module(
                        f'activation_{layer_idx}', activation_fnc)
                self.add_dropout(layer_idx)

            else:
                out_activation_fnc = self.add_activation(layer_idx, out=True)
                self.add_dropout(layer_idx) #in case of the last layer, place the dropout before the activation.

                if out_activation_fnc is not None:
                    self.add_module(
                        f'activation_{layer_idx}', out_activation_fnc)



    def add_activation(self, idx, out=False):
        """Adds an attribute to the module including the activation module to be used after every convolution. 

        Args:
            idx (int): Index of current block.
            out (bool, optional): If set to true, the activation is applied on the output, and it's named accordingly. Defaults to False.
        """
        if out:
            act = self.out_activation
        else:
            act = self.activation
        if act is None:
            return None

        activation_fncs = {
            'relu': nn.ReLU(),
            'leakyrelu': nn.LeakyReLU(negative_slope=act.get('negative_slope', 0.01)),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'softmax': nn.Softmax(dim=act.get('dim', -1)),
            'logsoftmax': nn.LogSoftmax(dim=act.get('dim', -1)),
        }

        if out:
            activation_fnc = activation_fncs.get(
                act['type'].lower(), None)
            assert activation_fnc is not None, 'The requested output activation function ' + \
                act['type'] + ' is not implemented.'
        else:
            activation = return_layer_setting(act, idx, self.n_layers)
            if activation is not None:
                activation_fnc = activation_fncs.get(
                    activation['type'].lower(), None)
                assert activation_fnc is not None, 'The requested activation function ' + \
                    act['type'] + ' is not implemented.'
            else:
                activation_fnc = None
        return activation_fnc

    def add_dropout(self, idx):
        if self.dropouts is None:
            return 

        dropout = return_layer_setting(self.dropouts, idx, self.n_layers)
        if dropout is not None:
            self.add_module(
                f'dropout1d_{idx}', nn.Dropout(dropout))

    def forward(self,x, **kwargs):
        if isinstance(x, dict): #In case of preceding the decoder with the variational module
            mu = x['mu']
            logvar = x['logvar']
            x = x['out']
        else:
            mu = None

        assert x.shape[-1] == self.input_channels, f'x shape: {x.shape}, input channels: {self.input_channels}'
        
        for module in self.children():
            x = module(x)

        if mu is not None:
            return {'out':x, 'mu':mu, 'logvar':logvar}
        else:
            return x

