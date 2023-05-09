
import torch
import torch.nn as nn

class ARmodule(nn.Module):
    def __init__(self, input_size, output_size, num_stacked, bias, dropout, bidirectional, freeze=False, type='gru', **kwargs):
        """ create an AR module (either GRU or LSTM) as defined on https://pytorch.org/docs/stable/generated/torch.nn.GRU.html
        Args:
            input_size (int): the dimensionality of the input latent space
            output_size (int): the dimensionality of the output vector
            num_stacked (int): the number of stacked GRUs
            bias (bool): use bias for learnable weights
            dropout (float): define the dropout probability on the output of each GRU layer, except the last
            bidirectional (bool): sets the GRU to be bidirectional
            freeze (bool, optional): If set to True, the model parameters are frozen (in the optimizer), and here the behavior is set to deterministic behavior (eval mode).
            name (str, optional): Name of this module. Defaults to 'encoder'.
        """
        super().__init__()
        self.name = type
        self.freeze = freeze
        assert type.lower() in ['lstm', 'gru']
        self.ar_module = eval('torch.nn.'+type.upper())(input_size=input_size, hidden_size=output_size, num_layers=num_stacked, bias=bias,
                                batch_first=False, dropout=dropout, bidirectional=bidirectional)
        self.ar_module.flatten_parameters()  # This supposedly speeds up learning for RNNs

    def train(self, mode=True):
        """Sets the module in training mode, except when the module is frozen."""
        if self.freeze:
            return super().train(mode=False)
        else:
            return super().train(mode)

    def forward(self, x, h0=None, **kwargs):
        """
        x (torch.Tensor): Tesnor of shape: [seq, batch_size, features]
        h0 (torch.Tensor, optional): Initialization of the hidden state of size [num_stacked*(1+bidirectional), batch_size, output_size]
        """
        
        if h0 is None:
            c_t, _ = self.ar_module(x)  # h0 is not given, and automatically initialized at zero
        else:
            c_t, _ = self.ar_module(x, h0)
            # h0 is given for the predicting GRU which starts off at h_n for the continuous GRU
        return c_t
