import numpy as np
import torch.nn as nn
import torch

class CombineDims(nn.Module):
    def __init__(self, dims, **kwargs):
        """[summary]

        Args:
            dims (list): List of integers pointing to the dimensions that should be concatenated. The dimensions must be adjacent.
        """
        super().__init__()
        self.dims = dims
        assert np.all(np.array(self.dims) >= 0), 'Only indicate positive dimensions.'

        assert(max(np.diff(self.dims)) == 1), 'The requested dimensions to be combined are not adjacent.'       

    def forward(self, x, **kwargs):
        x_shape = list(x.shape)

        new_dim_size = 1
        for dim in self.dims:
            new_dim_size *= x_shape[dim]
            
        # Place new size at the lowest index to be replaced and remove all higher indices that are now combined.
        x_shape[min(self.dims)] = new_dim_size
        for dim in self.dims[1:]:
            x_shape.pop(dim)

        return x.view(*x_shape)



class SwapDims(nn.Module):
    def __init__(self, dims, **kwargs):
        """[summary]

        Args:
            dims (list): List of length 2, pointing to integer dimensions to be swapped.
        """
        super().__init__()
        self.dims = dims

    def forward(self, x, **kwargs):
        return torch.swapaxes(x, self.dims[0], self.dims[1])


class Crop1d(nn.Module):
    def __init__(self, cropping, dim=-1, **kwargs):
        """
        Args:
            cropping (list, tuple): List or tuple containing the number of samples to crop from the left and right side. 
            dim (int, optional): Dimension where to apply the cropping. Defaults to the last dimension.
        """

        super().__init__()
        self.cropping = cropping
        self.dim = dim

    def forward(self, x, **kwargs):
        if self.dim == -1:
            return x[...,self.cropping[0]:-self.cropping[-1]]
        else:
            raise NotImplementedError


class ConstantPad1d(torch.nn.ConstantPad1d): #Wrapper that accepts kwargs
    def __init__(self, padding, value, **kwargs):
        super().__init__(padding, value)

    def forward(self,x,**kwargs):
        return super(ConstantPad1d, self).forward(x)

class Unsqueeze(nn.Module):
    def __init__(self, dim,**kwargs):
        super().__init__()
        self.dim = dim
    def forward(self, x, **kwargs):
        return x.unsqueeze(self.dim)

class Squeeze(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x, **kwargs):
        if list(x.shape)[0] == 1: #If the batch_size is 1 simply calling squeeze would squeeze the batch size dimension as well.
            squeeze_dims = []
            for dim, dim_size in enumerate(list(x.shape)):
                if dim > 0 and dim_size == 1:
                    squeeze_dims.append(dim)

            dims_squeezed = 0
            for dim in squeeze_dims:
                x = x.squeeze(dim-dims_squeezed)
                dims_squeezed += 1
            return x
        else:
            return x.squeeze()

class Unsqueeze(nn.Module):
    def __init__(self, dim,**kwargs):
        super().__init__()
        self.dim = dim
    def forward(self, x, **kwargs):
        if isinstance(x,dict): 
            out = x.pop('out')
            x.update({'out':out.unsqueeze(self.dim)})
            return x
        else:
            return x.unsqueeze(self.dim)

class Permute(nn.Module):
    def __init__(self, permutation, **kwargs):
        super().__init__()
        self.permutation = permutation

    def forward(self, x, **kwargs):
        return x.permute(self.permutation)

class SelectLastElement(nn.Module):
    def __init__(self, axis=-1, **kwargs):
        super().__init__()
        self.axis = axis

    def forward(self, x, **kwargs):
        if self.axis == 0:
            return x[-1]
        elif self.axis == -1:
            return x[...,-1]
        else:
            raise NotImplementedError

class Reshape(nn.Module):
    def __init__(self, new_shape,**kwargs):
        super().__init__()
        self.new_shape = new_shape
    def forward(self, x, **kwargs):
        if isinstance(x,dict): 
            out = x.pop('out')
                
            bs = out.shape[0]
            x.update({'out':out.view((bs,) + tuple(self.new_shape))})
            return x
        else:
            bs = x.shape[0]
            return x.view((bs,) + tuple(self.new_shape))
