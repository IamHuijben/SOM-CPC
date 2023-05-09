import numpy as np 
import torch
import warnings
import inspect

def tensor2array(tensor):
    """Converts a pytorch tensor to a numpy array

    Args:
        tensor (torch.tensor): A tensor 

    Returns:
        np.ndarray: A numpy array
    """

    if isinstance(tensor, np.ndarray):
        return tensor
    elif isinstance(tensor, list):
        return np.array(tensor)
    if tensor.is_cuda:
        tensor = tensor.clone().to('cpu')
    if tensor.requires_grad:
        tensor = tensor.detach()
    tensor = tensor.numpy()
    return tensor


def freeze_parameters(model, config):
    for param_name, param in model.named_parameters():

        # Check for freezing settings at the highest module level only.
        try:
            if config.model[param_name.split(".")[0]].get('freeze', False):
                param.requires_grad = False
        except:
            warnings.warn(f'no freezing setting for {param_name}')

def prepare_argument_dict(function_name, my_arg_dict):
    """ Removes unnecessary arguments, for functions that cannot take **kwargs

    Args:
        function_name (str): Name of the function.
        my_arg_dict (dict): Dictionary with string keys denoting arguments, and the items denoting their values.

    Returns:
        Dict: Returns a cleaned up dict that can be fed to the function, and the module class that is still needed to call the module.
    """
    
    required_args = inspect.getfullargspec(eval(function_name)).args

    common_args = set(required_args).intersection(set(my_arg_dict.keys()))
    non_used_args = set(my_arg_dict.keys()) - set(required_args)
    
    warnings.warn(f'{non_used_args} were provided as arguments to {function_name}, but they were automatically removed an remain unused.')
    
    new_arg_dict = {}
    for arg in common_args:
        new_arg_dict.update({arg:my_arg_dict[arg]})
        
    return new_arg_dict, my_arg_dict.get('class')