import numpy as np
import torch
import copy
from my_utils.pytorch import tensor2array

def one_hot(indices, depth, axis=-1, dtype=np.float32):
    """Returns a one-hot encoded numpy array of indices with equivalent behavior as the tensorflow 2.3 implementation.
    Args:
        indices (scalar or np.ndarray): Array that has to be converted to one-hot format.
        depth (int): Number of classes for the one-hot encoding
        axis (int): Axis on which to apply one-hot encoding. If not given, defaults to the last axis.
    Returns:
        np.ndarray: One-hot encoded array over the requested axis.
    """
    assert axis == -1, 'The one_hot function still has a bug for axis other than -1.'
    if indices is None: return None
        
    if not isinstance(indices, np.ndarray):
        indices = np.array([indices])

    if axis != -1:
        indices = np.moveaxis(indices, axis, -1)

    one_hot_transform = np.eye(depth)
    onehot = one_hot_transform[indices.flatten().astype(np.uint8)]
    onehot_reshaped = onehot.reshape(indices.shape+(depth,))

    return onehot_reshaped.astype(dtype)


def prepare_dict_for_yaml(my_dict):
    if not my_dict: return my_dict 
    
    # Convert all values in the dict to floats rather than numpy arrays or tensors for storage to yml file.
    for name, value in my_dict.items():
        if isinstance(value, dict):
            value = prepare_dict_for_yaml(value)
        else:
            if isinstance(value, torch.Tensor):
                value = tensor2array(value).tolist()
            elif isinstance(value, np.ndarray):
                value = value.tolist()
            
            if isinstance(value, list):
                my_dict[name] = [float(val) for val in value]
            else:
                my_dict[name] = float(value)

    if isinstance(list(my_dict.keys())[0],float):
        new_dict = copy.deepcopy(my_dict)
        for name,value in my_dict.items():
            new_dict[int(name)] = value
            new_dict.pop(name)

        my_dict = new_dict

    return my_dict

