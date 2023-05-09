import copy
import warnings

def return_layer_setting(setting_variable, idx, n_components):
    """Returns the variable corresponding to the given idx related to a block in the module. 

    Args:
        setting_variable (dict or value): A dict of settings for each block, where each key is a string representation of the block idx.
        idx (int): Integer referring to the index of the component at hand.
        n_components (int): Number of components in the module. This can be layers or blocks within a layer.
    """
    if setting_variable is None:
        return None
    elif isinstance(setting_variable, dict):
        try:
            for key in setting_variable.keys():
                int(key) #Check whether all keys are pointing to layer indices.
            layer_wise_dict = True
        except:
            layer_wise_dict = False

        if layer_wise_dict: # The dict has keys that denote layer or block indices
            if idx >= (n_components):
                warnings.warn(f'The following dict {setting_variable} has more keys than layers/blocks.')
            return setting_variable.get(str(idx), None)
        else: # The dict contains settings for all layers
            return setting_variable
        
    elif isinstance(setting_variable, list):
        assert len(setting_variable) == n_components
        return setting_variable[idx]
    else:
        return setting_variable

def return_full_layers_settings(setting_variables_dict, idx, n_components):
    """ Interprets the dict of settings where each key represent a specific setting.

    Args:
        setting_variables_dict (dict): Each key represents a setting for a layer. The value is a layer-dependent dict (each key is a layer index), a list or a value.
        idx (int): Index of the current block or layer.
        n_components (int): Total number of blocks or layers.

    Returns:
        dict: Dictionary containing the setting keys and their corresponding values for the given index idx.
    """
    
    settings_dict = copy.deepcopy(setting_variables_dict)

    remove_keys = []
    for key, value in settings_dict.items():
        value = return_layer_setting(value, idx, n_components)
        if value is not None:
            settings_dict[key] = value
        else:
            remove_keys.append(key)

    for key in remove_keys: settings_dict.pop(key)
    return settings_dict

