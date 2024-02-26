import warnings
from pathlib import Path
import copy
import torch
import torch.nn as nn
import yaml
from modules.classifiers import *
from modules.decoders import *
from modules.encoders import *
from modules.SOM import *
from modules.reshapings_rescalings import *
from modules.SOM import *
from modules.AR_modules import *
from my_utils.config import Config
from my_utils.pytorch import tensor2array, prepare_argument_dict
from my_utils.training import experiment_dir

from models.CPC.model_class import CPCModel
from models.SOM_VAE_and_DESOM.model_class import SOM_VAE
from models.SOM_CPC.model_class import SOM_CPC

class FeedForwardModel(nn.Module):
    def __init__(self, config, device):
        super().__init__()

        self.config = config
        self.device = device
        self.aggregate_pred_names = ['out']
        self.build_module_list_from_config()
        
    def build_module_list_from_config(self):

        for module_name in self.config.module_order:
            settings = self.config.get(module_name, {})
            if settings.get('checkpoint_path'):
                module, loaded_config = load_model_from_checkpoint(Path(settings.checkpoint_path), device=self.device, module=module_name)
                warnings.warn(f'The {module_name} module is loaded from a checkpoint and additional settings are ignored.')
                freeze_module = self.config[module_name].get('freeze', False)
                self.config[module_name] = loaded_config.model[module_name]

                # Reload lost information in the config and write again the freeze attribute to the module.
                self.config[module_name]['freeze'] = freeze_module
                module._modules[module_name].freeze = freeze_module
                self.config[module_name]['checkpoint_path'] = str(settings.checkpoint_path)
                if len(list(module.children())) == 1:
                    module = list(module.children())[0]
            else:
                if settings['class'].split('.')[0] == 'nn':
                    settings, module_class = prepare_argument_dict(settings['class'], settings)
                else: module_class = settings['class']
                module = eval(module_class)(**settings, config=self.config)

            self.add_module(module_name, module)

    def forward(self, inp, epoch=None):
        x = inp[0]

        for module in self.children():
            x = module(x, epoch=epoch)
            
        if isinstance(x, dict):
            return x
        else:
            return {'out':x}

    def aggregate_pred(self, y_pred_all_steps, pred, aggregate_pred_names, **kwargs):
        
        if aggregate_pred_names == True: #use the default output variable if a list is not provided.
            aggregate_pred_names = self.aggregate_pred_names
        else:
            aggregate_pred_names = aggregate_pred_names

        if not y_pred_all_steps:
            y_pred_all_steps = {k: [] for k in aggregate_pred_names}

        for out_var in aggregate_pred_names:
            # Only save the last window's prediction in case of an AR model
            if isinstance(pred[out_var], tuple) or isinstance(pred[out_var], list):
                y_pred_all_steps[out_var].append(tensor2array(pred[out_var][-1])) 
            else: # without AR model
                y_pred_all_steps[out_var].append(tensor2array(pred[out_var]))

        return y_pred_all_steps

    def convert_pred(self, y_pred_all_steps):
        """
        y_pred_all_steps (dic): List of np.arrays, these arrays the model predictions for each element in the data set. 

        Returns:
            all_preds (list): List that contains the output of the model for all elements in the data set.
        """
        # Loop over the multiple predictions (i.e. model outputs) that have been aggregated throughout the epoch.
        for out_var, value in y_pred_all_steps.items():

            # Concatenate the predictions of all batches.
            y_pred_all_steps[out_var] = np.concatenate(value)
        return y_pred_all_steps 
        

def load_model_from_checkpoint(checkpoint_path, device, module='all'):
    """ Returns a saved pytorch model and its configuration.

    Args:
        checkpoint_path (str): Absolute path to the checkpoint
        device (str): String indicating the device on which we will run this model.
        module (str, optional): Indicate the module to be loaded.

    Returns:
        torch model, Config 
    """

    config = Config(yaml.load(open(Path(checkpoint_path).parent.parent /
                       'config.yml'), Loader=yaml.FullLoader))
                       
    if not module == 'all':
        full_config = config.deep_copy()
        config = Config({'model':{'type':'sequential', 'module_order':[module]}}) #A partially loaded model must always result in a sequential model.
        for module_name in full_config.model.keys():
            if module_name == module:
                # Fill the new (partial) config with the specific settings of the loaded full config of the module that we need to load.
                config.model[module_name] = full_config.model[module_name]
        strict_loading = False
    else:
        strict_loading = True

    model = build_model_from_config(config, device)
    old_model_params = copy.deepcopy(dict(model.named_parameters()))

    state_dict = torch.load(Path(checkpoint_path), map_location=device)
    model.load_state_dict(state_dict, strict=strict_loading)

    new_model_params = dict(model.named_parameters())
    correctly_loaded = True
    wrongly_loaded = []
    for param_name, old_value in  old_model_params.items():
        if np.allclose(tensor2array(old_value), tensor2array(new_model_params[param_name])):
            # If the module was frozen when training that model, it is okay that the values do not change when the weights are loaded. 
            if new_model_params[param_name].requires_grad and not config.model.get(param_name.split(".")[0]) is None and not config.model[param_name.split(".")[0]].get('freeze', False):
                correctly_loaded = False
                wrongly_loaded.append(param_name)
    assert correctly_loaded, f'The parameters {wrongly_loaded} were possibly not correctly loaded as their values did not change.'

    return model.to(device), config

def build_model_from_config(config, device):
    if config.model.get('checkpoint_path') is not None:
        model, loaded_config = load_model_from_checkpoint(config.model.get('checkpoint_path'), device=device)
        config.model = loaded_config.model

    else:
        if config.model.type.lower() == 'sequential':
            model =  FeedForwardModel(config.model, device=device)
        elif 'cpc'  in config.model.type.lower():
            model = eval(config.model.type)(config.model, config.data.pos_samples, config.data.neg_samples, device=device)

            # Load pretrained modules from indicated checkpoint path
            old_model_params = copy.deepcopy(dict(model.named_parameters()))
            for module, settings in config.model.items():
                if isinstance(settings,Config) and 'checkpoint_path' in settings and settings.checkpoint_path is not None:
                    #loaded_module,_ = load_model_from_checkpoint(Path(settings.checkpoint_path), device, module=module)
                    loaded_state_dict = torch.load(Path(settings.checkpoint_path), map_location=device)
                    for module_in_loaded_state_dict in list(loaded_state_dict):
                        if not module in module_in_loaded_state_dict:
                            if module[-5:] == '_cont' or module[-5:] == '_disc':
                                if module[:-5] in module_in_loaded_state_dict:
                                    # Add an extra entry in the state dict with the name being compatible with the new module name
                                    loaded_state_dict[module + "."+'.'.join(module_in_loaded_state_dict.split(".")[1:])] = loaded_state_dict[module_in_loaded_state_dict]
                                
                            loaded_state_dict.pop(module_in_loaded_state_dict)
                    model.load_state_dict(loaded_state_dict, strict=False)
                    model._modules[module].freeze = settings.get('freeze', False)
                
            new_model_params = dict(model.named_parameters())
            for param_name, old_value in  old_model_params.items():
                if not np.allclose(tensor2array(old_value), tensor2array(new_model_params[param_name])):
                    print(f'{param_name} has been loaded from the given checkpoint')

        else:
            model = eval(config.model.type)(config.model, device=device)

    return model.to(device)



