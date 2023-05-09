import torch.nn as nn
from losses.contrastive_losses import *
from losses.torch_losses import *
from losses.SOM_losses import *


class CompoundedLoss(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.read_losses_from_config()
        
    def train(self, mode=True):
        """Sets the module and the sublosses in training mode."""      
        for sub_loss in self.losses:
            sub_loss.train(mode)
        return super().train(mode)

    def eval(self,):
        """Sets the module in the sublosses eval mode."""      
        for sub_loss in self.losses:
            sub_loss.eval()
        return super().train(mode=False)

    def read_losses_from_config(self):
        self.losses = []
        self.multipliers = []

        for _, settings in self.config.items():
            if settings.multiplier > 0:
                if settings['class'][:3] == 'nn.': # Use the custom wrapper for built-in losses
                    loss_cls = eval(settings['class'][3:]+'_wrapper')(config=settings)
                else:
                    loss_cls = eval(settings['class'])(**settings.get('arguments', {}))
                
                self.losses.append(loss_cls)
                self.multipliers.append(settings['multiplier'])

    def forward(self, input, label, pred, named_parameters, **kwargs):
        total_loss, all_extra_loss_info, sub_loss_dict = 0, {}, {}

        for mult, loss in zip(self.multipliers, self.losses):
            if isinstance(loss, MSELoss_wrapper):
                loss_value, extra_loss_info = loss(input=input, pred=pred, label=label)
            else:
                loss_value, extra_loss_info = loss(pred=pred, label=label, named_parameters=named_parameters)
            total_loss += mult*loss_value
            all_extra_loss_info.update(extra_loss_info)

            # Use the class name if no specific other name is given.
            name = loss.name if hasattr(loss,'name') else loss.__class__.__name__
            sub_loss_dict[name] = loss_value
            all_extra_loss_info.update(sub_loss_dict)

        return total_loss, all_extra_loss_info


def build_loss_from_config(config):
    return CompoundedLoss(config)