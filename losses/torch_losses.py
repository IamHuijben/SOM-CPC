import torch
import torch.nn as nn
import torch.nn.functional as F
from my_utils.pytorch import tensor2array
import numpy as np


class NLLLoss_wrapper(nn.Module):
    def __init__(self, config, name = 'neg_likelihood', **kwargs):
        super().__init__()
        self.__class__.__name__ = name
        self.config = config
        self.reduction = config.arguments.get('reduction', 'sum')
        self.apply_on_out_vars = config.arguments.get('apply_on_out_vars', 'all')

    def find_class_weighing(self, label, nr_classes):
        """
        Args:
            label (torch.tensor): Scalar label describing the class. 
            nr_classes (int): Number of classes

        Returns:
            torch.tensor: Class weights
        """

        if self.config.get('class_weighing', False) and self.training:
            if isinstance(self.config.get('class_weighing', False), list):
                class_weights = self.config.get('class_weighing')
            else:
                assert isinstance(self.config.get('class_weighing', False), bool)
                counts, _ = np.histogram(tensor2array(label), bins=np.arange(0, nr_classes+1)) 
                class_weights = sum(counts)/np.maximum(1,counts) 
        else: 
            class_weights = torch.ones((nr_classes,)) / nr_classes
        
        #Set the nan labels with a really low weight.
        if self.config.zero_class_nan:
            class_weights[0] = 1e-30 
            nr_classes -= 1

        #make sure that the weights add up to the nr of classs (- the nan class) to ensure that mean and sum reduction result in the same answer.
        class_weights = (class_weights / sum(class_weights)) * nr_classes
        return torch.as_tensor(class_weights, dtype=torch.float32, device=label.device)


    def forward(self, pred, label, **kwargs):
        """
        Args:
            pred (torch.tenosr): Log-softmax one-hot prediction
            label (torch.tensor): One-hot label
        """
        # For now assume only one output variable or one requested variable to apply this loss on.
        if self.apply_on_out_vars == 'all': 
            assert len(pred) == 1
            pred = list(pred.values())[0]
        elif len(self.apply_on_out_vars) == 1: 
            pred = pred[self.apply_on_out_vars[0]]
        else:
            raise NotImplementedError
                        
        nr_classes = label.shape[-1]         
        label = torch.argmax(label, -1)
        weights = self.find_class_weighing(label, nr_classes)
        return F.nll_loss(pred, label, weight=weights, reduction=self.reduction), {} #return an empty dict for the extra loss information.


class MSELoss_wrapper(nn.MSELoss):
    def __init__(self, config, name='mse', **kwargs):
        self.__class__.__name__ = name
        self.name = config.get('name', name)

        self.config = config
        self.final_reduction = config['arguments'].get('reduction', 'none')
        self.apply_on_out_vars = config.get('apply_on_out_vars', 'all')
        
        config['arguments'].pop('reduction')
        super().__init__(reduction='none', **config.get('arguments'))

        self.loss_fnc = super()

    def forward(self, input, pred, label,**kwargs):
        if self.apply_on_out_vars == 'all': 
            self.apply_on_out_vars = list(pred.keys())
        
        loss_wo_reduction = 0
        for out_var in self.apply_on_out_vars:

            #We either need to do cropping, or last-window selection in case of AR modules. 
            if pred[out_var].shape != label.shape:
                label = input

                #We assume that we have an AR module and loaded multiple windows for the label
                if label.shape[-1] > pred[out_var].shape[-1]:
                    label = label[...,(-pred[out_var].shape[-1]):]

                elif pred[out_var].shape[-1] > label.shape[-1]:
                    #We assume that we have an AR module and predicted multiple windows
                    if pred[out_var].shape[-1] % label.shape[-1] == 0: 
                         pred[out_var] = pred[out_var][...,(-label.shape[-1]):]

                    # We assume we have to remove some of the zeropadding from the start again. This hapens for the AE case.
                    else: 
                        remove_samples = np.array(pred[out_var].shape[-1]) - np.array(input.shape[-1])
                        pred[out_var] = pred[out_var][...,int(np.ceil(remove_samples/2)):-int(np.floor(remove_samples/2))]

            loss_wo_reduction += self.loss_fnc.forward(pred[out_var], label)
        
        # Compute reduction manually, because sum and mean reduce also the batch axis in this loss function.
        if loss_wo_reduction.dim() == 4:
            avg_loss_per_batch_elements = torch.mean(loss_wo_reduction,[-1,-2,-3])
        elif loss_wo_reduction.dim() == 3:
            avg_loss_per_batch_elements = torch.mean(loss_wo_reduction,[-1,-2])
        elif loss_wo_reduction.dim() == 2:
            raise NotImplementedError #This should never be the case
        elif loss_wo_reduction.dim() == 1:
            avg_loss_per_batch_elements = loss_wo_reduction
            
        if self.final_reduction == 'sum':
            loss = torch.sum(avg_loss_per_batch_elements,0)
        elif self.final_reduction == 'mean':
            loss = torch.mean(avg_loss_per_batch_elements,0)
        return loss,{}