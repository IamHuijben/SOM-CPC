import timeit
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

from torch.utils.data import DataLoader
from dataloading.librispeech_dataloader import *
from dataloading.sleep_dataloader import *
from dataloading.toy_dataloaders import *

from losses.build_loss import build_loss_from_config
from models.build_model import build_model_from_config 
from my_utils.config import *
from my_utils.pytorch import freeze_parameters, tensor2array
from my_utils.training import *

from callbacks.training import *
from callbacks.visualizations import *


def trainorval_epoch(epoch, model, data_loader, data_fold, callbacks, device, loss_fnc, aggregate_pred_names=True, optimizer=None):
    """
    aggregate_pred_names: can be a boolean, which indicates to either do or don't aggregate the default prediction indices for a model, or when it is a list of strings, 
                        which indicates which prediction to aggregate.
    """

    # Initialize all losses and prediction variables
    loss_epoch, extra_loss_epoch = 0, {}
    y_all_steps = [0] * len(data_loader)
    output_dict = {}
    if aggregate_pred_names:
        y_pred_all_steps = {} #[0] * len(data_loader)

    # Initialize the model in the right mode.
    assert data_fold in ['train', 'val']
    if data_fold == 'train':
        assert (optimizer is not None)
        model.train()
        loss_fnc.train()
    else:
        model.eval()
        loss_fnc.eval()
    
    for batch_idx, (x, y, *contrastive_samples) in enumerate(data_loader):
                
        if data_fold == 'train':
            for cb in callbacks: cb.on_step_begin(step=batch_idx)

        x = x.to(device, non_blocking=False)
        if y is not None: y = y.to(device, non_blocking=True)

        contrastive_samples = [el.to(device, non_blocking=True) for el in contrastive_samples]
        
        if data_fold == 'train':
            model.zero_grad()
            pred = model((x, *contrastive_samples), epoch=epoch)
        else:
            with torch.no_grad():
                pred = model((x, *contrastive_samples),epoch=epoch)

        if y is not None: y_all_steps[batch_idx] = (tensor2array(y))

        if aggregate_pred_names: 
            y_pred_all_steps = model.aggregate_pred(y_pred_all_steps, pred, aggregate_pred_names=aggregate_pred_names)
     
        if data_fold == 'train':
            loss, extra_loss_info = loss_fnc(input=x, label=y, pred=pred, named_parameters = dict(model.named_parameters()))
        else:
            with torch.no_grad():
                loss, extra_loss_info = loss_fnc(input=x, label=y, pred=pred, named_parameters = dict(model.named_parameters()))

        loss_epoch += loss
        for key,value in extra_loss_info.items():
            if key not in extra_loss_epoch.keys():
                extra_loss_epoch[key] = value
            else:
                extra_loss_epoch[key] += value

        if data_fold == 'train':
            # Backpropagation
            (loss/len(x)).backward()
            optimizer.step()
        else: 
            for cb in callbacks: cb.on_step_end(step=batch_idx)

    # Store targets
    if y is not None: 
        output_dict.update({'target': np.concatenate(y_all_steps,0)})

    # Store predictions
    if aggregate_pred_names:    
        all_preds = model.convert_pred(y_pred_all_steps)
        output_dict.update(all_preds)
                      
    # Compute final loss
    loss_epoch /= len(data_loader.dataset)
    for key in extra_loss_epoch.keys():
        extra_loss_epoch[key] /= len(data_loader.dataset)

    return loss_epoch, extra_loss_epoch, output_dict


def run_experiment(configuration=None, **kwargs):
    config = Config(configuration)
    set_random_seed(config.training.random_seed)
   
    "=============== SET GENERAL SETTINGS =============== "
    device = set_device_settings(cuda=config.device.get('cuda', True), gpu_idx=config.device.get('gpu_idx'))
    
    "=============== MODEL DEFINITION =============== "
    model = build_model_from_config(config, device=device)
    print(model)
    print('Model initialized!')

    save_dir = Path(config.get('logging'))
    if save_dir is not None:
        config.save_to_yaml(save_dir / 'config.yml')
    
    "=============== LOSS & METRICS =============== "
    loss_fnc = build_loss_from_config(config.losses)

    # Create a metrics dict that keeps track of all losses and metrics
    setattr(model, 'metrics_dict', {})
    model.metrics_dict['loss'] = {'train': np.zeros((config.training.n_epochs,))*np.nan}
    for loss, settings in config.losses.items():
        if settings.multiplier > 0:
            model.metrics_dict[loss] = {'train': np.zeros((config.training.n_epochs,))*np.nan}
    
    print('Losses and metrics initialized!')
    "=============== OPTIMIZER =============== "
    
    optim_name = config.optimizer.type
    lr = config.optimizer.get('lr', 0.001)
    opt = eval('optim.'+optim_name)

    freeze_parameters(model, config)
    optimizer = opt(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    print('Optimizer initialized!')
    "=============== DATA LOADING ==============="
    data_set_train = eval(config.data['class'])(config.data, 'train', seed=config.training.random_seed)  
    data_loader_train = DataLoader(data_set_train,
                                        batch_size=config.training.get(
                                            'train_batch_size'),
                                        shuffle= config.data.get('data_shuffle', True),
                                        drop_last=False,
                                        pin_memory=True,
                                        num_workers= 0,
                            )
    print('Train loader initialized!')

    data_set_val = eval(config.data['class'])(config.data, 'val', seed=config.training.random_seed)
    len_val = len(data_set_val)
    if len_val > 0:
        model.metrics_dict['loss']['val'] = np.zeros((config.training.n_epochs,))*np.nan
        for loss, settings in config.losses.items():
            if settings.multiplier > 0:
                model.metrics_dict[loss]['val'] = np.zeros((config.training.n_epochs,))*np.nan
        data_loader_val = DataLoader(data_set_val,
                                          batch_size=config.training.get(
                                              'val_batch_size', len_val),
                                        shuffle=False,
                                        drop_last=False,
                                        pin_memory=True,
                                        )
        
        print('Validation loader initialized!')
    
    "=============== CALLBACKS & HOOKS ==============="
    active_cbs = {cb:settings for cb, settings in config.callbacks.items() if settings.every_n_epochs>0}
    callbacks = [eval(cb)(config.training.n_epochs, data_class = data_set_train.__dict__, log_dir=save_dir, **settings) for cb, settings in active_cbs.items()]
    plot_metrics_cb = PlotMetrics(config.training.n_epochs, data_class=data_set_train.__dict__, every_n_epochs=config.callbacks.PlotLoss.every_n_epochs, log_dir=save_dir)

    print('Callbacks initialized!')
   
    "=============== TRAINING ==============="
    print('\n========================================> Start training and log in: ', str(save_dir), '\n')
    for cb in callbacks: cb.on_train_begin(state_dict=model.state_dict(), metrics_dict=model.metrics_dict,  model_named_parameters = dict(model.named_parameters()))
    
    for epoch_idx in range(config.training.n_epochs):
        time_start = timeit.default_timer()

        for cb in callbacks: cb.on_epoch_begin(epoch=epoch_idx)

        train_loss_epoch, extra_loss_info_train, train_dict = trainorval_epoch(epoch_idx, model, data_loader_train, 'train', callbacks, device, loss_fnc,config.training.get('aggregate_pred_names',True), optimizer)
        if len(data_set_val) > 0:
            val_loss, extra_loss_info_val, val_dict = trainorval_epoch(epoch_idx, model, data_loader_val, 'val', callbacks, device, loss_fnc, config.training.get('aggregate_pred_names',True))
        else: val_dict, extra_loss_info_val = {}, {}
        
        # Store train and val losses and print results.
        for loss, settings in config.losses.items():
            if settings.multiplier > 0: #Store all sub-losses and the compounded final loss.
                plot_metrics_cb.write_to_metric_dict(epoch_idx, model.metrics_dict, metric_name=loss, value=extra_loss_info_train[loss], data_fold='train')
        plot_metrics_cb.write_to_metric_dict(epoch_idx, model.metrics_dict, metric_name='loss', value=train_loss_epoch, data_fold='train')

        if len(data_set_val) > 0:
            for loss, settings in config.losses.items():
                if settings.multiplier > 0: #Store all sub-losses and the compounded final loss.
                    plot_metrics_cb.write_to_metric_dict(epoch_idx, model.metrics_dict, metric_name=loss, value=extra_loss_info_val[loss], data_fold='val')
            plot_metrics_cb.write_to_metric_dict(epoch_idx, model.metrics_dict, metric_name='loss', value=val_loss, data_fold='val')
            val_print= f' - Val_loss: {np.round(tensor2array(val_loss), 5)}'

        else:
            val_print = ''
        try:
            print(f'Epoch {epoch_idx}: ({np.round(timeit.default_timer()-time_start, 1)} sec.) Loss: {np.round(tensor2array(train_loss_epoch), 5)}'+val_print)
        except:
            print(f'Epoch {epoch_idx}')

        for cb in callbacks:
            cb.on_epoch_end(epoch=epoch_idx, state_dict=model.state_dict(), metrics_dict=model.metrics_dict, train_dict=train_dict, val_dict=val_dict, extra_loss_info=[extra_loss_info_train,extra_loss_info_val],  model_named_parameters = dict(model.named_parameters()))

        # Plot the sub-losses that are stored in the metrics dict.
        for name, settings in config.losses.items():
            if settings.multiplier > 0:
                plot_metrics_cb.on_epoch_end(epoch=epoch_idx, metric=model.metrics_dict[name],name=name)        



