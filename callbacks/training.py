
import copy
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import torch
import yaml
from losses.SOM_losses import *
from my_utils.python import prepare_dict_for_yaml
from sklearn.metrics import confusion_matrix
from sklearn.metrics import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_mutual_info_score as nmi_score_adj
import copy
from callbacks.callback import Callback


class PlotMetrics(Callback):
    def __init__(self, nr_epochs, data_class, zero_class_is_nan_class=True, every_n_steps=None, every_n_epochs=None, log_dir=None, apply_on_out_vars=['out'],**kwargs):
        super().__init__(nr_epochs, data_class, every_n_steps, every_n_epochs, log_dir)
        self.callback_type = 'pred_callback' 
        self.zero_class_is_nan_class = zero_class_is_nan_class
        self.apply_on_out_vars = apply_on_out_vars
        
        if log_dir:
            self.log_dir = log_dir / 'metrics'
            (self.log_dir).mkdir(exist_ok=True, parents=False)
        else:
            self.log_dir = log_dir



    def write_to_metric_dict(self, epoch, metrics_dict, metric_name, value, data_fold):
        metric = metrics_dict.get(metric_name, {})
        if not data_fold in metric.keys():
            metric[data_fold] = np.zeros((self.nr_epochs,))*np.nan 
        metric[data_fold][epoch] = value 

        if not metric_name in metrics_dict.keys():
            metrics_dict[metric_name] = metric

        return metrics_dict

    def on_epoch_end(self, epoch, metric, name, **kwargs):
        """[summary]

        Args:
            epoch (int): Epoch index
            metric (dict or list): Dict contains for train and val set the specific metric. If list is given, multiple metrics are provided and are plotted in a subplot.
            name (str or list): Name of the metric(s). First name in list will be used to name the saved subplot.
        """
        if self.check_epoch(epoch):    
            super().on_epoch_end(epoch)

            if not isinstance(name, list):
                name = [name]
                assert not isinstance(metric, list)
                metric = [metric]

            f, axs = plt.subplots(1,len(name),figsize=(6.4*len(name),4.8))
            if len(name) == 1:
                axs = [axs]
            plt.subplots_adjust(wspace=0.3) 
            
            for idx, (met_name, met) in enumerate(zip(name, metric)):           
                marker = 'o' if epoch == 0 else None
                
                axs[idx].grid()
                for data_fold, vals in met.items():
                    non_nan_vals = vals[:epoch+1][~np.isnan(vals[:epoch+1])]
                    non_nan_epochs = np.arange(1, epoch+2)[~np.isnan(vals[:epoch+1])]
                    axs[idx].plot(non_nan_epochs, non_nan_vals, marker=marker)
                axs[idx].set_xlabel('Epoch')
                axs[idx].set_ylabel(met_name)
                axs[idx].legend(list(met.keys()))

            if self.log_dir is not None:
                try:
                    plt.savefig(self.log_dir/ (name[0] +'.png'), bbox_inches='tight')
                except Exception as e:
                    print(e)
            plt.close(f)

    def at_inference(self, metrics_dict, data_fold, **kwargs):
        # Remove the nested structure where each data fold has its own dict, as during inference one metric dict is made per data fold.
        for name, metric in metrics_dict.items():
            if isinstance(metric[data_fold], list) or isinstance(metric[data_fold], tuple):
                metrics_dict[name] = metric[data_fold][0]
            else:
                metrics_dict[name] = metric[data_fold]
        return prepare_dict_for_yaml(metrics_dict)

    def on_train_end(self, metrics_dict, **kwargs):
        with open(self.log_dir / 'all_metrics_training.yml', 'w') as outfile:
                yaml.dump(prepare_dict_for_yaml(metrics_dict), outfile, default_flow_style=False)

class PlotLoss(PlotMetrics):
    def __init__(self, nr_epochs, data_class, every_n_steps=None, every_n_epochs=None, log_dir=None, **kwargs):
        super().__init__(nr_epochs=nr_epochs, data_class=data_class, every_n_steps=every_n_steps, every_n_epochs=every_n_epochs, log_dir=log_dir,**kwargs)

    def on_epoch_end(self, epoch, metrics_dict, **kwargs):
        if self.check_epoch(epoch):
            super().on_epoch_end(epoch, metric=metrics_dict['loss'], name='loss')

    def on_train_end(self,metrics_dict):
        super().on_train_end(metrics_dict)

    def at_inference(self, loss, data_fold, **kwargs):
        return super().at_inference({'loss': {data_fold: loss}}, data_fold)


class PlotAccuracy(PlotMetrics):
    def __init__(self, nr_epochs, data_class, every_n_steps=None, every_n_epochs=None, log_dir=None, apply_on_out_vars=['out'],**kwargs):
        super().__init__(nr_epochs=nr_epochs, data_class=data_class, every_n_steps=every_n_steps, every_n_epochs=every_n_epochs, log_dir=log_dir,apply_on_out_vars=apply_on_out_vars,**kwargs)
        assert len(apply_on_out_vars) == 1 # For now just assume one output variable.

    def compute_accuracies(self, epoch, metrics_dict, output_dict, data_fold):

        for out_var in self.apply_on_out_vars:
            scalar_label, scalar_pred = self.convert_onehot_to_scalar(output_dict['target'], output_dict[out_var])

            self.compute_accuracy_per_class(epoch, metrics_dict, scalar_label, scalar_pred, data_fold)
            self.compute_overall_accuracy(epoch, metrics_dict, scalar_label, scalar_pred, data_fold)
            
    def compute_accuracy_per_class(self, epoch, metrics_dict, scalar_label, scalar_pred, data_fold):
        cm_perc = confusion_matrix(scalar_label, scalar_pred, labels=np.arange(self.nr_classes), normalize='true')
        if self.zero_class_is_nan_class:
            averaged_acc_per_class = np.mean(np.sum(cm_perc * np.eye(self.nr_classes), -1)[1:]) # Remove the nan class for taking the mean.
        else:
            averaged_acc_per_class = np.mean(np.sum(cm_perc * np.eye(self.nr_classes), -1)) 
        self.write_to_metric_dict(epoch, metrics_dict, 'class_balanced_acc', averaged_acc_per_class, data_fold) 

    def compute_overall_accuracy(self, epoch, metrics_dict, scalar_label, scalar_pred, data_fold):
        total_acc = sklearn.metrics.accuracy_score(scalar_label, scalar_pred, normalize=True)            
        self.write_to_metric_dict(epoch, metrics_dict, 'accuracy', total_acc, data_fold) 

    def on_epoch_end(self, epoch, metrics_dict, train_dict, val_dict, **kwargs):
        
        self.nr_classes = train_dict['target'].shape[-1]
        data_folds = ['train', 'val'] if val_dict else ['train']
        for data_fold in data_folds:
            self.compute_accuracies(epoch, metrics_dict, eval(data_fold+'_dict'), data_fold)
        
        if self.check_epoch(epoch):    
            super().on_epoch_end(epoch, metric=[metrics_dict['accuracy'], metrics_dict['class_balanced_acc']], name=['accuracy', 'class balanced accuracy'])
    
    def on_train_end(self,metrics_dict):
        super().on_train_end(metrics_dict)

    def at_inference(self, pred, label, data_fold, **kwargs):
        metrics_dict = {}
        self.nr_classes = label.shape[-1]
        if isinstance(pred, dict):
            output_dict = {}
            output_dict.update(pred)
            output_dict.update({'target':label})
        else:
            output_dict = {self.apply_on_out_vars[0]:pred, 'target':label}
        self.compute_accuracies(epoch = 0, metrics_dict=metrics_dict, output_dict=output_dict, data_fold=data_fold)
        return super().at_inference(metrics_dict, data_fold)

class PlotCohensKappa(PlotMetrics):
    def __init__(self, nr_epochs, data_class, every_n_steps=None, every_n_epochs=None, log_dir=None, apply_on_out_vars=['out'], **kwargs):
        super().__init__(nr_epochs=nr_epochs, data_class=data_class, every_n_steps=every_n_steps, every_n_epochs=every_n_epochs, log_dir=log_dir, apply_on_out_vars=apply_on_out_vars,**kwargs)
        assert len(apply_on_out_vars) == 1 # For now just assume one output variable.

    def compute_cohens_kappa(self, epoch,  metrics_dict, output_dict, data_fold):

        for out_var in self.apply_on_out_vars:

            scalar_label, scalar_pred = self.convert_onehot_to_scalar(output_dict['target'], output_dict[out_var])
            cohens_kappa = sklearn.metrics.cohen_kappa_score(scalar_label, scalar_pred)
            self.write_to_metric_dict(epoch, metrics_dict, 'cohens_kappa', cohens_kappa, data_fold) 

    def on_epoch_end(self, epoch, metrics_dict, train_dict, val_dict, **kwargs):
        data_folds = ['train', 'val'] if val_dict else ['train']
        for data_fold in data_folds:
            self.compute_cohens_kappa(epoch, metrics_dict, eval(data_fold+'_dict'), data_fold)

        if self.check_epoch(epoch):
            super().on_epoch_end(epoch, metric=metrics_dict['cohens_kappa'], name='cohens_kappa')

    def at_inference(self, pred, label, data_fold, **kwargs):
        metrics_dict = {}
        if isinstance(pred, dict):
            output_dict = {}
            output_dict.update(pred)
            output_dict.update({'target':label})
        else:
            output_dict = {self.apply_on_out_vars[0]:pred, 'target':label}
        self.compute_cohens_kappa(epoch=0, metrics_dict=metrics_dict, output_dict=output_dict, data_fold=data_fold)
        return super().at_inference(metrics_dict, data_fold)
    
    def on_train_end(self,metrics_dict):
        super().on_train_end(metrics_dict)
           

class PlotNMI(PlotMetrics):
    def __init__(self, nr_epochs, data_class, every_n_steps=None, every_n_epochs=None, log_dir=None, apply_on_out_vars=['out'], **kwargs):
        super().__init__(nr_epochs=nr_epochs, data_class=data_class, every_n_steps=every_n_steps, every_n_epochs=every_n_epochs, log_dir=log_dir,apply_on_out_vars=apply_on_out_vars,**kwargs)

    def compute_NMIs(self, epoch, metrics_dict, output_dict, data_fold):

        for out_var in self.apply_on_out_vars:
            if output_dict.get(out_var) is not None:
                scalar_label, scalar_pred = self.convert_onehot_to_scalar(output_dict['target'], output_dict[out_var])

                self.compute_NMI(epoch, metrics_dict, scalar_label, scalar_pred, data_fold)
                self.compute_ajusted_NMI(epoch, metrics_dict, scalar_label, scalar_pred, data_fold)
                continue
            
    def compute_NMI(self, epoch, metrics_dict, scalar_label, scalar_pred, data_fold):
        nmi = nmi_score(scalar_label, scalar_pred)
        self.write_to_metric_dict(epoch, metrics_dict, 'NMI', nmi, data_fold) 

    def compute_ajusted_NMI(self, epoch, metrics_dict, scalar_label, scalar_pred, data_fold):
        nmi_adj = nmi_score_adj(scalar_label, scalar_pred)
        self.write_to_metric_dict(epoch, metrics_dict, 'adjusted_NMI', nmi_adj, data_fold) 

    def on_epoch_end(self, epoch, metrics_dict, train_dict, val_dict, **kwargs):
        
        self.nr_classes = train_dict['target'].shape[-1]
        data_folds = ['train', 'val'] if val_dict else ['train']
        for data_fold in data_folds:
            self.compute_NMIs(epoch, metrics_dict, eval(data_fold+'_dict'), data_fold)
        
        if self.check_epoch(epoch):    
            super().on_epoch_end(epoch, metric=[metrics_dict['NMI'], metrics_dict['adjusted_NMI']], name=['NMI', 'adjusted_NMI'])
    
    def on_train_end(self,metrics_dict):
        super().on_train_end(metrics_dict)

    def at_inference(self, pred, label, data_fold, **kwargs):
        metrics_dict = {}
        self.nr_classes = label.shape[-1]
        self.compute_accuracies(epoch = 0, metrics_dict=metrics_dict, output_dict={self.apply_on_out_vars[0]:pred[self.apply_on_out_vars[0]], 'target':label}, data_fold=data_fold)
        return super().at_inference(metrics_dict, data_fold)


class SaveModel(Callback):
    def __init__(self, nr_epochs, data_class, every_n_steps=None, every_n_epochs=None, log_dir=None, checking_metric='loss', **kwargs):
        super().__init__(nr_epochs=nr_epochs, data_class=data_class, every_n_steps=every_n_steps, every_n_epochs=every_n_epochs, log_dir=log_dir)

        if log_dir:
            self.log_dir = log_dir / 'checkpoints'
            (self.log_dir).mkdir(exist_ok=False, parents=False)
        else:
            self.log_dir = log_dir

        
        if checking_metric is not None:
            if not isinstance(checking_metric, list):
                self.checking_metric = [checking_metric]
            else:
                self.checking_metric = checking_metric
            self.best_val_loss = [np.inf]*len(self.checking_metric)
        else:
            self.checking_metric = checking_metric    

    def on_train_begin(self, state_dict, metrics_dict, **kwargs):
        super().on_train_begin()
        torch.save(state_dict, self.log_dir / f'model.pt')

    def on_epoch_end(self, state_dict, metrics_dict, epoch, **kwargs):
        if self.check_epoch(epoch):
            super().on_epoch_end(epoch) 

            # only save if <checking_metric> is lowest
            if self.checking_metric is not None and all('val' in metrics_dict[ch_metr].keys() for ch_metr in self.checking_metric):
                for metric_idx, ch_metr in enumerate(self.checking_metric):
                    if metrics_dict[ch_metr]['val'][epoch] < self.best_val_loss[metric_idx]:

                        self.best_val_loss[metric_idx] = metrics_dict[ch_metr]['val'][epoch]

                        print(f'Epoch {epoch}: new best model saved.')
                        torch.save(state_dict, self.log_dir / f'model_{epoch}.pt')

                        # Only save the last values rather than all values
                        metrics_dict_copy = copy.deepcopy(metrics_dict)
                        for metric_name, metric_vals in metrics_dict_copy.items():
                            for data_fold in metrics_dict_copy[metric_name]:
                                metrics_dict_copy[metric_name][data_fold] = metric_vals[data_fold][epoch]

                        with open(self.log_dir / 'metrics_dict.yml', 'w') as outfile:
                            yaml.dump(prepare_dict_for_yaml(metrics_dict_copy), outfile, default_flow_style=False)
                        break

            # If checking metrics is None or in case of no validation data, save every epoch.
            else: 
                if not self.checking_metric is None:
                    print(f'Epoch {epoch}: new best model saved because no validation data is present.')
                torch.save(state_dict, self.log_dir / f'model_{epoch}.pt')

                # Only save the last values rather than all values
                metrics_dict_copy = copy.deepcopy(metrics_dict)
                for metric_name, metric_vals in metrics_dict_copy.items():
                    for data_fold in metrics_dict_copy[metric_name]:

                        metrics_dict_copy[metric_name][data_fold] = metric_vals[data_fold][epoch]


                with open(self.log_dir / 'metrics_dict.yml', 'w') as outfile:
                    yaml.dump(prepare_dict_for_yaml(metrics_dict_copy), outfile, default_flow_style=False)

