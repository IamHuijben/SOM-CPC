
import matplotlib.pyplot as plt
import numpy as np  
import copy
from callbacks.callback import Callback
from callbacks.training import PlotCohensKappa, PlotAccuracy, PlotMetrics, PlotNMI
from my_utils.pytorch import tensor2array
from my_utils.python import one_hot


import numpy as np
from scipy.stats import mode
from somperf.metrics.internal import *

class SOMmap(Callback):
    def __init__(self, nr_epochs, data_class, every_n_steps=None, every_n_epochs=None, log_dir=None, nr_nodes=100, zero_class_is_nan_class=True, checking_metric = 'loss', **kwargs):
        super().__init__(nr_epochs, data_class, every_n_steps, every_n_epochs, log_dir)
        self.callback_type = 'pred_callback'
        self.nr_nodes = nr_nodes
        self.som_dim_1d = np.uint8(np.sqrt(self.nr_nodes))
        self.zero_class_is_nan_class = zero_class_is_nan_class
        self.checking_metric = checking_metric

        if log_dir:
            self.log_dir = log_dir / 'SOM_maps'
            (self.log_dir).mkdir(exist_ok=True, parents=False) 
        else:
            self.log_dir = log_dir

        self.best_val_loss = np.inf

        if isinstance(self.label_codes, list) or isinstance(self.label_codes, np.ndarray): #otherwise it is np.inf to indicate continuous labels
            self.acc_cb = PlotAccuracy(nr_epochs, data_class, every_n_steps, every_n_epochs, log_dir, apply_on_out_vars=['pred'],zero_class_is_nan_class=zero_class_is_nan_class,**kwargs)
            self.kappa_cb = PlotCohensKappa(nr_epochs, data_class, every_n_steps, every_n_epochs, log_dir, apply_on_out_vars=['pred'],zero_class_is_nan_class=zero_class_is_nan_class,**kwargs)
            self.nmi_cb = PlotNMI(nr_epochs, data_class, every_n_steps, every_n_epochs, log_dir, apply_on_out_vars=['all_selected_nodes', 'all_codebook_idxs'],zero_class_is_nan_class=zero_class_is_nan_class,**kwargs)
            self.plot_metric = PlotMetrics(nr_epochs, data_class, every_n_steps=every_n_steps, every_n_epochs=every_n_epochs, log_dir=log_dir, apply_on_out_vars=['pred'],zero_class_is_nan_class=zero_class_is_nan_class,**kwargs)
                                        
    
    def calculate_som_mode(self, k, labels, continuous_labels=False):
        """
        k (list, np.ndarray) Containing the som node indices for all windows
        labels (list, np.ndarray): Containing the corresponding ground truth labels
        continuous_labels (bool, optional): Whether the labels are continuous or not. Defaults to categorical labels.
        """

        if continuous_labels:
            median_labels = {} 
            for node in range(self.nr_nodes):
                if node in k:
                    median_label_value =  np.median(labels[tensor2array(k) == node])
                else:
                    median_label_value = np.nan
                median_labels.update({node: median_label_value})
            return median_labels
        else:
            # Returns a dict that has the mode label for all nodes. Label indices are as follows: [nan, N1,N2,N3,W,REM]
            node_histograms = self.calculate_som_histograms(k, labels)

            mode_labels, mode_labels_counts = {} , {}
            for node, counts in node_histograms.items():
                if isinstance(counts,list) or isinstance(counts,np.ndarray):
                    mode_labels.update({node:np.argmax(counts)})
                    mode_labels_counts.update({node:np.max(counts)}) #Count how many windows on the node correspond to the majority label.
                elif np.isnan(counts):
                    mode_labels.update({node:np.nan})
                    mode_labels_counts.update({node:np.nan})
                else:
                    raise ValueError

            return node_histograms, mode_labels, mode_labels_counts

    def calculate_purity(self, k, mode_labels_counts):
        """
        Computes the purity between nodes and class assignments. 
        # Compare to https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-clustering-1.html
        From: https://github.com/ratschlab/SOM-VAE/blob/master/som_vae/utils.py
        """
        return np.nansum(list(mode_labels_counts.values())) / len(k) #Do not count the nan nodes. NaN nodes typically only occur at the start of training anyway.

    
    def calculate_som_histograms(self, k, labels):
        # Returns a dict with unnormalized histograms for the labels 
        if len(labels.shape) == 2:
            nr_classes = labels.shape[-1] 
            labels = np.argmax(tensor2array(labels),-1)
        elif len(labels.shape) == 1:
            nr_classes = self.nr_classes
            labels = tensor2array(labels)
        else:
            assert hasattr(self, 'nr_classes')
            nr_classes = self.nr_classes
            labels = tensor2array(labels)

        node_histograms = {} 
        for node in range(self.nr_nodes):
            if node in k:
                counts, _ =  np.histogram(labels[tensor2array(k) == node], bins=np.arange(nr_classes+1), density=False)
            else:
                counts = np.nan
            node_histograms.update({node: counts})
        return node_histograms

    def plot_som(self,data_fold, epoch, label_coloring, fig=None, ax=None, labels=None, plot_alternative_som=False, plot_title=None, **kwargs): 
        """[summary]

        Args:
            data_fold (str): Data fold to be shown.
            epoch (int): Training epoch.
            label_coloring (dict): Class or continuous value per node to be used for coloring.
            labels (optional): 1D numpy array containing the labels assigned to each node or a dict with an entry for each node.
            plot_alternative_som (bool, optional): If set to True, provide plot_title, and do not automatically use vmin and vmax.
        """ 
        # Plot a SOM map on a som_dim x som_dim grid, coloured by its labels 

        k_x = np.arange(self.nr_nodes) // self.som_dim_1d 
        k_y = np.arange(self.nr_nodes) % self.som_dim_1d 

        if fig is None and ax is None:
            fig, ax = plt.subplots(1,1,figsize=(6.4, 4.8))

        if labels is None:
            if isinstance(label_coloring,dict):
                labels = list(label_coloring.values())
            else:
                labels = label_coloring

        if self.nr_classes < np.inf and not plot_alternative_som:
            cmap = self.return_cmap(nr_classes=self.nr_classes, zero_class_is_nan_class=self.zero_class_is_nan_class)
            vmin, vmax = 0, len(self.string_labels)
        elif plot_alternative_som: #in case of infinite classes (i.e .continuous case) and plot_alternative som
            cmap = 'hot_r'
            vmin,vmax = None,None
        else:
            cmap = self.return_cmap(nr_classes=self.nr_classes, zero_class_is_nan_class=self.zero_class_is_nan_class)
            vmin, vmax = 1,59 #np.min(np.array(labels)[~np.isnan(labels)]),np.max(np.array(labels)[~np.isnan(labels)])
        
        cax = ax.scatter(k_x, k_y, c=labels, s=100, cmap=cmap, vmin=vmin, vmax=vmax, zorder=10) # We can only choose two dimensions from the latent space to plot our latent space in this way 
        ax.grid()
        ax.set_axisbelow(True)
        ax.set_xlim([-0.5,self.som_dim_1d-0.5])
        ax.set_ylim([self.som_dim_1d-0.5,-0.5])

        
        if not plot_alternative_som and self.string_labels != np.inf:
            cbar = fig.colorbar(cax, ticks=np.arange(0.5,len(self.string_labels),1)) 
            cbar.ax.set_yticklabels(self.string_labels) 
        else:
            cbar = fig.colorbar(cax) 

        if plot_title is None: 
            assert data_fold is not None
            plot_title =  f'som_mode_{data_fold}_epoch_{epoch:04d}' 
        if self.log_dir:
            save_name = self.log_dir / f'{data_fold.split("_")[0]}' / plot_title
            (save_name.parent).mkdir(exist_ok=True, parents=False) 
            plt.savefig(save_name, bbox_inches='tight')
            plt.close(fig)

        return fig, ax, cbar

    def convert_nodes_to_preds(self, output_dict, mode_labels):
        if 'all_codebook_idxs' in output_dict:
            pred_nodes = tensor2array(output_dict['all_codebook_idxs'])
        elif 'all_sel_nodes' in output_dict:
            pred_nodes = tensor2array(output_dict['all_sel_nodes'])

        # Fill the nans in the dict by majority voting from the neig. These are nodes that were never assigned by the data set used for coloring.
        mode_labels_copy = copy.deepcopy(mode_labels)
        for k,mode_label in mode_labels_copy.items():
            if np.isnan(mode_label):        
                k_x = k // self.som_dim_1d
                k_y = k % self.som_dim_1d

                k_up = np.where(k_y > 0,k-1,k) 
                k_down = np.where(k_y<self.som_dim_1d-1,k+1,k)
                k_left = np.where(k_x >0,k-self.som_dim_1d,k)
                k_right = np.where(k_x<self.som_dim_1d-1,k+self.som_dim_1d,k)

                neighbour_mode = mode([mode_labels[int(k_left)], mode_labels[int(k_down)],mode_labels[int(k_right)],mode_labels[int(k_up)]])[0]
                if np.isnan(neighbour_mode): #If the neighbours are all non-assigned, just use the mode label of the full som at this moment.
                    mode_labels_copy[k] = int(mode(list(mode_labels.values()))[0])
                else: 
                    mode_labels_copy[k] = int(neighbour_mode)
        preds = np.stack([mode_labels_copy[idx] for idx in pred_nodes])
        return preds

    def on_epoch_end(self, epoch, metrics_dict, train_dict, val_dict, model_named_parameters, **kwargs):
        if not (isinstance(self.label_codes, list) or isinstance(self.label_codes, np.ndarray)) and self.label_codes == np.inf:  #continuous labels for synthetic case
            self.nr_classes = np.inf
            label_coloring_train = self.calculate_som_mode(train_dict.get('all_selected_nodes', train_dict.get('all_codebook_idxs')), train_dict['target'], continuous_labels=True)
            label_coloring_val = self.calculate_som_mode(val_dict.get('all_selected_nodes', val_dict.get('all_codebook_idxs')), val_dict['target'],continuous_labels=True) 

        else:
            self.nr_classes =  len(self.string_labels) 
            assert train_dict['target'].shape[-1] == self.nr_classes

            # Compute some classification metrics on the som.
            _, label_coloring_train, mode_labels_counts_train = self.calculate_som_mode(train_dict.get('all_selected_nodes', train_dict.get('all_codebook_idxs')), train_dict['target']) #Run this on the training set first to make sure the mode labels from the training set are used for the metrics.
            _, label_coloring_val, mode_labels_counts_val = self.calculate_som_mode(val_dict.get('all_selected_nodes', val_dict.get('all_codebook_idxs')), val_dict['target']) #Run this on the training set first to make sure the mode labels from the training set are used for the metrics.

            for data_fold in ['train', 'val']:
                preds = self.convert_nodes_to_preds(eval(data_fold+'_dict'), label_coloring_train) # Always use the train mode labels for converting the predictions of both sets.
                eval(data_fold+'_dict')['pred'] = one_hot(preds, self.nr_classes)

                # Purity is computed for each set separately. So coloring with the same set as evaluating the purity.
                purity = self.calculate_purity(eval(data_fold+'_dict').get('all_selected_nodes', eval(data_fold+'_dict').get('all_codebook_idxs')), eval(f'mode_labels_counts_{data_fold}'))
                self.plot_metric.write_to_metric_dict(epoch, metrics_dict, 'purity', purity, data_fold) 
                self.plot_metric.on_epoch_end(epoch, metrics_dict['purity'], 'purity')


            # These metrics are computed while coloring with the training set, since the convert_nodes_to_preds function colored with the training set.
            self.acc_cb.on_epoch_end(epoch, metrics_dict, train_dict, val_dict)
            self.kappa_cb.on_epoch_end(epoch, metrics_dict, train_dict, val_dict)
            self.nmi_cb.on_epoch_end(epoch, metrics_dict, train_dict, val_dict)
            
        if self.check_epoch(epoch):    
            super().on_epoch_end(epoch) 
        
            # Plot the SOMs as well
            if self.checking_metric is None or metrics_dict[self.checking_metric]['val'][epoch] < self.best_val_loss:
                self.plot_som('train', epoch, label_coloring_train) 
                self.plot_som('val', epoch, label_coloring_val)


                if self.checking_metric is not None:
                    self.best_val_loss = metrics_dict[self.checking_metric]['val'][epoch] 
                
