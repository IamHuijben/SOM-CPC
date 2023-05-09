from pathlib import Path
import torch
from my_utils.training import set_device_settings
from models.build_model import build_model_from_config
from my_utils.config import Config, load_config_from_yaml
from my_utils.pytorch import tensor2array
from callbacks.visualizations import SOMmap
import sklearn.metrics 
from sklearn.metrics import normalized_mutual_info_score as nmi_score
from somperf.metrics import topographic_error
from somperf.utils.topology import square_topology_dist
from scipy.stats import mode

from matplotlib import pyplot as plt
import numpy as np
import yaml 
import warnings

from dataloading.toy_dataloaders import MixtureOfSinusoids
from dataloading.sleep_dataloader import MassPSG



def sliding_windowing_x(x, window_size_in_sec, sliding_step_in_sec, fs, device):
    """ Returns x in shape: [nr_windows, ch, window_size] by performing sliding windowing

    Returns:
        torch.tensor: Tensor of shape [number of windows, channels, window size]
    """ 
    window_size = int(window_size_in_sec * fs)
    sliding_step = int(sliding_step_in_sec * fs)

    channels, time_dim = x.shape
    x = torch.as_tensor(x, device=device)

    if window_size_in_sec ==sliding_step_in_sec:
        #return np.swapaxes(x.reshape(channels, -1, window_size), 0, 1) 
        return torch.transpose(torch.reshape(x, (channels, -1, window_size)), 0, 1) #[nr of windows x channels x window length]
    else:
        windows = []
        start_sample = 0
        for start_sample in range(0,time_dim-window_size+1,sliding_step):            
            windows.append(x[:,start_sample: start_sample+window_size])
        return torch.stack(windows,0)
    
def convert_k_to_kx_ky(k, nr_nodes):
    som_dim_1d = int(np.sqrt(nr_nodes))
    k_x = k // som_dim_1d 
    k_y = k % som_dim_1d 
    return k_x, k_y


def sum_norm_over_time(k_x, k_y, ord=2):
    # Assuming k_x and k_y to be 1D arrays or lists containing all x and y coordinates
    return np.sum(np.linalg.norm(np.diff(np.stack([k_x,k_y])),ord=ord,axis=0))

def uniformity_loss(node_histograms):
    ass_per_node = []
    for k,v in node_histograms.items():
        if not isinstance(v, np.ndarray) and np.isnan(v):
            ass_per_node.append(0.)
        else:
            ass_per_node.append(v.sum())

    expected_ass_per_node = sum(ass_per_node) / len(node_histograms)
    return np.mean(np.abs(ass_per_node - expected_ass_per_node))


if __name__ == "__main__":
    """
    This script runs inference on all type of models that include a SOM (SOM-VAE, DESOM, SOM-CPC, CPC + SOM)    
    Provide a list of folder directories that point to runs for which you want to run inference and indicate the data_type: toycase or mass.
    """

    
    data_type = 'toycase' #or 'mass'
    config_paths = [
        <FILL IN>
    ]


    checkpoint_paths = []
    for config_path in config_paths:
        try:
            with open(str(Path(config_path)/ 'metrics' / 'all_metrics_training.yml')) as file:
                metrics = yaml.load(file, Loader=yaml.FullLoader)
                if 'infoNCE' in metrics: #SOM-CPC
                    best_epoch = np.argmin(metrics['infoNCE']['val'])
                elif 'infoNCEdirect_cont' in metrics:
                    best_epoch = np.argmin(metrics['infoNCEdirect_cont']['val'])
                elif 'mse_disc' in metrics: #SOM-VAE
                    best_epoch = np.argmin(np.array(metrics['mse_cont']['val']) + np.array(metrics['mse_disc']['val']))
                elif 'mse_cont' in metrics: #DESOM
                    best_epoch = np.argmin(metrics['mse_cont']['val'])
                else:
                    best_epoch = np.argmin(metrics['loss']['val'])
        except:
            print('Fill in the best epoch manually')
            best_epoch = None
            
        checkpoint_paths.append(Path(config_path) / 'checkpoints' / f'model_{best_epoch}.pt')
        print(f'Selected epoch {best_epoch}')


    ### Load model
    nr_nodes = 100
    for config_path, checkpoint_path in zip(config_paths, checkpoint_paths):
        config_path, checkpoint_path = Path(config_path), Path(checkpoint_path)
        
        config = load_config_from_yaml(Path(config_path) / 'config.yml')
        device = set_device_settings(cuda=config.device.get('cuda', True))

        base_config_path = Path('models') / 'supervised_classifier' / 'base_config.yml'
        base_config = load_config_from_yaml(base_config_path)

        base_config.model.encoder = config.model.encoder
        base_config.model.encoder.checkpoint_path = checkpoint_path
        base_config.model.quantizer = config.model.quantizer
        base_config.model.quantizer.checkpoint_path = checkpoint_path
        base_config.model.module_order = ['encoder', 'quantizer']

        base_config = Config(base_config)
        model = build_model_from_config(base_config, device=device)
        model.eval()
        model


        ### Load training data
        data_loader = 'MixtureOfSinusoids' if data_type == 'toycase' else 'MassPSG' 
        data_fold = 'train'
        data_set =  eval(data_loader)(config.data, data_fold)

        ### Load test data
        data_fold = 'test'
        config.data['include_test_subjects'] = 'all'
        data_set_test = eval(data_loader)(config.data, data_fold)
        

        ### Run training data through model
        all_preds, all_targets = [],[]
        for subject, data, in data_set.x.items():
            window_size_in_sec = config.data.crop_size
            sliding_step_in_sec = config.data.crop_size
                
            x_windows = sliding_windowing_x(data, window_size_in_sec, sliding_step_in_sec, data_set.fs, device).to(device) #[nr_widows, channels, window length]        
            with torch.no_grad():
                pred = model((x_windows,),epoch=0)   
            all_preds.append(pred['all_codebook_idxs'])
            all_targets.append(data_set.y[subject]) 

        all_preds = tensor2array(torch.cat(all_preds,0))
        all_targets = np.concatenate(all_targets,0) 
        
        #Target per sample, so has to be downsampled still for toy case
        if data_type == 'toycase':
            all_targets = np.median(np.reshape(all_targets,(-1,128)),-1)
        
        
        ### Compute node labels
        zero_class_is_nan_class = False if data_type == 'toycase' else True
        SOMmap_cb = SOMmap(0, data_set.__dict__, nr_nodes = nr_nodes, zero_class_is_nan_class=zero_class_is_nan_class)
        if data_type == 'toycase':
            SOMmap_cb.nr_classes = np.inf
            node_labels = SOMmap_cb.calculate_som_mode(all_preds, all_targets, continuous_labels=True) 
        else:
            SOMmap_cb.nr_classes = 6
            node_histograms_train, node_labels, mode_labels_counts_train = SOMmap_cb.calculate_som_mode(all_preds, all_targets, continuous_labels=False) 
        
        # Fill not-assigned nodes with neighbour mode
        som_dim_1d =int(np.sqrt(nr_nodes))
        for k, label_value in node_labels.items():
            if np.isnan(label_value):
                k_x = k // som_dim_1d
                k_y = k % som_dim_1d

                k_up = np.where(k_y > 0,k-1,k) 
                k_down = np.where(k_y<som_dim_1d-1,k+1,k)
                k_left = np.where(k_x >0,k-som_dim_1d,k)
                k_right = np.where(k_x<som_dim_1d-1,k+som_dim_1d,k)

                neighbour_mode = mode([node_labels[int(k_left)], node_labels[int(k_down)],node_labels[int(k_right)],node_labels[int(k_up)]])[0]
                if np.isnan(neighbour_mode): #If the neighbours are all non-assigned, just use the mode label of the full som at this moment.
                    node_labels[k] = int(mode(list(node_labels.values()))[0])
                    warnings.warn(f'node {k} and its neighbours were all not assigned by the train set. ')
                else: 
                    node_labels[k] = int(neighbour_mode)   

        ### Run test data through model
        all_preds_test, all_targets_test, all_z_cont_test, label_preds_test = [],[], [], []
        l2_velocities, SE_targets, topo_errors, kappas = [],[],[],[]

        dist_fun = square_topology_dist(map_size=int(np.sqrt(nr_nodes)))
        som = tensor2array(model.quantizer.embedding.weight)
            
        som_trajectories_per_test_subject, hypnogram_per_test_subject = {}, {}
        for subject, data, in data_set_test.x.items(): 

            x_windows = sliding_windowing_x(data, window_size_in_sec, sliding_step_in_sec, data_set_test.fs, device).to(device) #[nr_widows, channels, window length]

            with torch.no_grad():
                pred = model((x_windows,),epoch=0)   
            all_preds_test.append(tensor2array(pred['all_codebook_idxs']))
                                                                    
            node_preds = []
            for  node_pred in tensor2array(pred['all_codebook_idxs']):
                node_preds.append(node_labels[node_pred])
            label_preds_test.append(node_preds)
            
            all_z_cont_test.append(tensor2array(pred['all_z_cont']))
            
            if data_type == 'toycase':
                all_targets_test.append(np.median(np.reshape(tensor2array(data_set_test.y[subject]),(-1,128)),-1))
            else:
                all_targets_test.append(tensor2array(data_set_test.y[subject])) #[config.data.past_windows:])
                
            
            # Compute metrics per recording
            k_x, k_y = convert_k_to_kx_ky(tensor2array(all_preds_test[-1]), nr_nodes)
            vel_l2_norm = np.round(sum_norm_over_time(k_x,k_y,ord=2) / (len(k_x)-1),2) #norm2 distance / window
            l2_velocities.append(vel_l2_norm)
            if data_type == 'toycase':
                SE_targets.append(sklearn.metrics.mean_squared_error(label_preds_test[-1], all_targets_test[-1]))
            else:
                kappas.append(sklearn.metrics.cohen_kappa_score(all_targets_test[-1], label_preds_test[-1]))
                
            topo_errors.append(topographic_error(dist_fun=dist_fun, som=som, x=all_z_cont_test[-1]))

        all_preds_test = np.concatenate(all_preds_test,0)
        all_z_cont_test = np.concatenate(all_z_cont_test,0)
        label_preds_test = np.concatenate(label_preds_test,0) #frequency prediction per window
        all_targets_test = np.concatenate(all_targets_test,0)


        # Compute further metrics and print hem.
        if data_type == 'toycase':
            node_labels_test = SOMmap_cb.calculate_som_mode(all_preds_test, all_targets_test, continuous_labels=True) 
            topo_error = topographic_error(dist_fun=dist_fun, som=som, x=all_z_cont_test)
            
            print('SE_target: ', np.mean(SE_targets), np.std(SE_targets))
            print('L2_smooth: ', np.mean(l2_velocities), np.std(l2_velocities))
            print('TE: ', np.mean(topo_errors), np.std(topo_errors))
        
        elif data_type == 'mass':
            # Compute purity, NMI and Kappa
            _ , node_labels_test, mode_labels_counts_test = SOMmap_cb.calculate_som_mode(all_preds_test, all_targets_test, continuous_labels=False) 
            purity = SOMmap_cb.calculate_purity(all_preds_test, mode_labels_counts_test)
            nmi = nmi_score(all_targets_test, all_preds_test)
            kappa = sklearn.metrics.cohen_kappa_score(all_targets_test, label_preds_test)
            
            dist_fun = square_topology_dist(map_size=int(np.sqrt(nr_nodes)))
            som = tensor2array(model.quantizer.embedding.weight)
            topo_error = topographic_error(dist_fun=dist_fun, som=som, x=all_z_cont_test)
            
            print('Purity: ', purity)
            print('NMI: ', nmi)
            print('Cohens kappa: ', np.mean(kappas), np.std(kappas))
            print('L2_smooth: ', np.mean(l2_velocities), np.std(l2_velocities))
            print('TE: ', np.mean(topo_errors), np.std(topo_errors))
            

        #### Plot the som map colored with test set: node_labels_test
        k_x, k_y = convert_k_to_kx_ky(all_preds_test, nr_nodes)
        
        fig, ax = plt.subplots(1,1,figsize=(6,5))
        ax.set_xlim([-0.5,int(np.sqrt(nr_nodes))-0.5])
        ax.set_ylim([int(np.sqrt(nr_nodes))-0.5,-0.5])
        plt.xticks()
        plt.yticks()
        fig, ax, cbar = SOMmap_cb.plot_som(label_coloring=node_labels_test, data_fold=data_fold, epoch=1, fig=fig, ax=ax)
        plt.show()
        ax.set_xticks([])
        ax.set_yticks([])
        
        