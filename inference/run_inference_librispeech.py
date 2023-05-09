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

from matplotlib import pyplot as plt
import numpy as np
import yaml 

from dataloading.librispeech_dataloader import LibriSpeech
from scipy.stats import mode
import warnings


def sliding_windowing_x_incl_past_windows(x, past_windows, window_size_in_sec, sliding_step_in_sec, fs, device):
    """ Returns x in shape: [nr_windows, ch, window_size] by performing sliding windowing

    Returns:
        torch.tensor: Tensor of shape [number of windows, channels, window size]
    """ 
    window_size = int(window_size_in_sec * fs)
    sliding_step = int(sliding_step_in_sec * fs)
    past_window_samples = int(past_windows*window_size)

    channels, time_dim = x.shape
    x = torch.as_tensor(x, device=device)

    if window_size_in_sec ==sliding_step_in_sec and past_windows == 0:
        #return np.swapaxes(x.reshape(channels, -1, window_size), 0, 1) 
        return torch.transpose(torch.reshape(x, (channels, -1, window_size)), 0, 1) #[nr of windows x channels x window length]
    elif window_size_in_sec != sliding_step_in_sec and past_windows == 0:
        windows = []
        start_sample = 0
        for start_sample in range(0,time_dim-window_size+1,sliding_step):            
            windows.append(x[:,start_sample: start_sample+window_size])
        return torch.stack(windows,0)
    elif window_size_in_sec == sliding_step_in_sec and past_windows > 0:
        windows = []
        start_sample = 0
        for start_sample in range(past_window_samples,time_dim-window_size+1,sliding_step):            
            windows.append(x[:,(start_sample-past_window_samples): start_sample+window_size])
        return torch.stack(windows,0)
    else:
        raise NotImplementedError
    
def convert_k_to_kx_ky(k, nr_nodes):
    som_dim_1d = int(np.sqrt(nr_nodes))
    k_x = k // som_dim_1d 
    k_y = k % som_dim_1d 
    return k_x, k_y


def sum_norm_over_time(k_x, k_y, ord=2):
    # Assuming k_x and k_y to be 1D arrays or lists containing all x and y coordinates
    return np.sum(np.linalg.norm(np.diff(np.stack([k_x,k_y])),ord=ord,axis=0))

def fill_non_assigned_nodes(nr_nodes, node_labels):
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
                warnings.warn(f'The neighbours of node {k} were not assigned, so this node is colored with the mode of the full som.')
            else: 
                node_labels[k] = int(neighbour_mode)
    return node_labels


def plot_som_speakerid_gender(nr_nodes, label_coloring, node_shaping, log_dir=None, fontsize=14):

    som_dim_1d = int(np.sqrt(nr_nodes))
    k_x = np.arange(nr_nodes) // som_dim_1d 
    k_y = np.arange(nr_nodes) % som_dim_1d 

    node_shapes = np.array(list(node_shaping.values()))
    node_colors = np.array(list(label_coloring.values()))

    # Split in male (=0) and female (=1)
    k_x_0 = k_x[node_shapes ==0]
    k_y_0 = k_y[node_shapes ==0]
    k_x_1 = k_x[node_shapes ==1]
    k_y_1 = k_y[node_shapes ==1]

    fig, ax = plt.subplots(1,1,figsize=(6.4, 4.8))
    cmap = 'tab10' 
    cax = ax.scatter(k_x_0, k_y_0, c=node_colors[node_shapes==0], vmin=0,vmax=10, s=100, cmap=cmap, zorder=10, marker='o')
    cax = ax.scatter(k_x_1, k_y_1, c=node_colors[node_shapes==1], vmin=0,vmax=10, s=150, cmap=cmap, zorder=10, marker='*')
    ax.grid()
    
    plt.xticks(color='w')
    plt.yticks(color='w')

    cbar = fig.colorbar(cax, ticks=np.arange(0.5,10,1))
    cbar.ax.set_yticklabels(np.arange(10))
    cbar.ax.tick_params(labelsize=fontsize)
    ax.set_title('Clustering of speakers (colors)\n and genders (shapes)',fontsize=fontsize)

    if log_dir:
        plt.savefig(str(log_dir), bbox_inches='tight')
        plt.close(fig)


if __name__ == '__main__':
    
    """
    This script runs inference on all type of models that include a SOM ((GRU-)DESOM, SOM-CPC, CPC + SOM)    
    Provide a list of folder directories that point to runs for which you want to run inference and indicate the data_type: toycase or mass.
    """

    
    data_type = 'toycase' #or 'mass'
    config_paths = [
        Path.cwd() / "SOM_VAE_and_DESOM" / "Audio_experiments_ICML2023" / "DESOM_alpha1e-05_withGRU_decoderonlastC"
    ]



    checkpoint_paths = []
    for config_path in config_paths:
        
        try:
            with open(str(Path(config_path)/ 'metrics' / 'all_metrics_training.yml')) as file:
                metrics = yaml.load(file, Loader=yaml.FullLoader)
                if 'infoNCE' in metrics: #SOM-CPC
                    best_epoch = np.argmin(metrics['infoNCE']['val'])
                elif 'mse_disc' in metrics: #SOM-VAE
                    best_epoch = np.argmin(np.array(metrics['mse_cont']['val']) + np.array(metrics['mse_disc']['val']))
                elif 'mse_cont' in metrics: #DESOM
                    best_epoch = np.argmin(metrics['mse_cont']['val'])
                else:
                    best_epoch = np.argmin(metrics['loss']['val'])
        except:
            print('Fill in the best epoch yourself')
            best_epoch = None

        checkpoint_paths.append(Path(config_path) / 'checkpoints' / f'model_{best_epoch}.pt')
        print(f'Selected epoch {best_epoch}')
        
        

    batch_size = 256    
    for config_path, checkpoint_path in zip(config_paths, checkpoint_paths):
        
        if 'SOM_CPC' in config_path:
            model_type = 'SOM-CPC'
        elif 'SOM_VAE' in config_path and 'withGRU' in config_path:
            if 'decoderonall' in config_path:
                model_type = 'DESOM+GRU_decall'
            else:
                model_type = 'DESOM+GRU_declast'
        elif 'SOM_VAE' in config_path and 'noGRU' in config_path:
            model_type = 'DESOM'
        else:
            model_type = 'SOM'
        
        
        config_path, checkpoint_path = Path(config_path), Path(checkpoint_path)
        config = load_config_from_yaml(Path(config_path) / 'config.yml')
        device = set_device_settings(cuda=config.device.get('cuda', True))

        ### Load model
        # Make the SOM-CPC model a feedforward model to prevent drawing contrastive samples during inference.
        if model_type == 'SOM-CPC' or model_type == 'SOM':
            base_config_path = Path('models') / 'supervised_classifier' / 'base_config.yml'
            base_config = load_config_from_yaml(base_config_path)

            base_config.model.encoder = config.model.encoder
            base_config.model.encoder.checkpoint_path = checkpoint_path
            base_config.model.ARmodel = config.model.ARmodel
            base_config.model.ARmodel.checkpoint_path = checkpoint_path
            base_config.model.quantizer = config.model.quantizer
            base_config.model.quantizer.checkpoint_path = checkpoint_path
            base_config.model.permute['class'] = 'Permute'
            base_config.model.select_last_element['class'] = 'SelectLastElement'
            base_config.model.select_last_element['axis'] = 0
            base_config.model.module_order = ['encoder', 'permute', 'ARmodel', 'select_last_element', 'quantizer']
            
            base_config = Config(base_config)
            model = build_model_from_config(base_config, device=device)
        else:
            config.model.checkpoint_path = str(checkpoint_path)
            model = build_model_from_config(config, device=device)

        model.eval()
        model

    
        ### Load training data
        data_fold = 'train'
        data_set =  LibriSpeech(config.data, data_fold)

        ### Load test data
        data_fold = 'test'
        config.data['include_test_subjects'] = 'all'
        data_set_test =  LibriSpeech(config.data, data_fold)


        ### Run training data through model
        
        all_preds, all_phone_labels, all_speaker_id_labels, all_speaker_sex_labels = [],[],[],[]
        for file_id, data in data_set.x.items():
            
            window_size_in_sec = config.data.crop_size
            sliding_step_in_sec = config.data.crop_size
            past_windows = config.data.past_windows

            x_windows = sliding_windowing_x_incl_past_windows(np.expand_dims(data,0), past_windows, window_size_in_sec, sliding_step_in_sec, data_set.fs, device).to(device) #[nr_widows, channels, window length]        

            # Loop through the recording to prevent OOM issues
            prevent_one_elem_in_last_batch = False
            for batch_idx, start_batch_el in enumerate(range(0,len(x_windows),batch_size)): 

                if batch_idx < len(range(0,len(x_windows),batch_size)) - 1: #non-last batch
                    
                    # Check if last batch would be of size 1. If yes take one element less in this pen-ultimate batch to prevent a last batch-size of 1, which gives issues
                    if batch_idx == len(range(0,len(x_windows),batch_size)) - 2 and (start_batch_el+batch_size == len(x_windows) -1):
                        x_sub = x_windows[start_batch_el:(start_batch_el+batch_size-1)]    
                        prevent_one_elem_in_last_batch = True
                    else:
                        x_sub = x_windows[start_batch_el:(start_batch_el+batch_size)]
                        prevent_one_elem_in_last_batch = False
                else: #last batch
                    if prevent_one_elem_in_last_batch:
                        x_sub = x_windows[(start_batch_el-1):]
                    else:
                        x_sub = x_windows[start_batch_el:]

                with torch.no_grad():
                    pred = model((x_sub,),epoch=0)   
                all_preds.append(tensor2array(pred['all_codebook_idxs']))

            all_speaker_id_labels.append([data_set.speaker_id_labels[file_id]]*len(x_windows))
            all_speaker_sex_labels.append([data_set.speaker_sex_labels[file_id]]*len(x_windows))

        all_preds = np.concatenate(all_preds,0)
        all_speaker_id_labels = np.concatenate(all_speaker_id_labels, 0)
        all_speaker_sex_labels = np.concatenate(all_speaker_sex_labels, 0)

        ### Compute node labels phone id
        nr_nodes = config.model.quantizer.som_nodes

        ### Compute node labels speaker id
        data_set.__dict__['label_type'] = 'speaker_id'
        SOMmap_cb_speakerid = SOMmap(0, data_set.__dict__, nr_nodes = nr_nodes, zero_class_is_nan_class=False)
        SOMmap_cb_speakerid.nr_classes = data_set.speaker_classes
        node_histograms_train_speakerid, node_labels_speakerid, mode_labels_counts_train_speakerid = SOMmap_cb_speakerid.calculate_som_mode(all_preds, all_speaker_id_labels, continuous_labels=False) 
        node_labels_speakerid_filled = fill_non_assigned_nodes(nr_nodes, node_labels_speakerid)

        ### Compute node labels gender
        data_set.__dict__['label_type'] = 'speaker_sex'
        SOMmap_cb_speakersex = SOMmap(0, data_set.__dict__, nr_nodes = nr_nodes, zero_class_is_nan_class=False)
        SOMmap_cb_speakersex.nr_classes = data_set.gender_classes
        node_histograms_train_speakersex, node_labels_speakersex, mode_labels_counts_train_speakersex = SOMmap_cb_speakersex.calculate_som_mode(all_preds, all_speaker_sex_labels, continuous_labels=False) 
        node_labels_speakersex_filled = fill_non_assigned_nodes(nr_nodes, node_labels_speakersex)

        all_preds = tensor2array(all_preds)
        del all_preds
        
        
        ### Run test data through model
        all_preds_test, all_phone_labels_test, all_speaker_id_labels_test, all_speaker_sex_labels_test = [],[],[],[]
        speaker_id_preds_test = []    
        
        topo_errors_speakerid = []
        dist_fun = square_topology_dist(map_size=int(np.sqrt(nr_nodes)))
        som = tensor2array(model.quantizer.embedding.weight)


        for file_id, data in data_set_test.x.items():
            all_preds_one_rec, all_z_cont_one_rec = [], []

            x_windows = sliding_windowing_x_incl_past_windows(np.expand_dims(data,0), past_windows, window_size_in_sec, sliding_step_in_sec, data_set_test.fs, device).to(device) #[nr_widows, channels, window length]
                    
            prevent_one_elem_in_last_batch = False
            for batch_idx, start_batch_el in enumerate(range(0,len(x_windows),batch_size)):               
                if batch_idx < len(range(0,len(x_windows),batch_size)) - 1: #non-last batch
                    
                    # Check if last batch would be of size 1. If yes take one element less in this pen-ultimate batch
                    if batch_idx == len(range(0,len(x_windows),batch_size)) - 2 and (start_batch_el+batch_size == len(x_windows) -1):
                        x_sub = x_windows[start_batch_el:(start_batch_el+batch_size-1)]    
                        prevent_one_elem_in_last_batch = True
                    else:
                        x_sub = x_windows[start_batch_el:(start_batch_el+batch_size)]
                        prevent_one_elem_in_last_batch = False
                else: #last batch
                    if prevent_one_elem_in_last_batch:
                        x_sub = x_windows[(start_batch_el-1):]
                    else:
                        x_sub = x_windows[start_batch_el:]

                with torch.no_grad():
                    pred = model((x_sub,),epoch=0)   
                all_preds_one_rec.append(tensor2array(pred['all_codebook_idxs']))
                all_z_cont_one_rec.append(tensor2array(pred['all_z_cont']))
            

            all_speaker_id_labels_test.append([data_set_test.speaker_id_labels[file_id]]*len(x_windows))
            all_speaker_sex_labels_test.append([data_set_test.speaker_sex_labels[file_id]]*len(x_windows))
            
            all_preds_one_rec = np.concatenate(all_preds_one_rec,0)
            all_z_cont_one_rec = np.concatenate(all_z_cont_one_rec,0)
            
            # Compute TE per recording
            topo_errors_speakerid.append(topographic_error(dist_fun=dist_fun, som=som, x=all_z_cont_one_rec))
            all_preds_test.append(all_preds_one_rec)
            
            speaker_id_pred = []
            for pred in all_preds_one_rec:
                speaker_id_pred.append(node_labels_speakerid_filled[pred])
            speaker_id_preds_test.append(np.array(speaker_id_pred))
            

        all_preds_test = np.concatenate(all_preds_test,0)
        speaker_id_preds_test = np.concatenate(speaker_id_preds_test,0)
        all_speaker_id_labels_test = np.concatenate(all_speaker_id_labels_test, 0)
        all_speaker_sex_labels_test = np.concatenate(all_speaker_sex_labels_test, 0)

        

        # Compute purity, NMI and Kappa for speaker id labels
        _ , mode_labels_test_speakerid, mode_labels_counts_test_speakerid = SOMmap_cb_speakerid.calculate_som_mode(all_preds_test, all_speaker_id_labels_test, continuous_labels=False) 
        purity_speakerid = SOMmap_cb_speakerid.calculate_purity(speaker_id_preds_test, mode_labels_counts_test_speakerid)
        nmi_speakerid = nmi_score(all_speaker_id_labels_test, all_preds_test)
        kappa_speakerid = sklearn.metrics.cohen_kappa_score(all_speaker_id_labels_test, speaker_id_preds_test)

        # Gender
        _ , mode_labels_test_speakersex, mode_labels_counts_test_speakersex = SOMmap_cb_speakersex.calculate_som_mode(all_preds_test, all_speaker_sex_labels_test, continuous_labels=False) 
    
        config_path = Path(config_path)
        print('Purity: ', purity_speakerid)
        print('NMI: ', nmi_speakerid)
        print('Cohens kappa: ', kappa_speakerid)
        print('TE: ', np.mean(topo_errors_speakerid), np.std(topo_errors_speakerid))
        
        # Plot SOM modes colored with test set speaker ID and symbols with gender
        node_shaping = mode_labels_test_speakersex
        label_coloring = mode_labels_test_speakerid    
        
        plot_som_speakerid_gender(nr_nodes, label_coloring, node_shaping, log_dir=Path.cwd() / 'SOM.png')    

