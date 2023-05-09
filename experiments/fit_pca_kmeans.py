from sklearn.decomposition import PCA
from pathlib import Path
import yaml 
import numpy as np
import torch
from pathlib import Path
from my_utils.training import set_device_settings
from models.build_model import build_model_from_config
from my_utils.config import Config, load_config_from_yaml
from my_utils.pytorch import tensor2array
from dataloading.toy_dataloaders import *
from dataloading.sleep_dataloader import *
from dataloading.librispeech_dataloader import *
import pickle as pk
import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as nmi_score
from datetime import date


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
    

def calculate_cluster_histograms(k, labels, nr_clusters, nr_classes):
    """
    Same function as calculate_som_histograms(k,labels) from the SOM callback.

    Returns a dict with unnormalized histograms for the labels [nan, N1,N2,N3,W,REM].
    """
    node_histograms = {} 
    for node in range(nr_clusters):
        if node in k:
            counts, _ =  np.histogram(labels[tensor2array(k) == node], bins=np.arange(nr_classes+1), density=False)
        else:
            counts = np.nan
        node_histograms.update({node: counts})
    return node_histograms

def calculate_cluster_mode(k, labels, nr_classes=6, nr_clusters=100, continuous_labels=False):
    """
    Same function as calculate_som_mode() from the SOM callback
    
    k (list, np.ndarray) Containing the som node indices for all windows
    labels (list, np.ndarray): Containing the corresponding ground truth labels
    
    """
    if continuous_labels:
        median_labels = {} 
        for node in range(nr_clusters):
            if node in k:
                median_label_value =  np.median(labels[k == node])
            else:
                median_label_value = np.nan
            median_labels.update({node: median_label_value})
        return median_labels
    else:
        node_histograms = calculate_cluster_histograms(k, labels, nr_clusters, nr_classes)

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

def calculate_purity(k, mode_labels_counts):
    return np.nansum(list(mode_labels_counts.values())) / len(k) 

if __name__ == "__main__":

    # Indicate whether to apply K-means on top of PCA, or on top of the extracted latent space.
    kmeans_on_top_pca = True

    # Provide data type, choose from {'toycase', 'mass', 'librispeech'} and a path to the model directory.
    #Examples:
    
    data_type = 'toycase'
    config_path = <FILL IN>


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
        print('Fill in the best epoch manually')
        best_epoch = 2269 #None

    checkpoint_path = Path(config_path) / 'checkpoints' / f'model_{best_epoch}.pt'
    print(f'Selected epoch {best_epoch}')
        

    ### Load model
    config_path, checkpoint_path = Path(config_path), Path(checkpoint_path)

    config = load_config_from_yaml(Path(config_path) / 'config.yml')
    device = set_device_settings(cuda=config.device.get('cuda', True))

    if data_type == 'toycase' or data_type == 'mass':
        base_config_path = Path('models') / 'supervised_classifier' / 'base_config.yml'
        base_config = load_config_from_yaml(base_config_path)

        base_config.model.encoder = config.model.encoder
        base_config.model.encoder.checkpoint_path = checkpoint_path
        base_config.model.module_order = ['encoder']

        base_config = Config(base_config)
        model = build_model_from_config(base_config, device=device)
        model.eval()
        print(model)

    elif data_type == 'librispeech':
        if 'SOM_CPC' in str(config_path):
            model_type = 'SOM-CPC'
        elif 'SOM_VAE' in str(config_path) and 'withGRU' in str(config_path):
            if 'decoderonall' in str(config_path):
                model_type = 'DESOM+GRU_decall'
            else:
                model_type = 'DESOM+GRU_declast'
        elif 'SOM_VAE' in str(config_path) and 'noGRU' in str(config_path):
            model_type = 'DESOM'
        else:
            model_type = 'SOM'
            
        
        # Make the SOM-CPC model a feedforward model to prevent drawing contrastive samples during inference.
        if model_type == 'SOM-CPC' or model_type == 'SOM':
            base_config_path = Path('models') / 'supervised_classifier' / 'base_config.yml'
            base_config = load_config_from_yaml(base_config_path)

            base_config.model.encoder = config.model.encoder
            base_config.model.encoder.checkpoint_path = checkpoint_path
            base_config.model.ARmodel = config.model.ARmodel
            base_config.model.ARmodel.checkpoint_path = checkpoint_path
            base_config.model.permute['class'] = 'Permute'
            base_config.model.select_last_element['class'] = 'SelectLastElement'
            base_config.model.select_last_element['axis'] = 0
            base_config.model.module_order = ['encoder', 'permute', 'ARmodel', 'select_last_element']
            
            base_config = Config(base_config)
            model = build_model_from_config(base_config, device=device)
        else:
            config.model.checkpoint_path = str(checkpoint_path)
            model = build_model_from_config(config, device=device)

        model.eval()
        print(model)


    if data_type == 'toycase':
        data_loader = 'MixtureOfSinusoids'
    elif data_type == 'mass':
        data_loader = 'MassPSG'
    elif data_type == 'librispeech':
        data_loader = 'LibriSpeech' 


    ### Load training data
    data_fold = 'train'
    data_set =  eval(data_loader)(config.data, data_fold)


    ### Run training data through model
    all_z_cont, all_targets = [], []

    for subject, data, in data_set.x.items():
        if data_type == 'librispeech':
            data = np.expand_dims(data, 0)
            
        x_windows = sliding_windowing_x_incl_past_windows(data, config.data.past_windows, config.data.crop_size, config.data.crop_size, data_set.fs, device).to(device) #[nr_widows, channels, window length]        
        
        if data_type == 'toycase' or data_type == 'mass':
            with torch.no_grad():
                pred = model((x_windows,),epoch=0)   
            all_z_cont.append(tensor2array(pred['out'].squeeze()))
            all_targets.append(tensor2array(data_set.y[subject]))
            
        elif data_type == 'librispeech': #prevent OOM by batching the data
            prevent_one_elem_in_last_batch = False
            batch_size = 256

            for batch_idx, start_batch_el in enumerate(range(0,len(x_windows),batch_size)):        
                if batch_idx < len(range(0,len(x_windows),batch_size)) - 1: #non-last batches
                    
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
                all_z_cont.append(tensor2array(pred['out']))
                all_targets.append(np.ones((pred['out'].shape[0],))*(data_set.speaker_id_labels[subject]))

    all_z_cont = np.concatenate(all_z_cont,0)
    all_targets = np.concatenate(all_targets,0)

    # Fit PCA
    if kmeans_on_top_pca:
        pca = PCA(n_components=2)
        fitted_pca = pca.fit(all_z_cont)
        train_pca_proj = fitted_pca.transform(all_z_cont)
        pk.dump(fitted_pca, open(config_path / f"fitted_pca_{date.today()}.pkl","wb"))


    ### Fit K-means on train set ###

    if data_type == 'librispeech':
        kmeans = KMeans(n_clusters=225)
    else:
        kmeans = KMeans(n_clusters=100)
    config_path = Path(config_path)

    if kmeans_on_top_pca:    
        fitted_kmeans = kmeans.fit(train_pca_proj)
        pk.dump(fitted_kmeans, open(config_path / f"fitted_kmeans_onPCA_{date.today()}.pkl","wb"))    
        pred_train_clusters = fitted_kmeans.predict(train_pca_proj)

    else:
        
        fitted_kmeans = kmeans.fit(all_z_cont)
        pk.dump(fitted_kmeans, open(config_path / f"fitted_kmeans_onCPC_{date.today()}.pkl","wb"))
        pred_train_clusters = fitted_kmeans.predict(all_z_cont)
        

    if data_type == 'toycase':
        all_targets = np.median(np.reshape(all_targets,(-1,128)),-1)
        mode_labels = calculate_cluster_mode(pred_train_clusters, all_targets,nr_clusters=100, continuous_labels=True)
    else:
        if data_type == 'mass':
            nr_classes = 6
            nr_clusters = 100
        else:
            nr_classes = 10
            nr_clusters = 225
        node_histograms, mode_labels, mode_labels_counts = calculate_cluster_mode(pred_train_clusters, all_targets,nr_classes=nr_classes, nr_clusters=nr_clusters)
        

    ### Load test data
    data_fold = 'test'
    config.data['include_test_subjects'] = 'all'
    data_set_test =  eval(data_loader)(config.data, data_fold)

    ### Run test data through model
    all_z_cont_test, all_targets_test = [], []

    for subject, data, in data_set_test.x.items():
        if data_type == 'librispeech':
            data = np.expand_dims(data, 0)
            
        x_windows = sliding_windowing_x_incl_past_windows(data, config.data.past_windows, config.data.crop_size, config.data.crop_size, data_set.fs, device).to(device) #[nr_widows, channels, window length]        
        
        if data_type == 'toycase' or data_type == 'mass':
            with torch.no_grad():
                pred = model((x_windows,),epoch=0)   
            all_z_cont_test.append(tensor2array(pred['out'].squeeze()))
            all_targets_test.append(tensor2array(data_set_test.y[subject]))
            
        elif data_type == 'librispeech': #prevent OOM
            prevent_one_elem_in_last_batch = False
            batch_size = 256

            for batch_idx, start_batch_el in enumerate(range(0,len(x_windows),batch_size)):        
                if batch_idx < len(range(0,len(x_windows),batch_size)) - 1: #non-last batches
                    
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
                all_z_cont_test.append(tensor2array(pred['out']))
                all_targets_test.append(np.ones((pred['out'].shape[0],))*(data_set_test.speaker_id_labels[subject]))

    # Predict fitted K-means on test set
    label_preds_test = []
    pred_test_clusters = []
    for z_cont_test in all_z_cont_test:
        label_pred_one_series = []
        
        if kmeans_on_top_pca:
            pca_test_proj = fitted_pca.transform(z_cont_test)
            pred_test_clusters.append(fitted_kmeans.predict(pca_test_proj))
            
            for pca_z in pca_test_proj:
                label_pred_one_series.append(mode_labels[fitted_kmeans.predict(np.expand_dims(pca_z,0))[0]])
            label_preds_test.append(label_pred_one_series)
            
        else:
            pred_test_clusters.append(fitted_kmeans.predict(z_cont_test))
            for z in z_cont_test:
                label_pred_one_series.append(mode_labels[fitted_kmeans.predict(np.expand_dims(z,0))[0]])
            label_preds_test.append(label_pred_one_series)
            
    if data_type == 'toycase':
        SE_targets = []
        for target_test, pred_test in zip(all_targets_test, label_preds_test):
            median_target = np.median(np.reshape(target_test,(-1,128)),-1)
            SE_targets.append(sklearn.metrics.mean_squared_error(np.array(pred_test), median_target))
        print('SE_target: ', np.mean(SE_targets), np.std(SE_targets))

    elif data_type == 'mass' or data_type == 'librispeech':
        node_histograms_test, mode_labels_test ,mode_labels_counts_test = calculate_cluster_mode(np.concatenate(pred_test_clusters,0), np.concatenate(all_targets_test,0), nr_classes=nr_classes, nr_clusters=nr_clusters)
        test_purity = calculate_purity(np.concatenate(pred_test_clusters,0), mode_labels_counts_test)
        print('Purity: ', test_purity)

        nmi_test = nmi_score(np.concatenate(all_targets_test,0), np.concatenate(pred_test_clusters,0))
        print('NMI: ', nmi_test)

        if data_type == 'mass': #kappas per recording
            kappas_test = []
            for target_test, pred_test in zip(all_targets_test, label_preds_test):
                kappas_test.append(sklearn.metrics.cohen_kappa_score(target_test, pred_test))
            print('Cohens kappa: ', np.mean(kappas_test), np.std(kappas_test))
        else:
            kappa_test = sklearn.metrics.cohen_kappa_score(np.concatenate(all_targets_test,0), np.concatenate(label_preds_test,0))
            print('Cohens kappa: ', kappa_test)
            