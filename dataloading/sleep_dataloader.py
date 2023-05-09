import warnings
import numpy as np
import yaml
from my_utils.python import one_hot
from torch.utils.data import Dataset
from my_utils.training import data_root_dir
from pathlib import Path
import h5py

class MassGeneric():
    def __init__(self, config, data_fold, **kwargs):
        self.include_channels = [ch.lower() for ch in config.include_channels]

        self.data_fold = data_fold.lower()
        assert self.data_fold in ['train', 'val', 'test']   

        self.root_dir = data_root_dir() / Path(config.load_dir)
        self.data_folder = self.root_dir / self.data_fold

        config_key = f'include_{self.data_fold}_subjects'            
        if config.get(config_key, 'all') == 'all':
            self.include_subjects = self.subjects_in_folder()
        else:
            self.include_subjects = ['s'+str(subject_idx) for subject_idx in config[config_key]]

    def preload_data(self):

        for subject in self.include_subjects:     
            try:
                hf = h5py.File(self.data_folder /
                                f'{subject}.h5', 'r')

                self.signal_lengths[f'{subject}'] = hf.get('nr_samples')[0]
                data_array = np.zeros(
                    (len(getattr(self, f'include_channels')), self.signal_lengths[f'{subject}']), dtype=np.float32)
                
                for idx, channel in enumerate(getattr(self, f'include_channels')):
                    data_array[idx] = hf.get(channel)[()]
                
                self.x[f'{subject}'] = data_array
                self.y[f'{subject}'] = hf.get('labels')[()].astype(np.float32)                   
                hf.close()

            except Exception  as e:
                print(e)
                warnings.warn(
                    f'The file: {subject}.h5 could not be loaded from {self.data_folder}.')

    def subjects_in_folder(self):
        subject_paths = list(self.data_folder.rglob("*.h5"))
        return [subject.stem for subject in subject_paths]
        


class PSGDataset(Dataset):

    def __init__(self, config, labels_transitions=False): 
        """ Creates a pytorch dataset.

        Args:
            config (Config): A configuration containing all details for data loading.
            labels_transitions (bool): If true, not only the conventional sleep stage labels are loaded, but also the labels that include transition information.
        """ 
        
        self.read_meta_data()
        self.crop_size=config.crop_size
        self.crop_size_samples = self.crop_size * self.fs
        self.past_windows = config.get('past_windows',0)

        self.x, self.y = {}, {}
        self.signal_lengths = {}

        self.preload_data()
        self.find_all_crops()
        self.num_crops = len(self.all_crops)       

    def read_meta_data(self):
        processings_settings_file = list(self.root_dir.rglob("preprocessing_specifics*"))[0]

        with open(processings_settings_file) as meta_data_file:
            meta_data = yaml.load(meta_data_file, Loader=yaml.FullLoader)
            self.fs = meta_data['current_sample_rate']
            self.dataset = meta_data['dataset']
            self.sec_per_label = meta_data['sec_per_annotation']
            self.nr_classes = meta_data['nr_classes']

    def __len__(self):
        return self.num_crops
    
    def return_all_data_paths(self):
        return sorted(self.data_folder.rglob("*.h5"))

    def convert_sample_to_sleep_epoch(self, sample_idx):
        return int(np.floor(sample_idx / (self.fs * self.sec_per_label)))


    def get_xy_pair(self, idx):
        subject, start_sample_idx = self.all_crops[idx]
        sleep_epoch_idx = self.convert_sample_to_sleep_epoch(start_sample_idx)

        try:
            x = self.x[subject][:, (start_sample_idx-self.past_windows*self.crop_size_samples):(start_sample_idx+self.crop_size_samples)]
            y = self.y[subject][sleep_epoch_idx] # In case of past_windows > 1, only return the last label.
            return x, np.squeeze(one_hot(y, self.nr_classes, axis=-1)), subject, start_sample_idx
        except:
            warnings.warn(f'Not able to extract x,y for subject: {subject}, at start sample {start_sample_idx} and sleep epoch idx {sleep_epoch_idx}. Check for bugs!')
    

    def __getitem__(self, idx):
        x, y, _, _ = self.get_xy_pair(idx)
        return x,y

            
    def find_all_crops(self,  subtract_seconds_at_end=0):
        """ Returns a list containing tuples of all crops in the dataset.
            These tuples have format: (subject, start_idx)

        Args:
            subtract_seconds_at_end (int, optional): Indicate the number of seconds that should be ommitted when finding all possible crops. Defaults to 0.
        """

        self.all_crops = []

        for subject in self.x.keys():
            full_data_length = self.signal_lengths[subject] - (subtract_seconds_at_end*self.fs)
            # The floor operation makes sure the last samples of the data are not included if they cannot create a full patch of length crop_size
            start_idxs = np.arange(self.past_windows * self.crop_size_samples, np.floor(
                (full_data_length)/(self.crop_size_samples))*self.crop_size_samples, self.crop_size_samples, dtype=np.int32)

            self.all_crops.extend([(subject, start_idx)
                                   for start_idx in start_idxs])

    def get_subject(self, subject):
        """Returns all x,y pairs of a requested subject.

        Args:
            subject (str): Defines the unique string denoting a subject for which we want to return all data. String format is dataset dependent.

        Returns:
            np.ndarray, np.ndarray: x,y pairs of the requested subject.
        """
        return self.x.get(subject), np.squeeze(one_hot(self.y.get(subject), self.nr_classes, axis=-1))


class ContrastivePSG(PSGDataset):

    def __init__(self, config, labels_transitions=False):
        self.pos_samples = config.pos_samples
        assert self.pos_samples >= 1

        self.neg_samples = config.neg_samples * config.pos_samples # For each positive sample we want a different set of negative samples.
        assert config.neg_samples >= 1

        self.crop_size_cs = config.crop_size_cs
        assert self.crop_size_cs >= 0
        
        self.neg_sampling = config.neg_sampling.lower()
        assert self.neg_sampling.lower() in ['naive', 'within_patient', 'other_patients']

        self.past_windows = config.get('past_windows',0)

        self.crop_size_cs_samples = self.crop_size_cs * 128
        self.stride_pos_in_sec = config.get('stride_pos_in_sec', self.crop_size_cs_samples) #Number of seconds to ignore before the next positive sample is taken.
        self.stride_pos_samples = self.stride_pos_in_sec * 128

        super().__init__(config, labels_transitions)

        # Fs was is already needed in line29, so there it was assumed to be 128. Confirm this once self.fs has been created.
        assert self.fs == 128
    
    def sample_multiple_negative_crops(self, nr_samples, subject, exclude_time_samples=None):
        """Loads a crop from the data_path that serves as a negative sample.

        Args:
            nr_samples (int): Number of negative samples to select from this data_path
            subject (str): Unique subject string. Format is dataset dependent.
            exclude_time_samples (np.ndarray or list, optional): Indicating the time samples of the signal that may not be used to create a negative sample. Defaults to None.

        Returns:
            np.ndarray: 2D array of shape [channels, signal_length] or [nr_samples, channels, signal_length], containing data from one subject.
        """
        signal_length = self.signal_lengths[subject]

        neg_samples = []
        start_sample_idxs = np.random.randint(0,signal_length-self.crop_size_cs_samples, (nr_samples,))
        
        for sample in range(nr_samples):
            data_array = self.x[subject][:,start_sample_idxs[sample]:(start_sample_idxs[sample]+self.crop_size_cs_samples)]

            if nr_samples == 1:
                return data_array
            else:
                neg_samples.append(data_array)

        return np.stack(neg_samples)        

        
    def naive_neg_sampling(self):
        """ Samples neg_samples number of data crops naively/randomly from the entire data folder.

        Returns:
             np.ndarray: Array of shape [neg_samples, channels, crop_size_cs] containing negative samples.
        """
        all_data_paths = self.return_all_data_paths()
        neg_sample_path_idxs = np.random.permutation(len(all_data_paths))[:self.neg_samples]

        neg_samples = []
        for samp in range(self.neg_samples):
            subject_str = all_data_paths[neg_sample_path_idxs[samp]].stem
            neg_samples.append(self.sample_multiple_negative_crops(nr_samples = 1, subject = subject_str))

        return np.stack(neg_samples)

    def within_patient_neg_sampling(self, subject):
        """ Samples neg_samples number of data crops from the requested subject. 
            Note: The current implementation could sample the current sample of the patient as negative sample at this moment.

        Args:
            subject (str): String of format s<idx>_<odd_even>
        Returns:
             np.ndarray: Array of shape [neg_samples, channels, crop_size_cs] containing negative samples.
        """
        return self.sample_multiple_negative_crops(nr_samples=self.neg_samples, subject=subject)

    def other_patient_neg_sampling(self, subject):
        raise NotImplementedError


    def strided_slicing(self, array, L, S, striding_axis=-1): 
        """
        Extended from: https://stackoverflow.com/questions/40084931/taking-subarrays-from-numpy-array-with-given-stride-stepsize/40085052#40085052
        
        array (np.ndarray): Array to be sliced of shape [*ndims, slice_axis]
        L (int): window length
        S (int): stride 
        striding_axis (int, optional): Default to -1

        Returns:
        array in shape: [*ndims, nr_of_slices, L]
        """
        nrows = ((array.shape[striding_axis]-L)//S)+1
        if striding_axis == -1:
            return array[...,S*np.arange(nrows)[:,None] + np.arange(L)]
        else:
            raise NotImplementedError
            
    def positive_sampling(self, subject, start_sample_idx_pos_samples):
        
        if self.stride_pos_in_sec == self.crop_size_cs_samples:
            end_sample_idx = start_sample_idx_pos_samples + (self.pos_samples * self.crop_size_cs_samples)
            data_array = self.x[subject][:,start_sample_idx_pos_samples:end_sample_idx]

            # Reshape to: [pos_samples, channels, crop_size_cs_samples]
            data_array_r = data_array.reshape((-1, self.pos_samples, self.crop_size_cs_samples))
            return np.moveaxis(data_array_r, 0, 1)
            
        else:
            raise NotImplementedError
        

    def find_all_crops(self):
        return super().find_all_crops(subtract_seconds_at_end=self.pos_samples * self.stride_pos_in_sec)

    def __getitem__(self, idx):
        x, y, subject, start_sample_idx = self.get_xy_pair(idx)

        # Load neg_samples array of shape: [neg_samples, channels, self.crop_size_cs_samples]
        if self.neg_sampling == 'naive':
            neg_samples = self.naive_neg_sampling()

        elif self.neg_sampling == 'within_patient':
            neg_samples = self.within_patient_neg_sampling(subject)

        elif self.neg_sampling == 'other_patients':
            neg_samples = self.other_patient_neg_sampling(subject)

        # Load pos_samples array of shape: [pos_samples, channels, self.crop_size_cs_samples]
        pos_samples = self.positive_sampling(subject, start_sample_idx+self.crop_size_samples) 

        return x, y, neg_samples, pos_samples


class MassPSG(PSGDataset, MassGeneric):
    def __init__(self, config, data_fold, labels_transitions=False, **kwargs):
        MassGeneric.__init__(self, config, data_fold)
        PSGDataset.__init__(self, config, labels_transitions)

class MassContrastive(ContrastivePSG, MassGeneric):
    def __init__(self, config, data_fold, labels_transitions=False, **kwargs):
        MassGeneric.__init__(self, config, data_fold)
        ContrastivePSG.__init__(self, config, labels_transitions)

