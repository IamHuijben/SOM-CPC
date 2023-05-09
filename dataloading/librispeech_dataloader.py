import warnings
from pathlib import Path

import h5py
import numpy as np
import yaml
import random
from my_utils.python import one_hot
from my_utils.training import data_root_dir
from torch.utils.data import Dataset
from random import choices

class LibriSpeech(Dataset):

    def __init__(self, config, data_fold): 
        """ Creates a pytorch dataset.

        Args:
            config (Config): A configuration containing all details for data loading.
            data_fold (str): One of 'train', 'val', 'test
        """ 
        self.data_fold = data_fold.lower()
        assert self.data_fold in ['train', 'val', 'test']   

        self.root_dir = data_root_dir() / Path(config.load_dir)
        self.data_folder = self.root_dir / self.data_fold
        self.read_meta_data() 

        self.crop_size=config.crop_size
        self.crop_size_samples = int(self.crop_size * self.fs)
        self.past_windows = config.get('past_windows',0)


        config_key = f'include_{self.data_fold}_subjects'            
        if config.get(config_key, 'all') == 'all':
            self.include_subjects = self.subjects_in_folder()
        else:
            assert config.get(config_key) is not None
            self.include_subjects = config.get(config_key)
    
        self.x= {}
        self.speaker_id_labels = {}
        self.speaker_sex_labels = {}

        self.signal_lengths = {}
        self.preload_data()
        self.find_all_crops()
        self.num_crops = len(self.all_crops)
        

    def read_meta_data(self):
        
        with open(self.root_dir / 'preprocessing_specifics.yml') as meta_data_file:
            meta_data = yaml.load(meta_data_file, Loader=yaml.FullLoader)
            self.fs = int(meta_data['current_sample_rate'])
            self.dataset = meta_data['dataset']
            self.sec_per_label = meta_data['sec_per_annotation']

    def subjects_in_folder(self):
        all_paths = list(self.data_folder.rglob("*.h5"))

        all_files = []
        for path in all_paths:
            all_files.append(path.stem)

        return all_files

    def __len__(self):
        return self.num_crops

    def convert_sample_to_window_idx(self, sample_idx):
        return int(np.floor(sample_idx / (self.fs * self.sec_per_label)))

    def get_xy_pair(self, idx):
        file_id, start_sample_idx = self.all_crops[idx]
        window_idx = self.convert_sample_to_window_idx(start_sample_idx)

        try:
            x = self.x[file_id][(start_sample_idx-self.past_windows*self.crop_size_samples):(start_sample_idx+self.crop_size_samples)]
            x = np.expand_dims(x,0) # Make it a one-channel array #[1, crop_size_samples]
            return x, np.squeeze(one_hot(self.speaker_id_labels[file_id],  self.speaker_classes, axis=-1)), file_id, start_sample_idx

        except:
            warnings.warn(f'Not able to extract x,y for file: {file_id}, at start sample {start_sample_idx} and window idx {window_idx}. Check for bugs!')

    def __getitem__(self, idx):
        x, y, *_ = self.get_xy_pair(idx)
        return x, y
 
    def find_all_crops(self,  subtract_seconds_at_end=0):
        """ Returns a list containing tuples of all crops in the dataset.
            These tuples have format: (subject, start_idx)

        Args:
            subtract_seconds_at_end (int, optional): Indicate the number of seconds that should be ommitted when finding all possible crops. Defaults to 0.
        """

        self.all_crops = []
        self.all_crops_per_file = {}
        too_short_files = []

        for file_id in self.x.keys():
            full_data_length = self.signal_lengths[file_id] - (subtract_seconds_at_end*self.fs)
            start_idxs = np.arange(self.past_windows * self.crop_size_samples, np.floor(
                (full_data_length)/(self.crop_size_samples))*self.crop_size_samples, self.crop_size_samples, dtype=np.int32)

            if len(start_idxs) > 0:
                self.all_crops.extend([(file_id, start_idx)
                                   for start_idx in start_idxs])
                
                self.all_crops_per_file[file_id] = start_idxs
            else:
                too_short_files.append(file_id)

        for file in too_short_files: 
            self.x.pop(file)
            self.include_subjects.remove(file)

            
    def preload_data(self):

        for file_id in self.include_subjects:    
            try:
                hf = h5py.File(self.data_folder / (file_id+'.h5'), 'r')

                self.signal_lengths[file_id] = hf.get('nr_samples')[0]

                self.x[file_id] =  hf.get('waveform')[()]
                self.speaker_id_labels[file_id] = hf.get('speaker_id')[()].astype(np.float32)
                self.speaker_sex_labels[file_id] = hf.get('speaker_sex')[()].astype(np.float32)
                hf.close()

            except Exception  as e:
                print(e)
                warnings.warn(
                    f'The file: {file_id} could not be loaded from {self.data_folder}.')

        unique_speaker_ids = np.unique(list(self.speaker_id_labels.values()))
        if np.max(unique_speaker_ids) >= len(unique_speaker_ids):
            speaker_id_conversion_dict = dict(zip(unique_speaker_ids, np.arange(len(unique_speaker_ids))))
            for file_id, orig_label in self.speaker_id_labels.items():
                self.speaker_id_labels[file_id] = speaker_id_conversion_dict[orig_label]

        self.speaker_classes = len(unique_speaker_ids)
        self.gender_classes = len(np.unique(list(self.speaker_sex_labels.values())))
        
class LibriSpeechFast(LibriSpeech):
    """ 
    Instance of the LibriSpeech dataset, that only samples one window per recording in each epoch, instead of using exactly all windows in each epoch.
    
    """

    def __init__(self, config, data_fold, **kwargs): 
        super().__init__(config, data_fold)

    def __len__(self):
        return len(self.include_subjects)

    def get_xy_pair(self, idx, subtract_seconds_at_end=0):
        """
        subtract_seconds_at_end (int): Number of seconds that can not be used to draw a window, in case of necessary future windows that need to follow the drawn window.
        """
        file_id = self.include_subjects[idx]
        start_sample_idx = random.choice(self.all_crops_per_file[file_id])
        window_idx = self.convert_sample_to_window_idx(start_sample_idx)

        try:
            x = self.x[file_id][int(start_sample_idx-self.past_windows*self.crop_size_samples):int(start_sample_idx+self.crop_size_samples)]
            x = np.expand_dims(x,0) # Make it a one-channel array #[1, crop_size_samples]
            speaker_id_label = self.speaker_id_labels[file_id]

            return x, np.squeeze(one_hot(speaker_id_label,  self.speaker_classes, axis=-1)), file_id, start_sample_idx

        except:
            warnings.warn(f'Not able to extract x,y for file: {file_id}, at start sample {start_sample_idx} and window idx {window_idx}. Check for bugs!')


class LibriSpeechContrastive(LibriSpeechFast):
    def __init__(self, config, data_fold, **kwargs):
        self.pos_samples = config.pos_samples
        assert self.pos_samples >= 1

        self.neg_samples = config.neg_samples * config.pos_samples # For each positive sample we want a different set of negative samples.
        assert config.neg_samples >= 1

        self.crop_size_cs = config.crop_size_cs
        assert self.crop_size_cs >= 0
        
        self.neg_sampling = config.neg_sampling.lower()
        assert self.neg_sampling.lower() in ['within_sequence', 'naive']

        self.past_windows = config.get('past_windows',0)


        super().__init__(config, data_fold)
        self.crop_size_cs_samples = int(self.crop_size_cs * self.fs)

    def sample_multiple_negative_crops(self, nr_samples, file_id, exclude_time_samples=None):
        """Loads a crop from the data_path that serves as a negative sample.

        Args:
            nr_samples (int): Number of negative samples to select from this data_path
            file_id (str): Unique string. Format is dataset dependent.
            exclude_time_samples (np.ndarray or list, optional): Indicating the time samples of the signal that may not be used to create a negative sample. Defaults to None.

        Returns:
            np.ndarray: 2D array of shape [channels, signal_length] or [nr_samples, channels, signal_length], containing data from one sequence.
        """
        signal_length = self.signal_lengths[file_id]
        neg_samples = []
        start_sample_idxs = np.random.randint(0,signal_length-self.crop_size_cs_samples, (nr_samples,))
        
        for sample_idx in list(start_sample_idxs):
            data_array = self.x[file_id][sample_idx:(sample_idx+self.crop_size_cs_samples)]
            data_array = np.expand_dims(data_array,0)

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
        if len(self.include_subjects) < self.neg_samples: #this typically only happens in debug mode
            neg_sample_idxs = choices(np.arange(len(self.include_subjects)), k=self.neg_samples)
        else:
            neg_sample_idxs = np.random.permutation(len(self.include_subjects))[:self.neg_samples]

        neg_samples = []
        for file_idx in list(neg_sample_idxs):
            neg_samples.append(self.sample_multiple_negative_crops(nr_samples = 1, file_id = self.include_subjects[file_idx]))

        return np.stack(neg_samples)

    def within_sequence_neg_sampling(self, file_id):
        """ Samples neg_samples number of data crops from the requested sequence. 
            Note: The current implementation could sample the current sample of the sequence as negative sample at this moment.

        Args:
            file_id (str): String that denotes a unique file.
        Returns:
             np.ndarray: Array of shape [neg_samples, channels, crop_size_cs] containing negative samples.
        """
        return self.sample_multiple_negative_crops(nr_samples=self.neg_samples, file_id=file_id)


    def positive_sampling(self, file_id, start_sample_idx_pos_samples):
    
        end_sample_idx = start_sample_idx_pos_samples + (self.pos_samples * self.crop_size_cs_samples)
        data_array = self.x[file_id][start_sample_idx_pos_samples:end_sample_idx]

        # Reshape to: [pos_samples, channels, crop_size_cs_samples]
        data_array_r = data_array.reshape((-1, self.pos_samples, self.crop_size_cs_samples))
        return np.moveaxis(data_array_r, 0, 1)


    def find_all_crops(self):
        return super().find_all_crops(subtract_seconds_at_end=self.pos_samples * self.crop_size_cs)

    def __getitem__(self, idx):
        x, y, file_id, start_sample_idx = self.get_xy_pair(idx, subtract_seconds_at_end=self.pos_samples * self.crop_size_cs)

        # Load neg_samples array of shape: [neg_samples, channels, self.crop_size_cs_samples]
        if self.neg_sampling == 'naive':
            neg_samples = self.naive_neg_sampling()

        elif self.neg_sampling == 'within_sequence':
            neg_samples = self.within_sequence_neg_sampling(file_id)

        # Load pos_samples array of shape: [pos_samples, channels, self.crop_size_cs_samples]
        pos_samples = self.positive_sampling(file_id, int(start_sample_idx+self.crop_size_samples)) 

        return x, y, neg_samples, pos_samples

