import warnings
from pathlib import Path
import numpy as np
import yaml
from my_utils.training import data_root_dir
from torch.utils.data import Dataset
from random import choices
import h5py

class MixtureOfSinusoids(Dataset):

    def __init__(self, config, data_fold, **kwargs): 
        
        self.data_fold = data_fold.lower()
        assert self.data_fold in ['train', 'val', 'test']
       
        self.root_dir = data_root_dir() / Path(config.load_dir)
        self.data_folder = self.root_dir / self.data_fold
        
        config_key = f'include_{self.data_fold}_subjects'            
        if config[config_key] == 'all':
            self.include_subjects = self.subjects_in_folder()
        else:
            self.include_subjects = ['s'+str(subject_idx) for subject_idx in config[config_key]]
        self.include_channels = config.include_channels
                    
        self.read_meta_data()
        self.crop_size=config.crop_size
        self.crop_size_samples = int(self.crop_size * self.fs)
        self.past_windows = config.get('past_windows',0)

        self.x, self.y = {}, {}
        self.signal_lengths = {}

        self.preload_data()
        self.find_all_crops()
        self.num_crops = len(self.all_crops)   


    def preload_data(self):
        for subject in self.include_subjects:     
            try:
                hf = h5py.File(self.data_folder /
                                f'{subject}.h5', 'r')

                self.signal_lengths[f'{subject}'] = hf.get('nr_samples')[()]
                data_array = np.zeros(
                    (len(self.include_channels), self.signal_lengths[f'{subject}']), dtype=np.float32)
                for idx, channel in enumerate(self.include_channels):
                    data_array[idx] = hf.get(channel)[()]

                self.x[f'{subject}'] = data_array
                self.y[f'{subject}'] = hf.get('labels')[()].astype(np.float32)

                hf.close()

            except Exception  as e:
                print(e)
                warnings.warn(
                    f'The file: {subject}.h5 could not be loaded from {self.data_folder}.')

    def read_meta_data(self):
        
        with open(self.root_dir / 'preprocessing_specifics.yml') as meta_data_file:
            meta_data = yaml.load(meta_data_file, Loader=yaml.FullLoader)
            self.fs = meta_data.get('current_sample_rate', meta_data.get('fs'))
            self.dataset = meta_data.get('dataset')
            self.sec_per_label = meta_data['sec_per_annotation']
            self.nr_classes = meta_data['nr_classes']

    def __len__(self):
        return self.num_crops
    
    def get_xy_triplets(self, idx):
        subject, start_sample_idx = self.all_crops[idx]

        try:
            x = self.x[subject][:, (start_sample_idx-self.past_windows*self.crop_size_samples):(start_sample_idx+self.crop_size_samples)]
            # One label per sample, return the median of that window
            y = np.median(self.y[subject][(start_sample_idx-self.past_windows*self.crop_size_samples):(start_sample_idx+self.crop_size_samples)])
                
            return x, y, subject, start_sample_idx

        except:
            warnings.warn(f'Not able to extract x,y for subject: {subject}, at start sample {start_sample_idx}. Check for bugs!')
            

    def __getitem__(self, idx):
        x, y, _, _ = self.get_xy_triplets(idx)
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

            # Find the start_idxs of all windows in the data set
            start_idxs = np.arange(self.past_windows * self.crop_size_samples, np.floor(
                (full_data_length)/(self.crop_size_samples))*self.crop_size_samples, self.crop_size_samples, dtype=np.int32)

            self.all_crops.extend([(subject, start_idx) for start_idx in start_idxs])

    def subjects_in_folder(self):
        """
        Returns:
            list: List of absolute paths to all subjects in the data folder.
        """
        subject_paths = sorted(self.data_folder.rglob("*.h5"))
        subject_strings = [path.stem for path in subject_paths]
        return subject_strings


class MixtureOfSinusoidsContrastive(MixtureOfSinusoids):
    def __init__(self, config, data_fold, soft_labels=True, **kwargs):
        
        self.pos_samples = config.pos_samples
        assert self.pos_samples >= 1

        self.neg_samples = config.neg_samples * config.pos_samples # For each positive sample we want a different set of negative samples.
        assert config.neg_samples >= 1

        self.crop_size_cs = config.crop_size_cs
        assert self.crop_size_cs >= 0
       
        self.neg_sampling = config.neg_sampling.lower()
        assert self.neg_sampling.lower() in ['naive', 'within_patient']

        super().__init__(config, data_fold, soft_labels=True)
        self.crop_size_cs_samples = int(self.crop_size_cs * self.fs)

    
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

    def within_patient_neg_sampling(self, subject):
        """ Samples neg_samples number of data crops from the requested subject. 
            Note: The current implementation could sample the current sample of the patient as negative sample at this moment.

        Args:
            subject (str): String of format s<idx>_<odd_even>
        Returns:
             np.ndarray: Array of shape [neg_samples, channels, crop_size_cs] containing negative samples.
        """
        return self.sample_multiple_negative_crops(nr_samples=self.neg_samples, subject=subject)

    def naive_neg_sampling(self,):
        """ Samples neg_samples number of data crops naively/randomly from the entire data folder.

        Returns:
             np.ndarray: Array of shape [neg_samples, channels, crop_size_cs] containing negative samples.
        """
        random_subjects = choices(list(self.x.keys()),k=self.neg_samples)
        neg_samples = []
        for subject in random_subjects:
            neg_samples.append(self.sample_multiple_negative_crops(1, subject, exclude_time_samples=None))
        
        return np.stack(neg_samples)        


    def positive_sampling(self, subject, start_sample_idx_pos_samples):
        
        end_sample_idx = start_sample_idx_pos_samples + (self.pos_samples * self.crop_size_cs_samples)
        data_array = self.x[subject][:,start_sample_idx_pos_samples:end_sample_idx]

        # Reshape to: [pos_samples, channels, crop_size_cs_samples]
        data_array_r = data_array.reshape((-1, self.pos_samples, self.crop_size_cs_samples))
        return np.moveaxis(data_array_r, 0, 1)


    def find_all_crops(self):
        return super().find_all_crops(subtract_seconds_at_end=self.pos_samples * self.crop_size_cs)

    def __getitem__(self, idx):
        x, y, subject, start_sample_idx = self.get_xy_triplets(idx)

        # Load neg_samples array of shape: [neg_samples, channels, self.crop_size_cs_samples]
        if self.neg_sampling == 'naive':
            neg_samples = self.naive_neg_sampling()

        elif self.neg_sampling == 'within_patient':
            neg_samples = self.within_patient_neg_sampling(subject)


        # Load pos_samples array of shape: [pos_samples, channels, self.crop_size_cs_samples]
        pos_samples = self.positive_sampling(subject, start_sample_idx+self.crop_size_samples) 
        return x, y, neg_samples, pos_samples


