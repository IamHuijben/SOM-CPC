from pathlib import Path
import numpy as np
import yaml
import copy
import math
import warnings
from datetime import datetime
import h5py
import numpy as np
import yaml 
from scipy.signal import butter, filtfilt, resample_poly
import mne
from mne.io import read_raw_edf
import dhedfreader

class PreprocessData():
    """

    Args:
        load_dir (str): Absolute path indicating where the data should be loaded from.
        read_channels (list): List of strings or single string indicating the channels to be preprocessed.
        save_channels (list): List of strings or single string indicating the names of the channels for saving.
        read_subjects (list): List containing the (dataset-dependent) strings indicating the subjects to be preprocessed.
        save_subjects (list): List containing the strings to store the subjects.
        power_line (int): Number of Hertz where the powerline interference is to be expected for this dataset. Will be one of {50, 60}.
        nan_value (float or str): Indicates how nan values are indicated in the current dataset. If string, it must be 'nan'.
        low_fc (int): Low cutoff frequency used for the bandpass filter.
        high_fc (int): High cutoff frequency used for the bandpass filter.
        order_lpf (int): Order of the low-pass butterworth filter.
        order_hpf (int): Order of the high-pass butterworth filter.
        percentile_95 (bool):    If set to True, the minimum and maximum value are used for renormalization. If False, the 95 percentile is used.
        sample_rate_orig (list or int): The original sampling rate of the data. A list if different channels have different rates.
        sample_rate_new (int): The new sampling rate of the data to which the data must be resampled.
        orig_unit (str): Unit in which data is recording. One from {micro-v, v}
        outlier_threshold (float, optional): Sets abs. values higher than this threshold to zero. Defaults to 500 micro-V. 
        sec_per_annotation (int): Number of seconds of data that are annotation with one label. Default to 30 sec. 
        references (list or str, optional): List of strings (same lengths as channels) or single string indiciating the reference channels to be used. If None, the channels are assumed to already be referenced.

    Returns:
        preprocessed_input (dict): Dictionary containing the data from the requested channels.
        sleep_stages (np.ndarray): 1D numpy array containing the scored sleep stages.
    """
            
    def __init__(self, load_dir, read_channels, save_channels, read_subjects, save_subjects, power_line, nan_value, low_fc, high_fc, sample_rate_orig, sample_rate_new, orig_unit, outlier_threshold, sec_per_annotation, order_lpf, order_hpf, percentile_95, references=None, name_of_run=None):
        # Data-related parameters
        self.read_channels = read_channels
        self.save_channels = save_channels
        self.read_subjects = read_subjects
        self.save_subjects = save_subjects
        self.references = references
        self.sec_per_annotation = sec_per_annotation
        self.name_of_run = name_of_run

        # Loading and saving parameters
        self.load_dir = load_dir
        self.save_dir = Path.cwd() / 'data'/ 'mass_preprocessed'
        self.save_dir.mkdir(exist_ok=True, parents=True)

        # Sampling parameters
        self.sample_rate_orig = sample_rate_orig
        self.sample_rate_new = sample_rate_new

        ### Filtering ###
        self.nan_idx = None
        self.nan_value = nan_value
        self.power_line = power_line
        self.low_fc = low_fc
        self.high_fc = high_fc
        self.order_lpf = order_lpf 
        self.order_hpf = order_hpf

        # Normalization parameters
        self.orig_unit = orig_unit.lower()
        self.threshold = outlier_threshold
        self.percentile_95 = percentile_95

        self.sleep_stages = {}
        self.reference_data = {}
        self.raw_channel_data = {}
        self.preprocessed_input = {}
  

    def window_idx_to_samples(self, fs, sec_per_annotation, nr_epochs):
        return nr_epochs * fs * sec_per_annotation

    def align_signals_and_annotations(self, save_subject):
        """Crop the signal at the end such that the number of samples exactly match the number of labels, or delete labels at the end if unnecessary nan labels are attached that do not belong to a signal.
        """

        nr_labels = len(self.sleep_stages[save_subject])
        nr_samples = self.window_idx_to_samples(self.current_sample_rate, self.sec_per_annotation, nr_labels)
        removed_labels = False
        observed_too_much_data = False

        for ch, data in self.preprocessed_input[save_subject].items():
            if not observed_too_much_data and nr_samples <= 0.95*len(data): #Only give this warning once per patient.
                warnings.warn(f'More than 5% of the data of {save_subject} is being thrown away due to no labels being present. Check if this is correct.')
                observed_too_much_data = True
            if len(data) > nr_samples:
                self.preprocessed_input[save_subject][ch] = data[:nr_samples]
            elif len(data) < nr_samples:
                data_for_X_ann = int(len(data) // (self.current_sample_rate * self.sec_per_annotation))
                too_many_ann = nr_labels - data_for_X_ann
                if not removed_labels:
                    if not np.allclose(self.sleep_stages[save_subject][-too_many_ann:],0):
                        warnings.warn('The number of labels are more than is expected from the length of the data and the labels that are too many are not nans. Check this!')
                    
                    self.sleep_stages[save_subject] = self.sleep_stages[save_subject][:-too_many_ann]
                    removed_labels = True

                #In case the last label belonged to less than a full epoch, we also need to crop some data.
                cropped_data_len = data_for_X_ann * self.current_sample_rate * self.sec_per_annotation
                self.preprocessed_input[save_subject][ch] = data[:cropped_data_len]

        self.performed_functions['align_signals_and_annotations'] = True

    def interpolate_missing_data(self, save_subject, nan_value='nan'):
        """
        Missing data (often represented as zeros or nans) results in filtering artefacts. This function linearly interpolates these missing data points.

        Args:
            save_subject (str): String denoting the subject
            nan_value (float or str): Value in which this dataset annotates nans. If string, it must be 'nan'.
        """

        for ch, data in self.preprocessed_input[save_subject].items():
            if nan_value == 'nan':
                self.nan_idx = np.nonzero(np.isnan(data) + np.isinf(data))[0]
                non_nan_idx = np.nonzero(np.isfinite(data))[0]
            else:
                self.nan_idx = np.nonzero(data == nan_value)[0]
                non_nan_idx = np.nonzero(data != nan_value)[0]
            non_nan_vals = data[non_nan_idx]

            if list(self.nan_idx):
                interp_vals = np.interp(self.nan_idx, non_nan_idx, non_nan_vals)
                data[self.nan_idx] = interp_vals
                self.preprocessed_input[save_subject][ch] = data

        self.performed_functions['interpolate_missing_data'] = True

    def remove_leading_and_ending_data_with_nan_labels(self, save_subject):
        """ 
        Remove the sleep epochs at the start and end that were labeled as nans as this data is probably of low quality. 
        Assumed that the labels have been standardized already, i.e. nan equals 0.
        """
        
        sleep_stages = self.sleep_stages[save_subject]
        
        # Subtract the starting and end epochs with nan labels        
        self.nans_at_start = len(sleep_stages) - len(np.trim_zeros(sleep_stages, 'f'))
        nans_at_end = len(sleep_stages) - len(np.trim_zeros(sleep_stages, 'b'))

        # Convert epochs to samples to select the remove the corresponding data samples from the data as well.
        if self.nans_at_start > 0:
            remove_start_samples = self.window_idx_to_samples(
                self.current_sample_rate, self.sec_per_annotation, self.nans_at_start)

            # Update the sample nan idx accordingly
            if self.nan_idx is not None:
                self.nan_idx = self.nan_idx - remove_start_samples

        if nans_at_end > 0:
            remove_end_samples = self.window_idx_to_samples(
                self.current_sample_rate, self.sec_per_annotation, nans_at_end)

        self.sleep_stages[save_subject] = np.trim_zeros(sleep_stages, 'fb')
        
        if (self.nans_at_start > 0) and (nans_at_end > 0):
            for ch, data in self.preprocessed_input[save_subject].items():
                self.preprocessed_input[save_subject][ch] = data[remove_start_samples: -remove_end_samples]
        elif self.nans_at_start > 0:
            for ch, data in self.preprocessed_input[save_subject].items():
                self.preprocessed_input[save_subject][ch] = data[remove_start_samples:]
        elif nans_at_end > 0:
            for ch, data in self.preprocessed_input[save_subject].items():
                self.preprocessed_input[save_subject][ch] = data[: -remove_end_samples]
        else:
            for ch, data in self.preprocessed_input[save_subject].items():
                self.preprocessed_input[save_subject][ch] = data

        self.performed_functions['remove_leading_and_ending_data_with_nan_labels'] = True

    def balance_wake_epochs(self, save_subject):
        """
        Remove start and end wake epochs, such that amount of wake is not bigger than number of epochs on present averaged for all classes.
        This function assumes that the labels have already been standardized.
        """
        sleep_stages = self.sleep_stages[save_subject]
        # Check if wake is happening more than the largest available stage (non-wake and non-nan)
        sleep_stage_code, counts = np.unique(sleep_stages, return_counts=True)
        sleep_stage_hist = dict(zip(sleep_stage_code, counts))

        # Assume already standardized sleep stage scores: nan = 0, N1 = 1, N2 = 2, N3 = 3,  W = 4, REM = 5, 
        av_nr_epochs = int(np.ceil(np.mean([sleep_stage_hist.get(1,0), sleep_stage_hist.get(2,0), sleep_stage_hist.get(3,0), sleep_stage_hist.get(5,0)])))
        too_many_wake_epochs = sleep_stage_hist.get(4,0) - av_nr_epochs
        
        if too_many_wake_epochs > 0:
            first_non_wake_epoch_idx = np.argmax(sleep_stages != 4)

            # Find the first epoch from back that is a wake epoch and compute its index counting from the front based on the total number of sleep epochs.
            first_none_wake_epoch_idx_from_back = np.argmax(
                np.flip(sleep_stages != 4))

            # If the sequence starts with wake epochs, we can remove epochs if the wake class is overrepresented.
            if first_non_wake_epoch_idx > 1:
                remove_leading_wake_epochs = min(first_non_wake_epoch_idx - 1, too_many_wake_epochs) #Leave at least one wake epoch to learn the transition

                remove_start_samples = self.window_idx_to_samples(
                    self.current_sample_rate, self.sec_per_annotation, remove_leading_wake_epochs)

                # Already remove the labels
                sleep_stages = sleep_stages[remove_leading_wake_epochs:]

                # Update the sample nan idx accordingly
                if self.nan_idx is not None:
                    self.nan_idx = self.nan_idx - remove_start_samples
                    
            else:
                remove_leading_wake_epochs = 0

            # Compute how many overrepresented wake epochs we have left.
            too_many_wake_epochs_left = too_many_wake_epochs - remove_leading_wake_epochs
            
            # We can possibly remove end wake epochs.
            if (too_many_wake_epochs_left > 0) and (first_none_wake_epoch_idx_from_back > 1):

                # Leave at least one wake epoch to learn the transition
                remove_ending_wake_epochs = min(
                    first_none_wake_epoch_idx_from_back - 1, too_many_wake_epochs_left)

                remove_end_samples = self.window_idx_to_samples(
                    self.current_sample_rate, self.sec_per_annotation, remove_ending_wake_epochs)

                sleep_stages = sleep_stages[:-remove_ending_wake_epochs]
            else:
                remove_ending_wake_epochs = 0            

            # Store back the adapted label vector
            self.sleep_stages[save_subject] = sleep_stages

            for ch, data in self.preprocessed_input[save_subject].items():
                if (remove_leading_wake_epochs > 0) and (remove_ending_wake_epochs > 0):
                    assert remove_start_samples > 0
                    assert remove_end_samples > 0
                    self.preprocessed_input[save_subject][ch] = data[remove_start_samples: -remove_end_samples]
                elif remove_leading_wake_epochs > 0:
                    assert remove_start_samples > 0
                    self.preprocessed_input[save_subject][ch] = data[remove_start_samples:]
                elif remove_ending_wake_epochs > 0:
                    self.preprocessed_input[save_subject][ch] = data[:-remove_end_samples]
                else:
                    self.preprocessed_input[save_subject][ch] = data
            
        self.performed_functions['balance_wake_epochs'] = True

    def filtering(self, save_subject):
        """
        Filters the ExG data using a Butterworth bandpass filter [low_fc, sample_rate_new/2] and a notch filter at 50 Hz (or 60 Hz for data recorded in US).
        """
        # Band pass filter between low_fc and high_fc.
        # Low pass anti-aliasing filter
        assert self.high_fc <= self.sample_rate_new/2, 'The high cutoff frequency (high_fc) is larger than twice the new sampling frequency.'
        b, a = butter(self.order_lpf,self.high_fc/(self.current_sample_rate/2), 'lowpass')
        if self.low_fc > 0:
            d, c = butter(self.order_hpf,self.low_fc/(self.current_sample_rate/2), 'highpass')
    
        # Notch filter for powerline interference.
        if self.current_sample_rate > (self.power_line*2):
            f, e = butter(5, [(self.power_line-1)/(
                    self.current_sample_rate/2), (self.power_line+1)/(self.current_sample_rate/2)], 'stop')

        for ch, data in self.preprocessed_input[save_subject].items():
            assert len(data.shape) == 1, f'The stored PSG data is not 1D, so filtering might happen over the wrong axis.'

            data = filtfilt(b, a, data, axis=0) #lpf
            if self.low_fc > 0:
                data = filtfilt(d, c, data, axis=0) #hpf
            if self.current_sample_rate > (self.power_line*2):
                data = filtfilt(f, e, data, axis=0) #notch filter for powerline.
                
            self.preprocessed_input[save_subject][ch] = data

        self.performed_functions['filtering'] = True

    def nan_to_zero(self, save_subject):
        """
        Change nans (or interpolated nans) in the ExG data to zeros as the model can not deal with nans in the input.
        """
        assert not self.performed_functions.get('resampling', False), 'You can not replace the nans by zero after resampling has already taken place.'

        if self.nan_idx is not None:
            # By reducing the nan_idxs due to removal of start samples, the indices can have become negative. Remove those:
            self.nan_idx = self.nan_idx[self.nan_idx >= 0]

        for ch, data in self.preprocessed_input[save_subject].items():
            if self.nan_idx is not None: #If self.nan_idx is written the nans were interpolated before filtering.
                
                # By removing end samples, the nan idxs can contain samples that do not exist anymore, so remove then.
                nan_idx = self.nan_idx[self.nan_idx < len(data)]
                data[nan_idx] = 0.
            else:
                data[np.isnan(data)] = 0.

            self.preprocessed_input[save_subject][ch] = data

        self.performed_functions['nan_to_zero'] = True

    def init_upsample_subset_of_data(self, save_subject):
        """ In some datasets, some of the channels are sampled at a much lower frequency, for example 1Hz. In those cases, upsample those channels to the indicated original sampling rate in order to be able to process all channels in a same way.
        """
        
        for ch, data in self.preprocessed_input[save_subject].items():
            fs_orig = self.sampling_freqs[ch] 
            if fs_orig != self.sample_rate_orig:
                warnings.warn(f'The {ch} channel was sampled with {fs_orig} Hz, while you indicated {self.sample_rate_orig}')

                if fs_orig < self.sample_rate_orig: #need upsampling
                    assert self.sample_rate_orig % fs_orig == 0
                    up =  int(self.sample_rate_orig // fs_orig)
                    warnings.warn(f'The {ch} channel will be upsampled with a factor {up} before processing starts.')

                    self.preprocessed_input[save_subject][ch] = resample_poly(data, up=up, down=1)                


    def resampling(self, save_subject):
        """
        Resample the original signal to the new sampling rate.
        """
        assert self.performed_functions.get('nan_to_zero', False), 'First place the nans back to zero before applying resampling.'

        if self.current_sample_rate > self.sample_rate_new:
            # Integer downsampling
            if self.sample_rate_orig % self.sample_rate_new == 0:
                for ch, data in self.preprocessed_input[save_subject].items():
                    self.preprocessed_input[save_subject][ch] = np.array(
                        list(data)[0::(self.current_sample_rate // self.sample_rate_new)])
                
            # Non-integer downsampling
            else:
                up = self.sample_rate_new//math.gcd(self.current_sample_rate, self.sample_rate_new)
                down = self.sample_rate_orig//math.gcd(self.current_sample_rate, self.sample_rate_new)
                
                for ch, data in self.preprocessed_input[save_subject].items():
                    upsampled_sig = resample_poly(data, up=up, down=1)
                    self.preprocessed_input[save_subject][ch] = np.array(
                        list(upsampled_sig)[0::down])
                        
            self.current_sample_rate = self.sample_rate_new
        elif self.current_sample_rate < self.sample_rate_new:
            warnings.warn(f'The requested sampling rate is higher than the original sampling rate of {self.sample_rate_orig}')
        
        self.performed_functions['resampling'] = True

    def normalize(self, save_subject):
        """
        Renormalize the data to create zero-mean data between -1 and 1 and remove outliers with an absolute value higher than self.threshold. The data is assumed to be symmetric around its mean. 
        """

        for ch, data in self.preprocessed_input[save_subject].items():
            
            if self.orig_unit == 'micro-v':
                data = data / (1e6)

            # Remove extremely high artifacts
            if self.threshold:
                # Use median instead of mean here as the data still contains outliers and filter them out.
                shifted_data = data -  np.median(data) 
                data[abs(shifted_data) > self.threshold] = 0

            # Make the data zero-mean.
            unbiased_data = data - np.mean(data) 

            if self.percentile_95: 
                max_val = np.percentile(abs(unbiased_data), q=95) 
            else:
                max_val = np.max(abs(unbiased_data))

            self.preprocessed_input[save_subject][ch] = unbiased_data / max_val

        self.performed_functions['normalize'] = True

    def convert_to_float32(self, save_subject):
        for ch, data in self.preprocessed_input[save_subject].items():
            self.preprocessed_input[save_subject][ch] = np.float32(data)

        self.performed_functions['convert_to_float32'] = True

    def save_preprocessed_data(self, save_subject):
        

        hf = h5py.File(self.save_dir / f'{save_subject}.h5', 'a')

        for ch, data in self.preprocessed_input[save_subject].items():
            if ch in hf:
                warnings.warn(f'{ch} was already part of this stored hdf5 file. Ommitting it now!')
            else:
                hf.create_dataset(ch, data=data)
                assert (len(data) / (self.current_sample_rate * self.sec_per_annotation)) == len(self.sleep_stages[save_subject]), f'The number of sleep stages does not match the number of time samples for subject {save_subject}'
        
        if 'nr_samples' in hf:
            assert hf['nr_samples'][()][0] == len(data)
        else:
            hf.create_dataset('nr_samples', data=[len(data)])

        if 'removed_nans_at_start' in hf:
            assert hf['removed_nans_at_start'][()] == self.nans_at_start
        else:
            hf.create_dataset('removed_nans_at_start', data=self.nans_at_start)

        if 'fs' in hf:
            assert hf['fs'][()][0] == self.current_sample_rate
        else:
            hf.create_dataset('fs', data=[self.current_sample_rate])


        if 'labels' in hf:
            assert np.allclose(hf['labels'][()], self.sleep_stages[save_subject])
        else:
            hf.create_dataset('labels', data=self.sleep_stages[save_subject])

        hf.close()



    def save_meta_data(self):

        preprocessing_specifics = copy.deepcopy(self.__dict__)

        # Remove some meta-data that does not need to be stored.
        preprocessing_specifics.pop('raw_channel_data')
        preprocessing_specifics.pop('preprocessed_input')
        preprocessing_specifics.pop('reference_data')
        preprocessing_specifics.pop('sleep_stages')
        preprocessing_specifics.pop('load_dir')
        preprocessing_specifics.pop('save_dir')
        preprocessing_specifics.pop('nan_idx')
        preprocessing_specifics.pop('sampling_freqs')
        preprocessing_specifics.pop('read_subjects')
        preprocessing_specifics.pop('save_subjects')
        preprocessing_specifics.pop('nans_at_start')

        for key, value in preprocessing_specifics.items():
            if isinstance(value, np.ndarray):
                preprocessing_specifics[key] = value.tolist()

        if self.name_of_run is not None:
            with open(self.save_dir / f'preprocessing_specifics_{self.name_of_run}.yml', 'w') as outfile:
                yaml.dump(preprocessing_specifics, outfile, default_flow_style=False)
        else:
            with open(self.save_dir / f'preprocessing_specifics.yml', 'w') as outfile:
                yaml.dump(preprocessing_specifics, outfile, default_flow_style=False)

    def preprocess(self, preprocessing_steps = {}):
        """
        Applied the full preprocessing pipeline for all subjects provided in the list during initialization of this class.

        Args:
            preprocessing_steps (dict, optional): Each key refers to one preprocessing function and its value is either True or False. To be used to skip preprocessing steps for debugging or testing with different preprocessings.
        """

        for read_subject, save_subject in zip(self.read_subjects, self.save_subjects):
            
            # Keep track of which functions have been performed, as some are dependent on others.
            self.performed_functions = {}
            print(f'Subject: {read_subject}')
            self.current_sample_rate = np.nan

            self.raw_channel_data[save_subject] = {}
            self.preprocessed_input[save_subject] = {}
            self.reference_data = {}
            
            # Apply all preprocessing functions:
            self.load_x_y_pairs(read_subject, save_subject)

            if self.preprocessed_input[save_subject]: # Check if the subject was loaded. If not present in the folder, the dict here is empty.
                if preprocessing_steps.get('init_upsample_subset_of_data',True):
                    self.init_upsample_subset_of_data(save_subject)
                
                self.flexible_referencing(read_subject, save_subject)
                self.interpolate_missing_data(save_subject, nan_value=self.nan_value)
                
                if preprocessing_steps.get('align_signals_and_annotations', True):
                    self.align_signals_and_annotations(save_subject)
                if preprocessing_steps.get('remove_leading_and_ending_data_with_nan_labels', True):
                    self.remove_leading_and_ending_data_with_nan_labels(
                        save_subject)
                if preprocessing_steps.get('balance_wake_epochs', True):
                    self.balance_wake_epochs(save_subject)
                if preprocessing_steps.get('filtering', True):
                    self.filtering(save_subject)
                if preprocessing_steps.get('nan_to_zero', True):
                    self.nan_to_zero(save_subject)
                if preprocessing_steps.get('resampling', True):
                    self.resampling(save_subject)
                if preprocessing_steps.get('normalize', True):
                    self.normalize(save_subject)
                if preprocessing_steps.get('convert_to_float32', True):
                    self.convert_to_float32(save_subject)
                if preprocessing_steps.get('save_preprocessed_data', True):
                    self.save_preprocessed_data(save_subject)
            else:
                # Remove the still empty dictionary if no data was loaded for this subject.
                self.raw_channel_data.pop(save_subject)
                self.preprocessed_input.pop(save_subject)


        if preprocessing_steps.get('save_preprocessed_data', True):
            self.save_meta_data()


class PreprocessData_mass(PreprocessData):
    def __init__(self, load_dir, subset, channels, references, subjects, low_fc, high_fc, sample_rate_orig, sample_rate_new, outlier_threshold, order_lpf, order_hpf, percentile_95, name_of_run,**kwargs):
        power_line = 60
        sec_per_annotation = 30
        orig_unit = 'micro-v'
        nan_value = 'nan' #It seems that either exact zeros or NaNs do not occur in this dataset. 
        self.dataset = 'mass'

        self.subset = subset
        assert self.subset in [1,3]
        load_dir = Path(load_dir) / f'SS{self.subset}_EDF'

        read_subjects, save_subjects, read_channels, save_channels = self.standardize_subjects_and_channel_names(
            subjects, channels)

        super().__init__(load_dir=load_dir, read_channels=read_channels, save_channels=save_channels, references=references, read_subjects=read_subjects, save_subjects=save_subjects, power_line=power_line, nan_value=nan_value, low_fc=low_fc, high_fc=high_fc, sample_rate_orig=sample_rate_orig,
                         sample_rate_new=sample_rate_new, orig_unit=orig_unit, outlier_threshold=outlier_threshold, sec_per_annotation=sec_per_annotation, order_lpf=order_lpf, order_hpf=order_hpf, percentile_95=percentile_95, name_of_run=name_of_run)
            

    def standardize_subjects_and_channel_names(self, subjects, channels):
        """Convert both the list of subjects and channels to a list containing the strings to read and a list with standardized strings for saving.

        Args:
            channels (list or str): List of strings or single string indicating the channels to be preprocessed.
            subjects (list): List containing the subject indices to be processed.

        Returns:
            lists: Lists containing the (dataset-dependent) read and (dataset-independent) save strings for subjects and channels
        """
        read_subjects = []
        for subject_idx in subjects:
            if len(str(subject_idx)) == 1:
                read_subjects.append(f'01-0{self.subset}-000{subject_idx}')
            elif len(str(subject_idx)) == 2:
                read_subjects.append(f'01-0{self.subset}-00{subject_idx}')
            else:
                raise NotImplementedError

        save_subjects = ['s'+str(subject_idx) for subject_idx in subjects]

        read_channels = channels
        save_channels = [ch.lower() for ch in channels]
        return read_subjects, save_subjects, read_channels, save_channels

    def load_x_y_pairs(self, read_subject, save_subject):
        """
        Loads the PSG data and corresponding labels (if available). The PSG data is loaded as a dictionary, where each key is a channel, with a 1D time series as value.

        Args:
            read_subject (str): Unique (dataset-dependent) string describing the subject.
            save_subject (str): Unique standardized string describing the subject.
        
        """
        
        psg_file = Path(self.load_dir) / (read_subject + ' PSG.edf')
        label_file = Path(self.load_dir) / (read_subject + ' Base.edf')
    
        # Load the PSG data and its meta information
        f = open(psg_file, 'r', encoding="Latin-1")
        reader_raw = dhedfreader.BaseEDFReader(f)
        reader_raw.read_header()
        h_raw = reader_raw.header
        raw_start_dt = datetime.strptime(h_raw['date_time'], "%Y-%m-%d %H:%M:%S")
        f.close()
        
        sampling_freqs = dict(zip(h_raw['label'],np.array(h_raw['n_samples_per_record'])/h_raw['record_length']))
        self.sampling_freqs = {}
        other_fs_ch = []
        for read_ch, save_ch in zip(self.read_channels, self.save_channels):
            self.sampling_freqs[save_ch] = sampling_freqs[read_ch]
            if self.dataset == 'mass' and read_ch[:8] == 'EMG Chin':
                assert sampling_freqs[read_ch] == 128 or sampling_freqs[read_ch] == 256
                
            else:
                assert self.sample_rate_orig == sampling_freqs[read_ch] 
                if self.sampling_freqs[save_ch] < self.sample_rate_orig:
                    other_fs_ch.append(read_ch)
                elif self.sampling_freqs[save_ch] > self.sample_rate_orig:
                    assert ValueError, f'The indicated original sampling rate is lower than the true sampling rate of the {read_ch} channel.'
        self.current_sample_rate = int(sampling_freqs[read_ch])

        # Load the channels separately that have a deviating sampling frequency than the given sample_rate_orig otherwise it does not concatenate in one dataframe.
        raw = read_raw_edf(str(psg_file), preload=True, exclude=set(h_raw['label'])-set(self.read_channels) | set(other_fs_ch), stim_channel=None).to_data_frame()
        if len(other_fs_ch) > 0:
            raw_other_fs = read_raw_edf(str(psg_file), preload=True, exclude=set(h_raw['label'])-set(other_fs_ch), stim_channel=None).to_data_frame()

        for read_ch, save_ch in zip(self.read_channels, self.save_channels):
            if read_ch in raw:
                self.raw_channel_data[save_subject][save_ch] = raw[read_ch].values
            else:           
                self.raw_channel_data[save_subject][save_ch] = raw_other_fs[read_ch].values

        # Load Hypnogram and its meta information
        ann = mne.read_annotations(str(label_file)).description

        # Assert that raw and annotation files start at the same time and thus belong to each other.
        f = open(label_file, 'r', encoding="Latin-1")
        reader_ann = dhedfreader.BaseEDFReader(f)
        reader_ann.read_header()
        h_ann = reader_ann.header
        ann_start_dt = datetime.strptime(h_ann['date_time'], "%Y-%m-%d %H:%M:%S")
        f.close()
        assert raw_start_dt == ann_start_dt         
             
        self.sleep_stages[save_subject] = self.standardize_label_codes(ann)

        # Copy the raw data to the preprocessed_input dict and start all the preprocessing
        self.preprocessed_input[save_subject] = copy.deepcopy(self.raw_channel_data[save_subject])

    def standardize_label_codes(self, labels):
        """
            Ensure that the annotated sleep classes are labeled according to the following system independent of the used dataset.
            0: nan
            1: N1
            2: N2
            3: N3
            4: Wake
            5: REM

            Args:
                labels (np.ndarray)
        """
        self.nr_classes = 6

        # Recode NaN: ? --> 0
        # Recode Wake: W --> 4
        # Recode REM: R --> 5
        labels[labels == 'Sleep stage ?'] = 0
        labels[labels == 'Sleep stage W'] = 4
        labels[labels == 'Sleep stage R'] = 5

        labels[labels == 'Sleep stage 1'] = 1
        labels[labels == 'Sleep stage 2'] = 2
        labels[labels == 'Sleep stage 3'] = 3
        
        self.performed_functions['standardize_label_codes'] = True
        return labels.astype(int)

    def flexible_referencing(self, read_subject, save_subject):
        """
        References the channels to their corresponding reference.

        Args:
            read_subject (str): Unique (dataset-dependent) string describing the subject.
            save_subject (str): Unique standardized string describing the subject.
        
        """
        referenced_input = {save_subject:{}}
        for ch, ref_ch in zip(self.save_channels, self.references): 
            if ref_ch == '-' or ref_ch is None:
                referenced_input[save_subject].update({f'{ch}': self.preprocessed_input[save_subject][ch]})
            else:
                file = Path(self.load_dir) / (read_subject + ' PSG.edf')
                f = open(file, 'r', encoding="Latin-1")
                reader = dhedfreader.BaseEDFReader(f)
                reader.read_header()
                header = reader.header
                f.close()

                # Load the reference channel
                df = read_raw_edf(str(file), preload=True, exclude = set(header['label']) - set([ref_ch]), stim_channel=None) 
                df = df.to_data_frame()
                ref_data = df[ref_ch].values
                referenced_input[save_subject].update({f'{ch}_{ref_ch.lower()}': self.preprocessed_input[save_subject][ch] - ref_data})

        self.preprocessed_input[save_subject] = referenced_input[save_subject]        
        self.performed_functions['referencing'] = True

if __name__ == "__main__":

    """
    This file runs preprocessing of the MASS dataset. 
    The load_dir variable should point to the folder where the subfolders of the different MASS subsets are stored (e.g. SS3_EDF)
    After running this script, manually place each of the subjects in one of the train/val/test subfolders that are created.
    """
    
    load_dir = ... #Fill in
    
    # Run in groups of subjects to prevent OOM. Subject 43 and 49 are missing from thisdataset.
    subjects_list = [np.arange(1,20), np.arange(20,43),np.arange(44,49), np.arange(50,60), np.arange(60,65)] 

    settings_file = open(str(Path(__file__).parent / (f'preprocessing_settings_mass.yml'))) 
    settings = yaml.load(settings_file, Loader=yaml.FullLoader)

    save_dir = Path.cwd() / 'data'/ 'mass_preprocessed'
    save_dir.mkdir(parents=False, exist_ok=True)
    (save_dir/'train').mkdir(parents=False, exist_ok=True)
    (save_dir/'val').mkdir(parents=False, exist_ok=True)
    (save_dir/'test').mkdir(parents=False, exist_ok=True)

    for modality, modality_specific_settings in settings.items():
        for subjects in subjects_list:
                preprocess_inst = PreprocessData_mass(
                                    load_dir= load_dir, 
                                    subset = modality_specific_settings['subset'],
                                    channels=modality_specific_settings['all_channels'], 
                                    references=modality_specific_settings['all_references'], 
                                    subjects=subjects, 
                                    low_fc=modality_specific_settings['low_fc'],
                                    high_fc=modality_specific_settings['high_fc'],
                                    sample_rate_orig=modality_specific_settings['sample_rate_orig'],
                                    sample_rate_new=modality_specific_settings['sample_rate_new'],
                                    outlier_threshold=modality_specific_settings['outlier_threshold'],
                                    order_lpf = modality_specific_settings.get('order_lpf', 5), 
                                    order_hpf = modality_specific_settings.get('order_hpf', 5),
                                    percentile_95=True, 
                                    name_of_run = modality_specific_settings.get('channel_set_name')
                                    )

                preprocess_inst.preprocess()
        
    print('Done! Move the subjects to the train, val and test subfolders in the way you want to make your split.')