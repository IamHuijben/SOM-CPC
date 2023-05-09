import copy
import warnings
from datetime import date
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml  # requires pyyaml
import soundfile as sf        
from tqdm import tqdm


class PreprocessLibriSpeech():
    """
    Args:
        load_dir (str): Absolute path indicating where the data should be loaded from.
        subset_splits (dict): Dict containing the full path to the split files for each of the data folds.
        sample_rate_orig (list or int): The original sampling rate of the data. A list if different channels have different rates.
        sec_per_annotation (int): Number of seconds of data that are annotation with one label. Default to 30 sec. 
    """
            
    def __init__(self, load_dir, subset_splits, sec_per_annotation, sample_rate_orig):
        # Data-related parameters
        self.dataset = 'librispeech'
        self.sec_per_annotation = sec_per_annotation

        # Loading and saving parameters
        self.load_dir = Path(load_dir)
        self.find_all_sound_paths()

        self.label_dir = self.load_dir.parent / 'converted_aligned_phones.txt'
        self.load_labels()

        self.subset_splits = subset_splits
        self.load_split_files()

        self.save_dir = Path.cwd() / 'data'/ 'Librispeech_preprocessed'
        self.save_dir.mkdir(exist_ok=True, parents=True)
        (self.save_dir / 'train').mkdir(exist_ok=True, parents=True)
        if 'val' in self.subset_splits:
            (self.save_dir / 'val').mkdir(exist_ok=True, parents=True)
        if 'test' in self.subset_splits:
            (self.save_dir / 'test').mkdir(exist_ok=True, parents=True)

        # Sampling parameters
        self.sample_rate_orig = sample_rate_orig

    def window_idx_to_samples(self, fs, sec_per_annotation, nr_epochs):
        return nr_epochs * fs * sec_per_annotation

    def find_all_sound_paths(self):
        self.all_sound_files = sorted(self.load_dir.rglob('*flac'))

        # Create a dictionary with as keys the unique identifiers, and as values the corresponding paths. 
        self.file_path_dictionary = {}
        for file_path in self.all_sound_files:
            assert file_path.stem not in self.file_path_dictionary
            self.file_path_dictionary[file_path.stem] = file_path

    def load_waveform(self, file_id):
        """
        Loads the sound file and the correspond labels.

        Args:
            file_id (str): Unique file identifier        
        """

        ### Load data
        self.preprocessed_input = {}
        data, samplerate = sf.read(str(self.file_path_dictionary[file_id]))
        assert samplerate == self.sample_rate_orig
        self.current_sample_rate = self.sample_rate_orig
        self.preprocessed_input[file_id] = data

    def load_split_files(self):
        self.data_folds = {}
        for data_fold, split_path in self.subset_splits.items():
            with open(split_path) as f:
                all_unique_ids = f.readlines()
            for file_id in all_unique_ids:
                if file_id[-1:] == '\n': file_id = file_id[:-1]
                self.data_folds[file_id] = data_fold

    def load_labels(self):
        print('Processing the labels ... ')

        self.phone_label_dictionary, self.speaker_id_label_dictionary, self.speaker_sex_label_dictionary = {}, {}, {}    

        ### Load speaker information
        with open(Path(self.load_dir) / 'LibriSpeech' / 'SPEAKERS.txt') as f:
            speaker_information = f.readlines()[12:]

            speaker_information_ext = [ [] for _ in range(len(speaker_information)) ]
            for line_idx, speaker_line in enumerate(speaker_information):
                if '|CBW|Simon' in speaker_line:
                    speaker_line_split = ['60','M','train-clean-100','20.18','|CBW|Simon']
                else:
                    speaker_line_split = speaker_line.split("|")
                speaker_information_ext[line_idx]  = [substring.strip() for substring in speaker_line_split]
            speaker_df = pd.DataFrame(speaker_information_ext, columns=['speaker_id', 'gender', 'dataset' , 'duration', 'name'])


        ### Load phone labels
        with open(self.label_dir) as f:
            lines = f.readlines()

            # Read each line from the label txt file and split all strings into separate labels, ommitting the \n character
            # Each line represents one labeled sequence
            for line in tqdm(lines):
                split_line_in_substrings = line.split("\n")[0].split(" ")
                file_id = split_line_in_substrings[0]
                self.phone_label_dictionary[file_id] = [int(label) for label in split_line_in_substrings[1:]]
                self.speaker_id_label_dictionary[file_id] = int(file_id.split("-")[0])

                # Encode speaker sex label: 0:Male, 1:Female
                gender = speaker_df[speaker_df['speaker_id'] == str(self.speaker_id_label_dictionary[file_id])]['gender'].values[0]
                assert speaker_df[speaker_df['speaker_id'] == str(self.speaker_id_label_dictionary[file_id])]['dataset'].values[0] == 'train-clean-100'
                
                if gender == 'M':
                    self.speaker_sex_label_dictionary[file_id] = 0
                elif gender == 'F':
                    self.speaker_sex_label_dictionary[file_id] = 1
                else:
                    raise ValueError


    def align_signals_and_annotations(self, file_id):
        """Crop the signal at the end such that the number of samples exactly match the number of labels, or delete labels at the end if unnecessary nan labels are attached that do not belong to a signal.
        """

        nr_labels = len(self.phone_label_dictionary[file_id])
        nr_samples = int(self.window_idx_to_samples(self.current_sample_rate, self.sec_per_annotation, nr_labels))
        
        if nr_samples <= 0.95*len(self.preprocessed_input[file_id]):
            warnings.warn(f'More than 5% of the data of {file_id} is being thrown away due to no labels being present. Check if this is correct.')

        if len(self.preprocessed_input[file_id]) > nr_samples: #More data than labels
            self.preprocessed_input[file_id] = self.preprocessed_input[file_id][:nr_samples]
        elif len(self.preprocessed_input[file_id]) < nr_samples: # More labels than data
            data_for_X_ann = int(len(self.preprocessed_input[file_id]) // (self.current_sample_rate * self.sec_per_annotation))
            too_many_ann = nr_labels - data_for_X_ann

            warnings.warn(f'The number of labels are more than is expected from the length of the data for {file_id}. Check this!')
            self.phone_label_dictionary[file_id] = self.phone_label_dictionary[file_id][:-too_many_ann]

            #In case the last label belonged to less than a full epoch, we also need to crop some data.
            cropped_data_len = data_for_X_ann * self.current_sample_rate * self.sec_per_annotation
            self.preprocessed_input[file_id] = self.preprocessed_input[:cropped_data_len]

        self.performed_functions['align_signals_and_annotations'] = True

    def convert_to_float32(self, file_id):
        self.preprocessed_input[file_id] = np.float32(self.preprocessed_input[file_id])
        self.performed_functions['convert_to_float32'] = True

    def save_preprocessed_data(self, file_id):
        
        if self.save_dir is not None: 
            hf = h5py.File(self.save_dir / f'{self.data_folds[file_id]}' / f'{file_id}.h5', 'a')

            hf.create_dataset('waveform', data=self.preprocessed_input[file_id])
            hf.create_dataset('nr_samples', data=[len(self.preprocessed_input[file_id])])
            hf.create_dataset('fs', data=[self.current_sample_rate])
            hf.create_dataset('phone_labels', data=self.phone_label_dictionary[file_id])
            hf.create_dataset('speaker_id', data=self.speaker_id_label_dictionary[file_id])
            hf.create_dataset('speaker_sex', data=self.speaker_sex_label_dictionary[file_id])

            assert (len(self.preprocessed_input[file_id]) / (self.current_sample_rate * self.sec_per_annotation)) == len(self.phone_label_dictionary[file_id]), f'The number of phone labels does not match the number of time samples for file {file_id}'
            hf.close()

    def save_meta_data(self):
        if self.save_dir is not None:

            for k,v in self.subset_splits.items(): #Convert Paths to strings
                self.subset_splits[k] = str(v)

            preprocessing_specifics = copy.deepcopy(self.__dict__)

            preprocessing_specifics['label_dir'] = str(preprocessing_specifics['label_dir'])


            # Remove some meta-data that does not need to be stored.
            preprocessing_specifics.pop('preprocessed_input')
            preprocessing_specifics.pop('speaker_id_conversion_dict')
            preprocessing_specifics.pop('unique_sorted_speaker_ids')
            preprocessing_specifics.pop('phone_label_dictionary')
            preprocessing_specifics.pop('speaker_id_label_dictionary')
            preprocessing_specifics.pop('speaker_sex_label_dictionary')
            preprocessing_specifics.pop('file_path_dictionary')
            preprocessing_specifics.pop('data_folds')
            preprocessing_specifics.pop('all_sound_files')
            preprocessing_specifics.pop('load_dir')
            preprocessing_specifics.pop('save_dir')

            for key, value in preprocessing_specifics.items():
                if isinstance(value, np.ndarray):
                    preprocessing_specifics[key] = value.tolist()

            with open(self.save_dir / f'preprocessing_specifics.yml', 'w') as outfile:
                yaml.dump(preprocessing_specifics, outfile, default_flow_style=False)

    def standardize_speaker_ids(self):
        self.unique_sorted_speaker_ids = sorted(np.unique(list(self.speaker_id_label_dictionary.values())))
        new_ids = np.arange(len(self.unique_sorted_speaker_ids))

        self.speaker_id_conversion_dict = dict(zip(self.unique_sorted_speaker_ids, new_ids))

        for file_id, speaker_id in self.speaker_id_label_dictionary.items():
            self.speaker_id_label_dictionary[file_id] = self.speaker_id_conversion_dict[speaker_id]

        self.performed_functions['standardize_speaker_ids'] = True

    def preprocess(self, preprocessing_steps = {}):
        """
        Applied the full preprocessing pipeline for all subjects provided in the list during initialization of this class.

        Args:
            preprocessing_steps (dict, optional): Each key refers to one preprocessing function and its value is either True or False. To be used to skip preprocessing steps for debugging or testing with different preprocessings.
        """

        for file_id in self.file_path_dictionary:

            if not file_id in self.phone_label_dictionary:
                assert file_id not in self.data_folds
                continue

            # Keep track of which functions have been performed, as some are dependent on others.
            self.performed_functions = {}
            print(f'File: {file_id}')

            # Apply all preprocessing functions:
            self.load_waveform(file_id)
            self.standardize_speaker_ids()

            if preprocessing_steps.get('align_signals_and_annotations', True):
                self.align_signals_and_annotations(file_id)
            if preprocessing_steps.get('convert_to_float32', True):
                self.convert_to_float32(file_id)
            if preprocessing_steps.get('save_preprocessed_data', True):
                self.save_preprocessed_data(file_id)

        if self.save_dir is not None:
            self.save_meta_data()



if __name__ == "__main__":
    # Indicate the path to the extracted train-clean-100.tar file. Make sure that the converted_aligned_phones.txt file from https://drive.google.com/drive/folders/1BhJ2umKH3whguxMwifaKtSra0TgAbtfb
    # is saved in the parent directory of that path as well.

    data_load_dir =  <FILL IN>
    
    subset_splits = {'train':Path().absolute() / 'preprocessing' / 'Librispeech_10speakers_datasplits' / 'train_split.txt',
                    'val': Path().absolute() / 'preprocessing' /  'Librispeech_10speakers_datasplits' / 'validation_split.txt',
                    'test': Path().absolute() / 'preprocessing' / 'Librispeech_10speakers_datasplits' / 'test_split.txt'}   

    processing_inst = PreprocessLibriSpeech(data_load_dir, subset_splits, sec_per_annotation=10e-3, sample_rate_orig=16e3)
    processing_inst.preprocess()