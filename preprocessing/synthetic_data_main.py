from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import numpy as np
import yaml
import h5py
import copy



class SinusoidsRandomWalkFreqs(): 
    def __init__(self,
                n_sequences,
                nr_seconds_per_sequence,
                fs,
                start_freq_range, 
                min_freq,
                max_freq,
                step_set,
                prevent_clipping_on_min_max_freq,
                noise_var,
                save_dir,
                train_val_test_split,
               **kwargs):

        """
        prevent_clipping_on_min_max_freq (bool): If set to True, it alters the step probabilities at the boundaries to prevent that signals get stuck (i.e. clipped in their frequency) on the boundaries.
                                                Makes data generation extremely much slower!
        """
        

        self.fs = fs
        self.nr_steps = (nr_seconds_per_sequence*self.fs)-1
        self.start_freqs = np.expand_dims(np.random.choice(a=start_freq_range, size=n_sequences),1)
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.step_set = step_set  
        self.n_sequences = n_sequences
        self.noise_var = noise_var
        self.prevent_clipping_on_min_max_freq = prevent_clipping_on_min_max_freq
        self.save_dir = save_dir
        
        self.train_idxs = np.arange(0,train_val_test_split[0]*self.n_sequences)
        self.val_idxs = np.arange(self.train_idxs[-1]+1,self.train_idxs[-1]+1+train_val_test_split[1]*self.n_sequences)
        self.test_idxs = np.arange(self.val_idxs[-1]+1,self.val_idxs[-1]+1+train_val_test_split[2]*self.n_sequences)

    
    def generate_sequences(self,):
        step_shape = (self.n_sequences,self.nr_steps)

        if self.prevent_clipping_on_min_max_freq:
            max_step_size = np.max(self.step_set)
            self.frequencies = np.zeros(step_shape)*np.nan
            
            for seq_idx in range(self.n_sequences):
                prev_freq = self.start_freqs[seq_idx]
                
                #when the frequency moves beyond the boundaries, temporarilly adjust the step set and make it 50% change to move in again, and 50% chance to stay.
                for step_idx in range(self.nr_steps):
                    if prev_freq  <= self.min_freq: 
                        step_set_temp = [max_step_size, 0] 
                    elif prev_freq >= self.max_freq:
                        step_set_temp = [-max_step_size, 0]
                    else:
                        step_set_temp = self.step_set
                    self.frequencies[seq_idx, step_idx] = prev_freq + np.random.choice(a=step_set_temp, size=1)
                    prev_freq = self.frequencies[seq_idx, step_idx]

            self.frequencies = np.concatenate([self.start_freqs, self.frequencies],1)
        else:
            steps = np.random.choice(a=self.step_set, size=step_shape)
            self.frequencies =  np.minimum(np.maximum(np.concatenate([self.start_freqs, steps],1).cumsum(-1),self.min_freq),self.max_freq)
 
        self.sinusoids= np.sin(2*np.pi*self.frequencies*np.arange(0,self.nr_steps+1)/self.fs) 
        self.noisy_sinusoids = np.random.normal(self.sinusoids, scale=self.noise_var, size=self.sinusoids.shape) #[nr_sequences, nr_samples]

    def store_sequences_as_h5_dataset(self,):
        
        for idx, seq in enumerate(self.noisy_sinusoids):
            if idx in self.train_idxs: data_fold = 'train'
            elif idx in self.val_idxs: data_fold = 'val'
            elif idx in self.test_idxs: data_fold = 'test'

            seq_str = f's{idx}'
            hf = h5py.File(self.save_dir / data_fold / f'{seq_str}.h5', 'a')
            hf.create_dataset('x', data=seq)
            hf.create_dataset('labels', data=self.frequencies[idx])
            hf.create_dataset('nr_samples', data=len(seq))
            hf.create_dataset('fs', data=self.fs)     
            hf.close()

        preprocessing_specifics = copy.deepcopy(self.__dict__)
        preprocessing_specifics.pop('sinusoids')
        preprocessing_specifics.pop('noisy_sinusoids')
        preprocessing_specifics.pop('frequencies')
        preprocessing_specifics.pop('save_dir')
        preprocessing_specifics.update({'dataset':'random_walk_freq_sinusoids','sec_per_annotation':1/self.fs,'nr_classes':np.inf})

        for key, value in preprocessing_specifics.items():
            if isinstance(value, np.ndarray):
                preprocessing_specifics[key] = value.tolist()

        with open(self.save_dir / f'preprocessing_specifics.yml', 'w') as outfile:
            yaml.dump(preprocessing_specifics, outfile, default_flow_style=False)

    def generate_and_save_data(self):
        self.generate_sequences()
        self.store_sequences_as_h5_dataset()


if __name__ == '__main__':
    save_dir = Path.cwd() / 'data'/ 'random_walk_freq_sinusoids_preprocessed'

    save_dir.mkdir(parents=False, exist_ok=True)
    (save_dir/'train').mkdir(parents=False, exist_ok=True)
    (save_dir/'val').mkdir(parents=False, exist_ok=True)
    (save_dir/'test').mkdir(parents=False, exist_ok=True)

    data_gen = SinusoidsRandomWalkFreqs(
                n_sequences=200,
                nr_seconds_per_sequence=300,
                fs=128,
                start_freq_range=np.arange(20,40,0.1), 
                min_freq=1.,
                max_freq=60.,
                step_set=[-0.1,0,0,0,0,0,0,0,0,0.1],
                prevent_clipping_on_min_max_freq = True, 
                noise_var=0.01,
                save_dir=save_dir,
                train_val_test_split = [0.5,0.25,0.25]
               )

    data_gen.generate_and_save_data()
    