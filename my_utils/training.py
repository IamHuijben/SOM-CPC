from pathlib import Path
import torch
import numpy as np 
import random

            
def data_root_dir():
    data_root_dir = Path.cwd() / 'data'
    return data_root_dir

def experiment_dir():
    experiment_dir = Path.cwd()
    return experiment_dir

def set_random_seed(seed):
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic=True

def make_unique_path(save_dir):
    try:
        save_dir.mkdir(exist_ok=False, parents=True)
    except:
        unique_dir_found = False
        post_fix = 0
        while not unique_dir_found:
            try:
                Path(str(save_dir) +
                        f"_{post_fix}").mkdir(exist_ok=False, parents=True)
                unique_dir_found = True
                save_dir = Path(str(save_dir) + f"_{post_fix}")
            except:
                post_fix += 1
    return save_dir

def set_save_dir(config, experiment_type, experiment_name, sub_experiment_name, sub_sub_experiment_name=None):
    
    # Create save directory
    save_dir = config.get('logging', Path().absolute())
    if save_dir:
        save_dir = experiment_dir() / save_dir / experiment_type / experiment_name / sub_experiment_name
        save_dir = make_unique_path(save_dir)

        if sub_sub_experiment_name:
            save_dir = save_dir / sub_sub_experiment_name
            save_dir.mkdir(exist_ok=False, parents=True)
        config['logging'] = str(save_dir)

    
def set_save_dir_continue_training(orig_model_path_in_nas, new_experiment_name):
    save_dir = experiment_dir() / orig_model_path_in_nas / new_experiment_name 
    save_dir = make_unique_path(save_dir)
    return save_dir


def find_paths(sweep_file_path):
    "Returns different paths from the given sweep file path."
    experiment_folder = Path(sweep_file_path).parent
    checkout_folder = experiment_folder.parent.parent.parent
    experiment_name = experiment_folder.stem
    experiment_type = Path(sweep_file_path).parent.parent.stem
    return checkout_folder, experiment_folder, experiment_name, experiment_type

def get_gpu_memory():
    import subprocess as sp
    def _output_to_list(x): return x.decode('ascii').split('\n')[:-1]

    COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
    memory_free_values = [int(x.split()[0])
                            for i, x in enumerate(memory_free_info)]
    print('Available gpu memory: \n',memory_free_values)
    return memory_free_values

def choose_gpu():

    mem = get_gpu_memory()
    gpu_idx = np.argmax(mem)
    #print('GPU will be automatically chosen based on available memory')
    print('Selected GPU {} with {} MB of memory available'.format(
        gpu_idx, mem[gpu_idx]))

    torch.cuda.set_device(gpu_idx.tolist())
    return gpu_idx

def set_device_settings(cuda=True, gpu_idx=None):    

    # Set device settings
    if cuda and torch.cuda.is_available():
        if gpu_idx is None:
            gpu_idx = choose_gpu()

        device = torch.device("cuda:"+str(gpu_idx))
    else:
        device = "cpu"

    print('Use device: ', device)
    return device

