from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))
import numpy as np

from experiments.run import run_experiment
from my_utils.training import find_paths, set_save_dir
from my_utils.config import load_config_from_yaml

if __name__ == "__main__":

    checkout_folder, experiment_folder, experiment_name, experiment_type = find_paths(Path(__file__).resolve())

    # Read configuration
    config = load_config_from_yaml(experiment_folder / 'experiment_specific.yml')
    config.data.load_dir = Path.cwd() / 'data' / config.data.load_dir

    #### Create different runs #####
    """
    File to run the (GRU-)DESOM models for audio
    """
    runs = []
    runs_names = []

    # DESOM
    for alpha in [0.00001, 0.0001, 0.001, 0.01]:
        runs.append({
        'losses.SOM_loss.multiplier':alpha,
        'losses.commitment_loss.multiplier':alpha, 
        'data.past_windows':0,
        'model.ARmodel.class': None,
        'model.encoder.last_linear_layer.first_flatten': True,
        'model.decoder_cont.input_dim': [1024,1],
        })
        runs_names.append(f'DESOM_alpha{alpha}_noGRU')

    # DESOM + GRU and decoding last Ct
    # # When running with GRU (i.e. multiple windows), change the unflatten shape in the decoder, and undo the flatten in the encoder, since it would flatten windows and features together. 
    for alpha in [0.00001, 0.0001, 0.001, 0.01, 0.1]:
        runs.append({
        'losses.SOM_loss.multiplier':alpha,
        'losses.commitment_loss.multiplier':alpha, 
        })
        runs_names.append(f'DESOM_alpha{alpha}_withGRU_decoderonlastC')

    # DESOM + GRU and decoding full Ct
    for alpha in [0.00001, 0.0001, 0.001, 0.01]:
        runs.append({
        'losses.SOM_loss.multiplier':alpha,
        'losses.commitment_loss.multiplier':alpha, 
        'model.decoder_cont.first_linear_layer.unflatten_shape': [512, 256],
        'model.decoder_cont.input_dim': [131072], 
        'model.decoder_cont.decode_past_windows':True,
        })
        runs_names.append(f'DESOM_alpha{alpha}_withGRU_decoderonallC')


    for sub_experiment_name, run in zip(runs_names, runs):

        run_config = config.deep_copy()

        for config_entry, value in run.items():
            run_config.set_nested_key(config_entry, value)   

        set_save_dir(run_config, experiment_type, experiment_name, sub_experiment_name)
               
        # Run experiment
        run_experiment(configuration=run_config)
