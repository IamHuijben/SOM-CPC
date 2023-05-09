from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

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
    File to run the SOM-CPC models for audio
    """
    runs = []
    runs_names = []

    # SOM-CPC with SOM loss detached from encoder
    for alpha in [0.001, 0.01, 0.1, 1]:   
        runs.append({
        'losses.SOM_loss.multiplier':alpha,
        'losses.commitment_loss.multiplier':alpha,    
        'losses.SOM_loss.arguments.detach_from_encoder': True,
        })
        runs_names.append(f'alpha{alpha}_detachSOM_True')    

    # SOM-CPC without SOM loss detached from encoder
    for alpha in [0.00001, 0.0001, 0.001, 0.01]:
        runs.append({
        'losses.SOM_loss.multiplier':alpha,
        'losses.commitment_loss.multiplier':alpha,    
        'losses.SOM_loss.arguments.detach_from_encoder': False,
        })
        runs_names.append(f'alpha{alpha}_detachSOM_False')    


    for sub_experiment_name, run in zip(runs_names, runs):

        run_config = config.deep_copy()

        for config_entry, value in run.items():
            run_config.set_nested_key(config_entry, value)   

        set_save_dir(run_config, experiment_type, experiment_name, sub_experiment_name)

        # Run experiment
        run_experiment(configuration=run_config)
