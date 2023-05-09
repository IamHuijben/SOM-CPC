import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

from experiments.run import run_experiment
from my_utils.training import find_paths, set_save_dir_continue_training
from my_utils.config import load_config_from_yaml



if __name__ == "__main__":

    checkout_folder, experiment_folder, experiment_name, experiment_type = find_paths(Path(__file__).resolve())

    # Read configuration
    config = load_config_from_yaml(experiment_folder / 'experiment_specific.yml')
    config.data.load_dir = Path.cwd() / 'data' / config.data.load_dir
    
    #### Create different runs #####
    # Run a SOM on top of a trained CPC model
    runs, run_names = [],[]


    checkpoint_path = Path.cwd() / "CPC" / "Synthetic_experiments_ICML2023" / "CPC_synthetic" / "checkpoints" / "model_40.pt"
    runs.append({
        'model.encoder.checkpoint_path': Path(checkpoint_path),
        })
    run_names.append(f'')

    for run_name_comment, run in zip(run_names, runs):

        run_config = config.deep_copy()
        for config_entry, value in run.items():
            run_config.set_nested_key(config_entry, value)   

        epoch = run_config.model.encoder.checkpoint_path.stem.split("_")[-1]
        save_dir = set_save_dir_continue_training(Path(run['model.encoder.checkpoint_path']).parent.parent, f'SOM_e{epoch}{run_name_comment}')
        run_config['logging'] = str(save_dir)
    
            
        # Run experiment
        run_experiment(configuration=run_config)

