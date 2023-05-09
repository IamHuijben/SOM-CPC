from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

from experiments.run import run_experiment
from my_utils.training import find_paths, set_save_dir
from my_utils.config import load_config_from_yaml

if __name__ == "__main__":

    checkout_folder, experiment_folder, experiment_name, experiment_type = find_paths(Path(__file__).resolve())

    # Read configurations
    config = load_config_from_yaml(experiment_folder / 'experiment_specific.yml')
    config.data.load_dir = Path.cwd() / 'data' / config.data.load_dir

    #### Create different runs #####
    runs = []
    runs_names = []

    runs.append({})
    runs_names.append('CPC_physiological')

    runs.append({
    'model.classifier.input_channels': 2,
    'model.classifier.output_channels': [2],
    'model.encoder.number_of_features': 2,
    })
    runs_names.append('CPC_sleep_zdim2')


    for sub_experiment_name, run in zip(runs_names, runs):
        
        run_config = config.deep_copy()

        for config_entry, value in run.items():
            run_config.set_nested_key(config_entry, value)   

        set_save_dir(run_config, experiment_type, experiment_name, sub_experiment_name)
        
        # Run experiment
        run_experiment(configuration=run_config)

