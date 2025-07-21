from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

from experiments.run import run_experiment
from my_utils.training import find_paths, set_save_dir
from my_utils.config import load_config_from_yaml

if __name__ == "__main__":

    checkout_folder, experiment_folder, experiment_name, experiment_type = find_paths(Path(__file__).resolve())


    """
        File to run the SOM-CPC on different sleep datasets with different channel configurations
    """

    runs = []
    runs_names = []

    for dataset, config_file in zip(['healthbed', 'mass', 'healthbed_somnia'], ['config_healthbed.yml', 'config_mass.yml', 'config_healthbed_somnia.yml']):
        
        # Read configuration
        config = load_config_from_yaml(experiment_folder / 'experiment_specific.yml')
        config.data.load_dir = Path.cwd() / 'data' / config.data.load_dir

        #### Create different runs #####

        runs_names.append(f'{dataset}')
        runs.append({})


        for sub_experiment_name, run in zip(runs_names, runs):

            run_config = config.deep_copy()

            for config_entry, value in run.items():
                run_config.set_nested_key(config_entry, value)   

            set_save_dir(run_config, experiment_type, experiment_name, sub_experiment_name)
            
            # Run experiment
            run_experiment(configuration=run_config)
