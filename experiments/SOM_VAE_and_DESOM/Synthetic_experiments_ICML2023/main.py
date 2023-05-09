
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
    Run SOM-VAE, SOM-VAE-prob and DESOM models with different loss multipliers
    """
    runs = []
    runs_names = []

    # SOM-VAE
    for alpha in [0.001, 0.01, 0.1, 1]:
       runs.append({
       'losses.SOM_loss.multiplier':alpha/5,
       'losses.commitment_loss.multiplier':alpha, 
       'model.quantizer.gaussian_neighbourhood': None,
       })
       runs_names.append(f'SOMVAE_alpha{alpha}')

    # SOM-VAE + Gaussian
    for alpha in [0.00001, 0.0001, 0.001, 0.01]:
       runs.append({
       'losses.SOM_loss.multiplier':alpha,
       'losses.commitment_loss.multiplier':alpha, 
       })
       runs_names.append(f'SOMVAE_alpha{alpha}_Gaussian')

    # DESOM
    for alpha in [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]: 
        runs.append({
        'losses.SOM_loss.multiplier':alpha,
        'losses.commitment_loss.multiplier':alpha, 
        'losses.SOM_loss.arguments.detach_from_encoder': False,
        'losses.mse_disc.multiplier':0.,
        'model.decoder_disc.class': None,
        'training.aggregate_pred_names': ['all_codebook_idxs', 'x_hat_cont'],
        'callbacks.SaveModel.checking_metric': ['mse_cont', 'loss', 'commitment_loss', 'SOM_loss'],
        })
        runs_names.append(f'DESOM_alpha{alpha}')

    # SOM-VAE-prob
    for alpha in [0.001, 0.01, 0.1]:
        for tau in [alpha]: #smoothness loss is very similar to the commitment loss in terms of definition, so use the same multiplier. 
            for gamma in [tau/20, tau/25, tau/30]:
                runs.append({
                'losses.SOM_loss.multiplier':alpha/5,
                'losses.commitment_loss.multiplier':alpha, 
                'model.quantizer.gaussian_neighbourhood': None,
                'model.quantizer.transitions':True,
                'data.past_windows':1,
                'losses.smoothness_loss.multiplier':tau,
                'losses.transition_loss.multiplier': gamma,
                })
                runs_names.append(f'SOMVAE_prob_alpha{alpha}_gamma{gamma}_tau{tau}')

    for alpha in [0.1]:
        for tau in [alpha/10]:
            for gamma in [tau/20, tau/25, tau/30]:
                runs.append({
                'losses.SOM_loss.multiplier':alpha/5,
                'losses.commitment_loss.multiplier':alpha, 
                'model.quantizer.gaussian_neighbourhood': None,
                'model.quantizer.transitions':True,
                'data.past_windows':1,
                'losses.smoothness_loss.multiplier':tau,
                'losses.transition_loss.multiplier': gamma,
                })
                runs_names.append(f'SOMVAE_prob_alpha{alpha}_gamma{gamma}_tau{tau}')

    for sub_experiment_name, run in zip(runs_names, runs):

        run_config = config.deep_copy()

        for config_entry, value in run.items():
            run_config.set_nested_key(config_entry, value)   

        set_save_dir(run_config, experiment_type, experiment_name, sub_experiment_name)

        # Run experiment
        run_experiment(configuration=run_config)
