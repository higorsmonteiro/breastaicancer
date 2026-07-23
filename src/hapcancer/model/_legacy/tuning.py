'''
    Functions to perform bayesian optimization of hyperparameters using optuna.

    Fixed hyperparamters (does not influence the performance of the model):

    1.  Number of workers.
    2.  Batch size.
    3.  BIRADS: Initially we will consider all mammograms from 0 to 3. Another tuning must be
        done for 0 to 2 range.

    The loss function will be fixed: 'compositional_auc_loss' from libauc library. Other loss 
    functions were tested and we selected this one as it allowed very high AUROC values.

    The following hyperparameters will be considered for optimization:

    1.  Transformer's number of attention heads.
    2.  Transformer's number of layers.
    3.  Transformer's dropout.
    4.  Number of hidden layers of the MLP module (depth).
    5.  Number of neurons for each hidden layer of the MLP module.
    6.  Dropout for the MLP module.
    7.  Activation function for the MLP module: Relu, Gelu, Mish (Swish).
    8.  Learning rate.
    9.  Weight decay (regularization parameter).
'''

import optuna
import yaml
import os
import copy
from pathlib import Path
# -- relative imports
from .train_multiyear import train_for_tuning_multiyear
from .train_multiyear_ import train_for_tuning_singleyear

def objective_multiyear(trial, config):
    # -- don't mutate the original
    config = copy.deepcopy(config)

    
    heads = trial.suggest_categorical("transformer_num_heads", [2, 4, 8, 16])
    layers = trial.suggest_categorical("transformer_num_layers", [2, 4, 6, 8, 10, 12, 14, 16])
    ## -- filter valid layer values
    #valid_layers = [l for l in range(2, heads + 1) if l % heads == 0]
    #if not valid_layers:
    #    raise optuna.exceptions.TrialPruned()  # no valid value -> prune
    #layers = trial.suggest_categorical("transformer_num_layers", valid_layers)

    config['model']['transformer_num_heads'] = heads
    config['model']['transformer_num_layers'] = layers
    config['model']['transformer_dropout'] = trial.suggest_float("transformer_dropout", 0.0, 0.5)
    config['model']['mlp_config']['dropout'] = trial.suggest_float("dropout", 0.0, 0.5)
    config['model']['mlp_config']['activation'] = trial.suggest_categorical("activation", ['relu', 'gelu', 'mish'])
    # -- depth and layer size
    depth = trial.suggest_int("depth", 2, 6)
    hidden_layers = [
        trial.suggest_int(f"layer_{i}_units", 32, 128, step=16) for i in range(depth)
    ]
    config['model']['mlp_config']['hidden_layers'] = hidden_layers
    config['training']['learning_rate'] = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    config['training']['weight_decay'] = trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True)
    
    # -- run training (return average precision score)
    pr_score, rocauc_score = train_for_tuning_multiyear(config) 
    print(f"Average Precision: {pr_score:.5f}; ROC: {rocauc_score:.5f}")
    return rocauc_score

def objective_singleyear(trial, config):
    # -- don't mutate the original
    config = copy.deepcopy(config)
    
    heads = trial.suggest_categorical("transformer_num_heads", [2, 4, 8, 16])
    layers = trial.suggest_categorical("transformer_num_layers", [2, 4, 6, 8, 10, 12, 14, 16])
    ## -- filter valid layer values
    #valid_layers = [l for l in range(2, heads + 1) if l % heads == 0]
    #if not valid_layers:
    #    raise optuna.exceptions.TrialPruned()  # no valid value -> prune
    #layers = trial.suggest_categorical("transformer_num_layers", valid_layers)

    config['model']['transformer_num_heads'] = heads
    config['model']['transformer_num_layers'] = layers
    config['model']['transformer_dropout'] = trial.suggest_float("transformer_dropout", 0.0, 0.5)
    config['model']['mlp_config']['dropout'] = trial.suggest_float("dropout", 0.0, 0.5)
    config['model']['mlp_config']['activation'] = trial.suggest_categorical("activation", ['relu', 'gelu', 'mish'])
    # -- depth and layer size
    depth = trial.suggest_int("depth", 2, 6)
    hidden_layers = [
        trial.suggest_int(f"layer_{i}_units", 32, 128, step=16) for i in range(depth)
    ]
    config['model']['mlp_config']['hidden_layers'] = hidden_layers
    config['training']['learning_rate'] = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    config['training']['weight_decay'] = trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True)
    
    # -- run training (return average precision score)
    pr_score, rocauc_score = train_for_tuning_singleyear(config) 
    print(f"Average Precision: {pr_score:.5f}; ROC: {rocauc_score:.5f}")
    return rocauc_score

def run_study(base_config: dict, task: str = 'single'):
    '''
        Run an Optuna hyperparameter tuning study using a given base config.
    
        Args:
        -----
            base_config (dict): 
                Base config loaded from YAML
            n_trials (int): 
                Number of Optuna trials
            output_dir (str): 
                Folder to save results
            study_name (str): 
                Name of the Optuna study
            seed (int): 
                Seed for reproducibility
    '''
    num_trials = base_config['misc']['num_trials']
    study_name = base_config['misc']['study_name']
    optim_seed = base_config['misc']['optim_seed']
    checkpoint_path = Path(base_config['misc']['checkpoint_path'])
    os.makedirs(checkpoint_path.joinpath(study_name), exist_ok=True)
    
    def wrapped_objective(trial):
        # Deepcopy so we don't mutate base config
        config = copy.deepcopy(base_config)
        if task=='single':
            return objective_singleyear(trial, config)
        else:
            return objective_multiyear(trial, config)

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",  # because we're optimizing PR AUC/ ROC AUC (average precision, AUROC)
        sampler=optuna.samplers.TPESampler(seed=optim_seed)
    )
    study.optimize(wrapped_objective, n_trials=num_trials)

    print("best Trial")
    best_trial = study.best_trial
    for key, value in best_trial.params.items():
        print(f"{key}: {value}")
    print(f"best auroc: {best_trial.value:.4f}")

    # -- save best config
    best_config = copy.deepcopy(base_config)

    # -- inject best params into config structure
    best_config['misc']['model_name'] = 'best_model_optim'
    best_config['model']['transformer_num_heads'] = best_trial.params["transformer_num_heads"]
    best_config['model']['transformer_num_layers'] = best_trial.params["transformer_num_layers"]
    best_config['model']['transformer_dropout'] = best_trial.params["transformer_dropout"]
    best_config['model']['mlp_config']['dropout'] = best_trial.params["dropout"]
    best_config['model']['mlp_config']['activation'] = best_trial.params["activation"]
    best_config['model']['mlp_config']['hidden_layers'] = [
        best_trial.params[f"layer_{i}_units"]
        for i in range(best_trial.params["depth"])
    ]
    best_config['training']['learning_rate'] = best_trial.params["learning_rate"]
    best_config['training']['weight_decay'] = best_trial.params["weight_decay"]

    # -- save to file
    config_path = checkpoint_path.joinpath(study_name, f"best_config_{study_name}.yml")
    with open(config_path, "w") as f:
        yaml.dump(best_config, f)

    print(f"best config saved to: {config_path}")

    # -- save all scores of the optimization
    df = study.trials_dataframe()
    df.to_csv(checkpoint_path.joinpath(study_name, "all_trials.csv"), index=False)
    return study

