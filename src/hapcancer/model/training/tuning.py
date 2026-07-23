'''
    Functions to perform bayesian optimization of hyperparameters using optuna.

    Version without transformers.

    Fixed hyperparamters (does not influence the performance of the model):

    1.  Number of workers.
    2.  Batch size.
    3.  BIRADS: Initially we will consider all mammograms from 1 to 3. Another tuning must be
        done for 1 to 2 range.

    The following hyperparameters will be considered for optimization:

    1. Number of layers of the MLP module;
    2. Number of neurons in each layer of the MLP module;
    3. Dropout for the MLP module;
    4. Activation function for the MLP module: Relu, Gelu, Mish (Swish).
    5. Weight decay (regularization parameter);
    6. Loss function: (1) Binary Cross-Entropy; (2) Compositional AUC loss
    and (3) Average Precision Loss.
'''

import optuna
import yaml
import os
import copy
import datetime as dt
from pathlib import Path
from hapcancer.model.training.train_singleyear_mlflow import TrainingSingleYearTFIDF
#from hapcancer.model.utils.logger import Logger
from typing import Optional
from hapcancer.logger import Logger
from hapcancer.config_manager import ConfigInterface

class TuningSingleYear(ConfigInterface):
    def __init__(self, config_dir: str, config_defaults: dict, ablate: Optional[bool] = False):
        super().__init__(config_dir, config_defaults)
        self.config_dir = config_dir
        self.config_defaults = config_defaults
        self.ablate = ablate # -- used only for training without texts

        # -- tuning config
        self.tuning = self.tuning_cfg['tuning']
        self.study_name = self.tuning["study_name"]
        self.num_trials = self.tuning['num_trials']
        self.optim_seed = self.tuning['optim_seed']

        self.trainer = None
        self.total_epochs = 15
        self.study = None
        self.today = dt.datetime.today()

        self.split_dict = None
        self.target_year = None
        self.logger = Logger(self.tuning_path, self.study_name, overwrite=False)

    def _set_total_epochs_per_trial(self, total_epochs: int):
        self.total_epochs = total_epochs

    def set_split(self, split_dict: dict):
        self.split_dict = split_dict

    def set_target_year(self, target_year: int):
        self.target_year = target_year
    
    def objective(
        self,
        trial
    ):
        self.training_cfg['training']['epochs'] = self.total_epochs
        # -- choose MLP parameters
        self.training_cfg['model']['mlp_config']['dropout'] = trial.suggest_float("dropout", 0.0, 0.6)
        self.training_cfg['model']['mlp_config']['activation'] = 'relu'
        depth = trial.suggest_int("depth", 2, 5)
        hidden_layers = [
            trial.suggest_categorical(f"layer_{i}_units", [ 2**lyr for lyr in range(4,11) ]) for i in range(depth)
        ]
        # -- implement suggest_log (^2)
        self.training_cfg['model']['mlp_config']['hidden_layers'] = hidden_layers
        self.training_cfg['training']['weight_decay'] = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        self.training_cfg['training']['learning_rate'] = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
        self.training_cfg['training']['loss_function'] = 'cross_entropy'
        self.training_cfg['training']['optimizer'] = trial.suggest_categorical('optimizer', ['adam', 'sgd'])
        # -- run training
        trainer = TrainingSingleYearTFIDF(self.config_dir, self.config_defaults)
        pr_score, rocauc_score = trainer.train(self.split_dict, self.target_year, tuning=True, remove_text=self.ablate) 
        print(f"Average Precision: {pr_score:.5f}; ROC: {rocauc_score:.5f}")
        return rocauc_score

    def run_study(self):
        def wrapped_objective(trial):
            res_value = self.objective(trial)
            # -- log the trial result
            trial_info = {
                "config_dir_path": str(self.cfg_manager.config_dir),
                "split_config_name": self.cfg_manager['split'],
                "tuning_config_name": self.cfg_manager['tuning'],
                "trial_number": trial.number,
                "init_date": self.today.strftime("%Y-%m-%d %H:%M"),
                "load_id": self.load_id,
                "dataset_name": self.followup_cfg["dataset_name"],
                "target_year": self.target_year,
                "total_epochs_per_trial": self.total_epochs,
                "params": trial.params,
                "result": res_value
            }
            self.logger.log_info(trial_info)
            return res_value
        
        self.study = optuna.create_study(
            study_name=self.study_name,
            direction="maximize",  # because we're optimizing PR AUC/ ROC AUC (average precision, AUROC)
            sampler=optuna.samplers.TPESampler(seed=self.optim_seed)
        )
        self.study.optimize(wrapped_objective, n_trials=self.num_trials)


