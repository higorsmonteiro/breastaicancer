import subprocess
import argparse

# -------------------------------
# CLI argument: config-dir
# -------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--config-dir", required=True)
args = parser.parse_args()
 
config_dir = args.config_dir

# ----------------------------------------------
# 1. Tuning for all target years
# ----------------------------------------------
training_cfg = 'base.yml' # dummy
split_cfg = 'split_002.yml'
tuning_cfg = 'bce_mlflow_test.yml'
target_year = 5
# -- single split (split_002.yml)
followup_configs = [
    'fix_grace_90_birads5_interval_6_mlflow.yml'
]

for cur_followup_config in followup_configs:
    print(f"Target year: 5, Current follow-up age strat: {cur_followup_config}.")
    args = [
        'hapcancer', 'tuning', 
        '--config-dir', config_dir, 
        '--config-params', f'followup={cur_followup_config}',
        '--config-params', f'tuning={tuning_cfg}',
        '--config-params', f'split={split_cfg}', # 1,2 and 3.
        '--config-params', f'training_experiments={training_cfg}', # dummy
        '--target-year', f'{5}',
        '--seq-percentile', f'99.5', 
        '--n-splits', '5', '--total-epochs', '15'
    ]
    result = subprocess.run(args)