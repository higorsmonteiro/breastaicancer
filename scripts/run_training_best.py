import subprocess
import argparse

# -------------------------------
# CLI argument: config-dir
# -------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--config-dir", required=True)
args = parser.parse_args()
 
config_dir = args.config_dir

# ---------------------------------------------------------------------
# 1. CV training on the best params obtained by the tuning experiment
# ---------------------------------------------------------------------
training_cfg = 'base.yml' # dummy
tuning_cfg = 'bce_all_001.yml'
target_years = [ 5, 4, 3, 2, 1 ]
target_years = [ 1 ]
split_cfg = 'split_002.yml'
followup_cfg = 'age_strat_18_75.yml'
for cur_target_year in target_years:
    print(f"Target year: {cur_target_year}")
    args = [
        'hapcancer', 'cv-training-best', 
        '--config-dir', config_dir, 
        '--config-params', f'split={split_cfg}',
        '--config-params', f'tuning={tuning_cfg}',
        '--config-params', f'followup={followup_cfg}',
        '--config-params', f'training_experiments={training_cfg}',
        '--target-year', f'{cur_target_year}',
        '--seq-percentile', f'99.5', '--n-splits', '5',
    ]
    result = subprocess.run(args)