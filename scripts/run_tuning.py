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
tuning_cfg = 'bce_all_001.yml'
split_cfg = 'split_002.yml'
target_years = [ 5, 4, 3, 2, 1 ]
followup_cfg = 'age_strat_18_75.yml'
for cur_target_year in target_years:
    print(f"Target year: {cur_target_year}, Current follow-up age strat: {followup_cfg}.")
    args = [
        'hapcancer', 'tuning', 
        '--config-dir', config_dir, 
        '--config-params', f'followup={followup_cfg}',
        '--config-params', f'tuning={tuning_cfg}',
        '--config-params', f'split={split_cfg}',
        '--config-params', f'training_experiments={training_cfg}', # dummy
        '--target-year', f'{cur_target_year}',
        '--seq-percentile', f'99.5', 
        '--n-splits', '5', '--total-epochs', '15'
    ]
    result = subprocess.run(args)