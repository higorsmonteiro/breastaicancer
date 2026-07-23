import subprocess
import argparse

# -------------------------------
# CLI argument: config-dir
# -------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--config-dir", required=True)
args = parser.parse_args()

config_dir = args.config_dir

training_cfg = 'base.yml' # dummy
tuning_cfg = 'bce_all_001.yml'
split_cfg = 'split_002.yml'
followup_cfg = 'age_strat_18_75.yml'
target_years = [ 5, 4, 3, 2, 1 ]
target_years = [ 5 ]
fractions = [1.0, 1.0, 0.1, 0.1, 0.1 ] 
fractions = [ 1.0 ]

# ------------------------------------------------------
# 1. Evaluation: Discrimination and calibration metrics
# ------------------------------------------------------
for cur_target_year in target_years:
    print(f"Target year: {cur_target_year}, Current follow-up age strat: {followup_cfg}.")
    args = [
        'hapcancer', 'eval-metrics-best', 
        '--config-dir', config_dir, 
        '--config-params', f'followup={followup_cfg}',
        '--config-params', f'tuning={tuning_cfg}',
        '--config-params', f'split={split_cfg}',
        '--target-year', f'{cur_target_year}'
    ]
    result = subprocess.run(args)

# ------------------------------------------------------
# 2. Evaluation: Explanation attributions
# ------------------------------------------------------
# -- for 3 years or less, the cohort might too big to save attributions for all exams 
for cur_target_year, frac in zip(target_years, fractions):
    print(f"Target year: {cur_target_year}, Current follow-up age strat: {followup_cfg}.")
    args = [
        'hapcancer', 'eval-explain-best', 
        '--config-dir', config_dir, 
        '--config-params', f'followup={followup_cfg}',
        '--config-params', f'tuning={tuning_cfg}',
        '--config-params', f'split={split_cfg}',
        '--target-year', f'{cur_target_year}',
        '--fraction', f'{frac}'
    ]
    result = subprocess.run(args)