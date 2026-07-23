import subprocess
import argparse

# -------------------------------
# CLI argument: config-dir
# -------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--config-dir", required=True)
args = parser.parse_args()
 
# -- default: 'D:/hapvida/config/run_test_24022026'
config_dir = args.config_dir

# ----------------------------------------------
# 1. Apply elegibility and generate dataset
# ----------------------------------------------
strat_config = 'fix_grace_90_birads5_interval_6.yml'
print(f"Generating final dataset for strat configuration {strat_config} ...")
args = [
    'hapcancer', 'generate-dataset', 
    '--config-dir', config_dir,
    '--config-params', f'followup={strat_config}'
]
result = subprocess.run(args)

# ----------------------------------------------
# 2. Precompute past sequences
# ----------------------------------------------
print(f"Precompute embeddings for past mammograms ...")
gb_size = 14 # size in GB to store the past sequences
args = [
    'hapcancer', 'precompute-sequences', 
    '--config-dir', config_dir,
    '--config-params', f'followup={strat_config}',
    '--time-limit', '36', '--gb-size', f'{gb_size}',
    '--batch-size', '5000'
]
result = subprocess.run(args)