import argparse
import subprocess

# -------------------------------
# CLI argument: config-dir
# -------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--config-dir", required=True)
args = parser.parse_args()
 
config_dir = args.config_dir

db_origin_list = ['hsp', 'psc']
db_origin_list = ['psc']
raw_data_names = [
    'anamnesis', 'biopsy', 'person', 
    'user', 'patient', 'mammogram'
]
raw_data_names = ['mammogram']
# -- does not change the values below for a same collection in case we need to run this script more than once
chunk_sizes = {
    "hsp": 300_000,
    # for 'mammogram' and 'biopsy' it is common to find corrupted files in PSC that get in the way of collecting data.
    "psc": 10_000 
}
for raw_data_name in raw_data_names:
    for db_origin in db_origin_list:
        print(f'extraction state: {raw_data_name} ({db_origin})')
        chunk_size = chunk_sizes[db_origin]
        if raw_data_name=='anamnesis' and db_origin=='psc': chunk_size = 300_000
        args = [
            'hapcancer', 'extract-raw-data', 
            '--config-dir', config_dir,
            '--raw-data-name',  raw_data_name,
            '--db-origin', f'{db_origin}',
            '--chunk-size', f'{chunk_size}'
        ]
        result = subprocess.run(args)

#command = "validate-raw-data"
#args = [
#    'hapcancer', command, 
#    '--config-dir', config_dir,
#    '--fraction',  '0.4',
#]
#result = subprocess.run(args)