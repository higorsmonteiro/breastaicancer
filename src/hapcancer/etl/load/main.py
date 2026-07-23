import argparse
import yaml
import datetime as dt
# -- relative imports
from .setup_final_eligibility import SetupMammogramEligibility, SetupDataset
from .utils import load_config_file

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help="Relative path, filename included, to the YAML configuration file.", 
                        type=str, required=True)
    parser.add_argument('--mode', type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    config_path = args.config
    mode = args.mode
    config = load_config_file(config_path)
    
    if mode=="eligibility":
        setuper = SetupMammogramEligibility(config)
        setuper.setup_mammogram_data(verbose=True)
    elif mode=="dataset":
        splitsetup = SetupDataset(config)
        splitsetup.setup_dataset()
    elif mode=='all':
        setuper = SetupMammogramEligibility(config)
        setuper.setup_mammogram_data(verbose=True)
        splitsetup = SetupDataset(config)
        splitsetup.setup_dataset()
    elif mode=='all-inter':
        setuper = SetupMammogramEligibility(config)
        setuper.setup_mammogram_data(load_intermediate=True, verbose=True)
        splitsetup = SetupDataset(config)
        splitsetup.setup_dataset()
    else:
        pass
    
def generate_dataset_api(
    config: dict,
    mode: str
) -> None:
    if mode=="eligibility":
        setuper = SetupMammogramEligibility(config)
        setuper.setup_mammogram_data(verbose=True)
    elif mode=="dataset":
        splitsetup = SetupDataset(config)
        splitsetup.setup_dataset()
    elif mode=='all':
        setuper = SetupMammogramEligibility(config)
        setuper.setup_mammogram_data(verbose=True)
        splitsetup = SetupDataset(config)
        splitsetup.setup_dataset()
    elif mode=='all-inter':
        setuper = SetupMammogramEligibility(config)
        setuper.setup_mammogram_data(load_intermediate=True, verbose=True)
        splitsetup = SetupDataset(config)
        splitsetup.setup_dataset()
    else:
        pass


if __name__ == "__main__":
    main()