import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import Optional, List

from hapcancer.model.dataload.load_input import DatasetSplit, load_dataset, transform_eligibility_to_singleyear
from hapcancer.etl.utils import batching_parquet_file
from hapcancer.config_manager import ConfigInterface

class CohortInfo(ConfigInterface):
    '''

    '''
    def __init__(self, config_dir: str, config_defaults: dict):
        super().__init__(config_dir, config_defaults)
        self.config_dir = config_dir
        self.config_defaults = config_defaults

class EDA(ConfigInterface):
    def __init__(self, config_dir: str, config_defaults: dict):
        super().__init__(config_dir, config_defaults)
        self.config_dir = config_dir
        self.config_defaults = config_defaults

        self.dataset_splits = None
        self.key2info_order = None
        self.key2info = None
        self.complete_before_df = None
        self.positive_df = None
        self.input_loader = None

        self.feature_columns = self.model_fields_cfg["fields"]["feature_columns"]
        self.event_indicator_columns = self.model_fields_cfg["fields"]["event_indicator_columns"]
        self.followup_columns = self.model_fields_cfg["fields"]["followup_columns"]
        self.multi_elig_columns = self.model_fields_cfg["fields"]["multiyear_eligibility_columns"]

        self.struct_dataset_eligible_filename = self.files_and_folders_cfg["load"]["load_files"]["final_data_with_eligibility_filename"]

    def load_split(self, target_year: int):
        self.dataset_splits = DatasetSplit(self.config_dir, self.config_defaults)
        self.dataset_splits.split(target_year, n_splits=5, seq_percentile=99.5)

    def get_eligible_struct_dataset(self, target_year):
        elig_ids = self.get_eligible_ids()
        src = self.dataset_path.joinpath(self.struct_dataset_eligible_filename)
        struct_cols = self.feature_columns+self.event_indicator_columns+self.followup_columns+self.multi_elig_columns
        struct_dataset = load_dataset(src, struct_cols, elig_ids, batch_size=100_000)
        struct_dataset = transform_eligibility_to_singleyear(
            struct_dataset, 
            target_year=target_year, 
            followup_cols=self.followup_columns, 
            event_cols=self.event_indicator_columns,
            only_eligible=True
        )
        return struct_dataset

    def get_base_data(self):
        cols = [
            'CD_PESSOA', 'key', 'DT_ATENDIMENTO_MAMOGRAFIA', 'birads_labels', 
            'biopsy_results', 'biopsy_dates'
        ]
        base_df = []
        for batch in tqdm(batching_parquet_file(self.load_path.joinpath("base_merged_data.parquet"), columns=cols)):
            base_df.append(batch)
        base_df = pd.concat(base_df, ignore_index=True)
        return base_df

    def load_preprocessing_info(self):
        self.key2info_order = { 
            "names": [ "DT_NASCIMENTO_FUNDACAO", "zipcode_cat", "DT_ATENDIMENTO_MAMOGRAFIA", "birads_labels" ] 
        }
        self.key2info = {}
        for batch in tqdm(batching_parquet_file(self.load_path.joinpath("base_merged_data.parquet"))):
            keys = batch["key"].tolist()
            temp_ = list(
                zip(
                    batch[self.key2info_order["names"][0]].tolist(),
                    batch[self.key2info_order["names"][1]].tolist(),
                    batch[self.key2info_order["names"][2]].tolist(),
                    batch[self.key2info_order["names"][3]].tolist()
                )
            )
            self.key2info.update({
                key: temp_[idx] for idx, cur_key_list in enumerate(keys) for key in cur_key_list
            })

    def load_complete_before_elig(self, mammogram_ids: Optional[List[str]] = None):
        self.complete_before_df = []
        for batch in tqdm(batching_parquet_file(self.dataset_path.joinpath("complete_pop_before_eligibility_no_seq.parquet"))):
            if mammogram_ids is not None:
                batch = batch[batch["mammogram_id"].isin(mammogram_ids)].copy()
            self.complete_before_df.append(batch)
        self.complete_before_df = pd.concat(self.complete_before_df, ignore_index=True)

    def load_positive(self):
        cols = [
            "person_id", "mammogram_id", "earliest_positive_birads5", 
            "earliest_positive_birads6", "earliest_positive_biopsy"
        ]
        self.positive_df = []
        for batch in tqdm(batching_parquet_file(self.dataset_path.joinpath("mamm_seq_per_mammogram.parquet"), columns=cols)):
            self.positive_df.append(batch)
        self.positive_df = pd.concat(self.positive_df, ignore_index=True)
    
    def get_eligible_ids(self):
        split_dict = self.dataset_splits.cv_split_by_mammogram
        training_set = split_dict['fold 0']['train']
        validation_set = split_dict['fold 0']['validation']
        test_set = split_dict['test']
        elig_ids = training_set + validation_set + test_set
        return elig_ids