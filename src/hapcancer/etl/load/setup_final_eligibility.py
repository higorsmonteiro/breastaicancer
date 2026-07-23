import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import train_test_split
from . import utils

from hapcancer.etl.utils import batching_parquet_file
from hapcancer.config_manager import ConfigInterface

class SetupMammogramEligibility(ConfigInterface):
    def __init__(self, config_dir: str, config_defaults: dict):
        super().__init__(config_dir, config_defaults)        
        
        # -- field
        self.fields = self.fields_cfg["fields"]
        self.mamm_field_suffix = '_MAMOGRAFIA'
        self.anamnesis_field_suffix = '_ANAMNESE'
        
        self.merge_data_filename = "base_merged_data.parquet"
        self.merged_df = None

        self.shortcuts_path = self.transformed_data_path.joinpath(self.files_and_folders_cfg['transform']['transformed_files']['shortcut_terms']+'.parquet')

        self.mammograms_df = None

        self.gpd_aux = 14 # -- grace period days
        self.grace_period_start_in_days = int(self.followup_cfg["followup"]['grace_period_start_in_days'])
        self.maximum_months_of_followup = int(self.followup_cfg["followup"]['total_months_of_followup'])

        self.mamm_seq_filename = self.files_and_folders_cfg["load"]["load_files"]["seq_per_mammogram_filename"]
        self.output_filename = self.files_and_folders_cfg["load"]["load_files"]["final_data_with_eligibility_filename"]

    def _persist_structured_info(self, checkpoint_filename: str):
        cols = [
            'person_id', 'mammogram_id', 'mammogram_current_result',
            'monthly_payment_min', 'monthly_payment_max', 
            'bmi', 'menarche_age',
            'age_at_first_mammogram', 'first_mammogram_date', 'age_at_mammogram',
            'breastfeeding_cat', 'age_at_mammogram_old', 'ca_mama_mae_cat_-1.0',
            'ca_mama_mae_cat_0.0', 'ca_mama_mae_cat_1.0', 'ca_mama_irma_cat_-1.0',
            'ca_mama_irma_cat_0.0', 'ca_mama_irma_cat_1.0', 'ca_mama_avo_cat_-1.0',
            'ca_mama_avo_cat_0.0', 'ca_mama_avo_cat_1.0', 'ca_mama_tia_cat_-1.0',
            'ca_mama_tia_cat_0.0', 'ca_mama_tia_cat_1.0',
            'menopause_age', 'is_missing_children',
            'is_missing_miscarriage', 'number_of_children', 'number_of_miscarriage',
            'zipcode_cat_index', 'zipcode_embedding_0', 'zipcode_embedding_1',
            'zipcode_embedding_2', 'zipcode_embedding_3', 'zipcode_embedding_4',
            'zipcode_embedding_5', 'zipcode_embedding_6', 'zipcode_embedding_7',
            '14days_1yr_followup', '1yr_2yr_followup', '2yr_3yr_followup',
            '3yr_4yr_followup', '4yr_5yr_followup', '5yr_6yr_followup',
            '6yr_7yr_followup', '7yr_8yr_followup', '8yr_9yr_followup',
            '9yr_10yr_followup', 'event_indicator_1yr',
            'event_indicator_2yr', 'event_indicator_3yr', 'event_indicator_4yr',
            'event_indicator_5yr', 'event_indicator_6yr', 'event_indicator_7yr', 
            'event_indicator_8yr', 'event_indicator_9yr', 'event_indicator_10yr',
            'survival_time_1yr', 'survival_time_2yr', 'survival_time_3yr', 
            'survival_time_4yr', 'survival_time_5yr', 'survival_time_6yr', 
            'survival_time_7yr', 'survival_time_8yr', 'survival_time_9yr', 'survival_time_10yr',
            'eligibility_0yr_1yr', 'eligibility_1yr_2yr', 'eligibility_2yr_3yr',
            'eligibility_3yr_4yr', 'eligibility_4yr_5yr', 'eligibility_5yr_6yr',
            'eligibility_6yr_7yr', 'eligibility_7yr_8yr', 'eligibility_8yr_9yr',
            'eligibility_9yr_10yr', "shortcut_terms_flag"
        ]
        cols = [ nm for nm in cols if nm in self.mammograms_df.columns ]
        self.mammograms_df[cols].to_parquet(self.dataset_path.joinpath(checkpoint_filename))

    def _persist_seq_info(self, checkpoint_filename: str):
        cols = [
            'person_id', 'mammogram_id', 'mammogram_current_date',
            'mammogram_prior_codes', 'mammogram_prior_dates',
            'mammogram_prior_birads', 'mammogram_current_result', 
            'mammogram_complete_dates', 'mammogram_complete_birads', 
            'mammogram_complete_codes', 'biopsy_dates', 'biopsy_results', 
            'event_benign_birads_indices', 'event_benign_biopsy_indices', 
            'event_birads5_indices', 'event_birads6_indices', 
            'event_biopsy_indices', 'new_event_birads5_indices', 
            'earliest_positive_birads5', 'earliest_positive_birads6', 
            'earliest_positive_biopsy', 'event_date', 'interval_mammogram_to_event_date'
        ]
        self.mammograms_df[cols].to_parquet(self.dataset_path.joinpath(checkpoint_filename))
    
    def _map_transform_patients_to_mammograms(self, fixed_features_columns, timed_features_columns):
        '''
            Change the main format of the current dataset so that each row 
            represents a mammogram.

            Args:
            -----
                fixed_features_columns: List[str]. List of column names that represent
                fixed time features.
                timed_features_columns: List[str]. List of column names that represent
                features that change over time.
        '''
        fraction = 1.0       # e.g., 20%
        seed = 42             # set for reproducibility
        rng = np.random.default_rng(seed)

        self.mammograms_df = []
        SRC = self.load_path.joinpath(self.merge_data_filename)
        for batch in batching_parquet_file(SRC, batch_size=30000):
            mask = rng.random(len(batch)) < fraction
            if not mask.any():
                continue
            df_sub = batch.iloc[mask]
            print(batch.shape, df_sub.shape)
            
            table = utils.process_chunk(
                df_sub, self.fields, 
                fixed_features_columns, 
                timed_features_columns,
                mamm_field_suffix=self.mamm_field_suffix,
                anamnesis_field_suffix=self.anamnesis_field_suffix
            )  # same function as above
            self.mammograms_df.append(table)
            
        self.mammograms_df = pd.concat(self.mammograms_df, ignore_index=True).reset_index(drop=True)
        
        col_cd_pessoa = self.fields["person_id"]
        col_dt_nasc = self.fields["person_birthdate"]
        self.merged_df = pd.read_parquet(self.load_path.joinpath(self.merge_data_filename), columns=[col_cd_pessoa, col_dt_nasc])
        self.mammograms_df = self.mammograms_df.merge(self.merged_df, left_on="person_id", right_on=col_cd_pessoa, how="left")
        self.mammograms_df = self.mammograms_df.drop(columns=[col_cd_pessoa])
        self.mammograms_df = self.mammograms_df.rename({col_dt_nasc: "birthdate"}, axis=1)
    
    def _process_new_mammogram_data(self):
        '''
            ...
        '''
        col_cd_pessoa = self.fields["person_id"]
        col_dt_atend_mamografia = self.fields["mammogram_date"]+self.mamm_field_suffix
        col_cd_atend = self.fields["mammogram_id_final"]
        col_dt_nasc = self.fields["person_birthdate"]
        col_birads = "birads_labels"

        #assert col_dt_nasc in self.mammograms_df.columns, "birth date not included"

        self.mammograms_df["first_mammogram_date"] = self.mammograms_df["mammogram_complete_dates"].apply(lambda x: x[0]) 
        self.mammograms_df["age_at_first_mammogram"] = self.mammograms_df["first_mammogram_date"] - self.mammograms_df['birthdate']
        self.mammograms_df["age_at_first_mammogram"] = self.mammograms_df["age_at_first_mammogram"].apply(lambda x: np.timedelta64(x).astype('timedelta64[Y]')/np.timedelta64(1, 'Y') )

        self.mammograms_df["age_at_mammogram"] = self.mammograms_df['mammogram_current_date'] - self.mammograms_df['birthdate']
        self.mammograms_df["age_at_mammogram"] = self.mammograms_df["age_at_mammogram"].apply(lambda x: np.timedelta64(x).astype('timedelta64[Y]')/np.timedelta64(1, 'Y') )
        self.mammograms_df = self.mammograms_df.rename({
            'VL_MENSALIDADE_MAX': 'monthly_payment_max',
            'VL_MENSALIDADE_MIN': 'monthly_payment_min',
            'DS_MENARCA_FMT': 'menarche_age', 'BMI_PREDICT_RANDFOR': 'bmi',
            "DS_MENOPAUSA_FMT": 'menopause_age'
        }, axis=1)

        # -- calculate date of the event, when the case
        conditional_birads_val = lambda birads_arr, val: np.where(np.array(birads_arr) == val)[0]
        conditional_biopsy_val = lambda biopsy_arr, val: np.where(np.array(biopsy_arr) == val)[0]

        # ---- check which indices correspond to a benign case (1/2/3 birads or negative biopsy). birads 0/4 should not be considered as benign.
        self.mammograms_df["event_benign_birads_indices"] = self.mammograms_df["mammogram_complete_birads"].apply(lambda arr: np.concatenate((conditional_birads_val(arr, 1), conditional_birads_val(arr, 2), conditional_birads_val(arr, 3))) )
        self.mammograms_df["event_benign_biopsy_indices"] = self.mammograms_df["biopsy_results"].apply(lambda arr: conditional_biopsy_val(arr, 0) )
        # ---- check which indices correspond to a positive case (5/6 birads or positive biopsy)
        self.mammograms_df["event_birads5_indices"] = self.mammograms_df["mammogram_complete_birads"].apply(lambda arr: conditional_birads_val(arr, 5) )
        self.mammograms_df["event_birads6_indices"] = self.mammograms_df["mammogram_complete_birads"].apply(lambda arr: conditional_birads_val(arr, 6) )
        self.mammograms_df["event_biopsy_indices"] = self.mammograms_df["biopsy_results"].apply(lambda arr: conditional_biopsy_val(arr, 1) )

        # -- do a check (only when biopsy is included)
        #sub_example = self.mammograms_df[self.mammograms_df["biopsy_results"].apply(lambda x: True if type(x)==list and 1 in x else False)]
        #print(sub_example["biopsy_results"].iat[0], type(sub_example["biopsy_results"].iat[0]))
        #print(sub_example[["biopsy_results", "event_biopsy_indices", "event_birads5_indices", "event_birads6_indices", "event_benign_birads_indices"]])

        # ---- Validate the occurrences of BIRADS-5.
        subset_cols = [
            "event_birads5_indices", "biopsy_dates", "biopsy_results", "mammogram_complete_dates", 
            "event_birads6_indices", "event_biopsy_indices", "event_benign_biopsy_indices", 
            "event_benign_birads_indices"
        ]
        birads_5_val_version = self.followup_cfg["followup"]["birads_5"]["validation_version"]
        birads_5_val_interval_months = self.followup_cfg["followup"]["birads_5"]["validation_interval_months"]
        # -- define how validation for BI-RADS 5 should be done.
        birads_5_validation_f = utils.get_valid_birads_5_indices_v1 # v1
        if birads_5_val_version==0: # just keep as it is.
            birads_5_validation_f = utils.get_valid_birads_5_indices_v0
        elif birads_5_val_version==2: # v2
            birads_5_validation_f = utils.get_valid_birads_5_indices_v2
        self.mammograms_df["new_event_birads5_indices"] = self.mammograms_df[subset_cols].apply(lambda row: birads_5_validation_f(row, grace_period_days=7, interval_months=birads_5_val_interval_months), axis=1)

        self.mammograms_df["earliest_positive_birads5"] = self.mammograms_df[["mammogram_complete_dates", "new_event_birads5_indices"]].apply(lambda x: min(x["mammogram_complete_dates"][x["new_event_birads5_indices"]]) if len(x["new_event_birads5_indices"])>0 else np.nan, axis=1)
        self.mammograms_df["earliest_positive_birads6"] = self.mammograms_df[["mammogram_complete_dates", "event_birads6_indices"]].apply(lambda x: min(x["mammogram_complete_dates"][x["event_birads6_indices"]]) if len(x["event_birads6_indices"])>0 else np.nan, axis=1)
        self.mammograms_df["earliest_positive_biopsy"] = self.mammograms_df[["biopsy_dates", "event_biopsy_indices"]].apply(lambda x: min(x["biopsy_dates"][x["event_biopsy_indices"]]) if len(x["event_biopsy_indices"])>0 else np.nan, axis=1)
        self.mammograms_df["event_date"] = self.mammograms_df[["earliest_positive_birads5", "earliest_positive_birads6", "earliest_positive_biopsy"]].apply(lambda x: min([elem for elem in x if pd.notna(elem)]) if len([elem for elem in x if pd.notna(elem)])>0 else np.nan, axis=1)

        # ---- if the event date happens before the current mammogram, then we create a flag for this case
        self.mammograms_df["interval_mammogram_to_event_date"] = self.mammograms_df['event_date'] - self.mammograms_df['mammogram_current_date']
        self.mammograms_df["interval_mammogram_to_event_date"] = self.mammograms_df["interval_mammogram_to_event_date"].apply(lambda x: np.timedelta64(x).astype('timedelta64[D]')/np.timedelta64(1, 'D') if pd.notna(x) else np.nan)

        # 1000. Extra: fix menopause category
        #colnames = ["DS_MENOPAUSA_FMT_IMPUTATION", "age_at_mammogram"]
        #self.mammograms_df["DS_MENOPAUSA_FMT_IMPUTATION"] = self.mammograms_df[colnames].apply(lambda x: x[colnames[0]] if x[colnames[0]]<=x[colnames[1]] else np.nan, axis=1)
        #colnames = ["DS_MENOPAUSA_FMT_IMPUTATION", "DS_MENOPAUSA_FMT"]
        #self.mammograms_df["DS_MENOPAUSA_FMT"] = self.mammograms_df[colnames].apply(lambda x: x[colnames[1]] if pd.notna(x[colnames[1]]) else x[colnames[0]], axis=1)

        # -- create categories
        # ---- menopause
        #bins = [-1, 39, 44, 49, 54, 100]
        #labels = ['<=39', '40-44', '45-49', '50-54', '>=55']
        #self.mammograms_df['menopause_category'] = pd.cut(self.mammograms_df["DS_MENOPAUSA_FMT_IMPUTATION"], bins=bins, labels=labels)
        #self.mammograms_df['menopause_category'] = self.mammograms_df['menopause_category'].cat.add_categories('Not yet menopausal').fillna('Not yet menopausal')

        # 1001. Get the most recent information (not nan) of number of children and number of miscarriage
        self.mammograms_df["NU_GESTACAO_FMT"] = self.mammograms_df["NU_GESTACAO_FMT"].apply(lambda x: [ elem for elem in x if pd.notna(elem) ] if type(x)==list else x ).apply(lambda x: max(x) if type(x)==list and len(x)>0 else np.nan)
        self.mammograms_df["NU_GESTACAO_ABORTO_FMT"] = self.mammograms_df["NU_GESTACAO_ABORTO_FMT"].apply(lambda x: [ elem for elem in x if pd.notna(elem) ] if type(x)==list else x ).apply(lambda x: max(x) if type(x)==list and len(x)>0 else np.nan)

        self.mammograms_df["NU_GESTACAO_FMT"] = self.mammograms_df["NU_GESTACAO_FMT"].apply(lambda x: 6 if x>=6 else x) # create a group for 6 children or more
        self.mammograms_df["NU_GESTACAO_ABORTO_FMT"] = self.mammograms_df["NU_GESTACAO_ABORTO_FMT"].apply(lambda x: 5 if x>=5 else x) # create a group for 5 miscarriage or more

        # 1002. Impute missing monthly payment
        #self.mammograms_df["monthly_payment_min"] = self.mammograms_df["monthly_payment_min"].fillna(self.mammograms_df["monthly_payment_min"].mean())
        #self.mammograms_df["monthly_payment_max"] = self.mammograms_df["monthly_payment_max"].fillna(self.mammograms_df["monthly_payment_max"].mean())
#
        # 1003. Adjust familial history information
        subset_cols = ['FL_CA_MAMA_MAE_FMT', 'FL_CA_MAMA_AVO_FMT', 'FL_CA_MAMA_IRMA_FMT', 'FL_CA_MAMA_TIA_FMT', 'FL_ALEITAMENTO_FMT']
        for cur_col in subset_cols:
            self.mammograms_df[cur_col] = self.mammograms_df[cur_col].apply(lambda x: max(x) if type(x)!=float and len(x)>0 else np.nan)

        self.mammograms_df['ca_mama_mae_cat'] = self.mammograms_df['FL_CA_MAMA_MAE_FMT'].fillna(-1).astype('category')
        self.mammograms_df['ca_mama_irma_cat'] = self.mammograms_df['FL_CA_MAMA_IRMA_FMT'].fillna(-1).astype('category')
        self.mammograms_df['ca_mama_avo_cat'] = self.mammograms_df['FL_CA_MAMA_AVO_FMT'].fillna(-1).astype('category')
        self.mammograms_df['ca_mama_tia_cat'] = self.mammograms_df['FL_CA_MAMA_TIA_FMT'].fillna(-1).astype('category')
        self.mammograms_df['breastfeeding_cat'] = self.mammograms_df['FL_ALEITAMENTO_FMT'].fillna(0).astype('category')

        subset_cols = ['FL_MASTECTOMIA_MD_FMT', 'FL_MASTECTOMIA_ME_FMT', 'FL_PLASTICA_ME_FMT', 'FL_PLASTICA_MD_FMT']
        for cur_col in subset_cols:
            self.mammograms_df[cur_col] = self.mammograms_df[cur_col].apply(lambda x: max(x) if type(x)!=float and len(x)>0 else np.nan)

        # 1004. Impute menarche age (maybe should impute in another fashion? like conditioned on bmi) -> push to after splitting
        #menarche_mean, menarche_std = self.mammograms_df["menarche_age"].mean(), self.mammograms_df["menarche_age"].std()
        #self.mammograms_df["menarche_age"] = self.mammograms_df["menarche_age"].apply(lambda x: np.floor(np.random.normal(menarche_mean, menarche_std)) if pd.isna(x) else x)
        self.mammograms_df["biopsy_dates"] = self.mammograms_df["biopsy_dates"].apply(lambda x: [ elem for elem in x] if type(x)==np.ndarray and x.shape[0]>0 else [])
        remove_cols = [
            'FL_CA_MAMA_MAE_FMT', 'FL_CA_MAMA_AVO_FMT',
            'FL_CA_MAMA_IRMA_FMT', 'FL_CA_MAMA_TIA_FMT'
        ]
        self.mammograms_df = self.mammograms_df.drop(columns=remove_cols)
        
    def _transform_features(self):
        '''
        
        '''
        self.mammograms_df = self.mammograms_df.reset_index(drop=True)

        # -- current mammogram bi-rads
        self.mammograms_df["mammogram_current_result"] = self.mammograms_df["mammogram_current_result"].astype(int)

        # -- log transform of monthly payment (very skewed)
        #self.mammograms_df["monthly_payment_min"] = np.log(self.mammograms_df["monthly_payment_min"]+1)
        #self.mammograms_df["monthly_payment_max"] = np.log(self.mammograms_df["monthly_payment_max"]+1)

        # -- z-score for bmi
        # -- now, do it later in the training phase
        #bmi_mean, bmi_std = self.mammograms_df["bmi"].mean(), self.mammograms_df["bmi"].std()
        #self.mammograms_df["bmi"] = (self.mammograms_df["bmi"] - bmi_mean)/bmi_std

        # -- z-score for menarche age
        # -- now, do it later in the training phase
        #menarche_mean, menarche_std = self.mammograms_df["menarche_age"].mean(), self.mammograms_df["menarche_age"].std()
        #self.mammograms_df["menarche_age"] = (self.mammograms_df["menarche_age"] - menarche_mean)/menarche_std

        # -- z-score for age at mammogram
        # -- now, do it later in the training phase
        #zscore_mean, zscore_std = self.mammograms_df["age_at_first_mammogram"].mean(), self.mammograms_df["age_at_first_mammogram"].std()
        #self.mammograms_df["age_at_first_mammogram"] = (self.mammograms_df["age_at_first_mammogram"] - zscore_mean)/zscore_std

        # -- z-score for age at mammogram
        self.mammograms_df["age_at_mammogram_old"] = self.mammograms_df["age_at_mammogram"].copy()
        # -- now, do it later in the training phase
        #zscore_mean, zscore_std = self.mammograms_df["age_at_mammogram"].mean(), self.mammograms_df["age_at_mammogram"].std()
        #self.mammograms_df["age_at_mammogram"] = (self.mammograms_df["age_at_mammogram"] - zscore_mean)/zscore_std

        # -- one-hot encoding of familial history
        cols = ["ca_mama_mae_cat", "ca_mama_irma_cat", "ca_mama_avo_cat", "ca_mama_tia_cat"]
        self.mammograms_df = pd.get_dummies(self.mammograms_df, columns=cols)

        ## -- menopause category
        #age_group_mapping = {
        #    "<=39": 0,
        #    "40-44": 1,
        #    "45-49": 2,
        #    "50-54": 3,
        #    ">=55": 4,
        #    "Not yet menopausal": -1  # Special category
        #}

        #self.mammograms_df["menopause_category_ordered"] = self.mammograms_df["menopause_category"].map(age_group_mapping).astype(int)
        # ---- normalize (for neural network)
        # -- now, do it later in the training phase
        #self.mammograms_df['menopause_category_ordered'] = (self.mammograms_df['menopause_category_ordered'] - self.mammograms_df['menopause_category_ordered'].min()) / \
        #                                                  (self.mammograms_df['menopause_category_ordered'].max() - self.mammograms_df['menopause_category_ordered'].min())

        #self.mammograms_df = self.mammograms_df.drop(columns="menopause_category")

        # -- number of children
        self.mammograms_df['is_missing_children'] = self.mammograms_df['NU_GESTACAO_FMT'].isna().astype(int)
        self.mammograms_df['is_missing_miscarriage'] = self.mammograms_df['NU_GESTACAO_ABORTO_FMT'].isna().astype(int)

        self.mammograms_df['number_of_children'] = self.mammograms_df['NU_GESTACAO_FMT'].fillna(-1)
        self.mammograms_df['number_of_miscarriage'] = self.mammograms_df['NU_GESTACAO_ABORTO_FMT'].fillna(-1)

        # ---- normalize (Min-Max Scaling)
        # -- now, do it later in the training phase
        #self.mammograms_df['number_of_children'] = (self.mammograms_df['number_of_children'] - self.mammograms_df['number_of_children'].min()) / (self.mammograms_df['number_of_children'].max() - self.mammograms_df['number_of_children'].min())
        #self.mammograms_df['number_of_miscarriage'] = (self.mammograms_df['number_of_miscarriage'] - self.mammograms_df['number_of_miscarriage'].min()) / (self.mammograms_df['number_of_miscarriage'].max() - self.mammograms_df['number_of_miscarriage'].min())

        # -- zip code categories
        num_categories = self.mammograms_df["zipcode_cat"].unique().shape[0]  # Example with 50 unique categories
        zipcode_to_integer = defaultdict(lambda: -1, zip(self.mammograms_df["zipcode_cat"].unique(), [ n for n in range(num_categories) ]))
        self.mammograms_df["zipcode_cat_index"] = self.mammograms_df["zipcode_cat"].map(zipcode_to_integer)
        self.mammograms_df["zipcode_cat_index"] = self.mammograms_df["zipcode_cat_index"].astype("category").cat.codes
        # ---- create and pass through the embedding layer (there might be a problem here, when we add new categories)
        embedding_dim = 8 
        embedding_layer = nn.Embedding(num_categories, embedding_dim)
        category_tensor = torch.tensor(self.mammograms_df["zipcode_cat_index"].values, dtype=torch.long)
        embedded_output = pd.DataFrame(embedding_layer(category_tensor).detach().numpy())
        embedded_output.columns = [ f'zipcode_embedding_{n}' for n in range(embedding_dim) ]

        self.mammograms_df = pd.concat([self.mammograms_df, embedded_output], axis=1)
        # -- we are throwing important information before using them: previous implants and/or surgery
        remove_cols = [
            'FL_MASTECTOMIA_MD_FMT', 'FL_MASTECTOMIA_ME_FMT',
            'FL_PLASTICA_MD_FMT', 'FL_PLASTICA_ME_FMT', 'DT_MASTECTOMIA_MD_FMT', 'DT_MASTECTOMIA_ME_FMT',
            'DT_PLASTICA_MD_FMT', 'DT_PLASTICA_ME_FMT', 'NU_GESTACAO_FMT', 'NU_GESTACAO_ABORTO_FMT', 'zipcode_cat'
        ]
        self.mammograms_df = self.mammograms_df.drop(columns=remove_cols)


    def _calculate_followup(self):

        # -- follow-up flags
        # -- set the follow-up dates
        self.gpd_aux = 14 # -- for observability, we do not need to create a grace period (do we? it would make more robust, definetely)
        maxtf_aux = self.maximum_months_of_followup
        self.mammograms_df[f"{self.gpd_aux:.0f}days_1yr_followup_date"] = self.mammograms_df["mammogram_current_date"].apply(lambda x: pd.Timestamp(x) + pd.DateOffset(days=self.gpd_aux))
        self.mammograms_df["1yr_followup_date"] = self.mammograms_df["mammogram_current_date"].apply(lambda x: pd.Timestamp(x) + pd.DateOffset(months=12))
        self.mammograms_df["2yr_followup_date"] = self.mammograms_df["mammogram_current_date"].apply(lambda x: pd.Timestamp(x) + pd.DateOffset(months=24))
        self.mammograms_df["3yr_followup_date"] = self.mammograms_df["mammogram_current_date"].apply(lambda x: pd.Timestamp(x) + pd.DateOffset(months=36))
        self.mammograms_df["4yr_followup_date"] = self.mammograms_df["mammogram_current_date"].apply(lambda x: pd.Timestamp(x) + pd.DateOffset(months=48))
        self.mammograms_df["5yr_followup_date"] = self.mammograms_df["mammogram_current_date"].apply(lambda x: pd.Timestamp(x) + pd.DateOffset(months=60))
        self.mammograms_df["6yr_followup_date"] = self.mammograms_df["mammogram_current_date"].apply(lambda x: pd.Timestamp(x) + pd.DateOffset(months=72))
        self.mammograms_df["7yr_followup_date"] = self.mammograms_df["mammogram_current_date"].apply(lambda x: pd.Timestamp(x) + pd.DateOffset(months=84))
        self.mammograms_df["8yr_followup_date"] = self.mammograms_df["mammogram_current_date"].apply(lambda x: pd.Timestamp(x) + pd.DateOffset(months=96))
        self.mammograms_df["9yr_followup_date"] = self.mammograms_df["mammogram_current_date"].apply(lambda x: pd.Timestamp(x) + pd.DateOffset(months=108))
        self.mammograms_df["10yr_followup_date"] = self.mammograms_df["mammogram_current_date"].apply(lambda x: pd.Timestamp(x) + pd.DateOffset(months=120))


        # --------------------------------------------------------------------------------------------------------------
        '''
            The follow-up flags calculated in the code below are extremely important when defining the eligibility criteria
            for each mammogram to be used during model development. In this project, there may be an ongoing discussion on
            which eligibility criteria to apply for each case, therefore below we calculate each possible flag.

            For instance, consider the scenario where we are performing a 2-year prediction for each mammogram. Which eligibility
            to apply here? One obvious choice is to select only the mammograms where there is at least one follow-up exam
            (mammogram or biopsy) between X months and 2 years after the current exam. Or should we select a general approach
            where we select only a mammogram which has at least one follow-up exam between 1 and 5 years?

            This choice might not be so obvious if we perform prediciton at several different points, like one model for 2, 3, 4
            and 5 years.
        '''
        print("calculating follow-up flags ...")

        # ---- between X days and 1yr after the date of the current mammogram
        # ---- create a flag to signal whether there is at least one year of followup (biopsy) for the current mammogram
        lower_date_arr, upper_date_arr = self.mammograms_df[f"{self.gpd_aux:.0f}days_1yr_followup_date"], self.mammograms_df[f"1yr_followup_date"]
        biopsy_results_arr, biopsy_dates_arr = self.mammograms_df["biopsy_results"], self.mammograms_df["biopsy_dates"]
        benign_labels = [0]
        self.mammograms_df[f"{self.gpd_aux:.0f}days_1yr_followup_biopsy"] = utils.get_interval_label_for_benign_followup(
            lower_date_arr, upper_date_arr, biopsy_results_arr, biopsy_dates_arr, benign_labels, return_binary=False
        )
        # -- benign follow-up for mammogram
        mamm_results_arr, mamm_dates_arr = self.mammograms_df["mammogram_complete_birads"], self.mammograms_df["mammogram_complete_dates"]
        benign_labels = [0,1,2,3,4]
        self.mammograms_df[f"{self.gpd_aux:.0f}days_1yr_followup_mammogram"] = utils.get_interval_label_for_benign_followup(
            lower_date_arr, upper_date_arr, mamm_results_arr, mamm_dates_arr, benign_labels, return_binary=False
        )
        # -- unify
        subcols = [f"{self.gpd_aux:.0f}days_1yr_followup_biopsy", f"{self.gpd_aux:.0f}days_1yr_followup_mammogram"]
        self.mammograms_df[f"{self.gpd_aux:.0f}days_1yr_followup"] = self.mammograms_df[subcols].apply(lambda x: 1 if x[subcols[0]]>0 or x[subcols[1]]>0 else 0, axis=1)
        
        # -- now for interval 1-2, 2-3, 3-4, 4-5, 5-6, 6-7, 7-8, 8-9, 9-10.
        for year_k in [1,2,3,4,5,6,7,8,9]:
            # -- benign follow-up for biopsies
            lower_date_arr, upper_date_arr = self.mammograms_df[f"{year_k:.0f}yr_followup_date"], self.mammograms_df[f"{year_k+1:.0f}yr_followup_date"]
            biopsy_results_arr, biopsy_dates_arr = self.mammograms_df["biopsy_results"], self.mammograms_df["biopsy_dates"]
            benign_labels = [0]
            self.mammograms_df[f"{year_k:.0f}yr_{year_k+1:.0f}yr_followup_biopsy"] = utils.get_interval_label_for_benign_followup(
                lower_date_arr, upper_date_arr, biopsy_results_arr, biopsy_dates_arr, benign_labels, return_binary=False
            )

            # -- benign follow-up for mammogram
            mamm_results_arr, mamm_dates_arr = self.mammograms_df["mammogram_complete_birads"], self.mammograms_df["mammogram_complete_dates"]
            benign_labels = [0,1,2,3,4,5,6,7,8,9]
            self.mammograms_df[f"{year_k:.0f}yr_{year_k+1:.0f}yr_followup_mammogram"] = utils.get_interval_label_for_benign_followup(
                lower_date_arr, upper_date_arr, mamm_results_arr, mamm_dates_arr, benign_labels, return_binary=False
            )

            # -- unify (right now, final flag is binary)
            subcols = [f"{year_k:.0f}yr_{year_k+1:.0f}yr_followup_biopsy", f"{year_k:.0f}yr_{year_k+1:.0f}yr_followup_mammogram"]
            self.mammograms_df[f"{year_k:.0f}yr_{year_k+1:.0f}yr_followup"] = self.mammograms_df[subcols].apply(lambda x: 1 if x[subcols[0]]>0 or x[subcols[1]]>0 else 0, axis=1)

        print("calculating follow-up flags ... done")
        # --------------------------------------------------------------------------------------------------------------

        # -- calculate time to event
        cohort_end_date = pd.Timestamp(self.followup_cfg["followup"]["cohort_end_date"])

        interval_label_0_1, surv_timedelta_0_1, surv_timedelta_d_0_1 = utils.calculate_interval_label(self.mammograms_df["mammogram_current_date"], self.mammograms_df["event_date"], lower_limit_time=0, upper_limit_time=12, limit_date=cohort_end_date)
        interval_label_1_2, surv_timedelta_1_2, surv_timedelta_d_1_2 = utils.calculate_interval_label(self.mammograms_df["mammogram_current_date"], self.mammograms_df["event_date"], lower_limit_time=12, upper_limit_time=24, limit_date=cohort_end_date)
        interval_label_2_3, surv_timedelta_2_3, surv_timedelta_d_2_3 = utils.calculate_interval_label(self.mammograms_df["mammogram_current_date"], self.mammograms_df["event_date"], lower_limit_time=24, upper_limit_time=36, limit_date=cohort_end_date)
        interval_label_3_4, surv_timedelta_3_4, surv_timedelta_d_3_4 = utils.calculate_interval_label(self.mammograms_df["mammogram_current_date"], self.mammograms_df["event_date"], lower_limit_time=36, upper_limit_time=48, limit_date=cohort_end_date)
        interval_label_4_5, surv_timedelta_4_5, surv_timedelta_d_4_5 = utils.calculate_interval_label(self.mammograms_df["mammogram_current_date"], self.mammograms_df["event_date"], lower_limit_time=48, upper_limit_time=60, limit_date=cohort_end_date)
        interval_label_5_6, surv_timedelta_5_6, surv_timedelta_d_5_6 = utils.calculate_interval_label(self.mammograms_df["mammogram_current_date"], self.mammograms_df["event_date"], lower_limit_time=60, upper_limit_time=72, limit_date=cohort_end_date)
        interval_label_6_7, surv_timedelta_6_7, surv_timedelta_d_6_7 = utils.calculate_interval_label(self.mammograms_df["mammogram_current_date"], self.mammograms_df["event_date"], lower_limit_time=72, upper_limit_time=84, limit_date=cohort_end_date)
        interval_label_7_8, surv_timedelta_7_8, surv_timedelta_d_7_8 = utils.calculate_interval_label(self.mammograms_df["mammogram_current_date"], self.mammograms_df["event_date"], lower_limit_time=84, upper_limit_time=96, limit_date=cohort_end_date)
        interval_label_8_9, surv_timedelta_8_9, surv_timedelta_d_8_9 = utils.calculate_interval_label(self.mammograms_df["mammogram_current_date"], self.mammograms_df["event_date"], lower_limit_time=96, upper_limit_time=108, limit_date=cohort_end_date)
        interval_label_9_10, surv_timedelta_9_10, surv_timedelta_d_9_10 = utils.calculate_interval_label(self.mammograms_df["mammogram_current_date"], self.mammograms_df["event_date"], lower_limit_time=108, upper_limit_time=120, limit_date=cohort_end_date)

        self.mammograms_df["event_indicator_1yr"] = interval_label_0_1
        self.mammograms_df["event_indicator_2yr"] = interval_label_1_2
        self.mammograms_df["event_indicator_3yr"] = interval_label_2_3
        self.mammograms_df["event_indicator_4yr"] = interval_label_3_4
        self.mammograms_df["event_indicator_5yr"] = interval_label_4_5
        self.mammograms_df["event_indicator_6yr"] = interval_label_5_6
        self.mammograms_df["event_indicator_7yr"] = interval_label_6_7
        self.mammograms_df["event_indicator_8yr"] = interval_label_7_8
        self.mammograms_df["event_indicator_9yr"] = interval_label_8_9
        self.mammograms_df["event_indicator_10yr"] = interval_label_9_10

        # -- survival time will be used for the cases where events happened before the end of the grace period
        self.mammograms_df["survival_time_1yr"] = surv_timedelta_d_0_1
        self.mammograms_df["survival_time_2yr"] = surv_timedelta_d_1_2
        self.mammograms_df["survival_time_3yr"] = surv_timedelta_d_2_3
        self.mammograms_df["survival_time_4yr"] = surv_timedelta_d_3_4
        self.mammograms_df["survival_time_5yr"] = surv_timedelta_d_4_5
        self.mammograms_df["survival_time_6yr"] = surv_timedelta_d_5_6
        self.mammograms_df["survival_time_7yr"] = surv_timedelta_d_6_7
        self.mammograms_df["survival_time_8yr"] = surv_timedelta_d_7_8
        self.mammograms_df["survival_time_9yr"] = surv_timedelta_d_8_9
        self.mammograms_df["survival_time_10yr"] = surv_timedelta_d_9_10

    def _label_index_mammograms_with_shortcuts(self):
        '''
            ...
        '''
        if not self.shortcuts_path.is_file():
            print("no shortcuts file")

        shortcuts_df = pd.read_parquet(self.shortcuts_path)
        shortcuts_df = shortcuts_df[shortcuts_df['has_surgical_term']==True].copy()
        keys_with_terms = defaultdict(lambda: False, { key : True for key in shortcuts_df["key"].tolist() })

        print(f"Total reports with shortcut terms: {len(keys_with_terms)}")

        # -- create flag for reports with shortcut terms (including the ones in the past)
        self.mammograms_df["shortcut_terms_flag"] = self.mammograms_df["mammogram_prior_codes"].apply(lambda lst: any([ keys_with_terms[elem] for elem in lst ]))

    
    def _apply_eligibility_multiyear(self):
        '''
            ...
        '''
        # -- before applying eligibility, save the complete population structured stats
        self._persist_structured_info("complete_pop_before_eligibility_no_seq.parquet")
        
        self.mammograms_df = self.mammograms_df.reset_index(drop=True)

        # -- final processing and cohort numbers
        cohort_numbers_dict = dict()

        # ------- ELIGIBILITY -------
        # -- consider mammograms starting from a given date
        start_date_mammogram = pd.Timestamp(self.followup_cfg["followup"]["start_date_mammogram"])
        mammograms_before_start = self.mammograms_df[self.mammograms_df["mammogram_current_date"]<start_date_mammogram].shape[0]
        self.mammograms_df = self.mammograms_df[self.mammograms_df["mammogram_current_date"]>=start_date_mammogram].copy()

        # -- number of mammograms with a past confirmatory exam (v2 -> comment if we go back to v1 (why?))
        #past_positive_exam = self.mammograms_df[self.mammograms_df["interval_mammogram_to_event_date"]<0].shape[0]
        self.mammograms_df = self.mammograms_df[(self.mammograms_df["interval_mammogram_to_event_date"]>=0) | (pd.isna(self.mammograms_df["interval_mammogram_to_event_date"]))].copy()

        # -- age criteria
        minimum_age, maximum_age = self.followup_cfg["followup"]["minimum_age"], self.followup_cfg["followup"]["maximum_age"]
        n_age_invalid = self.mammograms_df[(self.mammograms_df["age_at_mammogram"]<minimum_age) | (self.mammograms_df["age_at_mammogram"]>maximum_age)].shape[0]
        self.mammograms_df = self.mammograms_df[(self.mammograms_df["age_at_mammogram"]>=minimum_age) & (self.mammograms_df["age_at_mammogram"]<=maximum_age)].copy()

        # -- no event within X days after mammogram (cancer detection x cancer prediction) - we should test using 90, 120, 150 and 180 months.
        n_min_days_event = self.grace_period_start_in_days
        self.mammograms_df = self.mammograms_df[~((self.mammograms_df["survival_time_1yr"]<=n_min_days_event) & (self.mammograms_df["event_indicator_1yr"]==1))]

        # -- this should not happen here
        count_dupli = self.mammograms_df["mammogram_id"].value_counts().reset_index()
        codes_to_remove = count_dupli[count_dupli["count"]>1]["mammogram_id"].tolist()
        #print(len(codes_to_remove))
        self.mammograms_df = self.mammograms_df[~self.mammograms_df["mammogram_id"].isin(codes_to_remove)].copy()

        # -- corrected form of elibility
        event_indicator_ = [
            'event_indicator_1yr', 'event_indicator_2yr', 'event_indicator_3yr',
            'event_indicator_4yr', 'event_indicator_5yr', 'event_indicator_6yr',
            'event_indicator_7yr', 'event_indicator_8yr', 'event_indicator_9yr',
            'event_indicator_10yr'
        ]
        self.gpd_aux = 14
        observability_cols = [
            f"{self.gpd_aux:.0f}days_1yr_followup", f"1yr_2yr_followup", f"2yr_3yr_followup",
            f"3yr_4yr_followup", f"4yr_5yr_followup", f"5yr_6yr_followup", f"6yr_7yr_followup",
            f"7yr_8yr_followup", f"8yr_9yr_followup", f"9yr_10yr_followup"
        ]
        E = self.mammograms_df[event_indicator_].astype(int).to_numpy()
        O = self.mammograms_df[observability_cols].copy()

        cum_prev = np.cumsum(E, axis=1)
        # -- shift so that the sum defining P is j < t, not j <= t
        P = (np.concatenate([np.zeros((len(self.mammograms_df),1), dtype=int), cum_prev[:,:-1]], axis=1) == 0).astype(int)
        F = np.fliplr((np.cumsum(np.fliplr(E), axis=1) > 0).astype(int))

        # -- R_t: observed through end (any follow-up in t..5)
        R = np.fliplr(np.cumsum(np.fliplr(O), axis=1) > 0).astype(int)
        # -- define eligibility (P [event-free at the start of the current interval] AND [ {an event occurs at the current interval} OR {observable until the end of the 
        # -- current interval, by existing an follow-up exam in the current interval or in any of the following ones} ])
        #A = (P * ((E + R) > 0).astype(int)).astype(int)
        A = (P * ((F | R).astype(int))).astype(int)

        elig_cols = [
            'eligibility_0yr_1yr','eligibility_1yr_2yr', 'eligibility_2yr_3yr',
            'eligibility_3yr_4yr','eligibility_4yr_5yr', 'eligibility_5yr_6yr',
            'eligibility_6yr_7yr', 'eligibility_7yr_8yr', 'eligibility_8yr_9yr',
            'eligibility_9yr_10yr'
        ]
        self.mammograms_df[elig_cols] = A
        self.mammograms_df["eligible_for_training"] = self.mammograms_df[elig_cols].sum(axis=1)
        self.mammograms_df = self.mammograms_df[self.mammograms_df["eligible_for_training"]>0].copy()

        # -- probably we need to remove the obvious one and then remove the rest in other parts of the code
        remove_cols = [
            'mammogram_current_date', 'mammogram_prior_codes', 'mammogram_prior_dates', 'mammogram_complete_codes',
            'mammogram_prior_birads', 'mammogram_complete_dates', 'mammogram_complete_birads', 'biopsy_results',
            'birthdate', 'earliest_positive_biopsy', 'earliest_positive_birads5',
            'earliest_positive_birads6', "eligible_for_training", "FL_ALEITAMENTO_FMT", f'{self.gpd_aux:.0f}days_1yr_followup_date',
            '1yr_followup_date', '2yr_followup_date', '3yr_followup_date', '4yr_followup_date', '5yr_followup_date',
            '6yr_followup_date', '7yr_followup_date', '8yr_followup_date', '9yr_followup_date', '10yr_followup_date',
            f'{self.gpd_aux:.0f}days_1yr_followup_biopsy', f'{self.gpd_aux:.0f}days_1yr_followup_mammogram',
            '1yr_2yr_followup_biopsy', '1yr_2yr_followup_mammogram', '2yr_3yr_followup_biopsy',
            '2yr_3yr_followup_mammogram', '3yr_4yr_followup_biopsy', '3yr_4yr_followup_mammogram',
            '4yr_5yr_followup_biopsy', '4yr_5yr_followup_mammogram', '5yr_6yr_followup_biopsy', '5yr_6yr_followup_mammogram',
            '6yr_7yr_followup_biopsy', '6yr_7yr_followup_mammogram', '7yr_8yr_followup_biopsy', '7yr_8yr_followup_mammogram',
            '8yr_9yr_followup_biopsy', '8yr_9yr_followup_mammogram', '9yr_10yr_followup_biopsy', '9yr_10yr_followup_mammogram'
            "biopsy_dates", "event_biopsy_indices", "event_date", "new_event_birads5_indices", "event_benign_birads_indices",
            'event_benign_biopsy_indices', 'event_birads5_indices', 'event_birads6_indices', 
            'interval_mammogram_to_event_date'
        ]
        #self.mammograms_df = self.mammograms_df.drop(columns=remove_cols)

    def setup_mammogram_data(self, load_intermediate=False, verbose=True):
        '''
            Define the base for the whole dataset: Identify each mammogram and each person and all the features 
            that should be associated with it.

            Args:
            -----
                config: dictionary.
        '''
        # -- features that are constant through time
        fixed_features_columns = [
            'VL_MENSALIDADE_MIN', 'VL_MENSALIDADE_MAX', 'BMI_PREDICT_RANDFOR',
            'DS_MENARCA_FMT', 'zipcode_cat', 'age_at_first_mammogram',
            "DS_MENOPAUSA_FMT_IMPUTATION", "DS_MENOPAUSA_FMT"
        ]
        # -- features that might change according to the mammogram date
        timed_features_columns = [
            'NU_GESTACAO_FMT', 'NU_GESTACAO_ABORTO_FMT', 'FL_CA_MAMA_MAE_FMT', 'FL_CA_MAMA_AVO_FMT', 
            'FL_CA_MAMA_IRMA_FMT', 'FL_CA_MAMA_TIA_FMT', 'FL_MASTECTOMIA_MD_FMT', 'FL_MASTECTOMIA_ME_FMT', 
            'FL_PLASTICA_ME_FMT', 'FL_PLASTICA_MD_FMT', 'DT_PLASTICA_ME_FMT', 'DT_PLASTICA_MD_FMT', 
            'DT_MASTECTOMIA_ME_FMT', 'DT_MASTECTOMIA_MD_FMT', 'FL_ALEITAMENTO_FMT'
        ]
        if verbose: print("[map] person-centered dataset to a mammogram-centered dataset ...")
        self._map_transform_patients_to_mammograms(fixed_features_columns, timed_features_columns)
        if verbose: print("[process] mammogram-centered dataset ...")
        self._process_new_mammogram_data()
        if verbose: print("[transform] final features for modeling ...")
        self._transform_features()
        if verbose: print("[calculate] follow-up and survival time ...")
        self._calculate_followup()
        if verbose: print("[identify] mammograms with shortcut terms in past mammograms ...")
        self._label_index_mammograms_with_shortcuts()
        if verbose: print("[apply] eligibility criteria ...")
        self._apply_eligibility_multiyear()
        if verbose: print("[persist] final non-splitted datasets ...")
        # -- persist sequence information for eligible
        self._persist_seq_info(self.mamm_seq_filename)
        # -- persist structured information for eligible
        self._persist_structured_info(self.output_filename)