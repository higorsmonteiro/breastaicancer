import gc
import lmdb
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from typing import Optional, List, Union, Tuple
from hapcancer.etl.utils import batching_parquet_file
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.data import DataLoader
from libauc.sampler import DualSampler
from hapcancer.model.dataload.datasets import CancerDatasetSingleYearTFIDFPrecomputed
from hapcancer.config_manager import ConfigInterface

# -------------------------------------------------------------------------------------------------- #
# ------------------------------------------- FUNCTIONS -------------------------------------------- #
# -------------------------------------------------------------------------------------------------- #

class LMDBEmbedStore:
    '''
        LMDB reader.

        Used to access the past mammogram history embeddings for each mammogram id.
    '''
    def __init__(self, lmdb_path: str):  # adjust dim
        self.env = lmdb.open(lmdb_path, subdir=False, readonly=True, lock=True, readahead=False)

    def get(self, mid):
        with self.env.begin() as txn:
            buf = txn.get(str(mid).encode())
            if buf is None:
                return None
            return np.frombuffer(buf, dtype=np.float16)
        
def load_dataset(
    src: Path,
    columns: List[str],
    ids: Optional[List[str]] = None,
    id_column: Optional[str] = "mammogram_id",
    batch_size: Optional[int] = 200_000
) -> pd.DataFrame:
    dataset = []
    for batch in batching_parquet_file(src, columns=columns, batch_size=batch_size):
        if ids is not None:
            batch = batch[batch[id_column].isin(ids)].copy()
        dataset.append(batch)
    dataset = pd.concat(dataset, ignore_index=True)
    return dataset

def filter_shortcut_terms(
   struct_dataset: pd.DataFrame,
   shortcut_terms_flag_col: str
) -> pd.DataFrame:
    struct_dataset = struct_dataset[struct_dataset[shortcut_terms_flag_col]==False].copy()
    struct_dataset = struct_dataset.drop(columns=[shortcut_terms_flag_col]).copy()
    return struct_dataset

def standardize_dataset(
    struct_dataset: pd.DataFrame
) -> pd.DataFrame:
    # -- menarche age imputation
    menarche_mean, menarche_std = struct_dataset["menarche_age"].mean(), struct_dataset["menarche_age"].std()
    struct_dataset["menarche_age"] = struct_dataset["menarche_age"].apply(lambda x: np.floor(np.random.normal(menarche_mean, menarche_std)) if pd.isna(x) else x)
    # -- z-score for menarche age
    struct_dataset["menarche_age"] = (struct_dataset["menarche_age"] - menarche_mean)/menarche_std

    # -- monthly payment
    struct_dataset["monthly_payment_min"] = struct_dataset["monthly_payment_min"].fillna(struct_dataset["monthly_payment_min"].mean())
    struct_dataset["monthly_payment_max"] = struct_dataset["monthly_payment_max"].fillna(struct_dataset["monthly_payment_max"].mean())
    # -- log transform of monthly payment (very skewed)
    struct_dataset["monthly_payment_min"] = np.log(struct_dataset["monthly_payment_min"]+1)
    struct_dataset["monthly_payment_max"] = np.log(struct_dataset["monthly_payment_max"]+1)

    # -- z-score for bmi
    bmi_mean, bmi_std = struct_dataset["bmi"].mean(), struct_dataset["bmi"].std()
    struct_dataset["bmi"] = (struct_dataset["bmi"] - bmi_mean)/bmi_std

    # -- z-score for age at first mammogram and at mammogram
    zscore_mean, zscore_std = struct_dataset["age_at_first_mammogram"].mean(), struct_dataset["age_at_first_mammogram"].std()
    struct_dataset["age_at_first_mammogram"] = (struct_dataset["age_at_first_mammogram"] - zscore_mean)/zscore_std
    zscore_mean, zscore_std = struct_dataset["age_at_mammogram"].mean(), struct_dataset["age_at_mammogram"].std()
    struct_dataset["age_at_mammogram"] = (struct_dataset["age_at_mammogram"] - zscore_mean)/zscore_std

    # -- menopause category
    # ---- impute menopause age
    median, std = struct_dataset['menopause_age'].median(), struct_dataset['menopause_age'].std()
    struct_dataset["menopause_age_imputation"] = struct_dataset["menopause_age"].apply(lambda x: np.floor(np.random.normal(median, std)) if pd.isna(x) else x)

    # ---- fix menopause category (if imputation is larger than age)
    colnames = ["menopause_age_imputation", "age_at_mammogram"]
    struct_dataset["menopause_age_imputation"] = struct_dataset[colnames].apply(lambda x: x[colnames[0]] if x[colnames[0]]<=x[colnames[1]] else np.nan, axis=1)
    colnames = ["menopause_age_imputation", "menopause_age"]
    struct_dataset["menopause_age"] = struct_dataset[colnames].apply(lambda x: x[colnames[1]] if pd.notna(x[colnames[1]]) else x[colnames[0]], axis=1)

    bins = [-1, 39, 44, 49, 54, 100]
    labels = ['<=39', '40-44', '45-49', '50-54', '>=55']
    struct_dataset['menopause_category'] = pd.cut(struct_dataset["menopause_age"], bins=bins, labels=labels)
    struct_dataset['menopause_category'] = struct_dataset['menopause_category'].cat.add_categories('Not yet menopausal').fillna('Not yet menopausal')

    # -- menopause category
    age_group_mapping = {
        "<=39": 0,
        "40-44": 1,
        "45-49": 2,
        "50-54": 3,
        ">=55": 4,
        "Not yet menopausal": -1  # Special category
    }
    struct_dataset["menopause_category_ordered"] = struct_dataset["menopause_category"].map(age_group_mapping).astype(int)
    # -- normalize categories
    struct_dataset['menopause_category_ordered'] = (struct_dataset['menopause_category_ordered'] - struct_dataset['menopause_category_ordered'].min()) / \
                                                      (struct_dataset['menopause_category_ordered'].max() - struct_dataset['menopause_category_ordered'].min())
    # -- keeps the original name (convenient)
    struct_dataset["menopause_age"] = struct_dataset["menopause_category_ordered"].copy()
    struct_dataset = struct_dataset.drop(columns=["menopause_category", "menopause_category_ordered", "menopause_age_imputation"])
    return struct_dataset


def transform_eligibility_to_singleyear(
    feature_df: pd.DataFrame,
    target_year: int,
    followup_cols: List[str],
    event_cols: List[str],
    only_eligible: Optional[bool] = True,
):
    '''
        Essential function to transform eligibility of individual interval of years
        to the eligibility for prediction within k years.
    '''
    followup_arr = feature_df[followup_cols].values
    event_arr = feature_df[event_cols].values
    followup_arr = (np.fliplr(np.cumsum(np.fliplr(followup_arr), axis=1))>0).astype(int)
    event_arr = (np.cumsum(event_arr, axis=1)>0).astype(int)
    feature_df['eligibility'] = ((followup_arr[:,target_year-1] + event_arr[:, target_year-1])>0).astype(int)
    feature_df['event_indicator'] = event_arr[:,target_year-1]

    if only_eligible:
        feature_df = feature_df[feature_df["eligibility"]==1].copy()
    return feature_df

def process_data_for_singleyear(
    features_df: pd.DataFrame,
    sampling: Optional[bool] = False,
    sampling_strategy: Optional[str] = "oversampling",
    positive_ratio: Optional[int] = 8,
    id_column_name: Optional[str] = "mammogram_id",
    verbose: Optional[bool] = True
) -> Tuple[np.ndarray, pd.DataFrame, np.ndarray, np.ndarray]:
    # -- sampling strategy
    sample_features_df_1 = features_df[features_df[f"event_indicator"]==1].copy()
    sample_features_df_0 = features_df[features_df[f"event_indicator"]==0].copy()
    sample_features_df_1["label"] = [ 1 for n in range(sample_features_df_1.shape[0]) ]
    sample_features_df_0["label"] = [ 0 for n in range(sample_features_df_0.shape[0]) ]

    # -- 1:positive_ratio positive to negative ratio for oversampling/undersampling
    if sampling and sampling_strategy=="oversampling":
        goal_n = int(sample_features_df_0.shape[0]/positive_ratio)
        sample_features_df_1 = sample_features_df_1.sample(n=goal_n, replace=True)
        if verbose: 
            print(f"expected number of positives after enrichment: {goal_n:,}")
            print("positive to negative ratio after enrichment: ", sample_features_df_1.shape[0]/sample_features_df_0.shape[0])
    elif sampling and sampling_strategy=="undersampling":
        goal_n = int(sample_features_df_1.shape[0]*positive_ratio)
        sample_features_df_0 = sample_features_df_0.sample(n=goal_n, replace=False)
        if verbose: 
            print(f"expected number of negatives after undersampling: {goal_n:,}")
            print("positive to negative ratio after enrichment: ", sample_features_df_1.shape[0]/sample_features_df_0.shape[0])
    else:
        pass
    sample_features = pd.concat([sample_features_df_1, sample_features_df_0])
    print(sample_features.shape)
    mammogram_ids = sample_features[id_column_name].values
    labels, eligibility_mask = sample_features['label'].values, sample_features["eligibility"].values
    sample_features = sample_features.drop(columns=['label', "event_indicator", "eligibility"])
    return mammogram_ids, sample_features, labels, eligibility_mask

def get_dataloaders_singleyear(
    config: dict,
    feature_df: dict,
    max_len: Optional[int] = 80,
    with_transformer: Optional[bool] = False,
    is_training: Optional[bool] = False,
    device: Optional[str] = 'cpu',
    verbose: Optional[bool] = False
) -> Tuple[DataLoader, float]:
    verbose = True
    batch_size = config['training']['batch_size']
    num_workers = config['training']['num_workers']
    sampling_strategy = config['training']['sampling_strategy']
    negative_to_positive_ratio = config['training']['negative_to_positive_ratio']

    cols_to_remove = config['fields']['followup_columns']
    cols_to_remove += config['fields']['event_indicator_columns']
    cols_to_remove += config['fields']['multiyear_eligibility_columns']

    # -- select custom dataset
    selected_dataset = CancerDatasetSingleYearTFIDFPrecomputed

    sampling = False
    if is_training:
        sampling = True
    
    mamm_ids, feature_after_df, labels, eligibility_mask = process_data_for_singleyear(
        feature_df, sampling=sampling, sampling_strategy=sampling_strategy, 
        positive_ratio=negative_to_positive_ratio, verbose=verbose
    )
    feature_after_df = feature_after_df.drop(columns=['mammogram_id']+cols_to_remove).values.astype(float)
    positive_n = labels.sum()
    imratio = positive_n / labels.shape[0] # -- how?
    #print("labels:", labels[:4])
    #print("eligibility mask:", eligibility_mask[:4])
    #print(feature_after_df.shape, labels.shape, eligibility_mask.shape)
    #print("number of positives:", positive_n)
    #print(imratio)

    # ---- define dataset and dataloader
    cur_dataset = selected_dataset(mamm_ids, feature_after_df, labels, eligibility_mask, device=device)
    if is_training:
        cur_loader_sampler = DualSampler(cur_dataset, labels=labels, batch_size=batch_size, sampling_rate=0.5)
        cur_loader = DataLoader(cur_dataset, batch_size=batch_size, sampler=cur_loader_sampler, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    else:
        cur_loader = DataLoader(cur_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return cur_loader, imratio


# -------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------- #

class InputLoader(ConfigInterface):
    '''
        Interface to create the dataloaders for the IDs parsed.
    '''
    def __init__(self, config_dir: str, config_defaults: dict):
        super().__init__(config_dir, config_defaults)
        # -- main path variables for dataset
        self.device = 'cpu'

        self.struct_dataset_eligible_filename = self.files_and_folders_cfg["load"]["load_files"]["final_data_with_eligibility_filename"]
        self.final_mamm_seq_filename = self.files_and_folders_cfg["load"]["load_files"]["seq_per_mammogram_filename"]
        self.precomputed_path = Path(self.followup_cfg["precomputed"]["path"])
        self.precomputed_filename = self.files_and_folders_cfg["load"]["load_files"]["precomputed_filename"]
        # -- for the sensitivity analysis
        self.precomputed_filename_test = Path(self.followup_cfg["precomputed"]["path"]).joinpath(self.followup_cfg["precomputed"]["filename"])
        self.vec_store = None

        self.dataloader_config = dict(self.training_cfg)
        self.dataloader_config.update({"fields": self.model_fields_cfg["fields"]})

        self.training_config = self.training_cfg
        
        # -- cols
        self.feature_columns = self.model_fields_cfg["fields"]["feature_columns"]
        self.event_indicator_columns = self.model_fields_cfg["fields"]["event_indicator_columns"]
        self.followup_columns = self.model_fields_cfg["fields"]["followup_columns"]
        self.multi_elig_columns = self.model_fields_cfg["fields"]["multiyear_eligibility_columns"]
        self.shortcut_terms_flag_col = "shortcut_terms_flag"
        self.struct_dataset = None
        
    def _connect_with_emb_store(self): # OLD
        self.vec_store = LMDBEmbedStore(str(self.dataset_path.joinpath(self.precomputed_filename)))

    def _connect_with_emb_store(self):
        print(f"loaded emb vector: {self.precomputed_path.joinpath(self.precomputed_filename)}")
        self.vec_store = LMDBEmbedStore(str(self.precomputed_path.joinpath(self.precomputed_filename)))

    def get_embeddings(
        self, 
        ids: List[str],
    ):
        if self.vec_store is None:
            self._connect_with_emb_store()
        return [ self.vec_store.get(cur_id) for cur_id in ids ]

    def get_dataloader(
        self,
        ids: List[str],
        target_year: int,
        is_training: bool,
        max_len: Optional[int] = 100, # so far, not necessary
        batch_size: Optional[int] = 50_000
    ):
        # -- load structured data
        src = self.dataset_path.joinpath(self.struct_dataset_eligible_filename)
        struct_cols = self.feature_columns+self.event_indicator_columns+self.followup_columns+self.multi_elig_columns+[self.shortcut_terms_flag_col]
        
        self.struct_dataset = load_dataset(src, struct_cols, ids, batch_size=batch_size)
        # -- filter mammograms with past shortcut terms
        print(f"before filtering of shortcut terms: {self.struct_dataset.shape}")
        self.struct_dataset = filter_shortcut_terms(self.struct_dataset, self.shortcut_terms_flag_col)
        print(f"after filtering of shortcut terms: {self.struct_dataset.shape}")

        # -- transformation is here
        self.struct_dataset = standardize_dataset(self.struct_dataset)
        print("standardization was done here")
        #print(self.struct_dataset.head())
        
        # -- transform dataset multi-year eligibility into one single year eligibility
        print("before transform:", self.struct_dataset.shape)
        self.struct_dataset = transform_eligibility_to_singleyear(
            self.struct_dataset, 
            target_year=target_year, 
            followup_cols=self.followup_columns, 
            event_cols=self.event_indicator_columns,
            only_eligible=True
        )
        print('after transform:', self.struct_dataset.shape)
        print('positives:', self.struct_dataset[self.struct_dataset["event_indicator"]==1].shape[0])
        # -- get the formatted dataloader for pytorch training
        dataloader, imratio = get_dataloaders_singleyear(
           self.dataloader_config, self.struct_dataset, max_len=max_len, 
           is_training=is_training, device=self.device
        )
        self.struct_dataset = None
        gc.collect()
        return dataloader, imratio
    
# -------------------------------------------------------------------------------------------------- #
# ------------------------------------- DATASET SPLIT ---------------------------------------------- #
# -------------------------------------------------------------------------------------------------- #

class DatasetSplit(ConfigInterface):
    def __init__(self, config_dir: str, config_defaults: dict):
        super().__init__(config_dir, config_defaults)

        self.struct_dataset_eligible_filename = self.files_and_folders_cfg["load"]["load_files"]["final_data_with_eligibility_filename"]
        self.final_mamm_seq_filename = self.files_and_folders_cfg["load"]["load_files"]["seq_per_mammogram_filename"]

        # -- split configuration
        self.training_size = self.split_cfg['split']['training_size']
        self.test_size = self.split_cfg["split"]["test_size"]
        self.kfold = self.split_cfg["split"]["kfold"]
        self.split_seed = self.split_cfg["split"]["seed"]
        self.birads = self.split_cfg["split"]["birads"]

        # -- task
        self.person_to_event = None
        self.person_to_mammogram = None
        self.event_indicator_task = None
        # -- splits
        self.train_test_split_by_person = None
        self.train_test_split_by_mammogram = None
        self.cv_split_by_person = None
        self.cv_split_by_mammogram = None

        # -- cols
        self.id_columns = ["person_id", "mammogram_id"]
        self.birads_column = self.model_fields_cfg["fields"]["birads_column"]
        self.feature_columns = self.model_fields_cfg["fields"]["feature_columns"]
        self.event_indicator_columns = self.model_fields_cfg["fields"]["event_indicator_columns"]
        self.followup_columns = self.model_fields_cfg["fields"]["followup_columns"]
        self.multi_elig_columns = self.model_fields_cfg["fields"]["multiyear_eligibility_columns"]

        self.eligible_dataset = None
        self.outliers = None

    def _get_eligible_dataset(
        self,
        target_year: int,
    ) -> None:
        src = self.dataset_path.joinpath(self.struct_dataset_eligible_filename)
        columns = self.id_columns + [ self.birads_column ] + self.event_indicator_columns + self.event_indicator_columns \
        + self.followup_columns + self.multi_elig_columns 
        self.eligible_dataset = load_dataset(src, columns, batch_size=100_000)
        # -- filter BI-RADS
        self.eligible_dataset = self.eligible_dataset[self.eligible_dataset[self.birads_column].isin(self.birads)].drop(columns=[self.birads_column])
        # the 'eligible_dataset' will contain only the eligible ids for the parsed
        # target year and it will also contain the labels for the stratified splitting.
        self.eligible_dataset = transform_eligibility_to_singleyear(
            self.eligible_dataset, target_year=target_year,
            followup_cols=self.followup_columns,
            event_cols=self.event_indicator_columns,
            only_eligible=True
        )

    def _create_single_split_by_person(self):
        unique_persons = self.eligible_dataset.sort_values(by=["person_id", "event_indicator"], ascending=True).drop_duplicates(subset=["person_id", "event_indicator"], keep="last")
        person_id_train, person_id_test, event_train, event_test = train_test_split(
            unique_persons["person_id"].values,
            unique_persons["event_indicator"].values, 
            test_size=self.test_size, 
            random_state=self.split_seed,
            stratify=unique_persons["event_indicator"].values
        )
        self.train_test_split_by_person = {
            "train_ids": person_id_train, "test_ids": person_id_test,
            "train_events": event_train, "test_events": event_test
        }

    def _define_kfold_split_by_person(
        self, 
        n_splits: Optional[int] = 5,
        verbose: Optional[bool] = False
    ):
        # -- split first training and test sets
        self._create_single_split_by_person()
        # -- access person ids and events for the training set
        person_id_train = self.train_test_split_by_person['train_ids']
        event_train = self.train_test_split_by_person['train_events']
        # -- Define the 'n_splits' cross-validation splits
        self.cv_split_by_person = {}
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.split_seed)
        for i, (train_indices, val_indices) in enumerate(skf.split(person_id_train, event_train)):
            if verbose: print(f"Fold {i}:")
            self.cv_split_by_person.update({
                i: { "train": person_id_train[train_indices], "validation": person_id_train[val_indices] }
            })
            if verbose: 
                print(f"Total Pop.: training ({len(self.cv_split_by_person[i]['train']):,})")
                print(f"Total Pop.: validation ({len(self.cv_split_by_person[i]['validation']):,})")

        # -- access person ids and events for the test set
        person_id_test = self.train_test_split_by_person['test_ids']
        event_test = self.train_test_split_by_person['test_events']
        self.cv_split_by_person.update({
            'test': person_id_test
        })
        if verbose: print(f"Total Pop.: test ({len(self.cv_split_by_person['test']):,})")


    def _define_split_by_mammogram(
        self,
        n_splits: Optional[int] = 5,
        verbose: Optional[bool] = False
    ):
        # -- define person id -> mammogram ids dictionary
        person_to_mammograms = defaultdict(lambda: [])
        pid, mid = self.eligible_dataset["person_id"], self.eligible_dataset["mammogram_id"]
        [ person_to_mammograms[person_id].append(mammogram_id) for person_id, mammogram_id in zip(pid, mid) ]
        
        # -- create 'n_splits' person splits for cross-validation
        self._define_kfold_split_by_person(n_splits=n_splits, verbose=verbose)
        # -- convert the person splits to a mammogram splits
        # ---- training set
        self.cv_split_by_mammogram = {}
        for fold_ix, v in self.cv_split_by_person.items():
            if fold_ix=="test": continue
            if verbose: print(f"Fold {fold_ix}")
            mammogram_train_ids = [ person_to_mammograms[person_id] for person_id in v['train'] ]
            mammogram_val_ids = [ person_to_mammograms[person_id] for person_id in v['validation'] ]

            self.cv_split_by_mammogram.update({
                f"fold {fold_ix}": {
                    "train": [ mamm_id for sublist in mammogram_train_ids for mamm_id in sublist ],
                    "validation": [ mamm_id for sublist in mammogram_val_ids for mamm_id in sublist ]
                }
            })
            if verbose: 
                print(f"Total Mammograms: training ({len(self.cv_split_by_mammogram[f'fold {fold_ix}']['train']):,})")
                print(f"Total Mammograms: validation ({len(self.cv_split_by_mammogram[f'fold {fold_ix}']['validation']):,})")

        # ---- test set
        person_test_ids = self.cv_split_by_person['test']
        mammogram_test_ids = [ person_to_mammograms[person_id] for person_id in person_test_ids ]
        self.cv_split_by_mammogram.update({
            "test": [ mamm_id for sublist in mammogram_test_ids for mamm_id in sublist ]
        })
        if verbose: print(f"Total Mammograms: test ({len(self.cv_split_by_mammogram['test']):,})")

    def _remove_outliers(self, seq_percentile=99):
        '''
            Some mammograms have an amount of past mammograms that are too
            high compared to the other. Here we classify them as outliers,
            identify and remove them.
        '''
        past_sequence_size = {
            "mammogram_id": [], "past_sequence_size": []
        }
        columns = [ 
            "person_id", "mammogram_id", "mammogram_current_date",
            "mammogram_prior_codes"
        ]
        src = self.dataset_path.joinpath(self.final_mamm_seq_filename)
        for batch_df in tqdm(batching_parquet_file(src, columns=columns, batch_size=100_000)):
            mammogram_list = batch_df["mammogram_id"].tolist()
            past_seq_size_list = batch_df["mammogram_prior_codes"].apply(len).tolist()
            past_sequence_size["mammogram_id"].extend(mammogram_list)
            past_sequence_size["past_sequence_size"].extend(past_seq_size_list)
        
        past_sequence_size = pd.DataFrame(past_sequence_size)
        max_past_seq_size = past_sequence_size["past_sequence_size"].max()
        n_percentile = np.percentile(past_sequence_size["past_sequence_size"].values, seq_percentile)
        print(f"maximum sequence size found: {max_past_seq_size}\nmaximum sequence size after removal of outliers: {n_percentile}")
        self.outliers = past_sequence_size[past_sequence_size["past_sequence_size"]>n_percentile]["mammogram_id"].tolist()

        # -- update splits by mammogram
        for key, v in self.cv_split_by_mammogram.items():
            if key!="test":
                self.cv_split_by_mammogram[key] = {
                    "train": list(set(v['train']) - set(self.outliers)),
                    "validation": list(set(v['validation']) - set(self.outliers))
                } 
            else:
                self.cv_split_by_mammogram[key] = list(set(v) - set(self.outliers))

    def split(
        self,
        target_year: int,
        n_splits: Optional[int] = 5,
        seq_percentile: Optional[float] = 99.0,
        verbose: Optional[float] = False
    ):
        self._get_eligible_dataset(target_year=target_year)
        if verbose: print("[define] cross-validation folds ...")
        self._define_split_by_mammogram(n_splits=n_splits, verbose=verbose)
        if verbose: print("[find] outliers ...")
        self._remove_outliers(seq_percentile=seq_percentile)