'''

'''
import h5py
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from libauc.sampler import DualSampler
from hapcancer.model.dataload.datasets import CancerDatasetSingleYear, CancerDatasetMultiYear, CancerDatasetSingleYearTFIDF, CancerDatasetSingleYearTFIDFFlat, create_past_sequence
from hapcancer.shared.embed_store import H5KeyedEmbeddings
from hapcancer.model.dataload.dim_reducer import PCADimReducer
from typing import Optional, List



class CancerRiskDatasetMultiYear_v2(Dataset):
    def __init__(
        self,
        subset_df,
        extra_features,
        labels,                # (N, 5) array
        eligibility_mask,      # (N, 5) array
        hdf5_path,
        max_len=156,
        embed_dim=128,
        device='cpu'
    ):
        """
        Dataset that loads precomputed mammogram sequences from HDF5 files
        and returns multiyear labels and eligibility masks.

        Here we do not use the precomputed files, but instead we use a embedding store
        to build the sequences.
        """
        self.subset_df = subset_df
        self.extra_features = torch.tensor(extra_features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)               # (N, 5)
        self.eligibility_mask = torch.tensor(eligibility_mask, dtype=torch.float32)  # (N, 5)
        self.hdf5_path = hdf5_path
        self.max_len = max_len
        self.embed_dim = embed_dim
        self.device = device

    def __len__(self):
        return len(self.subset_df)

    def __getitem__(self, idx):
        mammogram_id = str(self.subset_df.mammogram_id.iat[idx])

        with h5py.File(self.hdf5_path, "r") as f:
            if mammogram_id in f:
                embeddings = torch.tensor(f[mammogram_id]["embeddings"][:], dtype=torch.float32)
                time_diffs = torch.tensor(f[mammogram_id]["time_diffs"][:], dtype=torch.float32)
            else:
                embeddings = torch.zeros((0, self.embed_dim), dtype=torch.float32)
                time_diffs = torch.zeros((0,), dtype=torch.float32)

        seq_len = embeddings.shape[0]
        pad_len = self.max_len - seq_len

        if pad_len > 0:
            pad_embeddings = torch.zeros((pad_len, self.embed_dim))
            pad_time_diffs = torch.zeros((pad_len,))
            embeddings = torch.cat([pad_embeddings, embeddings], dim=0)
            time_diffs = torch.cat([pad_time_diffs, time_diffs], dim=0)
        elif pad_len < 0:
            embeddings = embeddings[-self.max_len:, :]
            time_diffs = time_diffs[-self.max_len:]

        attention_mask = torch.cat([torch.zeros(pad_len), torch.ones(seq_len)]) if pad_len > 0 else torch.ones(self.max_len)

        return (
            idx,
            mammogram_id,
            embeddings,
            time_diffs,
            attention_mask,
            self.extra_features[idx],
            self.labels[idx],               # shape (5,)
            self.eligibility_mask[idx]      # shape (5,)
        )


class CancerRiskDataset(Dataset):
    def __init__(self, subset_df, extra_features, labels, hdf5_path, max_len=156, embed_dim=128, device='cpu'):
        """
            Dataset that loads precomputed mammogram sequences from HDF5 files.
        
            Args:
            -----
                subset_df (pd.DataFrame): DataFrame with the patients included in training.
                extra_features (np.ndarray): Extra structured data features (patients, feature_dim).
                labels (np.ndarray): Labels (patients, 1).
                hdf5_path (str): Path to precomputed HDF5 file.
                max_len (int): Maximum sequence length for padding.
        """
        self.subset_df = subset_df
        self.extra_features = torch.tensor(extra_features, dtype=torch.float32)#.to(device)
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(-1)#.to(device)
        self.hdf5_path = hdf5_path
        self.max_len = max_len
        self.embed_dim = embed_dim
        self.device = device

    def __len__(self):
        return len(self.subset_df)

    def __getitem__(self, idx):
        """
            Load patient’s mammogram sequence from HDF5 and apply padding.
        """
        #mammogram_id = str(self.subset_df.index[idx])  # Convert to string for HDF5 lookup
        mammogram_id = str(self.subset_df.mammogram_id.iat[idx])
        #print(mammogram_id)

        with h5py.File(self.hdf5_path, "r") as f:
            if mammogram_id in f:
                embeddings = torch.tensor(f[mammogram_id]["embeddings"][:], dtype=torch.float32)
                time_diffs = torch.tensor(f[mammogram_id]["time_diffs"][:], dtype=torch.float32)
            else:
                # -- this will generate an attention mask vector fully masked -> it will generate problems during training/validation 
                embeddings = torch.zeros((0, self.embed_dim), dtype=torch.float32)  # Empty if no mammograms
                time_diffs = torch.zeros((0,), dtype=torch.float32)

        # Apply padding dynamically
        seq_len = embeddings.shape[0]
        pad_len = self.max_len - seq_len

        if pad_len > 0:  # Padding required
            pad_embeddings = torch.zeros((pad_len, self.embed_dim))
            pad_time_diffs = torch.zeros((pad_len,))

            embeddings = torch.cat([pad_embeddings, embeddings], dim=0)
            time_diffs = torch.cat([pad_time_diffs, time_diffs], dim=0)

        elif pad_len < 0:  # Truncate if needed
            embeddings = embeddings[-self.max_len:, :]
            time_diffs = time_diffs[-self.max_len:]

        # Attention mask: 1 for real values, 0 for padding
        attention_mask = torch.cat([torch.zeros(pad_len), torch.ones(seq_len)])

        return (
            idx,
            mammogram_id,
            embeddings,#.to(self.device),
            time_diffs,#.to(self.device),
            attention_mask,#.to(self.device),
            self.extra_features[idx],
            self.labels[idx]
        )

class CancerRiskDatasetMultiYear(Dataset):
    def __init__(
        self,
        subset_df,
        extra_features,
        labels,                # (N, 5) array
        eligibility_mask,      # (N, 5) array
        hdf5_path,
        max_len=156,
        embed_dim=128,
        device='cpu'
    ):
        """
        Dataset that loads precomputed mammogram sequences from HDF5 files
        and returns multiyear labels and eligibility masks.
        """
        self.subset_df = subset_df
        self.extra_features = torch.tensor(extra_features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)               # (N, 5)
        self.eligibility_mask = torch.tensor(eligibility_mask, dtype=torch.float32)  # (N, 5)
        self.hdf5_path = hdf5_path
        self.max_len = max_len
        self.embed_dim = embed_dim
        self.device = device

    def __len__(self):
        return len(self.subset_df)

    def __getitem__(self, idx):
        mammogram_id = str(self.subset_df.mammogram_id.iat[idx])

        with h5py.File(self.hdf5_path, "r") as f:
            if mammogram_id in f:
                embeddings = torch.tensor(f[mammogram_id]["embeddings"][:], dtype=torch.float32)
                time_diffs = torch.tensor(f[mammogram_id]["time_diffs"][:], dtype=torch.float32)
            else:
                embeddings = torch.zeros((0, self.embed_dim), dtype=torch.float32)
                time_diffs = torch.zeros((0,), dtype=torch.float32)

        seq_len = embeddings.shape[0]
        pad_len = self.max_len - seq_len

        if pad_len > 0:
            pad_embeddings = torch.zeros((pad_len, self.embed_dim))
            pad_time_diffs = torch.zeros((pad_len,))
            embeddings = torch.cat([pad_embeddings, embeddings], dim=0)
            time_diffs = torch.cat([pad_time_diffs, time_diffs], dim=0)
        elif pad_len < 0:
            embeddings = embeddings[-self.max_len:, :]
            time_diffs = time_diffs[-self.max_len:]

        attention_mask = torch.cat([torch.zeros(pad_len), torch.ones(seq_len)]) if pad_len > 0 else torch.ones(self.max_len)

        return (
            idx,
            mammogram_id,
            embeddings,
            time_diffs,
            attention_mask,
            self.extra_features[idx],
            self.labels[idx],               # shape (5,)
            self.eligibility_mask[idx]      # shape (5,)
        )


import h5py
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from libauc.sampler import DualSampler
from hapcancer.model.dataload.datasets import CancerDatasetSingleYear, CancerDatasetMultiYear, CancerDatasetSingleYearTFIDF, CancerDatasetSingleYearTFIDF_light
from hapcancer.shared.embed_store import H5KeyedEmbeddings
from typing import Optional, List

class CancerRiskDatasetFull:
    def __init__(
        self,
        mamm_seq_df,
        features_df,
        labels,
        eligibility_mask,
        embedding_store_path,
        max_len=128, # maximum amount of past mammograms to include 'max_seq_len'
        embedding_dim=1024,
        time_limit=None, # -- number of months to set the date limit in the past, e. g. 30 months -> consider only mammograms between [cur_date - 30mo, cur_date].
        device='cpu'
    ):
        #self.mamm_seq_df = mamm_seq_df,
        self.mamm_seq_codes = mamm_seq_df['mammogram_id'].tolist()
        self.mamm_seq_prior_codes = mamm_seq_df['mammogram_prior_codes'].tolist()
        self.mamm_seq_prior_dates = mamm_seq_df['mammogram_prior_dates'].tolist()
        self.features = torch.tensor(features_df, dtype=torch.float16)
        self.labels = torch.tensor(labels, dtype=torch.int64) # -- (N, 5)
        self.eligibility_mask = torch.tensor(eligibility_mask, dtype=torch.int64) # -- (N, 5)
        self.embedding_store_path = embedding_store_path
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.device = device
        self.time_limit = time_limit
        #self.embedding_store = H5KeyedEmbeddings(self.embedding_store_path)

    def __len__(self):
        return len(self.mamm_seq_codes)

    def __getitem__(self, idx):
        mammogram_id = self.mamm_seq_codes[idx]
        mammogram_prior_history = self.mamm_seq_prior_codes[idx]
        mammogram_prior_dates = self.mamm_seq_prior_dates[idx]
        mammogram_current_date = mammogram_prior_dates[-1]
        embedding_store = H5KeyedEmbeddings(self.embedding_store_path) # -- open only when getting item (multiprocessing problem might appear if opened at __init__)

        # -- create array of embeddings for past mammogram history using the created embedding store and its efficient wrapper.
        old_emb_n = None
        if mammogram_id in embedding_store:
            # -- sanity check: does 'get_many' return the embedding within the same order, right?
            embeddings, missing_keys = embedding_store.get_many(mammogram_prior_history, ignore_missing=True)
            old_emb_n = len(embeddings)
            
            # -- remove any key for which we didn't find an embedding for it.
            mammogram_prior_dates_valid = [ el for el in mammogram_prior_dates ]
            if len(missing_keys)>0:
                mammogram_prior_dates_valid = [ mamm_date for mamm_date, mamm_key in zip(mammogram_prior_dates, mammogram_prior_history) if mamm_key not in missing_keys ]
            
            assert len(embeddings) == len(mammogram_prior_dates_valid), "embeddings and dates arr not matching in size"

            earliest_date_possible = None
            if self.time_limit is not None:
                earliest_date_possible = pd.Timestamp(mammogram_current_date) - pd.DateOffset(months=self.time_limit)
                embeddings_tensor = torch.tensor(np.vstack([ cur_emb for cur_emb, cur_date in zip(embeddings, mammogram_prior_dates_valid) if cur_date>=earliest_date_possible  ]), dtype=torch.float16)
                mammogram_prior_dates_valid = [ cur_date for cur_emb, cur_date in zip(embeddings, mammogram_prior_dates_valid) if cur_date>=earliest_date_possible  ]
            
            list_of_time_difference = [ np.timedelta64(mammogram_current_date - current_date).astype(f'timedelta64[D]')/np.timedelta64(1, 'D') for current_date in mammogram_prior_dates_valid ]
            time_diffs = torch.tensor(list_of_time_difference, dtype=torch.int64)
            #print("example:")
            #print(embeddings_tensor.shape[0], len(missing_keys))
            #print(time_diffs.shape[0], len(mammogram_prior_dates_valid), mammogram_current_date, earliest_date_possible)
            #print(mammogram_prior_dates)
            #print(mammogram_prior_dates_valid)
        else:
            print("no ID!")
        
        assert embeddings_tensor.shape[1] == self.embedding_dim, "real embedding dimension not matching parsed embedding dimension"
        
        # -- create padding
        seq_mamm_len = embeddings_tensor.shape[0]
        padding_len = self.max_len - seq_mamm_len
        if padding_len > 0:
            pad_embeddings = torch.zeros((padding_len, self.embedding_dim))
            pad_time_diffs = torch.zeros((padding_len,))
            embeddings_tensor = torch.cat([pad_embeddings, embeddings_tensor], dim=0)
            time_diffs = torch.cat([pad_time_diffs, time_diffs], dim=0)
        elif padding_len < 0:
            embeddings_tensor = embeddings_tensor[-self.max_len:, :] # is that the behavior we want? prune 
            time_diffs = time_diffs[-self.max_len:]

        attention_mask = torch.cat([torch.zeros(padding_len), torch.ones(seq_mamm_len)]) if padding_len > 0 else torch.ones(self.max_len)

        return (
            idx,
            mammogram_id,
            embeddings_tensor,
            time_diffs,
            attention_mask,
            self.features[idx],
            self.labels[idx],               # shape (5,)
            self.eligibility_mask[idx]      # shape (5,)
        )

def prepare_data_for_model_v2(sequence_info_df, extra_features_df, valid_ids=None, oversampling=False, 
                              positive_years=[1,2], positive_ratio=8, prediction_year=2, verbose=False):
    '''
        ...

        Args:
        -----
            sequence_info_df:
                .
            extra_features_df:
                .
            valid_ids:
                .
            oversampling:
                .
            positive_years:
                .
            positive_ratio:
                .
            verbose:
                .

        Returns:
        --------
            sequence_info_df:
                .
            sample_extra_features:
                .
    '''
    if valid_ids is not None:
        sequence_info_df = sequence_info_df[sequence_info_df["mammogram_id"].isin(valid_ids)].copy() 

    mammogram_codes_sample = sequence_info_df["mammogram_id"].tolist()
    sample_extra_features_df = extra_features_df[extra_features_df["mammogram_id"].isin(mammogram_codes_sample)].copy()

    # -- predict cancer risk within X years after mammogram (original format is in survival time)
    # -- define the X years after mammogram
    #positive_condition = (sample_extra_features_df["survival_time_years"].isin(positive_years)) & (sample_extra_features_df["event_indicator"]==1)
    positive_condition = sample_extra_features_df[f"event_indicator_{prediction_year:.0f}yr"]==1

    sample_extra_features_df_1 = sample_extra_features_df[positive_condition].copy()
    sample_extra_features_df_0 = sample_extra_features_df[~positive_condition].copy()
    sample_extra_features_df_1["label"] = [ 1 for n in range(sample_extra_features_df_1.shape[0]) ]
    sample_extra_features_df_0["label"] = [ 0 for n in range(sample_extra_features_df_0.shape[0]) ]
    if verbose:
        print(sample_extra_features_df_0.columns)
        print("dimensions of training before enrichment: ", sample_extra_features_df_0.shape)
        print("positive to negative ratio before enrichment: ", sample_extra_features_df_1.shape[0]/sample_extra_features_df_0.shape[0])

    # -- 1:positive_ratio positive to negative ratio for oversampling
    if oversampling:
        goal_n = int(sample_extra_features_df_0.shape[0]/positive_ratio)
        if verbose: print(f"expected number of positives after enrichment: {goal_n:,}")
        sample_extra_features_df_1 = sample_extra_features_df_1.sample(n=goal_n, replace=True)
        if verbose: print("positive to negative ratio after enrichment: ", sample_extra_features_df_1.shape[0]/sample_extra_features_df_0.shape[0])
    sample_extra_features = pd.concat([sample_extra_features_df_1, sample_extra_features_df_0])
    # -- maybe not necessary
    for yr in [1,2,3,4,5]:
        if f"survival_time_{prediction_year:.0f}yr" in sample_extra_features.columns:
            sample_extra_features = sample_extra_features.drop(columns=[f"survival_time_{prediction_year:.0f}yr"])
        if f"event_indicator_{prediction_year:.0f}yr" in sample_extra_features.columns:
            sample_extra_features = sample_extra_features.drop(columns=[f"event_indicator_{prediction_year:.0f}yr"]) 

    sequence_info_df = sequence_info_df.set_index("mammogram_id")
    sequence_info_df = sequence_info_df.loc[sample_extra_features["mammogram_id"]].reset_index()
    #sample_extra_features = sample_extra_features.set_index("mammogram_id")
    if verbose:
        if 'mammogram_id' in sample_extra_features.columns:
            print("are identifiers aligned?", np.sum(sequence_info_df["mammogram_id"].values == sample_extra_features["mammogram_id"].values)==sample_extra_features.shape[0])
        else:
            print("are identifiers aligned?", np.sum(np.array(sequence_info_df.index) == np.array(sample_extra_features.index))==sample_extra_features.shape[0])
        print("final dimensions: ", sequence_info_df.shape, sample_extra_features.shape)

    # -- create a dummy unique index field
    sequence_info_df.index = [ n for n in range(sequence_info_df.shape[0]) ]
    sample_extra_features.index = [ n for n in range(sequence_info_df.shape[0]) ]
    labels = sample_extra_features['label']
    sample_extra_features = sample_extra_features.drop(columns=['label'])#.values.astype(float)
    return sequence_info_df, sample_extra_features, labels

def prepare_data_for_model_singleyear(
    mamm_seq_df: pd.DataFrame,
    features_df: pd.DataFrame,
    valid_ids: Optional[List[str]] = None,
    sampling: Optional[bool] = False,
    sampling_strategy: Optional[str] = "oversampling",
    positive_ratio: Optional[int] = 8,
    verbose: Optional[bool] = True
):
    if valid_ids is not None:
        mamm_seq_df = mamm_seq_df[mamm_seq_df["mammogram_id"].isin(valid_ids)].copy() 

    mammogram_codes_sample = mamm_seq_df["mammogram_id"].tolist()
    sample_features_df = features_df[features_df["mammogram_id"].isin(mammogram_codes_sample)].copy()

    positive_condition = sample_features_df[f"event_indicator"]==1

    sample_features_df_1 = sample_features_df[positive_condition].copy()
    sample_features_df_0 = sample_features_df[~positive_condition].copy()
    sample_features_df_1["label"] = [ 1 for n in range(sample_features_df_1.shape[0]) ]
    sample_features_df_0["label"] = [ 0 for n in range(sample_features_df_0.shape[0]) ]

    if verbose:
        print(sample_features_df_0.columns)
        print("dimensions of training before enrichment: ", sample_features_df_0.shape)
        print("positive to negative ratio before enrichment: ", sample_features_df_1.shape[0]/sample_features_df_0.shape[0])

    # -- 1:positive_ratio positive to negative ratio for oversampling/undersampling
    if sampling and sampling_strategy=="oversampling":
        goal_n = int(sample_features_df_0.shape[0]/positive_ratio)
        if verbose: print(f"expected number of positives after enrichment: {goal_n:,}")
        sample_features_df_1 = sample_features_df_1.sample(n=goal_n, replace=True)
        if verbose: print("positive to negative ratio after enrichment: ", sample_features_df_1.shape[0]/sample_features_df_0.shape[0])
    elif sampling and sampling_strategy=="undersampling":
        goal_n = int(sample_features_df_1.shape[0]*positive_ratio)
        if verbose: print(f"expected number of negatives after undersampling: {goal_n:,}")
        sample_features_df_0 = sample_features_df_0.sample(n=goal_n, replace=False)
        if verbose: print("positive to negative ratio after enrichment: ", sample_features_df_1.shape[0]/sample_features_df_0.shape[0])
    else:
        pass
    sample_features = pd.concat([sample_features_df_1, sample_features_df_0])

    mamm_seq_df = mamm_seq_df.set_index("mammogram_id")
    mamm_seq_df = mamm_seq_df.loc[sample_features["mammogram_id"]].reset_index()
    #sample_extra_features = sample_extra_features.set_index("mammogram_id")
    if verbose:
        if 'mammogram_id' in sample_features.columns:
            print("are identifiers aligned?", np.sum(mamm_seq_df["mammogram_id"].values == sample_features["mammogram_id"].values)==sample_features.shape[0])
        else:
            print("are identifiers aligned?", np.sum(np.array(mamm_seq_df.index) == np.array(sample_features.index))==sample_features.shape[0])
        print("final dimensions: ", mamm_seq_df.shape, sample_features.shape)

    # -- create a dummy unique index field
    mamm_seq_df.index = [ n for n in range(mamm_seq_df.shape[0]) ]
    sample_features.index = [ n for n in range(mamm_seq_df.shape[0]) ]
    labels = sample_features['label']
    eligibility_mask = sample_features["eligibility"]
    sample_features = sample_features.drop(columns=['label', "event_indicator", "eligibility"])#.values.astype(float)
    return mamm_seq_df, sample_features, labels, eligibility_mask


def prepare_data_for_model_multiyear(sequence_info_df, extra_features_df, valid_ids=None, oversampling=False, 
                                     positive_ratio=8, verbose=False):
    '''
        ...

        Args:
        -----
            sequence_info_df:
                .
            extra_features_df:
                .
            valid_ids:
                .
            oversampling:
                .
            positive_years:
                .
            positive_ratio:
                .
            verbose:
                .

        Returns:
        --------
            sequence_info_df:
                .
            sample_extra_features:
                .
    '''
    if valid_ids is not None:
        sequence_info_df = sequence_info_df[sequence_info_df["mammogram_id"].isin(valid_ids)].copy() 

    mammogram_codes_sample = sequence_info_df["mammogram_id"].tolist()
    sample_extra_features_df = extra_features_df[extra_features_df["mammogram_id"].isin(mammogram_codes_sample)].copy()

    # -- predict cancer risk within X years after mammogram (original format is in survival time)
    # -- define the X years after mammogram
    #eligibility_cols = [f'eligibility_{i}yr' for i in range(1, 6)]
    eligibility_cols = [f'eligibility_{i}yr_{i+1}yr' for i in range(0, 5)]
    label_cols = [f'event_indicator_{i}yr' for i in range(1, 6)]

    # -- element-wise multiplication (eligibility & label)
    eligible_positive = sample_extra_features_df[eligibility_cols].values * sample_extra_features_df[label_cols].values
    positive_condition = (eligible_positive.sum(axis=1) > 0)

    sample_extra_features_df_1 = sample_extra_features_df[positive_condition].copy()
    sample_extra_features_df_0 = sample_extra_features_df[~positive_condition].copy()
    if verbose:
        print(sample_extra_features_df_0.columns)
        print("dimensions of dataset before enrichment: ", sample_extra_features_df_0.shape)
        print("positive to negative ratio before enrichment: ", sample_extra_features_df_1.shape[0]/sample_extra_features_df_0.shape[0])

    # -- 1:positive_ratio positive to negative ratio for oversampling
    if oversampling:
        goal_n = int(sample_extra_features_df_0.shape[0]/positive_ratio)
        if verbose: print(f"expected number of positives after enrichment: {goal_n:,}")
        sample_extra_features_df_1 = sample_extra_features_df_1.sample(n=goal_n, replace=True)
        if verbose: print("positive to negative ratio after enrichment: ", sample_extra_features_df_1.shape[0]/sample_extra_features_df_0.shape[0])
    sample_extra_features = pd.concat([sample_extra_features_df_1, sample_extra_features_df_0])
    
    for yr in [1,2,3,4,5]:
        if f"survival_time_{yr:.0f}yr" in sample_extra_features.columns:
            sample_extra_features = sample_extra_features.drop(columns=[f"survival_time_{yr:.0f}yr"])

    sequence_info_df = sequence_info_df.set_index("mammogram_id")
    sequence_info_df = sequence_info_df.loc[sample_extra_features["mammogram_id"]].reset_index()
    #sample_extra_features = sample_extra_features.set_index("mammogram_id")
    if verbose:
        if 'mammogram_id' in sample_extra_features.columns:
            print("are identifiers aligned?", np.sum(sequence_info_df["mammogram_id"].values == sample_extra_features["mammogram_id"].values)==sample_extra_features.shape[0])
        else:
            print("are identifiers aligned?", np.sum(np.array(sequence_info_df.index) == np.array(sample_extra_features.index))==sample_extra_features.shape[0])
        print("final dimensions: ", sequence_info_df.shape, sample_extra_features.shape)

    # -- create a dummy unique index field
    sequence_info_df.index = [ n for n in range(sequence_info_df.shape[0]) ]
    sample_extra_features.index = [ n for n in range(sequence_info_df.shape[0]) ]
    labels_multiyear = sample_extra_features[label_cols].copy()
    eligibility_multiyear_mask = sample_extra_features[eligibility_cols].copy()
    sample_extra_features = sample_extra_features.drop(columns=label_cols+eligibility_cols)
    return sequence_info_df, sample_extra_features, labels_multiyear, eligibility_multiyear_mask

def get_dataloaders(config, structured_input, device, only_anamnesis=False, verbose=False):
    '''
        ...

        Args:
        -----
            config:
                dictionary. Obtained from a configuration file.
            structured_input:
                dictionary. Output from 'load_input' function. It stores the load datasets
                for training, validation and test.
    '''
    max_len = int(structured_input['max_len'])
    
    # -- data filenames and params
    verbose = config['misc']['verbose']
    batch_size = config['data']['batch_size']
    num_workers = config['data']['num_workers']
    precomputed_path = Path(config['data']['precomputed_path'])
    precomputed_file_extension = config['data']['precomputed_file_extension']
    precomputed_train_filename = config['data']['precomputed_train_filename']
    precomputed_val_filename = config['data']['precomputed_val_filename']
    precomputed_test_filename = config['data']['precomputed_test_filename']
    
    # -- if 'only_anamnesis' true -> select only the mammograms with anamnesis information.
    have_anamnesis_ids = None
    if only_anamnesis:
        data_path = Path(config['data']['path'])
        have_anamnesis_ids = pd.read_parquet(data_path.joinpath(config['data']['have_anamnesis_ids_filename']))['mammogram_id'].tolist()

    # ----------------------------- TRAINING SET -----------------------------
    # ------------------------------------------------------------------------
    negative_to_positive_ratio = config['data']['negative_to_positive_ratio']
    sequence_info_train_df, extra_features_train_df = structured_input['sequence']['train'], structured_input['extra_features']['train']
    sequence_info_train_df, extra_features_train_df, train_labels = prepare_data_for_model_v2(sequence_info_train_df, extra_features_train_df, valid_ids=have_anamnesis_ids,
                                                                    oversampling=True, positive_years=[1,2], positive_ratio=negative_to_positive_ratio, verbose=verbose, prediction_year=2)
    precomputed_train_path = precomputed_path.joinpath(precomputed_train_filename+precomputed_file_extension)
    
    # ---- filter patients with missing mammogram sequences
    train_index, train_mammogram_id = list(sequence_info_train_df.index), sequence_info_train_df.mammogram_id.tolist()
    with h5py.File(precomputed_train_path, "r") as f:
        valid_ids = [pid[0] for pid in zip(train_index, train_mammogram_id) if str(pid[1]) in f]
    if verbose:
        print('train before:', sequence_info_train_df.shape[0])
        print('train after: ', len(valid_ids))
    
    # -- do not drop 'mammogram_id' from sequence dataframe -> it will be used when retrieving mammogram embeddings
    sequence_info_train_df = sequence_info_train_df.loc[valid_ids]
    extra_features_train_df = extra_features_train_df.loc[valid_ids].drop(columns=['mammogram_id']).values.astype(float)
    train_labels = train_labels.loc[valid_ids].values
    imratio = train_labels.sum() / len(train_labels)
    
    # ---- training dataset and dataloader
    train_dataset = CancerRiskDataset(sequence_info_train_df, extra_features_train_df, train_labels, precomputed_train_path, max_len=max_len, device=device)
    train_loader_sampler = DualSampler(train_dataset, labels=train_labels, batch_size=batch_size, sampling_rate=0.3)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_loader_sampler, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    # ------------------------------------------------------------------------

    # ----------------------------- VALIDATION SET -----------------------------
    # --------------------------------------------------------------------------
    sequence_info_val_df, extra_features_val_df = structured_input['sequence']['validation'], structured_input['extra_features']['validation']
    sequence_info_val_df, extra_features_val_df, val_labels = prepare_data_for_model_v2(sequence_info_val_df, extra_features_val_df, valid_ids=have_anamnesis_ids,
                                                              oversampling=False, positive_years=[1,2], verbose=verbose, prediction_year=2)
    precomputed_val_path = precomputed_path.joinpath(precomputed_val_filename+precomputed_file_extension)

    # ---- filter patients with missing mammogram sequences
    val_index, val_mammogram_id = list(sequence_info_val_df.index), sequence_info_val_df.mammogram_id.tolist()
    with h5py.File(precomputed_val_path, "r") as f:
        valid_ids = [pid[0] for pid in zip(val_index, val_mammogram_id) if str(pid[1]) in f]
    if verbose:
        print('val before:', sequence_info_val_df.shape[0])
        print('val after: ', len(valid_ids))
    
    sequence_info_val_df = sequence_info_val_df.loc[valid_ids]
    extra_features_val_df = extra_features_val_df.loc[valid_ids].drop(columns=['mammogram_id']).values.astype(float)
    val_labels = val_labels.loc[valid_ids].values
    
    # ---- validation dataset and dataloader
    val_dataset = CancerRiskDataset(sequence_info_val_df, extra_features_val_df, val_labels, precomputed_val_path, max_len=max_len, device=device)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    # --------------------------------------------------------------------------

    # ----------------------------- TEST SET -----------------------------
    # --------------------------------------------------------------------
    sequence_info_test_df, extra_features_test_df = structured_input['sequence']['test'], structured_input['extra_features']['test']
    sequence_info_test_df, extra_features_test_df, test_labels = prepare_data_for_model_v2(sequence_info_test_df, extra_features_test_df, valid_ids=have_anamnesis_ids,
                                                                 oversampling=False, positive_years=[1,2], verbose=verbose, prediction_year=2)
    precomputed_test_path = precomputed_path.joinpath(precomputed_test_filename+precomputed_file_extension)

    # ---- filter patients with missing mammogram sequences
    test_index, test_mammogram_id = list(sequence_info_test_df.index), sequence_info_test_df.mammogram_id.tolist()
    with h5py.File(precomputed_test_path, "r") as f:
        valid_ids = [pid[0] for pid in zip(test_index, test_mammogram_id) if str(pid[1]) in f]
    if verbose:
        print('test before:', sequence_info_test_df.shape[0])
        print('test after: ', len(valid_ids))
    sequence_info_test_df = sequence_info_test_df.loc[valid_ids]
    extra_features_test_df = extra_features_test_df.loc[valid_ids].drop(columns=['mammogram_id']).values.astype(float)
    test_labels = test_labels.loc[valid_ids].values
    
    # ---- test dataset and dataloader
    test_dataset = CancerRiskDataset(sequence_info_test_df, extra_features_test_df, test_labels, precomputed_test_path, max_len=max_len, device=device)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    # --------------------------------------------------------------------
    
    return train_loader, val_loader, test_loader, imratio

def get_dataloaders_singleyear(
    config: dict,
    structured_input: dict,
    with_transformer: Optional[bool] = False,
    device: Optional[str] = 'cpu',
    verbose: Optional[bool] = False
):
    max_len = int(structured_input['max_len'])+1
    
    # -- data filenames and params
    verbose = config['misc']['verbose']
    batch_size = config['data']['batch_size']
    num_workers = config['data']['num_workers']
    time_limit_for_past_mammograms = config['data']['time_limit_for_past_mammograms']
    embedding_store_path = Path(config['data']['embedding_store_path']).joinpath(config['data']['embedding_store_name'])

    cols_to_remove = config['data']['followup_columns']
    cols_to_remove += config['data']['event_indicator_columns']
    cols_to_remove += config['data']['multiyear_eligibility_columns']

    sampling_strategy = config['data']['sampling_strategy'] 
    
    # -- if 'only_anamnesis' true -> select only the mammograms with anamnesis information.
    have_anamnesis_ids = None
    
    # -- select custom dataset
    selected_dataset = CancerDatasetSingleYear
    if not with_transformer:
        selected_dataset = CancerDatasetSingleYearTFIDF_light

    # ----------------------------- TRAINING SET -----------------------------
    # ------------------------------------------------------------------------
    negative_to_positive_ratio = config['data']['negative_to_positive_ratio']
    sequence_info_train_df, extra_features_train_df = structured_input['sequence']['train'], structured_input['extra_features']['train']
    sequence_info_train_df, extra_features_train_df, train_labels, train_eligibility_mask = prepare_data_for_model_singleyear(sequence_info_train_df, extra_features_train_df, 
                                                                                            valid_ids=have_anamnesis_ids, sampling=True, sampling_strategy=sampling_strategy, 
                                                                                            positive_ratio=negative_to_positive_ratio, verbose=verbose)

    train_labels = train_labels.values
    train_eligibility_mask = train_eligibility_mask.values
    extra_features_train_df = extra_features_train_df.drop(columns=['mammogram_id']+cols_to_remove).values.astype(float)
    
    positive_n = train_labels.sum()
    print("labels:", train_labels[:4])
    print("eligibility mask:", train_eligibility_mask[:4])
    print(extra_features_train_df.shape, train_labels.shape, train_eligibility_mask.shape)
    print("number of positives:", positive_n)
    imratio = positive_n / train_labels.shape[0] # -- how?
    print(imratio)
    
    
    # ---- training dataset and dataloader
    #train_dataset = selected_dataset(sequence_info_train_df, extra_features_train_df, train_labels, train_eligibility_mask, precomputed_train_path, max_len=max_len, device=device)
    train_dataset = selected_dataset(sequence_info_train_df, extra_features_train_df, train_labels, train_eligibility_mask, embedding_store_path, 
                                    max_len=max_len, device=device, time_limit=time_limit_for_past_mammograms)
    train_loader_sampler = DualSampler(train_dataset, labels=train_labels, batch_size=batch_size, sampling_rate=0.5)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_loader_sampler, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    # ------------------------------------------------------------------------

    # ----------------------------- VALIDATION SET -----------------------------
    # --------------------------------------------------------------------------
    sequence_info_val_df, extra_features_val_df = structured_input['sequence']['validation'], structured_input['extra_features']['validation']
    sequence_info_val_df, extra_features_val_df, val_labels, val_eligibility_mask = prepare_data_for_model_singleyear(sequence_info_val_df, extra_features_val_df, 
                                                                                    valid_ids=have_anamnesis_ids, sampling=False, sampling_strategy=sampling_strategy, 
                                                                                    positive_ratio=negative_to_positive_ratio)

    val_labels = val_labels.values
    val_eligibility_mask = val_eligibility_mask.values
    extra_features_val_df = extra_features_val_df.drop(columns=['mammogram_id']+cols_to_remove).values.astype(float)
    
    positive_n = val_labels.sum()
    print("labels:", val_labels[:4])
    print("eligibility mask:", val_eligibility_mask[:4])
    print("positive at least in one year:", positive_n)
    
    # ---- validation dataset and dataloader
    val_dataset = selected_dataset(sequence_info_val_df, extra_features_val_df, val_labels, val_eligibility_mask, embedding_store_path, 
                                  max_len=max_len, device=device, time_limit=time_limit_for_past_mammograms)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    # --------------------------------------------------------------------------

    # ----------------------------- TEST SET -----------------------------
    # --------------------------------------------------------------------
    sequence_info_test_df, extra_features_test_df = structured_input['sequence']['test'], structured_input['extra_features']['test']
    if sequence_info_test_df is not None and extra_features_test_df is not None:
        sequence_info_test_df, extra_features_test_df, test_labels, test_eligibility_mask = prepare_data_for_model_singleyear(sequence_info_test_df, extra_features_test_df, 
                                                                                            valid_ids=have_anamnesis_ids, sampling=False, sampling_strategy=sampling_strategy, 
                                                                                            positive_ratio=negative_to_positive_ratio)

        test_labels = test_labels.values
        test_eligibility_mask = test_eligibility_mask.values
        extra_features_test_df = extra_features_test_df.drop(columns=['mammogram_id']+cols_to_remove).values.astype(float)
    
        positive_n = test_labels.sum()
        print("labels:", test_labels[:4])
        print("eligibility mask:", test_eligibility_mask[:4])
        print("positive at least in one year:", positive_n)
    
        # ---- test dataset and dataloader
        test_dataset = selected_dataset(sequence_info_test_df, extra_features_test_df, test_labels, test_eligibility_mask, embedding_store_path, 
                                       max_len=max_len, device=device, time_limit=time_limit_for_past_mammograms)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        # --------------------------------------------------------------------
    else:
        test_loader = None
    
    return train_loader, val_loader, test_loader, imratio



def get_dataloaders_multiyear(config, structured_input, device, only_anamnesis=False, verbose=False):
    '''
        ...

        Args:
        -----
            config:
                dictionary. Obtained from a configuration file.
            structured_input:
                dictionary. Output from 'load_input' function. It stores the load datasets
                for training, validation and test.
    '''
    max_len = int(structured_input['max_len'])
    
    # -- data filenames and params
    verbose = config['misc']['verbose']
    batch_size = config['data']['batch_size']
    num_workers = config['data']['num_workers']
    precomputed_path = Path(config['data']['precomputed_path'])
    precomputed_file_extension = config['data']['precomputed_file_extension']
    precomputed_train_filename = config['data']['precomputed_train_filename']
    precomputed_val_filename = config['data']['precomputed_val_filename']
    precomputed_test_filename = config['data']['precomputed_test_filename']
    
    # -- if 'only_anamnesis' true -> select only the mammograms with anamnesis information.
    have_anamnesis_ids = None
    if only_anamnesis:
        data_path = Path(config['data']['path'])
        have_anamnesis_ids = pd.read_parquet(data_path.joinpath(config['data']['have_anamnesis_ids_filename']))['mammogram_id'].tolist()

    # -- define custom dataset
    SelectedDataSet = CancerRiskDatasetFull

    # ----------------------------- TRAINING SET -----------------------------
    # ------------------------------------------------------------------------
    negative_to_positive_ratio = config['data']['negative_to_positive_ratio']
    sequence_info_train_df, extra_features_train_df = structured_input['sequence']['train'], structured_input['extra_features']['train']
    sequence_info_train_df, extra_features_train_df, train_labels, train_eligibility_mask = prepare_data_for_model_multiyear(sequence_info_train_df, extra_features_train_df, 
                                                                                            valid_ids=have_anamnesis_ids, oversampling=True, positive_ratio=negative_to_positive_ratio, 
                                                                                            verbose=verbose)
    precomputed_train_path = precomputed_path.joinpath(precomputed_train_filename+precomputed_file_extension)
    
    # ---- filter patients with missing mammogram sequences
    train_index, train_mammogram_id = list(sequence_info_train_df.index), sequence_info_train_df.mammogram_id.tolist()
    with h5py.File(precomputed_train_path, "r") as f:
        valid_ids = [pid[0] for pid in zip(train_index, train_mammogram_id) if str(pid[1]) in f]
    if verbose:
        print('train before:', sequence_info_train_df.shape[0])
        print('train after: ', len(valid_ids))
    
    # -- do not drop 'mammogram_id' from sequence dataframe -> it will be used when retrieving mammogram embeddings
    sequence_info_train_df = sequence_info_train_df.loc[valid_ids]
    extra_features_train_df = extra_features_train_df.loc[valid_ids].drop(columns=['mammogram_id']).values.astype(float)
    train_labels = train_labels.loc[valid_ids].values
    train_eligibility_mask = train_eligibility_mask.loc[valid_ids].values

    # -- element-wise multiplication (eligibility & label)
    
    #eligibility_cols = [f'eligibility_{i}yr' for i in range(1, 6)]
    eligibility_cols = [f'eligibility_{i}_{i+1}yr' for i in range(0, 5)]
    label_cols = [f'event_indicator_{i}yr' for i in range(1, 6)]
    eligible_positive = train_labels * train_eligibility_mask
    positive_condition = (eligible_positive.sum(axis=1) > 0).astype(int)
    positive_n = positive_condition.sum()
    print("labels:", train_labels[:4])
    print("eligibility mask:", train_eligibility_mask[:4])
    print("eligible_positive:", eligible_positive[:4])
    print("positive at least in one year:", positive_n, positive_condition[:4])
    imratio = positive_n / train_labels.shape[0] # -- how?
    #print(imratio)
    
    # ---- training dataset and dataloader
    #train_dataset = SelectedDataset(sequence_info_train_df, extra_features_train_df, train_labels, train_eligibility_mask, precomputed_train_path, max_len=max_len, device=device)
    train_dataset = SelectedDataset(sequence_info_train_df, extra_features_train_df, train_labels, train_eligibility_mask, precomputed_train_path, max_len=max_len, device=device)
    train_loader_sampler = DualSampler(train_dataset, labels=positive_condition, batch_size=batch_size, sampling_rate=0.3)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_loader_sampler, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    # ------------------------------------------------------------------------

    # ----------------------------- VALIDATION SET -----------------------------
    # --------------------------------------------------------------------------
    sequence_info_val_df, extra_features_val_df = structured_input['sequence']['validation'], structured_input['extra_features']['validation']
    sequence_info_val_df, extra_features_val_df, val_labels, val_eligibility_mask = prepare_data_for_model_multiyear(sequence_info_val_df, extra_features_val_df, 
                                                                                    valid_ids=have_anamnesis_ids, oversampling=False, verbose=verbose)
    precomputed_val_path = precomputed_path.joinpath(precomputed_val_filename+precomputed_file_extension)

    # ---- filter patients with missing mammogram sequences
    val_index, val_mammogram_id = list(sequence_info_val_df.index), sequence_info_val_df.mammogram_id.tolist()
    with h5py.File(precomputed_val_path, "r") as f:
        valid_ids = [pid[0] for pid in zip(val_index, val_mammogram_id) if str(pid[1]) in f]
    if verbose:
        print('val before:', sequence_info_val_df.shape[0])
        print('val after: ', len(valid_ids))
    
    sequence_info_val_df = sequence_info_val_df.loc[valid_ids]
    extra_features_val_df = extra_features_val_df.loc[valid_ids].drop(columns=['mammogram_id']).values.astype(float)
    val_labels = val_labels.loc[valid_ids].values
    val_eligibility_mask = val_eligibility_mask.loc[valid_ids].values

    eligible_positive = val_labels * val_eligibility_mask
    positive_condition = (eligible_positive.sum(axis=1) > 0).astype(int)
    positive_n = positive_condition.sum()
    print("labels:", val_labels[:4])
    print("eligibility mask:", val_eligibility_mask[:4])
    print("eligible_positive:", eligible_positive[:4])
    print("positive at least in one year:", positive_n, positive_condition[:4])
    
    # ---- validation dataset and dataloader
    #val_dataset = SelectedDataset(sequence_info_val_df, extra_features_val_df, val_labels, val_eligibility_mask, precomputed_val_path, max_len=max_len, device=device)
    val_dataset = SelectedDataset(sequence_info_val_df, extra_features_val_df, val_labels, val_eligibility_mask, precomputed_val_path, max_len=max_len, device=device)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    # --------------------------------------------------------------------------

    # ----------------------------- TEST SET -----------------------------
    # --------------------------------------------------------------------
    sequence_info_test_df, extra_features_test_df = structured_input['sequence']['test'], structured_input['extra_features']['test']
    sequence_info_test_df, extra_features_test_df, test_labels, test_eligibility_mask = prepare_data_for_model_multiyear(sequence_info_test_df, extra_features_test_df, 
                                                                                        valid_ids=have_anamnesis_ids, oversampling=False, verbose=verbose)
    precomputed_test_path = precomputed_path.joinpath(precomputed_test_filename+precomputed_file_extension)

    # ---- filter patients with missing mammogram sequences
    test_index, test_mammogram_id = list(sequence_info_test_df.index), sequence_info_test_df.mammogram_id.tolist()
    with h5py.File(precomputed_test_path, "r") as f:
        valid_ids = [pid[0] for pid in zip(test_index, test_mammogram_id) if str(pid[1]) in f]
    if verbose:
        print('test before:', sequence_info_test_df.shape[0])
        print('test after: ', len(valid_ids))
    sequence_info_test_df = sequence_info_test_df.loc[valid_ids]
    extra_features_test_df = extra_features_test_df.loc[valid_ids].drop(columns=['mammogram_id']).values.astype(float)
    test_labels = test_labels.loc[valid_ids].values
    test_eligibility_mask = test_eligibility_mask.loc[valid_ids].values

    eligible_positive = test_labels * test_eligibility_mask
    positive_condition = (eligible_positive.sum(axis=1) > 0).astype(int)
    positive_n = positive_condition.sum()
    print("labels:", test_labels[:4])
    print("eligibility mask:", test_eligibility_mask[:4])
    print("eligible_positive:", eligible_positive[:4])
    print("positive at least in one year:", positive_n, positive_condition[:4])
    
    # ---- test dataset and dataloader
    #test_dataset = SelectedDataset(sequence_info_test_df, extra_features_test_df, test_labels, test_eligibility_mask, precomputed_test_path, max_len=max_len, device=device)
    test_dataset = SelectedDataset(sequence_info_test_df, extra_features_test_df, test_labels, test_eligibility_mask, precomputed_test_path, max_len=max_len, device=device)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    # --------------------------------------------------------------------
    
    return train_loader, val_loader, test_loader, imratio

def get_dataloaders_multiyear_v2(config, structured_input, device, only_anamnesis=False, verbose=False):
    '''
        ...

        Args:
        -----
            config:
                dictionary. Obtained from a configuration file.
            structured_input:
                dictionary. Output from 'load_input' function. It stores the load datasets
                for training, validation and test.
    '''
    max_len = int(structured_input['max_len'])+1
    
    # -- data filenames and params
    verbose = config['misc']['verbose']
    batch_size = config['data']['batch_size']
    num_workers = config['data']['num_workers']
    time_limit_for_past_mammograms = config['data']['time_limit_for_past_mammograms']
    embedding_store_path = Path(config['data']['embedding_store_path']).joinpath(config['data']['embedding_store_name'])
    
    # -- if 'only_anamnesis' true -> select only the mammograms with anamnesis information.
    have_anamnesis_ids = None

    #if verbose: print("fitting PCA to embeddings ...")
    #dimreducer = PCADimReducer(embedding_store_path, target_dim=156, sample_n=40000)
    
    # -- select custom dataset
    SelectedDataset = CancerRiskDatasetFull

    # ----------------------------- TRAINING SET -----------------------------
    # ------------------------------------------------------------------------
    negative_to_positive_ratio = config['data']['negative_to_positive_ratio']
    sequence_info_train_df, extra_features_train_df = structured_input['sequence']['train'], structured_input['extra_features']['train']
    sequence_info_train_df, extra_features_train_df, train_labels, train_eligibility_mask = prepare_data_for_model_multiyear(sequence_info_train_df, extra_features_train_df, 
                                                                                            valid_ids=have_anamnesis_ids, oversampling=True, positive_ratio=negative_to_positive_ratio, 
                                                                                            verbose=verbose)

    train_labels = train_labels.values
    train_eligibility_mask = train_eligibility_mask.values
    extra_features_train_df = extra_features_train_df.drop(columns=['mammogram_id']).values.astype(float)
    
    # -- element-wise multiplication (eligibility & label)
    eligibility_cols = [f'eligibility_{i}_{i+1}yr' for i in range(0, 5)]
    label_cols = [f'event_indicator_{i}yr' for i in range(1, 6)]
    eligible_positive = train_labels * train_eligibility_mask # must be array, not pandas?
    positive_condition = (eligible_positive.sum(axis=1) > 0).astype(int)
    positive_n = positive_condition.sum()
    print("labels:", train_labels[:4])
    print("eligibility mask:", train_eligibility_mask[:4])
    print("eligible_positive:", eligible_positive[:4])
    print(eligible_positive.shape, train_labels.shape, train_eligibility_mask.shape)
    print("positive at least in one year:", positive_n, type(positive_n), positive_condition[:4])
    imratio = positive_n / train_labels.shape[0] # -- how?
    print(imratio)

    
    
    # ---- training dataset and dataloader
    #train_dataset = SelectedDataset(sequence_info_train_df, extra_features_train_df, train_labels, train_eligibility_mask, precomputed_train_path, max_len=max_len, device=device)
    train_dataset = SelectedDataset(sequence_info_train_df, extra_features_train_df, train_labels, train_eligibility_mask, embedding_store_path, 
                                    max_len=max_len, device=device, time_limit=time_limit_for_past_mammograms)
    train_loader_sampler = DualSampler(train_dataset, labels=positive_condition, batch_size=batch_size, sampling_rate=0.3)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_loader_sampler, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    # ------------------------------------------------------------------------

    # ----------------------------- VALIDATION SET -----------------------------
    # --------------------------------------------------------------------------
    sequence_info_val_df, extra_features_val_df = structured_input['sequence']['validation'], structured_input['extra_features']['validation']
    sequence_info_val_df, extra_features_val_df, val_labels, val_eligibility_mask = prepare_data_for_model_multiyear(sequence_info_val_df, extra_features_val_df, 
                                                                                    valid_ids=have_anamnesis_ids, oversampling=False, verbose=verbose)

    val_labels = val_labels.values
    val_eligibility_mask = val_eligibility_mask.values
    extra_features_val_df = extra_features_val_df.drop(columns=['mammogram_id']).values.astype(float)
    
    eligible_positive = val_labels * val_eligibility_mask
    positive_condition = (eligible_positive.sum(axis=1) > 0).astype(int)
    positive_n = positive_condition.sum()
    print("labels:", val_labels[:4])
    print("eligibility mask:", val_eligibility_mask[:4])
    print("eligible_positive:", eligible_positive[:4])
    print("positive at least in one year:", positive_n, positive_condition[:4])
    
    # ---- validation dataset and dataloader
    val_dataset = SelectedDataset(sequence_info_val_df, extra_features_val_df, val_labels, val_eligibility_mask, embedding_store_path, 
                                  max_len=max_len, device=device, time_limit=time_limit_for_past_mammograms)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    # --------------------------------------------------------------------------

    # ----------------------------- TEST SET -----------------------------
    # --------------------------------------------------------------------
    sequence_info_test_df, extra_features_test_df = structured_input['sequence']['test'], structured_input['extra_features']['test']
    sequence_info_test_df, extra_features_test_df, test_labels, test_eligibility_mask = prepare_data_for_model_multiyear(sequence_info_test_df, extra_features_test_df, 
                                                                                        valid_ids=have_anamnesis_ids, oversampling=False, verbose=verbose)

    test_labels = test_labels.values
    test_eligibility_mask = test_eligibility_mask.values
    extra_features_test_df = extra_features_test_df.drop(columns=['mammogram_id']).values.astype(float)
    
    eligible_positive = test_labels * test_eligibility_mask
    positive_condition = (eligible_positive.sum(axis=1) > 0).astype(int)
    positive_n = positive_condition.sum()
    print("labels:", test_labels[:4])
    print("eligibility mask:", test_eligibility_mask[:4])
    print("eligible_positive:", eligible_positive[:4])
    print("positive at least in one year:", positive_n, positive_condition[:4])
    
    # ---- test dataset and dataloader
    test_dataset = SelectedDataset(sequence_info_test_df, extra_features_test_df, test_labels, test_eligibility_mask, embedding_store_path, 
                                   max_len=max_len, device=device, time_limit=time_limit_for_past_mammograms)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    # --------------------------------------------------------------------
    
    return train_loader, val_loader, test_loader, imratio