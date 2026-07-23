import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
#from hapcancer.shared.embed_store import H5KeyedEmbeddings
from typing import List, Union, Optional

#def create_past_sequence(
#    mammogram_id: str,
#    mammogram_current_date: pd.Timestamp, 
#    prior_history: List[str], 
#    prior_dates: List[pd.Timestamp], 
#    embedding_store_path: Union[Path, str],
#    time_limit: Optional[int] = None
#):
#    embedding_store = H5KeyedEmbeddings(embedding_store_path) # -- open only when getting item (multiprocessing problem might appear if opened at __init__)
#
#    # -- create array of embeddings for past mammogram history using the created embedding store and its wrapper.
#    if mammogram_id in embedding_store:
#        embeddings, missing_keys = embedding_store.get_many(prior_history, ignore_missing=True)
#        # -- remove any key for which we didn't find an embedding for it.
#        prior_dates_valid = [ el for el in prior_dates ]
#        if len(missing_keys)>0:
#            prior_dates_valid = [ mamm_date for mamm_date, mamm_key in zip(prior_dates, prior_history) if mamm_key not in missing_keys ]
#            
#        assert len(embeddings) == len(prior_dates_valid), "embeddings and dates arr not matching in size"
#
#        earliest_date_possible = None
#        if time_limit is not None:
#            earliest_date_possible = pd.Timestamp(mammogram_current_date) - pd.DateOffset(months=time_limit)
#            embeddings_tensor = torch.tensor(np.vstack([ cur_emb for cur_emb, cur_date in zip(embeddings, prior_dates_valid) if cur_date>=earliest_date_possible  ]), dtype=torch.float16)
#            prior_dates_valid = [ cur_date for cur_emb, cur_date in zip(embeddings, prior_dates_valid) if cur_date>=earliest_date_possible  ]
#            
#        list_of_time_difference = [ np.timedelta64(mammogram_current_date - current_date).astype(f'timedelta64[D]')/np.timedelta64(1, 'D') for current_date in prior_dates_valid ]
#        time_diffs = torch.tensor(list_of_time_difference, dtype=torch.int64)
#    else:
#        print("no ID!")
#        
#    #assert embeddings_tensor.shape[1] == self.embedding_dim, "real embedding dimension not matching parsed embedding dimension"
#    return embeddings_tensor, time_diffs

class CancerDatasetSingleYear(Dataset):
    def __init__(
        self, 
        mamm_seq_df, 
        features_df, 
        labels,
        eligibility_mask, 
        embedding_store_path, 
        max_len=156, 
        embedding_dim=1024,
        time_limit=None, 
        device='cpu'
    ):
        """
            Dataset that loads precomputed mammogram sequences from HDF5 files.
        
            Args:
            -----
                
        """
        self.mamm_seq_codes = mamm_seq_df['mammogram_id'].tolist()
        self.mamm_seq_prior_codes = mamm_seq_df['mammogram_prior_codes'].tolist()
        self.mamm_seq_prior_dates = mamm_seq_df['mammogram_prior_dates'].tolist()
        self.features = torch.tensor(features_df, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32) # -- (N, 1)
        self.eligibility_mask = torch.tensor(eligibility_mask, dtype=torch.float32) # -- (N, 1)
        self.embedding_store_path = embedding_store_path
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.device = device
        self.time_limit = time_limit

    def __len__(self):
        return len(self.mamm_seq_codes)

    def __getitem__(self, idx):
        """
            Load patient’s mammogram sequence from HDF5 and apply padding.
        """
        mammogram_id = self.mamm_seq_codes[idx]
        mammogram_prior_history = self.mamm_seq_prior_codes[idx]
        mammogram_prior_dates = self.mamm_seq_prior_dates[idx]
        mammogram_current_date = mammogram_prior_dates[-1]
        embeddings, time_diffs = create_past_sequence(
            mammogram_id, mammogram_current_date, mammogram_prior_history, mammogram_prior_dates, self.embedding_store_path, time_limit=self.time_limit
        )

        # Apply padding dynamically
        seq_len = embeddings.shape[0]
        pad_len = self.max_len - seq_len

        if pad_len > 0:  # Padding required
            pad_embeddings = torch.zeros((pad_len, self.embedding_dim))
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
            embeddings,
            time_diffs,
            attention_mask,
            self.features[idx],
            self.labels[idx],
            self.eligibility_mask[idx]
        )

class CancerDatasetMultiYear(Dataset):
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

    def __len__(self):
        return len(self.mamm_seq_codes)

    def __getitem__(self, idx):
        mammogram_id = self.mamm_seq_codes[idx]
        mammogram_prior_history = self.mamm_seq_prior_codes[idx]
        mammogram_prior_dates = self.mamm_seq_prior_dates[idx]
        mammogram_current_date = mammogram_prior_dates[-1]
        # -- sequence of mammograms and time differences
        embeddings_tensor, time_diffs = create_past_sequence(
            mammogram_id, mammogram_current_date, mammogram_prior_history, mammogram_prior_dates, self.embedding_store_path, time_limit=self.time_limit
        )
        
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


class CancerDatasetSingleYearTFIDF(Dataset):
    def __init__(
        self,
        mamm_seq_df,
        features_df,
        labels,
        eligibility_mask,
        embedding_store_path,
        max_len=128, # maximum amount of past mammograms to include 'max_seq_len'
        embedding_dim=2048,
        time_limit=None, # -- number of months to set the date limit in the past, e. g. 30 months -> consider only mammograms between [cur_date - 30mo, cur_date].
        device='cpu'
    ) -> None:
        self.mamm_seq_codes = mamm_seq_df['mammogram_id'].tolist()
        self.mamm_seq_prior_codes = mamm_seq_df['mammogram_prior_codes'].tolist()
        self.mamm_seq_prior_dates = mamm_seq_df['mammogram_prior_dates'].tolist()
        self.features = torch.tensor(features_df, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.eligibility_mask = torch.tensor(eligibility_mask, dtype=torch.float32)
        self.embedding_store_path = embedding_store_path
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.device = device
        self.time_limit = time_limit

    def __len__(self):
        return len(self.mamm_seq_codes)

    def __getitem__(self, idx):
        """
            Load patient’s mammogram sequence from HDF5 and apply padding.
        """
        mammogram_id = self.mamm_seq_codes[idx]
        mammogram_prior_history = self.mamm_seq_prior_codes[idx]
        mammogram_prior_dates = self.mamm_seq_prior_dates[idx]
        mammogram_current_date = mammogram_prior_dates[-1]
        embeddings, time_diffs = create_past_sequence(
            mammogram_id, mammogram_current_date, mammogram_prior_history, mammogram_prior_dates, self.embedding_store_path, time_limit=self.time_limit
        )
        X = np.hstack([embeddings, time_diffs.reshape(-1, 1)])
        mean_mammogram_seq = torch.tensor(X.mean(axis=0), dtype=torch.float32)

        return (
            idx,
            mammogram_id,
            mean_mammogram_seq,
            self.features[idx],
            self.labels[idx],
            self.eligibility_mask[idx]
        )

class CancerDatasetSingleYearTFIDF_light(Dataset):
    def __init__(
        self,
        mamm_seq_df,
        features_df,
        labels,
        eligibility_mask,
        embedding_store_path,
        max_len=128, # maximum amount of past mammograms to include 'max_seq_len'
        embedding_dim=2048,
        time_limit=None, # -- number of months to set the date limit in the past, e. g. 30 months -> consider only mammograms between [cur_date - 30mo, cur_date].
        device='cpu'
    ) -> None:
        self.mamm_seq_codes = mamm_seq_df['mammogram_id'].tolist()
        self.mamm_seq_prior_codes = mamm_seq_df['mammogram_prior_codes'].tolist()
        self.mamm_seq_prior_dates = mamm_seq_df['mammogram_prior_dates'].tolist()
        self.features = torch.tensor(features_df, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.eligibility_mask = torch.tensor(eligibility_mask, dtype=torch.float32)
        self.embedding_store_path = embedding_store_path
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.device = device
        self.time_limit = time_limit

    def __len__(self):
        return len(self.mamm_seq_codes)

    def __getitem__(self, idx):
        mammogram_id = self.mamm_seq_codes[idx]
        mean_mammogram_seq = -1

        return (
            idx,
            mammogram_id,
            mean_mammogram_seq,
            self.features[idx],
            self.labels[idx],
            self.eligibility_mask[idx]
        )

class CancerDatasetSingleYearTFIDFPrecomputed(Dataset):
    def __init__(
        self,
        mamm_ids,
        features_df,
        labels,
        eligibility_mask,
        device='cpu'
    ) -> None:
        self.mamm_ids = mamm_ids
        self.features = torch.tensor(features_df, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.eligibility_mask = torch.tensor(eligibility_mask, dtype=torch.float32)
        self.device = device

    def __len__(self):
        return len(self.mamm_ids)

    def __getitem__(self, idx):
        """

        """
        return (
            idx,
            self.mamm_ids[idx],
            self.features[idx],
            self.labels[idx],
            self.eligibility_mask[idx]
        )
