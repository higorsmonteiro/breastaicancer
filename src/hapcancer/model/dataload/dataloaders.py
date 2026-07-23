import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from libauc.sampler import DualSampler
from hapcancer.model.dataload.datasets import CancerDatasetSingleYear, CancerDatasetMultiYear, CancerDatasetSingleYearTFIDF, CancerDatasetSingleYearTFIDF_light
from hapcancer.model.dataload.datasets import CancerDatasetSingleYearTFIDF_light, CancerDatasetSingleYearTFIDFPrecomputed
from typing import Optional, List, Tuple

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
    verbose = config['misc']['verbose']
    batch_size = config['data']['batch_size']
    num_workers = config['data']['num_workers']
    sampling_strategy = config['data']['sampling_strategy']
    negative_to_positive_ratio = config['data']['negative_to_positive_ratio']

    cols_to_remove = config['data']['followup_columns']
    cols_to_remove += config['data']['event_indicator_columns']
    cols_to_remove += config['data']['multiyear_eligibility_columns']

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

def get_dataloaders_singleyear_(
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