import gc
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from torch import sigmoid
import torch.optim as optim
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import f1_score, precision_score, recall_score
from collections import defaultdict

# -- relative imports
from .utils.logger import Logger
from .dataload.dataloaders import get_dataloaders_multiyear_v2
from .dataload.load_input import load_input
from .model import build_model_multiyear
from .utils.config_loss import config_loss, make_masked_loss
from hapcancer.shared.embed_store import H5KeyedEmbeddings
from hapcancer.model.dataload.dim_reducer import PCADimReducer

def train_multiyear_old(config):
    '''
        Model training.

        Args:
        -----
            config:
                ...

        Returns:
        --------
            None.
    '''
    verbose = config['misc']['verbose']
    device = config['misc']['device']
    save_epochs = config['misc']['save_epochs']
    if verbose: print("device: ", device)
    
    # -- load sequence data and mammogram features (perform filtering of number of sequences and birads, if necessary)
    structured_input = load_input_multiyear(config)
    train_loader, val_loader, test_loader, imratio = get_dataloaders_multiyear_v2(
        config, structured_input, device,
        verbose=verbose, only_anamnesis=config['misc']['only_anamnesis']
    )
    model = build_model_multiyear(config, device).to(device)

    max_len = structured_input['max_len']
    batch_size = config['data']['batch_size']
    learning_rate = config['training']['learning_rate']
    weight_decay = config['training']['weight_decay']
    focal_gamma = config['training']['focal_gamma']
    focal_alpha = config['training']['focal_alpha']
    num_epochs = config['training']['epochs']
    patience = config['training']['patience']
    loss_function = config['training']['loss_function']
    max_training_batches_per_epoch = config['training']['max_training_batches_per_epoch']
    max_validation_batches_per_epoch = config['training']['max_validation_batches_per_epoch']
    is_sigmoid_applied = config['model']['mlp_config']['sigmoid']

    checkpoint_path = Path(config['misc']['checkpoint_path'])
    model_name = config['misc']['model_name']

    # -- instantiate logger
    logger_  = Logger(config)

    # -- define loss function
    criterion, optimizer = config_loss(
        config, model, 
        data_len=len(train_loader.dataset) 
    )
    if criterion is None:
        raise Exception("loss function and optimizer not properly defined.")

    # -- loss function suited for the eligibility mask
    masked_criterion = make_masked_loss(criterion)
    epochs, training_loss_v, validation_loss_v = [], [], []
    
    # -- training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        nan_count = 0
        loop_count = 0
        current_iter = 0
        for training_batch in train_loader:
            (
                indices,  # shape: (B, T)
                mammogram_id,
                padded_mammograms,     # shape: (B, T, C, H, W)
                padded_timediff,       # shape: (B, T)
                attention_mask,        # shape: (B, T)
                extra_features,        # shape: (B, T, F)
                labels,                # shape: (B, T)
                eligibility_mask       # shape: (B, T)
            ) = training_batch

            # -- move to device
            padded_mammograms = padded_mammograms.to(device)
            padded_timediff = padded_timediff.to(device).unsqueeze(-1)  # (B, T, 1)
            attention_mask = attention_mask.to(device)
            extra_features = extra_features.to(device)
            labels = labels.to(device)
            eligibility_mask = eligibility_mask.to(device)
            indices = indices.to(device)

            # -- forward
            optimizer.zero_grad()
            predictions = model(padded_mammograms, padded_timediff, attention_mask, extra_features)  # shape: (B, T)

            # -- sanity check for NaNs
            if torch.isnan(predictions).any():
                nan_count += 1
                print(f"Skipped {nan_count} NaN batch(es) so far")
                continue
            
            # -- flatten everything
            flat_preds = predictions.view(-1)               # shape: (B*T,)
            flat_labels = labels.view(-1).float()           # shape: (B*T,)
            flat_mask = eligibility_mask.view(-1).bool()    # shape: (B*T,)
            flat_index = indices.unsqueeze(1).expand(-1, labels.shape[1]).reshape(-1)

            # -- compute masked loss
            loss = masked_criterion(flat_preds, flat_labels, flat_mask, flat_index)

            # -- backward
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            current_iter+=1
            loop_count+=1
            if loop_count%400==0:
                print(loop_count, total_loss, loss)
            if current_iter>max_training_batches_per_epoch:
                break

        avg_train_loss = total_loss / len(train_loader)
        training_loss_v.append(avg_train_loss)

        # --- validation phase ---
        # -- containers
        per_year_preds = defaultdict(list)
        per_year_labels = defaultdict(list)

        model.eval()
        val_loss = 0.0
        current_iter = 0
        loop_count = 0
        with torch.no_grad():
            for validation_batch in val_loader:
                (
                    indices,  # shape: (B, T)
                    mammogram_id,
                    padded_mammograms,     # shape: (B, T, C, H, W)
                    padded_timediff,       # shape: (B, T)
                    attention_mask,        # shape: (B, T)
                    extra_features,        # shape: (B, T, F)
                    labels,                # shape: (B, T)
                    eligibility_mask       # shape: (B, T)
                ) = validation_batch

                # -- move to device
                padded_mammograms = padded_mammograms.to(device)
                padded_timediff = padded_timediff.to(device).unsqueeze(-1)  # (B, T, 1)
                attention_mask = attention_mask.to(device)
                extra_features = extra_features.to(device)
                labels = labels.to(device)
                eligibility_mask = eligibility_mask.to(device)
                indices = indices.to(device)
                B, T = labels.shape
                year_tensor = torch.arange(T).unsqueeze(0).expand(B, T)+1  # shape: (B, T)

                predictions = model(padded_mammograms, padded_timediff, attention_mask, extra_features)

                # -- flatten everything
                flat_preds = predictions.view(-1)               # shape: (B*T,)
                flat_labels = labels.view(-1).float()           # shape: (B*T,)
                flat_mask = eligibility_mask.view(-1).bool()    # shape: (B*T,)
                flat_index = indices.unsqueeze(1).expand(-1, labels.shape[1]).reshape(-1)
                flat_years = year_tensor.view(-1)               # shape: (B*T,)

                # -- apply mask
                valid_preds = flat_preds[flat_mask]
                valid_labels = flat_labels[flat_mask]
                valid_years = flat_years[flat_mask]

                # -- collect per year
                for pred, label, year in zip(valid_preds, valid_labels, valid_years):
                    per_year_preds[int(year.item())].append(pred.item())
                    per_year_labels[int(year.item())].append(label.item())

                # -- compute masked loss
                loss = masked_criterion(flat_preds, flat_labels, flat_mask, flat_index)
                val_loss += loss.item()

                current_iter+=1
                loop_count+=1
                if loop_count%200==0:
                    print(loop_count, val_loss, loss)
                if current_iter>max_validation_batches_per_epoch:
                    break
        
        # -- validation loss (if not X-risks)
        avg_val_loss = val_loss / max_validation_batches_per_epoch*batch_size
        validation_loss_v.append(avg_val_loss)
        epochs.append(epoch)

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}")
        
        # -- calculate validation metrics
        cur_metrics = {
            "year": [], "AUROC": [], "Average Precision": []
        }
        for year in sorted(per_year_preds.keys()):
            y_true = per_year_labels[year]
            y_score = per_year_preds[year]
            if len(set(y_true)) > 1:  # avoid error if only one class
                auc = roc_auc_score(y_true, y_score)
                prc = average_precision_score(y_true, y_score)
            else:
                auc, prc = float('nan'), float('nan')  # or handle however you prefer

            cur_metrics["year"].append(year)
            cur_metrics["AUROC"].append(auc)
            cur_metrics["Average Precision"].append(prc)
            if verbose:
                print(f"Year {year}: AUROC={auc:.4f}, Average Precision={prc:.4f}")

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "metrics_per_year": dict(cur_metrics)
        }
        logger_.logInfo(epoch_metrics)

        # -- save model from each epoch
        if save_epochs:
            checkpoint_path.joinpath(model_name).mkdir(parents=False, exist_ok=True)    
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'average_precision_score': val_pr,
                'roc_auc': val_auc,
            }, checkpoint_path.joinpath(model_name, f'epoch_{epoch:.0f}.pt'))

    # -- save training metrics into a parquet
    logger_.toParquet()

def train_multiyear(config):
    '''
        Model training.

        Args:
        -----
            config:
                ...

        Returns:
        --------
            None.
    '''
    verbose = config['misc']['verbose']
    device = config['misc']['device']
    save_epochs = config['misc']['save_epochs']
    save_best_epochs = config['misc']['save_best_epochs']
    if verbose: print("device: ", device)
    
    # -- load sequence data and mammogram features (perform filtering of number of sequences and birads, if necessary)
    structured_input = load_input(config)
    train_loader, val_loader, test_loader, imratio = get_dataloaders_multiyear_v2(
        config, structured_input, device,
        verbose=verbose, only_anamnesis=False
    )
    max_len = structured_input['max_len']
    model = build_model_multiyear(config, device).to(device)

    batch_size = config['data']['batch_size']
    learning_rate = config['training']['learning_rate']
    weight_decay = config['training']['weight_decay']
    focal_gamma = config['training']['focal_gamma']
    focal_alpha = config['training']['focal_alpha']
    num_epochs = config['training']['epochs']
    patience = config['training']['patience']
    loss_function = config['training']['loss_function']
    max_training_batches_per_epoch = config['training']['max_training_batches_per_epoch']
    max_validation_batches_per_epoch = config['training']['max_validation_batches_per_epoch']
    is_sigmoid_applied = config['model']['mlp_config']['sigmoid']

    checkpoint_path = Path(config['misc']['checkpoint_path'])
    model_name = config['misc']['model_name']

    # -- free memory of loaded datasets
    structured_input = None
    gc.collect()

    # -- instantiate logger
    logger_  = Logger(config)

    # -- define loss function
    criterion, optimizer = config_loss(
        config, model, 
        data_len=len(train_loader.dataset) 
    )
    if criterion is None:
        raise Exception("loss function and optimizer not properly defined.")

    # -- loss function suited for the eligibility mask
    masked_criterion = make_masked_loss(criterion)
    epochs, training_loss_v, validation_loss_v = [], [], []

    # -- metric used for assessment during optimization
    best_auroc_metric = 0.00
    best_ap_metric = 0.00
    auroc_metric_list, ap_metric_list = [], []
    
    # -- training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        nan_count = 0
        loop_count = 0
        current_iter = 0
        print("start training loop")
        for training_batch in train_loader:
            (
                indices,  # shape: (B, T)
                mammogram_id,
                padded_mammograms,     # shape: (B, T, C, H, W)
                padded_timediff,       # shape: (B, T)
                attention_mask,        # shape: (B, T)
                extra_features,        # shape: (B, T, F)
                labels,                # shape: (B, T)
                eligibility_mask       # shape: (B, T)
            ) = training_batch

            # -- move to device
            padded_mammograms = padded_mammograms.to(device)
            padded_timediff = padded_timediff.to(device).unsqueeze(-1)  # (B, T, 1)
            attention_mask = attention_mask.to(device)
            extra_features = extra_features.to(device)
            labels = labels.to(device)
            eligibility_mask = eligibility_mask.to(device)
            indices = indices.to(device)

            #print(padded_mammograms.shape)

            # -- forward
            optimizer.zero_grad()
            predictions = model(padded_mammograms, padded_timediff, attention_mask, extra_features)  # shape: (B, T)

            # -- sanity check for NaNs
            if torch.isnan(predictions).any():
                nan_count += 1
                print(f"Skipped {nan_count} NaN batch(es) so far")
                continue
            
            # -- flatten everything
            flat_preds = predictions.view(-1)               # shape: (B*T,)
            flat_labels = labels.view(-1).float()           # shape: (B*T,)
            flat_mask = eligibility_mask.view(-1).bool()    # shape: (B*T,)
            flat_index = indices.unsqueeze(1).expand(-1, labels.shape[1]).reshape(-1)

            # -- compute masked loss
            loss = masked_criterion(flat_preds, flat_labels, flat_mask, flat_index)

            # -- backward
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            current_iter+=1
            loop_count+=1
            if loop_count%200==0:
                print(loop_count, total_loss, loss)
            if current_iter>max_training_batches_per_epoch:
                break

        avg_train_loss = total_loss / len(train_loader)
        training_loss_v.append(avg_train_loss)

        # --- validation phase ---
        # -- containers
        per_year_preds = defaultdict(list)
        per_year_labels = defaultdict(list)

        model.eval()
        val_loss = 0.0
        current_iter = 0
        loop_count = 0
        with torch.no_grad():
            for validation_batch in val_loader:
                (
                    indices,  # shape: (B, T)
                    mammogram_id,
                    padded_mammograms,     # shape: (B, T, C, H, W)
                    padded_timediff,       # shape: (B, T)
                    attention_mask,        # shape: (B, T)
                    extra_features,        # shape: (B, T, F)
                    labels,                # shape: (B, T)
                    eligibility_mask       # shape: (B, T)
                ) = validation_batch

                # -- move to device
                padded_mammograms = padded_mammograms.to(device)
                padded_timediff = padded_timediff.to(device).unsqueeze(-1)  # (B, T, 1)
                attention_mask = attention_mask.to(device)
                extra_features = extra_features.to(device)
                labels = labels.to(device)
                eligibility_mask = eligibility_mask.to(device)
                indices = indices.to(device)
                B, T = labels.shape
                year_tensor = torch.arange(T).unsqueeze(0).expand(B, T)+1  # shape: (B, T)

                predictions = model(padded_mammograms, padded_timediff, attention_mask, extra_features)

                # -- flatten everything
                flat_preds = predictions.view(-1)               # shape: (B*T,)
                flat_labels = labels.view(-1).float()           # shape: (B*T,)
                flat_mask = eligibility_mask.view(-1).bool()    # shape: (B*T,)
                flat_index = indices.unsqueeze(1).expand(-1, labels.shape[1]).reshape(-1)
                flat_years = year_tensor.view(-1)               # shape: (B*T,)

                # -- apply mask
                valid_preds = flat_preds[flat_mask]
                valid_labels = flat_labels[flat_mask]
                valid_years = flat_years[flat_mask]

                # -- collect per year
                for pred, label, year in zip(valid_preds, valid_labels, valid_years):
                    per_year_preds[int(year.item())].append(pred.item())
                    per_year_labels[int(year.item())].append(label.item())

                # -- compute masked loss
                loss = masked_criterion(flat_preds, flat_labels, flat_mask, flat_index)
                val_loss += loss.item()

                current_iter+=1
                loop_count+=1
                if loop_count%200==0:
                    print(loop_count, val_loss, loss)
                if current_iter>max_validation_batches_per_epoch:
                    break
        
        # -- validation loss (if not X-risks)
        avg_val_loss = val_loss / max_validation_batches_per_epoch*batch_size
        validation_loss_v.append(avg_val_loss)
        epochs.append(epoch)

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}")
        
        # -- calculate validation metrics
        cur_metrics = {
            "year": [], "AUROC": [], "Average Precision": []
        }
        for year in sorted(per_year_preds.keys()):
            y_true = per_year_labels[year]
            y_score = per_year_preds[year]
            if len(set(y_true)) > 1:  # avoid error if only one class
                auc = roc_auc_score(y_true, y_score)
                prc = average_precision_score(y_true, y_score)
            else:
                auc, prc = float('nan'), float('nan')  # or handle however you prefer

            cur_metrics["year"].append(year)
            cur_metrics["AUROC"].append(auc)
            cur_metrics["Average Precision"].append(prc)
            if verbose:
                print(f"Year {year}: AUROC={auc:.4f}, Average Precision={prc:.4f}")

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "metrics_per_year": dict(cur_metrics)
        }
        logger_.logInfo(epoch_metrics)

        # -- save model from each epoch
        if save_epochs:
            checkpoint_path.joinpath(model_name).mkdir(parents=False, exist_ok=True)    
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'average_precision_score': val_pr,
                'roc_auc': val_auc,
            }, checkpoint_path.joinpath(model_name, f'epoch_{epoch:.0f}.pt'))

        # -- calculate metric for optimization
        year_weights = [0.1, 0.15, 0.3, 0.3, 0.15]
        current_auroc_metric = np.sum( [ year_weights[index]*cur_metrics["AUROC"][index] for index in range(5) ] )
        current_ap_metric = np.sum( [ year_weights[index]*cur_metrics["Average Precision"][index] for index in range(5) ] )
        auroc_metric_list.append(current_auroc_metric)
        ap_metric_list.append(current_ap_metric)

        if current_auroc_metric>best_auroc_metric:
            best_auroc_metric = current_auroc_metric
            best_ap_metric = current_ap_metric
        
            if save_best_epochs:
                checkpoint_path.joinpath(model_name).mkdir(parents=False, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'average_precision_score': best_ap_metric,
                    'roc_auc': best_auroc_metric,
                }, checkpoint_path.joinpath(model_name, f'best_model.pt'))   

    # -- save training metrics into a parquet
    logger_.toParquet()

def train_for_tuning_multiyear(config):
    '''
        Model training.

        Args:
        -----
            config:
                ...

        Returns:
        --------
            None.
    '''
    verbose = config['misc']['verbose']
    device = config['misc']['device']
    save_epochs = config['misc']['save_epochs']
    if verbose: print("device: ", device)

    batch_size = config['data']['batch_size']
    learning_rate = config['training']['learning_rate']
    weight_decay = config['training']['weight_decay']
    focal_gamma = config['training']['focal_gamma']
    focal_alpha = config['training']['focal_alpha']
    num_epochs = config['training']['epochs']
    patience = config['training']['patience']
    loss_function = config['training']['loss_function']
    max_training_batches_per_epoch = config['training']['max_training_batches_per_epoch']
    max_validation_batches_per_epoch = config['training']['max_validation_batches_per_epoch']
    is_sigmoid_applied = config['model']['mlp_config']['sigmoid']

    checkpoint_path = Path(config['misc']['checkpoint_path'])
    model_name = config['misc']['model_name']

    # -- instantiate logger
    #logger_  = Logger(config)

    # -- load input
    structured_input = load_input(config)
    max_len = structured_input['max_len']

    # -- dataloaders
    train_loader, val_loader, test_loader, imratio = get_dataloaders_multiyear_v2(
        config, structured_input, device,
        verbose=verbose, only_anamnesis=config['misc']['only_anamnesis']
    )
    model = build_model_multiyear(config, device).to(device)

    # -- free memory of loaded datasets
    structured_input = None
    gc.collect()

    # -- define loss function
    criterion, optimizer = config_loss(
        config, model, 
        data_len=len(train_loader.dataset) 
    )
    if criterion is None:
        raise Exception("loss function and optimizer not properly defined.")

    # -- loss function suited for the eligibility mask
    masked_criterion = make_masked_loss(criterion)

    # -- metric used for assessment during optimization
    best_auroc_metric = 0.00
    best_ap_metric = 0.00
    auroc_metric_list, ap_metric_list = [], []
    
    # -- training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        nan_count = 0
        loop_count = 0
        current_iter = 0
        for training_batch in train_loader:
            (
                indices,  # shape: (B, T)
                mammogram_id,
                padded_mammograms,     # shape: (B, T, C, H, W)
                padded_timediff,       # shape: (B, T)
                attention_mask,        # shape: (B, T)
                extra_features,        # shape: (B, T, F)
                labels,                # shape: (B, T)
                eligibility_mask       # shape: (B, T)
            ) = training_batch

            # -- move to device
            padded_mammograms = padded_mammograms.to(device)
            padded_timediff = padded_timediff.to(device).unsqueeze(-1)  # (B, T, 1)
            attention_mask = attention_mask.to(device)
            extra_features = extra_features.to(device)
            labels = labels.to(device)
            eligibility_mask = eligibility_mask.to(device)
            indices = indices.to(device)

            # -- forward
            optimizer.zero_grad()
            predictions = model(padded_mammograms, padded_timediff, attention_mask, extra_features)  # shape: (B, T)

            # -- sanity check for NaNs
            if torch.isnan(predictions).any():
                nan_count += 1
                print(f"Skipped {nan_count} NaN batch(es) so far")
                continue
            
            # -- flatten everything
            flat_preds = predictions.view(-1)               # shape: (B*T,)
            flat_labels = labels.view(-1).float()           # shape: (B*T,)
            flat_mask = eligibility_mask.view(-1).bool()    # shape: (B*T,)
            flat_index = indices.unsqueeze(1).expand(-1, labels.shape[1]).reshape(-1)

            # -- compute masked loss
            loss = masked_criterion(flat_preds, flat_labels, flat_mask, flat_index)

            # -- backward
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            current_iter+=1
            loop_count+=1
            if loop_count%5000==0:
                print(loop_count, total_loss, loss)
            if current_iter>max_training_batches_per_epoch:
                break

        avg_train_loss = total_loss / len(train_loader)

        # --- validation phase ---
        # -- containers
        per_year_preds = defaultdict(list)
        per_year_labels = defaultdict(list)

        model.eval()
        val_loss = 0.0
        current_iter = 0
        loop_count = 0
        with torch.no_grad():
            for validation_batch in val_loader:
                (
                    indices,  # shape: (B, T)
                    mammogram_id,
                    padded_mammograms,     # shape: (B, T, C, H, W)
                    padded_timediff,       # shape: (B, T)
                    attention_mask,        # shape: (B, T)
                    extra_features,        # shape: (B, T, F)
                    labels,                # shape: (B, T)
                    eligibility_mask       # shape: (B, T)
                ) = validation_batch

                # -- move to device
                padded_mammograms = padded_mammograms.to(device)
                padded_timediff = padded_timediff.to(device).unsqueeze(-1)  # (B, T, 1)
                attention_mask = attention_mask.to(device)
                extra_features = extra_features.to(device)
                labels = labels.to(device)
                eligibility_mask = eligibility_mask.to(device)
                indices = indices.to(device)
                B, T = labels.shape
                year_tensor = torch.arange(T).unsqueeze(0).expand(B, T)+1  # shape: (B, T)

                predictions = model(padded_mammograms, padded_timediff, attention_mask, extra_features)

                # -- flatten everything
                flat_preds = predictions.view(-1)               # shape: (B*T,)
                flat_labels = labels.view(-1).float()           # shape: (B*T,)
                flat_mask = eligibility_mask.view(-1).bool()    # shape: (B*T,)
                flat_index = indices.unsqueeze(1).expand(-1, labels.shape[1]).reshape(-1)
                flat_years = year_tensor.view(-1)               # shape: (B*T,)

                # -- apply mask
                valid_preds = flat_preds[flat_mask]
                valid_labels = flat_labels[flat_mask]
                valid_years = flat_years[flat_mask]

                # -- collect per year
                for pred, label, year in zip(valid_preds, valid_labels, valid_years):
                    per_year_preds[int(year.item())].append(pred.item())
                    per_year_labels[int(year.item())].append(label.item())

                # -- compute masked loss
                loss = masked_criterion(flat_preds, flat_labels, flat_mask, flat_index)
                val_loss += loss.item()

                current_iter+=1
                loop_count+=1
                if loop_count%5000==0:
                    print(loop_count, val_loss, loss)
                if current_iter>max_validation_batches_per_epoch:
                    break
            
        avg_val_loss = val_loss / max_validation_batches_per_epoch*batch_size
        if verbose:
            print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}")
        
        # -- calculate validation metrics
        cur_metrics = {
            "year": [], "AUROC": [], "Average Precision": []
        }
        for year in sorted(per_year_preds.keys()):
            y_true = per_year_labels[year]
            y_score = per_year_preds[year]
            if len(set(y_true)) > 1:  # avoid error if only one class
                auc = roc_auc_score(y_true, y_score)
                prc = average_precision_score(y_true, y_score)
            else:
                auc, prc = float('nan'), float('nan')  # or handle however you prefer

            cur_metrics["year"].append(year)
            cur_metrics["AUROC"].append(auc)
            cur_metrics["Average Precision"].append(prc)
            if verbose:
                print(f"Year {year}: AUROC={auc:.4f}, Average Precision={prc:.4f}")

        # -- calculate metric for optimization
        year_weights = [0.1, 0.15, 0.3, 0.3, 0.15]
        current_auroc_metric = np.sum( [ year_weights[index]*cur_metrics["AUROC"][index] for index in range(5) ] )
        current_ap_metric = np.sum( [ year_weights[index]*cur_metrics["Average Precision"][index] for index in range(5) ] )
        auroc_metric_list.append(current_auroc_metric)
        ap_metric_list.append(current_ap_metric)

        if current_auroc_metric>best_auroc_metric:
            best_auroc_metric = current_auroc_metric
            best_ap_metric = current_ap_metric

        if epoch>=4 and np.mean(np.array(auroc_metric_list))<0.53:
            break
    return best_ap_metric, best_auroc_metric


