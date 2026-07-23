import gc
import torch
import inspect
import numpy as np
from pathlib import Path
from collections import defaultdict

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import f1_score, precision_score, recall_score

import hapcancer.model.utils.training as training_utils

from hapcancer.model.utils.logger import Logger
from hapcancer.model.utils.config_loss_ import config_loss, config_loss_many, make_masked_loss
from hapcancer.model.dataload.dataloaders import get_dataloaders_multiyear_v2, get_dataloaders_singleyear, get_dataloaders_singleyear_flatdata
from hapcancer.model.model import build_model_singleyear, build_model_multiyear, build_model_singleyear_without_transformer
from hapcancer.model.dataload.load_input import load_input, transform_eligibility_to_singleyear

# -- relative imports
from .utils.model_io import configure_encoder_freeze
from .utils.custom_loss import interval_bce_loss

def train_multiyear(config):
    '''
        Model training (discrete-time survival product).
        Trains with masked BCE on per-interval logits; evaluates with within-k risk.
    '''
    verbose = config['misc']['verbose']
    device  = config['misc']['device']
    checkpoint_path = Path(config['misc']['checkpoint_path'])
    model_name = config['misc']['model_name']
    save_epochs = config['misc']['save_epochs']
    save_best_epochs = config['misc']['save_best_epochs']
    if verbose: print("device: ", device)

    # -- load datasets and instantiate model
    structured_input = load_input(config)
    train_loader, val_loader, test_loader, imratio = get_dataloaders_multiyear_v2(
        config, structured_input, device, verbose=verbose, only_anamnesis=False
    )
    max_len = structured_input['max_len']
    model = build_model_multiyear(config, device).to(device)

    # -- free memory
    structured_input = None
    gc.collect()

    logger_  = Logger(config)

    # -- optimizer (reuse your config_loss for optimizer if you want; we ignore its criterion)
    # ??? this should be used to load compositional auc loss
    _, optimizer = config_loss(config, model, data_len=len(train_loader.dataset))
    if optimizer is None:
        raise Exception("optimizer not properly defined.")

    # ==== training state ====
    epochs, training_loss_v, validation_loss_v = [], [], []
    best_auroc_metric = -np.inf
    best_ap_metric    = -np.inf
    auroc_metric_list, ap_metric_list = [], []

    # (optional) pos_weight per interval: set to None to start simple.
    pos_weight = None  # e.g., torch.tensor([...], device=device)

    # ================= loop =================
    num_epochs = config['training']['epochs']
    max_training_batches_per_epoch   = config['training']['max_training_batches_per_epoch']
    max_validation_batches_per_epoch = config['training']['max_validation_batches_per_epoch']
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        train_batches = 0

        nan_count = 0
        loop_count = 0
        current_iter = 0
        if verbose: print("start training loop")

        for training_batch in train_loader:
            (
                indices,              # (B, T)  [unused here, kept for compat]
                mammogram_id,
                padded_mammograms,    # (B, T, C, H, W)
                padded_timediff,      # (B, T)
                attention_mask,       # (B, T)
                extra_features,       # (B, T, F)
                labels,               # (B, T) interval labels (first-event-in-interval)
                eligibility_mask      # (B, T) 1 where label observed
            ) = training_batch

            # -- device
            padded_mammograms = padded_mammograms.to(device)
            padded_timediff   = padded_timediff.to(device).unsqueeze(-1)  # (B, T, 1)
            attention_mask    = attention_mask.to(device)
            extra_features    = extra_features.to(device)
            labels            = labels.to(device)
            eligibility_mask  = eligibility_mask.to(device)

            optimizer.zero_grad()
            out = model(padded_mammograms, padded_timediff, attention_mask, extra_features)
            logits_h = out["logits_h"]    # (B,T) interval hazard logits
            # risk = out["risk"]          # (B,T) within-k; not needed for training loss

            # NaN check on logits
            if torch.isnan(logits_h).any():
                nan_count += 1
                if verbose:
                    print(f"Skipped {nan_count} NaN batch(es) so far")
                continue

            loss = interval_bce_loss(logits_h, labels, eligibility_mask, pos_weight=pos_weight)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            train_batches += 1
            current_iter += 1
            loop_count += 1
            if verbose and loop_count % 400 == 0:
                print(loop_count, total_loss, loss.item())
            if current_iter > max_training_batches_per_epoch:
                break

        avg_train_loss = total_loss / max(1, train_batches)
        training_loss_v.append(avg_train_loss)

        # ---------- validation ----------
        model.eval()
        val_loss = 0.0
        val_batches = 0

        per_year_preds  = defaultdict(list)
        per_year_labels = defaultdict(list)

        current_iter = 0
        loop_count = 0
        with torch.no_grad():
            for validation_batch in val_loader:
                (
                    indices,              # (B, T)
                    mammogram_id,
                    padded_mammograms,    # (B, T, C, H, W)
                    padded_timediff,      # (B, T)
                    attention_mask,       # (B, T)
                    extra_features,       # (B, T, F)
                    labels,               # (B, T)
                    eligibility_mask      # (B, T)
                ) = validation_batch

                padded_mammograms = padded_mammograms.to(device)
                padded_timediff   = padded_timediff.to(device).unsqueeze(-1)
                attention_mask    = attention_mask.to(device)
                extra_features    = extra_features.to(device)
                labels            = labels.to(device)
                eligibility_mask  = eligibility_mask.to(device)

                out = model(padded_mammograms, padded_timediff, attention_mask, extra_features)
                logits_h = out["logits_h"]           # (B,T)
                risk     = out["risk"]               # (B,T) = P(cancer within k years)

                # val loss: same masked interval BCE on logits
                loss = interval_bce_loss(logits_h, labels, eligibility_mask, pos_weight=pos_weight)
                val_loss += loss.item()
                val_batches += 1

                # collect per-year metrics using within-k risk[:, k]
                B, T = labels.shape
                for k in range(T):  # k = 0..T-1 -> year = k+1
                    mk = eligibility_mask[:, k].bool()
                    if mk.any():
                        per_year_preds[k+1].extend(risk[mk, k].detach().cpu().tolist())
                        per_year_labels[k+1].extend(labels[mk, k].detach().cpu().tolist())

                current_iter += 1
                loop_count += 1
                if verbose and loop_count % 400 == 0:
                    print(loop_count, val_loss, loss.item())
                if current_iter > max_validation_batches_per_epoch:
                    break

        avg_val_loss = val_loss / max(1, val_batches)
        validation_loss_v.append(avg_val_loss)
        epochs.append(epoch)

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}")

        # -- metrics per year on within-k risk
        cur_metrics = {"year": [], "AUROC": [], "Average Precision": []}
        for year in sorted(per_year_preds.keys()):
            y_true  = per_year_labels[year]
            y_score = per_year_preds[year]
            if len(set(y_true)) > 1:
                auc = roc_auc_score(y_true, y_score)
                prc = average_precision_score(y_true, y_score)
            else:
                auc, prc = float('nan'), float('nan')
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

        # -- weighted objective for model selection
        year_weights = [0.1, 0.15, 0.3, 0.3, 0.15]  # assumes T=5
        # pad if fewer years are present
        K = len(cur_metrics["AUROC"])
        w = np.array(year_weights[:K], dtype=float)
        w = w / (w.sum() if w.sum() > 0 else 1)

        current_auroc_metric = float(np.nansum(w * np.array(cur_metrics["AUROC"][:K])))
        current_ap_metric    = float(np.nansum(w * np.array(cur_metrics["Average Precision"][:K])))
        auroc_metric_list.append(current_auroc_metric)
        ap_metric_list.append(current_ap_metric)

        # -- save checkpoints
        if save_epochs:
            checkpoint_path.joinpath(model_name).mkdir(parents=False, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'average_precision_score': current_ap_metric,
                'roc_auc': current_auroc_metric,
            }, checkpoint_path.joinpath(model_name, f'epoch_{epoch:.0f}.pt'))

        if current_auroc_metric > best_auroc_metric:
            best_auroc_metric = current_auroc_metric
            best_ap_metric    = current_ap_metric
            if save_best_epochs:
                checkpoint_path.joinpath(model_name).mkdir(parents=False, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'average_precision_score': best_ap_metric,
                    'roc_auc': best_auroc_metric,
                }, checkpoint_path.joinpath(model_name, f'best_model.pt'))

    # -- persist logs
    logger_.toParquet()

def train_singleyear(config):
    """
        Train one-horizon (e.g., 2-year) model with Compositional AUC (LibAUC) + PDSCA.
        Dataloader must return 1D labels/masks (B,).
    """
    verbose = config['misc']['verbose']
    device  = config['misc']['device']
    checkpoint_path = Path(config['misc']['checkpoint_path'])
    model_name = config['misc']['model_name']
    save_epochs = config['misc']['save_epochs']
    save_best_epochs = config['misc']['save_best_epochs']
    target_year = int(config['task']['target_year'])  # e.g., 2
    use_pretrained = config['misc']['pretrained']['load']
    freeze_transformer = config['model']['freeze_transformer']

    if verbose: print(f"device: {device} | target_year: {target_year}")

    # -- data & model
    print("load input & transform to a single year prediction & get dataloaders")
    structured_input = transform_eligibility_to_singleyear(config, target_year, load_test=False) # does load input and transform for a single year label
    train_loader, val_loader, test_loader, imratio = get_dataloaders_singleyear(
        config, structured_input, device, verbose=verbose
    )

    print("instantiate model")
    # -- if a pretrained model is to be loaded
    if use_pretrained:
        model = training_utils.load_pretrained_singleyear(config)
    else:
        model = build_model_singleyear(config, device).to(device)

    # -- freeze transformer?
    if config['model']['freeze_transformer']:
        configure_encoder_freeze(
            model.encoder,
            train_last_n_layers=0,      # change to 1 or 2 to train last blocks
            train_layernorms=False,
            train_input_proj=False,
            train_time_encoding=False,
            train_attention_pooling=False
        )
        for p in model.mlp.parameters():
            p.requires_grad = True
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name, param.data)

    # free memory
    structured_input = None
    gc.collect()

    logger_ = Logger(config)

    # -- define loss function
    print("instantiate loss function and optimizer")
    criterion, optimizer = config_loss_many(
        config, model, 
        data_len=len(train_loader.dataset),
        imratio=imratio
    )
    if criterion is None:
        raise Exception("loss function and optimizer not properly defined.")
    else:
        if verbose:
            print(type(criterion), type(optimizer))
        else:
            pass

    # ----- training state -----
    best_sel_metric = -np.inf  # model selection metric (AUROC or AUPRC)
    epochs, training_loss_v, validation_loss_v = [], [], []

    num_epochs = config['training']['epochs']
    max_train_batches = config['training']['max_training_batches_per_epoch']
    max_val_batches   = config['training']['max_validation_batches_per_epoch']
    
    # -- target learning rate for the warm-up phase
    target_lr = config['training']['learning_rate']
    warmup_steps = config['training']['warmup_steps']
    # -- when to anneal the learning rate
    epochs_to_update = {
        int(0.5*num_epochs), int(0.75*num_epochs), int(0.9*num_epochs)
    }
    opt_step_counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss, n_batches = 0.0, 0
        if verbose: print("starting training loop ...")

        # --------------------------------------------------
        # -------------------- training -------------------- 
        # --------------------------------------------------
        all_scores, all_labels = [], []
        for b_idx, batch in enumerate(train_loader):
            (
                indices,              # (B,)
                mammogram_id,
                padded_mammograms,    # (B, Tseq, C, H, W)
                padded_timediff,      # (B, Tseq)
                attention_mask,       # (B, Tseq)
                extra_features,       # (B, Tseq, F) or (B, F)
                labels,               # (B,)
                eligibility_mask      # (B,)
            ) = batch

            # device
            padded_mammograms = padded_mammograms.to(device)
            padded_timediff   = padded_timediff.to(device).unsqueeze(-1)
            attention_mask    = attention_mask.to(device)
            extra_features    = extra_features.to(device)
            labels            = labels.to(device) # 1d vector
            eligibility_mask  = eligibility_mask.to(device) # 1d vector

            optimizer.zero_grad()
            preds = model(padded_mammograms, padded_timediff, attention_mask, extra_features)
            if torch.isnan(preds).any():
                continue

            #loss = masked_criterion(preds, labels, eligibility_mask, index=indices)
            signature = inspect.signature(criterion)
            if index in signature.parameters:
                loss = criterion(preds, labels, index=indices)
            else:
                loss = criterion(preds, labels)
            loss.backward()
            # global-norm clipping (test to avoid exploding NaN in the transformers' output)
            # now: tested -> it does not help, it actually it worses the optimization. 
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=6)
            optimizer.step()
            # -- warm-up learning rate to improve stability
            #opt_step_counter+=1
            #if opt_step_counter<=warmup_steps:
            #    training_utils.set_outer_lr(optimizer, opt_step_counter, warmup_steps, target_lr, verbose=True)
            
            # -- apply warm-up of the learning rate
            
            # -- update learning rate (if the method exists - guaranteed using the CompositionalAUCLoss)
            #if epoch in epochs_to_update:
            #    optimizer.update_lr(decay_factor=10, decay_factor0=10)
            #    print(f"new learning rate: {optimizer.lr}")

            train_loss += float(loss.item())

            m = eligibility_mask.bool()
            if m.any():
                all_scores.append(preds[m].detach().cpu())
                all_labels.append(labels[m].detach().cpu())

            n_batches  += 1
            if verbose and (b_idx % 100 == 0):
                print(eligibility_mask.sum())
                print(f"[train] it={b_idx} loss={train_loss/max(1,n_batches):.5f}")
            if b_idx >= max_train_batches:
                break

        avg_train_loss = train_loss / max(1, n_batches)
        training_loss_v.append(avg_train_loss)

        # -- metrics on training
        train_auc, train_ap = training_utils.calculate_metrics(all_labels, all_scores)
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.5f} | Training AUROC: {train_auc:.4f} | Training AP: {train_ap:.4f}")

        # --------------------------------------------------
        # ------------------- validation ------------------- 
        # --------------------------------------------------
        model.eval()
        val_loss, nb = 0.0, 0
        all_scores, all_labels = [], []
        if verbose: print("starting validation loop ...")
        with torch.no_grad():
            for b_idx, batch in enumerate(val_loader):
                (
                    indices,
                    mammogram_id,
                    padded_mammograms,
                    padded_timediff,
                    attention_mask,
                    extra_features,
                    labels,
                    eligibility_mask
                ) = batch

                padded_mammograms = padded_mammograms.to(device)
                padded_timediff   = padded_timediff.to(device).unsqueeze(-1)
                attention_mask    = attention_mask.to(device)
                extra_features    = extra_features.to(device)
                labels            = labels.to(device)
                eligibility_mask  = eligibility_mask.to(device)

                preds = model(padded_mammograms, padded_timediff, attention_mask, extra_features)

                #loss = masked_criterion(preds, labels, eligibility_mask, index=indices)
                loss = criterion(preds, labels, index=indices)
                val_loss += float(loss.item()); nb += 1

                m = eligibility_mask.bool()
                if m.any():
                    all_scores.append(preds[m].detach().cpu())
                    all_labels.append(labels[m].detach().cpu())

                if verbose and (b_idx % 800 == 0):
                    print(f"[train] it={b_idx} loss={val_loss/max(1,n_batches):.5f}")
                
                if b_idx >= max_val_batches:
                    break

        avg_val_loss = val_loss / max(1, nb)
        validation_loss_v.append(avg_val_loss)
        epochs.append(epoch)

        # metrics on validation
        val_auc, val_ap = training_utils.calculate_metrics(all_labels, all_scores)
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f} | AUROC: {val_auc:.4f} | AP: {val_ap:.4f}")

        # --> log
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "training metrics": {"AUROC": float(train_auc), "Average Precision": float(train_ap)},
            "validation metrics": {"AUROC": float(val_auc), "Average Precision": float(val_ap)},
            "target_year": target_year
        }
        logger_.logInfo(epoch_metrics)

        # selection metric: pick what matters (AUPRC is good under imbalance)
        sel = val_ap if np.isfinite(val_ap) else (val_auc if np.isfinite(val_auc) else -np.inf)

        # save checkpoints
        if save_epochs:
            checkpoint_path.joinpath(model_name).mkdir(parents=False, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_ap': val_ap, 'val_auc': val_auc,
                'target_year': target_year,
            }, checkpoint_path.joinpath(model_name, f'epoch_{epoch:.0f}.pt'))

        if sel > best_sel_metric:
            best_sel_metric = sel
            if save_best_epochs:
                checkpoint_path.joinpath(model_name).mkdir(parents=False, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_ap': val_ap, 'val_auc': val_auc,
                    'target_year': target_year,
                }, checkpoint_path.joinpath(model_name, f'best_model.pt'))

    # persist logs
    logger_.toParquet()

def train_singleyear_no_transformer(config):
    """
        Train one-horizon (e.g., 2-year) model with Compositional AUC (LibAUC) + PDSCA.
        Dataloader must return 1D labels/masks (B,).
    """
    verbose = config['misc']['verbose']
    device  = config['misc']['device']
    checkpoint_path = Path(config['misc']['checkpoint_path'])
    model_name = config['misc']['model_name']
    save_epochs = config['misc']['save_epochs']
    save_best_epochs = config['misc']['save_best_epochs']
    target_year = int(config['task']['target_year'])  # e.g., 2
    use_pretrained = config['misc']['pretrained']['load']

    if verbose: print(f"device: {device} | target_year: {target_year}")

    # -- data & model
    print("load input & transform to a single year prediction & get dataloaders")
    structured_input = transform_eligibility_to_singleyear(config, target_year, load_test=False) # does load input and transform for a single year label
    train_loader, val_loader, test_loader, imratio = get_dataloaders_singleyear(
        config, structured_input, with_transformer=False, verbose=verbose, device=device
    )

    print("instantiate model")
    # -- if a pretrained model is to be loaded
    if use_pretrained:
        model = training_utils.load_pretrained_singleyear(config)
    else:
        #model = build_model_singleyear(config, device).to(device)
        model = build_model_singleyear_without_transformer(config, device).to(device)

    # free memory
    structured_input = None
    gc.collect()

    logger_ = Logger(config)

    # -- define loss function
    print("instantiate loss function and optimizer")
    criterion, optimizer = config_loss_many(
        config, model, 
        data_len=len(train_loader.dataset),
        imratio=imratio
    )
    if criterion is None:
        raise Exception("loss function and optimizer not properly defined.")
    else:
        if verbose:
            print(type(criterion), type(optimizer))
        else:
            pass

    # ----- training state -----
    best_sel_metric = -np.inf  # model selection metric (AUROC or AUPRC)
    epochs, training_loss_v, validation_loss_v = [], [], []

    num_epochs = config['training']['epochs']
    max_train_batches = config['training']['max_training_batches_per_epoch']
    max_val_batches   = config['training']['max_validation_batches_per_epoch']
    
    # -- target learning rate for the warm-up phase
    target_lr = config['training']['learning_rate']
    warmup_steps = config['training']['warmup_steps']
    # -- when to anneal the learning rate
    epochs_to_update = {
        int(0.5*num_epochs), int(0.75*num_epochs), int(0.9*num_epochs)
    }
    opt_step_counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss, n_batches = 0.0, 0
        if verbose: print("starting training loop ...")

        # --------------------------------------------------
        # -------------------- training -------------------- 
        # --------------------------------------------------
        all_scores, all_labels = [], []
        for b_idx, batch in enumerate(train_loader):
            (
                indices,              # (B,)
                mammogram_id,
                mean_mammogram_vec,    # (B, Tseq, C, H, W)
                extra_features,       # (B, Tseq, F) or (B, F)
                labels,               # (B,)
                eligibility_mask      # (B,)
            ) = batch

            mean_mammogram_vec = mean_mammogram_vec.to(device)
            extra_features    = extra_features.to(device)
            labels            = labels.to(device) # 1d vector
            eligibility_mask  = eligibility_mask.to(device) # 1d vector (not used here, but it might very useful if we extend to the multi-year model)

            optimizer.zero_grad()
            preds = model(mean_mammogram_vec, extra_features)
            preds = preds.squeeze(-1) # (B,1) -> (B,)
            #preds = model(padded_mammograms, padded_timediff, attention_mask, extra_features)
            if torch.isnan(preds).any():
                continue

            signature = inspect.signature(criterion)
            if 'index' in signature.parameters:
                loss = criterion(preds, labels, index=indices)
            else:
                loss = criterion(preds, labels)

            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())

            m = eligibility_mask.bool()
            if m.any():
                all_scores.append(preds[m].detach().cpu())
                all_labels.append(labels[m].detach().cpu())

            n_batches  += 1
            if verbose and (b_idx % 50 == 0):
                print(eligibility_mask.sum())
                print(f"[train] it={b_idx} loss={train_loss/max(1,n_batches):.5f}")
            if b_idx >= max_train_batches:
                break

        avg_train_loss = train_loss / max(1, n_batches)
        training_loss_v.append(avg_train_loss)

        # -- metrics on training
        train_auc, train_ap = training_utils.calculate_metrics(all_labels, all_scores)
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.5f} | Training AUROC: {train_auc:.4f} | Training AP: {train_ap:.4f}")

        # --------------------------------------------------
        # ------------------- validation ------------------- 
        # --------------------------------------------------
        model.eval()
        val_loss, n_batches = 0.0, 0
        all_scores, all_labels = [], []
        if verbose: print("starting validation loop ...")
        with torch.no_grad():
            for b_idx, batch in enumerate(val_loader):
                (
                    indices,              # (B,)
                    mammogram_id,
                    mean_mammogram_vec,    # (B, Tseq, C, H, W)
                    extra_features,       # (B, Tseq, F) or (B, F)
                    labels,               # (B,)
                    eligibility_mask      # (B,)
                ) = batch

                mean_mammogram_vec = mean_mammogram_vec.to(device)
                extra_features    = extra_features.to(device)
                labels            = labels.to(device) # 1d vector
                eligibility_mask  = eligibility_mask.to(device) # 1d vector (not used here, but it might very useful if we extend to the multi-year model)

                preds = model(mean_mammogram_vec, extra_features)
                preds = preds.squeeze(-1) # (B,1) -> (B,)

                signature = inspect.signature(criterion)
                if 'index' in signature.parameters:
                    loss = criterion(preds, labels, index=indices)
                else:
                    loss = criterion(preds, labels)
                val_loss += float(loss.item())
                n_batches += 1

                m = eligibility_mask.bool()
                if m.any():
                    all_scores.append(preds[m].detach().cpu())
                    all_labels.append(labels[m].detach().cpu())

                if verbose and (b_idx % 50 == 0):
                    print(f"[train] it={b_idx} loss={val_loss/max(1,n_batches):.5f}")
                
                if b_idx >= max_val_batches:
                    break

        avg_val_loss = val_loss / max(1, n_batches)
        validation_loss_v.append(avg_val_loss)
        epochs.append(epoch)

        # metrics on validation
        val_auc, val_ap = training_utils.calculate_metrics(all_labels, all_scores)
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f} | AUROC: {val_auc:.4f} | AP: {val_ap:.4f}")

        # --> log
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "training metrics": {"AUROC": float(train_auc), "Average Precision": float(train_ap)},
            "validation metrics": {"AUROC": float(val_auc), "Average Precision": float(val_ap)},
            "target_year": target_year
        }
        logger_.logInfo(epoch_metrics)

        # selection metric: pick what matters (AUPRC is good under imbalance)
        sel = val_ap if np.isfinite(val_ap) else (val_auc if np.isfinite(val_auc) else -np.inf)

        # -----------------------------------------------------
        # -------------------- CHECKPOINTS --------------------
        # -----------------------------------------------------
        
        # -- save every epoch
        if save_epochs:
            checkpoint_path.joinpath(model_name).mkdir(parents=False, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_ap': val_ap, 'val_auc': val_auc,
                'target_year': target_year,
            }, checkpoint_path.joinpath(model_name, f'epoch_{epoch:.0f}.pt'))

        # -- save only the best epoch (based on chosen metric)
        if sel > best_sel_metric:
            best_sel_metric = sel
            if save_best_epochs:
                checkpoint_path.joinpath(model_name).mkdir(parents=False, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_ap': val_ap, 'val_auc': val_auc,
                    'target_year': target_year,
                }, checkpoint_path.joinpath(model_name, f'best_model.pt'))

    # -- persists logs
    logger_.toParquet()

def train_singleyear_no_transformer_flatdata(config):
    """
        Train one-horizon (e.g., 2-year) model with Compositional AUC (LibAUC) + PDSCA.
        Dataloader must return 1D labels/masks (B,).
    """
    verbose = config['misc']['verbose']
    device  = config['misc']['device']
    checkpoint_path = Path(config['misc']['checkpoint_path'])
    model_name = config['misc']['model_name']
    save_epochs = config['misc']['save_epochs']
    save_best_epochs = config['misc']['save_best_epochs']
    target_year = int(config['task']['target_year'])  # e.g., 2
    use_pretrained = config['misc']['pretrained']['load']

    if verbose: print(f"device: {device} | target_year: {target_year}")

    # -- data & model
    print("load input & transform to a single year prediction & get dataloaders")
    structured_input = transform_eligibility_to_singleyear(config, target_year, load_test=True) # does load input and transform for a single year label
    train_loader, val_loader, test_loader, imratio = get_dataloaders_singleyear_flatdata(
        config, structured_input, verbose=verbose, device=device
    )

    print("instantiate model")
    # -- if a pretrained model is to be loaded
    if use_pretrained:
        model = training_utils.load_pretrained_singleyear(config)
    else:
        #model = build_model_singleyear(config, device).to(device)
        model = build_model_singleyear_without_transformer(config, device).to(device)

    # free memory
    structured_input = None
    gc.collect()

    logger_ = Logger(config)

    # -- define loss function
    print("instantiate loss function and optimizer")
    criterion, optimizer = config_loss_many(
        config, model, 
        data_len=len(train_loader.dataset),
        imratio=imratio
    )
    if criterion is None:
        raise Exception("loss function and optimizer not properly defined.")
    else:
        if verbose:
            print(type(criterion), type(optimizer))
        else:
            pass

    # ----- training state -----
    best_sel_metric = -np.inf  # model selection metric (AUROC or AUPRC)
    epochs, training_loss_v, validation_loss_v = [], [], []

    num_epochs = config['training']['epochs']
    max_train_batches = config['training']['max_training_batches_per_epoch']
    max_val_batches   = config['training']['max_validation_batches_per_epoch']
    
    # -- target learning rate for the warm-up phase
    target_lr = config['training']['learning_rate']
    warmup_steps = config['training']['warmup_steps']
    # -- when to anneal the learning rate
    epochs_to_update = {
        int(0.5*num_epochs), int(0.75*num_epochs), int(0.9*num_epochs)
    }
    opt_step_counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss, n_batches = 0.0, 0
        if verbose: print("starting training loop ...")

        # --------------------------------------------------
        # -------------------- training -------------------- 
        # --------------------------------------------------
        all_scores, all_labels = [], []
        for b_idx, batch in enumerate(train_loader):
            (
                indices,              # (B,)
                mean_mammogram_vec,    # (B, Tseq, C, H, W)
                extra_features,       # (B, Tseq, F) or (B, F)
                labels,               # (B,)
                eligibility_mask      # (B,)
            ) = batch

            mean_mammogram_vec = mean_mammogram_vec.to(device)
            extra_features    = extra_features.to(device)
            labels            = labels.to(device) # 1d vector
            eligibility_mask  = eligibility_mask.to(device) # 1d vector (not used here, but it might very useful if we extend to the multi-year model)

            optimizer.zero_grad()
            preds = model(mean_mammogram_vec, extra_features)
            preds = preds.squeeze(-1) # (B,1) -> (B,)
            #preds = model(padded_mammograms, padded_timediff, attention_mask, extra_features)
            if torch.isnan(preds).any():
                continue

            signature = inspect.signature(criterion)
            if 'index' in signature.parameters:
                loss = criterion(preds, labels, index=indices)
            else:
                loss = criterion(preds, labels)

            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())

            m = eligibility_mask.bool()
            if m.any():
                all_scores.append(preds[m].detach().cpu())
                all_labels.append(labels[m].detach().cpu())

            n_batches  += 1
            if verbose and (b_idx % 50 == 0):
                print(eligibility_mask.sum())
                print(f"[train] it={b_idx} loss={train_loss/max(1,n_batches):.5f}")
            if b_idx >= max_train_batches:
                break

        avg_train_loss = train_loss / max(1, n_batches)
        training_loss_v.append(avg_train_loss)

        # -- metrics on training
        train_auc, train_ap = training_utils.calculate_metrics(all_labels, all_scores)
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.5f} | Training AUROC: {train_auc:.4f} | Training AP: {train_ap:.4f}")

        # --------------------------------------------------
        # ------------------- validation ------------------- 
        # --------------------------------------------------
        model.eval()
        val_loss, n_batches = 0.0, 0
        all_scores, all_labels = [], []
        if verbose: print("starting validation loop ...")
        with torch.no_grad():
            for b_idx, batch in enumerate(val_loader):
                (
                    indices,              # (B,)
                    mean_mammogram_vec,    # (B, Tseq, C, H, W)
                    extra_features,       # (B, Tseq, F) or (B, F)
                    labels,               # (B,)
                    eligibility_mask      # (B,)
                ) = batch

                mean_mammogram_vec = mean_mammogram_vec.to(device)
                extra_features    = extra_features.to(device)
                labels            = labels.to(device) # 1d vector
                eligibility_mask  = eligibility_mask.to(device) # 1d vector (not used here, but it might very useful if we extend to the multi-year model)

                preds = model(mean_mammogram_vec, extra_features)
                preds = preds.squeeze(-1) # (B,1) -> (B,)

                signature = inspect.signature(criterion)
                if 'index' in signature.parameters:
                    loss = criterion(preds, labels, index=indices)
                else:
                    loss = criterion(preds, labels)
                val_loss += float(loss.item())
                n_batches += 1

                m = eligibility_mask.bool()
                if m.any():
                    all_scores.append(preds[m].detach().cpu())
                    all_labels.append(labels[m].detach().cpu())

                if verbose and (b_idx % 50 == 0):
                    print(f"[train] it={b_idx} loss={val_loss/max(1,n_batches):.5f}")
                
                if b_idx >= max_val_batches:
                    break

        avg_val_loss = val_loss / max(1, n_batches)
        validation_loss_v.append(avg_val_loss)
        epochs.append(epoch)

        # metrics on validation
        val_auc, val_ap = training_utils.calculate_metrics(all_labels, all_scores)
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f} | AUROC: {val_auc:.4f} | AP: {val_ap:.4f}")

        # --> log
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "training metrics": {"AUROC": float(train_auc), "Average Precision": float(train_ap)},
            "validation metrics": {"AUROC": float(val_auc), "Average Precision": float(val_ap)},
            "target_year": target_year
        }
        logger_.logInfo(epoch_metrics)

        # selection metric: pick what matters (AUPRC is good under imbalance)
        sel = val_ap if np.isfinite(val_ap) else (val_auc if np.isfinite(val_auc) else -np.inf)

        # -----------------------------------------------------
        # -------------------- CHECKPOINTS --------------------
        # -----------------------------------------------------
        
        # -- save every epoch
        if save_epochs:
            checkpoint_path.joinpath(model_name).mkdir(parents=False, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_ap': val_ap, 'val_auc': val_auc,
                'target_year': target_year,
            }, checkpoint_path.joinpath(model_name, f'epoch_{epoch:.0f}.pt'))

        # -- save only the best epoch (based on chosen metric)
        if sel > best_sel_metric:
            best_sel_metric = sel
            if save_best_epochs:
                checkpoint_path.joinpath(model_name).mkdir(parents=False, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_ap': val_ap, 'val_auc': val_auc,
                    'target_year': target_year,
                }, checkpoint_path.joinpath(model_name, f'best_model.pt'))

    # -- persists logs
    logger_.toParquet()

# ----------------------------------------------------------------------------------
# -------------------------------- TUNING FUNCTIONS --------------------------------
# ----------------------------------------------------------------------------------

def train_for_tuning_singleyear(config):
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
    device  = config['misc']['device']
    checkpoint_path = Path(config['misc']['checkpoint_path'])
    model_name = config['misc']['model_name']
    save_epochs = config['misc']['save_epochs']
    save_best_epochs = config['misc']['save_best_epochs']
    target_year = int(config['task']['target_year'])  # e.g., 2
    if verbose: print(f"device: {device} | target_year: {target_year}")

    # -- data & model
    print("load input & transform to a single year prediction & get dataloaders")
    structured_input = transform_eligibility_to_singleyear(config, target_year) # does load input and transform for a single year label
    train_loader, val_loader, test_loader, imratio = get_dataloaders_singleyear(
        config, structured_input, device, verbose=verbose
    )
    print("instatiate model")
    model = build_model_singleyear(config, device).to(device)

    # free memory
    structured_input = None
    gc.collect()

    # -- define loss function
    print("instantiate loss function and optimizer")
    criterion, optimizer = config_loss(
        config, model, 
        data_len=len(train_loader.dataset) 
    )
    if criterion is None:
        raise Exception("loss function and optimizer not properly defined.")
    else:
        if verbose:
            print(type(criterion), type(optimizer))
        else:
            pass
    masked_criterion = make_masked_loss(criterion)

    # -- metric used for assessment during optimization
    best_auroc_metric = -np.inf
    best_ap_metric = -np.inf
    #auroc_metric_list, ap_metric_list = [], []
    epochs, training_loss_v, validation_loss_v = [], [], []

    num_epochs = config['training']['epochs']
    max_train_batches = config['training']['max_training_batches_per_epoch']
    max_val_batches   = config['training']['max_validation_batches_per_epoch']
    
    # -- training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss, n_batches = 0.0, 0
        if verbose: print("start training loop")

        for b_idx, batch in enumerate(train_loader):
            (
                indices,              # (B,)
                mammogram_id,
                padded_mammograms,    # (B, Tseq, C, H, W)
                padded_timediff,      # (B, Tseq)
                attention_mask,       # (B, Tseq)
                extra_features,       # (B, Tseq, F) or (B, F)
                labels,               # (B,)
                eligibility_mask      # (B,)
            ) = batch

            # device
            padded_mammograms = padded_mammograms.to(device)
            padded_timediff   = padded_timediff.to(device).unsqueeze(-1)
            attention_mask    = attention_mask.to(device)
            extra_features    = extra_features.to(device)
            labels            = labels.to(device) # 1d vector
            eligibility_mask  = eligibility_mask.to(device) # 1d vector

            optimizer.zero_grad()
            preds = model(padded_mammograms, padded_timediff, attention_mask, extra_features)
            if torch.isnan(preds).any():
                continue

            loss = masked_criterion(preds, labels, eligibility_mask, index=indices)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            n_batches  += 1
            #if verbose and (b_idx % 100 == 0):
            #    print(f"[train] it={b_idx} loss={total_loss/max(1,n_batches):.5f}")
            if b_idx >= max_train_batches:
                break

        avg_train_loss = total_loss / max(1, n_batches)
        training_loss_v.append(avg_train_loss)

        # ---- validation ----
        model.eval()
        val_loss, nb = 0.0, 0
        all_scores, all_labels = [], []
        with torch.no_grad():
            for b_idx, batch in enumerate(val_loader):
                (
                    indices,
                    mammogram_id,
                    padded_mammograms,
                    padded_timediff,
                    attention_mask,
                    extra_features,
                    labels,
                    eligibility_mask
                ) = batch

                padded_mammograms = padded_mammograms.to(device)
                padded_timediff   = padded_timediff.to(device).unsqueeze(-1)
                attention_mask    = attention_mask.to(device)
                extra_features    = extra_features.to(device)
                labels            = labels.to(device)
                eligibility_mask  = eligibility_mask.to(device)

                preds = model(padded_mammograms, padded_timediff, attention_mask, extra_features)

                loss = masked_criterion(preds, labels, eligibility_mask, index=indices)
                val_loss += float(loss.item()); nb += 1

                m = eligibility_mask.bool()
                if m.any():
                    all_scores.append(preds[m].detach().cpu())
                    all_labels.append(labels[m].detach().cpu())

                if b_idx >= max_val_batches:
                    break

        avg_val_loss = val_loss / max(1, nb)
        validation_loss_v.append(avg_val_loss)
        epochs.append(epoch)
            
        # metrics on masked validation
        if len(all_labels) > 0:
            y_true  = torch.cat(all_labels).numpy()
            y_score = torch.cat(all_scores).numpy()
            if len(np.unique(y_true)) > 1:
                val_auc = roc_auc_score(y_true, y_score)
                val_ap  = average_precision_score(y_true, y_score)
                #try:
                #    val_brier = brier_score_loss(y_true, y_score)
                #except Exception:
                #    val_brier = np.nan
            else:
                val_auc, val_ap = np.nan, np.nan
        else:
            val_auc, val_ap = np.nan, np.nan

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f} | AUROC: {val_auc:.4f} | AP: {val_ap:.4f}")

        # -- calculate metric for optimization
        #year_weights = [0.1, 0.15, 0.3, 0.3, 0.15]
        #current_auroc_metric = np.sum( [ year_weights[index]*cur_metrics["AUROC"][index] for index in range(5) ] )
        #current_ap_metric = np.sum( [ year_weights[index]*cur_metrics["Average Precision"][index] for index in range(5) ] )
        #auroc_metric_list.append(current_auroc_metric)
        #ap_metric_list.append(current_ap_metric)

        if val_auc>best_auroc_metric:
            best_auroc_metric = val_auc
            best_ap_metric = val_ap

        if epoch>=4 and val_auc<0.55:
            break
    return best_ap_metric, best_auroc_metric
