import gc
import torch
import inspect
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Optional
from collections import defaultdict

import mlflow
import mlflow.pytorch

from hapcancer.model.dataload.load_input import InputLoader
from hapcancer.model.architecture.prediction_model import build_model_singleyear_with_tfidf
from hapcancer.model.loss.config_loss import config_loss_many
from hapcancer.model.training.utils import calculate_metrics, set_outer_lr
from hapcancer.etl.utils import batching_parquet_file

from hapcancer.logger import Logger
from hapcancer.config_manager import ConfigInterface


# ------------------------------------------------------------------- #
# --------------------- TRAINING CORE BEHAVIOR ---------------------- #
# ------------------------------------------------------------------- #

class TrainingBase(ConfigInterface):
    '''

    '''
    def __init__(self, config_dir: str, config_defaults: dict):
        super().__init__(config_dir, config_defaults)
        self.device = self.extra_cfg["extra"]["device"]
        self.verbose = self.extra_cfg["extra"]["verbose"]
        self.save_epochs = self.extra_cfg["extra"]['save_epochs']
        self.save_best_epochs = self.extra_cfg["extra"]['save_best_epochs']
        self.model_name = self.training_cfg['training']['model_name']
        # -- if a pretrained model is to be loaded
        self.load_pretrained = self.training_cfg['training']['pretrained']['load']
        self.pretrained_checkpoint_name = self.training_cfg['training']['pretrained']['model_name']
        self.pretrained_file_name = self.training_cfg['training']['pretrained']['file_name']
        
        # -- training specs
        self.num_epochs = self.training_cfg['training']['epochs']
        self.max_train_batches = self.training_cfg['training']['max_training_batches_per_epoch']
        self.max_val_batches = self.training_cfg['training']['max_validation_batches_per_epoch']
        self.target_lr = self.training_cfg['training']['learning_rate']
        self.warmup_steps = self.training_cfg['training']['warmup_steps']
        self.loss_function_nm = self.training_cfg['training']['loss_function']

        # -- tuning config
        self.study_name = None
        if self.tuning_cfg is not None:
            self.study_name = self.tuning_cfg["tuning"]["study_name"]

        # -- training metrics
        self.best_sel_metric = -np.inf
        self.best_auc_metric = -np.inf
        self.best_ap_metric = -np.inf
        self.epochs = []
        self.training_loss_per_epoch = []
        self.validation_loss_per_epoch = []
        # -- when to anneal the learning rate
        self.epochs_to_update_lr = None

        # -- loaders
        self.structured_input = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.imratio = None

        # -- model
        self.model = None

        # -- loss and optimizer
        self.criterion = None
        self.optimizer = None
        self._logger = None

    def _log_info(
        self,
        epoch: int,
        target_year: int,
        training_metrics: dict,
        validation_metrics: dict
    ):
        epoch_metrics = {
            "config_dir_path": str(self.cfg_manager.config_dir),
            "split_config_name": self.cfg_manager["split"],
            "training_config_name": self.cfg_manager["training_experiments"],
            "epoch": epoch,
            "target year": target_year,
            "training metrics": training_metrics,
            "validation metrics": validation_metrics,
        }
        self._logger.log_info(epoch_metrics)

    def _batch_to_embeddings(self, mammogram_ids):
        arr = np.vstack(self.input_loader.get_embeddings(list(mammogram_ids)))
        return torch.tensor(arr, dtype=torch.float16)

    def _get_current_lr(self):
        if self.loss_function_nm == "cross_entropy":
            return float(self.optimizer.param_groups[0]["lr"])
        return float(self.optimizer.lr)

    def _select_metric(self, val_auc: float, val_ap: float) -> float:
        return val_auc if np.isfinite(val_auc) else (val_ap if np.isfinite(val_ap) else -np.inf)

    def _reset_best_metrics_for_fold(self):
        self.best_sel_metric = -np.inf
        self.best_auc_metric = -np.inf
        self.best_ap_metric = -np.inf

    def _maybe_save_checkpoints(self, epoch: int, fold_ix: int, sel: float):
        # -- epoch checkpoint
        if self.save_epochs:
            if fold_ix is None:
                fname = f"epoch_{epoch:.0f}.pt"
            else:
                fname = f"epoch_{epoch:.0f}_fold_{fold_ix:.0f}.pt"

            self.checkpoint_path.joinpath(self.model_name).mkdir(parents=False, exist_ok=True)
            ckpt_path = self.checkpoint_path.joinpath(self.model_name, fname)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                },
                ckpt_path,
            )

        # -- best checkpoint (based on selected metric)
        if sel > self.best_sel_metric:
            self.best_sel_metric = sel
            if self.save_best_epochs:
                if fold_ix is None:
                    fname = f"best_model_epoch_{epoch:.0f}.pt"
                else:
                    fname = f"best_model_fold_{fold_ix:.0f}.pt"
                self.checkpoint_path.joinpath(self.model_name).mkdir(parents=False, exist_ok=True)
                ckpt_path = self.checkpoint_path.joinpath(self.model_name, fname)
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                    },
                    ckpt_path,
                )
                # ── MLflow: log best checkpoint as artifact ──────────────────
                if mlflow.active_run():
                    mlflow.log_artifact(str(ckpt_path), artifact_path="checkpoints/best")

    def _run_epoch(
        self,
        loader,
        is_train: bool,
        epoch: int,
        opt_step_counter: int,
        max_batches: int,
    ):
        if is_train:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        n_batches = 0
        scores, labels_all = [], []

        # grad only in training
        ctx = torch.enable_grad() if is_train else torch.no_grad()
        with ctx:
            for b_idx, batch in enumerate(loader):
                (
                    indices,              # (B,)
                    mammogram_ids,
                    extra_features,       # (B, Tseq, F) or (B, F)
                    labels,               # (B,)
                    eligibility_mask      # (B,)
                ) = batch

                mean_mammogram_vec = self._batch_to_embeddings(mammogram_ids)
                if is_train:
                    self.optimizer.zero_grad()
                preds = self.model(mean_mammogram_vec, extra_features).squeeze(-1)
                if torch.isnan(preds).any(): continue

                loss = None
                if self.loss_function_nm != "cross_entropy":
                    if is_train:
                        loss = self.criterion(preds, labels, index=indices)
                    else:
                        loss = None
                else:
                    loss = self.criterion(preds, labels)
                if loss is not None:
                    total_loss += float(loss.item())

                if is_train:
                    if loss is not None:
                        loss.backward()
                        self.optimizer.step()
                    # -- warm-up
                    opt_step_counter += 1
                    if opt_step_counter <= self.warmup_steps:
                        set_outer_lr(
                            self.optimizer,
                            opt_step_counter,
                            self.warmup_steps,
                            self.target_lr,
                            verbose=False,
                        )

                m = eligibility_mask.bool()
                if m.any():
                    scores.append(preds[m].detach().cpu())
                    labels_all.append(labels[m].detach().cpu())

                n_batches += 1
                if b_idx >= max_batches:
                    break

        avg_loss = total_loss / max(1, n_batches)
        auc, ap = calculate_metrics(labels_all, scores)
        lr = self._get_current_lr()
        out = {"loss": avg_loss, "auc": auc, "ap": ap, "lr": lr}
        return out, opt_step_counter


# ------------------------------------------------------------------- #
# ----------------------- TRAINING SPECIFICS ------------------------ #
# ------------------------------------------------------------------- #

class TrainingSingleYearTFIDF(TrainingBase):
    def __init__(self, config_dir: str, config_defaults: dict):
        super().__init__(config_dir, config_defaults)
        self.bce_optimizer = self.training_cfg['training']['optimizer']
        self.input_loader = InputLoader(config_dir, config_defaults)

    # ── MLflow helper: collect all hyperparams from training_cfg ─────────
    def _get_mlflow_params(self, target_year: int) -> dict:
        t = self.training_cfg['training']
        m = self.training_cfg.get('model', {})
        mlp = m.get('mlp_config', {})
        return {
            "model_name":        self.model_name,
            "target_year":       target_year,
            "epochs":            t.get('epochs'),
            "learning_rate":     t.get('learning_rate'),
            "weight_decay":      t.get('weight_decay'),
            "warmup_steps":      t.get('warmup_steps'),
            "loss_function":     t.get('loss_function'),
            "optimizer":         t.get('optimizer'),
            "max_train_batches": t.get('max_training_batches_per_epoch'),
            "max_val_batches":   t.get('max_validation_batches_per_epoch'),
            "mlp_dropout":       mlp.get('dropout'),
            "mlp_activation":    mlp.get('activation'),
            "mlp_hidden_layers": str(mlp.get('hidden_layers')),
            "device":            self.device,
        }

    def _load_pretrained_model(self):
        loaded_state = torch.load(
            self.checkpoint_path.joinpath(self.pretrained_checkpoint_name, self.pretrained_file_name), weight_only=True, map=self.device
        )
        self.model.load_state_dict(loaded_state['model_state_dict'])
    
    def _build_model(self):
        self.model = build_model_singleyear_with_tfidf(self.training_cfg, self.device).to(self.device)
        if self.load_pretrained:
            self._load_pretrained_model()

    def init_logger(self, mode: str):
        match mode:
            case 'tuning':
                self._logger = None
            case 'full':
                self._logger = Logger(self.model_logging_path, self.model_name+'_full')
            case _:
                self._logger = Logger(self.model_logging_path, self.model_name)

    def _training_cv_loop(
        self,
        split_dict: dict, 
        target_year: Optional[int] = 5,
        tuning: Optional[bool] = False,
        verbose: Optional[bool] = False,
        remove_text: Optional[bool] = False
    ):
        if tuning:
            self.init_logger('tuning')
        else:
            self.init_logger('any')

        fold_keys = [ k for k, v in split_dict.items() if k!="test" ]
        fold_eval = { k: [] for k in fold_keys }

        for fold_ix, fold_key in enumerate(fold_keys):
            print(f"Fold {fold_ix}:")

            # -- reset the best metrics for each fold
            self.best_sel_metric = -np.inf
            self.best_auc_metric = -np.inf
            self.best_ap_metric = -np.inf

            # ── MLflow: one run per fold (skipped during tuning) ─────────
            run_name = f"{self.model_name}_year{target_year}_fold{fold_ix}"
            mlflow_ctx = (
                mlflow.start_run(run_name=run_name, nested=tuning)
                if not tuning
                else _nullcontext()
            )

            with mlflow_ctx:
                if not tuning:
                    mlflow.log_params(self._get_mlflow_params(target_year))
                    mlflow.set_tag("fold", fold_ix)
                    mlflow.set_tag("mode", "cv_training")

                # --------------------------------------------
                # 1. Split configuration
                # --------------------------------------------
                self._build_model()
                train_ids = split_dict[fold_key]['train']
                validation_ids = split_dict[fold_key]['validation']
                train_loader, train_imratio = self.input_loader.get_dataloader(
                    train_ids, target_year=target_year, is_training=True
                )
                val_loader, val_imratio = self.input_loader.get_dataloader(
                    validation_ids, target_year=target_year, is_training=False
                )

                self.criterion, self.optimizer = config_loss_many(
                    self.training_cfg, self.model,
                    device=self.device, 
                    data_len=len(train_loader.dataset),
                    imratio=train_imratio,
                    bce_optimizer=self.bce_optimizer
                )

                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode='max',
                    factor=0.5,
                    patience=3,
                    min_lr=1e-6,
                )

                # ----------------------------------------------
                # 2. Start the training loop
                # ----------------------------------------------
                opt_step_counter = 0
                for epoch in tqdm(range(self.num_epochs)):
                    self.model.train()
                    train_loss, n_batches = 0.0, 0
                    scores, cur_labels = [], []

                    for b_idx, batch in enumerate(train_loader):
                        (
                            indices,
                            mammogram_ids,
                            extra_features,
                            labels,
                            eligibility_mask
                        ) = batch
                        mean_mammogram_vec = torch.tensor(np.vstack(self.input_loader.get_embeddings(list(mammogram_ids))), dtype=torch.float16)
                        if remove_text: # for ablation
                            mean_mammogram_vec = torch.zeros(mean_mammogram_vec.shape, dtype=torch.float16)

                        self.optimizer.zero_grad()
                        preds = self.model(mean_mammogram_vec, extra_features)
                        preds = preds.squeeze(-1)
                        if torch.isnan(preds).any():
                            continue

                        if self.loss_function_nm!='cross_entropy':
                            loss = self.criterion(preds, labels, index=indices)
                        else:
                            loss = self.criterion(preds, labels)

                        loss.backward()
                        self.optimizer.step()
                        train_loss += float(loss.item())

                        opt_step_counter+=1
                        if opt_step_counter<=self.warmup_steps:
                            set_outer_lr(self.optimizer, opt_step_counter, self.warmup_steps, self.target_lr, verbose=False)

                        m = eligibility_mask.bool()
                        if m.any():
                            scores.append(preds[m].detach().cpu())
                            cur_labels.append(labels[m].detach().cpu())

                        n_batches += 1
                        if b_idx >= self.max_train_batches:
                            break

                    avg_train_loss = train_loss / max(1, n_batches)
                    self.training_loss_per_epoch.append(avg_train_loss)

                    train_auc, train_ap = calculate_metrics(cur_labels, scores)
                    if verbose:
                        print(f"Epoch {epoch+1}/{self.num_epochs} | Train Loss: {avg_train_loss:.5f} | Training AUROC: {train_auc:.4f} | Training AP: {train_ap:.4f}")

                    # VALIDATION
                    self.model.eval()
                    val_loss, n_batches = 0.0, 0
                    scores, cur_labels = [], []
                    with torch.no_grad():
                        for b_idx, batch in enumerate(val_loader):
                            (
                                indices,
                                mammogram_ids,
                                extra_features,
                                labels,
                                eligibility_mask
                            ) = batch

                            mean_mammogram_vec = torch.tensor(np.vstack(self.input_loader.get_embeddings(list(mammogram_ids))), dtype=torch.float16)
                            if remove_text: # for ablation
                                mean_mammogram_vec = torch.zeros(mean_mammogram_vec.shape, dtype=torch.float16)
                            preds = self.model(mean_mammogram_vec, extra_features)
                            preds = preds.squeeze(-1)

                            if self.loss_function_nm!='cross_entropy':
                                pass
                            else:
                                loss = self.criterion(preds, labels)
                                val_loss += float(loss.item())

                            n_batches += 1
                            m = eligibility_mask.bool()
                            if m.any():
                                scores.append(preds[m].detach().cpu())
                                cur_labels.append(labels[m].detach().cpu())

                            if b_idx >= self.max_val_batches:
                                break

                    avg_val_loss = val_loss / max(1, n_batches)
                    self.validation_loss_per_epoch.append(avg_val_loss)
                    self.epochs.append(epoch)

                    val_auc, val_ap = calculate_metrics(cur_labels, scores)
                    if self.loss_function_nm=="cross_entropy":
                        dummy_lr = self.optimizer.param_groups[0]['lr']
                    else:
                        dummy_lr = self.optimizer.lr
                    
                    if verbose:
                        print(f"CV: {fold_ix} | Epoch {epoch+1}/{self.num_epochs} | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f} | AUROC: {val_auc:.4f} | AP: {val_ap:.4f} | LR: {dummy_lr:.6f}")

                    scheduler.step(val_auc)

                    # ── MLflow: log per-epoch metrics ─────────────────────
                    if not tuning and mlflow.active_run():
                        mlflow.log_metrics({
                            "train/loss":  avg_train_loss,
                            "train/auroc": float(train_auc),
                            "train/ap":    float(train_ap),
                            "val/loss":    avg_val_loss,
                            "val/auroc":   float(val_auc),
                            "val/ap":      float(val_ap),
                            "lr":          float(dummy_lr),
                        }, step=epoch)

                    # -- log epoch info (existing custom logger)
                    training_metrics = {  
                        "CV": fold_ix, "Loss": avg_train_loss, "AUROC": float(train_auc), "Average Precision": float(train_ap)
                    }
                    validation_metrics = {
                        "CV": fold_ix, "Loss": avg_val_loss, "AUROC": float(val_auc), "Average Precision": float(val_ap), "Learning Rate": float(dummy_lr)
                    }
                    if not tuning: self._log_info(epoch, target_year, training_metrics, validation_metrics)

                    sel = val_auc if np.isfinite(val_auc) else (val_ap if np.isfinite(val_ap) else -np.inf)

                    if tuning:
                        if val_auc > self.best_auc_metric:
                            self.best_auc_metric = val_auc
                        if val_ap > self.best_ap_metric:
                            self.best_ap_metric = val_ap

                    # CHECKPOINTS
                    if not tuning:
                        if self.save_epochs:
                            self.checkpoint_path.joinpath(self.model_name).mkdir(parents=False, exist_ok=True)
                            torch.save({
                                'epoch': epoch, 'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict()
                            }, self.checkpoint_path.joinpath(self.model_name, f'epoch_{epoch:.0f}_fold_{fold_ix:.0f}.pt'))

                        if sel > self.best_sel_metric:
                            self.best_sel_metric = sel
                            if self.save_best_epochs:
                                best_ckpt = self.checkpoint_path.joinpath(self.model_name, f'best_model_fold_{fold_ix:.0f}.pt')
                                self.checkpoint_path.joinpath(self.model_name).mkdir(parents=False, exist_ok=True)
                                torch.save({
                                    'epoch': epoch, 'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict()
                                }, best_ckpt)
                                # ── MLflow: log best model artifact ──────────────
                                if mlflow.active_run():
                                    mlflow.log_artifact(str(best_ckpt), artifact_path="checkpoints/best")

                # ── MLflow: summary metrics at run end ────────────────────
                if not tuning and mlflow.active_run():
                    mlflow.log_metrics({
                        "best_val_auroc": float(self.best_auc_metric),
                        "best_val_ap":    float(self.best_ap_metric),
                    })
                    mlflow.pytorch.log_model(self.model, artifact_path="model")

            if tuning and fold_ix==0:
                break
                    
    def _cross_validation_loop(
        self,
        split_dict: dict, 
        target_year: Optional[int] = 5,
        tuning: Optional[bool] = False,
        verbose: Optional[bool] = False
    ) -> None:

        if tuning:
            self.init_logger('tuning')
        else:
            self.init_logger('any')

        fold_keys = [k for k in split_dict.keys() if k != "test"]
        for fold_ix, fold_key in enumerate(fold_keys):
            self._reset_best_metrics_for_fold()

            run_name = f"{self.model_name}_year{target_year}_fold{fold_ix}"
            mlflow_ctx = (
                mlflow.start_run(run_name=run_name, nested=tuning)
                if not tuning
                else _nullcontext()
            )

            with mlflow_ctx:
                if not tuning:
                    mlflow.log_params(self._get_mlflow_params(target_year))
                    mlflow.set_tag("fold", fold_ix)
                    mlflow.set_tag("mode", "cv_training")

                self._build_model()
                train_ids, val_ids = split_dict[fold_key]["train"], split_dict[fold_key]["validation"]
                train_loader, train_imratio = self.input_loader.get_dataloader(
                    train_ids, target_year=target_year, is_training=True
                )
                val_loader, _ = self.input_loader.get_dataloader(
                    val_ids, target_year=target_year, is_training=False
                )
                self.criterion, self.optimizer = config_loss_many(
                    self.training_cfg,
                    self.model,
                    device=self.device,
                    data_len=len(train_loader.dataset),
                    imratio=train_imratio,
                    bce_optimizer=self.bce_optimizer,
                )
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode="max",
                    factor=0.5,
                    patience=3,
                    min_lr=1e-6,
                )
                opt_step_counter = 0
                for epoch in tqdm(range(self.num_epochs)):
                    train_out, opt_step_counter = self._run_epoch(
                        loader=train_loader,
                        is_train=True,
                        epoch=epoch,
                        opt_step_counter=opt_step_counter,
                        max_batches=self.max_train_batches,
                    )
                    val_out, _ = self._run_epoch(
                        loader=val_loader,
                        is_train=False,
                        epoch=epoch,
                        opt_step_counter=opt_step_counter,
                        max_batches=self.max_val_batches,
                    )
                    self.training_loss_per_epoch.append(train_out["loss"])
                    self.validation_loss_per_epoch.append(val_out["loss"])
                    self.epochs.append(epoch)
                    if verbose:
                        print(
                            f"CV: {fold_ix} | Epoch {epoch+1}/{self.num_epochs} | "
                            f"Train Loss: {train_out['loss']:.5f} | "
                            f"Val Loss: {val_out['loss']:.5f} | "
                            f"Train AUROC: {train_out['auc']:.4f} | AP: {train_out['ap']:.4f} | "
                            f"AUROC: {val_out['auc']:.4f} | AP: {val_out['ap']:.4f} | "
                            f"LR: {val_out['lr']:.6f}"
                        )
                    scheduler.step(val_out["auc"])
                    if not tuning:
                        self._log_info(
                            epoch,
                            target_year,
                            training_metrics={
                                "CV": fold_ix,
                                "Loss": train_out["loss"],
                                "AUROC": float(train_out["auc"]),
                                "Average Precision": float(train_out["ap"]),
                            },
                            validation_metrics={
                                "CV": fold_ix,
                                "Loss": val_out["loss"],
                                "AUROC": float(val_out["auc"]),
                                "Average Precision": float(val_out["ap"]),
                                "Learning Rate": float(val_out["lr"]),
                            },
                        )

                    # ── MLflow: log per-epoch metrics ─────────────────────
                    if not tuning and mlflow.active_run():
                        mlflow.log_metrics({
                            "train/loss":  train_out["loss"],
                            "train/auroc": float(train_out["auc"]),
                            "train/ap":    float(train_out["ap"]),
                            "val/loss":    val_out["loss"],
                            "val/auroc":   float(val_out["auc"]),
                            "val/ap":      float(val_out["ap"]),
                            "lr":          float(val_out["lr"]),
                        }, step=epoch)

                    sel = self._select_metric(val_auc=val_out["auc"], val_ap=val_out["ap"])
                    if tuning:
                        self.best_auc_metric = max(self.best_auc_metric, val_out["auc"])
                        self.best_ap_metric = max(self.best_ap_metric, val_out["ap"])
                    else:
                        self._maybe_save_checkpoints(
                            epoch=epoch,
                            fold_ix=fold_ix,
                            sel=sel,
                        )

                # ── MLflow: summary at run end ────────────────────────────
                if not tuning and mlflow.active_run():
                    mlflow.log_metrics({
                        "best_val_auroc": float(self.best_auc_metric),
                        "best_val_ap":    float(self.best_ap_metric),
                    })
                    mlflow.pytorch.log_model(self.model, artifact_path="model")

            if tuning:
                break

    def _full_training(
        self,
        split_dict: dict, 
        target_year: Optional[int] = 5,
        verbose: Optional[bool] = False
    ) -> None:
        self.init_logger('full')
        fold_dummy = [k for k in split_dict.keys() if k != "test"][0]
        self._reset_best_metrics_for_fold()
        self._build_model()

        train_ids = split_dict[fold_dummy]["train"]
        val_ids = split_dict[fold_dummy]["validation"]
        test_ids = split_dict["test"]
        train_loader, train_imratio = self.input_loader.get_dataloader(
            train_ids+val_ids, target_year=target_year, is_training=True
        )
        test_loader, _ = self.input_loader.get_dataloader(
            test_ids, target_year=target_year, is_training=False
        )
        self.criterion, self.optimizer = config_loss_many(
            self.training_cfg,
            self.model,
            device=self.device,
            data_len=len(train_loader.dataset),
            imratio=train_imratio,
            bce_optimizer=self.bce_optimizer,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.num_epochs, eta_min=1e-6
        )

        run_name = f"{self.model_name}_year{target_year}_full"
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(self._get_mlflow_params(target_year))
            mlflow.set_tag("mode", "full_training")

            opt_step_counter = 0
            for epoch in tqdm(range(self.num_epochs)):
                train_out, opt_step_counter = self._run_epoch(
                    loader=train_loader,
                    is_train=True,
                    epoch=epoch,
                    opt_step_counter=opt_step_counter,
                    max_batches=self.max_train_batches,
                )
                self.training_loss_per_epoch.append(train_out["loss"])
                self.epochs.append(epoch)
                if verbose:
                    print(
                        f"Epoch {epoch+1}/{self.num_epochs} | "
                        f"Train Loss: {train_out['loss']:.5f} | "
                        f"LR: {train_out['lr']:.6f}"
                    )
                scheduler.step()
                self._log_info(
                    epoch,
                    target_year,
                    training_metrics={
                        "Loss": train_out["loss"],
                        "AUROC": float(train_out["auc"]),
                        "Average Precision": float(train_out["ap"]),
                        "Learning Rate": float(train_out["lr"])
                    },
                    validation_metrics={}
                )

                # ── MLflow: log per-epoch metrics ─────────────────────────
                mlflow.log_metrics({
                    "train/loss":  train_out["loss"],
                    "train/auroc": float(train_out["auc"]),
                    "train/ap":    float(train_out["ap"]),
                    "lr":          float(train_out["lr"]),
                }, step=epoch)

                self.save_epochs = True
                self._maybe_save_checkpoints(epoch=epoch, fold_ix=None, sel=-np.inf)

            # ── MLflow: log final model ───────────────────────────────────
            mlflow.pytorch.log_model(self.model, artifact_path="model")

        self.save_epochs = self.extra_cfg["extra"]['save_epochs']
    
    def train(
        self,
        split_dict: dict,
        target_year: Optional[int] = 5,
        tuning: Optional[bool] = False,
        verbose: Optional[bool] = False,
        remove_text: Optional[bool] = False # for ablation
    ):
        print("training ...")
        self._training_cv_loop(split_dict, target_year, tuning=tuning, verbose=True, remove_text=remove_text)

        if tuning:
            return self.best_ap_metric, self.best_auc_metric

    def full_training(
        self,
        split_dict: dict,
        target_year: Optional[int] = 5,
        verbose: Optional[bool] = False
    ) -> None:
        self._full_training(split_dict, target_year, verbose=verbose)


# ── tiny helper so tuning paths can skip the mlflow context manager ──
from contextlib import contextmanager

@contextmanager
def _nullcontext():
    yield