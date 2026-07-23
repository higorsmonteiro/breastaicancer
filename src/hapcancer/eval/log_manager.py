import json
import yaml
import torch
import random
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Any, Optional, List, Dict, Tuple, Sequence

from captum.attr import IntegratedGradients, FeatureAblation

from hapcancer.config_manager import ConfigInterface
from hapcancer.schemas.enums import ConfigFolderNames
from hapcancer.model.dataload.load_input import InputLoader, DatasetSplit
from hapcancer.model.architecture.prediction_model import build_model_singleyear_with_tfidf
from hapcancer.schemas.logs import TuningBCELog, TrainingBCELog
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

def load_log(
    log_path: str,
    config: dict,
    mode: str
) -> List[dict]:
    log_path = Path(log_path)
    match mode:
        case 'tuning':
            study_name = config['tuning']['study_name']
            log_filename = log_path.joinpath(study_name)
        case 'training':
            model_name = config['training']['model_name']
            log_filename = log_path.joinpath(model_name)
        case _:
            raise ValueError(f"'mode' only options are: 'training' or 'tuning'.")
    with open(log_filename.joinpath("metrics.jsonl"), "r") as f:
        lines = [ json.loads(line) for line in f ]
    return lines

# ------------------------------------------------------------ #
# ----------------------- TUNING LOGS ------------------------ #
# ------------------------------------------------------------ #

def get_tuning_log(
    tuning_path: str, 
    tuning_config: dict, 
    strat_str: str, 
    target_year: int,
    split_str: Optional[str] = None
) -> List[dict]:
    lines = load_log(tuning_path, tuning_config, mode="tuning")
    if split_str is None:
        return [
            line for line in lines if line['dataset_name']==strat_str and line['target_year']==target_year
        ]
    else:
        return [
            line for line in lines if line['dataset_name']==strat_str and line['target_year']==target_year and line['split_config_name']==split_str
        ]

def get_tuning_results(
    tuning_path: str, 
    tuning_config: dict, 
    strat_str: str, 
    target_year: int,
    split_str: Optional[str] = None
) -> List[float]:
    lines = get_tuning_log(tuning_path, tuning_config, strat_str, target_year, split_str)
    results = [ elem['result'] for elem in lines ]
    return results

def get_trial_params(
    trial_number: int, 
    tuning_path: str, 
    tuning_config: dict, 
    strat_str: str, 
    target_year: int,
    split_str: Optional[str] = None
) -> Tuple[int, dict, float]:
    lines = get_tuning_log(tuning_path, tuning_config, strat_str, target_year, split_str)
    params = [ (elem['trial_number'], elem['params'], elem['result']) for elem in lines if elem['trial_number']==trial_number ]
    return params[0] if len(params) else []

def get_best_trial_params(
    tuning_path: str, 
    tuning_config: dict, 
    strat_str: str, 
    target_year: int,
    split_str: Optional[str] = None
) -> Tuple[int, dict, float]:
    lines = get_tuning_log(tuning_path, tuning_config, strat_str, target_year, split_str)
    results = [ (elem['trial_number'], elem['params'], elem['result']) for elem in lines ]
    if results:
        results = sorted(results, key=lambda x: x[2])[::-1][0]
        return results
    return ()

def get_trial_info(
    trial_number: int, 
    tuning_path: str, 
    tuning_config: dict, 
    strat_str: str, 
    target_year: int,
    split_str: Optional[str] = None
) -> dict:
    lines = get_tuning_log(tuning_path, tuning_config, strat_str, target_year, split_str)
    info = [ elem for elem in lines if elem['trial_number']==trial_number ]
    return info[0] if len(info) else []

class TuningLogManager(ConfigInterface):
    '''
        Interface to make easier the handling of the output of tuning experiments.
    '''
    def __init__(self, config_dir: str, config_defaults: dict):
        super().__init__(config_dir, config_defaults)
        self.config_dir = config_dir
        self.config_defaults = config_defaults

        self._filename_ = None

    def validate_logs(self, line):
        TuningBCELog.model_validate(line)

    def list_tuning_files(self) -> List[str]:
        cfg_model_foldername = ConfigFolderNames.model_foldername.value
        cfg_tuning_foldername = ConfigFolderNames.tuning_foldername.value
        tuning_exp_files = list(Path(self.config_dir).joinpath(cfg_model_foldername, cfg_tuning_foldername).glob("*"))
        return tuning_exp_files

    def get_results(self, strat_str: str, target_year: int, split_str: Optional[str] = None):
        return get_tuning_results(self.tuning_path, self.tuning_cfg, strat_str, target_year, split_str)

    def get_best_trial_params(self, strat_str: str, target_year: int, split_str: Optional[str] = None):
        return get_best_trial_params(self.tuning_path, self.tuning_cfg, strat_str, target_year, split_str)

    def get_trial_params(self, trial_number: int, strat_str: str, target_year: int, split_str: Optional[str] = None):
        return get_trial_params(trial_number, self.tuning_path, self.tuning_cfg, strat_str, target_year, split_str)

    def get_trial_info(self, trial_number: int, strat_str: str, target_year: int, split_str: Optional[str] = None):
        return get_trial_info(trial_number, self.tuning_path, self.tuning_cfg, strat_str, target_year, split_str)

    def define_training_config(self, trial_number: int, strat_str: str, target_year: int, split_str: Optional[str] = None) -> dict:
        if self.training_cfg is None:
            raise Exception("No base training configuration provided.")

        base_config = dict(self.training_cfg)
        cur_trial = self.get_trial_params(trial_number, strat_str, target_year, split_str)
        params = cur_trial[1]
        base_config['model']['mlp_config']['dropout'] = params['dropout']
        base_config['training']['weight_decay'] = params['weight_decay']
        base_config['training']['learning_rate'] = params['learning_rate']
        base_config['training']['optimizer'] = params['optimizer']
        base_config['training']['epochs'] = 25
        
        depth = params['depth']
        base_config['model']['mlp_config']['hidden_layers'] = [ params[f'layer_{ix}_units'] for ix in range(depth) ]
        return base_config

    def get_model_name(self, trial_number: int, strat_str: str, target_year: int, split_str: Optional[str] = None) -> str:
        trial_params = self.get_trial_info(trial_number, strat_str, target_year, split_str)
        experiment_id_base = f'{trial_number}-{strat_str}-{target_year}-{trial_params["config_dir_path"]}-{trial_params["tuning_config_name"]}-{trial_params["split_config_name"]}'
        experiment_id = hashlib.sha1(experiment_id_base.encode('utf-8')).hexdigest()[:16]
        filename = f"trial_{trial_number}_{target_year}yr_id_{experiment_id}"
        return filename

    def persist_training_config(self, trial_number: int, strat_str: str, target_year: int, split_str: Optional[str] = None) -> None:
        trial_params = self.get_trial_info(trial_number, strat_str, target_year, split_str)
        base_config = self.define_training_config(trial_number, strat_str, target_year, split_str)
        base_config['description'] = f'trial number:{trial_number};' \
        f'strat:{strat_str};' \
        f'target year:{target_year};' \
        f'configuration directory:{trial_params["config_dir_path"]};' \
        f'tuning configuration file:{trial_params["tuning_config_name"]};' \
        f'split configuration file:{trial_params["split_config_name"]}'

        self._filename_ = self.get_model_name(trial_number, strat_str, target_year, split_str) + ".yml"
        path_to = Path(trial_params["config_dir_path"]).joinpath("model", "training_experiments")

        base_config['training']['model_name'] = Path(self._filename_).stem
        base_config['training']['pretrained']['model_name'] = Path(self._filename_).stem

        #print(path_to, self._filename_)

        with open(path_to.joinpath(self._filename_), "w") as f:
            yaml.dump(base_config, f, sort_keys=False)

    

# ------------------------------------------------------------ #
# ---------------------- TRAINING LOGS ----------------------- #
# ------------------------------------------------------------ #

def get_training_results(
    training_path: str,
    training_config: dict,
    fold_number: Optional[int] = None
) -> List[dict]:
    lines = load_log(training_path, training_config, mode="training")
    model_name = training_config['training']['model_name']
    lines = [ line for line in lines if line['training_config_name']==model_name ]
    if fold_number is not None:
        lines = [ line for line in lines if line["training metrics"]["CV"]==fold_number ]
    return lines

def load_best_model(
    checkpoint_path: str,
    training_config: dict,
    fold_number: int
) -> Any:
    model_file_name = f"best_model_fold_{fold_number:.0f}.pt"
    model_name = training_config['training']['model_name']
    checkpoint_path = Path(checkpoint_path).joinpath(model_name)
    model = build_model_singleyear_with_tfidf(training_config, 'cpu')
    checkpoint = torch.load(checkpoint_path.joinpath(model_file_name))
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def model_eval(
    loader, 
    model, 
    input_loader,
    remove_text: Optional[bool] = False
) -> Dict[str, List]:
    output = {
        "preds": [], "labels": [], "ids": []
    }
    with torch.no_grad():
        for b_idx, batch in tqdm(enumerate(loader)):
            ( indices, mammogram_ids, extra_features, labels, eligibility_mask ) = batch
            mean_mammogram_vec = torch.tensor(np.vstack(input_loader.get_embeddings(list(mammogram_ids))), dtype=torch.float16)
            if remove_text: # for ablation
                mean_mammogram_vec = torch.zeros(mean_mammogram_vec.shape, dtype=torch.float16)
            preds = model(mean_mammogram_vec, extra_features)
            preds = torch.sigmoid(preds.squeeze(-1)) # (B,1) -> (B,)
            m = eligibility_mask.bool()
            m_np = m.cpu().numpy()
            ids_arr = np.array(list(mammogram_ids))
            if m.any():
                output["preds"].append(preds[m].detach().cpu())
                output["labels"].append(labels[m].detach().cpu())
                output["ids"].append(ids_arr[m_np])
    output["preds"], output["labels"] = torch.cat(output["preds"]).numpy(), torch.cat(output["labels"]).numpy()
    output["ids"] = np.concatenate(output["ids"])
    return output

def calibrate_probabilities(
    val_output: dict, 
    test_output: dict
) -> Tuple[dict, dict]:
    eps = 1e-15
    clip_f = lambda p: np.clip(p, eps, 1 - eps)
    p_val, p_test = clip_f(val_output['preds']), clip_f(test_output['preds'])
    y_val, y_test = val_output['labels'], test_output['labels']
    X_val = np.column_stack([np.log(p_val), np.log(1 - p_val)])
    # -- fit calibration model (beta calibration)
    cal = LogisticRegression(solver="lbfgs")
    cal.fit(X_val, y_val)
    beta_calibrate = lambda p, cal: cal.predict_proba(np.column_stack([np.log(p), np.log(1 - p)]))[:, 1]
    p_val_cal = beta_calibrate(p_val, cal)
    p_test_cal = beta_calibrate(p_test, cal)

    val_output["preds_calibrated"] = p_val_cal
    val_output["brier_before_calibration"] = brier_score_loss(y_val, p_val)
    val_output["brier_after_calibration"] = brier_score_loss(y_val, p_val_cal)
    test_output["preds_calibrated"] = p_test_cal
    test_output["brier_before_calibration"] = brier_score_loss(y_test, p_test)
    test_output["brier_after_calibration"] = brier_score_loss(y_test, p_test_cal)
    return val_output, test_output

def explain_model_ig(
    loader, 
    model,
    input_loader,
    extra_features_baseline,
    device='cpu',
    n_steps: Optional[int] = 32,
    frac: Optional[float] = None,
    remove_text: Optional[bool] = False # ablate
) -> Dict[str, np.ndarray]:
    model.eval()
    ig = IntegratedGradients(lambda mean_vec, extra_feat: model(mean_vec, extra_feat).squeeze(-1))

    all_attr_mv, all_attr_ef = [], []
    all_preds, all_labels, all_ids = [], [], []
    for b_idx, batch in tqdm(enumerate(loader)):
        ( indices, mammogram_ids, extra_features, labels, eligibility_mask ) = batch 

        mean_mammogram_vec = torch.tensor(np.vstack(input_loader.get_embeddings(list(mammogram_ids))), dtype=torch.float32, device=device)
        if remove_text: # for ablation
            mean_mammogram_vec = torch.zeros(mean_mammogram_vec.shape, dtype=torch.float16, device=device)
        # -- since not in 'with torch.no_grad()' this might be already on, but just to make sure
        mean_mammogram_vec = mean_mammogram_vec.detach().requires_grad_(True)
        extra_features = extra_features.detach().requires_grad_(True)

        # -- baselines
        # -- text baseline: all-zero embedding vector
        mean_mammogram_vec_base = torch.zeros_like(mean_mammogram_vec)
        extra_base = torch.as_tensor(extra_features_baseline, device=device, dtype=torch.float32)
        extra_base = extra_base.unsqueeze(0).expand_as(extra_features)

        # -- attributions (for logits)
        attr_mv, attr_ef = ig.attribute(
            inputs=(mean_mammogram_vec, extra_features),
            baselines=(mean_mammogram_vec_base, extra_base),
            n_steps=n_steps
        )

        with torch.no_grad():
            logits = model(mean_mammogram_vec.detach(), extra_features.detach()).squeeze(-1)
            preds = torch.sigmoid(logits)

        # -- when we have no interest (or capacity of storing every individual case)
        # -- con: we are still running the model on each unit (slow even if 'frac' is small)
        if frac is not None and frac<1:
            neg_ixs = [ ix for ix, label in enumerate(labels) if label==0 ]
            pos_ixs = [ ix for ix, label in enumerate(labels) if label==1 ]
            sel_neg_ixs = random.sample(neg_ixs, int(len(neg_ixs)*frac))
            sel_ixs = sel_neg_ixs + pos_ixs
        else:
            sel_ixs = list(range(len(labels)))
        
        all_attr_mv.append(attr_mv.detach().cpu()[sel_ixs])
        all_attr_ef.append(attr_ef.detach().cpu()[sel_ixs])
        all_preds.append(preds.detach().cpu()[sel_ixs])
        all_labels.append(labels.detach().cpu()[sel_ixs])
        all_ids.append(np.array(list(mammogram_ids))[sel_ixs])

    return {
        "attr_mean_vec": torch.cat(all_attr_mv).numpy(),  # (N,5001)
        "attr_extra": torch.cat(all_attr_ef).numpy(),     # (N,33)
        "preds": torch.cat(all_preds).numpy(),
        "labels": torch.cat(all_labels).numpy(),
        "mammogram_ids": np.concatenate(all_ids),
    }


# -- Feature ablation options: (1) ablate only by the two main blocks: text and structured data; (2) top-k text vector and individual structured data
def build_mask_two_blocks(d_text=5001, d_struct=33) -> Tuple[torch.Tensor, torch.Tensor]:
    mask_text = torch.zeros(d_text, dtype=torch.long)     # one group
    mask_struct = torch.zeros(d_struct, dtype=torch.long) # one group
    return mask_text, mask_struct

def build_mask_topk_text(
    topk_text_indices,
    struct_names,
    d_text=5001,
    group_zip=True
) -> Tuple[torch.Tensor, torch.Tensor]:
    topk_text_indices = sorted(set(int(i) for i in topk_text_indices))

    # text: 0 = "all other dims", 1..K = selected dims individually
    mask_text = torch.zeros(d_text, dtype=torch.long)
    for j, idx in enumerate(topk_text_indices, start=1):
        if 0 <= idx < d_text:
            mask_text[idx] = j

    # structured: group zipcode dims, others separate
    mask_struct = make_struct_mask_zip_group(struct_names, group_zip=group_zip)
    return mask_text, mask_struct

def explain_model_ablation(
    loader: Any,
    model: Any,
    input_loader: InputLoader,
    extra_features_baseline: np.ndarray,
    feature_mask: Optional[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]],
    device: Optional[str] = "cpu",
) -> Dict[str, np.ndarray]:
    model.eval()
    ablator = FeatureAblation(lambda mean_vec, extra_feat: model(mean_vec, extra_feat).squeeze(-1))

    all_attr_mv, all_attr_ef = [], []
    all_preds, all_labels, all_ids = [], [], []
    with torch.no_grad():
        for b_idx, batch in tqdm(enumerate(loader), total=len(loader)):
            indices, mammogram_ids, extra_features, labels, eligibility_mask = batch
            # -- keep the eligibility logic (just in case we decide to create a multi-year model later)
            m = eligibility_mask.bool()
            if not m.any():
                continue
            ids_sel = np.array(list(mammogram_ids))[m.cpu().numpy()]
            extra = extra_features[m].to(device=device, dtype=torch.float32)
            y = labels[m].to(device=device)

            mv_np = np.vstack(input_loader.get_embeddings(list(ids_sel))).astype(np.float32)
            mean_vec = torch.from_numpy(mv_np).to(device=device, dtype=torch.float32)
            # -- baselines
            mean_vec_base = torch.zeros_like(mean_vec)
            extra_base_1d = torch.as_tensor(extra_features_baseline, device=device, dtype=torch.float32)
            extra_base = extra_base_1d.unsqueeze(0).expand_as(extra)

            # -- expand masks from 1D (D,), expand to (B,D)
            mask_text, mask_struct = feature_mask
            if mask_text.dim() != 1 or mask_struct.dim() != 1:
                raise Exception(f'masks have dimensions {mask_text.shape} and {mask_struct.shape}, but only (D,) is allowed.')
            mask_text_b = mask_text.unsqueeze(0).expand(mean_vec.shape[0], -1)
            mask_struct_b = mask_struct.unsqueeze(0).expand(extra.shape[0], -1)
            fm = (mask_text_b, mask_struct_b)

            # --> ablate
            #print(mean_vec.shape, extra.shape, mask_struct_b.shape, mask_text_b.shape, mask_struct.shape, mask_text.shape)
            attr_mv, attr_ef = ablator.attribute(
                inputs=(mean_vec, extra),
                baselines=(mean_vec_base, extra_base),
                feature_mask=fm
            )

            logits = model(mean_vec, extra).squeeze(-1)
            preds = torch.sigmoid(logits)

            all_attr_mv.append(attr_mv.cpu())
            all_attr_ef.append(attr_ef.cpu())
            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())
            all_ids.append(ids_sel)

    return {
        "attr_mean_vec": torch.cat(all_attr_mv).numpy(),
        "attr_extra": torch.cat(all_attr_ef).numpy(),
        "mammogram_ids": np.concatenate(all_ids),
    }

class TrainingLogManager(ConfigInterface):
    '''
        Interface to make easier the handling of the output of training experiments.
    '''
    def __init__(self, config_dir: str, config_defaults: dict):
        super().__init__(config_dir, config_defaults)
        self.config_dir = config_dir
        self.config_defaults = config_defaults
        self.n_folds = self.split_cfg['split']['kfold']
        self.seq_percentile = self.split_cfg['split']['seq_percentile']

        self.feature_columns = self.model_fields_cfg['fields']['feature_columns']
        self.feature_columns = [ elem for elem in self.feature_columns if elem!="mammogram_id"]

    def list_experiment_files(self) -> List[str]:
        cfg_model_foldername = ConfigFolderNames.model_foldername.value
        cfg_training_foldername = ConfigFolderNames.training_experiments_foldername.value
        training_exp_files = list(Path(self.config_dir).joinpath(cfg_model_foldername, cfg_training_foldername).glob("*"))
        return training_exp_files

    def validate_logs(self, line):
        TrainingBCELog.model_validate(line)

    def get_results(self, fold_number: Optional[int] = None):
        return get_training_results(self.model_logging_path, self.training_cfg, fold_number)

    def load_best_model(self, fold_number: int):
        return load_best_model(self.checkpoint_path, self.training_cfg, fold_number)

    def _get_split_ids(self, fold_number: int) -> Tuple[List[str], List[str], List[str]]:
        result = self.get_results(fold_number)
        data_split = DatasetSplit(self.config_dir, self.config_defaults)
        data_split.split(
            target_year=result[0]['target year'], n_splits=self.n_folds, 
            seq_percentile=self.seq_percentile
        )
        split_dict = data_split.cv_split_by_mammogram
        training_ids = split_dict[f'fold {fold_number}']['train']
        val_ids = split_dict[f'fold {fold_number}']['validation']
        test_ids = split_dict['test']
        return (training_ids, val_ids, test_ids)
    
    def get_training_scores(self, fold_number: int) -> Dict[str, Sequence]:
        model = self.load_best_model(fold_number)
        model.eval()

        # -- check whether current split is correct
        result = self.get_results(fold_number)
        parsed_split_file = Path(self.config_defaults['split']).stem
        if parsed_split_file!=result[0]['split_config_name']:
            raise ValueError('split configuration file parsed in defaults is not the same as the one used in training.')

        # -- generate the adequate splitting and get the dataloaders
        training_ids, val_ids, test_ids = self._get_split_ids(fold_number)
        input_loader = InputLoader(self.config_dir, self.config_defaults)
        # -- NEED TO BE CAREFUL HERE, SINCE WE ONLY USE THE IDS AFTER UNDERSAMPLING (so, maybe 'is_training'=True).
        train_loader, imratio = input_loader.get_dataloader(training_ids, target_year=result[0]['target year'], is_training=True)
        train_output = model_eval(train_loader, model, input_loader)
        return train_output
    
    def get_model_output(self, fold_number: int, remove_text: Optional[bool] = False) -> Tuple[Dict[str, Sequence], Dict[str, Sequence]]:
        model = self.load_best_model(fold_number)
        model.eval()

        # -- check whether current split is correct
        result = self.get_results(fold_number)
        print(self.training_cfg['training']['model_name'])
        print(result[0])
        parsed_split_file = Path(self.config_defaults['split']).stem
        if parsed_split_file!=result[0]['split_config_name']:
            raise ValueError('split configuration file parsed in defaults is not the same as the one used in training.')

        # -- generate the adequate splitting and get the dataloaders
        training_ids, val_ids, test_ids = self._get_split_ids(fold_number)
        input_loader = InputLoader(self.config_dir, self.config_defaults)
        val_loader, imratio = input_loader.get_dataloader(val_ids, target_year=result[0]['target year'], is_training=False)
        test_loader, imratio = input_loader.get_dataloader(test_ids, target_year=result[0]['target year'], is_training=False)

        val_output = model_eval(val_loader, model, input_loader, remove_text)
        test_output = model_eval(test_loader, model, input_loader, remove_text)
        # -- calibrate probabilities
        val_output, test_output = calibrate_probabilities(val_output, test_output)
        return (val_output, test_output)

    def _get_extra_features_baseline(
        self,
        ids: Optional[List[str]] = None
    ) -> np.ndarray:
        feat_columns = self.model_fields_cfg['fields']['feature_columns']
        # -- if not ID is provided, then returns an all-zero vector
        if ids is None:
            ndim = len(feat_columns) - 1 # do not consider mammogram id column
            baseline = np.zeros(ndim, dtype=np.float32)
        else:
            temp_df = pd.concat([ df for df in self._iter_final_data(mammogram_ids=ids) ], ignore_index=True)
            temp_df = temp_df[feat_columns].drop(columns=["mammogram_id"]).copy()
            baseline = temp_df.mean(axis=0).to_numpy(dtype=np.float32)
        return baseline

    def explain_model_outputs(
        self, 
        fold_number: int, 
        frac: Optional[float] = None,
        remove_text: Optional[bool] = False
    ) -> Dict[str, Sequence]:
        model = self.load_best_model(fold_number)

        # -- check whether current split is correct
        result = self.get_results(fold_number)
        parsed_split_file = Path(self.config_defaults['split']).stem
        if parsed_split_file!=result[0]['split_config_name']:
            raise ValueError('split configuration file parsed in defaults is not the same as the one used in training.')

        training_ids, val_ids, test_ids = self._get_split_ids(fold_number)
        input_loader = InputLoader(self.config_dir, self.config_defaults)
        test_loader, imratio = input_loader.get_dataloader(test_ids, target_year=result[0]['target year'], is_training=False)

        # -- definition of baseline is crucial
        # -- if no list of IDs is parsed below, then it will use an all-zero vector for all features (text and structured) as baseline for integrated gradients.
        # -- When a list of IDs is parsed, then it will use the mean of features over these IDs.
        extra_features_baseline = self._get_extra_features_baseline()
        attr = explain_model_ig(test_loader, model, input_loader, extra_features_baseline, remove_text=remove_text)
        return attr

    def feature_ablation(
        self, 
        fold_number: int
    ) -> Dict[str, Sequence]:
        # -- load best model and check whether current split is correct
        model = self.load_best_model(fold_number)
        result = self.get_results(fold_number)
        parsed_split_file = Path(self.config_defaults['split']).stem
        if parsed_split_file!=result[0]['split_config_name']:
            raise ValueError('split configuration file parsed in defaults is not the same as the one used in training.')

        training_ids, val_ids, test_ids = self._get_split_ids(fold_number)
        input_loader = InputLoader(self.config_dir, self.config_defaults)
        test_loader, imratio = input_loader.get_dataloader(test_ids, target_year=result[0]['target year'], is_training=False)

        d_text = self.training_cfg['model']['mammogram_input_dim']+1
        d_struct = self.training_cfg['model']['extra_features_dim']
        
        # -- set the baseline (so far: all-zero vector as baseline)
        extra_features_baseline = self._get_extra_features_baseline()
        
        # -- set feature_mask
        # ---- default: ablate two blocks: text block and structured block
        # ---- we should change this to include other options later.
        mask_text, mask_struct = build_mask_two_blocks(d_text=d_text, d_struct=d_struct)
        fm = (mask_text.to('cpu'), mask_struct.to('cpu'))

        attr_ablate = explain_model_ablation(test_loader, model, input_loader, extra_features_baseline, fm, device='cpu')
        return attr_ablate

