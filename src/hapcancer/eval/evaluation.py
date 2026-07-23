import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
from hapcancer.eval.log_manager import TrainingLogManager
from hapcancer.eval.metrics import GenerateCurvesForPlotting, Plotting
from hapcancer.config_manager import ConfigInterface

def persist_csv(
    df: pd.DataFrame, 
    model_eval_path: Path, 
    filename: str
) -> None:
    if not model_eval_path.is_dir():
        model_eval_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(model_eval_path.joinpath(f"{filename}.csv"))

def persist_parquet(
    df: pd.DataFrame, 
    model_eval_path: Path, 
    filename: str
) -> None:
    if not model_eval_path.is_dir():
        model_eval_path.mkdir(parents=True, exist_ok=True)
    df.to_parquet(model_eval_path.joinpath(f"{filename}.parquet"))

def get_models_outputs(
    log_manager: TrainingLogManager, 
    n_folds: int,
    remove_text: Optional[bool] = False # ablate
) -> pd.DataFrame:
    models_output = {
        "mammogram_id": [], "fold_id": [], "split_id": [], 
        "label": [], "pred": [],  "pred_calibrated": []
    }
    
    val_outputs, test_outputs = {}, {}
    print("Extracting model outputs: ")
    for k in tqdm(range(n_folds)):
        val_out, test_out = log_manager.get_model_output(k, remove_text=remove_text)
        # -- validation
        models_output['mammogram_id'].extend([ str(mid) for mid in val_out['ids'] ])
        models_output['fold_id'].extend([ k for elem in val_out['labels'] ])
        models_output['label'].extend(val_out['labels'])
        models_output['pred'].extend(val_out['preds'])
        models_output['pred_calibrated'].extend(val_out['preds_calibrated'])
        models_output['split_id'].extend([ 'validation' for elem in val_out['labels'] ])
        # -- test
        models_output['mammogram_id'].extend([ str(mid) for mid in test_out['ids'] ])
        models_output['fold_id'].extend([ k for elem in test_out['labels'] ])
        models_output['label'].extend(test_out['labels'])
        models_output['pred'].extend(test_out['preds'])
        models_output['pred_calibrated'].extend(test_out['preds_calibrated'])
        models_output['split_id'].extend([ 'test' for elem in test_out['labels'] ])

    models_output = pd.DataFrame(models_output)
    return models_output

def get_split_ids(log_manager: TrainingLogManager) -> pd.DataFrame:
    training_ids, val_ids, test_ids = log_manager._get_split_ids(fold_number=0)
    split_label = [ 'train' for elem in training_ids ]
    split_label += [ 'train' for elem in val_ids ]
    split_label += [ 'test' for elem in test_ids ]
    split_df = pd.DataFrame({
        "mammogram_id": training_ids+val_ids+test_ids,
        "split_label": split_label
    })
    return split_df

def create_roc_figures(
    outputs_df: pd.DataFrame, 
    n_folds: int,
    n_bootstraps: Optional[int] = 10,
    **kwargs
) -> Dict[int, Tuple[Any, Any]]:
    roc_figures = {}
    for k in tqdm(range(n_folds)):
        filter_df = outputs_df[outputs_df["fold_id"]==k].copy()

        cur_results = {
            "validation": {
                "preds": filter_df[filter_df["split_id"]=="validation"]["pred"],
                "labels": filter_df[filter_df["split_id"]=="validation"]["label"]
            },
            "test": {
                "preds": filter_df[filter_df["split_id"]=="test"]["pred"],
                "labels": filter_df[filter_df["split_id"]=="test"]["label"]
            }
        }

        curve_generator = Plotting(cur_results)
        curve_generator._calculate_roc_with_ci(n_bootstraps=n_bootstraps)
        fig, ax = curve_generator.plot_roc(**kwargs)
        roc_figures[k] = (fig, ax)
    return roc_figures

def create_topk_figures(
    outputs_df: pd.DataFrame, 
    n_folds: int,
    n_bootstraps: Optional[int] = 10,
    K: Optional[int] = 200,
    upper_K_perc: Optional[int] = 200,
    percent_viz: Optional[List[int]] = [4, 8, 12],
    **kwargs
) -> Dict[int, Tuple[Any, Any]]:
    topk_figures = {} 
    topk_values_val, topk_values_test = {}, {}
    for k in tqdm(range(n_folds)):
        filter_df = outputs_df[outputs_df["fold_id"]==k].copy()

        cur_results = {
            "validation": {
                "preds": filter_df[filter_df["split_id"]=="validation"]["pred"],
                "labels": filter_df[filter_df["split_id"]=="validation"]["label"]
            },
            "test": {
                "preds": filter_df[filter_df["split_id"]=="test"]["pred"],
                "labels": filter_df[filter_df["split_id"]=="test"]["label"]
            }
        }

        curve_generator = Plotting(cur_results)
        curve_generator._calculate_pr_with_ci(n_bootstraps=n_bootstraps, K=K, upper_K_perc=upper_K_perc)
        fig, ax = curve_generator.plot_topk(percent_viz=percent_viz, **kwargs)
        topk_figures[k] = (fig, ax)
    return topk_figures

def get_topk_values(
    outputs_df: pd.DataFrame,
    n_folds: int
) -> pd.DataFrame:
    topk_values = {}
    for k in tqdm(range(n_folds)):
        filter_df = outputs_df[outputs_df["fold_id"]==k].copy()

        cur_results = {
            "validation": {
                "preds": filter_df[filter_df["split_id"]=="validation"]["pred"],
                "labels": filter_df[filter_df["split_id"]=="validation"]["label"]
            },
            "test": {
                "preds": filter_df[filter_df["split_id"]=="test"]["pred"],
                "labels": filter_df[filter_df["split_id"]=="test"]["label"]
            }
        }
        curve_generator = Plotting(cur_results)
        curve_generator._calculate_pr_with_ci(n_bootstraps=10, K=250, upper_K_perc=1.0)
        topk_val_df = pd.DataFrame(curve_generator.pr_results_val)
        topk_test_df = pd.DataFrame(curve_generator.pr_results_test)
        
        # -- merge validation and test dataframes into a single one
        cols_to_not_suffix = ["K_values", "K_perc"]
        topk_val_df = topk_val_df.rename(columns={ col: col+"_val" for col in topk_val_df.columns if col not in cols_to_not_suffix})
        topk_test_df = topk_test_df.rename(columns={ col: col+"_test" for col in topk_test_df.columns if col not in cols_to_not_suffix})
        topk_val_df = topk_val_df.merge(topk_test_df.drop(columns=["K_perc"]), on="K_values", how="left")
        topk_values[k] = topk_val_df
    return topk_values

def extract_epoch_metrics(
    epoch_metrics: List[dict], 
    fold_number: int
) -> pd.DataFrame:
    res_schema = {
        "epochs": [], "training loss": [], "validation loss": [],
        "training auroc": [], "validation auroc": [],
        "training auprc": [], "validation auprc": []
    }
    res_schema["epochs"] = [ elem["epoch"] for elem in epoch_metrics if elem["training metrics"]["CV"]== fold_number ]
    res_schema["training loss"] = [ elem["training metrics"]["Loss"] for elem in epoch_metrics if elem["training metrics"]["CV"]==fold_number ]
    res_schema["validation loss"] = [ elem["validation metrics"]["Loss"] for elem in epoch_metrics if elem["training metrics"]["CV"]==fold_number ]
    res_schema["training auroc"] = [ elem["training metrics"]["AUROC"] for elem in epoch_metrics if elem["training metrics"]["CV"]==fold_number ]
    res_schema["validation auroc"] = [ elem["validation metrics"]["AUROC"] for elem in epoch_metrics if elem["training metrics"]["CV"]==fold_number ]
    res_schema["training auprc"] = [ elem["training metrics"]["Average Precision"] for elem in epoch_metrics if elem["training metrics"]["CV"]==fold_number ]
    res_schema["validation auprc"] = [ elem["validation metrics"]["Average Precision"] for elem in epoch_metrics if elem["training metrics"]["CV"]==fold_number ]
    res_schema = pd.DataFrame(res_schema)
    return res_schema

def get_training_epochs_figures(
    log_manager: TrainingLogManager,
    n_folds: int
) -> Dict[int, Tuple[Any, Any, Any]]:
    epoch_figures = {}
    for k in tqdm(range(n_folds)):
        fold_results = log_manager.get_results(k)
        results_df = extract_epoch_metrics(fold_results, k)

        curve_generator = Plotting({})
        fig, (ax1, ax2) = curve_generator.plot_training_epochs(results_df)
        epoch_figures[k] = (fig, ax1, ax2)
    return epoch_figures
        

def attr_block_reliance(
    ids: List[str],
    attr_text: np.ndarray, 
    attr_struct: np.ndarray
) -> pd.DataFrame:
    text_contrib = np.abs(attr_text).sum(axis=1)
    struct_contrib = np.abs(attr_struct).sum(axis=1)
    share_text = text_contrib / (text_contrib + struct_contrib + 1e-12)
    return pd.DataFrame({
        "mammogram_id": ids,
        "text_contrib": text_contrib,
        "struct_contrib": struct_contrib,
        "text_share": share_text,
    })


def explain_models_ig(
    log_manager: TrainingLogManager,
    emb_model: Any,
    n_folds: int,
    frac: Optional[float] = None,
    remove_text: Optional[bool] = False
) -> Tuple[Dict[int, pd.DataFrame], pd.DataFrame]:
    # tf-idf model
    text_terms = emb_model.get_feature_names_out()
    attr_blocks = {}
    individual_blocks = {
        "CV": [], "mammogram_id": [],
        "text_terms": [], "text_attributions": [],
        "exam_interval_attribution": [],
        "extra_attributions": []
    }
    for k in tqdm(range(n_folds)):
        cur_attrs = log_manager.explain_model_outputs(k, frac=frac, remove_text=remove_text)
        # -- block reliance
        block_df = attr_block_reliance(cur_attrs['mammogram_ids'], cur_attrs['attr_mean_vec'], cur_attrs['attr_extra'])
        attr_blocks[k] = block_df
        # -- rank
        ids = cur_attrs["mammogram_ids"]
        extra_attr = cur_attrs['attr_extra']
        text_attr = cur_attrs['attr_mean_vec']

        individual_blocks["CV"] += [ k for elem in ids ]
        individual_blocks["mammogram_id"] += list(ids)

        # -- confusing (find a more intuitive numpy way for this)
        individual_blocks["text_terms"] += [ text_terms[np.argsort(np.abs(elem[:-1]))[::-1][:elem[:-1][np.abs(elem[:-1])>1e-4].shape[0]]] for elem in text_attr ]
        individual_blocks["text_attributions"] += [ elem[:-1][np.argsort(np.abs(elem[:-1]))[::-1][:elem[:-1][np.abs(elem[:-1])>1e-4].shape[0]]] for elem in text_attr ]
        individual_blocks["exam_interval_attribution"] += [ elem[-1] for elem in text_attr  ]
        individual_blocks["extra_attributions"] += [ elem for elem in extra_attr ]
    individual_blocks = pd.DataFrame(individual_blocks)
    return attr_blocks, individual_blocks

def explain_models_feat_ablation(
    log_manager: TrainingLogManager,
    emb_model: Any,
    n_folds: int,
    agg_func: Optional[str] = 'mean'
) -> Tuple[Dict[int, pd.DataFrame], pd.DataFrame]:
    
    agg_f = lambda arr: arr
    if agg_func == 'mean':
        agg_f = lambda arr: np.mean(arr)
    
    attr_blocks = {}
    for k in tqdm(range(n_folds)):
        cur_attrs = log_manager.feature_ablation(k)
        ids = cur_attrs["mammogram_ids"]
        extra_attr = cur_attrs['attr_extra']
        text_attr = cur_attrs['attr_mean_vec']
        
        attr_blocks[k] = pd.DataFrame({
            "CV": [ k for elem in ids ], 
            "mammogram_id": [ elem for elem in ids ], 
            "text_attributions": [ agg_f(elem) for elem in text_attr ], 
            "extra_attributions": [ agg_f(elem) for elem in extra_attr ]
        })
    return attr_blocks

class Evaluation(ConfigInterface):
    def __init__(self, config_dir: str, config_defaults: dict):
        super().__init__(config_dir, config_defaults)
        self.config_dir = config_dir
        self.config_defaults = config_defaults

        self.training_log_manager = TrainingLogManager(config_dir, config_defaults)

        self.n_folds = self.split_cfg['split']['kfold']
        self.model_name = self.training_cfg['training']['model_name']
        self.model_eval_path = self.eval_path.joinpath(self.model_name)
        self.model_path = self.checkpoint_path.joinpath(self.model_name)

        # -- load language model (TF-IDF)
        self.emb_model_name = self.embeddings_cfg['tfidf']['model_name']
        self.emb_model_path = self.fitted_models_folder_path.joinpath(f'{self.emb_model_name}.joblib')
        self.emb_model = None

        self.temp_outputs = None

    def _load_embedding_model(self):
        self.emb_model = joblib.load(self.emb_model_path)
    
    def _get_models_outputs(self, force: Optional[bool] = False, remove_text: Optional[bool] = False) -> None:
        if self.temp_outputs is None:
            self.temp_outputs = get_models_outputs(self.training_log_manager, self.n_folds, remove_text=remove_text)
        elif force:
            self.temp_outputs = get_models_outputs(self.training_log_manager, self.n_folds, remove_text=remove_text)

    def save_model_outputs(self, remove_text: Optional[bool] = False) -> None:
        if self.model_path.is_dir(): # -- if the model file exists
            self._get_models_outputs(remove_text=remove_text)
            persist_parquet(self.temp_outputs, self.model_eval_path, "models_outputs")
        else:
            raise ValueError(f"{self.model_name} does not have a checkpoint folder.")

    def save_cohort_ids(self) -> None:
        split_df = get_split_ids(self.training_log_manager)
        persist_parquet(split_df, self.model_eval_path, "cohort_ids")
    
    def save_roc_figures(
        self, 
        n_bootstraps: Optional[int] = 20, 
        remove_text: Optional[bool] = False,
        **kwargs
    ) -> None:       
        self._get_models_outputs(remove_text=remove_text)
        roc_figures = create_roc_figures(self.temp_outputs, self.n_folds, n_bootstraps, **kwargs)
        # -- persist figures
        for fold, (fig, ax) in roc_figures.items():
            fig.savefig(self.model_eval_path.joinpath(f"roc_fold_{fold}.png"), dpi=300, bbox_inches="tight")
        return

    def save_topk_figures(
        self, 
        n_bootstraps: Optional[int] = 20, 
        K: Optional[int] = 200,
        upper_K_perc: Optional[float] = 0.2,
        percent_viz: Optional[List[int]] = [4, 8, 12],
        remove_text: Optional[bool] = False,
        **kwargs
    ) -> None:
        self._get_models_outputs(remove_text=remove_text)
        topk_figures = create_topk_figures(
            self.temp_outputs, self.n_folds, n_bootstraps=n_bootstraps, 
            K=K, upper_K_perc=upper_K_perc, percent_viz=percent_viz
        )
        # -- persist figures
        for fold, (fig, ax) in topk_figures.items():
            fig.savefig(self.model_eval_path.joinpath(f"topk_pr_fold_{fold}.png"), dpi=300, bbox_inches="tight")
        return

    def save_topk_table(self, remove_text: Optional[bool] = False) -> None:
        self._get_models_outputs(remove_text=remove_text)
        topk_values = get_topk_values(self.temp_outputs, self.n_folds)
        # -- persist tables
        for fold, topk_df in topk_values.items():
            persist_parquet(topk_df, self.model_eval_path, f"topk_values_fold_{fold}")

    def save_training_epochs_figures(self) -> None:
        epoch_figures = get_training_epochs_figures(self.training_log_manager, self.n_folds)
        # -- persist figures
        for fold, (fig, ax1, ax2) in epoch_figures.items():
            fig.savefig(self.model_eval_path.joinpath(f"epochs_metrics_fold_{fold}.png"), dpi=300, bbox_inches="tight")
        return 

    def save_calibration_plot(self):
        pass

    def save_model_attributions_ig(self, frac: Optional[float] = None, remove_text: Optional[bool] = False):
        self._load_embedding_model()
        attr_blocks, attr_df = explain_models_ig(self.training_log_manager, self.emb_model, self.n_folds, frac, remove_text=remove_text)
        for fold, block_df in attr_blocks.items():
            persist_parquet(block_df, self.model_eval_path, f"block_ig_attributions_fold_{fold}")
        persist_parquet(attr_df, self.model_eval_path, "individual_ig_attributions")

    def save_feature_ablation_results(self):
        self._load_embedding_model()
        attr_blocks = explain_models_feat_ablation(self.training_log_manager, self.emb_model, self.n_folds, 'mean')
        for fold, block_df in attr_blocks.items():
            persist_parquet(block_df, self.model_eval_path, f"feat_ablation_block_attr_fold_{fold}")


        


