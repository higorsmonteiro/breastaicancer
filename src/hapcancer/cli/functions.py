'''
    Auxiliary functions for the CLI interface.
'''
from typing import List, Union, Optional

from hapcancer.mlflow_setup import configure_mlflow
from hapcancer.config_manager import ConfigInterface

from hapcancer.schemas.enums import ConfigFolderNames
from hapcancer.etl.extract.extractor import Extractor
from hapcancer.schemas.validation import DataValidator
from hapcancer.etl.transform.preprocessing.preprocess import (
    TransformAnamnesis, TransformBiopsyData, TransformMammograms, TransformPersonUser
)
from hapcancer.etl.transform.preprocessing.shortcut_terms import FindShortcutTerms 
from hapcancer.etl.transform.process_birads.get_birads import GetBirads
from hapcancer.etl.transform.process_birads.birads_classifier import BiradsClassifier
from hapcancer.etl.transform.embeddings.embed import FitTFIDF

from hapcancer.etl.load.precompute import (
    PrecomputeSequenceTFIDF,
    PrecomputeSequenceBERT,
    PrecomputeSequenceTFIDFTimeWeighted,
    PrecomputeSequenceBERTTimeWeighted,
)

from hapcancer.etl.load.setup_final_eligibility import SetupMammogramEligibility
from hapcancer.etl.transform.merge_sources import MergeSources
#from hapcancer.model.training.train_singleyear import TrainingSingleYearTFIDF
from hapcancer.model.training.train_singleyear_mlflow import TrainingSingleYearTFIDF
#from hapcancer.model.training.tuning import TuningSingleYear
from hapcancer.model.training.tuning_mlflow import TuningSingleYear
from hapcancer.model.dataload.load_input import InputLoader, DatasetSplit

from hapcancer.eval.log_manager import TuningLogManager, TrainingLogManager
from hapcancer.eval.evaluation import Evaluation

RAW_DATA_NAMES = [
    "mammogram", "biopsy", "anamnesis", 
    "user", "person", "patient" 
]

# (embedding_id, aggregation) → class
_PRECOMPUTE_REGISTRY = {
    ('tfidf',     'mean'):          PrecomputeSequenceTFIDF,
    ('tfidf',     'time_weighted'): PrecomputeSequenceTFIDFTimeWeighted,
    ('bert',      'mean'):          PrecomputeSequenceBERT,
    ('bert',      'time_weighted'): PrecomputeSequenceBERTTimeWeighted,
    ('bertimbau', 'mean'):          PrecomputeSequenceBERT,
    ('bertimbau', 'time_weighted'): PrecomputeSequenceBERTTimeWeighted,
    ('biobertpt', 'mean'):          PrecomputeSequenceBERT,
    ('biobertpt', 'time_weighted'): PrecomputeSequenceBERTTimeWeighted,
    ('ollama', 'mean'):          PrecomputeSequenceBERT,       # reuses BERT class — dense ndarray, same interface
    ('ollama', 'time_weighted'): PrecomputeSequenceBERTTimeWeighted,
}

def extract_raw_data_api(
    config_dir: str, 
    config_defaults: dict,
    raw_data_name: str,
    db_origin: str,
    chunk_size: int,
    logger: Optional[bool] = True
) -> None:
    with Extractor(config_dir, config_defaults, logger=logger) as extractor:
        match raw_data_name:
            case 'mammogram':
                extractor.fetch_mammograms_paginated(
                    db_origin=db_origin, timer=0.5, chunk_size=chunk_size, 
                    verbose=True, max_retries=5, start_year=2016
                )
            case 'biopsy':
                extractor.fetch_biopsy_paginated(
                    db_origin=db_origin, timer=0.5, chunk_size=chunk_size, 
                    verbose=True, max_retries=5, start_year=2018
                )
            case 'anamnesis':
                extractor.fetch_anamnesis_paginated(
                    db_origin=db_origin, timer=0.5, chunk_size=chunk_size, 
                    verbose=True, max_retries=5
                )
            case cohort_type if cohort_type in ['user', 'person', 'patient']:
                extractor.fetch_cohort_from_mammograms(
                    db_origin=db_origin, cohort_type=cohort_type, timer=0.5, 
                    chunk_size=chunk_size, verbose=True, max_retries=5
                )
            case _:
                raise ValueError(f"{raw_data_name} is not a valid input. Valid inputs are: {RAW_DATA_NAMES}.")

def validate_raw_data_api(
    config_dir: str, 
    config_defaults: dict,
    fraction: float
) -> None:
    data_validator = DataValidator(config_dir, config_defaults)
    data_validator.validate_raw_data(fraction=fraction)

def get_birads_api(
    config_dir: str, 
    config_defaults: dict,
    process_mode: str
) -> None:
    match process_mode:
        case 'extract':
            get_birads = GetBirads(config_dir, config_defaults)
            get_birads.get()
        case 'infer':
            birads_clf = BiradsClassifier(config_dir, config_defaults)
            birads_clf.fit_and_infer()

def preprocess_api(
    config_dir: str, 
    config_defaults: dict,
    raw_data_name: str
) -> None:
    match raw_data_name:
        case "mammogram":
            mamm_process = TransformMammograms(config_dir, config_defaults)
            mamm_process.transform()
        case "anamnesis":
            anamnesis_preprocess = TransformAnamnesis(config_dir, config_defaults)
            anamnesis_preprocess.transform()
        case "cohort":
            cohort_preprocess = TransformPersonUser(config_dir, config_defaults)
            cohort_preprocess.transform()
        case _:
            raise ValueError(f"{raw_data_name} is not a valid input. Valid inputs are: {RAW_DATA_NAMES}.")

def preprocess_biopsy_api(
    config_dir: str, 
    config_defaults: dict,
    mode: str
) -> None:
    biopsy_preprocess = TransformBiopsyData(config_dir, config_defaults)
    match mode:
        case 'get-reports':
            biopsy_preprocess.get_breast_biopsy_reports()
        case 'classify-reports':
            biopsy_preprocess.classify_breast_biopsy_reports()
        case 'get-results':
            biopsy_preprocess.get_breast_biopsy_results()

def find_shortcuts_terms_api(
    config_dir: str,
    config_defaults: dict
) -> None:
    terms_filter = FindShortcutTerms(config_dir, config_defaults)
    terms_filter.find_terms()

def merge_sources_api(config_dir: str, config_defaults: dict) -> None:
    merger = MergeSources(config_dir, config_defaults)
    merger.merge()

def train_tfidf_embedding_model_api(
    config_dir: str,
    config_defaults: dict,
    sampling_ratio: float,
    mammogram_ids: Optional[List[str]] = None
) -> None:
    tfidf_fit = FitTFIDF(config_dir, config_defaults)
    tfidf_fit.fit(sampling_ratio=sampling_ratio, mammogram_ids=mammogram_ids)

def generate_dataset_api(
    config_dir: str, 
    config_defaults: dict,
    verbose: Optional[bool] = True
) -> None:
    final_data_setup = SetupMammogramEligibility(config_dir, config_defaults)
    final_data_setup.setup_mammogram_data(verbose=verbose)

#def precompute_sequences_api(
#    config_dir: str, 
#    config_defaults: dict,
#    time_limit: Optional[int] = 36,
#    gb_size: Optional[int] = 10,
#    batch_size: Optional[int] = 5000
#) -> None:
#    precompute = PrecomputeSequenceTFIDF(config_dir, config_defaults)
#    precompute.precompute(time_limit=time_limit, batch_size=batch_size, gb_size=gb_size)

def precompute_sequences_api(
    config_dir: str,
    config_defaults: dict,
    time_limit: Optional[int] = 36,
    gb_size: Optional[int] = 10,
    batch_size: Optional[int] = 5000,
) -> None:
    config_manager = ConfigInterface(config_dir, config_defaults)
    emb_config = config_manager.embeddings_cfg
    embedding_id = emb_config['embedding_id']                        # e.g. 'tfidf', 'bert', 'ollama'
    aggregation  = emb_config.get('aggregation', 'mean')             # default keeps old behaviour

    key = (embedding_id, aggregation)
    precompute_cls = _PRECOMPUTE_REGISTRY.get(key)
    if precompute_cls is None:
        raise ValueError(
            f"No precompute class for combination {key}. "
            f"Available: {list(_PRECOMPUTE_REGISTRY.keys())}"
        )
    precompute = precompute_cls(config_dir, config_defaults)
    precompute.precompute(time_limit=time_limit, batch_size=batch_size, gb_size=gb_size)

def tuning_api(
    config_dir: str, 
    config_defaults: dict,
    target_year: Optional[int] = 5,
    n_splits: Optional[int] = 5,
    seq_percentile: Optional[int] = 99.5,
    total_epochs_per_trial: Optional[int] = 15,
    ablate: Optional[bool] = False
) -> None:
    # -- set MLflow experiment
    configure_mlflow(tracking_uri="sqlite:///D:/hapvida/mlruns.db", experiment_name="hapcancer")
    
    data_split = DatasetSplit(config_dir, config_defaults)
    data_split.split(
        target_year=target_year, n_splits=n_splits, 
        seq_percentile=seq_percentile
    )
    split_dict = data_split.cv_split_by_mammogram

    tuner = TuningSingleYear(config_dir, config_defaults, ablate)
    tuner.set_split(split_dict=split_dict)
    tuner.set_target_year(target_year)
    tuner._set_total_epochs_per_trial(total_epochs_per_trial)
    tuner.run_study()

def cv_training_tfidf_api(
    config_dir: str, 
    config_defaults: dict,
    target_year: Optional[int] = 5,
    n_splits: Optional[int] = 5,
    seq_percentile: Optional[int] = 99.5,
    ablate: Optional[bool] = False
) -> None:
    # -- set MLflow experiment
    configure_mlflow(tracking_uri="sqlite:///D:/hapvida/mlruns.db", experiment_name="hapcancer")

    data_split = DatasetSplit(config_dir, config_defaults)
    data_split.split(
        target_year=target_year, n_splits=n_splits, 
        seq_percentile=seq_percentile
    )
    split_dict = data_split.cv_split_by_mammogram
    trainer = TrainingSingleYearTFIDF(config_dir, config_defaults)
    trainer.train(split_dict, target_year, tuning=False, remove_text=ablate)

# -- coupled with 'cv_training_tfidf_api'
def cv_training_best_api(
    config_dir: str, 
    config_defaults: dict,
    target_year: Optional[int] = 5,
    n_splits: Optional[int] = 5,
    seq_percentile: Optional[int] = 99.5,
    ablate: Optional[bool] = False
) -> None:
    # -- set MLflow experiment
    configure_mlflow(tracking_uri="sqlite:///D:/hapvida/mlruns.db", experiment_name="hapcancer")

    # -- which split file we are using
    split_str = config_defaults['split'].split(".")[0]
    # -- for a given target year, get the best set of model params
    tuning_log_manager = TuningLogManager(config_dir, config_defaults)
    strat_str = tuning_log_manager.followup_cfg['dataset_name'] # -- better to define internally
    best_params = tuning_log_manager.get_best_trial_params(strat_str=strat_str, target_year=target_year, split_str=split_str)
    best_trial_number = best_params[0]
    # -- generate the config file for the training experiment of the best trial
    tuning_log_manager.persist_training_config(trial_number=best_trial_number, strat_str=strat_str, target_year=target_year, split_str=split_str)

    config_filename = tuning_log_manager.get_model_name(trial_number=best_trial_number, strat_str=strat_str, target_year=target_year, split_str=split_str)
    print(f"Filename of the training configuration: {config_filename}") 
    train_exps_cfg_entry = ConfigFolderNames.training_experiments_foldername.value
    # -- set the training experiment with the recently created file
    config_defaults[train_exps_cfg_entry] = config_filename
    # -- perform the CV training for the recently created file
    cv_training_tfidf_api(
        config_dir, config_defaults,
        target_year, n_splits, seq_percentile, 
        ablate=ablate
    )

def eval_metrics_best_api(
    config_dir: str, 
    config_defaults: dict,
    target_year: Optional[int] = 5,
    ablate: Optional[bool] = False
) -> None:
    # -- which split file we are using
    split_str = config_defaults['split'].split(".")[0]
    # -- for a given target year, get the best set of model params
    tuning_log_manager = TuningLogManager(config_dir, config_defaults)
    strat_str = tuning_log_manager.followup_cfg['dataset_name'] # -- better to define internally
    best_params = tuning_log_manager.get_best_trial_params(strat_str=strat_str, target_year=target_year, split_str=split_str)
    best_trial_number = best_params[0]
    config_filename = tuning_log_manager.get_model_name(trial_number=best_trial_number, strat_str=strat_str, target_year=target_year, split_str=split_str)
    # -- set the training experiment with the corresponding configuration file
    train_exps_cfg_entry = ConfigFolderNames.training_experiments_foldername.value
    config_defaults[train_exps_cfg_entry] = config_filename
    # -- evaluate and generate the output metrics files
    evaluator = Evaluation(config_dir, config_defaults)
    evaluator.save_model_outputs(remove_text=ablate)
    evaluator.save_cohort_ids()
    evaluator.save_roc_figures(remove_text=ablate)
    #evaluator.save_topk_figures()
    #evaluator.save_topk_table()
    evaluator.save_training_epochs_figures()

def eval_explain_best_api(
    config_dir: str, 
    config_defaults: dict,
    target_year: Optional[int] = 5,
    fraction: Optional[float] = 1.0,
    ablate: Optional[bool] = False
) -> None:
    # -- for a given target year, get the best set of model params
    tuning_log_manager = TuningLogManager(config_dir, config_defaults)
    strat_str = tuning_log_manager.followup_cfg['dataset_name'] # -- better to define internally
    best_params = tuning_log_manager.get_best_trial_params(strat_str=strat_str, target_year=target_year)
    best_trial_number = best_params[0]
    config_filename = tuning_log_manager.get_model_name(trial_number=best_trial_number, strat_str=strat_str, target_year=target_year)
    # -- set the training experiment with the corresponding configuration file
    train_exps_cfg_entry = ConfigFolderNames.training_experiments_foldername.value
    config_defaults[train_exps_cfg_entry] = config_filename
    # -- evaluate and generate the output attribution files
    evaluator = Evaluation(config_dir, config_defaults)
    evaluator.save_model_attributions_ig(frac=fraction, remove_text=ablate)
    evaluator.save_feature_ablation_results()