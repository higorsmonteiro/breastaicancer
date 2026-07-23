import typer
from typing import Optional, List
from hapcancer.cli.functions import *

CONFIG_DEFAULTS_HELP = "Repeatable key=value (e.g., -p alpha=0.1 -p debug=true)."
CONFIG_DIR_HELP = "Path to the configuration directory."

app = typer.Typer(help='''CLI interfaces to automate essential ETL 
                    pipelines and model tuning/training.''')

def parse_key_value_params(items):
    # -- default
    params = {
        "birads_classifier": "birads_clf_001.yml",
        "embeddings": "tfidf_001.yml",
        "split": "split_001.yml",
        "followup": None,
        "bmi_model": None,
        "tuning": None,
        "training_experiments": None,
        "eval": None
    }
    if not items:
        return params
    accepted_keys = params.keys()
    for item in items:
        if "=" not in item:
            raise typer.BadParameter(f"Invalid 'params' value '{item}'. Use key=value format.")
        key, val = item.split("=", 1) # split on the first '='
        key = key.strip()
        if not key or key not in accepted_keys:
            raise typer.BadParameter(f"Invalid key in '{item}'.")
        params[key] = val.strip() 
    return params

# ------------------------------------------------------------ #
# ------------------------- COMMANDS ------------------------- #
# ------------------------------------------------------------ #                    

@app.command("extract-raw-data")
def extract(
    config_dir: str = typer.Option(None, "--config-dir", "-d", help=CONFIG_DIR_HELP),
    config_params: List[str] = typer.Option(None, "--config-params", "-p", help=CONFIG_DEFAULTS_HELP),
    raw_data_name: str = typer.Option(None, "--raw-data-name", "-r", help=""),
    db_origin: str = typer.Option(None, "--db-origin", "-o", help=""),
    chunk_size: int = typer.Option(None, "--chunk-size", "-s", help=""),
    logger: bool = typer.Option(True, "--logger", "-l", help="Turn on the logger.")
):
    config_defaults = parse_key_value_params(config_params)
    extract_raw_data_api(config_dir, config_defaults, raw_data_name, db_origin, chunk_size, logger=logger)

@app.command("validate-raw-data")
def validate_raw_data(
    config_dir: str = typer.Option(None, "--config-dir", "-d", help=CONFIG_DIR_HELP),
    config_params: List[str] = typer.Option(None, "--config-params", "-p", help=CONFIG_DEFAULTS_HELP),
    fraction: float = typer.Option(0.5, "--fraction", "-f", help="Fraction of the raw data to be validated.")
):
    config_defaults = parse_key_value_params(config_params)
    validate_raw_data_api(config_dir, config_defaults, fraction)

@app.command("process-birads")
def process_birads(
    config_dir: str = typer.Option(None, "--config-dir", "-d", help=CONFIG_DIR_HELP),
    config_params: List[str] = typer.Option(None, "--config-params", "-p", help=CONFIG_DEFAULTS_HELP),
    process_mode: str = typer.Option("extract", "--process-mode", "-m", help='''Options: 'extract' or 'infer'. Whether to extract BI-RADS using RE or 
                                                                                infer using a ML model. Correct order: 'extract' -> 'infer'.''')
):
    config_defaults = parse_key_value_params(config_params)
    get_birads_api(config_dir, config_defaults, process_mode)

@app.command("preprocess")
def preprocess(
    config_dir: str = typer.Option(None, "--config-dir", "-d", help=CONFIG_DIR_HELP),
    config_params: List[str] = typer.Option(None, "--config-params", "-p", help=CONFIG_DEFAULTS_HELP),
    raw_data_name: str = typer.Option(None, "--raw-data-name", "-r", help="")
):
    config_defaults = parse_key_value_params(config_params)
    preprocess_api(config_dir, config_defaults, raw_data_name)

@app.command("find-shortcuts")
def find_shortcuts(
    config_dir: str = typer.Option(None, "--config-dir", "-d", help=CONFIG_DIR_HELP),
    config_params: List[str] = typer.Option(None, "--config-params", "-p", help=CONFIG_DEFAULTS_HELP),
):
    config_defaults = parse_key_value_params(config_params)
    find_shortcuts_terms_api(config_dir, config_defaults)

@app.command("preprocess-biopsy")
def preprocess_biopsy(
    config_dir: str = typer.Option(None, "--config-dir", "-d", help=CONFIG_DIR_HELP),
    config_params: List[str] = typer.Option(None, "--config-params", "-p", help=CONFIG_DEFAULTS_HELP),
    mode: str = typer.Option(None, "--mode", "-m", help="")
):
    config_defaults = parse_key_value_params(config_params)
    preprocess_biopsy_api(config_dir, config_defaults, mode)

@app.command("merge-sources")
def merge_sources(
    config_dir: str = typer.Option(None, "--config-dir", "-d", help=CONFIG_DIR_HELP),
    config_params: List[str] = typer.Option(None, "--config-params", "-p", help=CONFIG_DEFAULTS_HELP),
):
    config_defaults = parse_key_value_params(config_params)
    merge_sources_api(config_dir, config_defaults)

@app.command('train-tfidf')
def train_tfidf_embedding_model(
    config_dir: str = typer.Option(None, "--config-dir", "-d", help=CONFIG_DIR_HELP),
    config_params: List[str] = typer.Option(None, "--config-params", "-p", help=CONFIG_DEFAULTS_HELP),
    sampling_ratio: float = typer.Option(4, "--sampling-ratio", "-s", help="Sampling ratio between low risk and high risk mammograms for training.")
):
    config_defaults = parse_key_value_params(config_params)
    train_tfidf_embedding_model_api(config_dir, config_defaults, sampling_ratio)

@app.command("generate-dataset")
def generate_dataset(
    config_dir: str = typer.Option(None, "--config-dir", "-d", help=CONFIG_DIR_HELP),
    config_params: List[str] = typer.Option(None, "--config-params", "-p", help=CONFIG_DEFAULTS_HELP),
    verbose: Optional[bool] = True
):
    print(config_params)
    print(verbose)
    config_defaults = parse_key_value_params(config_params)
    generate_dataset_api(config_dir, config_defaults, verbose)

@app.command("precompute-sequences")
def precompute_sequences(
    config_dir: str = typer.Option(None, "--config-dir", "-d", help=CONFIG_DIR_HELP),
    config_params: List[str] = typer.Option(None, "--config-params", "-p", help=CONFIG_DEFAULTS_HELP),
    time_limit: int = typer.Option(36, "--time-limit", "-t", help="Time window (in months) of the past mammograms."),
    gb_size: int = typer.Option(10, "--gb-size", "-s", help="Amount of memory (in gigabytes) to reserve for the embedding of the past sequences."),
    batch_size: int = typer.Option(5000, "--batch-size", "-b", help="Size of the batch used for the precomputation.")
):
    config_defaults = parse_key_value_params(config_params)
    precompute_sequences_api(config_dir, config_defaults, time_limit, gb_size, batch_size)

@app.command("tuning")
def tuning(
    config_dir: str = typer.Option(None, "--config-dir", "-d", help=CONFIG_DIR_HELP),
    config_params: List[str] = typer.Option(None, "--config-params", "-p", help=CONFIG_DEFAULTS_HELP),
    target_year: int = typer.Option(5, "--target-year", "-t", help="Predict cancer within $target_year years."),
    n_splits: int = typer.Option(5, "--n-splits", "-n", help="Number of folds used to define the size of the validation set."),
    seq_percentile: float = typer.Option(99.5, "--seq-percentile", "-s", help="Upper percentile used to remove outliers in the number of past mammograms."),
    total_epochs_per_trial:  int = typer.Option(15, "--total-epochs", "-e", help="Total number of epochs to use in each tuning trial.") 

):
    config_defaults = parse_key_value_params(config_params)
    tuning_api(
        config_dir, config_defaults,
        target_year, n_splits, seq_percentile,
        total_epochs_per_trial
    )

@app.command("cv-training-tfidf")
def cross_validation_training(
    config_dir: str = typer.Option(None, "--config-dir", "-d", help=CONFIG_DIR_HELP),
    config_params: List[str] = typer.Option(None, "--config-params", "-p", help=CONFIG_DEFAULTS_HELP),
    target_year: int = typer.Option(5, "--target-year", "-t", help="Predict cancer within $target_year years."),
    n_splits: int = typer.Option(5, "--n-splits", "-n", help="Number of folds used to define the size of the validation set."),
    seq_percentile: float = typer.Option(99.5, "--seq-percentile", "-s", help="Upper percentile used to remove outliers in the number of past mammograms."),
):
    config_defaults = parse_key_value_params(config_params)
    cv_training_tfidf_api(
        config_dir, config_defaults,
        target_year, n_splits, seq_percentile
    )

@app.command("cv-training-best")
def cross_validation_training_best(
    config_dir: str = typer.Option(None, "--config-dir", "-d", help=CONFIG_DIR_HELP),
    config_params: List[str] = typer.Option(None, "--config-params", "-p", help=CONFIG_DEFAULTS_HELP),
    target_year: int = typer.Option(5, "--target-year", "-t", help="Predict cancer within $target_year years."),
    n_splits: int = typer.Option(5, "--n-splits", "-n", help="Number of folds used to define the size of the validation set."),
    seq_percentile: float = typer.Option(99.5, "--seq-percentile", "-s", help="Upper percentile used to remove outliers in the number of past mammograms."),
):
    config_defaults = parse_key_value_params(config_params)
    cv_training_best_api(
        config_dir, config_defaults,
        target_year, n_splits, seq_percentile
    )

@app.command("eval-metrics-best")
def eval_metrics_best(
    config_dir: str = typer.Option(None, "--config-dir", "-d", help=CONFIG_DIR_HELP),
    config_params: List[str] = typer.Option(None, "--config-params", "-p", help=CONFIG_DEFAULTS_HELP),
    target_year: int = typer.Option(5, "--target-year", "-t", help="Predict cancer within $target_year years.")
):
    config_defaults = parse_key_value_params(config_params)
    eval_metrics_best_api(
        config_dir, config_defaults, target_year
    )

@app.command("eval-explain-best")
def eval_explain_best(
    config_dir: str = typer.Option(None, "--config-dir", "-d", help=CONFIG_DIR_HELP),
    config_params: List[str] = typer.Option(None, "--config-params", "-p", help=CONFIG_DEFAULTS_HELP),
    target_year: int = typer.Option(5, "--target-year", "-t", help="Predict cancer within $target_year years."),
    fraction: float = typer.Option(1.0, "--fraction", "-f", help="Fraction of exams to apply the explainability calculations."),
):
    config_defaults = parse_key_value_params(config_params)
    eval_explain_best_api(
        config_dir, config_defaults, target_year, fraction
    )

    
