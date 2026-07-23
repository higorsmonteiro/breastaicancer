🇺🇸 English | [🇧🇷 Português](how_to_run.pt-BR.md)

# How to Run

This document describes the recommended execution workflow for the project after a
configuration directory has been created and validated.

The pipeline is executed via a small set of top-level scripts. Running these scripts
in order produces all pipeline artifacts:

- Raw extracted datasets
- Processed / merged datasets
- BI-RADS classifier for reports
- Final cohort dataset for modeling
- Tuned models and best checkpoints
- Evaluation metrics, plots, and result tables

All scripts rely on the package CLI entrypoints, which require the project to be
installed in the active environment (editable install recommended).

---

## 1. Preconditions

Before running anything, ensure:

1) A configuration directory exists and follows the required schema. (See: docs/configuration_setup.md)

2) The environment is installed and active:

```
conda env create -f environment.yml<br>
conda activate hapcancer
```

3) The project is installed in editable mode (from repository root):

```
pip install -e .
```

4) Defaults mapping is defined for the configuration directory you are using
   (followup, embeddings, split, training_experiments, tuning, etc.).

---

## 2. Overview: the six main scripts

The full pipeline is executed by running the following scripts in order:

1) run_extract.py:<br>
   Extract raw data from source databases in chunked form.
   Requires explicit `--config-dir`.

2) run_transform.py:<br>
   Orchestrate transformation and feature-generation steps over extracted data.
   Supports partial execution via CLI flags.

3) run_load.py:<br>
   Apply eligibility criteria, build the cohort dataset, and precompute
   historical aggregates needed for training.

4) run_tuning.py:<br>
   Run hyperparameter tuning experiments and store tuning outputs.

5) run_training_best.py:<br>
   Train the best-performing model configuration (selected from tuning) and write final
   checkpoints and training logs.

6) run_eval.py:<br>
   Run evaluation on a held-out test set and generate metrics, tables, and plots.

By the end of run_eval.py, the run directory should contain everything required to
reproduce the results.

---

## 3. Execution model: scripts are thin wrappers around CLI commands

The scripts above are wrappers around CLI commands exposed by the package.
These commands become available after:

```
pip install -e .
```

In particular, the entrypoint:

```
hapcancer
```

is provided by the CLI module (hapcancer/cli/cli.py) and supports commands like:

    hapcancer extract-raw-data [args]
    hapcancer process-birads [args]
    hapcancer preprocess [args]
    hapcancer train-tfidf [args]
    hapcancer merge-sources [args]
    hapcancer generate-dataset [args]
    hapcancer precompute-sequences [args]
    hapcancer tuning [args]
    hapcancer cv-training-best [args]
    hapcancer eval-metrics-best [args]
    hapcancer eval-explain-best [args]

The wrapper scripts typically exist to:

- Run many CLI calls in a safe loop
- Handle database-origin variations (e.g., HSP vs PSC)
- Ensure consistent chunk sizing and re-runnability
- Reduce manual copy-paste in the terminal

All wrapper scripts require the configuration directory to be passed explicitly
via the `--config-dir` argument. Hardcoded configuration paths are intentionally
avoided to ensure reproducibility and to prevent accidental execution on the
wrong run directory.

---

## 4. Step 1: run_extract.py

Run:

    python run_extract.py --config-dir <config_dir>

Example:

    python run_extract.py --config-dir runs/collection_06012026

Purpose:
    Extract raw datasets from the source databases and store them as Parquet files
    in the configured extract directory.

Required configuration:
    - etl/paths.yml<br>
    - etl/files_and_folders.yml<br>

Typical execution pattern:
    The extraction is run per data source and per origin database, usually in chunks.

Notes:

- Because `config_dir` is explicit, the same script can be safely reused across
  different collections without code changes.
- If a given extraction step fails due to corrupted rows, re-run only that specific
  (raw_data_name, db_origin) combination.
- If interruption occurs during extraction, running again will place the script where it
  stopped in the last execution.
- Keep chunk sizes stable (do not change!) within the same collection to avoid mixing 
  chunk boundaries across re-runs.

Expected outputs:
    Parquet files under:

    extract_path/<raw_data_folder>/*.parquet

Where extract_path is resolved from `etl/paths.yml` and `etl/files_and_folders.yml`.

---

## 5. Step 2: run_transform.py

Purpose:
    Orchestrate transformation steps over extracted raw data, including cleaning,
    normalization, BI-RADS extraction and inference, auxiliary model training,
    and final source merging.

Inputs:
    - Extracted raw Parquet files from run_extract.py

Outputs:
    - Processed data containing extract and infered BI-RADS from raw mammogram exams.
    - Processed data for each type of raw data.
    - Classified biopsy reports.
    - Fitted TF-IDF models.
    - Merged data from all the sources.

Run (full pipeline):

    python run_transform.py --config-dir <config_dir>

Run a single step:

    python run_transform.py --config-dir <config_dir> --step birads-extract

Resume from a given step:

    python run_transform.py --config-dir <config_dir> --from preprocess

Inspect execution plan without running:

    python run_transform.py --config-dir <config_dir> --all --dry-run

---

## 6. Step 3: run_load.py

Purpose:
    Apply eligibility criteria, build the modeling cohort dataset,
    and precompute historical aggregates of past mammograms for efficient 
    training.

Inputs:
    - Transform the merged dataset from run_transform.py

Outputs:
    - load_path/<dataset_name>/... (final datasets)

Run:

    python run_load.py --config-dir <config_dir>

Configuration files can be changed in the script itself, such as `strat_cfg` and `gb_sizes`. `strat_cfg` corresponds to the follow-up configuration file.
For instance, if we want to change inclusion criteria, such as age interval, `strat_cfg` will defined which configuration file to use. `gb_sizes` refers to the dedicated memory
space for the precomputed vectors representing the aggregates of past mammogram embeddings. Since the precomputed vectors are stored in a LMDB database, the memory space should be 
predetermined.

---

## 7. Step 4: run_tuning.py

Purpose:
    Perform hyperparameter tuning experiments using model/tuning/*.yml settings.

Inputs:
    - Final cohort dataset
    - tuning config selection (defaults mapping)

Outputs:
    - tuning logs
    - tuning results tables
    - candidate checkpoints (depending on implementation)

Run:

    python run_tuning.py --config-dir <config_dir>

Configuration files that can be changed in the script are: `tuning_cfg`, `split_cfg` and `followup_cfg`. `tuning_cfg` refers to the configuration file to determine the parameters
of the tuning experiment (seed, number of trials, output folder, etc). `split_cfg` refers to the configuration file to determine the data splitting parameters, such as training and
test fractions and which BI-RADS to include in the cohort (this should be the same splitting of training later). `followup_cfg` refers to the cohort to be used for tuning. it should
refer to a configuration file already used during execution of `run_load.py`.

---

## 8. Step 5: run_training_best.py

Purpose:
    Train a final model using the best configuration selected from tuning.

Inputs:
    - Final cohort dataset
    - Selected best hyperparameters (output of tuning)

Outputs:
    - Final model checkpoints
    - Training logs
    - Selected config snapshot (recommended)

Run:

    python run_training_best.py --config-dir <config_dir>

Configuration files that can be changed in the script are: `tuning_cfg`, `split_cfg` and `followup_cfg`. `tuning_cfg` refers to the configuration file used during the referred tuning
experiment and it will define the best parameter configuration obtained. `split_cfg` refers to the configuration file to determine the data splitting parameters, such as training and
test fractions and which BI-RADS to include in the cohort (it should be the same splitting used during tuning). `followup_cfg` refers to the cohort to be used for training. it should
refer to a configuration file already used during execution of `run_load.py`.

---

## 9. Step 6: run_eval.py

Purpose:
    Evaluate the final model on the held-out test set, generate all final metrics,
    plots, and result tables.

Inputs:
    - Final checkpoint(s)
    - Evaluation configuration

Outputs:
    - Metrics tables
    - Plots (ROC, PR, calibration, etc.)
    - Explainability artifacts (if enabled)

Run:

    python run_eval.py --config-dir <config_dir>

Configuration files that can be changed in the script are: `tuning_cfg`, `split_cfg` and `followup_cfg`. `tuning_cfg` refers to the configuration file used during the referred tuning
experiment and it will define the best parameter configuration obtained. `split_cfg` refers to the configuration file to determine the data splitting parameters, such as training and
test fractions and which BI-RADS to include in the cohort (it should be the same splitting used during tuning). `followup_cfg` refers to the cohort to be used for training. it should
refer to a configuration file already used during execution of `run_load.py`.

---

## 10. Recommended run discipline

1) Do not modify YAML configs mid-run without recording the change.
2) For each run collection, snapshot:
   - the defaults mapping used
   - the selected YAML files
3) Avoid partial reruns unless you understand upstream dependencies.

A good habit is to treat each configuration directory as immutable once results
are generated, and create a new configuration directory for new collections. Tuning and
training experiments could be done freely within the same configuration directory.