🇺🇸 English | [🇧🇷 Português](configuration_setup.pt-BR.md)

# Configuration Setup

This project uses a folder-based configuration system. A “configuration directory”
is a self-contained folder that contains a fixed schema of subfolders and YAML files.
The class `ConfigInterface` (and all classes that inherit from it) relies on this
directory to load configuration, resolve defaults, and construct all filesystem
paths used across the pipeline. This pattern is used for reducing boilerplate code
and for centralizing important (and numerous) parameter/argument assignments.

The configuration system is implemented in:

- hapcancer/config_manager.py
  - ConfigManager (resolve directory schema and load files)
  - ConfigInterface (high-level configuration setup)

In this document, we explain the implementation details involved.

---

## 1. What a “configuration directory” is

A configuration directory is a folder with a known structure that contains
YAML configuration files grouped by pipeline area. The illustrative example
below follows the current schema for a configuration directory:

    collection_folder/
      etl/
        paths.yml
        files_and_folders.yml
        fields.yml
        followup/
          age_strat_18_75.yml
          age_strat_40_75.yml
        birads_classifier/
          birads_clf_001.yml
        bmi_model/
          bmi_model_001.yml
        embeddings/
          tfidf_001.yml
          tfidf_002.yml

      model/
        extra.yml
        fields.yml
        splits/
          split_001.yml
          split_002.yml
        training_experiments/
          base.yml
          trial_*.yml
        tuning/
          bce_all_001.yml
          bce_all_002.yml

This directory can be swapped (e.g., different experiments or collections),
and the code will load a different configuration tree accordingly.

### 1.1. Descriptions

1. **ETL**  
   Coordinates the paths, filenames, and parameters related to the first extraction
   from Hapvida databases up to the final dataset ready for model development.

   1.1. **paths.yml**  
        Sets IDs and storage paths for ETL. Also sets the path to the `.env` file
        containing database access credentials.<br>

   1.2. **files_and_folders.yml**  
        Sets folder names for each ETL stage and output filenames.<br>

   1.3. **fields.yml**  
        Fields used during analysis. Currently fixed (and possibly unnecessary
        as a configuration file).<br>

   1.4. **followup**  
        Sets important cohort restrictions.<br>

   1.5. **birads_classifier**  
        A BI-RADS classifier (from reports) is built during ETL. Here we set the
        model parameters.<br>

   1.6. **bmi_model**  
        This can be removed later. The only thing needed is a BMI estimator from
        sex and age.<br>

   1.7. **embeddings**  
        Parameters for the TF-IDF embedding model. Other models can be included
        later.<br>

2. **MODEL**  
   Coordinates model architecture definitions, training procedures, evaluation
   setup, and experiment variants.

   2.1. **extra.yml**  
        Global model-level paths and runtime options (logging, checkpoints, device).<br>

   2.2. **fields.yml**  
        Defines feature columns, targets, follow-up windows, and eligibility masks.<br>

   2.3. **splits**  
        Defines data splitting strategies and related parameters.<br>

   2.4. **training_experiments**  
        Training configurations (architecture and training hyperparameters).<br>

   2.5. **tuning**  
        Hyperparameter tuning configuration.<br>

---


### 1.2. High-level access via `ConfigInterface`

The following pattern is used for interaction with the configuration folder
and configuration files:

```python
from hapcancer.config_manager import ConfigInterface

config_dir = "path/to/config_folder"
config_defaults = {
    "birads_classifier": "birads_clf_001.yml",
    "embeddings": "tfidf_001.yml",
    "split": "split_001.yml",
    "followup": "age_strat_18_75.yml",
    "bmi_model": "bmi_model_001.yml",
    "tuning": "bce_all_001.yml",
    "training_experiments": None
}

config = ConfigInterface(config_dir, config_defaults)
```

When `ConfigInterface` is instantiated, it handles several values defined in the parsed configuration files. Most classes defined in the software inherit directly from `ConfigInterface`, sharing the same initialization pattern.

```python
from hapcancer.etl.extract.extractor import Extractor

with Extractor(config_dir, config_defaults) as extractor:
    extractor.fetch_mammograms_paginated(
        db_origin="hsp",
        start_year=2016,
        timer=0.5,
        chunk_size=200_000,
        verbose=True,
        max_retries=5,
    )
```

Within the same configuration folder, it is possible to test different parameters for different parts of the pipeline. For example, when using a TF-IDF model to represent mammogram reports, multiple embedding configurations may exist under `etl/embeddings/` (e.g., `tfidf_001.yml`, `tfidf_002.yml`). The config_defaults dictionary defines the exchangeable components of the pipeline. To change the embedding model used by default, one can set:

```python
config_defaults["embeddings"] = "tfidf_002.yml"
```

and reinitialize the class.

During standard production runs, this level of interaction is not required. These details are included in case further development or experimentation is needed. `ConfigInterface` is built on top of the `ConfigManager`.

## 2. How configuration files are loaded

On initialization, `ConfigManager` scans the configuration directory recursively
and builds an in-memory tree of the same shape as the folder structure.

Key behaviors:

- Each folder becomes a nested dictionary key.
- Each file is loaded and stored under a key equal to its filename stem.
  Example:
      etl/embeddings/tfidf_001.yml
  is accessible as:
      ["etl", "embeddings", "tfidf_001"]

Notes:

- If two files in the same folder have the same stem (e.g., a.yml and a.json),
  one will overwrite the other. Avoid stem collisions.
- The file loader is `hapcancer.etl.utils.load_config_file(filepath)`.

### 2.1. Access pattern

Configuration is accessed by a list of keys:

    cfg_manager.get(["etl", "paths"])
    cfg_manager.get(["etl", "embeddings", "tfidf_001"])
    cfg_manager.get(["model", "splits", "split_001"])

This path-based API is intentionally explicit and mirrors the folder hierarchy.

### 2.2. Defaults: selecting “which config to use”

Many folders contain multiple candidate YAML files (e.g., multiple split configs,
multiple embedding configs, multiple follow-up definitions). The system selects
which one to use via a “defaults mapping”.

Defaults are set by calling:

    cfg_manager.set_defaults(config_defaults)

The defaults mapping points to filenames (strings). Internally, the manager
converts each filename to its stem for lookup in the tree.

Example:

    config_defaults = {
      "followup": "age_strat_30_75.yml",
      "embeddings": "tfidf_003.yml",
      "birads_classifier": "birads_clf_001.yml",
      "bmi_model": "bmi_model_001.yml",
      "split": "split_001.yml",
      "training_experiments": "base.yml",
      "tuning": "bce_all_001.yml",
      "eval": null
    }

After validation, these are stored as stems:

    followup = "age_strat_30_75"
    embeddings = "tfidf_003"
    split = "split_001"
    ...

Defaults are validated by the Pydantic schema:

    hapcancer.schemas.validation_models.ConfigDefaults

If defaults do not conform to the schema, initialization fails.

Observation:
Accessing and using `ConfigManager` methods and attributes is only needed in case more
functionality is required beyond what `ConfigInterface` provides. In typical usage,
high-level classes should inherit from `ConfigInterface` and avoid interacting with
`ConfigManager` directly.

---

## 3. What ConfigInterface loads by default

`ConfigInterface` is the standard high-level interface used by ETL and modeling
classes. It wraps `ConfigManager`, applies defaults, and exposes commonly used
configuration blocks as attributes.

ETL configs loaded:

    etl/paths.yml                       -> self.paths_cfg
    etl/files_and_folders.yml           -> self.files_and_folders_cfg
    etl/fields.yml                      -> self.fields_cfg

    etl/followup/<DEFAULT>.yml          -> self.followup_cfg
    etl/birads_classifier/<DEFAULT>.yml -> self.birads_clf_cfg
    etl/bmi_model/<DEFAULT>.yml         -> self.bmi_models_cfg
    etl/embeddings/<DEFAULT>.yml        -> self.embeddings_cfg

Model configs loaded:

    model/extra.yml                          -> self.extra_cfg
    model/fields.yml                         -> self.model_fields_cfg
    model/splits/<DEFAULT>.yml               -> self.split_cfg
    model/training_experiments/<DEFAULT>.yml -> self.training_cfg
    model/tuning/<DEFAULT>.yml               -> self.tuning_cfg
    model/eval/<DEFAULT>.yml                 -> self.eval_cfg   (if configured)

This means: if the folder schema is present and defaults are valid,
the complete pipeline can locate all needed configuration through this interface.

---

## 4. Path resolution and run IDs

The file etl/paths.yml must define at least:

- extract.path and extract.id
- transform.path and transform.id
- load.path and load.id
- (optional) logging_path entries

`ConfigInterface` combines base paths with IDs to create run-specific directories:

    extract_path   = extract.path   / extract.id
    transform_path = transform.path / transform.id
    load_path      = load.path      / load.id

It also defines derived paths such as:

    transformed_data_path = transform_path / "transformed"
    dataset_path = load_path / followup_cfg["dataset_name"]   (if followup is configured)

These IDs are how the pipeline separates different runs/collections on disk.

---

## 5. Expected folder names and side effects (important)

`ConfigInterface` reads folder names from etl/files_and_folders.yml and creates
multiple directories on initialization.

This has side effects:

- Directories are created automatically if missing.
- Incorrect paths or IDs in configuration will create directories in the wrong place.
- Always verify paths.yml before running.

Examples of directories created:

- Raw extract folders (mammograms, biopsy, anamnesis, patient/person/user)
- Transform folders (birads outputs, fitted models, cache, embedding store)
- Load folder and dataset folder

---

## 6. Minimal checklist for a new configuration directory

To create a new configuration folder (new collection), ensure:

1) Folder structure exists:

    config_dir/<br>
      etl/<br>
      model/<br>

2) Required base files exist:

    etl/paths.yml<br>
    etl/files_and_folders.yml<br>
    etl/fields.yml<br>
    model/extra.yml<br>
    model/fields.yml<br>

3) Required option subfolders exist (even if only one config initially):

    etl/followup/<br>
    etl/embeddings/<br>
    etl/birads_classifier/<br>
    etl/bmi_model/<br>
    model/splits/<br>
    model/training_experiments/<br>
    model/tuning/<br>

4) Defaults mapping selects valid filenames that exist in those folders.

5) All paths in paths.yml point to valid writable locations.

---

## 7. Practical usage pattern

Recommended usage:

- Create a new configuration directory per “run collection” (e.g., per date or study batch).
- Keep all experiment variants as separate YAML files under model/training_experiments/ and model/tuning/.
- Version-control the configuration folder (or at least the YAMLs) so experiments are reproducible.

The configuration folder is part of the scientific record of each experiment set.

---

## 8. Common pitfalls and guardrails

1) Stem collisions
   If you create two files with the same stem under the same folder, one will overwrite the other in memory.
   Always keep unique stems per folder.

2) Missing defaults
   If `config_defaults` references a filename that does not exist, the default will become a stem that cannot
   be resolved later. Make sure defaults always point to real files in the configuration directory.

3) Silent `None` paths in access
   The helper `access_nested_dict` returns `None` immediately if any key in the path is `None`.
   If you allow optional defaults (e.g., eval = null), ensure your downstream code handles missing configs.

4) Side-effect directories
   `ConfigInterface` creates directories on initialization.
   Always validate that etl/paths.yml points to the correct output root before running.

5) Reproducibility: always snapshot configs
   For each run, copy the full configuration directory (or at least the YAML files used) into the output folder.
   This prevents “config drift” from breaking reproducibility.

---

## 9. Suggested improvement (optional): run metadata snapshot

A recommended pattern for experiment tracking:

- Create an output folder for each run (extract_id, transform_id, load_id).
- Save a copy of:
    - the defaults mapping used
    - the resolved config stems
    - the full relevant YAMLs (or the entire configuration folder)
    - the git commit hash

This can be implemented as a small helper function called at the beginning of each CLI run.

This is not strictly required, but it makes results auditable and reproducible.

---

# Schemas for configuration files

This can be improved (probably very messy as it is).

## etl/paths.yml

    extract:
        id: str
        path: str
        env_path: str # path to .env
        logging_path: str
    transform:
      id: str
      path: str
      logging_path: str
    load:
      id: str
      path: str
      logging_path: str

## etl/files_and_folders.yml

    extract:
      folders:
        biopsy: str
        mammogram_exams: str
        anamnesis: str
        user_person_data: str
    transform:
      folders:
        fitted_models: str
        embedding: str
        embedding_store: str
        birads: str
        birads_model: str
        cache: str
      transformed_files:
        anamnesis: str
        person: str
        user: str
        similarity_data: str
        valid_person_patient: str
        breast_biopsy: str
        breast_biopsy_classified: str 
        person_biopsy: str
        person_mammogram: str
    load:
      load_files:
        merged_data: str # with .parquet extension
        seq_per_mammogram_filename: str # with .parquet extension
        final_data_before_eligibility_filename: str # with .parquet extension
        final_data_with_eligibility_filename: str # with .parquet extension
        precomputed_filename: str # with .lmdb extension

## etl/fields.yml (legacy)

    fields:
      person_id: CD_PESSOA
      patient_id: CD_PACIENTE
      user_id: CD_USUARIO
      mammogram_id: CD_ATENDIMENTO
      mammogram_id_final: key
      person_birthdate: DT_NASCIMENTO_FUNDACAO
      mammogram_date: DT_ATENDIMENTO
      mammogram_text: DS_LAUDO_MEDICO
      anamnesis_id: CD_ATENDIMENTO
      anamnesis_date: DT_ATENDIMENTO
      anamnesis_raw_features: [ 
        DS_MENARCA, DS_MENOPAUSA, NU_GESTACAO_, NU_GESTACAO_ABORTO, FL_CA_MAMA_MAE, 
        FL_CA_MAMA_IRMA, FL_CA_MAMA_AVO, FL_CA_MAMA_TIA, FL_MASTECTOMIA_MD, 
        FL_MASTECTOMIA_ME, FL_PLASTICA_ME , DT_PLASTICA_ME, FL_PLASTICA_MD , 
        DT_PLASTICA_MD, DT_ATENDIMENTO, CD_ATENDIMENTO, FL_ALEITAMENTO 
      ] 

## etl/birads_classifier/*.yml

    phase: transform
    birads_classifier:
      max_samples_per_class: int
      val_size: float
      split_random_state: int
      tfidf_max_features: int
      ngram_range_max: in
      clf_penalty: str
      clf_solver: str
      re_processed_filename: str # with .parquet extension
      ml_infered_filename: str # with .parquet extension

## etl/followup/*.yml

    phase: load
    dataset_name: str
    followup:
      minimum_age: int
      maximum_age: int
      grace_period_start_in_days: int # -- set an interval of D days after the index mammogram so that we do not consider follow-up in this period.
      total_months_of_followup: 120 # keep
      start_date_mammogram: str # YYYY-MM-DD format
      cohort_end_date: str # YYYY-MM-DD format
      birads_5:
        validation_version: 1
        validation_interval_months: 6
        validation_explanation: [
          v1 -> keeps BI-RADS 5 only if within an interval of time after its date there isn't any benign BI-RADS or benign biopsy.,
          v2 -> keeps BI-RADS 5 if there is either a BI-RADS 6 or confirmatory biopsy after its date.
        ]
    precomputed:
      path: str
      filename: str (with extension)

## etl/embeddings/*.yml

    phase: transform
    embedding_id: str
    tfidf: # applies only when using tf-idf
      svd: bool
      svd_dim: int
      max_features: int
      min_df: int
      max_df: float
      ngram_range: list[int]
      model_name: str

## etl/bmi_model/*.yml (to be removed)

    bmi_model:
        path: str
        linreg_model: str # with .pkl extension
        randfor_model: str # with .pkl extension

## model/fields.yml

    fields:
        feature_columns: [
          mammogram_id, mammogram_current_result, monthly_payment_min, monthly_payment_max, 
          bmi, menarche_age, age_at_first_mammogram, age_at_mammogram, ca_mama_mae_cat_-1.0, 
          ca_mama_mae_cat_0.0, ca_mama_mae_cat_1.0, ca_mama_irma_cat_-1.0, ca_mama_irma_cat_0.0, 
          ca_mama_irma_cat_1.0, ca_mama_avo_cat_-1.0, ca_mama_avo_cat_0.0, ca_mama_avo_cat_1.0, 
          ca_mama_tia_cat_-1.0, ca_mama_tia_cat_0.0, ca_mama_tia_cat_1.0, menopause_category_ordered, 
          is_missing_children, is_missing_miscarriage, number_of_children, number_of_miscarriage, 
          zipcode_embedding_0, zipcode_embedding_1, zipcode_embedding_2, zipcode_embedding_3, 
          zipcode_embedding_4, zipcode_embedding_5, zipcode_embedding_6, zipcode_embedding_7, 
          breastfeeding_cat
        ] # fixed for now
        event_indicator_columns: [
          event_indicator_1yr, event_indicator_2yr, event_indicator_3yr, 
          event_indicator_4yr, event_indicator_5yr, event_indicator_6yr,
          event_indicator_7yr, event_indicator_8yr, event_indicator_9yr,
          event_indicator_10yr
        ] # fixed
        followup_columns: [
          14days_1yr_followup, 1yr_2yr_followup, 2yr_3yr_followup, 
          3yr_4yr_followup, 4yr_5yr_followup, 5yr_6yr_followup,
          6yr_7yr_followup, 7yr_8yr_followup, 8yr_9yr_followup,
          9yr_10yr_followup
        ] # fixed
        multiyear_eligibility_columns: [
          eligibility_0yr_1yr, eligibility_1yr_2yr, eligibility_2yr_3yr, 
          eligibility_3yr_4yr, eligibility_4yr_5yr, eligibility_5yr_6yr,
          eligibility_6yr_7yr, eligibility_7yr_8yr, eligibility_8yr_9yr,
          eligibility_9yr_10yr
        ] # fixed
        birads_column: str

## model/extra.yml

    extra:
        checkpoint_path: str
        logging_path: str
        tuning_path: str
        eval_path: str
        use_amp: true
        device: str # either 'cpu' or 'cuda'
        verbose: bool
        save_epochs: bool
        save_best_epochs: bool

## model/split/*.yml

    description: str
    split:
      training_size: float # < 1.0
      test_size: float # sums to 1.0 with training_size
      kfold: int
      seed: int
      birads: list[int] # e g. [1,2,3] 

## model/tuning/*.yml

    description: tuning for several age strats and different target years. 
    model:
      mammogram_input_dim: 5000
      extra_features_dim: 33
      embed_dim: 128
      transformer_num_heads: int # not needed for tf-idf embeddings
      transformer_num_layers: int # not needed for tf-idf embeddings
      transformer_dropout: float # not needed for tf-idf embeddings
      freeze_transformer: bool
      sigmoid_output: bool
      mlp_config:
        hidden_layers: list[int]
        dropout: float
        activation: str  # options: relu, gelu and mish
        use_batchnorm: true
        sigmoid: false

    training:
      epochs: int
      early_stop: bool
      patience: int
      learning_rate: float
      weight_decay: float
      warmup_steps: int
      loss_function: cross_entropy
      optimizer: str # 'adam' or 'sgd'
      max_training_batches_per_epoch: int
      max_validation_batches_per_epoch: int

    tuning:
      path: str # for logging
      num_trials: int
      optim_seed: int
      study_name: str


## model/training_experiments/*.yml

    model:
      mammogram_input_dim: 5000
      extra_features_dim: 33
      embed_dim: 128
      transformer_num_heads: int # not needed for tf-idf embeddings
      transformer_num_layers: int # not needed for tf-idf embeddings
      transformer_dropout: float # not needed for tf-idf embeddings
      freeze_transformer: bool
      sigmoid_output: bool
      mlp_config:
        hidden_layers: list[int]
        dropout: float
        activation: str  # options: relu, gelu and mish
        use_batchnorm: true
        sigmoid: false
    training:
      epochs: int
      early_stop: bool
      patience: int
      learning_rate: float
      weight_decay: float
      warmup_steps: int
      loss_function: cross_entropy
      optimizer: str # 'adam' or 'sgd'
      max_training_batches_per_epoch: int
      max_validation_batches_per_epoch: int
      model_name: str
      target_year: int
      sampling_strategy: str # either 'undersampling' or 'oversampling'
      negative_to_positive_ratio: int
      num_workers: int
      batch_size: int
      pretrained:
        load: bool
        model_name: str
        file_name: best_model.pt
    description: str

