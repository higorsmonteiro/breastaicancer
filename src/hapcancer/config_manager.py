from pathlib import Path
from pydantic import BaseModel, ValidationError
from typing import Optional, Iterable, List, Union, Any, Dict

import pandas as pd
import numpy as np
import datetime as dt

from hapcancer.etl.utils import sha1, load_config_file, batching_parquet_file
from hapcancer.schemas.enums import MammogramColumns, BiopsyColumns, PersonColumns, AnamnesisColumns
from hapcancer.schemas.validation_models import ConfigDefaults


#  ----------------------------------------------------------------
#  ---------------------- Auxiliar Functions ----------------------
# -----------------------------------------------------------------

def access_nested_dict(dict_: dict, ordered_keys: List[str]) -> Dict[Any, Any]:
    """
        Traverse a nested dictionary using an ordered list of keys.

        This helper walks a dictionary-of-dictionaries structure by repeatedly
        indexing into the current level with each key in `ordered_keys`.

        Important behavior:
        - If any key in `ordered_keys` is `None`, the function returns `None`
          immediately.
        - If a key is not found in the current dictionary level, the traversal
          does *not* raise; it simply stops descending for that key and continues
          to the next key, returning whatever dictionary/value was last reached.

        Args:
        ----------
        dict_ : dict
            Root dictionary to traverse.
        ordered_keys : list[str]
            Keys representing the hierarchical path to traverse.

        Returns:
        -------
        cur_dict: dict.
            The nested value reached after walking the provided keys. If traversal
            cannot fully descend due to missing keys, the last reachable value
            is returned. If any key is `None`, returns `None`.

        Notes
        -----
        This function does not validate that the intermediate values are
        dictionaries. If the traversal reaches a non-dict value and subsequent
        keys are provided, membership checks may fail (or raise) depending on the
        type of that value.
    """
    cur_dict = dict_
    for key in ordered_keys:
        if key is None: return None
        if key in cur_dict:
            cur_dict = cur_dict[key]
    return cur_dict

def build_directory_tree(path: Path) -> Dict[Any, Any]:
    """
        Build a nested dictionary representation of a directory tree.

        Given a directory path, this function produces a nested dictionary where:
        - Each directory is represented as a dict keyed by its immediate subfolder names.
        - Each directory-level dictionary may include a special key `'files'`
          containing a list of file paths (as strings) that live under that directory.

        Args:
        ----------
        path : pathlib.Path
            Root directory to scan.

        Returns:
        -------
        nested_dir_tree: dict
            Nested dictionary representing the directory structure. Folder names
            map to sub-dictionaries. Each folder dictionary may include a `'files'`
            list containing stringified absolute/relative file paths discovered via
            recursive search.
    """
    def nested_directory_tree(path: Path):
        # create a nested dictionary representing the directory (no files)
        nested_tree = {}
        if path.is_dir():
            next_dirs = [ elem for elem in path.glob("*") if elem.is_dir() ]
            for next_dir in next_dirs:
                nested_tree[next_dir.name] = nested_directory_tree(next_dir)
        return nested_tree
    
    # -- nested dict without files
    nested_dir_tree = nested_directory_tree(path)
    # -- for each dictionary, add a 'files' key with the filepaths for its directory level
    rfiles = [ elem for elem in path.rglob("*") if elem.is_file() ]
    for current_filepath in rfiles:
        tree_dict_path = current_filepath.parent.relative_to(path).parts
        current_level_dict = access_nested_dict(nested_dir_tree, tree_dict_path)
        if 'files' in current_level_dict:
            current_level_dict['files'].append(str(current_filepath))
        else: current_level_dict['files'] = [ str(current_filepath) ]
    return nested_dir_tree

def create_config_tree(config_dir: str | Path):
    """
        Create a configuration tree by loading config files from a directory structure.

        This function:
        1) Builds a directory tree (folders as dicts and file paths under `'files'`).
        2) Recursively traverses that tree and, whenever a `'files'` list is present,
           loads each file with `load_config_file(filepath)`.
        3) Replaces the `'files'` key with keys corresponding to each file stem
           (`Path(filepath).stem`) mapping to the loaded configuration content.

        Args:
        ----------
        config_dir : str or pathlib.Path
            Root configuration directory.

        Returns:
        -------
        tree: dict
            A nested dictionary mirroring the directory structure, where configuration
            files are loaded into memory and exposed under keys equal to their file stem.

        Notes
        -----
        - If two files in the same folder share the same stem (e.g., `a.yml` and `a.json`),
          later entries will overwrite earlier ones due to dictionary key collisions.
        - The loader used is `hapcancer.etl.utils.load_config_file`.
    """
    path = config_dir if type(config_dir)==Path else Path(config_dir)
    dir_tree = build_directory_tree(path)
    tree = dict(dir_tree)

    def traverse_nested_dict(dir_tree: dict):
        tree = dir_tree
        temp = {}
        for key, value in tree.items():
            if type(value)==dict:
                traverse_nested_dict(value)
            else: # or a list
                for filepath in value:
                    temp[Path(filepath).stem] = load_config_file(filepath)
        if temp:
            #print(tree.keys())
            tree.pop('files')
            for key, value in temp.items():
                #print(key, value)
                tree[key] = value
        return tree
    tree = traverse_nested_dict(tree)
    return tree

class ConfigManager:
    '''
        Manage access to configuration files stored in a directory.

        `ConfigManager` builds an in-memory configuration tree from a configuration
        directory and provides:
        - Path-based access into the nested tree via `.get([...])`
        - A mechanism for declaring and validating default configuration selections
          (e.g., which embedding config to use) via `.set_defaults(...)`
        - Dictionary-style access to default selections via `__getitem__`

        Args:
        ----------
        config_dir : str or pathlib.Path
            Root directory containing configuration subfolders and config files.

        Attributes:
        ----------
        config_dir : str or pathlib.Path
            Root configuration directory.
        config_defaults : dict or None
            Mapping of pipeline component -> selected config name (stored as file stem),
            validated against `hapcancer.schemas.validation_models.ConfigDefaults`.
    '''
    def __init__(
        self, 
        config_dir: Union[str, Path]
    ):
        self.config_dir = config_dir
        self.config_defaults = None
        self.tree = create_config_tree(self.config_dir)

    def get(self, tree_path: List[str]):
        """
            Retrieve a configuration object from the loaded configuration tree.

            Args:
            ------
            tree_path : list[str]
                Hierarchical path of keys used to index into the configuration tree.
                Example:
                    `["etl", "birads_classifier", "birads_clf_001"]`

            Returns:
            -------
            Dict
                The configuration object located at the given path, typically a dictionary
                returned by `load_config_file`.

            Notes
            -----
            This method delegates traversal to `access_nested_dict`.
        """
        return access_nested_dict(self.tree, tree_path)
    
    def set_defaults(self, config_defaults: dict):
        '''
            Set and validate the default configuration selections for pipeline components.

            This stores a mapping from pipeline component name to a default config filename.
            After validation via `ConfigDefaults`, each default value is converted to the
            file stem (e.g., `"tfidf_001.yml"` -> `"tfidf_001"`) for convenient lookup
            within the configuration tree.

            Args:
            ------
            config_defaults : dict
                Mapping of component name -> filename (or `None`) indicating which
                configuration should be used by default for each pipeline component.

            Raises
            ------
            ValueError
                If `config_defaults` does not conform to the expected Pydantic schema
                (`hapcancer.schemas.validation_models.ConfigDefaults`).

            Side Effects
            ------------
            Sets `self.config_defaults`. On validation failure, resets it to `None`.
        '''
        self.config_defaults = dict(config_defaults)
        try:
            ConfigDefaults.model_validate(self.config_defaults)
            # -- after validated, make each field a Path object for easy stemming.
            for k, v in self.config_defaults.items():
                self.config_defaults[k] = Path(v).stem if type(v)==str else None
        except ValidationError as e:
            self.config_defaults = None
            raise ValueError(f"Default does not follow the expected schema: error: {e.errors()}")

    def __getitem__(self, key):
        if self.config_defaults is None:
            raise ValueError(f"defaults are not defined.")
        return self.config_defaults[key]
             

class ConfigInterface:
    """
        High-level configuration interface for ETL and modeling components.

        This class:
        - Instantiates a `ConfigManager`
        - Validates and stores default config selections
        - Loads commonly used configuration sections into attributes
        - Builds and stores key filesystem paths used across the pipeline
        - Ensures required directory structure exists by creating folders on disk

        Args:
        ----------
        config_dir : str
            Root directory containing configuration files.
        config_defaults : dict
            Mapping of component -> default config filename (or `None`), validated by
            `ConfigManager.set_defaults`.

        Attributes (selected)
        ---------------------
        cfg_manager : ConfigManager
            Underlying configuration manager.
        paths_cfg, files_and_folders_cfg, fields_cfg, followup_cfg, ...
            Loaded configuration dictionaries for major pipeline sections.
        extract_path, transform_path, load_path : pathlib.Path
            Main pipeline output directories (with run IDs applied).
        raw_mammograms_folder_path, raw_biopsy_folder_path, raw_patient_folder_path : pathlib.Path
            Key input folders for raw data.
        processed_birads_folder_path, fitted_models_folder_path : pathlib.Path
            Key output folders for processed artifacts and models.

        Side Effects
        ------------
        Creates multiple directories on initialization using `Path.mkdir(...)`.
    """
    def __init__(
            self,
            config_dir: str,
            config_defaults: dict 
        ):
        # -- create the manager of all configuration files of the directory  
        self.cfg_manager = ConfigManager(config_dir)
        self.cfg_manager.set_defaults(config_defaults)

        # -- main configuration files: ETL
        self.paths_cfg = self.cfg_manager.get(["etl", "paths"])
        self.files_and_folders_cfg = self.cfg_manager.get(["etl", "files_and_folders"])
        self.fields_cfg = self.cfg_manager.get(["etl", "fields"])
        self.followup_cfg = self.cfg_manager.get(["etl", "followup", self.cfg_manager["followup"]])
        self.birads_clf_cfg = self.cfg_manager.get(["etl", "birads_classifier", self.cfg_manager["birads_classifier"]])
        self.bmi_models_cfg = self.cfg_manager.get(["etl", "bmi_model", self.cfg_manager["bmi_model"]])
        self.embeddings_cfg = self.cfg_manager.get(["etl", "embeddings", self.cfg_manager["embeddings"]])

        # -- main configurations files: Model
        self.extra_cfg = self.cfg_manager.get(["model", "extra"])
        self.model_fields_cfg = self.cfg_manager.get(["model", "fields"])
        self.eval_cfg = self.cfg_manager.get(["model", "eval", self.cfg_manager["eval"]])
        self.split_cfg = self.cfg_manager.get(["model", "splits", self.cfg_manager["split"]])
        self.training_cfg = self.cfg_manager.get(["model", "training_experiments", self.cfg_manager["training_experiments"]])
        self.tuning_cfg = self.cfg_manager.get(["model", "tuning", self.cfg_manager["tuning"]])

        # -- set main path variables
        self.extract_id = self.paths_cfg["extract"]["id"]
        self.transform_id = self.paths_cfg["transform"]["id"]
        self.load_id = self.paths_cfg["load"]["id"]
        self.extract_path = Path(self.paths_cfg["extract"]["path"]).joinpath(self.extract_id)
        self.transform_path = Path(self.paths_cfg["transform"]["path"]).joinpath(self.transform_id)
        self.transformed_data_path = self.transform_path.joinpath("transformed")
        self.load_path = Path(self.paths_cfg["load"]["path"]).joinpath(self.load_id)
        self.dataset_path = self.load_path.joinpath(self.followup_cfg["dataset_name"]) if self.followup_cfg is not None else None
        # -- collect ETL folders names and create them (if do not exist)
        self.extract_folders = self.files_and_folders_cfg['extract']['folders']
        self.transform_folders = self.files_and_folders_cfg['transform']['folders']

        # -- create ETL directories
        # -- create (if not exists) extract folders
        for pth in ( self.extract_path.joinpath(cur_folder) for key, cur_folder in self.extract_folders.items()):
            pth.mkdir(parents=True, exist_ok=True)
        # -- create (if not exists) transform folders
        for pth in (self.transform_path.joinpath(cur_folder) for key, cur_folder in self.transform_folders.items()):
            pth.mkdir(parents=True, exist_ok=True)
        self.transformed_data_path.mkdir(parents=True, exist_ok=True)
        # -- create (if not exists) load folder
        self.load_path.mkdir(exist_ok=True, parents=True)
        if self.dataset_path is not None: self.dataset_path.mkdir(exist_ok=True, parents=True)

        # -- some useful paths
        # -- raw data paths
        self.raw_mammograms_folder_path = self.extract_path.joinpath(self.files_and_folders_cfg['extract']['folders']['mammogram_exams'])
        self.raw_anamnesis_folder_path = self.extract_path.joinpath(self.files_and_folders_cfg['extract']['folders']['anamnesis'])
        self.raw_biopsy_folder_path = self.extract_path.joinpath(self.files_and_folders_cfg['extract']['folders']['biopsy'])
        self.raw_patient_folder_path = self.extract_path.joinpath(self.files_and_folders_cfg['extract']['folders']['user_person_data'])
        # ---- bi-rads paths
        self.processed_birads_folder_path = self.transform_path.joinpath(self.files_and_folders_cfg['transform']['folders']['birads'])
        self.fitted_models_folder_path = self.transform_path.joinpath(self.files_and_folders_cfg['transform']['folders']['fitted_models'])
        self.embedding_store_folder_path = self.transform_path.joinpath(self.files_and_folders_cfg['transform']['folders']['embedding_store']) # so far, not used
        self.cache_path = self.transform_path.joinpath(self.files_and_folders_cfg['transform']['folders']['cache']) # so far, not used

        # -- model pipeline paths
        self.checkpoint_path = Path(self.extra_cfg["extra"]["checkpoint_path"])
        self.tuning_path = Path(self.extra_cfg["extra"]["tuning_path"])
        self.extract_logging_path = Path(self.paths_cfg["extract"]["logging_path"])
        self.transform_logging_path = Path(self.paths_cfg["transform"]["logging_path"])
        self.load_logging_path = Path(self.paths_cfg["load"]["logging_path"])
        self.model_logging_path = Path(self.extra_cfg["extra"]["logging_path"])
        self.eval_path = Path(self.extra_cfg["extra"]["eval_path"])
    
    # -----------------------------------------------------
    # Basic useful functions: raw data iteration functions
    # -----------------------------------------------------
    def _iter_raw_anamnesis_data(
        self,
        columns: Optional[List[str]] = None,
        date_columns: Optional[List[str]] = None,
        file_glob_pattern: Optional[str] = "*.parquet"
    ) -> Iterable[pd.DataFrame]:
        """
            Iterate over raw anamnesis Parquet files as cleaned DataFrames.

            Args:
            ------
            columns : list[str], optional
                Subset of columns to read from each Parquet file (passed to `pd.read_parquet`).
            file_glob_pattern : str, optional
                Glob pattern for selecting files within the mammograms folder.
                Defaults to `"*.parquet"`.

            Yields
            ------
            pandas.DataFrame generator
        """
        for cur_file in list(self.raw_anamnesis_folder_path.glob(file_glob_pattern)):
            cur_df = pd.read_parquet(cur_file, columns=columns)
            if date_columns is not None:
                # -- avoid outofboundstimestamp error
                for col in date_columns:
                    cur_df[col] = cur_df[col].apply(lambda x: x if pd.notna(x) and x < dt.datetime(2029, 1, 1) and x>dt.datetime(1960,1,1) else np.nan)
                    cur_df[col] = pd.to_datetime(cur_df[col], errors="coerce")
            yield cur_df
    
    def _iter_raw_mammograms_data(
        self,
        columns: Optional[List[str]] = None,
        file_glob_pattern: Optional[str] = "*.parquet"
    ) -> Iterable[pd.DataFrame]:
        """
            Iterate over raw mammogram Parquet files as cleaned DataFrames.

            Args:
            ------
            columns : list[str], optional
                Subset of columns to read from each Parquet file (passed to `pd.read_parquet`).
            file_glob_pattern : str, optional
                Glob pattern for selecting files within the mammograms folder.
                Defaults to `"*.parquet"`.

            Yields
            ------
            pandas.DataFrame generator
                DataFrame with at least:
                - `"text"`: stripped report text
                - `"raw_text_hash"`: SHA1 hash of text
                - `"key"`: string identifier derived from attendance code and hash
        """
        for cur_file in list(self.raw_mammograms_folder_path.glob(file_glob_pattern)):
            cur_df = pd.read_parquet(cur_file, columns=columns)
            cur_df = cur_df[pd.notna(cur_df[MammogramColumns.DS_LAUDO_MEDICO.value])].copy()
            cur_df["text"] = cur_df[MammogramColumns.DS_LAUDO_MEDICO.value].apply(lambda x: x.strip() if pd.notna(x) else np.nan)
            cur_df["raw_text_hash"] = cur_df["text"].apply(lambda x: sha1(x) if pd.notna(x) else np.nan)
            cur_df["key"] = cur_df[MammogramColumns.CD_ATENDIMENTO.value].apply(lambda x: f"{x:.0f}") + cur_df["raw_text_hash"]
            yield cur_df

    def _iter_raw_biopsy_data(
        self,
        columns: Optional[List[str]] = None,
        frac: Optional[float] = None, # -- for testing
        file_glob_pattern: Optional[str] = "*.parquet"
    ) -> Iterable[pd.DataFrame]:
        """
            Iterate over raw biopsy Parquet files as cleaned DataFrames.

            Args:
            ------
            columns : list[str], optional
                Subset of columns to read from each Parquet file.
            frac : float, optional
                If provided, sample this fraction of rows from each file (via `DataFrame.sample`).
            file_glob_pattern : str, optional
                Glob pattern for selecting files within the biopsy folder.
                Defaults to `"*.parquet"`.

            Yields
            ------
            pandas.DataFrame generator
                Cleaned biopsy DataFrame including `"raw_text_hash"` and with date bounds applied.

            Notes
            -----
            - Date bounds enforced: between 1970-01-01 and 2030-01-01 (inclusive by filter logic).
        """
        for cur_file in list(self.raw_biopsy_folder_path.glob(file_glob_pattern)):
            cur_df = pd.read_parquet(cur_file, columns=columns)
            if frac is not None:
                cur_df = cur_df.sample(frac=frac).copy()
            cur_df = cur_df[pd.notna(cur_df[BiopsyColumns.DS_LAUDO_MEDICO.value])].copy()
            cur_df[BiopsyColumns.DS_LAUDO_MEDICO.value] = cur_df[BiopsyColumns.DS_LAUDO_MEDICO.value].apply(lambda x: x.strip() if pd.notna(x) else np.nan)
            cur_df["raw_text_hash"] = cur_df[BiopsyColumns.DS_LAUDO_MEDICO.value].apply(lambda x: sha1(x) if pd.notna(x) else np.nan)
            # -- filter out of bounds dates
            cur_df = cur_df[(cur_df[BiopsyColumns.DT_ATENDIMENTO.value]<=dt.datetime(2030,1,1)) & (cur_df[BiopsyColumns.DT_PROCEDIMENTO_REALIZADO.value]<=dt.datetime(2030,1,1))].copy()
            cur_df = cur_df[(cur_df[BiopsyColumns.DT_ATENDIMENTO.value]>=dt.datetime(1970,1,1)) & (cur_df[BiopsyColumns.DT_PROCEDIMENTO_REALIZADO.value]>=dt.datetime(1970,1,1))].copy()            
            yield cur_df

    def _iter_raw_person_data(
        self,
        columns: Optional[List[str]] = None,
        deduple_columns: Optional[List[str]] = None,
    ) -> Iterable[pd.DataFrame]:
        """
            Iterate over raw person-level Parquet files as DataFrames.

            This generator reads files matching `"person*.parquet"` in
            `self.raw_patient_folder_path`, optionally selecting columns and deduplicating
            rows.

            Args:
            ------
            columns : list[str], optional
                Subset of columns to read from each Parquet file.
            deduple_columns : list[str], optional
                If provided, drop duplicate rows based on these columns.

            Yields
            ------
            pandas.DataFrame generator
                DataFrame read from each matching Parquet file, optionally deduplicated.
        """
        for cur_file in list(self.raw_patient_folder_path.glob("person*.parquet")):
            cur_df = pd.read_parquet(cur_file, columns=columns)
            if deduple_columns is not None:
                cur_df = cur_df.drop_duplicates(deduple_columns)
            yield cur_df

    def _iter_raw_patient_data(
        self,
        columns: Optional[List[str]] = None,
        deduple_columns: Optional[List[str]] = None
    ) -> Iterable[pd.DataFrame]:
        """
            Iterate over raw patient-level Parquet files as DataFrames.

            This generator reads files matching `"patient*.parquet"` in
            `self.raw_patient_folder_path`, optionally selecting columns and deduplicating
            rows.

            Same behavior as self._iter_raw_person_data().
        """
        for cur_file in list(self.raw_patient_folder_path.glob("patient*.parquet")):
            cur_df = pd.read_parquet(cur_file, columns=columns)
            if deduple_columns is not None:
                cur_df = cur_df.drop_duplicates(deduple_columns)
            yield cur_df

    def _iter_raw_user_data(
        self,
        columns: Optional[List[str]] = None,
        deduple_columns: Optional[List[str]] = None
    ) -> Iterable[pd.DataFrame]:
        """
            Iterate over raw user-level Parquet files as DataFrames.

            This generator reads files matching `"user*.parquet"` in
            `self.raw_patient_folder_path`, optionally selecting columns and deduplicating
            rows.

            Same behavior as self._iter_raw_person_data().
        """
        for cur_file in list(self.raw_patient_folder_path.glob("user*.parquet")):
            cur_df = pd.read_parquet(cur_file, columns=columns)
            if deduple_columns is not None:
                cur_df = cur_df.drop_duplicates(deduple_columns)
            yield cur_df

    # -----------------------------------------------------------
    # Basic useful functions: final data iteration functions
    # -----------------------------------------------------------
    def _iter_base_merged_data(
        self,
        person_ids: Optional[List[str]] = None,
        columns: Optional[List[str]] = None,
        batch_size: Optional[int] = 10000
    ) -> Iterable[pd.DataFrame]:
        merged_data_filename = self.files_and_folders_cfg['load']['load_files']['merged_data']
        if not self.load_path.joinpath(merged_data_filename).is_file():
            raise Exception(f"{merged_data_filename} does not exist in {self.load_path}.")
        for batch in batching_parquet_file(self.load_path.joinpath(merged_data_filename), columns=columns, batch_size=batch_size):
            if person_ids is not None:
                batch = batch[batch[PersonColumns.CD_PESSOA.value].isin(person_ids)].copy()
            yield batch

    def _iter_mamm_sequence_data(
        self,
        mammogram_ids: Optional[List[str]] = None,
        person_ids: Optional[List[str]] = None,
        columns: Optional[List[str]] = None,
        batch_size: Optional[int] = 10000
    ) -> Iterable[pd.DataFrame]:
        seq_data_filename = self.files_and_folders_cfg['load']['load_files']['seq_per_mammogram_filename']
        if not self.dataset_path.joinpath(seq_data_filename).is_file():
            raise Exception(f"{seq_data_filename} does not exist in {self.dataset_path}.")
        for batch in batching_parquet_file(self.dataset_path.joinpath(seq_data_filename), columns=columns, batch_size=batch_size):
            if mammogram_ids is not None:
                batch = batch[batch["mammogram_id"].isin(mammogram_ids)].copy()
            if person_ids is not None:
                batch = batch[batch[PersonColumns.CD_PESSOA.value].isin(person_ids)].copy()
            yield batch

    def _iter_final_data(
        self,
        mammogram_ids: Optional[List[str]] = None,
        person_ids: Optional[List[str]] = None,
        columns: Optional[List[str]] = None,
        fraction: Optional[float] = 1.0,
        batch_size: Optional[int] = 10000
    ) -> Iterable[pd.DataFrame]:
        final_data_filename = self.files_and_folders_cfg['load']['load_files']['final_data_before_eligibility_filename']
        if not self.dataset_path.joinpath(final_data_filename).is_file():
            raise Exception(f"{final_data_filename} does not exist in {self.dataset_path}.")
        for batch in batching_parquet_file(self.dataset_path.joinpath(final_data_filename), columns=columns, batch_size=batch_size):
            if mammogram_ids is not None:
                batch = batch[batch["mammogram_id"].isin(mammogram_ids)].copy()
            if person_ids is not None:
                batch = batch[batch[PersonColumns.CD_PESSOA.value].isin(person_ids)].copy()
            if fraction<1.0:
                batch = batch.sample(frac=fraction)
            yield batch
