from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from hapcancer.schemas.enums import (
    AnamnesisColumns, MammogramColumns, PersonColumns, PatientColumns, BiopsyColumns,
    MergedSourcesColumns
)
from hapcancer.config_manager import ConfigInterface

def load_bmi_models(path_to_model, filename):
    path_to_bmi_models = Path(path_to_model)
    model = joblib.load(path_to_bmi_models.joinpath(filename))
    return model

class MergeSources(ConfigInterface):
    def __init__(self, config_dir: str, config_defaults):
        super().__init__(config_dir, config_defaults)
        
        self.transformed_data_path = self.transform_path.joinpath("transformed")
        self.fields = self.fields_cfg["fields"]
        self.mamm_field_suffix = '_MAMOGRAFIA'
        self.anamnesis_field_suffix = '_ANAMNESE'

        ext = ".parquet"
        self.transformed_filenames = {
            'anamnesis': self.transformed_data_path.joinpath(self.files_and_folders_cfg['transform']['transformed_files']['anamnesis']+ext),
            'person': self.transformed_data_path.joinpath(self.files_and_folders_cfg['transform']['transformed_files']['person']+ext),
            'user': self.transformed_data_path.joinpath(self.files_and_folders_cfg['transform']['transformed_files']['user']+ext),
            'similarity_data': self.transformed_data_path.joinpath(self.files_and_folders_cfg['transform']['transformed_files']['similarity_data']+ext),
            'valid_person_patient': self.transformed_data_path.joinpath(self.files_and_folders_cfg['transform']['transformed_files']['valid_person_patient']+ext),
            'person_mammogram': self.transformed_data_path.joinpath(self.files_and_folders_cfg['transform']['transformed_files']['person_mammogram']+ext),
            'biopsy': self.transformed_data_path.joinpath(self.files_and_folders_cfg['transform']['transformed_files']['person_biopsy']+ext)
        }
        
        self.transformed_data = None
        self.merged_data = None

    def _load_transformed_data(self) -> None:
        self.transformed_data = {
            'anamnesis': pd.read_parquet(self.transformed_filenames['anamnesis']),
            'person': pd.read_parquet(self.transformed_filenames['person']),
            'user': pd.read_parquet(self.transformed_filenames['user']),
            'similarity_data': pd.read_parquet(self.transformed_filenames['similarity_data']),
            'valid_person_patient': pd.read_parquet(self.transformed_filenames['valid_person_patient']),
            'biopsy': pd.read_parquet(self.transformed_filenames['biopsy']),
            'person_mammogram': pd.read_parquet(self.transformed_filenames['person_mammogram'])#.sample(frac=0.1) # test,
        }

    def _set_base_mammogram(self):
        # -- base data hold sequences with the ordered performed mammograms
        base_info_mamm_cols = [ PersonColumns.CD_PESSOA.value, MammogramColumns.DT_ATENDIMENTO.value, 'birads_labels', 'key' ]
        self.merged_data = self.transformed_data['person_mammogram'].reset_index().copy()#[base_info_mamm_cols].copy()
        self.merged_data = self.merged_data.rename({ MammogramColumns.DT_ATENDIMENTO.value: MammogramColumns.DT_ATENDIMENTO.value+self.mamm_field_suffix }, axis=1)

    def _merge_user_person(self):
        col_cd_pessoa = self.fields["person_id"]
        col_cd_user = self.fields["user_id"]
        col_dt_nasc = self.fields["person_birthdate"]
        col_dt_atend_mamografia = self.fields["mammogram_date"]+self.mamm_field_suffix
        col_cd_atend = self.fields["mammogram_id_final"]
        col_birads = "birads_labels"        
        base_info_user_cols = [col_cd_pessoa, col_cd_user, "VL_MENSALIDADE_MIN", "VL_MENSALIDADE_MAX"]
        base_info_person_cols = [col_cd_pessoa, col_dt_nasc, 'FL_SEXO_ML', 'zipcode_cat']

        person_info = self.transformed_data['person'].copy()
        user_info = self.transformed_data['user'].copy()

        self.merged_data = self.merged_data.merge(user_info.reset_index()[base_info_user_cols], on=col_cd_pessoa, how="left")
        self.merged_data = self.merged_data.merge(person_info[base_info_person_cols], on=col_cd_pessoa, how="left")

        self.merged_data["age_at_first_mammogram"] = self.merged_data["first_mammogram_date"] - self.merged_data[col_dt_nasc]
        self.merged_data["age_at_first_mammogram"] = self.merged_data["age_at_first_mammogram"].apply(lambda x: np.timedelta64(x).astype('timedelta64[Y]')/np.timedelta64(1, 'Y') if pd.notna(x) else np.nan )
        # -- remove null
        self.merged_data = self.merged_data[pd.notna(self.merged_data["age_at_first_mammogram"])].copy()

        print(self.bmi_models_cfg)
        lin_reg = load_bmi_models(self.bmi_models_cfg["bmi_model"]["path"], self.bmi_models_cfg["bmi_model"]["linreg_model"])
        randfor = load_bmi_models(self.bmi_models_cfg["bmi_model"]["path"], self.bmi_models_cfg["bmi_model"]["randfor_model"])
        self.merged_data["BMI_PREDICT_LINREG"] = lin_reg.predict(self.merged_data[["FL_SEXO_ML", "age_at_first_mammogram"]].values)
        self.merged_data["BMI_PREDICT_RANDFOR"] = randfor.predict(self.merged_data[["FL_SEXO_ML", "age_at_first_mammogram"]].values)

    def _merge_biopsy_anamnesis(self):
        # -- Biopsy data
        biopsy_cols = [ PatientColumns.CD_PESSOA.value, "biopsy_results", "biopsy_dates" ]
        biopsy_info = self.transformed_data['biopsy'].reset_index()
        self.merged_data = self.merged_data.merge(biopsy_info[biopsy_cols], on=PatientColumns.CD_PESSOA.value, how="left")

        # -- Anamnesis data
        anamnesis_info = self.transformed_data['anamnesis'].drop(columns=[AnamnesisColumns.CD_ATENDIMENTO.value])
        anamnesis_info = anamnesis_info.rename({AnamnesisColumns.DT_ATENDIMENTO.value: AnamnesisColumns.DT_ATENDIMENTO.value+self.anamnesis_field_suffix}, axis=1)
        self.merged_data = self.merged_data.merge(anamnesis_info, on=PatientColumns.CD_PESSOA.value, how="left")

    def merge(self, verbose=True):
        if verbose: print("[load] transformed data ...")
        self._load_transformed_data()
        if verbose: print("[set] the base mammogram sequence data ...")
        self._set_base_mammogram()
        if verbose: print("[merge] user/person data ...")
        self._merge_user_person()
        if verbose: print("[merge] biopsy/anamnesis data ...")
        self._merge_biopsy_anamnesis()
        print(self.merged_data.columns)
        base_merged_filename = self.files_and_folders_cfg['load']['load_files']['merged_data']
        self.merged_data.to_parquet(self.load_path.joinpath(base_merged_filename))