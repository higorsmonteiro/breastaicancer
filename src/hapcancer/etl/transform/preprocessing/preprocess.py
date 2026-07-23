import re
import csv
import pandas as pd
import numpy as np
import datetime as dt
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from typing import List, Any, Optional

from hapcancer.schemas.enums import (
    MammogramColumns, PersonColumns, PatientColumns,
    BiopsyColumns, AnamnesisColumns, UserColumns
)
from hapcancer.etl.transform.preprocessing.biopsy_report_classifier import BiopsyReportClassifier
import hapcancer.etl.transform.preprocessing.utils as utils

from hapcancer.logger import Logger
from hapcancer.config_manager import ConfigInterface


# ------------------------------------------------------------------------------ #
# ---------------------------- AUXILIARY FUNCTIONS ----------------------------- #
# ------------------------------------------------------------------------------ #

def filter_integers(data):
    valid_integers = []
    for item in data:
        try:
            # attempt to convert to integer and strip any spaces
            number = int(item.strip())
            valid_integers.append(number)
        except ValueError:
            # if conversion fails, skip the item
            pass
    return valid_integers

def flatten(xss):
    return [x for xs in xss for x in xs]

def flatten_menarche(xss):
    return [ x for xs in xss if xs is not None and any(pd.notna(xs)) for x in xs ]

def flatten_menarche(outer_list):
    return [y for el in outer_list for y in (el if isinstance(el, list) else [el])]

def load_patient_to_person(
    obj: ConfigInterface, 
    pat_subset_columns: List[str]
) -> pd.DataFrame:
    obj.patient_to_person = []
    for cur_df in obj._iter_raw_patient_data(columns=pat_subset_columns, deduple_columns=pat_subset_columns):
        obj.patient_to_person.append(cur_df)
    obj.patient_to_person = pd.concat(obj.patient_to_person, ignore_index=True).drop_duplicates(pat_subset_columns)

# ------------------------------------------------------------------------------ #
# --------------------------------- ANAMNESIS ---------------------------------- #
# ------------------------------------------------------------------------------ #

class TransformAnamnesis(ConfigInterface):
    def __init__(self, config_dir: str, config_defaults: dict):
        super().__init__(config_dir, config_defaults)
    
        self.anamnesis_path = self.extract_path.joinpath(self.extract_folders['anamnesis'])
        self.anamnesis_df = None

        self.all_columns = [ elem.value for elem in AnamnesisColumns ]
        self.date_columns = [
            AnamnesisColumns.DT_BIOPSIA_MD.value, AnamnesisColumns.DT_ATENDIMENTO.value,
            AnamnesisColumns.DT_BIOPSIA_ME.value, AnamnesisColumns.DT_QUADRANTECTOMIA_MD.value,
            AnamnesisColumns.DT_QUADRANTECTOMIA_ME.value, AnamnesisColumns.DT_MASTECTOMIA_MD.value,
            AnamnesisColumns.DT_MASTECTOMIA_ME.value, AnamnesisColumns.DT_PLASTICA_MD.value,
            AnamnesisColumns.DT_PLASTICA_ME.value 
        ]

    def _load_data(self):
        self.anamnesis_df = pd.concat([
            cur_df for cur_df in self._iter_raw_anamnesis_data(columns=self.all_columns, date_columns=self.date_columns)
        ], ignore_index=True).reset_index(drop=True)
        #self.anamnesis_df = self.anamnesis_df.sample(frac=0.05) # -- just for debugging

    def _adjust_children_count(self):
        pattern = r'\bfilhos?-?\d*\b'
        ds_ind, nu_gest = AnamnesisColumns.DS_INDICACAO_QUEIXA.value, AnamnesisColumns.NU_GESTACAO.value
        self.anamnesis_df['contains_filho'] = self.anamnesis_df[ds_ind].apply(lambda x: x.lower() if pd.notna(x) else x).str.contains(pattern, regex=True).apply(lambda x: False if pd.isna(x) else x)
        self.anamnesis_df['nu_filhos_extracao'] = self.anamnesis_df[[ds_ind, "contains_filho"]].apply(utils.extract_children_count, axis=1)
        self.anamnesis_df["NU_GESTACAO_"] = self.anamnesis_df[[nu_gest, "nu_filhos_extracao"]].apply(lambda x: x[nu_gest] if pd.notna(x[nu_gest]) else x["nu_filhos_extracao"] , axis=1)
    
    def _standardize_columns(self):
        cd_person = AnamnesisColumns.CD_PESSOA.value
        cd_atend, dt_atend = AnamnesisColumns.CD_ATENDIMENTO.value, AnamnesisColumns.DT_ATENDIMENTO.value
        ds_menarca_cl = AnamnesisColumns.DS_MENARCA.value
        ds_meno_cl = AnamnesisColumns.DS_MENOPAUSA.value
        nu_gest_aborto_cl = AnamnesisColumns.NU_GESTACAO_ABORTO
        flca_mae, flca_avo = AnamnesisColumns.FL_CA_MAMA_MAE.value, AnamnesisColumns.FL_CA_MAMA_AVO.value
        flca_irma, flca_tia = AnamnesisColumns.FL_CA_MAMA_IRMA.value, AnamnesisColumns.FL_CA_MAMA_TIA.value
        fl_aleitamento = AnamnesisColumns.FL_ALEITAMENTO.value
        fl_mast_md, fl_mast_me = AnamnesisColumns.FL_MASTECTOMIA_MD.value, AnamnesisColumns.FL_MASTECTOMIA_ME.value
        fl_plas_me, fl_plas_md = AnamnesisColumns.FL_PLASTICA_ME.value, AnamnesisColumns.FL_PLASTICA_MD.value

        map_val = defaultdict(lambda: np.nan, {'S': 1, 'N': 0})
        self.anamnesis_df["DS_MENARCA_FMT"] = self.anamnesis_df[ds_menarca_cl].apply(lambda x: [ filter_integers(re.findall('\d+', elem)) if pd.notna(elem) else np.nan for elem in x ])
        self.anamnesis_df["DS_MENOPAUSA_FMT"] = self.anamnesis_df[ds_meno_cl].apply(lambda x: [ filter_integers(re.findall('\d+', elem)) if pd.notna(elem) else np.nan for elem in x ])
        self.anamnesis_df["NU_GESTACAO_FMT"] = self.anamnesis_df["NU_GESTACAO_"].apply(lambda x: [ elem if pd.notna(elem) else np.nan for elem in x ])#.apply(lambda x: max(x) if len(x)>0 else np.nan)
        self.anamnesis_df["NU_GESTACAO_ABORTO_FMT"] = self.anamnesis_df[nu_gest_aborto_cl].apply(lambda x: [ elem if pd.notna(elem) else np.nan for elem in x ])#.apply(lambda x: max(x) if len(x)>0 else np.nan)

        ## -- if the person has an anamnesis exam, but data on familial history is missing, we could consider as a negative result, right? if no, change last 0 to -1.
        self.anamnesis_df["FL_CA_MAMA_MAE_FMT"] = self.anamnesis_df[flca_mae].apply(lambda x: [ elem if pd.notna(elem) else 'N' for elem in x ]).apply(lambda x: [map_val[elem] for elem in x] )#.apply(lambda x: 1 if 'S' in x else ( 0 if 'N' in x else 0 ) ) 
        self.anamnesis_df["FL_CA_MAMA_AVO_FMT"] = self.anamnesis_df[flca_avo].apply(lambda x: [ elem if pd.notna(elem) else 'N' for elem in x ]).apply(lambda x: [map_val[elem] for elem in x] )#.apply(lambda x: 1 if 'S' in x else ( 0 if 'N' in x else 0 ) )
        self.anamnesis_df["FL_CA_MAMA_IRMA_FMT"] = self.anamnesis_df[flca_irma].apply(lambda x: [ elem if pd.notna(elem) else 'N' for elem in x ]).apply(lambda x: [map_val[elem] for elem in x] )#.apply(lambda x: 1 if 'S' in x else ( 0 if 'N' in x else 0 ) )
        self.anamnesis_df["FL_CA_MAMA_TIA_FMT"] = self.anamnesis_df[flca_tia].apply(lambda x: [ elem if pd.notna(elem) else 'N' for elem in x ]).apply(lambda x: [map_val[elem] for elem in x] )#.apply(lambda x: 1 if 'S' in x else ( 0 if 'N' in x else 0 ) )
        self.anamnesis_df["FL_ALEITAMENTO_FMT"] = self.anamnesis_df[fl_aleitamento].apply(lambda x: [ elem if pd.notna(elem) else 'N' for elem in x ]).apply(lambda x: [map_val[elem] for elem in x] )
        ## -- breast removal
        self.anamnesis_df["FL_MASTECTOMIA_MD_FMT"] = self.anamnesis_df[fl_mast_md].apply(lambda x: [ elem if pd.notna(elem) else 'N' for elem in x ]).apply(lambda x: [map_val[elem] for elem in x] )#.apply(lambda x: 1 if 'S' in x else ( 0 if 'N' in x else 0 ) )
        self.anamnesis_df["FL_MASTECTOMIA_ME_FMT"] = self.anamnesis_df[fl_mast_me].apply(lambda x: [ elem if pd.notna(elem) else 'N' for elem in x ]).apply(lambda x: [map_val[elem] for elem in x] )#.apply(lambda x: 1 if 'S' in x else ( 0 if 'N' in x else 0 ) )
        ## -- date of first confirmed removal
        self.anamnesis_df["DT_MASTECTOMIA_ME_FMT"] = self.anamnesis_df[["FL_MASTECTOMIA_ME_FMT", dt_atend]].apply(lambda x: [ x[dt_atend][pos] if val==1 else np.nan for pos, val in enumerate(x["FL_MASTECTOMIA_ME_FMT"]) ], axis=1 )
        self.anamnesis_df["DT_MASTECTOMIA_MD_FMT"] = self.anamnesis_df[["FL_MASTECTOMIA_MD_FMT", dt_atend]].apply(lambda x: [ x[dt_atend][pos] if val==1 else np.nan for pos, val in enumerate(x["FL_MASTECTOMIA_MD_FMT"]) ], axis=1 )
        #
        ## -- breast implants
        self.anamnesis_df["FL_PLASTICA_ME_FMT"] = self.anamnesis_df[fl_plas_me].apply(lambda x: [ elem if pd.notna(elem) else 'N' for elem in x ]).apply(lambda x: [map_val[elem] for elem in x] )
        self.anamnesis_df["FL_PLASTICA_MD_FMT"] = self.anamnesis_df[fl_plas_md].apply(lambda x: [ elem if pd.notna(elem) else 'N' for elem in x ]).apply(lambda x: [map_val[elem] for elem in x] )
        ## -- date of first confirmed 'plástica' (left breast) - date of when the information was first available
        self.anamnesis_df["DT_PLASTICA_ME_FMT"] = self.anamnesis_df[["FL_PLASTICA_ME_FMT", dt_atend]].apply(lambda x: [ x[dt_atend][pos] if val==1 else np.nan for pos, val in enumerate(x["FL_PLASTICA_ME_FMT"]) ], axis=1 )
        self.anamnesis_df["DT_PLASTICA_MD_FMT"] = self.anamnesis_df[["FL_PLASTICA_MD_FMT", dt_atend]].apply(lambda x: [ x[dt_atend][pos] if val==1 else np.nan for pos, val in enumerate(x["FL_PLASTICA_MD_FMT"]) ], axis=1 )
        
        selected_cols = [
            cd_person, cd_atend, dt_atend, 'DS_MENARCA_FMT', 'DS_MENOPAUSA_FMT', 
            'NU_GESTACAO_FMT', 'NU_GESTACAO_ABORTO_FMT', 'FL_CA_MAMA_MAE_FMT',
            'FL_CA_MAMA_AVO_FMT', 'FL_CA_MAMA_IRMA_FMT', 'FL_CA_MAMA_TIA_FMT',
            'FL_MASTECTOMIA_MD_FMT', 'FL_MASTECTOMIA_ME_FMT', 'FL_PLASTICA_ME_FMT',
            'FL_PLASTICA_MD_FMT', 'DT_PLASTICA_ME_FMT', 'DT_PLASTICA_MD_FMT',
            'DT_MASTECTOMIA_ME_FMT', 'DT_MASTECTOMIA_MD_FMT', 'FL_ALEITAMENTO_FMT'
        ]
        self.anamnesis_df = self.anamnesis_df[selected_cols].copy()

    def _process_info(self):
        '''
        
        '''
        self.anamnesis_df["DT_PRIMEIRA_ANAMNESE"] = self.anamnesis_df[AnamnesisColumns.DT_ATENDIMENTO.value].apply(lambda x: x[0])
        #self.anamnesis_df["DS_MENARCA_FMT"] = self.anamnesis_df["DS_MENARCA_FMT"].apply(lambda x: flatten_menarche(x) if any(pd.notna(x)) else np.nan)
        self.anamnesis_df["DS_MENARCA_FMT"] = self.anamnesis_df["DS_MENARCA_FMT"].apply(flatten_menarche).apply(lambda x: min(x) if len(x)>0 else np.nan)
        #self.anamnesis_df["DS_MENARCA_FMT"] = self.anamnesis_df["DS_MENARCA_FMT"].apply(lambda x: min(x) if len(x)>0 else np.nan)
        #self.anamnesis_df["DS_MENOPAUSA_FMT"] = self.anamnesis_df["DS_MENOPAUSA_FMT"].apply(lambda x: flatten_menarche(x) if any(pd.notna(x)) else np.nan)
        self.anamnesis_df["DS_MENOPAUSA_FMT"] = self.anamnesis_df["DS_MENOPAUSA_FMT"].apply(flatten_menarche).apply(lambda x: np.floor(np.mean(x)) if len(x)>0 else np.nan)
        #self.anamnesis_df["DS_MENOPAUSA_FMT"] = self.anamnesis_df["DS_MENOPAUSA_FMT"].apply(lambda x: np.floor(np.mean(x)) if len(x)>0 else np.nan)

        menarche_no_nan = self.anamnesis_df[pd.notna(self.anamnesis_df["DS_MENARCA_FMT"])]["DS_MENARCA_FMT"]
        menopause_no_nan = self.anamnesis_df[pd.notna(self.anamnesis_df["DS_MENOPAUSA_FMT"])]["DS_MENOPAUSA_FMT"]

        # -- remove outliers
        p5, p95 = np.percentile(menarche_no_nan, 2), np.percentile(menarche_no_nan, 98)
        self.anamnesis_df["DS_MENARCA_FMT"] = self.anamnesis_df["DS_MENARCA_FMT"].apply(lambda x: x if pd.notna(x) and x>p5 and x<p95 else np.nan)
        min_age_menopause, max_age_menopause = 25, 65
        self.anamnesis_df["DS_MENOPAUSA_FMT"] = self.anamnesis_df["DS_MENOPAUSA_FMT"].apply(lambda x: x if pd.notna(x) and x>min_age_menopause and x<max_age_menopause else np.nan)

        # -- within the anamnesis data if number of children or miscarriage is not filled, then we set it to zero.
        self.anamnesis_df['NU_GESTACAO_ABORTO_FMT'] = self.anamnesis_df['NU_GESTACAO_ABORTO_FMT'].apply(lambda x: [ elem if pd.notna(elem) else 0.0 for elem in x ] )
        self.anamnesis_df["NU_GESTACAO_FMT"] = self.anamnesis_df["NU_GESTACAO_FMT"].apply(lambda x: [ elem if pd.notna(elem) else 0.0 for elem in x ])
        # -- menopause category (must be done after splitting)
        #median = self.anamnesis_df[self.anamnesis_df['DS_MENOPAUSA_FMT']>=min_age_menopause]['DS_MENOPAUSA_FMT'].median()
        #std = self.anamnesis_df['DS_MENOPAUSA_FMT'].std()
        # -- imputation of menopause age (must be handled after when we have information of the age - so that information is consistent with age)
        #self.anamnesis_df['DS_MENOPAUSA_FMT_IMPUTATION'] = self.anamnesis_df["DS_MENOPAUSA_FMT"].apply(lambda x: np.floor(np.random.normal(median, std)) if pd.isna(x) else x)

    def transform(self):
        print("[load] anamnesis data ...")
        self._load_data()
        self._adjust_children_count()
        print("[formatting] sequential anamnesis data per person ...")
        self.anamnesis_df = self.anamnesis_df.groupby(AnamnesisColumns.CD_PESSOA.value).agg({
            feature_nm : list for feature_nm in self.anamnesis_df.columns if feature_nm != AnamnesisColumns.CD_PESSOA.value
        }).reset_index()
        self.anamnesis_df["argsort"] = self.anamnesis_df[AnamnesisColumns.DT_ATENDIMENTO.value].apply(lambda x: np.argsort(x))
        
        # -- rearrange columns to obey correct date order
        for cur_col in [ elem for elem in self.all_columns if elem != AnamnesisColumns.CD_PESSOA.value ]:
            self.anamnesis_df[cur_col] = self.anamnesis_df[[cur_col, "argsort"]].apply(lambda x: np.array(x[cur_col])[x["argsort"]], axis=1)
        self.anamnesis_df = self.anamnesis_df.drop(columns=["argsort"])
        print("[processing] standardization and processing of columns ...")
        self._standardize_columns()
        self._process_info()
        # -- careful that in other steps we might use DT_ATENDIMENTO_ANAMNESE instead of DT_ATENDIMENTO
        anamnesis_filename = self.files_and_folders_cfg['transform']['transformed_files']['anamnesis']+'.parquet'
        self.anamnesis_df.to_parquet(self.transformed_data_path.joinpath(anamnesis_filename))


# ------------------------------------------------------------------------------ #
# ----------------------------------- BIOPSY ----------------------------------- #
# ------------------------------------------------------------------------------ #

class TransformBiopsyData(ConfigInterface):
    def __init__(self, config_dir: str, config_defaults: dict, logger: Optional[bool] = False) -> None:
        super().__init__(config_dir, config_defaults)
        self.config_dir = config_dir
        self.config_defaults = config_defaults

        self.user_person_path = self.extract_path.joinpath(self.extract_folders['user_person_data'])
        self.biopsy_path = self.extract_path.joinpath(self.extract_folders['biopsy'])

        self.logger = None
        if logger:
            self.logger = Logger(self.transform_logging_path, run_name=self.transform_id, overwrite=False)

        self.biopsy_df = None
        self.biopsy_breast_df = None
        self.patient_to_person = None
        self.person_biopsy_timing = None

        ext = ".parquet"
        self.transformed_filenames = {
            'breast_biopsy': self.transformed_data_path.joinpath(self.files_and_folders_cfg['transform']['transformed_files']['breast_biopsy']+ext),
            'breast_biopsy_classified': self.transformed_data_path.joinpath(self.files_and_folders_cfg['transform']['transformed_files']['breast_biopsy_classified']+'.csv'),
            'person_biopsy': self.transformed_data_path.joinpath(self.files_and_folders_cfg['transform']['transformed_files']['person_biopsy']+ext),
            'similarity_data': self.transformed_data_path.joinpath(self.files_and_folders_cfg['transform']['transformed_files']['similarity_data']+ext),
            'valid_person_patient': self.transformed_data_path.joinpath(self.files_and_folders_cfg['transform']['transformed_files']['valid_person_patient']+ext),
        }

    # ------------------------- LOGGING HELPERS ------------------------- #
    def _log(self, event: str, **kwargs) -> None:
        if self.logger is None: return
        payload = {"event": event, "entity": "biopsy_transform"}
        payload.update(kwargs)
        self.logger.log_info(payload)
    
    # ---------------------------- TRANSFORM ---------------------------- #
    def _load_data(self, frac: Optional[float] = None):
        # -- biopsy dataframe (big load here**)
        self.biopsy_df = pd.concat([
            cur_df for cur_df in self._iter_raw_biopsy_data(frac=frac)
        ], ignore_index=True)

        # -- patient to person codes
        pat_subset_columns = ["CD_PACIENTE", "CD_PESSOA"]
        load_patient_to_person(self, pat_subset_columns)

    def _simple_deduple(self):
        self.biopsy_df = self.biopsy_df.drop_duplicates(subset=[
            BiopsyColumns.CD_ATENDIMENTO.value, BiopsyColumns.CD_PACIENTE.value, "raw_text_hash"
        ])
        self.biopsy_df["key"] = self.biopsy_df[BiopsyColumns.CD_ATENDIMENTO.value].apply(lambda x: f"{x:.0f}") + self.biopsy_df["raw_text_hash"]

    def _identify_if_breast(self):
        self.biopsy_df["is_breast"] = self.biopsy_df[BiopsyColumns.DS_LAUDO_MEDICO.value].apply(lambda x: utils.is_breast_biopsy(x) if pd.notna(x) else False)
        n_true = int((self.biopsy_df["is_breast"] == True).sum())
        n_total = int(self.biopsy_df.shape[0])
        self._log("identify_breast_done", is_breast_true=n_true, total=n_total, pct=round(100 * n_true / max(n_total, 1), 4))

    def _check_if_biopsy(self):
        BIO_PATTERNS = utils.compile_patterns(utils.BIOPSY_STRONG, weight=2) + utils.compile_patterns(utils.BIOPSY_BASE, weight=1)
        MAM_PATTERNS = utils.compile_patterns(utils.MAMMO_STRONG, weight=2) + utils.compile_patterns(utils.MAMMO_BASE, weight=1)
        REC_PATTERNS = utils.compile_patterns(utils.RECOMMEND, weight=1)
        self.biopsy_df["biopsy_check"] = self.biopsy_df[BiopsyColumns.DS_LAUDO_MEDICO.value].apply(lambda x: utils.classify(x, BIO_PATTERNS, MAM_PATTERNS, REC_PATTERNS) if pd.notna(x) else np.nan)
        self.biopsy_df["is_biopsy"] = self.biopsy_df["biopsy_check"].apply(lambda x: x[0] if pd.notna(x) else np.nan)
        self.biopsy_df["compare_points"] = self.biopsy_df["biopsy_check"].apply(lambda x: x[1]['biopsy'] - x[1]['mammogram'] if pd.notna(x) else np.nan)
        self.biopsy_df["biopsy_points"] = self.biopsy_df["biopsy_check"].apply(lambda x: x[1]['biopsy'] if pd.notna(x) else np.nan)
        self.biopsy_df["mammogram_points"] = self.biopsy_df["biopsy_check"].apply(lambda x: x[1]['mammogram'] if pd.notna(x) else np.nan)
        self.biopsy_df = self.biopsy_df.drop(columns=["biopsy_check"])
    
    def _filter_for_breast(self):
        # -- we can refine this but setting a rule using the biopsy and mammogram points defined in '_check_if_biopsy'.
        self.biopsy_breast_df = self.biopsy_df[self.biopsy_df["is_breast"]==True].copy()

    def _get_person_code(self):
        if self.biopsy_df is None or self.patient_to_person is None:
            raise Exception("Either biopsy data or patient data is not loaded.")
        if self.biopsy_breast_df is None:
            raise Exception("'biopsy_breast' not valid.")
        
        self.biopsy_breast_df = self.biopsy_breast_df.merge(self.patient_to_person,
                                                            on=PatientColumns.CD_PACIENTE.value, 
                                                            how="left")
        self.biopsy_breast_df = self.biopsy_breast_df.drop_duplicates(subset=["key", PatientColumns.CD_PACIENTE.value, PersonColumns.CD_PESSOA.value])

    def _map_to_person(self):
        if not self.transformed_filenames['breast_biopsy_classified'].is_file():
            raise Exception(f'No classified reports for breast biopsy exists.')
        if self.biopsy_breast_df is None:
            self.biopsy_breast_df = pd.read_parquet(self.transformed_filenames['breast_biopsy'])

        self.clf_breast_biopsy_df = pd.read_csv(self.transformed_filenames['breast_biopsy_classified'], encoding='latin1') # -- portuguese
        self.clf_breast_biopsy_df = self.clf_breast_biopsy_df.merge(self.biopsy_breast_df[[
            "key", BiopsyColumns.CD_PACIENTE.value, BiopsyColumns.CD_ATENDIMENTO.value, 
            PersonColumns.CD_PESSOA.value, BiopsyColumns.DT_ATENDIMENTO.value
        ]], on="key", how="left").drop_duplicates(subset="key")

        def codify_biopsy_label(x):
            if pd.isna(x):
                return np.nan
            if 'benigno' in x:
                return 0
            elif 'maligno' in x:
                return 1
            return 0
        self.clf_breast_biopsy_df["biopsy_dates"] = self.clf_breast_biopsy_df[BiopsyColumns.DT_ATENDIMENTO.value].copy()
        self.clf_breast_biopsy_df['biopsy_results'] = self.clf_breast_biopsy_df['content'].apply(codify_biopsy_label)

        # -- define biopsies by person
        self.person_biopsy_timing = self.clf_breast_biopsy_df.groupby(PersonColumns.CD_PESSOA.value).agg({
            "key": list, BiopsyColumns.CD_PACIENTE.value: list, "biopsy_results": list, "biopsy_dates": list
        })
        # -- sort by date
        self.person_biopsy_timing["argsort"] = self.person_biopsy_timing["biopsy_dates"].apply(lambda x: np.argsort(x))
        self.person_biopsy_timing["key"] = self.person_biopsy_timing[["key", "argsort"]].apply(lambda x: np.array(x['key'])[x['argsort']], axis=1)
        self.person_biopsy_timing[BiopsyColumns.CD_PACIENTE.value] = self.person_biopsy_timing[[BiopsyColumns.CD_PACIENTE.value, "argsort"]].apply(lambda x: np.array(x[BiopsyColumns.CD_PACIENTE.value])[x['argsort']], axis=1)
        self.person_biopsy_timing["biopsy_dates"] = self.person_biopsy_timing[["biopsy_dates", "argsort"]].apply(lambda x: np.array(x['biopsy_dates'])[x['argsort']], axis=1)
        self.person_biopsy_timing["biopsy_results"] = self.person_biopsy_timing[["biopsy_results", "argsort"]].apply(lambda x: np.array(x['biopsy_results'])[x['argsort']], axis=1)
        self.person_biopsy_timing = self.person_biopsy_timing.drop(columns=["argsort"])

    def _save_breast_biopsy_reports(self):
        self.biopsy_breast_df.to_parquet(self.transformed_filenames['breast_biopsy'])
        self._log(
            "saved_breast_biopsy", path=str(self.transformed_filenames['breast_biopsy']),
            rows=int(self.biopsy_breast_df.shape[0]) if self.biopsy_breast_df is not None else None
        )

    def _save_biopsy_results_by_person(self):
        self.person_biopsy_timing.to_parquet(self.transformed_filenames['person_biopsy'])
        self._log(
            "saved_person_biopsy", path=str(self.transformed_filenames['person_biopsy']),
            rows=int(self.person_biopsy_timing.shape[0]) if self.person_biopsy_timing is not None else None
        )

    def get_breast_biopsy_reports(self):
        t0 = dt.datetime.now()
        self._log("pipeline_start", name="get_breast_biopsy_reports")
        self._load_data(frac=1.0)
        self._simple_deduple()
        self._identify_if_breast()
        self._check_if_biopsy()
        self._filter_for_breast()
        self._get_person_code()
        self._save_breast_biopsy_reports()
        self._log("pipeline_end", name="get_breast_biopsy_reports", duration_s=(dt.datetime.now() - t0).total_seconds())

    def classify_breast_biopsy_reports(self, timer: Optional[int] = 1):
        self._log("pipeline_start", name="classify_breast_biopsy_reports")
        clf = BiopsyReportClassifier(self.config_dir, self.config_defaults)
        clf._load_data()
        clf.classify_reports(timer)
        self._log("pipeline_end", name="classify_breast_biopsy_reports")

    def get_breast_biopsy_results(self):
        self._log("pipeline_start", name="get_breast_biopsy_results")
        self._map_to_person()
        self._save_biopsy_results_by_person()
        self._log("pipeline_end", name="get_breast_biopsy_results")


# ------------------------------------------------------------------------------ #
# --------------------------------- COHORT DATA -------------------------------- #
# ------------------------------------------------------------------------------ #

class TransformPersonUser(ConfigInterface):
    def __init__(self, config_dir: str, config_defaults: dict):
        super().__init__(config_dir, config_defaults)

        # -- extract paths
        self.user_person_path = self.extract_path.joinpath(self.extract_folders['user_person_data'])

        # -- get all extracted files for mammograms and patients data
        self.person_from_mammograms_files = list(self.user_person_path.glob("person*.parquet"))
        self.user_from_mammograms_files = list(self.user_person_path.glob("user*.parquet"))
        self.patient_from_mammograms_files = list(self.user_person_path.glob("patient*.parquet"))

        # -- cols
        self.person_all_columns = [ elem.value for elem in PersonColumns ]
        self.patient_all_columns = [ elem.value for elem in PatientColumns ]
        self.user_all_columns = [ elem.value for elem in UserColumns ]

        self.person_unique_df = None # -- unique for "CD_PESSOA"
        self.patient_unique_df = None # -- unique for "CD_PACIENTE"
        self.person_df = None # -- unique for pairs ("CD_PESSOA", "CD_PACIENTE", "CD_USUARIO")
        self.user_df = None # -- unique for pairs ("CD_PESSOA", "CD_PACIENTE", "CD_USUARIO")
        self.similarity_df = None
        self.valid_person_patient_pairs = None

        self.processed_user = None
        self.processed_person = None

    def _load_person_data(self) -> None:
        # -- load unique
        self.person_unique_df = []
        subset = [
            PersonColumns.CD_PESSOA.value, PersonColumns.NM_PESSOA_RAZAO_SOCIAL.value,
            PersonColumns.DT_NASCIMENTO_FUNDACAO
        ]
        for cur_file in tqdm(self.person_from_mammograms_files):
            cur_df = pd.read_parquet(cur_file)[subset].drop_duplicates(subset=[PersonColumns.CD_PESSOA.value])
            self.person_unique_df.append( cur_df )
        self.person_unique_df = pd.concat(self.person_unique_df, ignore_index=True).reset_index(drop=True)
        self.person_unique_df = self.person_unique_df.drop_duplicates(subset=[PersonColumns.CD_PESSOA.value])
        
        # -- load pairs and covariates
        self.person_df = []
        subset_dup = [
            PersonColumns.CD_PESSOA.value, PersonColumns.CD_PACIENTE.value,
            PersonColumns.CD_USUARIO.value
        ]
        for cur_file in tqdm(self.person_from_mammograms_files):
            cur_df = pd.read_parquet(cur_file)[self.person_all_columns].copy()
            self.person_df.append(cur_df.drop_duplicates(subset=subset_dup))
        self.person_df = pd.concat(self.person_df, ignore_index=True).reset_index(drop=True)
        self.person_df = self.person_df.drop_duplicates(subset=subset_dup)

    def _load_user_data(self) -> None:
        self.user_df = []
        subset_dup = [
            UserColumns.CD_PESSOA.value, UserColumns.CD_PACIENTE.value,
            UserColumns.CD_USUARIO.value
        ]
        for cur_file in tqdm(self.user_from_mammograms_files):
            cur_df = pd.read_parquet(cur_file)[self.user_all_columns].copy()
            self.user_df.append(cur_df.drop_duplicates(subset=subset_dup))
        self.user_df = pd.concat(self.user_df, ignore_index=True).reset_index(drop=True)
        self.user_df = self.user_df.drop_duplicates(subset=subset_dup)

    def _load_patient_data(self) -> None:
        self.patient_unique_df = []
        subset = [
            PatientColumns.CD_PACIENTE.value, PatientColumns.NM_PACIENTE.value,
            PatientColumns.DT_NASCIMENTO.value
        ]
        # -- concat all chunks
        for cur_file in tqdm(self.patient_from_mammograms_files):
            cur_df = pd.read_parquet(cur_file)[subset].drop_duplicates(subset=[PatientColumns.CD_PACIENTE.value])
            cur_df = cur_df[(cur_df[PatientColumns.DT_NASCIMENTO.value]>=dt.datetime(1880,1,1)) & (cur_df[PatientColumns.DT_NASCIMENTO.value]<=dt.datetime(2030,1,1))]
            self.patient_unique_df.append(cur_df)
        self.patient_unique_df = pd.concat(self.patient_unique_df, ignore_index=True).reset_index(drop=True)
        self.patient_unique_df = self.patient_unique_df.drop_duplicates(subset=[PatientColumns.CD_PACIENTE.value])

    def _calculate_consistency(self, threshold=0.85):
        subset_dup = [PersonColumns.CD_PESSOA.value, PatientColumns.CD_PACIENTE.value]
        temp_ = self.person_df[subset_dup].copy().drop_duplicates(subset=subset_dup)
        temp_ = temp_.merge(self.person_unique_df, on=PersonColumns.CD_PESSOA.value, how="left")
        temp_ = temp_.merge(self.patient_unique_df, on=PatientColumns.CD_PACIENTE.value, how="left")

        # -- check whether the pairs really refer to the same patient
        temp_ = utils.add_person_similarity(
            temp_, 
            name_col_a=PersonColumns.NM_PESSOA_RAZAO_SOCIAL.value,
            name_col_b=PatientColumns.NM_PACIENTE.value,
            dob_col_a=PersonColumns.DT_NASCIMENTO_FUNDACAO,
            dob_col_b=PatientColumns.DT_NASCIMENTO.value,
            weight_name=0.70,
            weight_dob=0.30,
            out_col="similarity"
        )
        self.similarity_df = temp_.copy()
        self.valid_person_patient_pairs = self.similarity_df[self.similarity_df["similarity"]>=threshold][subset_dup].copy()

    def _process_user_info(self) -> None:
        # -- filter the person-patient pair that are valid (defined by the consistency check)
        #self.valid_person_patient_pairs["aux_pair"] = self.valid_person_patient_pairs["CD_PESSOA"].apply(lambda x: f"{x:.0f}") + self.valid_person_patient_pairs["CD_PACIENTE"].apply(lambda x: f"{x:.0f}")
        #self.user_df["aux_pair"] = self.user_df["CD_PESSOA"].apply(lambda x: f"{x:.0f}") + self.user_df["CD_PACIENTE"].apply(lambda x: f"{x:.0f}")

        #temp_user = self.user_df[self.user_df["aux_pair"].isin(self.valid_person_patient_pairs["aux_pair"])].copy()
        self.processed_user = self.user_df.groupby(UserColumns.CD_PESSOA.value).agg({
            UserColumns.CD_USUARIO.value: list, UserColumns.DT_REFERENCIA_CARENCIA.value: min, 
            UserColumns.VL_MENSALIDADE.value: min
        }
        ).rename({UserColumns.VL_MENSALIDADE.value: "VL_MENSALIDADE_MIN"}, axis=1)
        temp_2 = self.user_df.groupby(UserColumns.CD_PESSOA.value).agg({
            UserColumns.VL_MENSALIDADE.value: max
        }).rename({UserColumns.VL_MENSALIDADE.value: "VL_MENSALIDADE_MAX"}, axis=1)
        
        self.processed_user = self.processed_user.merge(temp_2, left_index=True, right_index=True, how="left")

    def _process_person_info(self) -> None:
        self.person_df = self.person_df.drop_duplicates(subset=[PersonColumns.CD_PESSOA.value], keep='first')
        self.person_df[PersonColumns.FL_SEXO.value] = self.person_df[PersonColumns.FL_SEXO.value].fillna("F").apply(lambda x: x.upper())
        self.person_df["FL_SEXO_ML"] = self.person_df[PersonColumns.FL_SEXO.value].map({"F": 0, "M": 1})
        self.person_df['zipcode_cat'] = self.person_df[PersonColumns.CD_CEP_ENDERECO.value].apply(
            lambda x: x.strip()[:3] if pd.notna(x) and len(x)==8 else np.nan
        ).astype("category").cat.add_categories("no_zip_code").fillna("no_zip_code")
    
    def transform(self) -> None:
        print("[load] data on person/patient/user ...")
        self._load_patient_data()
        self._load_user_data()
        self._load_person_data()
        print("[calculate] similarity between pairs ...")
        self._calculate_consistency(threshold=0.86)
        print("[process] personal info ...")
        self._process_user_info()
        self._process_person_info()
        print("[save] processed data ...")
        self.similarity_df.to_parquet(self.transformed_data_path.joinpath("person_similarity.parquet"))
        self.valid_person_patient_pairs.to_parquet(self.transformed_data_path.joinpath("valid_person_patient_pairs.parquet"))
        self.processed_user.to_parquet(self.transformed_data_path.joinpath("user_info_transformed.parquet"))
        self.person_df.to_parquet(self.transformed_data_path.joinpath("person_info_transformed.parquet"))


# ------------------------------------------------------------------------------ #
# ---------------------------------- MAMMOGRAMS -------------------------------- #
# ------------------------------------------------------------------------------ #

class TransformMammograms(ConfigInterface):
    def __init__(self, config_dir: str, config_defaults: dict):
        super().__init__(config_dir, config_defaults)

        # -- extract paths
        self.user_person_path = self.extract_path.joinpath(self.extract_folders['user_person_data'])

        # -- get all extracted files for mammograms and patients data
        self.person_from_mammograms_files = list(self.user_person_path.glob("person*.parquet"))
        self.patient_from_mammograms_files = list(self.user_person_path.glob("patient*.parquet"))

        # -- transform paths
        self.birads_folder = self.transform_path.joinpath(self.files_and_folders_cfg["transform"]["folders"]["birads"])

        # -- get file with the extracted birads for all mammograms
        # -- bi-rads were extracted through two approaches: 1) Regular expression (majority); 2) Bag-of-Words classifier;
        # -- by the stage where this routine is run, exists one file for each approach
        self.re_birads_final_file = list(self.birads_folder.glob("processed*.parquet"))
        self.ml_birads_final_file = list(self.birads_folder.glob("infered*.parquet"))

        self.patient_to_person = None
        self.person_unique_df = None
        self.patient_unique_df = None
        self.birads_final_df = None
        self.person_df = None
        self.similarity_df = None
        self.valid_person_patient_pairs = None

        self.transformed_data_path = self.transform_path.joinpath("transformed")
        self.transformed_data_path.mkdir(parents=True, exist_ok=True)

        ext = ".parquet"
        self.transformed_filenames = {
            'anamnesis': self.transformed_data_path.joinpath(self.files_and_folders_cfg['transform']['transformed_files']['anamnesis']+ext),
            'person': self.transformed_data_path.joinpath(self.files_and_folders_cfg['transform']['transformed_files']['person']+ext),
            'user': self.transformed_data_path.joinpath(self.files_and_folders_cfg['transform']['transformed_files']['user']+ext),
            'similarity_data': self.transformed_data_path.joinpath(self.files_and_folders_cfg['transform']['transformed_files']['similarity_data']+ext),
            'valid_person_patient': self.transformed_data_path.joinpath(self.files_and_folders_cfg['transform']['transformed_files']['valid_person_patient']+ext),
            'biopsy': self.transformed_data_path.joinpath(self.files_and_folders_cfg['transform']['transformed_files']['person_biopsy']+ext),
            'person_mammogram': self.transformed_data_path.joinpath(self.files_and_folders_cfg['transform']['transformed_files']['person_mammogram']+ext),
        }

    def _load_person_data(self) -> None:
        # -- load unique
        self.person_unique_df = []
        subset = [PersonColumns.CD_PESSOA.value, PersonColumns.NM_PESSOA_RAZAO_SOCIAL.value, PersonColumns.DT_NASCIMENTO_FUNDACAO.value]
        for cur_df in tqdm(self._iter_raw_person_data(deduple_columns=[PersonColumns.CD_PESSOA.value])):
            self.person_unique_df.append( cur_df )
        self.person_unique_df = pd.concat(self.person_unique_df, ignore_index=True).reset_index(drop=True)
        self.person_unique_df = self.person_unique_df.drop_duplicates(subset=[PersonColumns.CD_PESSOA.value])

        # -- load pairs and covariates
        self.person_df = []
        subset_dup = [PersonColumns.CD_PESSOA.value, PersonColumns.CD_PACIENTE.value, PersonColumns.CD_USUARIO.value]
        # -- can be modified in case some of these variables should be used in the model.
        drop_col = [
            "NM_PAI", "NM_MAE", "FL_TIPO_ENDERECO", "NM_BAIRRO_ENDERECO",
            "CD_FATOR_RH", "CD_GRUPO_SANGUINEO", "CD_COR"
        ]
        for cur_df in tqdm(self._iter_raw_person_data(deduple_columns=subset_dup)):
            self.person_df.append( cur_df.drop(columns=drop_col) )
        self.person_df = pd.concat(self.person_df, ignore_index=True).reset_index(drop=True)
        self.person_df = self.person_df.drop_duplicates(subset=subset_dup)

    def _load_patient_data(self) -> None:
        self.patient_unique_df = []
        subset = [PatientColumns.CD_PACIENTE.value, PatientColumns.NM_PACIENTE.value, PatientColumns.DT_NASCIMENTO.value]
        # -- concat all chunks
        for cur_df in tqdm(self._iter_raw_patient_data(deduple_columns=[PatientColumns.CD_PACIENTE.value])):
            cur_df = cur_df[(cur_df[PatientColumns.DT_NASCIMENTO.value]>=dt.datetime(1880,1,1)) & (cur_df[PatientColumns.DT_NASCIMENTO.value]<=dt.datetime(2040,1,1))]
            self.patient_unique_df.append(cur_df)
        self.patient_unique_df = pd.concat(self.patient_unique_df, ignore_index=True).reset_index(drop=True)
        self.patient_unique_df = self.patient_unique_df.drop_duplicates(subset=[PatientColumns.CD_PACIENTE.value])

        self.patient_to_person = []
        subset = [PatientColumns.CD_PACIENTE.value, PatientColumns.CD_PESSOA.value]
        # -- concat all chunks
        for cur_df in tqdm(self._iter_raw_patient_data(columns=subset)):
            self.patient_to_person.append(cur_df)
        self.patient_to_person = pd.concat(self.patient_to_person, ignore_index=True).reset_index(drop=True)
        self.patient_to_person = self.patient_to_person.drop_duplicates(subset=subset)

    def _calculate_consistency(self, threshold=0.85):
        subset_dup = [PersonColumns.CD_PESSOA.value, PatientColumns.CD_PACIENTE.value]
        temp_ = self.person_df[subset_dup].copy().drop_duplicates(subset=subset_dup).drop(columns=[PatientColumns.CD_PACIENTE.value])
        temp_ = temp_.merge(self.person_unique_df, on=PersonColumns.CD_PESSOA.value, how="left")
        temp_ = temp_.merge(self.patient_unique_df.drop(columns=[PersonColumns.CD_PESSOA.value]), on=PatientColumns.CD_PACIENTE.value, how="left")

        # -- check whether the pairs really refer to the same patient
        temp_ = utils.add_person_similarity(
            temp_, 
            name_col_a=PersonColumns.NM_PESSOA_RAZAO_SOCIAL.value,
            name_col_b=PatientColumns.NM_PACIENTE.value,
            dob_col_a=PersonColumns.DT_NASCIMENTO_FUNDACAO.value,
            dob_col_b=PatientColumns.DT_NASCIMENTO.value,
            weight_name=0.70,
            weight_dob=0.30,
            out_col="similarity"
        )
        self.similarity_df = temp_.copy()
        self.valid_person_patient_pairs = self.similarity_df[self.similarity_df["similarity"]>=threshold][subset_dup].copy()
    
    def _load_final_birads_data(self) -> None:
        # ================== RE EXTRACTED BI-RADS
        # -- birads data holds no info on MammogramColumns.CD_PESSOA.value, only on MammogramColumns.CD_PACIENTE.value.
        subset = [
            "key", MammogramColumns.CD_ATENDIMENTO.value, MammogramColumns.DT_ATENDIMENTO.value,
            MammogramColumns.CD_PACIENTE.value, "processed_birads"
        ]
        self.birads_final_df = pd.read_parquet(self.re_birads_final_file[0])
        self.birads_final_df["key"] = self.birads_final_df[MammogramColumns.CD_ATENDIMENTO.value].apply(lambda x: f"{x:.0f}") + self.birads_final_df["raw_text_hash"]
        self.birads_final_df = self.birads_final_df[subset].copy()

        # -- sanity check
        self.birads_final_df = self.birads_final_df[pd.notna(self.birads_final_df["processed_birads"])]
        self.birads_final_df = self.birads_final_df[(self.birads_final_df["processed_birads"]>=0) & (self.birads_final_df["processed_birads"]<=6)]

        # ================= ML EXTRACTED BI-RADS
        subset = [
            "key", MammogramColumns.CD_ATENDIMENTO.value, MammogramColumns.DT_ATENDIMENTO.value, 
            MammogramColumns.CD_PACIENTE.value, "predicted_birads"
        ]
        ml_birads_final_df = pd.read_parquet(self.ml_birads_final_file[0])[subset]
        ml_birads_final_df = ml_birads_final_df[(ml_birads_final_df["predicted_birads"]>=0) & (ml_birads_final_df["predicted_birads"]<=6)]
        self.birads_final_df = pd.concat([self.birads_final_df, ml_birads_final_df], axis=0)
        #self.birads_final_df = self.birads_final_df.merge(ml_birads_final_df[["key", "predicted_birads"]], on="key", how="left")
        birads_f = lambda x: x["processed_birads"] if pd.notna(x["processed_birads"]) else x["predicted_birads"]
        self.birads_final_df["processed_birads"] = self.birads_final_df[["processed_birads", "predicted_birads"]].apply(birads_f, axis=1)
        self.birads_final_df = self.birads_final_df.drop(columns=["predicted_birads"])
        self.birads_final_df = self.birads_final_df[pd.notna(self.birads_final_df["processed_birads"])]
        

    def transform(self) -> None:
        print("[load] patient and birads data ...")
        self._load_patient_data()
        self._load_person_data()
        self._load_final_birads_data()

        print("[calculate] consistency between person and patient data ...")
        self._calculate_consistency(threshold=0.85)

        # -- generate pairs MammogramColumns.CD_PESSOA.value-MammogramColumns.CD_PACIENTE.value
        self.birads_final_df = self.birads_final_df.merge(self.patient_to_person, on=MammogramColumns.CD_PACIENTE.value, how="left")

        self.valid_person_patient_pairs["aux_pair"] = self.valid_person_patient_pairs[PersonColumns.CD_PESSOA.value].apply(lambda x: f"{x:.0f}") + self.valid_person_patient_pairs[MammogramColumns.CD_PACIENTE.value].apply(lambda x: f"{x:.0f}")
        self.birads_final_df["aux_pair"] = self.birads_final_df[PersonColumns.CD_PESSOA.value].apply(lambda x: f"{x:.0f}") + self.birads_final_df[MammogramColumns.CD_PACIENTE.value].apply(lambda x: f"{x:.0f}")

        # -- keep only the consistent pairs
        self.birads_final_df = self.birads_final_df[self.birads_final_df["aux_pair"].isin(self.valid_person_patient_pairs["aux_pair"])].copy()
        self.birads_final_df = self.birads_final_df.drop(columns=["aux_pair"])
        self.valid_person_patient_pairs = self.valid_person_patient_pairs.drop(columns=["aux_pair"])

        print("[formatting] sequential mammogram data per person ...")
        self.person_mammogram_timing = self.birads_final_df.groupby(PersonColumns.CD_PESSOA.value).agg({
            "key": list, MammogramColumns.CD_ATENDIMENTO.value: list, MammogramColumns.DT_ATENDIMENTO.value: list, 
            "processed_birads": list, MammogramColumns.CD_PACIENTE.value: list,
        })

        # -- sort sequences by date
        print("[sorting] sequences ...")
        self.person_mammogram_timing['argsort'] = self.person_mammogram_timing[MammogramColumns.DT_ATENDIMENTO.value].apply(lambda x: np.argsort(x))
        self.person_mammogram_timing["key"] = self.person_mammogram_timing[["key", 'argsort']].apply(lambda x: np.array(x['key'])[x['argsort']], axis=1)
        self.person_mammogram_timing[MammogramColumns.CD_ATENDIMENTO.value] = self.person_mammogram_timing[[MammogramColumns.CD_ATENDIMENTO.value, 'argsort']].apply(lambda x: np.array(x[MammogramColumns.CD_ATENDIMENTO.value])[x['argsort']], axis=1)
        self.person_mammogram_timing[MammogramColumns.DT_ATENDIMENTO.value] = self.person_mammogram_timing[[MammogramColumns.DT_ATENDIMENTO.value, 'argsort']].apply(lambda x: np.array(x[MammogramColumns.DT_ATENDIMENTO.value])[x['argsort']], axis=1)
        self.person_mammogram_timing["birads_labels"] = self.person_mammogram_timing[["processed_birads", 'argsort']].apply(lambda x: np.array(x['processed_birads'])[x['argsort']], axis=1)
        self.person_mammogram_timing = self.person_mammogram_timing.drop(columns=["argsort", "processed_birads"])
        
        # -- identify the indices of the sequences where bi-rads are 4 or higher -> goal: find the first one.
        conditional_birads_val = lambda birads_arr, val: np.where(np.array(birads_arr) == val)[0]
        birads456_indices_f = lambda arr: np.concatenate(( conditional_birads_val(arr, 4), conditional_birads_val(arr, 5), conditional_birads_val(arr, 6) ))
        self.person_mammogram_timing["birads456_indices"] = self.person_mammogram_timing['birads_labels'].apply(birads456_indices_f)
        self.person_mammogram_timing["first_high_birads_idx"] = self.person_mammogram_timing["birads456_indices"].apply(lambda x: min(x) if len(x)>0 else np.nan)
        
        # -- we do not follow the patient from the first high bi-rads (4 or higher) forward
        self.person_mammogram_timing["birads_upto3"] = self.person_mammogram_timing[['birads_labels', "first_high_birads_idx"]].apply(lambda x: x['birads_labels'][:int(x["first_high_birads_idx"])] if pd.notna(x["first_high_birads_idx"]) else x['birads_labels'], axis=1)
        self.person_mammogram_timing["mammogram_codes_upto3"] = self.person_mammogram_timing[['key', "first_high_birads_idx"]].apply(lambda x: x['key'][:int(x["first_high_birads_idx"])] if pd.notna(x["first_high_birads_idx"]) else x['key'], axis=1)
        self.person_mammogram_timing["mammogram_dates_upto3"] = self.person_mammogram_timing[[MammogramColumns.DT_ATENDIMENTO.value, "first_high_birads_idx"]].apply(
            lambda x: x[MammogramColumns.DT_ATENDIMENTO.value][:int(x["first_high_birads_idx"])] if pd.notna(x["first_high_birads_idx"]) else x[MammogramColumns.DT_ATENDIMENTO.value], axis=1
        )
        self.person_mammogram_timing["first_mammogram_date"] = self.person_mammogram_timing[MammogramColumns.DT_ATENDIMENTO.value].apply(lambda x: x[0]) 
        self.person_mammogram_timing["last_benign_mammogram_date"] = self.person_mammogram_timing["mammogram_dates_upto3"].apply(lambda x: x[-1] if len(x)>0 else np.nan)
        self.person_mammogram_timing = self.person_mammogram_timing.drop(columns=["birads456_indices", "first_high_birads_idx"])

        # -- persist
        self.similarity_df.to_parquet(self.transformed_filenames['similarity_data'])
        self.valid_person_patient_pairs.to_parquet(self.transformed_filenames['valid_person_patient'])
        self.person_mammogram_timing.to_parquet(self.transformed_filenames['person_mammogram'])
