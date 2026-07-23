import re
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import hashlib
from collections import defaultdict
from typing import List, Optional, Dict, Tuple

from hapcancer.schemas.enums import MammogramColumns
from hapcancer.config_manager import ConfigInterface
import hapcancer.etl.transform.process_birads.utils as utils
from hapcancer.etl.utils import sha1

# ------------------------------------------------------------------------------ #
# ---------------------------- AUXILIARY FUNCTIONS ----------------------------- #
# ------------------------------------------------------------------------------ #

def load_data(
    SELF_OBJ: ConfigInterface
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    hsp_df = [ cur_df for cur_df in SELF_OBJ._iter_raw_mammograms_data(file_glob_pattern="*hsp*") ]
    psc_df = [ cur_df for cur_df in SELF_OBJ._iter_raw_mammograms_data(file_glob_pattern="*psc*") ]
    hsp_df = pd.concat(hsp_df, ignore_index=True).reset_index(drop=True)
    psc_df = pd.concat(psc_df, ignore_index=True).reset_index(drop=True)

    temp_df = hsp_df[MammogramColumns.CD_ATENDIMENTO.value].value_counts().reset_index()
    temp_df = temp_df[temp_df["count"]>1]
    hsp_df_single = hsp_df[~hsp_df[MammogramColumns.CD_ATENDIMENTO.value].isin(temp_df[MammogramColumns.CD_ATENDIMENTO.value])]
    hsp_df_multiple = hsp_df[hsp_df[MammogramColumns.CD_ATENDIMENTO.value].isin(temp_df[MammogramColumns.CD_ATENDIMENTO.value])]

    temp_df = psc_df[MammogramColumns.CD_ATENDIMENTO.value].value_counts().reset_index()
    temp_df = temp_df[temp_df["count"]>1]
    psc_df_single = psc_df[~psc_df[MammogramColumns.CD_ATENDIMENTO.value].isin(temp_df[MammogramColumns.CD_ATENDIMENTO.value])]
    psc_df_multiple = psc_df[psc_df[MammogramColumns.CD_ATENDIMENTO.value].isin(temp_df[MammogramColumns.CD_ATENDIMENTO.value])]
    return (hsp_df_single, hsp_df_multiple, psc_df_single, psc_df_multiple)

def extract_birads(
    hsp_df_single: pd.DataFrame, 
    hsp_df_multiple: pd.DataFrame, 
    psc_df_single: pd.DataFrame, 
    psc_df_multiple: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    
    # -- version 1 regular expression extraction
    hsp_df_single["birads_v1"] = hsp_df_single[MammogramColumns.DS_LAUDO_MEDICO.value].apply(lambda x: utils.extract_birads_v1(x) if pd.notna(x) else np.nan)
    hsp_df_multiple["birads_v1"] = hsp_df_multiple[MammogramColumns.DS_LAUDO_MEDICO.value].apply(lambda x: utils.extract_birads_v1(x) if pd.notna(x) else np.nan)
    psc_df_single["birads_v1"] = psc_df_single[MammogramColumns.DS_LAUDO_MEDICO.value].apply(lambda x: utils.extract_birads_v1(x) if pd.notna(x) else np.nan)
    psc_df_multiple["birads_v1"] = psc_df_multiple[MammogramColumns.DS_LAUDO_MEDICO.value].apply(lambda x: utils.extract_birads_v1(x) if pd.notna(x) else np.nan)
        
    # -- version 2 regular expression extraction for the ones not captured by the previous version
    setcols = [MammogramColumns.DS_LAUDO_MEDICO.value, "birads_v1"]
    hsp_df_single["birads_v2"] = hsp_df_single[setcols].apply(lambda x: utils.extract_birads_v2(x[setcols[0]]) if pd.notna(x[setcols[0]]) and pd.isna(x["birads_v1"]) else np.nan, axis=1)
    hsp_df_multiple["birads_v2"] = hsp_df_multiple[setcols].apply(lambda x: utils.extract_birads_v2(x[setcols[0]]) if pd.notna(x[setcols[0]]) and pd.isna(x["birads_v1"]) else np.nan, axis=1)
    psc_df_single["birads_v2"] = psc_df_single[setcols].apply(lambda x: utils.extract_birads_v2(x[setcols[0]]) if pd.notna(x[setcols[0]]) and pd.isna(x["birads_v1"]) else np.nan, axis=1)
    psc_df_multiple["birads_v2"] = psc_df_multiple[setcols].apply(lambda x: utils.extract_birads_v2(x[setcols[0]]) if pd.notna(x[setcols[0]]) and pd.isna(x["birads_v1"]) else np.nan, axis=1)

    setcols = [MammogramColumns.DS_LAUDO_MEDICO.value, "birads_v1", "birads_v2"]
    hsp_df_single["is_for_breast"] = hsp_df_single[setcols].apply(lambda x: utils.is_breast_exam(x[setcols[0]]) if pd.notna(x[setcols[0]]) and all(pd.isna( [ x[setcols[1]], x[setcols[2]] ])) else np.nan, axis=1)
    hsp_df_multiple["is_for_breast"] = hsp_df_multiple[setcols].apply(lambda x: utils.is_breast_exam(x[setcols[0]]) if pd.notna(x[setcols[0]]) and all(pd.isna( [ x[setcols[1]], x[setcols[2]] ])) else np.nan , axis=1)
    psc_df_single["is_for_breast"] = psc_df_single[setcols].apply(lambda x: utils.is_breast_exam(x[setcols[0]]) if pd.notna(x[setcols[0]]) and all(pd.isna( [ x[setcols[1]], x[setcols[2]] ])) else np.nan , axis=1)
    psc_df_multiple["is_for_breast"] = psc_df_multiple[setcols].apply(lambda x: utils.is_breast_exam(x[setcols[0]]) if pd.notna(x[setcols[0]]) and all(pd.isna( [ x[setcols[1]], x[setcols[2]] ])) else np.nan , axis=1)

    return (hsp_df_single, hsp_df_multiple, psc_df_single, psc_df_multiple)

# ------------------------------------------------------------------------------ #
# -------------------------------- BIRADS CLASS -------------------------------- #
# ------------------------------------------------------------------------------ #

# -- not very efficient storage and processing (it gets too much RAM)
class GetBirads(ConfigInterface):
    def __init__(self, config_dir: str, config_defaults: dict):
        super().__init__(config_dir, config_defaults)
    
        self.hsp_df_single = None
        self.psc_df_single = None
        self.hsp_df_multiple = None
        self.psc_df_multiple = None

        self.hsp_sample_single = None
        self.psc_sample_single = None
        self.hsp_sample_multiple = None
        self.psc_sample_multiple = None

    def load_data(self):
        self.hsp_df_single, self.hsp_df_multiple, self.psc_df_single, self.psc_df_multiple = load_data(self)

    def extract_birads(
        self, 
        frac : Optional[float] = 1.0,
        persist: Optional[bool] = True
    ) -> None:
        hsp_single_active, hsp_multiple_active = self.hsp_df_single, self.hsp_df_multiple
        psc_single_active, psc_multiple_active = self.psc_df_single, self.psc_df_multiple
        if frac<1.0:
            hsp_single_active, hsp_multiple_active = self.hsp_df_single.sample(frac=frac), self.hsp_df_multiple.sample(frac=frac)
            psc_single_active, psc_multiple_active = self.psc_df_single.sample(frac=frac), self.psc_df_multiple.sample(frac=frac)

        res_tuple = extract_birads(hsp_single_active, hsp_multiple_active, psc_single_active, psc_multiple_active)
        if persist:
            res_tuple[0].to_parquet(self.processed_birads_folder_path.joinpath("hsp_single_exam_birads.parquet"))
            res_tuple[1].to_parquet(self.processed_birads_folder_path.joinpath("hsp_multiple_exam_birads.parquet"))
            res_tuple[2].to_parquet(self.processed_birads_folder_path.joinpath("psc_single_exam_birads.parquet"))
            res_tuple[3].to_parquet(self.processed_birads_folder_path.joinpath("psc_multiple_exam_birads.parquet"))

    def process_extracted_birads_for_ml(self):
        birads_files_single = list(self.processed_birads_folder_path.glob("*single*.parquet"))
        birads_files_multiple = list(self.processed_birads_folder_path.glob("*multiple*.parquet"))
        print("[display] available files: ")
        print([ fname.name for fname in birads_files_single ] )
        print([ fname.name for fname in birads_files_multiple ])

        # -- open birads files
        print("load files ...")
        self.single_df = pd.concat([pd.read_parquet(birads_files_single[0]), pd.read_parquet(birads_files_single[1])], ignore_index=True).reset_index(drop=True)
        self.multiple_df = pd.concat([pd.read_parquet(birads_files_multiple[0]), pd.read_parquet(birads_files_multiple[1])], ignore_index=True).reset_index(drop=True)

        print("consolidate extracted strings ...")
        agg_birads_f = lambda x: x["birads_v1"]["value"] if pd.notna(x["birads_v1"]) else ( x["birads_v2"]["value"] if pd.notna(x["birads_v2"]) else np.nan )
        self.single_df["birads"] = self.single_df[["birads_v1", "birads_v2"]].apply(agg_birads_f, axis=1)
        self.multiple_df["birads"] = self.multiple_df[["birads_v1", "birads_v2"]].apply(agg_birads_f, axis=1)

        print("define hash for the raw text ...")
        self.single_df["raw_text_hash"] = self.single_df[MammogramColumns.DS_LAUDO_MEDICO.value].apply(lambda x: sha1(x.strip()) if pd.notna(x) else np.nan)
        self.multiple_df["raw_text_hash"] = self.multiple_df[MammogramColumns.DS_LAUDO_MEDICO.value].apply(lambda x: sha1(x.strip()) if pd.notna(x) else np.nan)

        self.single_df = self.single_df.drop_duplicates(subset=[MammogramColumns.CD_ATENDIMENTO.value, "raw_text_hash"], keep="first")
        self.multiple_df = self.multiple_df.drop_duplicates(subset=[MammogramColumns.CD_ATENDIMENTO.value, "raw_text_hash"], keep="first")

        # -- fix category 0
        print("refine extracted birads ...")
        pat = re.compile(r'(?<![\d/\-:\.])(0[0-9])(?![\d/\-:\.])')
        self.single_df["birads_v1_0"] = self.single_df[self.single_df["birads"]=="0"][["birads_v1", "birads"]].apply(lambda x: re.findall(pat, x["birads_v1"]["clause"]) if pd.notna(x["birads_v1"]) and x["birads"]=="0" else [], axis=1)
        self.single_df["birads_v2_0"] = self.single_df[self.single_df["birads"]=="0"][["birads_v2", "birads"]].apply(lambda x: re.findall(pat, x["birads_v2"]["clause"]) if pd.notna(x["birads_v2"]) and x["birads"]=="0" else [], axis=1)

        self.multiple_df["birads_v1_0"] = self.multiple_df[self.multiple_df["birads"]=="0"][["birads_v1", "birads"]].apply(lambda x: re.findall(pat, x["birads_v1"]["clause"]) if pd.notna(x["birads_v1"]) and x["birads"]=="0" else [], axis=1)
        self.multiple_df["birads_v2_0"] = self.multiple_df[self.multiple_df["birads"]=="0"][["birads_v2", "birads"]].apply(lambda x: re.findall(pat, x["birads_v2"]["clause"]) if pd.notna(x["birads_v2"]) and x["birads"]=="0" else [], axis=1)

        self.single_df["birads_aux0"] = self.single_df[["birads", "birads_v1_0", "birads_v2_0"]].apply(lambda x: x["birads"] if pd.isna(x["birads"]) or x["birads"]!="0" else ( x["birads_v1_0"] if len(x["birads_v1_0"])>0 else x["birads_v2_0"] ), axis=1)
        self.single_df["birads"] = self.single_df[["birads", "birads_aux0"]].apply(lambda x: x["birads"] if pd.isna(x["birads"]) or x["birads"]!="0" else ( x["birads"] if len(x["birads_aux0"])==0 else x["birads_aux0"][0] ), axis=1)

        self.multiple_df["birads_aux0"] = self.multiple_df[["birads", "birads_v1_0", "birads_v2_0"]].apply(lambda x: x["birads"] if pd.isna(x["birads"]) or x["birads"]!="0" else ( x["birads_v1_0"] if len(x["birads_v1_0"])>0 else x["birads_v2_0"] ), axis=1)
        self.multiple_df["birads"] = self.multiple_df[["birads", "birads_aux0"]].apply(lambda x: x["birads"] if pd.isna(x["birads"]) or x["birads"]!="0" else ( x["birads"] if len(x["birads_aux0"])==0 else x["birads_aux0"][0] ), axis=1)

        self.single_df = self.single_df.drop(columns=["birads_v1_0", "birads_v2_0", "birads_aux0"])
        self.multiple_df = self.multiple_df.drop(columns=["birads_v1_0", "birads_v2_0", "birads_aux0"])

        cl4 = defaultdict(lambda: np.nan, {'4A': '4', '4B': '4', '4C': '4'})
        def process_value(val):
            conv_val = cl4[val]
            if pd.notna(conv_val):
                val = conv_val
            try:
                val = int(val)
            except:
                val = np.nan
            if val<0 or val>6:
                val = np.nan
            return val

        print("final processing ...")
        self.single_df["processed_birads"] = self.single_df["birads"].apply(process_value)
        self.multiple_df["processed_birads"] = self.multiple_df["birads"].apply(process_value)

        # -- keep only the exams for the breast
        print("keep exams for breast and persist file ...")
        self.single_df = self.single_df[self.single_df["is_for_breast"].apply(lambda x: True if pd.isna(x) else x["is_breast"])].copy()
        self.multiple_df = self.multiple_df[self.multiple_df["is_for_breast"].apply(lambda x: True if pd.isna(x) else x["is_breast"])].copy()
        self.single_df = pd.concat([self.single_df, self.multiple_df], ignore_index=True).reset_index(drop=True)
        self.single_df = self.single_df.drop_duplicates(subset=[MammogramColumns.CD_ATENDIMENTO.value, "raw_text_hash"], keep='first')
        self.single_df = self.single_df.drop(columns=["birads", MammogramColumns.DS_LAUDO_MEDICO, MammogramColumns.NM_EXAME.value])
        self.single_df.to_parquet(self.processed_birads_folder_path.joinpath("processed_birads_for_training.parquet"))

    def get(self):
        self.load_data()
        self.extract_birads(frac=1.0, persist=True)
        self.process_extracted_birads_for_ml()




