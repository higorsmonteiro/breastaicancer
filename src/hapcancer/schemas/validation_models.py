'''
    Define the expected schemas for the raw data used in the ETL pipeline.
'''
from pydantic import BaseModel, Field
from typing import List, Sequence, Optional, Union

import numpy as np
import pandas as pd 
from datetime import datetime

from hapcancer.schemas.enums import (
    AnamnesisColumns, BiopsyColumns, MammogramColumns,
    PersonColumns, PatientColumns, UserColumns,
    MergedSourcesColumns
)

# ---------------------------------------------------------- #
# ------------------------- CONFIG ------------------------- #
# ---------------------------------------------------------- #

class ConfigDefaults(BaseModel):
    birads_classifier: Optional[str]
    embeddings: Optional[str]
    split: Optional[str]
    bmi_model: Optional[str]
    followup: Optional[str]
    tuning: Optional[str]
    training_experiments: Optional[str]

# ---------------------------------------------------------- #
# ------------------------ RAW DATA ------------------------ #
# ---------------------------------------------------------- #

class RawAnamnesisData(BaseModel):
    CD_ATENDIMENTO: Union[int, float] = Field(alias=AnamnesisColumns.CD_ATENDIMENTO.value)
    CD_PACIENTE: Union[int, float] = Field(alias=AnamnesisColumns.CD_PACIENTE.value)
    CD_PESSOA: Union[int, float] = Field(alias=AnamnesisColumns.CD_PESSOA.value)
    CD_USUARIO: Optional[str] = Field(alias=AnamnesisColumns.CD_USUARIO.value)
    DS_INDICACAO_QUEIXA: Optional[Union[str, float]] = Field(alias=AnamnesisColumns.DS_INDICACAO_QUEIXA.value)
    DT_ATENDIMENTO: datetime = Field(alias=AnamnesisColumns.DT_ATENDIMENTO.value)
    DS_MENARCA: Optional[str] = Field(alias=AnamnesisColumns.DS_MENARCA.value)
    DS_MENOPAUSA: Optional[str] = Field(alias=AnamnesisColumns.DS_MENOPAUSA.value)
    NU_GESTACAO: Optional[float] = Field(alias=AnamnesisColumns.NU_GESTACAO.value)
    NU_GESTACAO_ABORTO: Optional[float] = Field(alias=AnamnesisColumns.NU_GESTACAO_ABORTO.value)
    FL_ALEITAMENTO: Optional[str] = Field(alias=AnamnesisColumns.FL_ALEITAMENTO.value)
    FL_CA_MAMA_MAE: Optional[str] = Field(alias=AnamnesisColumns.FL_CA_MAMA_MAE.value)
    FL_CA_MAMA_IRMA: Optional[str] = Field(alias=AnamnesisColumns.FL_CA_MAMA_IRMA.value)
    FL_CA_MAMA_AVO: Optional[str] = Field(alias=AnamnesisColumns.FL_CA_MAMA_AVO.value)
    FL_CA_MAMA_TIA: Optional[str] = Field(alias=AnamnesisColumns.FL_CA_MAMA_TIA.value)
    DT_BIOPSIA_ME: Optional[datetime] = Field(alias=AnamnesisColumns.DT_BIOPSIA_ME.value)
    DT_BIOPSIA_MD: Optional[datetime] = Field(alias=AnamnesisColumns.DT_BIOPSIA_MD.value)
    DT_QUADRANTECTOMIA_ME: Optional[datetime] = Field(alias=AnamnesisColumns.DT_QUADRANTECTOMIA_ME.value)
    DT_QUADRANTECTOMIA_MD: Optional[datetime] = Field(alias=AnamnesisColumns.DT_QUADRANTECTOMIA_MD.value)
    DT_MASTECTOMIA_ME: Optional[datetime] = Field(alias=AnamnesisColumns.DT_MASTECTOMIA_ME.value)
    DT_MASTECTOMIA_MD: Optional[datetime] = Field(alias=AnamnesisColumns.DT_MASTECTOMIA_MD.value)
    DT_PLASTICA_ME: Optional[datetime] = Field(alias=AnamnesisColumns.DT_PLASTICA_ME.value)
    DT_PLASTICA_MD: Optional[datetime] = Field(alias=AnamnesisColumns.DT_PLASTICA_MD.value)
    FL_BIOPSIA_ME: Optional[str] = Field(alias=AnamnesisColumns.FL_BIOPSIA_ME.value)
    FL_BIOPSIA_MD: Optional[str] = Field(alias=AnamnesisColumns.FL_BIOPSIA_MD.value)
    FL_QUADRANTECTOMIA_ME: Optional[str] = Field(alias=AnamnesisColumns.FL_QUADRANTECTOMIA_ME.value)
    FL_QUADRANTECTOMIA_MD: Optional[str] = Field(alias=AnamnesisColumns.FL_QUADRANTECTOMIA_MD.value)
    FL_MASTECTOMIA_ME: Optional[str] = Field(alias=AnamnesisColumns.FL_MASTECTOMIA_ME.value)
    FL_MASTECTOMIA_MD: Optional[str] = Field(alias=AnamnesisColumns.FL_MASTECTOMIA_MD.value)

    model_config = {"populate_by_name": True, "extra": "ignore"}
    
class RawMammogramData(BaseModel):
    NM_EXAME: Optional[str] = Field(alias=MammogramColumns.NM_EXAME.value)
    CD_ATENDIMENTO: Union[int, float] = Field(alias=MammogramColumns.CD_ATENDIMENTO.value)
    DT_ATENDIMENTO: datetime = Field(alias=MammogramColumns.DT_ATENDIMENTO.value)
    DS_LAUDO_MEDICO: Optional[str] = Field(alias=MammogramColumns.DS_LAUDO_MEDICO.value)
    CD_PACIENTE: Union[int, float] = Field(alias=MammogramColumns.CD_PACIENTE.value)

    model_config = {"populate_by_name": True, "extra": "ignore"}

class RawUserData(BaseModel):
    CD_USUARIO: str = Field(alias=UserColumns.CD_USUARIO.value)
    CD_PACIENTE: Optional[Union[int, float]] = Field(alias=UserColumns.CD_PACIENTE.value)
    CD_PESSOA: Optional[Union[int, float]] = Field(alias=UserColumns.CD_PESSOA.value)
    NU_USUARIO: Union[int, float] = Field(alias=UserColumns.NU_USUARIO.value)
    NU_TITULAR: Union[int, float] = Field(alias=UserColumns.NU_TITULAR.value)
    CD_CANCELAMENTO: Optional[Union[int, float]] = Field(alias=UserColumns.CD_CANCELAMENTO.value)
    DT_REFERENCIA_CARENCIA: Optional[datetime] = Field(alias=UserColumns.DT_REFERENCIA_CARENCIA.value)
    DT_CADASTRAMENTO: Optional[datetime] = Field(alias=UserColumns.DT_CADASTRAMENTO.value)
    DT_CANCELAMENTO: Optional[datetime] = Field(alias=UserColumns.DT_CANCELAMENTO.value)
    DT_MENSALIDADE_A: Optional[datetime] = Field(alias=UserColumns.DT_MENSALIDADE_A.value)
    VL_MENSALIDADE_A: Optional[float] = Field(alias=UserColumns.VL_MENSALIDADE_A.value)
    VL_MENSALIDADE: Optional[float] = Field(alias=UserColumns.VL_MENSALIDADE.value)
    VL_TAXA_ADESAO: Optional[float] = Field(alias=UserColumns.VL_TAXA_ADESAO.value)

    model_config = {"populate_by_name": True, "extra": "ignore"}

class RawPatientData(BaseModel):
    CD_PACIENTE: Union[int, float] = Field(alias=PatientColumns.CD_PACIENTE.value)
    CD_USUARIO: Optional[str] = Field(alias=PatientColumns.CD_USUARIO.value)
    CD_PESSOA: Optional[Union[int, float]] = Field(alias=PatientColumns.CD_PESSOA.value)
    DT_NASCIMENTO: Optional[datetime] = Field(alias=PatientColumns.DT_NASCIMENTO.value)
    CD_SEXO: Optional[Union[int, str]] = Field(alias=PatientColumns.CD_SEXO.value)
    NM_MUNICIPIO: Optional[str] = Field(alias=PatientColumns.NM_MUNICIPIO.value)
    CD_UF: Optional[Union[int, float, str]] = Field(alias=PatientColumns.CD_UF.value)
    NM_MAE: Optional[str] = Field(alias=PatientColumns.NM_MAE.value)
    NM_PAI: Optional[str] = Field(alias=PatientColumns.NM_PAI.value)
    CD_TIPO_ENDERECO: Optional[str] = Field(alias=PatientColumns.CD_TIPO_ENDERECO.value)
    NM_BAIRRO: Optional[str] = Field(alias=PatientColumns.NM_BAIRRO.value)
    NU_CEP: Optional[str] = Field(alias=PatientColumns.NU_CEP.value)

    model_config = {"populate_by_name": True, "extra": "ignore"}

class RawPersonData(BaseModel):
    CD_PESSOA: Union[int, float] = Field(alias=PersonColumns.CD_PESSOA.value)
    CD_PACIENTE: Optional[Union[int, float]] = Field(alias=PersonColumns.CD_PACIENTE.value)
    CD_USUARIO: Optional[str] = Field(alias=PersonColumns.CD_USUARIO.value)
    DT_NASCIMENTO_FUNDACAO: Optional[datetime] = Field(alias=PersonColumns.DT_NASCIMENTO_FUNDACAO.value)
    NM_PESSOA_RAZAO_SOCIAL: Optional[str] = Field(alias=PersonColumns.NM_PESSOA_RAZAO_SOCIAL.value)
    NM_MAE: Optional[str] = Field(alias=PersonColumns.NM_MAE.value)
    FL_SEXO: Optional[str] = Field(alias=PersonColumns.FL_SEXO.value)
    DS_ALTURA: Optional[Union[str, float, int]] = Field(alias=PersonColumns.DS_ALTURA.value)
    DS_PESO: Optional[Union[str, float, int]] = Field(alias=PersonColumns.DS_PESO.value)
    NM_CIDADE_ENDERECO: Optional[str] = Field(alias=PersonColumns.NM_CIDADE_ENDERECO.value)
    CD_CEP_ENDERECO: Optional[str] = Field(alias=PersonColumns.CD_CEP_ENDERECO.value)

    model_config = {"populate_by_name": True, "extra": "ignore"}

class RawBiopsyData(BaseModel):
    CD_PACIENTE: Optional[Union[int, float]] = Field(alias=BiopsyColumns.CD_PACIENTE.value)
    CD_ATENDIMENTO: Union[int, float] = Field(alias=BiopsyColumns.CD_ATENDIMENTO.value)
    DT_ATENDIMENTO: datetime = Field(alias=BiopsyColumns.DT_ATENDIMENTO.value)
    CD_PROCEDIMENTO: str = Field(alias=BiopsyColumns.CD_PROCEDIMENTO.value)
    DT_PROCEDIMENTO_REALIZADO: Optional[datetime] = Field(alias=BiopsyColumns.DT_PROCEDIMENTO_REALIZADO.value)
    NM_EXAME: Optional[str] = Field(alias=BiopsyColumns.NM_EXAME.value)
    DS_LAUDO_MEDICO: Optional[str] = Field(alias=BiopsyColumns.DS_LAUDO_MEDICO.value)
    CD_OCORRENCIA: Optional[Union[int, float]] = Field(alias=BiopsyColumns.CD_OCORRENCIA.value)
    CD_ORDEM: Optional[Union[int, float]] = Field(alias=BiopsyColumns.CD_ORDEM.value)
    CD_MODELO_LAUDO: Optional[str] = Field(alias=BiopsyColumns.CD_MODELO_LAUDO.value)

    model_config = {"populate_by_name": True, "extra": "ignore"}

# ---------------------------------------------------------------- #
# ------------------------ MERGED SOURCES ------------------------ #
# ---------------------------------------------------------------- #

class MergedSourcesData(BaseModel):
    person_id: int | float = Field(alias=MergedSourcesColumns.person_id.value)
    patient_ids: Optional[List[Union[int, float]]] = Field(alias=MergedSourcesColumns.patient_ids.value)
    user_ids: Optional[List[Union[int, float]]] = Field(default=None, alias=MergedSourcesColumns.user_ids.value)
    mammogram_ids: Optional[List[str]] = Field(default=None, alias=MergedSourcesColumns.mammogram_ids.value)
    mammogram_codes: Optional[List[Union[int, float]]] = Field(default=None, alias=MergedSourcesColumns.mammogram_codes.value)
    mammogram_dates: Optional[List[datetime]] = Field(default=None, alias=MergedSourcesColumns.mammogram_dates.value)
    birthdate: Optional[datetime] = Field(default=None, alias=MergedSourcesColumns.birthdate.value)
    sex_code: Optional[Union[int, float, str]] = Field(default=None, alias=MergedSourcesColumns.sex_code.value)
    birads_labels: Optional[List[Union[int, float]]] = Field(default=None, alias=MergedSourcesColumns.birads_labels.value)
    birads_upto3: Optional[List[Union[int, float]]] = Field(default=None, alias=MergedSourcesColumns.birads_upto3.value)
    biopsy_results: Optional[List[Union[int, float]]] = Field(default=None, alias=MergedSourcesColumns.biopsy_results.value)
    biopsy_dates: Optional[List[datetime]] = Field(default=None, alias=MergedSourcesColumns.biopsy_dates.value)
    mammogram_codes_upto3: Optional[List[Union[str]]] = Field(default=None, alias=MergedSourcesColumns.mammogram_codes_upto3.value)
    first_mammogram_date: Optional[datetime] = Field(default=None, alias=MergedSourcesColumns.first_mammogram_date.value)
    last_benign_mammogram_date: Optional[datetime] = Field(default=None, alias=MergedSourcesColumns.last_benign_mammogram_date.value)
    monthly_payment_min: Optional[Union[int, float]] = Field(default=None, alias=MergedSourcesColumns.monthly_payment_min.value)
    monthly_payment_max: Optional[Union[int, float]] = Field(default=None, alias=MergedSourcesColumns.monthly_payment_max.value)
    zipcode: Optional[Union[int, float, str]] = Field(default=None, alias=MergedSourcesColumns.zipcode.value)
    age_at_first_mammogram: Optional[Union[int, float]] = Field(default=None, alias=MergedSourcesColumns.age_at_first_mammogram.value)
    #bmi_linreg: Optional[Union[int, float]] = Field(default=None, alias=MergedSourcesColumns.bmi_linreg.value)
    bmi_randfor: Optional[Union[int, float]] = Field(default=None, alias=MergedSourcesColumns.bmi_randfor.value)
    anamnesis_dates: Optional[List[datetime]] = Field(default=None, alias=MergedSourcesColumns.anamnesis_dates.value)
    first_anamnesis: Optional[datetime] = Field(default=None, alias=MergedSourcesColumns.first_anamnesis.value)
    menarche_age: Optional[Union[int, float, str]] = Field(default=None, alias=MergedSourcesColumns.menarche_age.value)
    menopause_age: Optional[Union[int, float, str]] = Field(default=None, alias=MergedSourcesColumns.menopause_age.value)
    menopause_age_imput: Optional[Union[int, float, str]] = Field(default=None, alias=MergedSourcesColumns.menopause_age_imput.value)
    num_children: Optional[Union[int, float, str]] = Field(default=None, alias=MergedSourcesColumns.num_children.value)
    num_miscarriage: Optional[Union[int, float, str]] = Field(default=None, alias=MergedSourcesColumns.num_miscarriage.value)
    cancer_history_mother: Optional[Union[int, float, bool]] = Field(default=None, alias=MergedSourcesColumns.cancer_history_mother.value)
    cancer_history_sister: Optional[Union[int, float, bool]] = Field(default=None, alias=MergedSourcesColumns.cancer_history_sister.value)
    cancer_history_grandmother: Optional[Union[int, float, bool]] = Field(default=None, alias=MergedSourcesColumns.cancer_history_grandmother.value)
    cancer_history_aunt: Optional[Union[int, float, bool]] = Field(default=None, alias=MergedSourcesColumns.cancer_history_aunt.value)
    mastec_surg_right_flags: Optional[List[Union[int, float, bool]]] = Field(default=None, alias=MergedSourcesColumns.mastec_surg_right_flags.value)
    mastec_surg_left_flags: Optional[List[Union[int, float, bool]]] = Field(default=None, alias=MergedSourcesColumns.mastec_surg_left_flags.value)
    implant_surg_right_flags: Optional[List[Union[int, float, bool]]] = Field(default=None, alias=MergedSourcesColumns.implant_surg_right_flags.value)
    implant_surg_left_flags: Optional[List[Union[int, float, bool]]] = Field(default=None, alias=MergedSourcesColumns.implant_surg_left_flags.value)
    mastec_surg_right_dates: Optional[List[datetime]] = Field(default=None, alias=MergedSourcesColumns.mastec_surg_right_dates.value)
    mastec_surg_left_dates: Optional[List[datetime]] = Field(default=None, alias=MergedSourcesColumns.mastec_surg_left_dates.value)
    implant_surg_right_dates: Optional[List[datetime]] = Field(default=None, alias=MergedSourcesColumns.implant_surg_right_dates.value)
    implant_surg_left_dates: Optional[List[datetime]] = Field(default=None, alias=MergedSourcesColumns.implant_surg_left_dates.value)

    model_config = {"populate_by_name": True, "extra": "ignore"}
    