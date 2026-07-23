from enum import Enum 

# --------------------------------------------------------------- #
# ------------------------ CONFIGURATION ------------------------ #
# --------------------------------------------------------------- #

class ConfigFolderNames(str, Enum):
    etl_foldername = "etl"
    model_foldername = "model"
    birads_clf_foldername = "birads_classifier"
    embeddings_foldername = "embeddings"
    split_foldername = "splits"
    bmi_model_foldername = "bmi_model"
    training_experiments_foldername = "training_experiments"
    tuning_foldername = "tuning"
    followup_foldername = "followup"
    

# ---------------------------------------------------------- #
# ------------------------ RAW DATA ------------------------ #
# ---------------------------------------------------------- #

class AnamnesisColumns(str, Enum):
    CD_ATENDIMENTO = "CD_ATENDIMENTO"
    CD_PACIENTE = "CD_PACIENTE"
    CD_PESSOA = "CD_PESSOA"
    CD_USUARIO = "CD_USUARIO"
    DS_INDICACAO_QUEIXA = "DS_INDICACAO_QUEIXA"
    DT_ATENDIMENTO = "DT_ATENDIMENTO"
    DS_MENARCA = "DS_MENARCA"
    DS_MENOPAUSA = "DS_MENOPAUSA"
    NU_GESTACAO = "NU_GESTACAO"
    NU_GESTACAO_ABORTO = "NU_GESTACAO_ABORTO"
    FL_ALEITAMENTO = "FL_ALEITAMENTO"
    FL_CA_MAMA_MAE = "FL_CA_MAMA_MAE"
    FL_CA_MAMA_IRMA = "FL_CA_MAMA_IRMA"
    FL_CA_MAMA_AVO = "FL_CA_MAMA_AVO"
    FL_CA_MAMA_TIA = "FL_CA_MAMA_TIA"
    DT_BIOPSIA_ME = "DT_BIOPSIA_ME"
    DT_BIOPSIA_MD = "DT_BIOPSIA_MD"
    DT_QUADRANTECTOMIA_ME = "DT_QUADRANTECTOMIA_ME"
    DT_QUADRANTECTOMIA_MD = "DT_QUADRANTECTOMIA_MD"
    DT_MASTECTOMIA_ME = "DT_MASTECTOMIA_ME"
    DT_MASTECTOMIA_MD = "DT_MASTECTOMIA_MD"
    FL_PLASTICA_ME = "FL_PLASTICA_ME"
    FL_PLASTICA_MD = "FL_PLASTICA_MD"
    DT_PLASTICA_ME = "DT_PLASTICA_ME"
    DT_PLASTICA_MD = "DT_PLASTICA_MD"
    FL_BIOPSIA_ME = "FL_BIOPSIA_ME"
    FL_BIOPSIA_MD = "FL_BIOPSIA_MD"
    FL_QUADRANTECTOMIA_ME = "FL_QUADRANTECTOMIA_ME"
    FL_QUADRANTECTOMIA_MD = "FL_QUADRANTECTOMIA_MD"
    FL_MASTECTOMIA_ME = "FL_MASTECTOMIA_ME"
    FL_MASTECTOMIA_MD = "FL_MASTECTOMIA_MD"

class MammogramColumns(str, Enum):
    NM_EXAME = "NM_EXAME"
    CD_ATENDIMENTO = "CD_ATENDIMENTO"
    DT_ATENDIMENTO = "DT_ATENDIMENTO"
    DS_LAUDO_MEDICO = "DS_LAUDO_MEDICO"
    CD_PACIENTE = "CD_PACIENTE"

class BiopsyColumns(str, Enum):
    NM_EXAME = "NM_EXAME"
    CD_ATENDIMENTO = "CD_ATENDIMENTO"
    DT_ATENDIMENTO = "DT_ATENDIMENTO"
    CD_PROCEDIMENTO = "CD_PROCEDIMENTO"
    CD_OCORRENCIA = "CD_OCORRENCIA"
    CD_ORDEM = "CD_ORDEM"
    DT_PROCEDIMENTO_REALIZADO = "DT_PROCEDIMENTO_REALIZADO"
    CD_MODELO_LAUDO = "CD_MODELO_LAUDO"
    DS_LAUDO_MEDICO = "DS_LAUDO_MEDICO"
    CD_PACIENTE = "CD_PACIENTE"

class UserColumns(str, Enum):
    CD_PACIENTE = "CD_PACIENTE"
    CD_USUARIO = "CD_USUARIO"
    CD_PESSOA = "CD_PESSOA"
    NU_USUARIO = "NU_USUARIO"
    NU_TITULAR = "NU_TITULAR"
    CD_PLANO = "CD_PLANO"
    CD_CANCELAMENTO = "CD_CANCELAMENTO"
    CD_USUARIO_EMPRESA_PARCEIRA = "CD_USUARIO_EMPRESA_PARCEIRA"
    CD_TIPO_DEPENDENTE_USUARIO = "CD_TIPO_DEPENDENTE_USUARIO"
    NU_ORDEM_USUARIO = "NU_ORDEM_USUARIO"
    DT_REFERENCIA_CARENCIA = "DT_REFERENCIA_CARENCIA"
    DT_CADASTRAMENTO = "DT_CADASTRAMENTO"
    DT_CANCELAMENTO = "DT_CANCELAMENTO"
    DT_MENSALIDADE_A = "DT_MENSALIDADE_A"
    VL_MENSALIDADE_A = "VL_MENSALIDADE_A"
    VL_MENSALIDADE = "VL_MENSALIDADE"
    VL_TAXA_ADESAO = "VL_TAXA_ADESAO"

class PatientColumns(str, Enum):
    CD_PACIENTE = "CD_PACIENTE"
    CD_USUARIO = "CD_USUARIO"
    CD_PESSOA = "CD_PESSOA"
    NU_USUARIO = "NU_USUARIO"
    NU_TITULAR = "NU_TITULAR"
    NM_PACIENTE = "NM_PACIENTE"
    DT_NASCIMENTO = "DT_NASCIMENTO"
    CD_SEXO = "CD_SEXO"
    NM_MUNICIPIO = "NM_MUNICIPIO"
    CD_UF = "CD_UF"
    CD_ESTADO_CIVIL = "CD_ESTADO_CIVIL"
    CD_PROFISSIONAL = "CD_PROFISSIONAL"
    NM_MAE = "NM_MAE"
    NM_PAI = "NM_PAI"
    CD_TIPO_ENDERECO = "CD_TIPO_ENDERECO"
    NM_BAIRRO = "NM_BAIRRO"
    NU_CEP = "NU_CEP"

class PersonColumns(str, Enum):
    CD_PESSOA = "CD_PESSOA"
    CD_PACIENTE = "CD_PACIENTE"
    CD_USUARIO = "CD_USUARIO"
    DT_NASCIMENTO_FUNDACAO = "DT_NASCIMENTO_FUNDACAO"
    NM_PESSOA_RAZAO_SOCIAL = "NM_PESSOA_RAZAO_SOCIAL"
    NM_MAE = "NM_MAE"
    FL_SEXO = "FL_SEXO"
    DS_ALTURA = "DS_ALTURA"
    DS_PESO = "DS_PESO"
    NM_CIDADE_ENDERECO = "NM_CIDADE_ENDERECO"
    CD_CEP_ENDERECO = "CD_CEP_ENDERECO"


# ---------------------------------------------------------------- #
# ------------------------ MERGED SOURCES ------------------------ #
# ---------------------------------------------------------------- #

class MergedSourcesColumns(str, Enum):
    person_id = "CD_PESSOA"
    patient_ids = "CD_PACIENTE"
    user_ids = "CD_USUARIO"
    mammogram_ids = "key"
    mammogram_codes = "CD_ATENDIMENTO"
    mammogram_dates = "DT_ATENDIMENTO"
    birthdate = "DT_NASCIMENTO_FUNDACAO"
    sex_code = "FL_SEXO_ML" 
    birads_labels = "birads_labels"
    birads_upto3 = "birads_upto3"
    biopsy_results = "biopsy_results"
    biopsy_dates = "biopsy_dates"
    mammogram_codes_upto3 = "mammogram_codes_upto3"
    first_mammogram_date = "first_mammogram_date"
    last_benign_mammogram_date = "last_benign_mammogram_date"
    monthly_payment_min = "VL_MENSALIDADE_MIN"
    monthly_payment_max = "VL_MENSALIDADE_MAX"
    zipcode = "zipcode_cat"
    age_at_first_mammogram = "age_at_first_mammogram"
    bmi_linreg = "BMI_PREDICT_LINREG"
    bmi_randfor = "BMI_PREDICT_RANDFOR"
    anamnesis_dates = "DT_ATENDIMENTO_ANAMNESE"
    first_anamnesis = "DT_PRIMEIRA_ANAMNESE"
    menarche_age = "DS_MENARCA_FMT"
    menopause_age = "DS_MENOPAUSA_FMT"
    menopause_age_imput = "DS_MENOPAUSA_FMT_IMPUTATION"
    num_children = "NU_GESTACAO_FMT"
    num_miscarriage = "NU_GESTACAO_ABORTO_FMT"
    cancer_history_mother = 'FL_CA_MAMA_MAE_FMT'
    cancer_history_sister = 'FL_CA_MAMA_IRMA_FMT'
    cancer_history_grandmother = 'FL_CA_MAMA_AVO_FMT'
    cancer_history_aunt = 'FL_CA_MAMA_TIA_FMT'
    mastec_surg_right_flags = 'FL_MASTECTOMIA_MD_FMT'
    mastec_surg_left_flags = 'FL_MASTECTOMIA_ME_FMT'
    implant_surg_right_flags = 'FL_PLASTICA_MD_FMT'
    implant_surg_left_flags = 'FL_PLASTICA_ME_FMT'
    mastec_surg_right_dates = 'DT_MASTECTOMIA_MD_FMT'
    mastec_surg_left_dates = 'DT_MASTECTOMIA_ME_FMT'
    implant_surg_right_dates = 'DT_PLASTICA_MD_FMT'
    implant_surg_left_dates = 'DT_PLASTICA_ME_FMT'


# ------------------------------------------------------------------------- #
# ------------------------ MAMMOGRAM SEQUENCE DATA ------------------------ #
# ------------------------------------------------------------------------- #