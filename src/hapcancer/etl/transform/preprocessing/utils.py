import re
import yaml
import hashlib
import bisect
import pandas as pd
import numpy as np
import unicodedata
from tqdm import tqdm
from collections import Counter
from datetime import date, datetime
from rapidfuzz import fuzz, distance 
from typing import List, Optional, Tuple, Dict

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("latin")).hexdigest()

# ===========================================================

def extract_children_count(var):
    text = var["DS_INDICACAO_QUEIXA"]
    filho_signal = var["contains_filho"]
    if not filho_signal:
        return np.nan

    # -- check for specific patterns using regex
    match = re.search(r'FILHOS?\s*(-?)(\d+|NAO)', text, re.IGNORECASE)
    
    if match:
        # -- extract the captured group (either a number or "NAO")
        count = match.group(2)
        # -- convert "NAO" to 0 children
        if count.upper() == "NAO":
            return 0
        else:
            # -- return the integer value of children count
            return int(count)
    return np.nan  # -- if no match is found, return None

def load_config_file(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


# --------------------------------------------------------------------------------



# ------------------------
# Name normalization
# ------------------------
_PT_SMALL = {"da","de","do","das","dos","e","du","d’","d"}

def _strip_accents(s: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def _normalize_name(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    s = str(s).strip().lower()
    s = _strip_accents(s)
    s = re.sub(r"[^a-z\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    tokens = [t for t in s.split() if t and t not in _PT_SMALL]
    return " ".join(tokens)

def _initials(name):
    toks = name.split()
    if not toks:
        return ("","")
    return toks[0][0], toks[-1][0]

def _rf_norm(x: float) -> float:
    # RapidFuzz's fuzz.* return 0..100; distance.*.normalized_similarity may return 0..1.
    return x/100.0 if x > 1.0 else x

def name_similarity_rapidfuzz(a, b) -> float:
    na = _normalize_name(a)
    nb = _normalize_name(b)
    if not na and not nb:
        return np.nan
    if not na or not nb:
        return 0.0

    r_plain   = _rf_norm(fuzz.ratio(na, nb))
    r_tok_set = _rf_norm(fuzz.token_set_ratio(na, nb))
    r_tok_sort= _rf_norm(fuzz.token_sort_ratio(na, nb))
    r_partial = _rf_norm(fuzz.partial_ratio(na, nb))
    r_jw      = _rf_norm(distance.JaroWinkler.normalized_similarity(na, nb))

    # robust aggregate: emphasize token_set, token_sort, and JW; keep others as tie-breakers
    core = max(r_tok_set, r_tok_sort, r_jw, r_plain, r_partial)

    # tiny bonus if first & last initials align (guards against reordered/middle names)
    bonus = 0.05 if _initials(na) == _initials(nb) and _initials(na) != ("","") else 0.0
    return float(min(core + bonus, 1.0))

# ------------------------
# DOB similarity (datetimes already)
# ------------------------
def _to_date(x):
    if pd.isna(x):
        return None
    if isinstance(x, pd.Timestamp):
        return x.date()
    if isinstance(x, datetime):
        return x.date()
    if isinstance(x, date):
        return x
    # last resort: try pandas (but you said it's already datetime)
    try:
        return pd.to_datetime(x, errors="coerce").date()
    except Exception:
        return None

def dob_similarity_dt(d1, d2) -> float:
    D1 = _to_date(d1)
    D2 = _to_date(d2)
    if D1 is None and D2 is None:
        return np.nan
    if D1 is None or D2 is None:
        return 0.0

    if D1 == D2:
        return 1.0

    delta = abs((pd.Timestamp(D1) - pd.Timestamp(D2)).days)
    if delta <= 1:
        return 0.95

    if D1.year == D2.year:
        if D1.month == D2.month:
            return 0.8
        return 0.6

    if abs(D1.year - D2.year) == 1 and (D1.month, D1.day) == (D2.month, D2.day):
        return 0.85

    if abs(D1.year - D2.year) == 1 and D1.month == D2.month:
        return 0.5

    return 0.0

# ------------------------
# Main entry
# ------------------------
def add_person_similarity(
    df: pd.DataFrame,
    name_col_a: str = "name_a",
    dob_col_a: str = "birthdate_a",
    name_col_b: str = "name_b",
    dob_col_b: str = "birthdate_b",
    weight_name: float = 0.65,
    weight_dob: float = 0.35,
    adaptive_weights: bool = True,
    out_col: str = "similarity"
) -> pd.DataFrame:
    """
        ...
    """

    def row_score(row):
        ns = name_similarity_rapidfuzz(row.get(name_col_a), row.get(name_col_b))
        ds = dob_similarity_dt(row.get(dob_col_a), row.get(dob_col_b))

        ns_eff = 0.0 if np.isnan(ns) else ns
        ds_eff = 0.0 if np.isnan(ds) else ds

        wn, wd = weight_name, weight_dob
        if adaptive_weights:
            if ds_eff >= 0.95:
                wn, wd = 0.45, 0.55
            elif ds_eff == 0.0:
                wn, wd = 0.85, 0.15

        score = wn * ns_eff + wd * ds_eff
        if ns_eff >= 0.8 and ds_eff >= 0.8:
            score = min(1.0, score + 0.05)
        return float(score)

    out = df.copy()
    out[out_col] = out.apply(row_score, axis=1)
    return out



    
# ===================== biopsy =====================
def is_breast_biopsy(text: str) -> bool:
    BREAST_REGEX = re.compile(
        r"\b(mama(s)?|mam[áa]ria(s)?|mam[áa]rio(s)?)\b",
        flags=re.IGNORECASE
    )
    return bool(BREAST_REGEX.search(text))

BIOPSY_STRONG = ["core needle biopsy","core biopsy","tru cut","fine needle aspiration",
                 "biopsia core","biopsia incisional","biopsia excisional","puncao por agulha fina"]
BIOPSY_BASE = ["biopsy","biopsia","puncao aspirativa","paf","histopatologico","imuno histoquimica","parafina","microtomia","fragmento tecidual","fragmentos teciduais"]

BIOPSY_STRONG += [
    "guiada por ultrassonografia", "guiada por usg", "sob anestesia local",
    "cuidados pos procedimento", "procedimento realizado",
    "fixados em formaldeido", "fixados em formalina",
    "clip metalico", "marcador clip", "tumor", "cassete",
    "agulha tru cut", "tru cut", "agulha core", "passagens de agulha",
    "peca cirurgica", "produto de setorectomia", "quadrantectomia"
    "macroscopia", "microscopia optica",
    "cortes histologicos", "laminas", "blocagem", "bloc o", "bloco de parafina",
    "material recebido", "frasco",
    "diagnostico:", "tipo histologico", "grau histologico", "nottingham",
    "estadiamento patologico", "ptnm",
    "linfonodo sentinela", "linfonodos regionais",
    "margens cirurgicas", "margens de resseccao",
]

BIOPSY_BASE += [
    "foram retirados", "fragmentos fixados", "fragmento tecidual", "fragmentos teciduais",
    "carcinoma ductal invasivo", "carcinoma lobular in situ", "carcinoma ductal in situ",
    "tamanho do tumor", "focalidade do tumor",
    "invasao vascular", "invasao linfatica",
    "microcalcificacoes",
    "negativo para malignidade", "nao ha sinais de malignidade",
    "cnb", "puncao aspirativa", "paf", "puncao por agulha fina", "agulha fina"
]


MAMMO_STRONG = ["tomosynthesis","digital breast tomosynthesis","dbt","tomossintese","bi-rads","birads",
                "screening mammogram","diagnostic mammogram","mamografia digital","mediolateral oblique","mlo","craniocaudal"]

MAMMO_STRONG += [
    "mamografia", "mamografia digital",
    "birads", "bi rads", "categoria birads",  # qualquer forma de birads
    "tomossintese", "digital breast tomosynthesis", "dbt",
    "projecao craniocaudal", "craniocaudal",
    "projecao mediolateral obliqua", "mlo",
    "compressao", "dose glandular", "incidencia", "radiografia", "radiografico",
    "rastreamento", "screening", "diagnostica", "diagnostico mamografico",
    "laudo mamografico"
]


MAMMO_BASE = ["mammogram","mammography","mamografia","calcificacoes","assimetria","densidade mamaria"]

# -- Key words for exams that are recommendations and not results.
RECOMMEND = ["recommend","recommended","suggest","sugerida","sugerido","indicar","indicado",
             "agendar","programar","to be performed","a realizar"]

RECOMMEND += [
    "recomend", "suger", "orienta", "encaminh", "solicit", "pedido de", "requisit",
    "indica", "indicada", "indicado", "indicacao",
    "a realizar", "a ser realizada", "realizar se a", "sera realizada", "serao realizados",
    "program", "agendar", "agendada", "agendado", "agendamento", "pre agendada",
    "marcada", "marcado", "marcacao",
    "agendada para", "marcada para",
    "retornar para biopsia", "retorno para biopsia",
    "comparecer para", "preparo para", "jejum de",
    "autorizacao pendente", "aguardando autorizacao", "guia de procedimento",
    "laudo inconclusivo", "inconclusivo", "a esclarecer",
]

CANCER_BREAST_TERMS = [
    "maligno", "espiculado", "carcinoma",
    "neoplasia", "pleomórfico", "lesão sólida",
    "hipoecoica", "calcificações"
]

def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", s.lower()).strip()

def compile_patterns(terms: List[str], weight=1):
    pats = []
    for t in terms:
        t_norm = norm(t)
        # -- word boundary; allow hyphens/spaces variants
        t_esc = re.sub(r"\s+", r"[\\s-]+", re.escape(t_norm))
        pats.append((re.compile(rf"\b{t_esc}\b"), weight))
    return pats

def token_spans(text: str) -> List[Tuple[str,int,int]]:
    # get tokens and positions
    toks, pos = [], 0
    for m in re.finditer(r"\w+|\S", text):
        toks.append((m.group(), m.start(), m.end()))
    return toks

def near_negation(text_norm: str, hit_span: Tuple[int, int], toks: Tuple[str, int, int], REC_PATTERNS, window_tokens=8):
    # find token index only using precomputed toks
    starts = [s for _,s,_ in toks]   # precomputed list of start positions
    hit_idx = bisect.bisect_left(starts, hit_span[0])
    
    #hit_idx = next((i for i,(tok,s,e) in enumerate(toks) if s >= hit_span[0]), len(toks)-1)
    s = max(0, hit_idx - window_tokens)
    e = min(len(toks), hit_idx + window_tokens + 1)
    window_text = text_norm[toks[s][1] : toks[e-1][2]]
    return any(p.search(window_text) for p,_ in REC_PATTERNS)

def class_scores(doc: str, BIO_PATTERNS, MAM_PATTERNS, REC_PATTERNS) -> Dict[str, int]:
    t = norm(doc)
    toks = token_spans(t)       # computed ONCE

    b_score, m_score = 0, 0
    for p,w in BIO_PATTERNS:
        for m in p.finditer(t):
            if near_negation(t, (m.start(),m.end()), toks, REC_PATTERNS):
                continue
            b_score += w

    for p,w in MAM_PATTERNS:
        for _ in p.finditer(t):
            m_score += w

    return {"biopsy": b_score, "mammogram": m_score}

def classify(doc: str, BIO_PATTERNS, MAM_PATTERNS, REC_PATTERNS, margin=1, min_score=2) -> Tuple[str,Dict[str,int]]:
    s = class_scores(doc, BIO_PATTERNS, MAM_PATTERNS, REC_PATTERNS)
    if s["biopsy"] >= min_score and s["biopsy"] >= s["mammogram"] + margin:
        return "biopsy", s
    if s["mammogram"] >= min_score and s["mammogram"] >= s["biopsy"] + margin:
        return "mammogram", s
    return "unknown", s

