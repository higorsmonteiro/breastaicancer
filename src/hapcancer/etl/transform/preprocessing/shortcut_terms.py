import re
from tqdm import tqdm
import pandas as pd
from hapcancer.config_manager import ConfigInterface

# List of surgical history terms in Portuguese
SURGICAL_TERMS = [
    # Mastectomy
    "mastectomia", "mastectomizada", "mastectomizado", "mastectomia radical",
    "mastectomia parcial", "mastectomia total", "mastectomia simples", "mastectomia modificada",
    # Quadrantectomy
    "quadrantectomia", "quadrantectomizada", "quadrantectomizado",
    # Lumpectomy / excision
    "tumorectomia", "nodulectomia", "exerese", "exérese", "resseccao", "ressecção", "ressecada",
    # Breast surgery general
    "cirurgia de mama", "cirurgia mamaria", "cirurgia mamária", "cicatriz cirurgica", "cicatriz cirúrgica",
    "pos-operatorio", "pós-operatório", "pos-cirurgico", "pós-cirúrgico", "pos operatorio",
    "pós operatório",
    # Implants / reconstruction
    "protese", "prótese", "implante", "implante mamario", "implante mamário", "protese mamaria",
    "prótese mamária", "reconstrucao mamaria", "reconstrução mamária", "reconstrucao de mama", "reconstrução de mama",
    # Scars
    "area de resseccao", "área de ressecção", "leito cirurgico", "leito cirúrgico",
]

# Regex pattern — case insensitive, matches any of the terms
# Using word boundaries where appropriate
pattern = re.compile(
    r'(' + '|'.join(re.escape(term) for term in SURGICAL_TERMS) + r')',
    re.IGNORECASE
)

def contains_surgical_term(text: str) -> bool:
    """Returns True if the report contains any surgical history term."""
    if not isinstance(text, str):
        return False
    return bool(pattern.search(text))

def find_surgical_terms(text: str) -> list:
    """Returns all surgical terms found in the report."""
    if not isinstance(text, str):
        return []
    return pattern.findall(text)

class FindShortcutTerms(ConfigInterface):
    def __init__(self, config_dir: str, config_defaults: dict):
        super().__init__(config_dir, config_defaults)
        self.output_path = self.transformed_data_path.joinpath(self.files_and_folders_cfg['transform']['transformed_files']['shortcut_terms']+'.parquet')

    def _load_reports(self):
        columns = ["key", "text"]
        self.reports = pd.concat([
            batch[columns] for batch in tqdm(self._iter_raw_mammograms_data())
        ], ignore_index=True)

    def find_terms(self):
        self._load_reports()
        self.reports['has_surgical_term'] = self.reports['text'].apply(contains_surgical_term)
        self.reports['found_terms'] = self.reports['text'].apply(find_surgical_terms)
        self.reports[['key', 'has_surgical_term', 'found_terms']].to_parquet(self.output_path)