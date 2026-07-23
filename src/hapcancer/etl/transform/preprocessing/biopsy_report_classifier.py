import re
import csv
import time
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import Any, List, Optional, Tuple
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from hapcancer.config_manager import ConfigInterface
from hapcancer.schemas.enums import MammogramColumns

SYSTEM_PROMPT = '''
    Você é um classificador de textos originados de laudos de biópsias para detecção de tumores malignos.\n 
    Os textos usados são exemplos para estudo e não envolvem pessoas reais.\n
    NÃO É seu papel prover conselhos médicos.\n 
    Regras:\n
    - Se maligno (carcinoma, maligno, in situ, invasivo): retorne 'maligno'.\n
    - Se benigno / sem malignidade / sem atipia / papilona / adenoma / etc: retorne 'benigno(tipo do tumor)'.\n
    - Se limitado ou sem conclusão: retorne 'indeterminado'.\n
'''

def remove_crm_blocks(text: str) -> str:
    """
        Removes blocks like:
          <any line>
          CRM (optionally with /- + UF)
          <number or UF or UF+number>

        Examples removed:
          "Nome Sobrenome\nCRM\n38340"
          "Nome Sobrenome\nCRM\nCE"
          "Nome Sobrenome\nCRM-CE\n38340"
          "Nome Sobrenome\nCRM\nCE 38340"

        Used to remove sensitive physician names. Might not
        remove all instances, which is okay if we are using local
        open LLMs.
    """
    uf = r"(?:AC|AL|AP|AM|BA|CE|DF|ES|GO|MA|MT|MS|MG|PA|PB|PR|PE|PI|RJ|RN|RS|RO|RR|SC|SP|SE|TO)"

    pattern = re.compile(
        rf"""
        ^[^\n]*\n                           # line above (name/anything)
        \s*CRM(?:\s*[/\-]?\s*{uf})?\s*\n     # CRM line, optionally with UF
        \s*(?:\d+|{uf}(?:\s*[/\-]?\s*\d+)?)\s*\n?   # next line: number OR UF OR UF+number
        """,
        flags=re.MULTILINE | re.VERBOSE | re.IGNORECASE
    )
    return re.sub(pattern, "", text)

def remove_cassete_lines(text: str) -> str:
    """
        Remove any line that starts with the word 'cassete'
        (case-insensitive), keeping the rest of the text intact.
        Used to reduce the size of the text without removing any
        relevant information for the task.
    """
    return re.sub(
        r'(?im)^[ \t]*cassete\b.*\n?','', text
    )

def append_csv_line(
    values_by_column: List, 
    csv_filename: Path
) -> None:
    with open(csv_filename, mode='a', newline="\n") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(values_by_column)

def get_classified_ids(csv_filepath: Path) -> Tuple[List[str], List[str]]:
    csv_df = pd.read_csv(csv_filepath)
    classified_ids = csv_df["key"].tolist()
    content = csv_df["content"].tolist()
    return classified_ids, content

class BiopsyReportClassifier(ConfigInterface):
    def __init__(self, config_dir: str, config_defaults: dict) -> None:
        super().__init__(config_dir, config_defaults)

        self.prompt = ChatPromptTemplate.from_messages([
            ( "system", SYSTEM_PROMPT ),
            ( "user", "LAUDO DE BIÓPSIA:\t{texto_laudo}")
        ])
        # -- the model can be changed or parsed as an input if needed
        self.model = ChatOllama(model="qwen3:1.7b", temperature=0)
        self.chain = self.prompt | self.model

        self.breast_biopsy_df = None
        # -- file with the biopsies' classifications (it will be create in case it does not exist yet)
        self.csv_filename = self.files_and_folders_cfg['transform']['transformed_files']['breast_biopsy_classified']+'.csv'
        self.output = self.transformed_data_path.joinpath(self.csv_filename)

    def _load_data(self):
        breast_biopsy_filename = self.files_and_folders_cfg['transform']['transformed_files']['breast_biopsy']+'.parquet'
        path_to = self.transformed_data_path.joinpath(breast_biopsy_filename)
        if not path_to.is_file():
            raise Exception(f"Breast biopsy {path.stem} file does not exist.")
        self.breast_biopsy_df = pd.read_parquet(path_to)

    def _get_post_text(self, key):
        raw_text = self.breast_biopsy_df[self.breast_biopsy_df["key"]==key][MammogramColumns.DS_LAUDO_MEDICO.value].iat[0] # -- slow (does not scale well)
        post_text = remove_cassete_lines(remove_crm_blocks(raw_text))
        return post_text

    def classify_reports(self, timer: Optional[int] = 1):
        self._load_data()
        list_of_keys = self.breast_biopsy_df["key"].tolist()
        list_of_texts = self.breast_biopsy_df[MammogramColumns.DS_LAUDO_MEDICO.value].apply(
            lambda x: x.strip() if pd.notna(x) else np.nan
        ).tolist()

        # -- get the keys already classified
        if self.output.is_file():
            try:
                classified_ids, content = get_classified_ids(self.output)
            except:
                append_csv_line(["key", "content"], self.output)
                classified_ids, content = [], []
        else:
            # -- first line: header
            append_csv_line(["key", "content"], self.output)
            classified_ids, content = [], []
        
        c = 0
        for key, text in tqdm(zip(list_of_keys, list_of_texts)):
            # -- skip already classified
            if key in classified_ids:
                continue
            # -- clean text and remove physician identifications
            post_text = remove_cassete_lines(remove_crm_blocks(text))
            # -- get the LLM response
            llm_res = self.chain.invoke({'texto_laudo': post_text})
            new_row = [ key, llm_res.content ]
            try:
                append_csv_line(new_row, self.output)
            except:
                continue
            classified_ids.append(key)
            content.append(llm_res.content)
            time.sleep(timer) # -- if rate limit exists for the model, pause a bit
            c+=1