import os
import json
import time
import sqlite3
import ollama
import joblib
import numpy as np
import pandas as pd
import datetime as dt
from tqdm import tqdm
from pathlib import Path
from typing import Iterable, List, Optional, Dict, Any, Tuple
from sklearn.decomposition import TruncatedSVD

from hapcancer.schemas.enums import MammogramColumns
from hapcancer.etl.transform.embeddings import utils
from hapcancer.etl.utils import batching_parquet_file
from hapcancer.config_manager import ConfigInterface

# ----------------------------
# Transformer
# ----------------------------
class MammogramsEmbeddingTransform:
    """
        - Reads rows from input files
        - Concats specified text columns
        - Chunks long texts
        - Embeds with Ollama (nomic-embed-text by default)
        - Pools chunk vectors to a single row vector
        - Caches by (model, SHA1(text))
        - Writes a Parquet with columns: [id, embedding, embedding_dim, raw_text_hash, model, created_at]
    """
    def __init__(self, config: dict) -> None:
        self.config = config
        self.verbose = self.config['verbose']
        self.extract_path = Path(self.config['extract']['path']).joinpath(self.config['extract']['id'])
        self.raw_mammograms_folder_path = self.extract_path.joinpath(self.config['extract']['folders']['mammogram_exams'])
        self.transform_id = self.config['transform']['transform_id']
        self.transform_path = Path(self.config['transform']['transform_path'])
        self.ollama_embedding_model = self.config['transform']['ollama_embedding_model']
        self.deberta_flag = self.config['transform']['use_deberta_for_embedding']
        self.embedding_id = self.config['transform']['embedding_id']
        
        if self.deberta_flag:
            self.embedding_id = 'debertav3_base'

        for pth in (
            self.transform_path.joinpath(self.transform_id),
            self.transform_path.joinpath(self.transform_id).joinpath(self.config['transform']['folders']['embedding']),
            self.transform_path.joinpath(self.transform_id).joinpath(self.config['transform']['folders']['cache']),
            Path(self.config['transform']['logging_path']).joinpath(self.transform_id)
        ):
            pth.mkdir(parents=True, exist_ok=True)

        self.cache_path = self.transform_path.joinpath(self.transform_id).joinpath(self.config['transform']['folders']['cache'])
        self.cache = EmbedCache(self.cache_path.joinpath("embed_cache.sqlite3"))
        
        if self.verbose:
            SETUP_MSG = f'''
                Task initiated: Transform {self.transform_id}.
                
                Model name ID: {self.embedding_id}.
            '''
            print(SETUP_MSG)

    # ---------- iteration for existing mammogram files ----------
    def _iter_input_files(self) -> Iterable[Path]:
        yield from sorted(self.raw_mammograms_folder_path.glob("*.parquet"))

    def _get_mammogram_text(self, row: pd.Series) -> str:
        val = row[self.config['transform']['fields']['mammogram_text']]  # singular, not list
        if pd.isna(val):
            return ""
        return str(val).strip()

    def _embed_text_with_retries(self, text: str) -> List[float]:
        max_retries = 4
        for attempt in range(max_retries):
            try:
                #vec = utils.embed_text(text=text, model=self.config['transform']['ollama_embedding_model'])
                # -- just for now while nomic v2 is not available in ollama
                vec = utils.embed_text_nomic_v2(text=text, model=self.embedding_model, truncation=512)
                #vec = vec.astype(np.float16)
                return vec
            except Exception as e:
                wait = 1
                time.sleep(wait)
                if attempt == max_retries - 1:
                    raise e
        return []

    def _embed_document(self, text: str) -> Tuple[List[float], Dict[str, Any]]:
        '''
            Chunk -> embed each chunk -> pool
        '''
        max_words = self.config['transform']['chunk_max_words_embedding']
        stride_words = self.config['transform']['chunk_overlap_stride_embedding']
        pool_strategy = self.config['transform']['chunk_pool_strategy']
        chunks = utils.word_chunks(text, max_words=max_words, stride_words=stride_words)
        if not chunks:
            return [], {"num_chunks": 0}

        vectors = [self._embed_text_with_retries(ch) for ch in chunks]
        if pool_strategy == "mean":
            pooled = utils.mean_pool(vectors) # -- maybe not the best approach
        elif pool_strategy == "concat":
            pooled = [f for vec in vectors for f in vec]
        else:
            raise Exception("No valid pool strategy was specified.")

        meta = {"num_chunks": len(chunks), "chunk_sizes": [len(c.split()) for c in chunks]}
        return pooled, meta

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        """
            ...
        """
        if self.ollama_embedding_model.lower()!='none':
            vecs = ollama.embed(input=texts, model=self.ollama_embedding_model)
            vecs = np.asarray(vecs['embeddings'])
        elif self.deberta_flag:
            vecs = utils.embed_text_deberta_v3(input=texts, max_length = 512, model_name = "microsoft/deberta-v3-base")
            vecs = np.asarray(vecs)
        else:
            vecs = self.embedding_model.encode(
                texts,
                batch_size=128,          # e.g., 128 on GPU, 32–64 on CPU
                convert_to_numpy=True,
                show_progress_bar=False,
                device='cuda'
            )
        return vecs.astype(np.float16)

    # ---------- Public API ----------
    def run(self) -> None:
        """
            Embeds all input files.
            - One parquet per input file.
            - Uses cache to avoid recomputing if interrupted.
            - Skips writing if parquet already exists.
            Returns the embeddings directory.
        """
        embedding_folder = self.config['transform']['folders']['embedding']
        embedding_path = self.transform_path.joinpath(self.transform_id).joinpath(embedding_folder)
        self.ollama_embedding_model = self.config["transform"]["ollama_embedding_model"]
        mammogram_id_col = self.config["transform"]["fields"]["mammogram_id"]
        for filename in self._iter_input_files():
            # -- check whether we have already generated embeddings for this file.
            current_emb_file = embedding_path.joinpath(f"{filename.stem}_embeddings.parquet")
            if current_emb_file.exists():
                print(f"[skip] {filename.name} -> {current_emb_file.name} already exists.")
                continue

            print(f"[process] {filename.name} ...")
            df = pd.read_parquet(filename)

            needed_cols = [ colname_tp[1] for colname_tp in list(self.config['transform']['fields'].items()) if 'mammogram' in colname_tp[0]]
            missing = [c for c in needed_cols if c not in df.columns]
            if missing:
                raise ValueError(f"{filename} is missing required columns: {missing}")
            df = df[needed_cols].copy()

            records = []
            cache_records = []
            for _, row in tqdm(df.iterrows()):
                text = self._get_mammogram_text(row)
                mammogram_id = int(row[mammogram_id_col])
                text_hash = utils.sha1(text) if text else ''

                # -- store cache info
                cache_records.append(
                    (mammogram_id, text_hash)
                )

                if not text:
                    records.append({
                        mammogram_id_col: row[mammogram_id_col],
                        "embedding": [],
                        "embedding_dim": 0,
                        "raw_text_hash": text_hash,
                        "model": self.embedding_id,
                        "created_at": utils.now_iso(),
                        "num_chunks": 0,
                    })
                    continue

                vec, meta = self._embed_with_cache(text, mammogram_id)
                records.append({
                    mammogram_id_col: row[mammogram_id_col],
                    "embedding": vec,
                    "embedding_dim": len(vec),
                    "raw_text_hash": text_hash,
                    "model": self.embedding_id,
                    "created_at": utils.now_iso(),
                    "num_chunks": meta.get("num_chunks"),
                })

            out_df = pd.DataFrame.from_records(records)
            out_df.to_parquet(current_emb_file, index=False)
            print(f"[done] wrote {len(out_df)} rows -> {current_emb_file.name}")

            # -- caching only after the embedding file is created
            [ self.cache.put(self.embedding_id, elem[0], elem[1]) for elem in cache_records ]

    def run_batch(self) -> None:
        """
            ...
        """
        embedding_folder = self.config['transform']['folders']['embedding']
        embedding_path = self.transform_path.joinpath(self.transform_id).joinpath(embedding_folder)

        mammogram_id_col = self.config["transform"]["fields"]["mammogram_id"]
        text_col         = self.config["transform"]["fields"]["mammogram_text"]

        if not [ fname for fname in self._iter_input_files()]:
            raise Exception("raw mammograms' files not found.")
        
        for filename in self._iter_input_files():
            current_emb_file = embedding_path.joinpath(f"{filename.stem}_{self.embedding_id}_embeddings.parquet")
            if current_emb_file.exists():
                print(f"[skip] {filename.name} -> {current_emb_file.name} already exists.")
                continue

            print(f"[process] {filename.name} ...")
            df = pd.read_parquet(filename)

            needed_cols = [mammogram_id_col, text_col]
            missing = [c for c in needed_cols if c not in df.columns]
            if missing:
                raise ValueError(f"{filename} is missing required columns: {missing}")
            df = df[needed_cols].copy()

            records = []
            batch_ids: list = []
            batch_texts: list[str] = []

            def flush_batch():
                """Embed current buffer and extend `records`."""
                if not batch_texts:
                    return
                embs = self._embed_batch(batch_texts)  # (B, D) float32
                now = utils.now_iso
                for rid, txt, vec in zip(batch_ids, batch_texts, embs):
                    records.append({
                        mammogram_id_col: rid,
                        "embedding": vec.tolist(),
                        "embedding_dim": int(vec.shape[0]),
                        "raw_text_hash": utils.sha1(txt) if txt else None,
                        "model": self.embedding_id,
                        "created_at": now(),
                        "num_chunks": 1,
                    })
                batch_ids.clear(); batch_texts.clear()

            # iterate efficiently; each row is (id, text)
            for rid, txt in tqdm(df[[mammogram_id_col, text_col]].itertuples(index=False, name=None)):
                # empty/NaN text → emit an empty record (no embedding call)
                if not isinstance(txt, str) or not txt.strip():
                    records.append({
                        mammogram_id_col: rid,
                        "embedding": [],
                        "embedding_dim": 0,
                        "raw_text_hash": None,
                        "model": self.embedding_id,
                        "created_at": utils.now_iso(),
                        "num_chunks": 0,
                    })
                    continue

                batch_ids.append(rid)
                batch_texts.append(txt.strip())

                if len(batch_texts) >= 1000:
                    flush_batch()

                if len(records) and (len(records) % 5000 == 0):
                    print(f"  .. {len(records):,} rows prepared")

            # tail batch
            flush_batch()

            out_df = pd.DataFrame.from_records(records)
            out_df.to_parquet(current_emb_file, index=False)
            print(f"[done] wrote {len(out_df)} rows -> {current_emb_file.name}")



class BaseFitEmbedModel(ConfigInterface):
    def __init__(self, config_dir: str, config_defaults: dict) -> None:
        super().__init__(config_dir, config_defaults)
        self.verbose = True
        self.processed_birads_filenames = [
            "processed_birads_for_training.parquet", 
            "infered_birads_bov.parquet"
        ]
        self.selected_texts = []

    def _iter_input_files(self) -> Iterable[Path]:
        yield from sorted(self.raw_mammograms_folder_path.glob("*.parquet"))
    
    def _get_key_to_birads(
        self,
        batch_size: Optional[int] = 100000,
        minimum_date: Optional[str] = '2010-01-01'
    ) -> pd.DataFrame:
        key_to_birads_df = {}
        for current_filename in self.processed_birads_filenames:
            current_src = self.processed_birads_folder_path.joinpath(self.processed_birads_filenames[0])
            for current_batch in batching_parquet_file(current_src, batch_size=batch_size):
                current_batch = current_batch[(pd.notna(current_batch["processed_birads"])) & (current_batch["DT_ATENDIMENTO"]>=minimum_date)].copy()
                if current_batch.shape[0]==0:
                    continue
                current_batch["key"] = current_batch["CD_ATENDIMENTO"].apply(lambda x: f"{x:.0f}") + current_batch["raw_text_hash"]
                key_to_birads_df.update(dict(zip(current_batch["key"], current_batch["processed_birads"])))
        key_to_birads_df = list(key_to_birads_df.items())
        key_to_birads_df = pd.DataFrame(
            {
                "key": [ i[0] for i in key_to_birads_df ],
                "birads": [ i[1] for i in key_to_birads_df ]
            }
        )
        return key_to_birads_df

    def _select_reports_for_fitting(
        self, 
        sample_benign_texts: Optional[bool] = True,
        sampling_ratio: Optional[int] = 4,
        batch_size: Optional[int] = 100000,
        minimum_date: Optional[str] = '2010-01-01',
        mammogram_ids: Optional[List[str]] = None
    ) -> None:
        '''
            Not very productive (and prone to bias) if we train the model on all
            reports. Therefore, a sample is obtained following a ratio between
            low and high risk reports (according to the BI-RADS).
        '''
        key_to_birads_df = self._get_key_to_birads(batch_size, minimum_date)
        if sample_benign_texts:
            inconclusive, low_risk, high_risk = [0], [1,2,3], [4,5,6]
            df_inc = key_to_birads_df[key_to_birads_df['birads'].isin(inconclusive)]
            df_high = key_to_birads_df[key_to_birads_df['birads'].isin(high_risk)]
            df_low = key_to_birads_df[key_to_birads_df['birads'].isin(low_risk)]
            if mammogram_ids:
                df_low = df_low[df_low["key"].isin(mammogram_ids)].copy()

            n_high = len(df_high)
            n_low = int(min(len(df_low), sampling_ratio * n_high))
            n_inc = int(min(len(df_inc), n_high))

            df_low_sample = df_low.sample(n=n_low, random_state=42)
            df_inc_sample = df_low.sample(n=n_inc, random_state=42)

            df_sampled = pd.concat([df_high, df_low_sample, df_inc_sample], ignore_index=True)
            valid_keys = df_sampled["key"].tolist()
        else:
            valid_keys = key_to_birads_df["key"].tolist()

        # -- legacy already (config interface already does this)
        for current_raw_file in self._iter_input_files():
            df = pd.read_parquet(current_raw_file)

            df['raw_text_hash'] = df["DS_LAUDO_MEDICO"].apply(lambda x: utils.sha1(x.strip()) if pd.notna(x) else np.nan)
            df['key'] = df["CD_ATENDIMENTO"].apply(lambda x: f'{x:.0f}') + df['raw_text_hash']

            df = df[df["key"].isin(valid_keys)]
            if df.shape[0]>0:
                self.selected_texts.extend( df["DS_LAUDO_MEDICO"].tolist() )

    def fit(self) -> None:
        pass


class FitTFIDF(BaseFitEmbedModel):
    def __init__(self, config_dir: str, config_defaults: dict) -> None:
        super().__init__(config_dir, config_defaults)

        self.tfidf_config = self.embeddings_cfg["tfidf"]
        self.model_name = self.tfidf_config['model_name']
        self.fext = ".joblib"
        self.to_fit_svd = self.tfidf_config["svd"]
        self.svd_dim = self.tfidf_config["svd_dim"]

        self.selected_texts = []
        self.model_vec = None
        self.svd = None

    def _fit_tfidf(self):
        self.model_vec = utils.fit_tfidf_vec(self.selected_texts, params=self.tfidf_config)
        model_save_path = self.fitted_models_folder_path.joinpath(self.model_name+self.fext)
        joblib.dump(self.model_vec, model_save_path)

    def _fit_svd(
        self,
        svd_random_state: Optional[int] = 42
    ):
        self.svd = TruncatedSVD(n_components=self.svd_dim, random_state=42)
        X_train = self.model_vec.transform(self.selected_texts)
        self.svd.fit(X_train) 
        print(np.cumsum(self.svd.explained_variance_ratio_))  
        joblib.dump(self.svd, self.fitted_models_folder_path.joinpath(f"svd_{self.svd_dim:.0f}_{self.model_name}"+self.fext))

    def fit(
        self,
        sample_benign_texts: Optional[bool] = True,
        sampling_ratio: Optional[int] = 3,
        batch_size: Optional[int] = 100000,
        minimum_date: Optional[str] = '2010-01-01',
        mammogram_ids: Optional[List[str]] = None # included for a sensitivity analysis
    ):
        self._select_reports_for_fitting(sample_benign_texts, sampling_ratio, batch_size, minimum_date, mammogram_ids)
        self._fit_tfidf()
        if self.to_fit_svd:
            self._fit_svd()



        

