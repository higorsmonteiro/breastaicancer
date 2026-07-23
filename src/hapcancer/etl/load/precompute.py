import lmdb
import joblib
import torch
import ollama
import pyarrow as pa
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
import pyarrow.parquet as pq
from collections import defaultdict
from typing import List, Optional, Sequence, Dict, Union, Tuple, Iterable, Callable

from transformers import AutoTokenizer, AutoModel

from hapcancer.schemas.enums import MammogramColumns
from hapcancer.etl.transform.embeddings import utils as emb_utils
from hapcancer.etl.utils import batching_parquet_file, sha1
from hapcancer.config_manager import ConfigInterface

# --------------------------------------
# Load embedding model functions
# --------------------------------------
def load_tfidf_emb_model(
    emb_config: dict, 
    path_to_model: Union[str, Path]
) -> Callable[[List[str]], List[np.ndarray]]:    
    emb_model_config = emb_config['tfidf']
    use_svd = emb_model_config['svd']
    svd_dim = emb_model_config['svd_dim']
    model_name = emb_model_config['model_name']
    path_to_tfidf_model = path_to_model.joinpath(model_name+'.joblib')
    path_to_svd_model = path_to_model.joinpath(f"svd_{svd_dim:.0f}_{model_name}"+'.joblib')

    tfidf_model = joblib.load(path_to_tfidf_model)
    if use_svd:
        svd_model = joblib.load(path_to_svd_model)
        return lambda x: svd_model.transform(tfidf_model.transform(x))
    return lambda x: tfidf_model.transform(x)

def load_bert_emb_model(
    emb_config: dict,
    device: Optional[str] = None,
) -> Callable[[List[str]], np.ndarray]:
    """
        Loads a HuggingFace encoder model (BERTimbau, BioBERTpt, etc.) and returns
        a callable with the same signature as the TF-IDF one:
 
            model(texts: List[str]) -> np.ndarray  shape (N, hidden_dim)
 
        Config keys expected under emb_config['bert']:
            model_name  : HuggingFace model identifier, e.g.
                            'neuralmind/bert-base-portuguese-cased'   (BERTimbau)
                            'pucpr/biobertpt-clin'                    (BioBERTpt clinical)
            batch_size  : int, texts per forward pass (default 32)
            max_length  : int, tokenizer max length    (default 512)
            pooling     : 'mean' | 'cls'               (default 'mean')
    """
    bert_cfg   = emb_config['bert']
    model_name = bert_cfg['model_name']
    batch_size = bert_cfg.get('batch_size', 32)
    max_length = bert_cfg.get('max_length', 512)
    pooling    = bert_cfg.get('pooling', 'mean')
 
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModel.from_pretrained(model_name)
    model.eval()
    model.to(device)
 
    def _pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> np.ndarray:
        if pooling == 'cls':
            return last_hidden_state[:, 0, :].cpu().numpy()
        # mean pooling: ignore padding tokens
        mask_expanded = attention_mask.unsqueeze(-1).float()
        summed  = (last_hidden_state * mask_expanded).sum(dim=1)
        counts  = mask_expanded.sum(dim=1).clamp(min=1e-9)
        return (summed / counts).cpu().numpy()
 
    def embed(texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, model.config.hidden_size), dtype=np.float16)
        all_vecs = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            encoded = tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt',
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            with torch.no_grad():
                output = model(**encoded)
            vecs = _pool(output.last_hidden_state, encoded['attention_mask'])
            all_vecs.append(vecs)
        return np.vstack(all_vecs).astype(np.float16)
    return embed

def load_ollama_emb_model(
    emb_config: dict,
) -> Callable[[List[str]], np.ndarray]:
    """
    Returns an embedding callable backed by a locally running Ollama server,
    using the official ollama Python library.

    Config keys expected under emb_config['ollama']:
        model_name  : str   Ollama model tag, e.g. 'qwen3:embedding'
        batch_size  : int   texts per request  (default 32)
    """
    ollama_cfg = emb_config['ollama']
    model_name = ollama_cfg['model_name']
    batch_size = ollama_cfg.get('batch_size', 32)
    dimensions = ollama_cfg.get('ndim', None)

    def embed(texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0,), dtype=np.float16)
        all_vecs = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            response = ollama.embed(model=model_name, input=chunk, dimensions=dimensions)
            vecs = np.array(response['embeddings'], dtype=np.float16)
            all_vecs.append(vecs)
        return np.vstack(all_vecs)
    return embed

def define_model(
    emb_config: dict, 
    path_to_model: Optional[Union[str,Path]] = None
) -> Callable[[List[str]], List[np.ndarray]]:
    if emb_config['embedding_id']=='tfidf':
        return load_tfidf_emb_model(emb_config, path_to_model)
    else:
        return lambda x: x

def define_model(
    emb_config: dict,
    path_to_model: Optional[Union[str, Path]] = None,
    device: Optional[str] = None,
) -> Callable[[List[str]], np.ndarray]:
    """
        Extended version of define_model() that also handles BERT-based models.
        Keeps the original TF-IDF branch untouched.
    """
    eid = emb_config['embedding_id']
    if eid == 'tfidf':
        return load_tfidf_emb_model(emb_config, path_to_model)
    elif eid in ('bert', 'bertimbau', 'biobertpt'):
        return load_bert_emb_model(emb_config, device=device)
    elif eid == 'ollama':
        return load_ollama_emb_model(emb_config)
    else:
        raise ValueError(f"Unknown embedding_id: '{eid}'")

def create_past_sequences_model_mean(
    key2text: Dict[str, str],
    embedding_model: Callable[[List[str]], np.ndarray],
    mammogram_ids: List[str],
    mammogram_current_dates: List[pd.Timestamp],
    prior_histories: List[List[str]],
    prior_dates: List[List[pd.Timestamp]],
    time_limit: Optional[int] = None,
    is_sparse: Optional[bool] = True
) -> Tuple[List[List[np.ndarray]], List[np.ndarray]]:
    '''
        For each mammogram id, returns the list of past mammogram exams (embeddings)
        and the timing between the current mammogram and the other exams (in days).

        Args:
        -----
            key2text: Dict[str, str]. Dictionary containing an the raw exam text for
            each mammogram id.
            embedding_model: Callable[[List[str]], np.ndarray]. Embedding model who converts
            a list of texts to their respective representation vectors.
            mammogram_ids: List[str]. A list of mammogram ids for further computation.
            mammogram_current_dates: List[pd.Timestamp]. A list of dates for each mammogram id.
            prior_histories: List[List[str]]. List containing lists of ids of the past mammograms
            of each current mammogram id.
            prior_dates: List[List[str]]. List containing lists of dates of the past mammograms
            of each current mammogram id.
            time_limit: Optional[int] = None. Number of months in the past to consider past exams.
        
        Returns:
        --------
            embs_per_mammogram. List[List[np.ndarray].
            time_diff_per_mammogram. List[np.ndarray].
    '''
    embs_per_mammogram = []
    time_diff_per_mammogram = []
    for ix, cur_mammogram_id in enumerate(mammogram_ids):
        cur_date = pd.Timestamp(mammogram_current_dates[ix])
        cur_prior_history = prior_histories[ix]
        cur_prior_dates = prior_dates[ix]
        
        earliest = None
        if time_limit is not None:
            earliest = cur_date - pd.DateOffset(months=time_limit)

        # -- define sequence of embeddings and sequence of time deltas.
        cur_text_date_seq = [
            (key2text[cur_key], cur_date_) for cur_key, cur_date_ in zip(cur_prior_history, cur_prior_dates) if cur_key in key2text
        ]
        cur_emb_seq = embedding_model([ text for text, date in cur_text_date_seq ])
        if is_sparse:
            cur_emb_seq = np.array(cur_emb_seq.toarray(), dtype=np.float16) # -- maybe work only for tf-idf csr matrices?
        cur_emb_date_seq = [ date for text, date in cur_text_date_seq ]

        # -- calculate time deltas (in days)
        cur_dt64 = np.datetime64(cur_date, 'D')
        cur_emb_date_seq = np.array([np.datetime64(dt, 'D') for dt in cur_emb_date_seq])
        diffs_days = (cur_dt64 - cur_emb_date_seq).astype('timedelta64[D]').astype(np.int64)

        # -- if 'time_limit' equals zero, it signals that we are going to use only the index mammogram
        if time_limit==0:
            cur_emb_seq = cur_emb_seq[:1]
            diffs_days = diffs_days[:1]
        
        embs_per_mammogram.append(cur_emb_seq)
        time_diff_per_mammogram.append(diffs_days)

    return embs_per_mammogram, time_diff_per_mammogram


# =============================================================================
# X.  AGGREGATION HELPERS
#     create_past_sequences_model_mean is reused unchanged for collecting
#     raw embeddings + time diffs.  The aggregation step lives in the
#     Precompute classes below, mirroring the existing pattern.
# =============================================================================

def aggregate_mean(
    embs: np.ndarray,           # shape (T, D+1)  — embeddings already hstacked with time diff
) -> np.ndarray:                # shape (D+1,)
    """Simple mean — matches current behaviour."""
    return embs.mean(axis=0)

def aggregate_time_weighted_mean(
    embs: np.ndarray,           # shape (T, D)
    time_diffs: np.ndarray,     # shape (T,)  in days, already ≥ 0
    decay: str  = 'exponential',
    lam:   float = 0.001,       # λ for exponential:  w = exp(-λ * Δt_days)
                                # a value of 0.001 gives ~half-weight at ~693 days (~2 yrs)
) -> np.ndarray:                # shape (D,)
    """
    Weighted mean of embedding vectors, where weights reflect document recency.
 
    decay options
    -------------
    'exponential'  w_i = exp(-λ * Δt_i)            most principled, smoothly decays
    'inverse'      w_i = 1 / (1 + Δt_i)            slower decay for very recent docs
    'linear'       w_i = max(0,  1 - Δt_i / max_t) oldest doc gets weight 0
 
    The time_diff column is NOT included in the output vector (unlike aggregate_mean
    which appends it).  This keeps the output dimensionality equal to the embedding
    dimension, which is cleaner.  Adjust if your downstream model expects it.
    """
    if len(embs) == 0:
        return np.zeros(embs.shape[1], dtype=np.float16)
 
    t = time_diffs.astype(np.float32)
 
    if decay == 'exponential':
        weights = np.exp(-lam * t)
    elif decay == 'inverse':
        weights = 1.0 / (1.0 + t)
    elif decay == 'linear':
        max_t = t.max() if t.max() > 0 else 1.0
        weights = np.clip(1.0 - t / max_t, 0.0, 1.0)
    else:
        raise ValueError(f"Unknown decay: '{decay}'")
 
    weights = weights / weights.sum()                      # normalise → sum to 1
    weighted = (embs.astype(np.float32) * weights[:, None]).sum(axis=0)
    return weighted.astype(np.float16)

class BasePrecompute(ConfigInterface):
    '''
        The precomputation task depends heavily on what type of model is
        used to process the mammograms' texts.

        If an embedding model is used to process each text, we might either 
        embed text as we perform the precomputation or we might create a
        vector store to collect during the precomputation. If instead of an
        embedding model, we need to collect the sequence of raw texts, then 
        another approch need to be used. The same goes for which type of 
        calculation is done (e. g. mean vector of sequence of embeddings or
        weighted mean, etc).

        This class forms the base for each main precomputation choice to be
        used during model development.
    '''
    def __init__(self, config_dir: str, config_defaults: dict):
        super().__init__(config_dir, config_defaults)
        self.mamm_seq_elig_filename = self.files_and_folders_cfg["load"]["load_files"]["seq_per_mammogram_filename"]
        self.structured_elig_filename = self.files_and_folders_cfg["load"]["load_files"]["final_data_with_eligibility_filename"]

        self.bcols = {
            "id": "mammogram_id",
            "current_date": "mammogram_current_date",
            "prior_codes": "mammogram_prior_codes",
            "prior_dates": "mammogram_prior_dates"
        }
        self.embedding_model = None

    def _define_model(self):
        self.embedding_model = None

    def _collect_keys(self, batch_size: Optional[int] = 10_000) -> None:
        '''
            For the final sequence dataset, collect all the keys
        '''
        seq_cols = [ col_nm for k, col_nm in self.bcols.items() ]
        src = self.dataset_path.joinpath(self.mamm_seq_elig_filename)
        # -- go through final sequence dataset
        self.valid_keys = []
        for batch_df in tqdm(batching_parquet_file(src, columns=seq_cols, batch_size=batch_size)):
            current_ids, prior_codes = batch_df[self.bcols["id"]].tolist(), batch_df[self.bcols["prior_codes"]].tolist()
            self.valid_keys.extend(current_ids)
            [ self.valid_keys.extend(cur_list) for cur_list in prior_codes ]
            self.valid_keys = list(set(self.valid_keys))

    def _define_key_to_text(self) -> None:
        self.key2text = {}
        #self.valid_keys = set(self.valid_keys)
        valid_keys_df = pd.DataFrame({'key': list(self.valid_keys)})
        for raw_batch_df in tqdm(self._iter_raw_mammograms_data()):
            #raw_batch_df = raw_batch_df[raw_batch_df["key"].isin(self.valid_keys)].copy()
            filtered = raw_batch_df.merge(valid_keys_df, on='key', how='inner')[['key', 'text']]
            self.key2text.update(dict(zip(filtered["key"], filtered["text"])))
        self.valid_keys = None
    
    def precompute(self):
        pass

# =============================================================================
# X.  PRECOMPUTE CLASSES
#     Each class is a self-contained variant.  Internal structure mirrors
#     PrecomputeSequenceTFIDF exactly.
# =============================================================================

class PrecomputeSequenceTFIDF(BasePrecompute):
    def __init__(self, config_dir: str, config_defaults: dict):
        super().__init__(config_dir, config_defaults)
        # -- configuration file holds info for tf-idf model
        self.tfidf_config = self.embeddings_cfg["tfidf"]
        self.path_to_model = self.fitted_models_folder_path

        # -- data is not supposed to become much larger than what we have now, so load all valid keys is fine
        self.valid_keys = None 
        self.key2text = None
        self.embedding_model = None

        self.suffix = self.tfidf_config['model_name']
        self.output_filepath = None

    def _define_model(self):
        self.embedding_model = define_model(self.embeddings_cfg, self.path_to_model) # -- returns a callable

    def _embed_text(self, texts: List[str]):
        return self.embedding_model(texts)

    def _precompute_sequences(
        self,
        time_limit: Optional[int] = 36,
        batch_size: Optional[int] = 1000,
        gb_size: Optional[int] = 10
    ) -> None:
        
        print(f"TIME LIMIT: {time_limit}\nBATCH_SIZE: {batch_size}\nGB_SIZE: {gb_size}")
        self.embs, self.time_diffs = None, None

        # -- Open or create an LMDB storage for the embedding vectors
        lmdb_path = str(self.dataset_path.joinpath(f"mammogram_id_embeddings_{time_limit}_{self.suffix}.lmdb"))
        env = lmdb.open(
            lmdb_path,
            map_size=gb_size*1024**3,   # allocate gb_size GB virtual space, disk stays small
            subdir=False,
            lock=True,
            readahead=False,          # important for random access performance
        )

        seq_cols = [col_nm for k, col_nm in self.bcols.items()]
        src = self.dataset_path.joinpath(self.mamm_seq_elig_filename)

        with env.begin(write=True) as txn:
            for batch_df in tqdm(batching_parquet_file(src, columns=seq_cols, batch_size=batch_size)):
                current_ids = batch_df[self.bcols["id"]].tolist()
                current_date = batch_df[self.bcols["current_date"]].tolist()
                prior_codes = batch_df[self.bcols["prior_codes"]].tolist()
                prior_dates = batch_df[self.bcols["prior_dates"]].tolist()

                # UPDATE
                self.embs, self.time_diffs = create_past_sequences_model_mean(
                    self.key2text, self._embed_text,
                    current_ids, current_date, prior_codes, prior_dates,
                    time_limit
                )

                mean_mammogram_seqs = [
                    np.hstack([
                        np.vstack(cur_embs),
                        cur_time_diffs.reshape(-1, 1)
                    ]).astype(np.float16).mean(axis=0)
                    for cur_embs, cur_time_diffs in zip(self.embs, self.time_diffs)
                ]

                # LMDB WRITE: id → encoded embedding
                for mid, vec in zip(current_ids, mean_mammogram_seqs):
                    txn.put(str(mid).encode(), vec.tobytes())

    def precompute(
        self,
        time_limit: Optional[int] = 36,
        batch_size: Optional[int] = 25000,
        gb_size: Optional[int] = 10
    ):
        print("[collect] keys and define key2text dictionary ...")
        self._collect_keys()
        self._define_key_to_text()
        print("[define] embedding model ...")
        self._define_model()
        print("[precompute] sequence ...")
        self._precompute_sequences(time_limit, batch_size, gb_size)

class PrecomputeSequenceTFIDFTimeWeighted(BasePrecompute):
    """
    TF-IDF embeddings with exponential time-weighted mean aggregation.
    Isolates the contribution of the aggregation strategy vs the baseline.
 
    Extra config keys under embeddings_cfg:
        time_weighted:
            decay : 'exponential' | 'inverse' | 'linear'   (default 'exponential')
            lam   : float                                    (default 0.001)
    """
    def __init__(self, config_dir: str, config_defaults: dict):
        super().__init__(config_dir, config_defaults)
        self.tfidf_config = self.embeddings_cfg['tfidf']
        self.tw_config    = self.embeddings_cfg.get('time_weighted', {})
        self.path_to_model = self.fitted_models_folder_path
 
        self.valid_keys      = None
        self.key2text        = None
        self.embedding_model = None
 
        self.suffix = 'tw_' + self.tfidf_config['model_name']
 
    def _define_model(self):
        self.embedding_model = define_model(
            self.embeddings_cfg, path_to_model=self.path_to_model
        )
 
    def _embed_text(self, texts: List[str]) -> np.ndarray:
        return self.embedding_model(texts)
 
    def _precompute_sequences(
        self,
        time_limit: Optional[int] = 36,
        batch_size: Optional[int] = 1000,
        gb_size:    Optional[int] = 10,
    ) -> None:
        decay = self.tw_config.get('decay', 'exponential')
        lam   = self.tw_config.get('lam',   0.001)
 
        lmdb_path = str(
            self.dataset_path.joinpath(
                f"mammogram_id_embeddings_{time_limit}_{self.suffix}.lmdb"
            )
        )
        env = lmdb.open(lmdb_path, map_size=gb_size * 1024**3,
                        subdir=False, lock=True, readahead=False)
 
        seq_cols = [col_nm for col_nm in self.bcols.values()]
        src      = self.dataset_path.joinpath(self.mamm_seq_elig_filename)
 
        with env.begin(write=True) as txn:
            for batch_df in tqdm(batching_parquet_file(src, columns=seq_cols, batch_size=batch_size)):
                current_ids   = batch_df[self.bcols['id']].tolist()
                current_dates = batch_df[self.bcols['current_date']].tolist()
                prior_codes   = batch_df[self.bcols['prior_codes']].tolist()
                prior_dates   = batch_df[self.bcols['prior_dates']].tolist()
 
                embs, time_diffs = create_past_sequences_model_mean(
                    self.key2text, self._embed_text,
                    current_ids, current_dates, prior_codes, prior_dates,
                    time_limit, is_sparse=True
                )
 
                tw_seqs = []
                for cur_embs, cur_time_diffs in zip(embs, time_diffs):
                    cur_embs = np.array(cur_embs.toarray(), dtype=np.float16)
                    if cur_embs.ndim == 1:
                        cur_embs = cur_embs[None, :]
                    vec = aggregate_time_weighted_mean(cur_embs, cur_time_diffs, decay=decay, lam=lam)
                    tw_seqs.append(vec)
 
                for mid, vec in zip(current_ids, tw_seqs):
                    txn.put(str(mid).encode(), vec.tobytes())
 
    def precompute(
        self,
        time_limit: Optional[int] = 36,
        batch_size: Optional[int] = 25_000,
        gb_size:    Optional[int] = 10,
    ):
        print('[collect] keys and define key2text dictionary ...')
        self._collect_keys()
        self._define_key_to_text()
        print('[define] TF-IDF embedding model ...')
        self._define_model()
        print('[precompute] time-weighted sequence ...')
        self._precompute_sequences(time_limit, batch_size, gb_size)

class PrecomputeSequenceBERT(BasePrecompute):
    """
        Precomputation using a HuggingFace BERT-based encoder (BERTimbau or
        BioBERTpt) with simple mean aggregation.
 
        Expected embeddings_cfg keys:
            embedding_id : 'bert' | 'bertimbau' | 'biobertpt'
            bert:
                model_name : str    (HuggingFace identifier)
                batch_size : int    (default 32)
                max_length : int    (default 512)
                pooling    : str    ('mean' | 'cls', default 'mean')
    """
    def __init__(self, config_dir: str, config_defaults: dict, device: Optional[str] = None):
        super().__init__(config_dir, config_defaults)
        #self.bert_config = if self.embeddings_cfg.get('bert', None) is None else self.embeddings_cfg.get('ollama', None)
        self.bert_config = self.embeddings_cfg.get('bert', None) if self.embeddings_cfg.get('bert', None) is not None else self.embeddings_cfg.get('ollama', None)
        #self.bert_config = self.embeddings_cfg['bert']
        self.device      = device
 
        self.valid_keys      = None
        self.key2text        = None
        self.embedding_model = None
 
        # suffix used in output filenames — use short model name
        _raw = self.bert_config['model_name']           # e.g. 'neuralmind/bert-base-portuguese-cased'
        self.suffix = _raw.split('/')[-1]               # e.g. 'bert-base-portuguese-cased'
        self.output_filepath = None
 
    # ------------------------------------------------------------------
    # model
    # ------------------------------------------------------------------
    def _define_model(self):
        self.embedding_model = define_model(
            self.embeddings_cfg, device=self.device
        )
 
    def _embed_text(self, texts: List[str]) -> np.ndarray:
        return self.embedding_model(texts)
 
    # ------------------------------------------------------------------
    # precompute — same skeleton, different embed call
    # ------------------------------------------------------------------
    def _precompute_sequences(
        self,
        time_limit: Optional[int] = 36,
        batch_size: Optional[int] = 256,      # smaller default: BERT is heavier
        gb_size:    Optional[int] = 10,
    ) -> None:
        lmdb_path = str(
            self.dataset_path.joinpath(
                f"mammogram_id_embeddings_{time_limit}_{self.suffix}.lmdb"
            )
        )
        env = lmdb.open(lmdb_path, map_size=gb_size * 1024**3,
                        subdir=False, lock=True, readahead=False)
 
        seq_cols = [col_nm for col_nm in self.bcols.values()]
        src      = self.dataset_path.joinpath(self.mamm_seq_elig_filename)
 
        with env.begin(write=True) as txn:
            for batch_df in tqdm(batching_parquet_file(src, columns=seq_cols, batch_size=batch_size)):
                current_ids   = batch_df[self.bcols['id']].tolist()
                current_dates = batch_df[self.bcols['current_date']].tolist()
                prior_codes   = batch_df[self.bcols['prior_codes']].tolist()
                prior_dates   = batch_df[self.bcols['prior_dates']].tolist()
 
                embs, time_diffs = create_past_sequences_model_mean(
                    self.key2text, self._embed_text,
                    current_ids, current_dates, prior_codes, prior_dates,
                    time_limit, is_sparse=False
                )
 
                # NOTE: BERT embeddings are dense ndarray already — no .toarray() needed.
                # create_past_sequences_model_mean calls .toarray() which only works for
                # sparse matrices.  We override that step here.
                mean_seqs = []
                for cur_embs, cur_time_diffs in zip(embs, time_diffs):
                    if cur_embs.ndim == 1:          # single past document
                        cur_embs = cur_embs[None, :]
                    vec = np.hstack([
                        cur_embs,
                        cur_time_diffs.reshape(-1, 1),
                    ]).astype(np.float16).mean(axis=0)
                    mean_seqs.append(vec)
 
                for mid, vec in zip(current_ids, mean_seqs):
                    txn.put(str(mid).encode(), vec.tobytes())
 
    def precompute(
        self,
        time_limit: Optional[int] = 36,
        batch_size: Optional[int] = 256,
        gb_size:    Optional[int] = 10,
    ):
        print('[collect] keys and define key2text dictionary ...')
        self._collect_keys()
        self._define_key_to_text()
        print('[define] BERT embedding model ...')
        self._define_model()
        print('[precompute] sequence ...')
        self._precompute_sequences(time_limit, batch_size, gb_size)

class PrecomputeSequenceBERTTimeWeighted(BasePrecompute):
    """
    BERT-based embeddings (BERTimbau / BioBERTpt) with time-weighted mean
    aggregation.  The combination of both extensions.
 
    Config keys: same as PrecomputeSequenceBERT + time_weighted block.
    """
    def __init__(self, config_dir: str, config_defaults: dict, device: Optional[str] = None):
        super().__init__(config_dir, config_defaults)
        #self.bert_config = self.embeddings_cfg['bert']
        self.bert_config = self.embeddings_cfg.get('bert', None) if self.embeddings_cfg.get('bert', None) is not None else self.embeddings_cfg.get('ollama', None)
        self.tw_config   = self.embeddings_cfg.get('time_weighted', {})
        self.device      = device
 
        self.valid_keys      = None
        self.key2text        = None
        self.embedding_model = None
 
        _raw = self.bert_config['model_name']
        self.suffix = 'tw_' + _raw.split('/')[-1]
 
    def _define_model(self):
        self.embedding_model = define_model(
            self.embeddings_cfg, device=self.device
        )
 
    def _embed_text(self, texts: List[str]) -> np.ndarray:
        return self.embedding_model(texts)
 
    def _precompute_sequences(
        self,
        time_limit: Optional[int] = 36,
        batch_size: Optional[int] = 256,
        gb_size:    Optional[int] = 10,
    ) -> None:
        decay = self.tw_config.get('decay', 'exponential')
        lam   = self.tw_config.get('lam',   0.001)
 
        lmdb_path = str(
            self.dataset_path.joinpath(
                f"mammogram_id_embeddings_{time_limit}_{self.suffix}.lmdb"
            )
        )
        env = lmdb.open(lmdb_path, map_size=gb_size * 1024**3,
                        subdir=False, lock=True, readahead=False)
 
        seq_cols = [col_nm for col_nm in self.bcols.values()]
        src      = self.dataset_path.joinpath(self.mamm_seq_elig_filename)
 
        with env.begin(write=True) as txn:
            for batch_df in tqdm(batching_parquet_file(src, columns=seq_cols, batch_size=batch_size)):
                current_ids   = batch_df[self.bcols['id']].tolist()
                current_dates = batch_df[self.bcols['current_date']].tolist()
                prior_codes   = batch_df[self.bcols['prior_codes']].tolist()
                prior_dates   = batch_df[self.bcols['prior_dates']].tolist()
 
                embs, time_diffs = create_past_sequences_model_mean(
                    self.key2text, self._embed_text,
                    current_ids, current_dates, prior_codes, prior_dates,
                    time_limit, is_sparse=True
                )
 
                tw_seqs = []
                for cur_embs, cur_time_diffs in zip(embs, time_diffs):
                    # BERT returns dense ndarray — no .toarray()
                    if cur_embs.ndim == 1:
                        cur_embs = cur_embs[None, :]
                    vec = aggregate_time_weighted_mean(cur_embs, cur_time_diffs, decay=decay, lam=lam)
                    tw_seqs.append(vec)
 
                for mid, vec in zip(current_ids, tw_seqs):
                    txn.put(str(mid).encode(), vec.tobytes())
 
    def precompute(
        self,
        time_limit: Optional[int] = 36,
        batch_size: Optional[int] = 256,
        gb_size:    Optional[int] = 10,
    ):
        print('[collect] keys and define key2text dictionary ...')
        self._collect_keys()
        self._define_key_to_text()
        print('[define] BERT embedding model ...')
        self._define_model()
        print('[precompute] time-weighted sequence ...')
        self._precompute_sequences(time_limit, batch_size, gb_size)
