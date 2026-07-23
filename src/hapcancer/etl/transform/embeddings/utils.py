import hashlib
import ollama
import yaml
import torch
import joblib
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import datetime as dt
from pathlib import Path
from typing import List, Optional, Sequence, Union, Any, Callable

from transformers import DataCollatorWithPadding
from transformers import TrainingArguments
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

def embed_text(text: str, model: str = "nomic-embed-text") -> List[float]:
    res = ollama.embed(model=model, input=text)
    return np.array(res["embeddings"][0])

def embed_text_deberta_v3(
    input: List[str], 
    max_length: Optional[int] = 512,
    model_name: Optional[str] = "microsoft/deberta-v3-base"
) -> Sequence:
    # -- load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
    # -- load the mode (embedding output)
    model = AutoModel.from_pretrained(
        model_name,
        device_map="cuda",
        torch_dtype=torch.float16,
        use_auth_token=True,
    )
    
    def create_embeddings(input, model, tokenizer, batch_size=16, max_length=max_length, pooling="cls"):
        """
        Create embeddings for a list of texts using the specified model and tokenizer.

        Args:
            input (list of str): List of input texts.
            model: Hugging Face model (AutoModel).
            tokenizer: Hugging Face tokenizer (AutoTokenizer).
            batch_size (int): Batch size for processing.
            max_length (int): Maximum token length for truncation/padding.
            pooling (str): Pooling strategy ('mean' or 'cls').

        Returns:
            torch.Tensor: Embeddings of shape (num_texts, hidden_size).
        """
        # -- tokenize the input texts
        encoded_texts = tokenizer(
            input,
            padding=True,
            #truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        #print(input)

        # -- prepare DataLoader for batching
        dataset = DeBertaDataset(encoded_texts["input_ids"], encoded_texts["attention_mask"])
        dataloader = DataLoader(dataset, batch_size=batch_size)

        # -- move model to device
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = "cuda"
        model.to(device)
        model.eval()

        # -- collect embeddings
        embeddings = []
        torch.cuda.empty_cache()
        with torch.no_grad():
            for batch in dataloader:
                input_ids, attention_mask = batch
                input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)

                # Get model outputs
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                hidden_states = outputs.last_hidden_state  # Shape: (batch_size, seq_length, hidden_size)

                # Pooling
                if pooling == "mean":
                    # Mean pooling: Average embeddings over non-padded tokens
                    mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                    pooled = torch.sum(hidden_states * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
                elif pooling == "cls":
                    # CLS pooling: Take the embedding of the first token ([CLS])
                    pooled = hidden_states[:, 0, :]
                else:
                    raise ValueError("Pooling must be 'mean' or 'cls'")

                embeddings.append(pooled.detach().cpu().numpy())

        return embeddings

    # -- Generate embeddings for training data
    embeddings_chunk = []
    print("Generating embeddings for input texts ...")
    embeddings_chunk = create_embeddings(input, model, tokenizer, batch_size=4, max_length=max_length)
    return embeddings_chunk


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("latin")).hexdigest()

def now_iso() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")

def word_chunks(text: str, max_words: int, stride_words: int) -> List[str]:
    words = text.split()
    if not words:
        return []
    if max_words <= 0:
        return [" ".join(words)]
    chunks = []
    i = 0
    n = len(words)
    while i < n:
        j = min(i + max_words, n)
        chunks.append(" ".join(words[i:j]))
        if j == n:
            break
        i = j - stride_words if stride_words > 0 else j
        if i < 0:
            i = 0
    return chunks

def mean_pool(vectors: List[List[float]]) -> List[float]:
    if not vectors:
        return []
    m = len(vectors[0])
    acc = [0.0] * m
    for v in vectors:
        for i in range(m):
            acc[i] += v[i]
    return [x / len(vectors) for x in acc]

def load_config_file(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


# ----------- TF-IDF vectorizer for new embeddings -----------

def tfidf_doc_iterator(
    list_of_docs: List[str]
):
    for text in list_of_docs:
        yield text


def fit_tfidf_vec(
    list_of_docs: List[str],
    params: Optional[dict] = None
):
    params_ = dict()
    if params is not None:
        params_['ngram_range'] = tuple(params['ngram_range'])
        params_['min_df'] = params['min_df']
        params_['max_df'] = params['max_df']
        params_['max_features'] = params['max_features']
    
    vec = TfidfVectorizer(
        analyzer="word",
        stop_words=stopwords.words('portuguese'),
        lowercase=True,
        ngram_range=params_.get('ngram_range', (1)),
        min_df=params_.get('min_df', 5),          # tune
        max_df=params_.get('max_df', 1.0),         # tune
        max_features=params_.get('max_features', 50000),
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32
    )
    it = tqdm(tfidf_doc_iterator(list_of_docs))
    vec.fit(it)
    return vec

# --------------------------------------
# Load embedding model functions
# --------------------------------------
def load_tfidf_emb_model(config: dict) -> Callable[[List[str]], List[np.ndarray]]:
    transform_path = Path(config["transform"]["path"]).joinpath(config["transform"]["id"])
    fitted_models_folder_path = config["transform"]["folders"]["fitted_models"]
    
    emb_model_config = config['transform']['embedding']
    use_svd = emb_model_config['tfidf']['svd']
    svd_dim = emb_model_config['tfidf']['svd_dim']
    model_name = emb_model_config['tfidf']['model_name']
    path_to_tfidf_model = transform_path.joinpath(fitted_models_folder_path, model_name+'.joblib')
    path_to_svd_model = transform_path.joinpath(fitted_models_folder_path, f"svd_{svd_dim:.0f}_{model_name}"+'.joblib')

    tfidf_model = joblib.load(path_to_tfidf_model)
    if use_svd:
        svd_model = joblib.load(path_to_svd_model)
        return lambda x: svd_model.transform(tfidf_model.transform(x))
    return lambda x: tfidf_model.transform(x)