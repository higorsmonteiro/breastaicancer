import yaml
import torch
import hashlib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from pathlib import Path
from collections import defaultdict
from typing import Union, List, Optional, Dict, Tuple, Iterable, Any

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("latin")).hexdigest()

def load_config_file(config_path: Union[str, Path]) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def batching_parquet_file(
    src: Union[str, Path],
    columns: Optional[List[str]] = None,
    batch_size: Optional[int] = 10000
) -> Iterable[pd.DataFrame]:
    pf = pq.ParquetFile(src)
    for batch in pf.iter_batches(columns=columns, batch_size=batch_size):
        yield batch.to_pandas()

