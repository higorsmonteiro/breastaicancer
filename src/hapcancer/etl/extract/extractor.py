import time
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from dotenv import dotenv_values
from typing import Dict, Any, Iterator, Optional, Sequence

import oracledb
oracledb.defaults.fetch_lobs = False

from hapcancer.logger import Logger
from hapcancer.etl.extract.queries import *
from hapcancer.config_manager import ConfigInterface

# ------------------------------------------------------------- #
# ---------------------- QUERY FUNCTIONS ---------------------- #
# ------------------------------------------------------------- #

def query_data_pool(query_str: str, pool: oracledb.ConnectionPool, batchsize: int = 5000) -> Dict[str, Any]:
    """
    Execute a SQL query using an Oracle connection pool, fetching results in batches.

    Notes:
        - Returns a dict with 'COLUMNS', 'TYPES', and 'ROWS'.
        - Uses tqdm to show progress of batch fetches (one tick per fetchmany call).
        - Keeps the existing behavior of accumulating all rows in memory.

    Args:
        query_str: SQL query string to execute.
        pool: Oracle connection pool.
        batchsize: Number of rows to fetch per round-trip.

    Returns:
        Dictionary containing columns, types, and the fetched rows.
    """
    schema_data = {"COLUMNS": [], "TYPES": [], "ROWS": []}
    with pool.acquire() as connection:
        cursor = connection.cursor()
        cursor.execute(query_str)

        schema_data["COLUMNS"] = [nm[0] for nm in cursor.description]
        schema_data["TYPES"] = [nm[1] for nm in cursor.description]

        pbar = tqdm()
        while True:
            pbar.update(1)
            rows = cursor.fetchmany(size=batchsize)
            if not rows:
                break
            schema_data["ROWS"] += [row for row in rows]
        pbar.close()

    return schema_data


def perform_query(query: str, connection: oracledb.ConnectionPool) -> pd.DataFrame:
    """
        Convenience helper: run a query through the pool and return a DataFrame.

        Args:
        -----
            query: SQL query string.
            connection: Oracle connection pool (kept name to avoid changing usage).

        Returns:
        --------
            DataFrame with the query results.
    """
    qdata = query_data_pool(query, connection)
    return pd.DataFrame(qdata["ROWS"], columns=qdata["COLUMNS"])


def fetch_query_iter(query: str, pool: oracledb.ConnectionPool, batchsize: int = 5000) -> Iterator[pd.DataFrame]:
    """
        Stream query results as DataFrames, one DataFrame per fetched batch.

        Args:
        -----
            query: SQL query string.
            pool: Oracle connection pool.
            batchsize: Number of rows per batch.

        Yields:
        -------
            DataFrame chunks.
    """
    with pool.acquire() as connection:
        cursor = connection.cursor()
        cursor.execute(query)
        columns = [desc[0] for desc in cursor.description]

        while True:
            rows = cursor.fetchmany(batchsize)
            if not rows:
                break
            yield pd.DataFrame(rows, columns=columns)


def call_query(query: str, pool: oracledb.ConnectionPool, timer: int, batchsize: int = 5000) -> Optional[pd.DataFrame]:
    """
        Fetch a full query result by iterating over `fetch_query_iter` and concatenating chunks.

        Notes:
            - Preserves your throttle logic: sleeps `timer` seconds per fetched chunk.
            - Returns None if no chunks were returned.

        Args:
            query: SQL query string.
            pool: Oracle connection pool.
            timer: Sleep duration (seconds) after each chunk.
            batchsize: Number of rows per fetched chunk.

        Returns:
            Concatenated DataFrame, or None if the query returned no rows.
    """
    chunks = []
    for df_chunk in tqdm(fetch_query_iter(query, pool, batchsize=batchsize)):
        chunks.append(df_chunk)
        time.sleep(timer)

    if chunks:
        return pd.concat(chunks, ignore_index=True)
    return None

def log_persistence(
    logger: Logger, 
    entity: str, 
    chunk_df: pd.DataFrame,
    chunk_id: int,
    offset: int,
    out_path: str    
) -> None:
    logger.log_info({
        "event": "chunk_saved", "entity": entity,
        "chunk_id": chunk_id, "offset": offset, 
        "rows": len(chunk_df), "path": str(out_path),
    })

def log_errors(
    logger: Logger,
    entity: str,
    err: oracledb.DatabaseError,
    chunk_id: int,
    attempt: int,
) -> None:
    logger.log_info({
        "event": "oracle_error",
        "entity": entity,
        "chunk_id": chunk_id,
        "error_code": err.code,
        "error_message": err.message,
        "attempt": attempt + 1,
    })


# ----------------------------------------------------------------- #
# ---------------------- EXTRACTOR INTERFACE ---------------------- #
# ----------------------------------------------------------------- #

class Extractor(ConfigInterface):
    """
        Wrapper class to perform the necessary extraction tasks for the project.

        Args:
        -----
            config_dir: Base directory for configuration.
            config_defaults: Defaults passed to ConfigInterface.
    """

    def __init__(self, config_dir: str, config_defaults: dict, logger: Optional[bool] = False) -> None:
        super().__init__(config_dir, config_defaults)

        self.pool: Optional[oracledb.ConnectionPool] = None

        # --- environment / credentials
        # .................................

        # Keep your helper mapping unchanged
        self.query_helper = dict(QUERY_HELPER)

    # --------- CONTEXT MANAGER ---------
    def __enter__(self) -> "Extractor":
        """Open the shared pool connection for a `with Extractor(...) as ex:` usage."""
        self.create_pool()
        return self

    def __exit__(self, *_exc) -> None:
        """Close pool on context manager exit."""
        self.close()

    # --------- POOL CONNECTION ---------
    def create_pool(self) -> None:
        """
            Create a shared Oracle pool connection.

            Note:
                Pool parameters could be moved to config, but kept inline to preserve behavior.
        """
        self.pool = oracledb.create_pool(
            user=self.user,
            password=self.pw,
            dsn=self.dsn,
            port=self.port,
            min=0,
            max=4,
            increment=1,
            expire_time=4,
        )
        if self.logger:
            self.logger.log_info({
                "event": "pool_created",
                "dsn": self.dsn,
                "max": 4,
            })

    def close(self) -> None:
        """Close and clear the pool if it exists."""
        if self.pool is not None:
            self.pool.close()
            self.pool = None

    # ----------------------------------------------------------------- #
    # ------------------------- QUERY METHODS ------------------------- #
    # ----------------------------------------------------------------- #
    def fetch_biopsy_paginated(
        self,
        db_origin: Optional[str] = "hsp",
        start_year: Optional[int] = 2014,
        timer: Optional[int] = 1,
        chunk_size: int = 50000,
        max_retries: int = 3,
        verbose: Optional[bool] = False,
    ) -> None:
        """
            Fetch biopsy data (paginated) and persist chunks as parquet.

            Notes:
                - Uses counter query to get total rows per procedure.
                - Skips already-persisted chunks using *_part{chunk_id}.parquet detection.
                - Retries on Oracle errors; recreates pool on known session-loss codes.

            Args:
            -----
                db_origin: Options are {'hsp', 'psc'}.
                timer: Pause (seconds) between chunk fetches inside `call_query`.
                chunk_size: Page size for persistence blocks.
                max_retries: Max attempts per chunk.
                verbose: Print progress details.
        """
        if self.logger:
            self.logger.log_info({
                "event": "fetch_start", "entity": "biopsy",
                "db_origin": db_origin, "chunk_size": chunk_size,
            })
        
        FNAME_PREFIX = f"biopsy_table_{db_origin.lower()}"
        Q_FUNCTION = self.query_helper["biopsy"]["content"]
        Q_COUNTER = self.query_helper["biopsy"]["counter"]
        base_path = self.extract_path.joinpath(self.extract_folders["biopsy"])

        for cur_procedure in LIST_PROCEDURES:
            if verbose:
                print(f"Counting total rows for biopsy {db_origin}, procedure {cur_procedure} ...")
            total_rows_df = perform_query(Q_COUNTER(cur_procedure, db_origin, start_year), self.pool)
            total_rows = int(total_rows_df.iloc[0, 0])
            if total_rows == 0:
                continue
            if verbose: print(f"Total rows to fetch: {total_rows}")

            exam_prefix = f"{FNAME_PREFIX}_{cur_procedure}"
            completed_chunks = {
                int(f.name.split("_part")[1].split(".")[0])
                for f in base_path.glob(f"{exam_prefix}_part*.parquet")
            }

            if self.logger:
                self.logger.log_info({
                    "event": "row_count", "entity": "biopsy",
                    "procedure": cur_procedure, "total_rows": total_rows,
                })

            offset = 0
            chunk_id = 0
            while offset < total_rows:
                if chunk_id in completed_chunks:
                    if verbose:
                        print(f"Skipping existing chunk {chunk_id}")
                    offset += chunk_size
                    chunk_id += 1
                    continue

                for attempt in range(max_retries):
                    try:
                        paginated_query = Q_FUNCTION(cur_procedure, db_origin, start_year, offset=offset, limit=chunk_size)
                        chunk_df = call_query(paginated_query, self.pool, timer)

                        out_path = base_path / f"{exam_prefix}_part{chunk_id}.parquet"
                        chunk_df.to_parquet(out_path)
                        if self.logger:
                            log_persistence(
                                logger=self.logger, entity="biopsy", chunk_df=chunk_df, 
                                chunk_id=chunk_id, offset=offset, out_path=out_path
                            )

                        if verbose:
                            print(f"Saved chunk {chunk_id} to {out_path.name}")
                        break

                    except oracledb.DatabaseError as e:
                        err, = e.args
                        if self.logger:
                            log_errors(self.logger, 'biopsy', err, chunk_id, attempt)
                        print(f"Error on chunk {chunk_id}: {err.message}")
                        if err.code in (28, 3113, 1080):
                            print("Reconnecting pool after session loss...")
                            self.create_pool()
                        else:
                            raise ValueError("")

                    except UnicodeDecodeError as e:
                        print(f"Skipping chunk {chunk_id} due to decode error: {e}")
                        break

                    time.sleep(1)
                else:
                    print(f"Max retries exceeded for chunk {chunk_id}, skipping.")
                    break

                offset += chunk_size
                chunk_id += 1
        
        if self.logger:
            self.logger.log_info({
                "event": "fetch_end", "entity": "biopsy", "db_origin": db_origin,
            })

    def fetch_anamnesis_paginated(
        self,
        db_origin: Optional[str] = "hsp",
        timer: Optional[int] = 1,
        chunk_size: int = 50000,
        max_retries: int = 3,
        verbose: Optional[bool] = False,
    ) -> None:
        """
            Fetch anamnesis data (paginated) and persist chunks as parquet.

            Args:
            -----
                db_origin: ...
                timer: Pause (seconds) between chunk fetches inside `call_query`.
                chunk_size: Page size for persistence blocks.
                max_retries: Max attempts per chunk.
                verbose: Print progress details.
        """
        if self.logger:
            self.logger.log_info({
                "event": "fetch_start", "entity": "anamnesis",
                "db_origin": db_origin, "chunk_size": chunk_size,
            })

        FNAME_PREFIX = f"anamnesis_table_{db_origin}"
        Q_FUNCTION = self.query_helper["anamnesis"][db_origin]["content"]
        Q_COUNTER = self.query_helper["anamnesis"][db_origin]["counter"]
        base_path = self.extract_path.joinpath(self.extract_folders["anamnesis"])

        if verbose: print(f"Counting total rows for anamnesis {db_origin}...")
        total_rows_df = perform_query(Q_COUNTER(db_origin), self.pool)
        total_rows = int(total_rows_df.iloc[0, 0])
        if verbose: print(f"Total rows to fetch: {total_rows}")

        completed_chunks = {
            int(f.name.split("_part")[1].split(".")[0])
            for f in base_path.glob(f"{FNAME_PREFIX}_part*.parquet")
        }

        if self.logger:
            self.logger.log_info({
                "event": "row_count", "entity": "anamnesis",
                "procedure": 'anamnesis', "total_rows": total_rows,
            })

        offset = 0
        chunk_id = 0
        while offset < total_rows:
            if chunk_id in completed_chunks:
                if verbose:
                    print(f"Skipping existing chunk {chunk_id}")
                offset += chunk_size
                chunk_id += 1
                continue

            for attempt in range(max_retries):
                try:
                    paginated_query = Q_FUNCTION(db_origin, offset=offset, limit=chunk_size)
                    chunk_df = call_query(paginated_query, self.pool, timer)

                    out_path = base_path / f"{FNAME_PREFIX}_part{chunk_id}.parquet"
                    chunk_df.to_parquet(out_path)
                    if self.logger:
                        log_persistence(
                            logger=self.logger, entity="anamnesis", chunk_df=chunk_df, 
                            chunk_id=chunk_id, offset=offset, out_path=out_path
                        )

                    if verbose:
                        print(f"Saved chunk {chunk_id} to {out_path.name}")
                    break

                except oracledb.DatabaseError as e:
                    err, = e.args
                    if self.logger:
                        log_errors(self.logger, 'anamnesis', err, chunk_id, attempt)
                    print(f"Error on chunk {chunk_id}: {err.message}")
                    if err.code in (28, 3113, 1080):
                        print("Reconnecting pool after session loss...")
                        self.create_pool()
                    else:
                        raise ValueError("Connection lost.")

                except UnicodeDecodeError as e:
                    print(f"Skipping chunk {chunk_id} due to decode error: {e}")
                    break

                time.sleep(1)
            else:
                print(f"Max retries exceeded for chunk {chunk_id}, skipping.")
                break

            offset += chunk_size
            chunk_id += 1

        if self.logger:
            self.logger.log_info({
                "event": "fetch_end", "entity": "anamnesis", "db_origin": db_origin,
            })

    def fetch_mammograms_paginated(
        self,
        db_origin: Optional[str] = "hsp",
        start_year: Optional[int] = 2014,
        timer: Optional[int] = 1,
        chunk_size: int = 5000,
        max_retries: int = 3,
        verbose: Optional[bool] = False,
    ) -> None:
        """
            Fetch data on mammogram reports (paginated) and persist chunks as parquet.

            Important:
                Your current logic only computes `total_rows` when `verbose=True`.
                I preserved that behavior, but it can break when verbose=False.

            Args:
            -----
                db_origin: ...
                timer: Pause (seconds) between chunk fetches inside `call_query`.
                chunk_size: Page size for persistence blocks.
                max_retries: Max attempts per chunk.
                verbose: Print progress details (also controls total_rows computation, as per original code).
        """
        if self.logger:
            self.logger.log_info({
                "event": "fetch_start", "entity": "mammograms",
                "db_origin": db_origin, "chunk_size": chunk_size,
            })

        FNAME_PREFIX = f"mammograms_table_{db_origin}"
        Q_FUNCTION = self.query_helper["mammogram"][db_origin]["content"]
        Q_COUNTER = self.query_helper["mammogram"][db_origin]["counter"]
        base_path = self.extract_path.joinpath(self.extract_folders["mammogram_exams"])

        for current_exam in LIST_PROCEDURES_MAMMOGRAM:
            if verbose: print(f"Exam: {current_exam}")
            total_rows_df = perform_query(Q_COUNTER(current_exam, db_origin, start_year), self.pool)
            total_rows = int(total_rows_df.iloc[0, 0])
            if verbose: print(f"Total rows (exams): {total_rows}")

            exam_prefix = f"{FNAME_PREFIX}_{current_exam.lower().replace(' ', '_').replace(':','_').replace('.','_')}"
            completed_chunks = {
                int(f.name.split("_part")[1].split(".")[0])
                for f in base_path.glob(f"{exam_prefix}_part*.parquet")
            }

            if self.logger:
                self.logger.log_info({
                    "event": "row_count", "entity": "mammograms",
                    "procedure": current_exam, "total_rows": total_rows,
                })

            offset = 0
            chunk_id = 0
            while offset < total_rows:
                if chunk_id in completed_chunks:
                    if verbose:
                        print(f"Skipping chunk {chunk_id} (already exists)")
                    offset += chunk_size
                    chunk_id += 1
                    continue

                for attempt in range(max_retries):
                    try:
                        paginated_query = Q_FUNCTION(current_exam, db_origin, start_year, offset=offset, limit=chunk_size)
                        chunk_df = call_query(paginated_query, self.pool, timer)
                        if chunk_df is None:
                            break

                        out_path = base_path / f"{exam_prefix}_part{chunk_id}.parquet"
                        chunk_df.to_parquet(out_path)
                        if self.logger:
                            log_persistence(
                                logger=self.logger, entity="mammograms", chunk_df=chunk_df, 
                                chunk_id=chunk_id, offset=offset, out_path=out_path
                            )

                        if verbose:
                            print(f"Saved chunk {chunk_id} at {out_path.name}")
                        break

                    except oracledb.DatabaseError as e:
                        err, = e.args
                        if self.logger:
                            log_errors(self.logger, 'mammograms', err, chunk_id, attempt)
                        print(f"Error on chunk {chunk_id}: {err.message}")
                        if err.code in (28, 3113, 1080):
                            print("Reconnecting after session loss...")
                            self.create_pool()
                        else:
                            raise ValueError("Connection lost.")

                    except UnicodeDecodeError as e:
                        print(f"Skipping chunk {chunk_id} due to decode error: {e}")
                        break

                    time.sleep(1)
                else:
                    print(f"Max retries exceeded for chunk {chunk_id}, skipping.")
                    break

                offset += chunk_size
                chunk_id += 1
        
        if self.logger:
            self.logger.log_info({
                "event": "fetch_end", "entity": "mammograms", "db_origin": db_origin,
            })

    def fetch_cohort_from_mammograms(
        self,
        cohort_type: Optional[str] = "person",
        db_origin: Optional[str] = "hsp",
        start_year: Optional[int] = 2014,
        timer: Optional[int] = 1,
        chunk_size: int = 5000,
        max_retries: int = 3,
        verbose: Optional[bool] = False,
    ) -> None:
        """
            Fetch cohort data derived from mammogram exams (paginated) and persist chunks as parquet.

            Args:
            -----
                cohort_type: ...
                db_origin: ...
                timer: Pause (seconds) between chunk fetches inside `call_query`.
                chunk_size: Page size for persistence blocks.
                max_retries: Max attempts per chunk.
                verbose: Print progress details (also controls total_rows computation, as per original code).
        """
        if self.logger:
            self.logger.log_info({
                "event": "fetch_start", "entity": f"cohort({cohort_type})",
                "db_origin": db_origin, "chunk_size": chunk_size,
            })

        FNAME_PREFIX = f"{cohort_type}_from_mammograms_{db_origin}"
        Q_FUNCTION = self.query_helper[cohort_type][db_origin]["content"]
        Q_COUNTER = self.query_helper["mammogram"][db_origin]["counter"]
        base_path = self.extract_path.joinpath(self.extract_folders["user_person_data"])

        for current_exam in LIST_PROCEDURES_MAMMOGRAM:
            if verbose: print(f"Exam: {current_exam}")
            total_rows_df = perform_query(Q_COUNTER(current_exam, db_origin, start_year), self.pool)
            total_rows = int(total_rows_df.iloc[0, 0])
            if verbose: print(f"Total rows (exams): {total_rows}")

            exam_prefix = f"{FNAME_PREFIX}_{current_exam.lower().replace(' ', '_').replace(':','_').replace('.','_')}"
            completed_chunks = {
                int(f.name.split("_part")[1].split(".")[0])
                for f in base_path.glob(f"{exam_prefix}_part*.parquet")
            }

            if self.logger:
                self.logger.log_info({
                    "event": "row_count", "entity": f"cohort({cohort_type})",
                    "procedure": "cohort", "total_rows": total_rows,
                })

            offset = 0
            chunk_id = 0
            while offset < total_rows:
                if chunk_id in completed_chunks:
                    if verbose:
                        print(f"Skipping chunk {chunk_id} (already exists)")
                    offset += chunk_size
                    chunk_id += 1
                    continue

                for attempt in range(max_retries):
                    try:
                        paginated_query = Q_FUNCTION(current_exam, db_origin, start_year, offset=offset, limit=chunk_size)
                        chunk_df = call_query(paginated_query, self.pool, timer)
                        if chunk_df is None:
                            break

                        out_path = base_path / f"{exam_prefix}_part{chunk_id}.parquet"
                        chunk_df.to_parquet(out_path)
                        if self.logger:
                            log_persistence(
                                logger=self.logger, entity=f"cohort({cohort_type})", chunk_df=chunk_df, 
                                chunk_id=chunk_id, offset=offset, out_path=out_path
                            )

                        if verbose:
                            print(f"Saved chunk {chunk_id} at {out_path.name}")
                        break

                    except oracledb.DatabaseError as e:
                        err, = e.args
                        if self.logger:
                            log_errors(self.logger, f'cohort({cohort_type})', err, chunk_id, attempt)
                        print(f"Error on chunk {chunk_id}: {err.message}")
                        if err.code in (28, 3113, 1080):
                            print("Reconnecting after session loss...")
                            self.create_pool()
                        else:
                            raise

                    time.sleep(1)
                else:
                    print(f"Max retries exceeded for chunk {chunk_id}, skipping.")
                    break

                offset += chunk_size
                chunk_id += 1
        
        if self.logger:
            self.logger.log_info({
                "event": "fetch_end", "entity": f"cohort({cohort_type})", "db_origin": db_origin,
            })

    def fetch_prescriptions(self) -> None:
        pass

    def fetch_blood_samples(self) -> None:
        pass
