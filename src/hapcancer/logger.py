import logging
import json
import pandas as pd
import datetime as dt
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, Any


@dataclass(frozen=True)
class RunLogPaths:
    """
    Container for all filesystem paths associated with a single run.

    Attributes:
        run_dir: Base directory for the run.
        metrics_jsonl: Path to the structured metrics log (JSON Lines).
        text_log: Path to the human-readable text log.
    """
    run_dir: Path
    metrics_jsonl: Path
    text_log: Path


class Logger:
    """
        Lightweight run-based logger with:
          - a human-readable text log (run.log)
          - a structured JSONL metrics log (metrics.jsonl)

        This class is designed for ETL / data extraction runs where:
          - each execution has its own run directory
          - metrics and progress information should be machine-readable
          - logs should *not* propagate to the root logger (to avoid duplication)

        Notes:
            - Uses a single logger instance (per module name).
            - Ensures handlers are not duplicated across multiple instantiations.
            - Metrics are logged as JSON strings (one object per line).
    """

    def __init__(
        self,
        base_dir: Union[Path, str] = "runs",
        run_name: Optional[str] = None,
        overwrite: bool = True,
        level: int = logging.INFO,
    ) -> None:
        """
            Initialize a run logger.

            Args:
            -----
                base_dir: Root directory where run folders will be created.
                run_name: Optional explicit run name. If None, a timestamp + UUID suffix is used.
                overwrite: Whether to overwrite existing log files if they exist.
                level: Logging level (e.g., logging.INFO, logging.DEBUG).
        """
        self.base_dir = Path(base_dir)

        if run_name is None:
            ts = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            short = uuid.uuid4().hex[:6]
            run_name = f"{ts}_{short}"

        self.run_dir = self.base_dir / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.paths = RunLogPaths(
            run_dir=self.run_dir,
            metrics_jsonl=self.run_dir / "metrics.jsonl",
            text_log=self.run_dir / "run.log",
        )

        # -- optionally reset logs
        if overwrite:
            for p in (self.paths.metrics_jsonl, self.paths.text_log):
                if p.exists():
                    p.unlink()

        # -- logger configuration
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level)
        self.logger.propagate = False  # prevent double logging via root

        self._ensure_handlers()

    def _ensure_handlers(self) -> None:
        """
            Attach file handlers if they are not already present.

            This avoids duplicated logs when the Logger is instantiated multiple times
            within the same Python process.
        """
        # -- avoid adding duplicate handlers
        if any(
            isinstance(h, logging.FileHandler)
            and getattr(h, "baseFilename", "") == str(self.paths.text_log)
            for h in self.logger.handlers
        ):
            return

        # -- human-readable text log
        text_handler = logging.FileHandler(self.paths.text_log, mode="a", encoding="utf-8")
        text_handler.setLevel(logging.INFO)
        text_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        self.logger.addHandler(text_handler)

        # -- structured metrics log (JSONL)
        metrics_handler = logging.FileHandler(self.paths.metrics_jsonl, mode="a", encoding="utf-8")
        metrics_handler.setLevel(logging.INFO)
        metrics_handler.setFormatter(logging.Formatter("%(message)s"))
        self.logger.addHandler(metrics_handler)

    def log_info(self, dictionary_info: dict) -> None:
        """
            Log a structured dictionary as a JSON line.

            Intended use:
                - metrics
                - counters
                - progress checkpoints
                - run metadata

            Args:
            -----
                dictionary_info: Dictionary to be serialized and logged as JSON.
        """
        self.logger.info(json.dumps(dictionary_info))
