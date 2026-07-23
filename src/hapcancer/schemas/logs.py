from pydantic import BaseModel, Field
from typing import Optional, Iterable, List, Union

class TuningBCEParams(BaseModel):
    dropout: float = Field(alias="dropout")
    depth: int = Field(alias="depth")
    layer_0_units: int = Field(alias="layer_0_units")
    weight_decay: float = Field(alias="weight_decay")
    learning_rate: float = Field(alias="learning_rate")
    optimizer: str = Field(alias="optimizer")

    model_config = {"populate_by_name": True, "extra": "ignore"}

class TuningBCELog(BaseModel):
    trial_number: int = Field(alias="trial_number")
    init_date: str = Field(alias="init_date")
    load_id: str = Field(alias="load_id")
    dataset_name: str = Field(alias="dataset_name")
    target_year: int = Field(alias="target_year")
    total_epochs_per_trial: int = Field(alias="total_epochs_per_trial")
    result: float = Field(alias="result")
    params: TuningBCEParams = Field(alias="params")

    model_config = {"populate_by_name": True, "extra": "ignore"}

class TrainingMetricsLog(BaseModel):
    cv: int = Field(alias="CV")
    loss: float = Field(alias="Loss")
    auroc: float = Field(alias="AUROC")
    average_precision: float = Field(alias="Average Precision")

    model_config = {"populate_by_name": True, "extra": "ignore"}

class TrainingBCELog(BaseModel):
    config_dir_path: str = Field(alias="config_dir_path")
    split_config_name: str = Field(alias="split_config_name")
    training_config_name: str = Field(alias="training_config_name")
    epoch: int = Field(alias="epoch")
    target_year: int = Field(alias="target year")
    training_metrics: TrainingMetricsLog = Field(alias="training metrics")
    validation_metrics: TrainingMetricsLog = Field(alias="validation metrics")

    model_config = {"populate_by_name": True, "extra": "ignore"}
