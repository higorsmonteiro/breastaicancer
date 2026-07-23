"""
mlflow_setup.py
---------------
Call `configure_mlflow()` once at the top of any CLI entrypoint
(e.g. in functions.py, before tuning_api / cv_training_tfidf_api).
 
You can also set these via environment variables instead of calling
configure_mlflow() explicitly — MLflow will pick them up automatically:
 
    export MLFLOW_TRACKING_URI=http://localhost:5000   # or a file path
    export MLFLOW_EXPERIMENT_NAME=hapcancer
"""
import mlflow
 
def configure_mlflow(
    tracking_uri: str = "mlruns",          # local folder; swap for remote URI
    experiment_name: str = "hapcancer",
) -> None:
    """
        Configure MLflow tracking URI and set (or create) the experiment.
        To be called this once before any training/tuning API function.
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    print(f"[MLflow] tracking_uri={tracking_uri!r}  experiment={experiment_name!r}")