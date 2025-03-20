import pandas as pd
import hopsworks
import src.config as config

from pathlib import Path
from datetime import timedelta
from typing import Tuple

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from hsfs.feature_store import FeatureStore
from hopsworks.project import Project

from src.paths import PARENT_DIR
from src.data import prepare_feature_store_data_for_training

"""
TODO: 
    - Clean up config for hopsworks by adding in new script holding Hopsworks metadata
"""

def get_hopsworks_project() -> Project:

    return hopsworks.login(
        project=config.HOPSWORKS_PROJECT_NAME,
        api_key_value=config.HOPSWORKS_API_KEY,
        )


def get_feature_store() -> FeatureStore:
    
    project = get_hopsworks_project()
    return project.get_feature_store()


def load_batch_of_features_for_inference(current_date: str) -> pd.DataFrame:

    # Adding some padding so that we don't lose any data
    to_date = current_date + timedelta(days=2)
    from_date = current_date - timedelta(days=config.DAYS_HISTORICAL + 14)

    feature_store = get_feature_store()

    feature_view = feature_store.get_feature_view(
        name=config.FEATURE_VIEW_NAME, 
        version=config.FEATURE_VIEW_VERSION
    )

    data = feature_view.get_batch_data(
        start_time=from_date,
        end_time=to_date,
    )

    return prepare_feature_store_data_for_training(data)


def load_model_from_registry(
    hopsworks_model_dir: str = "models",
    pipeline_name: str = "preprocessing_pipeline",
    model_name: str = "lgbm",
    ) -> Tuple[BaseEstimator, Pipeline]:
    """
    Loads saved model files from the Hopsworks model registry.

    Args:
        hopsworks_model_dir: name of folder holding the model artifact
        pipeline_name: name of pickled preprocessing pipeline
        model_name: name of pickled trained model

    Returns:
        (model, preprocessing_pipeline)
    """

    import shutil
    import os
    import joblib
    
    project = get_hopsworks_project()

    model_registry = project.get_model_registry()
    model_files = model_registry.get_model(
        name=config.MODEL_NAME,
        version=config.MODEL_VERSION,
    )

    model_dir = model_files.download()

    filepath = Path(model_dir) / f"{hopsworks_model_dir}.zip"
    extract_dir = PARENT_DIR / "downloaded_model_bundle"
    if not extract_dir.exists():
        os.mkdir(extract_dir)

    shutil.unpack_archive(filepath, extract_dir)

    preprocessing_pipeline = joblib.load(extract_dir / f"{pipeline_name}.pkl")
    model = joblib.load(extract_dir / f"{model_name}.pkl")

    return model, preprocessing_pipeline


def get_model_predictions(
    model: BaseEstimator,
    preprocessing_pipeline: Pipeline,
    X: pd.DataFrame,
    features_end: str,
) -> pd.DataFrame:
    """
    Uses {model} to make predictions of daily demand for the day after {features_end}
    using features {X}.

    Args:
        model: a fitted SKLearn model
        preprocessing pipeline: the pipeline used to train {model}
        X: the features for inference
        features_end: the date corresponding to the day before the prediction date,
            in format, e.g., '2025-03-15'

    Returns:
        A dataframe holding the predictions.

        Columns:
            ba_code: str
            predicted_demand: float64
            datetime: datetime64
    """
    features_end = pd.Timestamp(features_end)
    features_start = features_end - pd.offsets.Day(config.DAYS_HISTORICAL)

    # Filter to appropriate date range
    inference_data = X.loc[
        (X["datetime"] <= features_end) & (X["datetime"] >= features_start)
    ].copy()

    # Transform data with fitted pipeline
    inference_data_t = preprocessing_pipeline.transform(inference_data)

    predictions = model.predict(inference_data_t)

    ba_codes = inference_data["ba_code"].unique()

    predictions_df = pd.DataFrame(
        {"ba_code": ba_codes, "predicted_demand": predictions}
    )

    # Predictions are for the day after {features_end}
    predictions_df["datetime"] = features_end + pd.offsets.Day(1)

    return predictions_df.round()