import pandas as pd
import numpy as np
from typing import Tuple, Optional

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

from feature_engine.datetime import DatetimeFeatures
from feature_engine.imputation import DropMissingData
from feature_engine.selection import DropFeatures
from feature_engine.timeseries.forecasting import (
    LagFeatures,
    WindowFeatures,
)
from feature_engine.encoding import OrdinalEncoder


class FeatureEngineerByBA(BaseEstimator, TransformerMixin):
    """
    Applies a given transformer to each unique {ba_code} group in the dataset.

    This class loops over the unique values of {ba_code} in the dataset and applies
    the provided transformer independently to each group. The fitted transformers
    are stored for later transformations.

    Attributes:
        transformer: The transformation to be applied.
        fitted_transformers: Stores fitted transformers per {ba_code}.
    """

    def __init__(self, transformer: TransformerMixin) -> None:
        """
        Initializes the FeatureEngineerByBA with a given transformer.

        Args:
            transformer: The transformer to be applied to each BA time series.
        """
        self.transformer = transformer
        self.fitted_transformers: dict[str, TransformerMixin] = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "FeatureEngineerByBA":
        """
        Fits the transformer to each {ba_code} time series.

        Args:
            X: Input dataset.
            y: Target variable.

        Returns:
            The fitted instance.
        """
        for ba_code in X["ba_code"].unique():
            tmp = X.loc[X["ba_code"] == ba_code, :].copy()
            self.fitted_transformers[ba_code] = self.transformer.fit(tmp)
        return self

    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Transforms the dataset using the fitted transformers for each {ba_code} time series.

        Args:
            X: Input dataset.
            y: Target variable.

        Returns:
            The transformed dataset.
        """
        output = pd.DataFrame()
        for ba_code in X["ba_code"].unique():
            tmp = X.loc[X["ba_code"] == ba_code, :].copy()
            tmp = self.fitted_transformers[ba_code].transform(tmp)
            tmp["ba_code"] = ba_code
            output = pd.concat([output, tmp])
        return output


class ScaleByBA(BaseEstimator, TransformerMixin):
    """
    Applies a given scaler to demand-related columns for each {ba_code} time series.

    This class loops over the unique values of {ba_code} in the dataset and applies
    the provided scaler to columns related to demand.

    Attributes:
        scaler (TransformerMixin): The scaler to be applied.
        fitted_scalers (dict[str, TransformerMixin]): Stores fitted scalers per {ba_code}.
    """

    def __init__(self, scaler: TransformerMixin) -> None:
        """
        Initializes the ScaleByBA with a given scaler.

        Args:
            scaler: The scaler to be applied.
        """
        self.scaler = scaler
        self.fitted_scalers: dict[str, TransformerMixin] = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "ScaleByBA":
        """
        Fits the scaler to demand-related columns for each {ba_code} time series.

        Args:
            X: Input dataset.
            y: Target variable.

        Returns:
            The fitted instance.
        """
        demand_cols = X.filter(like="demand").columns
        for ba_code in X["ba_code"].unique():
            tmp = X.loc[X["ba_code"] == ba_code, demand_cols].copy()
            self.fitted_scalers[ba_code] = (
                self.scaler.__class__().set_output(transform="pandas").fit(tmp)
            )
        return self

    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Transforms demand-related columns using the fitted scalers for each {ba_code} time series.

        Args:
            X: Input dataset containing a {ba_code} column.
            y: Target variable.

        Returns:
            The transformed dataset.
        """
        X_ = X.copy()
        output = pd.DataFrame()
        demand_cols = X.filter(like="demand").columns
        for ba_code in X["ba_code"].unique():
            tmp = X.loc[X["ba_code"] == ba_code, demand_cols].copy()
            tmp = self.fitted_scalers[ba_code].transform(tmp)
            output = pd.concat([output, tmp])
        X_[demand_cols] = output[demand_cols]
        return X_

def make_pipeline() -> Pipeline:
    """
    Helper function to streamline preprocessing pipeline.
    """
    dtf = DatetimeFeatures(
        variables=["datetime"],
        features_to_extract=[
            "month",
            "week",
            "day_of_week",
            "day_of_month",
            "weekend",
        ],
        drop_original=False,
    )

    lag_transformer = FeatureEngineerByBA(
        LagFeatures(
            variables=["demand"],
            periods=[1, 2, 3, 4, 5, 6, 7, 30, 180, 365],
            drop_original=False,
        )
    )

    window_transformer = FeatureEngineerByBA(
        WindowFeatures(
            variables=["demand"],
            window=[3, 7, 14],
            freq=None,
            functions=["mean", "std", "max", "min"],
            missing_values="ignore",
        )
    )

    minmax_scaler = ScaleByBA(MinMaxScaler())

    # Introduce missing date when using lags and windows so need to drop these NaNs
    drop_missing = DropMissingData()

    # Ordinal encoding for BA feature
    ordinal_enc = OrdinalEncoder(variables=["ba_code"], encoding_method="arbitrary")

    # Also drop the target from the training set
    drop_target = DropFeatures(features_to_drop=["demand", "datetime"])

    pipe = Pipeline(
        [
            ("datetime", dtf),
            ("lags", lag_transformer),
            ("windf", window_transformer),
            ("minmax_scaling", minmax_scaler),
            ("drop_missing", drop_missing),
            ("ordinal_enc", ordinal_enc),
            ("drop_target", drop_target),
        ]
    )

    return pipe

def make_baseline_predictions(
    test_data: pd.DataFrame,
) -> Tuple[pd.DataFrame, float, list]:
    """
    Wrapper to return a batch of predictions using the best baseline model to assess
    performance.

    Args:
        test_data: data that we are using for evaluation

    Returns:
        Tuple containing the predictions, the average MAE over all BAs, and the individual MAE
        for each BA.
    """
    def _get_lag_columns(
        df: pd.DataFrame, lag: int, target_columns: list[str]
        ) -> pd.DataFrame:

        lag_columns = {
            f"lag{lag}_{col}".replace("ba_", ""): df[col].shift(periods=lag)
            for col in target_columns
        }

        # Convert the dictionary to a DataFrame
        lag_df = pd.DataFrame(lag_columns)

        # Concatenate the original and new DataFrame along the column axis
        df = pd.concat([df, lag_df], axis=1)

        return df

    # Append lag column for each BA
    predictions = _get_lag_columns(
        data=test_data, lag=1, target_columns=test_data.filter(like="ba_").columns
    )

    predictions.dropna(inplace=True)

    # Number of BAs
    num_bas = test_data.filter(like="ba_").shape[1]

    maes = [
        mean_absolute_error(predictions.iloc[:, i], predictions.iloc[:, i + num_bas])
        for i in range(num_bas)
    ]

    average_mae = np.mean(maes)

    return predictions, average_mae, maes


def forwardfill_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    df = df.replace(-1, np.nan).ffill()
    return df
