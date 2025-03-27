import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_absolute_error
from feature_engine.timeseries.forecasting import WindowFeatures
from feature_engine.imputation import DropMissingData

from src.inference import load_batch_of_features, load_batch_of_predictions


def load_predictions_and_actual_values_from_feature_store(
    current_date: datetime.date, days_in_past: int = 30
    ) -> pd.DataFrame:
    """
    Loads predictions and actuals for last {days_in_past} days before {current_date} and combines 
    them into a single dataframe.

    Args:
        current_date: The date up to which we want the data to run to
        days_in_past: The number of days before {current_date} we want the data to run from

    Returns:
        A merged dataframe containing the predictions and actual demand values for each (date, BA) pair.
    """

    predictions = load_batch_of_predictions(current_date, days_in_past)
    actuals = load_batch_of_features(current_date, days_in_past)

    return predictions.merge(actuals, on=["datetime", "ba_code"], how="inner")


def get_mae_df(monitoring_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the MAE between the demand and the predicted demand
    over all BAs.

    Args:
        monitoring_df: a dataframe containing the date, ba_code,
            demand and predicted demand

    Returns:
        A dataframe containing the MAE for each date and the 7-day MAE moving average
    """

    df = (
        pd.DataFrame(
            monitoring_df.sort_values(by="datetime")
            .groupby("datetime")
            .apply(lambda x: mean_absolute_error(x["demand"], x["predicted_demand"]))
        )
        .reset_index()
        .rename(columns={0: "mae"})
    )

    df["datetime"] = pd.to_datetime(df["datetime"])
    df["datetime"] = df["datetime"].dt.tz_localize(None)

    window_transformer = WindowFeatures(
        variables=["mae"],
        window=[7],
        freq=None,
        functions=["mean"],
        missing_values="ignore",
    )

    mae_df = window_transformer.fit_transform(df)

    return DropMissingData().fit_transform(mae_df)
