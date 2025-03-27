import pandas as pd
from sklearn.metrics import mean_absolute_error
from feature_engine.timeseries.forecasting import WindowFeatures

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

    return mae_df
