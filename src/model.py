import pandas as pd
import numpy as np
from typing import Tuple

from sklearn.metrics import mean_absolute_error


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
