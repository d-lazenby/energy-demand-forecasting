import calendar
import holidays
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os

import requests
import pandas as pd
from tqdm import tqdm

from typing import Tuple

from sklearn.pipeline import Pipeline

from src.paths import PARENT_DIR, RAW_DATA_DIR, TRANSFORMED_DATA_DIR

load_dotenv(PARENT_DIR / ".env")
EIA_API_KEY = os.environ["EIA_API_KEY"]


def download_new_batch_of_data(year: int, month: int) -> pd.DataFrame:
    """
    Downloads one batch of raw data from EIA for {year}, {month}. Note that 
    length of dataset will vary depending on how far through {month} the user
    is when they call the function.

    Args:
        year: Year of data collection
        month: Month of data collection

    Returns:
        Dataframe of raw data
    """

    # Need the number of days in the current (year, month)
    _, num_days = calendar.monthrange(year, month)

    # To ensure we don't return an empty response
    from_date = datetime(year, month, 1).date() - timedelta(days=30)
    from_year, from_month = from_date.year, from_date.month

    URL = (
        "https://api.eia.gov/v2/electricity/rto/daily-region-data/data/"
        "?frequency=daily"
        "&data[0]=value"
        "&facets[timezone][]=Eastern"
        "&facets[type][]=D"
        f"&start={from_year}-{from_month:02d}-01"
        f"&end={year}-{month:02d}-{num_days}"
        "&sort[0][column]=period"
        "&sort[0][direction]=desc"
        "&offset=0"
        "&length=5000"
        f"&api_key={EIA_API_KEY}"
    )

    response = requests.get(url=URL).json()["response"]["data"]
    data = pd.DataFrame(response)

    # Tidies dataframe and saves to csv
    data = data[["period", "respondent", "value"]].copy()
    data.rename(
        columns={
            "period": "datetime",
            "value": "demand",
            "respondent": "ba_code",
        },
        inplace=True,
    )

    data["datetime"] = pd.to_datetime(data["datetime"])

    return data


def download_and_save_raw_data(year: int, month: int):
    """
    Downloads one month of raw data from EIA for {year}, {month}
    and saves to CSV.
    
    Args:
        year: Year of data collection
        month: Month of data collection
    """
    file_path = RAW_DATA_DIR / f"demand_{year}_{month}.csv"
    if file_path.exists():
        print(f"File demand_{year}_{month}.csv exists locally already, try next URL")
    else:
        data = download_new_batch_of_data(year=year, month=month)

        data.to_csv(file_path, index=False)
        print(
            f"Data for {year}_{month} successfully downloaded to demand_{year}_{month}.csv"
        )


def fill_missing_demand_values(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures that all BAs have demand values for the full range of dates.
    Adds in entries for any missing dates and fills these with demand = -1.

    Args:
        raw_df: dataframe of concatenated demand for all BAs

    Returns:
        extended dataframe
    """
    ba_codes = raw_df["ba_code"].unique()
    raw_df["datetime"] = pd.to_datetime(raw_df["datetime"])

    # Make a full range of datetime indexes.
    full_range = pd.date_range(
        raw_df["datetime"].min(), raw_df["datetime"].max(), freq="D"
    )

    # For storing the modified time series
    output = pd.DataFrame()

    for ba_code in tqdm(ba_codes):

        # Temporary df holding demand with current ba code
        tmp = raw_df.loc[raw_df["ba_code"] == ba_code, ["datetime", "demand"]]
        tmp.set_index("datetime", inplace=True)
        tmp = tmp[~tmp.index.duplicated(keep="first")]
        tmp.index = pd.DatetimeIndex(tmp.index)

        # Fill missing values with -1 for the moment for flexibility in modelling stage
        tmp = tmp.reindex(full_range, fill_value=-1)

        # Add back in the location id
        tmp["ba_code"] = ba_code

        output = pd.concat([output, tmp])

    # Move demand date from index to column
    output = output.reset_index().rename(columns={"index": "datetime"})

    return output


def prepare_raw_data_for_feature_store() -> pd.DataFrame:
    """
    Prepares raw data for feature store by concatenating raw data and inserting
    missing demand values with -1. Saves modified dataset to CSV.
    """
    concat_demand = pd.DataFrame(columns=["datetime", "ba_code", "demand"])

    for file_path in RAW_DATA_DIR.glob("*.csv"):
        with open(file_path, "rb"):
            tmp = pd.read_csv(file_path)
            concat_demand = pd.concat([concat_demand, tmp])

    # To deal with downcasting when filling NaNs
    concat_demand["demand"] = concat_demand["demand"].astype(int)
    concat_demand["datetime"] = pd.to_datetime(concat_demand["datetime"])

    output = fill_missing_demand_values(concat_demand)

    # For annotating the file name
    min_month, min_year = (
        output["datetime"].min().month,
        output["datetime"].min().year,
    )

    max_month, max_year = (
        output["datetime"].max().month,
        output["datetime"].max().year,
    )

    output.to_csv(
        TRANSFORMED_DATA_DIR
        / f"ts_tabular_{min_year}_{min_month}_to_{max_year}_{max_month}.csv",
        index=False,
    )

    print(
        f"Data transformed and saved at",
        TRANSFORMED_DATA_DIR
        / f"ts_tabular_{min_year}_{min_month}_to_{max_year}_{max_month}.csv",
    )

    return output


def fetch_batch_raw_data(from_date: datetime, to_date: datetime) -> pd.DataFrame:
    """
    Downloads raw data between {from_date} and {to_date}.

    Args:
        from_date: date that we want the data to range from
        to_date: date that we want the data to range to

    Returns:
        Dataframe of demand
    """
    # Download full month
    from_batch = download_new_batch_of_data(from_date.year, from_date.month)
    # Filter out unwanted rows
    from_batch = from_batch[from_batch["datetime"] >= pd.Timestamp(from_date)]

    # Download full month
    to_batch = download_new_batch_of_data(to_date.year, to_date.month)
    # Filter out unwanted rows
    to_batch = to_batch[to_batch["datetime"] < pd.Timestamp(to_date)]

    data = pd.concat([from_batch, to_batch])
    data.drop_duplicates(inplace=True)

    # To deal with downcasting when filling NaNs
    data["demand"] = data["demand"].astype(int)

    data = fill_missing_demand_values(data)

    data.sort_values(by=["ba_code", "datetime"], inplace=True)

    return data


def prepare_feature_store_data_for_training(data: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares feature store data for training with SKForecast. Target series are
    moved to individual columns and the timestamp is set as the index.

    Args:
        data: dataframe from Hopsworks feature store

    Returns:
        pd.DataFrame
    """

    from src.config import BAS

    # Filter out unwanted BAs
    data = data[data["ba_code"].isin(BAS)].copy()

    data["datetime"] = pd.to_datetime(data["datetime"]).dt.date
    data = data.set_index("datetime")

    data = pd.pivot_table(
        data=data, values="demand", index="datetime", columns="ba_code"
    )
    # Resetting column names
    data.columns.name = None
    data.columns = [f"ba_{ba_code}" for ba_code in data.columns]

    # Explicitly set frequency of index
    data = data.asfreq("1D")

    data = data.sort_index()

    return data


def prepare_raw_data_for_training():
    """
    Concatentates all raw data and prepares it for training. 
    Missing values forward-filled with previous days' demand. 
    """
    concat_demand = pd.DataFrame(columns=["datetime", "ba_code", "demand"])

    for file_path in RAW_DATA_DIR.glob("*.csv"):
        with open(file_path, "rb"):
            tmp = pd.read_csv(file_path)
            concat_demand = pd.concat([concat_demand, tmp])

    # To deal with downcasting when filling NaNs
    concat_demand["demand"] = concat_demand["demand"].astype(int)

    # For annotating the file name
    min_month, min_year = (
        datetime.strptime(concat_demand["datetime"].min(), "%Y-%m-%d").month,
        datetime.strptime(concat_demand["datetime"].min(), "%Y-%m-%d").year,
    )

    max_month, max_year = (
        datetime.strptime(concat_demand["datetime"].max(), "%Y-%m-%d").month,
        datetime.strptime(concat_demand["datetime"].max(), "%Y-%m-%d").year,
    )

    data = pd.pivot_table(
        data=concat_demand, values="demand", index="datetime", columns="ba_code"
    )
    # Resetting column names
    data.columns.name = None
    data.columns = [f"ba_{ba_code}" for ba_code in data.columns]

    # Forward fill NaNs with previous days demand
    data = data.ffill()

    data = data.sort_index()

    data.to_csv(
        TRANSFORMED_DATA_DIR
        / f"ts_tabular_{min_year}_{min_month}_to_{max_year}_{max_month}.csv"
    )

    print(
        f"Data transformed and saved at",
        TRANSFORMED_DATA_DIR
        / f"ts_tabular_{min_year}_{min_month}_to_{max_year}_{max_month}.csv",
    )


def split_data(
    df: pd.DataFrame, train_end: str, days_of_historic_data: int
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Splits data into training and test sets. Training set is from the earliest date
    in {df} to {train_end}. The test set is from {days_of_historic data} days before {train_end}
    to the latest date in {df}.

    Args:
        df: The data to split
        train_end: The date at which the data is split
        days_of_historic_data: The number of days before {train_end} that are required to make
            predictions on the test set

    Returns:
        Tuple containing targets and features for the training and test sets
    """

    data = df.copy()

    train_end = pd.Timestamp(train_end)

    X_train = data.loc[data["datetime"] < train_end]
    X_test = data.loc[
        data["datetime"] >= train_end - pd.offsets.Day(days_of_historic_data)
    ]

    y_train = data.loc[data["datetime"] < train_end]["demand"]
    y_test = data.loc[
        data["datetime"] >= train_end - pd.offsets.Day(days_of_historic_data)
    ]["demand"]

    print(f"Data successfully split at {train_end.date()}:")
    print(
        f"\t{X_train.shape=}: {X_train['datetime'].min().strftime('%Y-%m-%d')} --- {X_train['datetime'].max().strftime('%Y-%m-%d')}"
    )
    print(f"\t{y_train.shape=}")
    print(
        f"\t{X_test.shape=}: {X_test['datetime'].min().strftime('%Y-%m-%d')} --- {X_test['datetime'].max().strftime('%Y-%m-%d')}"
    )
    print(f"\t{y_test.shape=}")

    return X_train, y_train, X_test, y_test


def transform_training_data(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    preprocessing_pipeline: Pipeline,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Applies {preprocessing_pipeline} of transformations to the train and test sets for training and
    validation.

    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target

    Returns:
        Tuple containing transformed targets and features for the train and test sets.
    """

    # Fit and transform train set
    X_train_t = preprocessing_pipeline.fit_transform(X_train)
    # Align indexes
    y_train_t = y_train.loc[X_train_t.index]

    # Transform test set
    X_test_t = preprocessing_pipeline.transform(X_test)
    # Align indexes
    y_test_t = y_test.loc[X_test_t.index]

    assert all(
        y_train_t.index == X_train_t.index
    ), "Indexes of target don't match features in training set"
    assert all(
        y_test_t.index == X_test_t.index
    ), "Indexes of target don't match features in testing set"

    return X_train_t, y_train_t, X_test_t, y_test_t
