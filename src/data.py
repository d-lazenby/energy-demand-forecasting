import calendar
import holidays
from datetime import datetime
from dotenv import load_dotenv
import os

import requests
import pandas as pd

from typing import Tuple

from src.paths import PARENT_DIR, RAW_DATA_DIR, TRANSFORMED_DATA_DIR

load_dotenv(PARENT_DIR / ".env")
EIA_API_KEY = os.environ["EIA_API_KEY"]


def download_raw_data(year: int, month: int):
    """
    Downloads one month of raw data from EIA for {year}, {month}.
    
    Args:
        year: Year of data collection
        month: Month of data collection
    """
    file_path = RAW_DATA_DIR / f"demand_{year}_{month}.csv"
    if file_path.exists():
        print(f"File demand_{year}_{month}.csv exists locally already, try next URL")
    else:
        # Need the number of days in the current (year, month)
        _, num_days = calendar.monthrange(year, month)

        URL = (
            "https://api.eia.gov/v2/electricity/rto/daily-region-data/data/"
            "?frequency=daily"
            "&data[0]=value"
            "&facets[timezone][]=Eastern"
            "&facets[type][]=D"
            f"&start={year}-{month:02d}-01"
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

        data.to_csv(file_path, index=False)
        print(
            f"Data for {year}_{month} successfully downloaded to demand_{year}_{month}.csv"
        )

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

def load_training_data() -> pd.DataFrame:
    """
    Loads data from CSV and prepares for training. 
    """
    
    data = pd.read_csv(TRANSFORMED_DATA_DIR / "ts_tabular_2022_10_to_2024_10.csv")
    # Wrangling index for deriving exog features
    data["datetime"] = pd.to_datetime(data["datetime"])
    data = data.set_index("datetime")
    # Explicitly set freqency of index
    data = data.asfreq("1D")
    
    return data

def make_exog_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds columns to indicate holidays, month, day of week, weekend and season.
    
    Args:
        df: dataframe containing the time-series data. 
        
    Returns:
        Modified dataframe with additional columns of exogenous features.
    """
    data = df.copy()
    
    us_holidays = holidays.US(years=[2022, 2023, 2024])
    data["exog_is_holiday"] = data.index.map(lambda day: day in us_holidays).astype(int)

    data["exog_month"] = data.index.month
    data["exog_day_of_week"] = data.index.dayofweek
    data["exog_is_weekend"] = data["exog_day_of_week"].isin([5, 6]).astype(int)

    # Winter = 12, 1, 2; Spring = 3, 4, 5; ...
    data["exog_season"] = ((data["exog_month"] % 12) // 3) + 1

    return data


def split_data(
    df: pd.DataFrame, end_train: str = "2023-10-31 23:59:00"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the data into training and testing sets, taking all rows before {end_train} as
    training data and all rows after as testing data.

    Args:
        df: dataframe containing the time-series data.
        end_train: string containing the cutoff date. Should be in format, e.g., '2023-10-31 23:59:00' to avoid 
            rows appearing in both train and test sets.
    
    Returns:
        (data_train, data_test)
    """

    data = df.copy()

    data_train = data.loc[:end_train, :].copy()
    data_test = data.loc[end_train:, :].copy()

    print(f"{data_train.shape=}")
    print(f"{data_test.shape=}")
    
    print(
        f"Train dates : {data_train.index.min()} --- {data_train.index.max()}   "
        f"(n={len(data_train)})"
    )
    print(
        f"Test dates  : {data_test.index.min()} --- {data_test.index.max()}   "
        f"(n={len(data_test)})"
    )

    return data_train, data_test
