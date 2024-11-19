from typing import List
import pandas as pd
import plotly.express as px


def plot_demand(data: pd.DataFrame, bas: List[str] | None = None):
    """
    Plots time-series data for single or multiple balancing authorities.
    
    Args: 
        data: dataframe with datetime index and columns ba_code and demand.
        bas: list of BAs to plot.  
    """
    # Wrangling data for plotly
    df_plot = data.reset_index()
    df_melted = df_plot.melt(
        id_vars=df_plot.columns[0], var_name="ba_code", value_name="demand"
    )
    df_melted = df_melted.sort_values(by=["ba_code", "datetime"])
    df_melted = df_melted[["datetime", "demand", "ba_code"]]

    df_melted = df_melted[df_melted["ba_code"].isin(bas)] if bas else df_melted

    fig = px.line(
        df_melted,
        x="datetime",
        y="demand",
        color="ba_code",
        template="none",
    )

    fig.show()
