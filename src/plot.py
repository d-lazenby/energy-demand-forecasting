from typing import List, Optional
import pandas as pd
import plotly.express as px


def plot_demand(demand: pd.DataFrame, bas: Optional[List[int]] = None):
    demand_to_plot = demand[demand["ba_code"].isin(bas)] if bas else demand

    fig = px.line(
        demand_to_plot,
        x="datetime",
        y="demand",
        color="ba_code",
        template="none",
    )

    fig.show()
