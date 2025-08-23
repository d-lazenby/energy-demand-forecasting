import streamlit as st
from datetime import datetime

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from src.monitoring import (
    load_predictions_and_actual_values_from_feature_store, 
    get_mae_df,
    get_mae_by_ba_df
    )
from src.plot import plot_mae_df


st.set_page_config(layout="wide")

current_date = datetime.now().date()
st.title("Monitoring Dashboard for Electricity Demand Prediction Service ⚡️📈📊")
st.header(f"{current_date}")

with st.sidebar:
    st.markdown(
        """
        - :red[**Metrics:**] The charts plot the mean absolute error (MAE) between the actual demand values 
            and those predicted by the model overall (top chart) and for each individual Balancing Authority (BA). 
            The red line indicates the seven-day moving average of the MAE.
        - :red[**Data:**] The predictions and actual values used to calculate the MAE are fetched from an online feature store; 
            the values are updated daily (midnight Eastern Time) via feature and inference pipelines triggered by GitHub Actions.
        """
    )


st.sidebar.header("Working... 🐝")
progress_bar = st.sidebar.progress(0)

n_steps = 4

# Load predictions and actuals
with st.spinner(text="Fetching predictions and actual demand values from the feature store..."):
    monitoring_df = load_predictions_and_actual_values_from_feature_store(
        current_date=current_date, days_in_past=60
    )
    st.sidebar.write("✅ Values fetched successfully")
    progress_bar.progress(1 / n_steps)

# Calculate Metrics
with st.spinner(text="Calculating MAE..."):
    mae_df = get_mae_df(monitoring_df)
    mae_by_ba_df = get_mae_by_ba_df(monitoring_df)
    st.sidebar.write("✅ Metrics calculated")
    progress_bar.progress(2 / n_steps)

# Plot MAE for all BAs
with st.spinner(text="Plotting MAE for all BAs..."):
    fig = plot_mae_df(mae_df)
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    st.sidebar.write("✅ Overall MAE plotted")
    progress_bar.progress(3 / n_steps)

# Plot MAE by BA
with st.spinner(text="Plotting MAE for all BAs..."):
    ba_codes = mae_by_ba_df["ba_code"].unique()
    for ba_code in ba_codes:
        tmp = mae_by_ba_df.loc[mae_by_ba_df["ba_code"] == ba_code].copy()
        fig = plot_mae_df(tmp, all_bas=False)
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)

    st.sidebar.write("✅ MAE plotted for all BAs")
    progress_bar.progress(4 / n_steps)
