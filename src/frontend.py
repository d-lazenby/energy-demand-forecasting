import streamlit as st
from datetime import datetime, timedelta

from src.plot import plot_demand_with_prediction
from src.inference import (
    load_batch_of_features_for_inference,
    load_batch_of_predictions,
)

st.set_page_config(layout="wide")

current_date = datetime.now().date()
st.title("Predicting Daily Electricity Demand in the US ⚡️")
st.header(f"{current_date}")

with st.sidebar:
    st.markdown(
        """
        - :red[**Data:**] The data is fetched daily from the 
            [US Energy Information Administration (EIA) API](https://www.eia.gov/opendata/) via a feature 
            pipeline triggered with a GitHub Action. 
        - :red[**Charts:**] The plots show a year's worth of data (blue trace) along with the prediction 
            for tomorrow's demand (red circle) for each Balancing Authority (BA).
        - :red[**Modelling:**] A simple global LGBMRegressor model was trained over the entire dataset. The demand 
            scales from $10^3{-}10^7$ MWh across the BAs and the time series 
            are volatile making it a challenging modelling exercise.
        """
    )

st.sidebar.header("Working... 🐝")
progress_bar = st.sidebar.progress(0)

n_steps = 3

with st.spinner(text="Fetching historical data..."):
    inference_data = load_batch_of_features_for_inference(current_date)
    st.sidebar.write("✅ Features fetched successfully")
    progress_bar.progress(1 / n_steps)

with st.spinner(text="Fetching predictions from the feature store..."):
    predictions = load_batch_of_predictions(
        current_date=current_date + timedelta(days=1), days_in_past=2
    )
    st.sidebar.write("✅ Predictions made")
    progress_bar.progress(2 / n_steps)

with st.spinner(text="Preparing plots..."):
    for ba_code in inference_data["ba_code"].unique():
        tmp_pred = predictions.loc[predictions["ba_code"] == ba_code]
        pred = (tmp_pred["datetime"].iloc[-1], tmp_pred["predicted_demand"].iloc[-1])

        fig = plot_demand_with_prediction(inference_data, ba_code, pred)

        st.plotly_chart(fig, theme="streamlit", use_container_width=True)

    st.sidebar.write("✅ Plotting complete")
    progress_bar.progress(3 / n_steps)
