import streamlit as st
from datetime import datetime

from src.plot import plot_demand_with_prediction
from src.inference import (
    load_batch_of_features_for_inference,
    load_model_from_registry,
    get_model_predictions,
)

st.set_page_config(layout="wide")

current_date = datetime.now().date()
st.title("US Daily Electricity Demand Prediction ⚡️")
st.header(f"{current_date}")

st.sidebar.header("Working... 🐝")
progress_bar = st.sidebar.progress(0)

n_steps = 4

with st.spinner(text="Fetching batch of features for inference..."):
    inference_data = load_batch_of_features_for_inference(current_date)
    st.sidebar.write("✅ Features fetched successfully")
    progress_bar.progress(1 / n_steps)

with st.spinner(text="Loading model from registry..."):
    model, preprocessing_pipeline = load_model_from_registry()
    st.sidebar.write("✅ Model loaded")
    progress_bar.progress(2 / n_steps)

with st.spinner(text="Model is making inferences..."):
    features_end = str(inference_data["datetime"].dt.date.max())

    predictions = get_model_predictions(
        model=model,
        preprocessing_pipeline=preprocessing_pipeline,
        X=inference_data,
        features_end=features_end,
    )
    st.sidebar.write("✅ Predictions made")
    progress_bar.progress(3 / n_steps)

with st.spinner(text="Preparing plots..."):
    for ba_code in inference_data["ba_code"].unique():
        tmp_pred = predictions.loc[predictions["ba_code"] == ba_code]
        pred = (tmp_pred["datetime"].iloc[0], tmp_pred["predicted_demand"].iloc[0])

        fig = plot_demand_with_prediction(inference_data, ba_code, pred)

        st.plotly_chart(fig, theme="streamlit", use_container_width=True)

    st.sidebar.write("✅ Plotting complete")
    progress_bar.progress(4 / n_steps)
