# Energy demand forecasting in the US

A machine learning engineering project that predicts daily electricity demand for different balancing authorities (BAs) across the United States. This project emphasizes ML engineering best practices including modular design, feature store and model registry integration, and automated pipelines.

Links to the frontend and monitoring dashboards are [here](https://frontendpy-us-energy-demand-forecasting.streamlit.app/) and here.

## Project Overview

This project demonstrates end-to-end ML engineering practices for time-series forecasting, featuring:

- Automated data ingestion from the [US Energy Information Administration (EIA)](https://www.eia.gov/opendata/) API
- Feature engineering and storage using Hopsworks feature store
- Model training with LightGBM regression
- Model registry for versioning and deployment
- Automated pipeline for generating inferences
- Streamlit applications for visualizing predictions and monitoring model performance

## Architecture
Below is an overview of the system architecture. It follows the [FTI (feature-training-inference)](https://www.hopsworks.ai/post/mlops-to-ml-systems-with-fti-pipelines) design in which the component pipelines are uncoupled from one another and can be operated and developed independently.

![](architecture.drawio.svg)

The system consists of several components:

1. **Feature Pipeline**: US electricity demand data is fetched daily from the EIA API and uploaded to the feature store. This process is triggered via a GitHub Actions workflow.
2. **Training Pipeline**: Features and labels are fetched from the feature store and passed to the training pipeline. This notebook preprocesses them and trains a simple LightGBM regressor and the trained model and fitted preprocessing pipeline is uploaded to a model registry.
3. **Inference Pipeline**: On completion of the feature pipeline, another GitHub Actions workflow is triggered that fetches features and the model from Hopsworks, makes a batch of predictions and writes these back to the feature store. 
4. **Visualization**: The stored features, labels and predictions are used by a Streamlit frontend for displaying forecasts alongside historical data and the next day's prediction.
5. **Monitoring**: A second Streamlit app takes the labels and predictions from the feature store and calculates and plots metrics for monitoring purposes.

## Repository Structure

```
.
├── README.md
├── architecture.drawio.svg
├── .env
├── notebooks/
├── ├── 01_download_raw_data.ipynb
├── ├── 02_transform_raw_data_into_ts_data.ipynb
├── ├── 03_transform_ts_into_features_and_targets.ipynb
├── ├── 04_baseline_model.ipynb
├── ├── 05_lgbm_model.ipynb
├── ├── 06_lgbm_model_with_hp_tuning.ipynb
├── ├── 07_backfill_feature_store.ipynb
├── ├── 08_feature_pipeline.ipynb
├── ├── 09_training_pipeline.ipynb
├── └── 10_inference_pipeline.ipynb
├── poetry.lock
├── pyproject.toml
├── src/
├── ├── __init__.py
├── ├── config.py
├── ├── data.py
├── ├── frontend.py
├── ├── inference.py
├── ├── model.py
├── ├── paths.py
└── └── plot.py
```

## Getting Started

### Prerequisites
The project was written in Python 3.11.10 and uses `poetry` for dependency management. See `.env.sample` for how to set up environment credentials for the EIA and Hopsworks services.

## Future Improvements

- Training and evaluating additional model architectures
- Implementing automated monitoring of model performance drift
- Setting up triggered retraining when performance degrades
- Adding more advanced features from external data sources
- Enhancing the visualization dashboard with additional metrics