from typing import Optional
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt


def plot_demand(demand: pd.DataFrame, bas: Optional[list[str]] = None):
    demand_to_plot = demand[demand["ba_code"].isin(bas)] if bas else demand

    fig = px.line(
        demand_to_plot,
        x="datetime",
        y="demand",
        color="ba_code",
        template="none",
    )

    fig.show()


def plot_demand_with_prediction(
    demand: pd.DataFrame, ba: str, prediction: Optional[tuple] = None
    ):
    demand_to_plot = demand[demand["ba_code"] == ba].copy()

    default_colors = px.colors.qualitative.G10_r

    fig = px.line(
        demand_to_plot,
        x="datetime",
        y="demand",
        markers=False,
        color="ba_code",
        template="plotly_dark",
    )

    fig.update_traces(opacity=0.6, line=dict(color=default_colors[0]))

    if prediction:
        fig.add_scatter(
            x=[prediction[0]],
            y=[prediction[1]],
            mode="markers",
            marker=dict(
                size=10,
                symbol="circle-open",
                line=dict(width=2),
            ),
            name="Prediction",
        )

    fig.update_layout(
        title=f"Predicted demand for {ba} on {prediction[0].date()}",
        xaxis_title=None,
        yaxis_title="Electricity Demand (MW)",
        showlegend=False,
        yaxis=dict(showgrid=False),
    )

    fig.show()


def plot_predictions_against_actuals(
    train_pred: pd.Series,
    train_actual: pd.Series,
    test_pred: pd.Series,
    test_actual: pd.Series,
):
    """
    Plots residuals for training and test sets.

    Args:
        train_pred: Predictions for training set
        train_actual: Actual values for training set
        test_pred: Predictions for test set
        test_actual: Actual values for test set
    """

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
    
    ax1.scatter(
        test_pred,
        test_actual,
        c="limegreen",
        marker="s",
        edgecolor="white",
        label="Test data",
    )
    
    ax2.scatter(
        train_pred,
        train_actual,
        c="steelblue",
        marker="o",
        edgecolor="white",
        label="Training data",
    )
    
    ax1.set_ylabel("Actual values")
    for ax in (ax1, ax2):
        ax.set_xlabel("Predicted values")
        ax.legend(loc="upper left")
    plt.suptitle("Predicted vs Actual Values")
    plt.tight_layout()
    plt.show()


def plot_residuals(
    train_pred: pd.Series,
    train_actual: pd.Series,
    test_pred: pd.Series,
    test_actual: pd.Series,
):
    """
    Plots residuals for training and test sets.

    Args:
        train_pred: Predictions for training set
        train_actual: Actual values for training set
        test_pred: Predictions for test set
        test_actual: Actual values for test set
    """
    x_max = np.max([np.max(train_pred), np.max(test_pred)])
    x_min = np.min([np.min(train_pred), np.min(test_pred)])
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
    
    ax1.scatter(
        test_pred,
        test_pred - test_actual,
        c="limegreen",
        marker="s",
        edgecolor="white",
        label="Test data",
    )
    
    ax2.scatter(
        train_pred,
        train_pred - train_actual,
        c="steelblue",
        marker="o",
        edgecolor="white",
        label="Training data",
    )
    
    ax1.set_ylabel("Residuals")
    for ax in (ax1, ax2):
        ax.set_xlabel("Predicted values")
        ax.legend(loc="upper left")
        ax.hlines(y=0, xmin=x_min - 100, xmax=x_max + 100, color="black", lw=2)
    plt.suptitle("Residual Plots")
    plt.tight_layout()
    plt.show()
