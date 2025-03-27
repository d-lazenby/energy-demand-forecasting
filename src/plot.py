from typing import Optional
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
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
    demand: pd.DataFrame, ba: str, prediction: tuple[pd.Timestamp, float]
    ) -> go.Figure:
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
        yaxis_title="Electricity Demand (MWh)",
        showlegend=False,
        yaxis=dict(showgrid=False),
    )

    return fig


def plot_mae_df(mae_df: pd.DataFrame, all_bas: bool = True) -> go.Figure:
    """
    Plots a bar chart of the MAE along with a line plot of the 7-day MAE moving average. If {mae_df}
    has been derived per BA, then it should accept a slice for a particular BA and {all_bas} can be
    set to False to plot the MAE for that BA.

    Args:
        mae_df: a dataframe containing the date, MAE and 7-day MAE moving average

    Returns:
        The combined bar and line plot
    """
    if not all_bas:
        ba_code = mae_df["ba_code"].unique()[0]
        mae_df = mae_df[[col for col in mae_df.columns if col != "ba_code"]]
        title = f"MAE for the last 30 days from {mae_df.iloc[-1]['datetime'].date()} for {ba_code}"
    else:
        title = f"MAE for the last 30 days from {mae_df.iloc[-1]['datetime'].date()}"

    fig = px.bar(
        mae_df,
        x="datetime",
        y="mae",
        template="plotly_dark",
    )

    default_colors = px.colors.qualitative.G10_r

    fig.add_trace(
        go.Scatter(
            x=mae_df["datetime"],
            y=mae_df["mae_window_7_mean"],
            mode="lines",
            line=dict(color=default_colors[1], width=2),
            opacity=0.6,
            name="7-Day Avg MAE",
        )
    )

    fig.update_traces(opacity=0.6, marker=dict(color=default_colors[0]))

    fig.update_layout(
        title=f"{title}",
        xaxis_title=None,
        yaxis_title="MAE (MWh)",
        showlegend=True,
        yaxis=dict(showgrid=False),
    )

    return fig


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
