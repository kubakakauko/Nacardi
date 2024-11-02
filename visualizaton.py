# visualization.py

import os
import sys

import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# ----------------------------- #
#       CSV Files and Columns   #
# ----------------------------- #

# Define the required CSV files and their expected columns
REQUIRED_CSV_FILES = {
    "nadaraya_watson_data.csv": [
        "Timestamp",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "y",
        "Upper_Band",
        "Lower_Band",
        "Buy_Signal",
        "Sell_Signal",
        "EMA",  # Optional if TEMA is used
        "TEMA",  # Optional if EMA is used
        "RSI",  # Ensure RSI is present
    ],
    "trade_log.csv": [
        "Timestamp",
        "Trade_ID",
        "Action",
        "Price",
        "PnL",
        "Position",
    ],
    "performance_log.csv": [
        "Timestamp",
        "Total Trades",
        "Win Trades",
        "Win Ratio",
        "Total PnL",
        "Max Drawdown",
        "Sharpe Ratio",
    ],
    "historical_trade_log.csv": [
        "Timestamp",
        "Signal",
        "Price",
    ],
}


def check_and_load_csv(file_path: str, required_columns: list) -> pd.DataFrame:
    """
    Check if the CSV file exists and has the required columns.
    If the file does not exist, create it with the required columns.
    If columns are missing, add them with NaN values.
    Returns the loaded DataFrame.
    """
    if not os.path.exists(file_path):
        print(
            f"CSV file '{file_path}' not found. Creating a new one with required columns."
        )
        df = pd.DataFrame(columns=required_columns)
        df.to_csv(file_path, index=False)
    else:
        df = pd.read_csv(file_path)
        existing_columns = df.columns.tolist()
        missing_columns = [
            col for col in required_columns if col not in existing_columns
        ]
        if missing_columns:
            print(
                f"CSV file '{file_path}' is missing columns: {missing_columns}. Adding them with NaN values."
            )
            for col in missing_columns:
                df[col] = pd.NA
            df.to_csv(file_path, index=False)
    # Load the CSV with proper parsing
    try:
        df = pd.read_csv(file_path, parse_dates=["Timestamp"])
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True)
        df.sort_values("Timestamp", inplace=True)
    except Exception as e:
        print(f"Error loading '{file_path}': {e}")
        sys.exit(1)
    return df


# ----------------------------- #
#          Load Data            #
# ----------------------------- #

# Load all required CSV files
data_log = check_and_load_csv(
    "nadaraya_watson_data.csv", REQUIRED_CSV_FILES["nadaraya_watson_data.csv"]
)
trade_log = check_and_load_csv("trade_log.csv", REQUIRED_CSV_FILES["trade_log.csv"])
performance_log = check_and_load_csv(
    "performance_log.csv", REQUIRED_CSV_FILES["performance_log.csv"]
)
historical_trade_log = check_and_load_csv(
    "historical_trade_log.csv", REQUIRED_CSV_FILES["historical_trade_log.csv"]
)

# ----------------------------- #
#        Create Figure          #
# ----------------------------- #

# Determine if TEMA is used based on the presence of 'TEMA' column with non-NA values
USE_TEMA = "TEMA" in data_log.columns and data_log["TEMA"].notna().any()
indicator_name = "TEMA" if USE_TEMA else "EMA"

fig = make_subplots(
    rows=5,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.02,
    subplot_titles=(
        "Price with Indicators and Signals",
        "Volume",
        "Cumulative PnL",
        "Performance Metrics",
        "RSI",
    ),
    row_width=[
        0.2,  # Price with indicators and signals
        0.2,  # Volume
        0.2,  # Cumulative PnL
        0.2,  # Performance Metrics
        0.2,  # RSI
    ],  # Proportion of the figure allocated to each subplot
)

# ----------------------------- #
#     Add Candlestick Chart     #
# ----------------------------- #

try:
    fig.add_trace(
        go.Candlestick(
            x=data_log["Timestamp"],
            open=data_log["Open"],
            high=data_log["High"],
            low=data_log["Low"],
            close=data_log["Close"],
            name="Price",
            increasing_line_color="green",
            decreasing_line_color="red",
        ),
        row=1,
        col=1,
    )
except Exception as e:
    print(f"Error adding candlestick chart: {e}")

# ----------------------------- #
#   Add Nadaraya-Watson Estimator #
# ----------------------------- #

try:
    fig.add_trace(
        go.Scatter(
            x=data_log["Timestamp"],
            y=data_log["y"],
            mode="lines",
            name="Nadaraya-Watson Estimator",
            line=dict(color="orange", width=2),
        ),
        row=1,
        col=1,
    )

    # Add Upper and Lower Bands
    fig.add_trace(
        go.Scatter(
            x=data_log["Timestamp"],
            y=data_log["Upper_Band"],
            mode="lines",
            name="Upper Band",
            line=dict(color="purple", dash="dash"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=data_log["Timestamp"],
            y=data_log["Lower_Band"],
            mode="lines",
            name="Lower Band",
            line=dict(color="brown", dash="dash"),
        ),
        row=1,
        col=1,
    )
except Exception as e:
    print(f"Error adding Nadaraya-Watson Estimator or Bands: {e}")

# ----------------------------- #
#      Add EMA/TEMA Indicator   #
# ----------------------------- #

try:
    if USE_TEMA and "TEMA" in data_log.columns:
        fig.add_trace(
            go.Scatter(
                x=data_log["Timestamp"],
                y=data_log["TEMA"],
                mode="lines",
                name=f"TEMA (20)",
                line=dict(color="blue", width=2),
            ),
            row=1,
            col=1,
        )
    elif "EMA" in data_log.columns:
        fig.add_trace(
            go.Scatter(
                x=data_log["Timestamp"],
                y=data_log["EMA"],
                mode="lines",
                name=f"EMA (20)",
                line=dict(color="blue", width=2),
            ),
            row=1,
            col=1,
        )
except Exception as e:
    print(f"Error adding EMA/TEMA Indicator: {e}")

# ----------------------------- #
#    Add Buy and Sell Signals   #
# ----------------------------- #

try:
    buy_signals = data_log.dropna(subset=["Buy_Signal"])
    sell_signals = data_log.dropna(subset=["Sell_Signal"])

    fig.add_trace(
        go.Scatter(
            x=buy_signals["Timestamp"],
            y=buy_signals["Buy_Signal"],
            mode="markers",
            name="Buy Signal",
            marker=dict(symbol="triangle-up", color="lime", size=12),
            hovertemplate="Buy Signal<br>Price: %{y:.2f}<br>Time: %{x}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=sell_signals["Timestamp"],
            y=sell_signals["Sell_Signal"],
            mode="markers",
            name="Sell Signal",
            marker=dict(symbol="triangle-down", color="maroon", size=12),
            hovertemplate="Sell Signal<br>Price: %{y:.2f}<br>Time: %{x}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Add historical trades
    historical_buys = historical_trade_log[historical_trade_log["Signal"] == "BUY"]
    historical_sells = historical_trade_log[historical_trade_log["Signal"] == "SELL"]

    fig.add_trace(
        go.Scatter(
            x=historical_buys["Timestamp"],
            y=historical_buys["Price"],
            mode="markers",
            name="Historical Buy",
            marker=dict(symbol="triangle-up", color="cyan", size=8),
            hovertemplate="Historical Buy<br>Price: %{y:.2f}<br>Time: %{x}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=historical_sells["Timestamp"],
            y=historical_sells["Price"],
            mode="markers",
            name="Historical Sell",
            marker=dict(symbol="triangle-down", color="magenta", size=8),
            hovertemplate="Historical Sell<br>Price: %{y:.2f}<br>Time: %{x}<extra></extra>",
        ),
        row=1,
        col=1,
    )
except Exception as e:
    print(f"Error adding Buy/Sell Signals: {e}")

# ----------------------------- #
#  Add Trade Entries and Exits  #
# ----------------------------- #

try:
    open_trades = trade_log[trade_log["Action"].isin(["Open LONG", "Open SHORT"])]
    close_trades = trade_log[trade_log["Action"].isin(["Close LONG", "Close SHORT"])]

    # Add Trade Entries
    fig.add_trace(
        go.Scatter(
            x=open_trades["Timestamp"],
            y=open_trades["Price"],
            mode="markers",
            name="Trade Entry",
            marker=dict(symbol="star", color="blue", size=15),
            text=open_trades["Action"],
            hovertemplate="<b>%{text}</b><br>Price: %{y:.2f}<br>Time: %{x}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Add Trade Exits
    fig.add_trace(
        go.Scatter(
            x=close_trades["Timestamp"],
            y=close_trades["Price"],
            mode="markers",
            name="Trade Exit",
            marker=dict(symbol="x", color="black", size=15),
            text=close_trades["Action"],
            hovertemplate="<b>%{text}</b><br>Price: %{y:.2f}<br>Time: %{x}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Connect Trade Entries and Exits
    for trade_id in trade_log["Trade_ID"].unique():
        trade_actions = trade_log[trade_log["Trade_ID"] == trade_id].sort_values(
            "Timestamp"
        )
        if len(trade_actions) == 2:
            entry = trade_actions.iloc[0]
            exit_ = trade_actions.iloc[1]
            fig.add_trace(
                go.Scatter(
                    x=[entry["Timestamp"], exit_["Timestamp"]],
                    y=[entry["Price"], exit_["Price"]],
                    mode="lines",
                    line=dict(color="grey", dash="dot"),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1,
                col=1,
            )
except Exception as e:
    print(f"Error adding Trade Entries/Exits: {e}")

# ----------------------------- #
#        Add Volume Bars        #
# ----------------------------- #

try:
    fig.add_trace(
        go.Bar(
            x=data_log["Timestamp"],
            y=data_log["Volume"],
            name="Volume",
            marker_color="lightblue",
            opacity=0.7,
        ),
        row=2,
        col=1,
    )
except Exception as e:
    print(f"Error adding Volume Bars: {e}")

# ----------------------------- #
#      Add Cumulative PnL       #
# ----------------------------- #

try:
    # Calculate Cumulative PnL from performance_log
    performance_log = performance_log.sort_values("Timestamp")
    performance_log["Cumulative_PnL"] = performance_log["Total PnL"]

    fig.add_trace(
        go.Scatter(
            x=performance_log["Timestamp"],
            y=performance_log["Cumulative_PnL"],
            mode="lines",
            name="Cumulative PnL",
            line=dict(color="green", width=2),
        ),
        row=3,
        col=1,
    )
except Exception as e:
    print(f"Error adding Cumulative PnL: {e}")

# ----------------------------- #
#     Add Performance Metrics   #
# ----------------------------- #

try:
    # Plot Max Drawdown and Sharpe Ratio
    fig.add_trace(
        go.Scatter(
            x=performance_log["Timestamp"],
            y=performance_log["Max Drawdown"],
            mode="lines",
            name="Max Drawdown",
            line=dict(color="red", width=2),
        ),
        row=4,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=performance_log["Timestamp"],
            y=performance_log["Sharpe Ratio"],
            mode="lines",
            name="Sharpe Ratio",
            line=dict(color="purple", width=2),
        ),
        row=4,
        col=1,
    )
except KeyError as e:
    print(f"Missing expected column in performance_log.csv: {e}")
except Exception as e:
    print(f"Error adding Performance Metrics: {e}")

# ----------------------------- #
#          Add RSI Plot         #
# ----------------------------- #

try:
    if "RSI" in data_log.columns:
        fig.add_trace(
            go.Scatter(
                x=data_log["Timestamp"],
                y=data_log["RSI"],
                mode="lines",
                name="RSI",
                line=dict(color="orange", width=2),
            ),
            row=5,
            col=1,
        )

        # Add overbought and oversold lines
        fig.add_hline(y=75, line_dash="dash", line_color="red", row=5, col=1)
        fig.add_hline(y=25, line_dash="dash", line_color="green", row=5, col=1)
    else:
        print("RSI data not found in data_log.csv. Skipping RSI plot.")
except Exception as e:
    print(f"Error adding RSI plot: {e}")

# ----------------------------- #
#        Update Layout          #
# ----------------------------- #

fig.update_layout(
    title="Nadaraya-Watson Live Trading Signals and Performance",
    hovermode="x unified",
    width=1600,
    height=1400,
    margin=dict(l=50, r=50, t=80, b=50),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        bordercolor="Black",
        borderwidth=1,
    ),
)

# ----------------------------- #
#       Customize Axes          #
# ----------------------------- #

try:
    # Customize x-axes
    fig.update_xaxes(
        rangeslider_visible=False,
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,
        row=1,
        col=1,
    )
    fig.update_xaxes(
        tickformat="%Y-%m-%d %H:%M",
        tickangle=45,
        row=5,
        col=1,
    )

    # Customize y-axes for price
    fig.update_yaxes(
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,
        row=1,
        col=1,
        title_text="Price",
    )

    # Customize y-axes for volume
    fig.update_yaxes(
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,
        row=2,
        col=1,
        title_text="Volume",
    )

    # Customize y-axes for cumulative PnL
    fig.update_yaxes(
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,
        row=3,
        col=1,
        title_text="Cumulative PnL",
    )

    # Customize y-axes for performance metrics
    fig.update_yaxes(
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,
        row=4,
        col=1,
        title_text="Performance Metrics",
    )

    # Customize y-axes for RSI
    fig.update_yaxes(
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,
        row=5,
        col=1,
        title_text="RSI",
        range=[0, 100],
    )
except Exception as e:
    print(f"Error customizing axes: {e}")

# ----------------------------- #
#          Show Plot            #
# ----------------------------- #

try:
    fig.show()
except Exception as e:
    print(f"Error displaying the plot: {e}")
