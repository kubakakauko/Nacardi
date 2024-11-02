# nadaraya_watson_live_trading.py

import logging
import os
import time
import uuid  # For generating unique trade IDs
from datetime import datetime, timedelta, timezone
from logging.handlers import RotatingFileHandler
from typing import Tuple

import numpy as np
import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException
from scipy.ndimage import gaussian_filter1d

# ----------------------------- #
#        Configuration          #
# ----------------------------- #

# Binance API credentials
# Replace with your actual API and secret, this can later be adjusted for demo trading

API_KEY = "YOUR_API_KEY"
API_SECRET = "YOUR_API_SECRET"
# Trading parameters
SYMBOL = "BTCUSDT"  # Trading pair
INTERVAL = "1m"  # Candle interval (e.g., '1m', '5m', '15m', '1h')
START_TIME = "1 day ago UTC"  # Start time for historical data

# Nadaraya-Watson parameters
NADARAYA_H = 8  # Smoothing parameter (sigma)
NADARAYA_MULT = 3  # Multiplier for MAE

# Other signal parameters

EMA_PERIOD = 20

# Calculate ATR and add to data_df
ATR_PERIOD = 14

# Calculate RSI and add to data_df
RSI_PERIOD = 14

# Log files
DATA_LOG_FILE = "nadaraya_watson_data.csv"
TRADE_LOG_FILE = "trade_log.csv"
PERFORMANCE_LOG_FILE = "performance_log.csv"
DEBUG_LOG_FILE = "nadaraya_watson_trading_debug.log"

# Visualization update interval in minutes
VISUALIZATION_UPDATE_INTERVAL = 10  # Minutes

# ----------------------------- #
#          Logging Setup        #
# ----------------------------- #

# Configure logging with rotation to prevent log files from growing indefinitely
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Debug log handler
debug_handler = RotatingFileHandler(
    DEBUG_LOG_FILE,
    maxBytes=5 * 1024 * 1024,  # 5 MB
    backupCount=5,
)
debug_formatter = logging.Formatter(
    "%(asctime)s %(levelname)s:%(message)s", "%Y-%m-%d %H:%M:%S"
)
debug_handler.setFormatter(debug_formatter)
debug_handler.setLevel(logging.DEBUG)
logger.addHandler(debug_handler)

# Console log handler (optional, for real-time monitoring)
console_handler = logging.StreamHandler()
console_formatter = logging.Formatter(
    "%(asctime)s %(levelname)s:%(message)s", "%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(console_formatter)
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)

# ----------------------------- #
#        Global Variables       #
# ----------------------------- #

POSITION_SIDE = None  # Current position: 'LONG', 'SHORT', or None
ENTRY_PRICE = 0.0  # Entry price for the current position
POSITION_SIZE = 0.0  # Current position size
TOTAL_PNL = 0.0  # Total Profit and Loss
WIN_TRADES = 0  # Number of profitable trades
TOTAL_TRADES = 0  # Total number of trades executed

# For Performance Metrics
CUMULATIVE_PNL = []  # List to track cumulative PnL over time
RETURN_HISTORY = []  # List to track returns for Sharpe Ratio

# Position sizing parameters
BASE_POSITION_SIZE = 1.0  # Base size for initial positions
MIN_DCA_SIZE = 0.5  # Minimum DCA size
MAX_DCA_SIZE = 2.0  # Maximum DCA size

# Risk management parameters
STOP_LOSS_PERCENT = 0.02  # 2% stop loss
TAKE_PROFIT_PERCENT = 0.04  # 4% take profit

# Initialize Binance Client
client = Client(API_KEY, API_SECRET)

# Initialize DataFrame
data_df = pd.DataFrame()

# ----------------------------- #
#            Functions          #
# ----------------------------- #


def append_data_log(
    timestamp,
    open_,
    high,
    low,
    close,
    volume,
    y,
    upper_band,
    lower_band,
    buy_signal,
    sell_signal,
):
    """
    Append a new row to the data log CSV file.
    """
    try:
        df = pd.DataFrame(
            [
                {
                    "Timestamp": timestamp,
                    "Open": open_,
                    "High": high,
                    "Low": low,
                    "Close": close,
                    "Volume": volume,
                    "y": y,
                    "Upper_Band": upper_band,
                    "Lower_Band": lower_band,
                    "Buy_Signal": buy_signal,
                    "Sell_Signal": sell_signal,
                }
            ]
        )
        df.to_csv(
            DATA_LOG_FILE,
            mode="a",
            header=not os.path.exists(DATA_LOG_FILE),
            index=False,
        )
        logging.debug(f"Appended data to {DATA_LOG_FILE} at {timestamp}.")
    except Exception as e:
        logging.error(f"Error appending data to log: {e}")


def log_trade(action, price, pnl, position=None, trade_id=None):
    """
    Log trade actions to the trade log CSV file.
    """
    try:
        if trade_id is None:
            trade_id = str(uuid.uuid4())  # Generate a unique ID if not provided

        df = pd.DataFrame(
            [
                {
                    "Timestamp": datetime.now(timezone.utc),
                    "Trade_ID": trade_id,
                    "Action": action,
                    "Price": price,
                    "PnL": pnl,
                    "Position": position,
                }
            ]
        )
        df.to_csv(
            TRADE_LOG_FILE,
            mode="a",
            header=not os.path.exists(TRADE_LOG_FILE),
            index=False,
        )
        logging.debug(
            f"Logged trade: Trade_ID={trade_id}, Action={action}, Price={price}, PnL={pnl}, Position={position}."
        )
    except Exception as e:
        logging.error(f"Error logging trade: {e}")


def calculate_max_drawdown(cumulative_pnl: list) -> float:
    """
    Calculate the maximum drawdown from the cumulative PnL series.
    """
    if not cumulative_pnl:
        return 0.0

    cumulative = np.array(cumulative_pnl)
    peak = np.maximum.accumulate(cumulative)
    drawdown = peak - cumulative
    max_dd = np.max(drawdown)
    return max_dd


def calculate_sharpe_ratio(returns: list, risk_free_rate: float = 0.0) -> float:
    """
    Calculate the Sharpe Ratio from a list of returns.
    """
    if not returns:
        return 0.0

    returns_array = np.array(returns)
    excess_returns = returns_array - risk_free_rate
    avg_excess_return = np.mean(excess_returns)
    std_excess_return = np.std(excess_returns)
    if std_excess_return == 0:
        return 0.0
    sharpe_ratio = avg_excess_return / std_excess_return
    return sharpe_ratio


def log_performance():
    """
    Log the performance metrics to a CSV file.
    """
    global CUMULATIVE_PNL, RETURN_HISTORY

    try:
        if TOTAL_TRADES > 0:
            win_ratio = WIN_TRADES / TOTAL_TRADES
        else:
            win_ratio = 0.0

        # Calculate Max Drawdown
        max_drawdown = calculate_max_drawdown(CUMULATIVE_PNL)

        # Calculate Sharpe Ratio (assuming returns are per trade)
        sharpe_ratio = calculate_sharpe_ratio(RETURN_HISTORY)

        perf_entry = pd.DataFrame(
            [
                {
                    "Timestamp": datetime.now(timezone.utc),
                    "Total Trades": TOTAL_TRADES,
                    "Win Trades": WIN_TRADES,
                    "Win Ratio": win_ratio,
                    "Total PnL": TOTAL_PNL,
                    "Max Drawdown": max_drawdown,
                    "Sharpe Ratio": sharpe_ratio,
                }
            ]
        )
        perf_entry.to_csv(
            PERFORMANCE_LOG_FILE,
            mode="a",
            header=not os.path.exists(PERFORMANCE_LOG_FILE),
            index=False,
        )
        logging.info(
            f"Performance Logged: Total Trades={TOTAL_TRADES}, Win Trades={WIN_TRADES}, "
            f"Win Ratio={win_ratio:.2f}, Total PnL={TOTAL_PNL:.2f}, "
            f"Max Drawdown={max_drawdown:.2f}, Sharpe Ratio={sharpe_ratio:.2f}"
        )
    except Exception as e:
        logging.error(f"Error logging performance: {e}")


def live_data(
    symbol: str = SYMBOL, interval: str = INTERVAL, start_str: str = START_TIME
) -> pd.DataFrame:
    """
    Fetch historical candle data from Binance and return as a pandas DataFrame.
    """
    try:
        logging.debug(
            f"Fetching historical data for {symbol} with interval {interval} starting from {start_str}."
        )
        candles = client.get_historical_klines(symbol, interval, start_str)
        logging.debug(f"Fetched {len(candles)} candle data points.")
    except BinanceAPIException as e:
        logging.error(f"Binance API Exception while fetching data: {e}")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"Error fetching historical data: {e}")
        return pd.DataFrame()

    if not candles:
        logging.error("No candle data fetched.")
        return pd.DataFrame()

    # Define column names as per Binance API response
    columns = [
        "Open time",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Close time",
        "Quote asset volume",
        "Number of trades",
        "Taker buy base asset volume",
        "Taker buy quote asset volume",
        "Ignore",
    ]

    # Create DataFrame
    df = pd.DataFrame(candles, columns=columns)

    # Convert relevant columns to float
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = df[col].astype(float)

    # Convert timestamp to datetime and set as index (timezone-aware)
    df["Open time"] = pd.to_datetime(df["Open time"], unit="ms", utc=True)
    df.set_index("Open time", inplace=True)

    logging.debug(
        f"DataFrame prepared with columns: {df.columns.tolist()} and index range from {df.index.min()} to {df.index.max()}."
    )

    return df[["Open", "High", "Low", "Close", "Volume"]]


def calculate_nadaraya_watson(
    data: np.ndarray, h: int = NADARAYA_H, mult: float = NADARAYA_MULT
) -> Tuple[np.ndarray, float]:
    """
    Calculate the Nadaraya-Watson estimator and Mean Absolute Error (MAE).
    """
    logging.debug(f"Calculating Nadaraya-Watson estimator with h={h} and mult={mult}.")

    # Apply Gaussian filter for Nadaraya-Watson estimation
    y = gaussian_filter1d(data, sigma=h)
    logging.debug(f"Nadaraya-Watson estimator calculated.")

    # Calculate Mean Absolute Error
    sum_e = np.sum(np.abs(data - y))
    mae = (sum_e / len(data)) * mult
    logging.debug(f"Mean Absolute Error (MAE) calculated: {mae}.")

    return y, mae


def calculate_ema(prices: pd.Series, period: int) -> pd.Series:
    """
    Calculate the Exponential Moving Average (EMA).
    """
    return prices.ewm(span=period, adjust=False).mean()


def calculate_atr(data: pd.DataFrame, period: int) -> pd.Series:
    """
    Calculate the Average True Range (ATR).
    """
    high = data["High"]
    low = data["Low"]
    close = data["Close"]
    previous_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - previous_close).abs()
    tr3 = (low - previous_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr


def calculate_rsi(prices: pd.Series, period: int) -> pd.Series:
    """
    Calculate the Relative Strength Index (RSI).
    """
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=(period - 1), adjust=False).mean()
    avg_loss = loss.ewm(com=(period - 1), adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def generate_signals(
    y: np.ndarray, data: np.ndarray, mae: float, data_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Generate Buy/Sell signals based on price crossing the upper and lower bands and EMA confirmation.
    """
    logging.debug("Generating Buy/Sell signals and calculating bands.")

    df_nw = pd.DataFrame({"y": y, "data": data})

    # Calculate Upper and Lower Bands
    df_nw["Upper_Band"] = df_nw["y"] + mae
    df_nw["Lower_Band"] = df_nw["y"] - mae
    logging.debug("Upper and Lower bands calculated.")

    # Compute previous data points
    df_nw["data_prev"] = df_nw["data"].shift(1)
    df_nw["Upper_Band_prev"] = df_nw["Upper_Band"].shift(1)
    df_nw["Lower_Band_prev"] = df_nw["Lower_Band"].shift(1)

    # Include EMA in the DataFrame
    df_nw["EMA"] = data_df["EMA"].values

    # Generate Buy Signal when price crosses below Lower Band from above and above EMA
    df_nw["Buy_Signal"] = np.where(
        (df_nw["data_prev"] > df_nw["Lower_Band_prev"])
        & (df_nw["data"] <= df_nw["Lower_Band"])
        & (df_nw["data"] > df_nw["EMA"]),  # Price above EMA
        df_nw["data"],
        np.nan,
    )

    # Generate Sell Signal when price crosses above Upper Band from below and below EMA
    df_nw["Sell_Signal"] = np.where(
        (df_nw["data_prev"] < df_nw["Upper_Band_prev"])
        & (df_nw["data"] >= df_nw["Upper_Band"])
        & (df_nw["data"] < df_nw["EMA"]),  # Price below EMA
        df_nw["data"],
        np.nan,
    )

    # Log the number of signals
    num_buy_signals = df_nw["Buy_Signal"].notna().sum()
    num_sell_signals = df_nw["Sell_Signal"].notna().sum()
    logging.info(
        f"Total Buy Signals: {num_buy_signals}, Total Sell Signals: {num_sell_signals}."
    )

    return df_nw[
        [
            "y",
            "Buy_Signal",
            "Sell_Signal",
            "Upper_Band",
            "Lower_Band",
        ]
    ]


def execute_trade(
    signal: str, current_price: float, CURRENT_ATR: float, CURRENT_RSI: float
):
    """
    Execute trade based on the signal.
    """
    global \
        POSITION_SIDE, \
        ENTRY_PRICE, \
        POSITION_SIZE, \
        TOTAL_PNL, \
        WIN_TRADES, \
        TOTAL_TRADES, \
        CUMULATIVE_PNL, \
        RETURN_HISTORY

    logging.debug(
        f"Executing trade: Signal={signal}, Current Price={current_price}, Current Position={POSITION_SIDE}, Entry Price={ENTRY_PRICE}, Position Size={POSITION_SIZE}"
    )

    trade_id = str(uuid.uuid4())  # Unique identifier for each trade action

    # Calculate ATR-based volatility factor
    ATR_mean = data_df["ATR"].mean()
    ATR_min = data_df["ATR"].min()
    ATR_max = data_df["ATR"].max()

    # Normalize the CURRENT_ATR
    if ATR_max - ATR_min != 0:
        ATR_normalized = (CURRENT_ATR - ATR_min) / (ATR_max - ATR_min)
    else:
        ATR_normalized = 0.5  # default value

    # Invert the ATR for sizing (lower ATR leads to higher size)
    volatility_factor = 1 - ATR_normalized

    # Determine DCA size within limits
    DCA_size = BASE_POSITION_SIZE * volatility_factor
    DCA_size = max(min(DCA_size, MAX_DCA_SIZE), MIN_DCA_SIZE)

    if signal == "BUY":
        if POSITION_SIDE == "LONG":
            # Only DCA if RSI is within favorable range
            if 30 < CURRENT_RSI < 60:
                # Adjust POSITION_SIZE
                previous_size = POSITION_SIZE
                POSITION_SIZE += DCA_size

                # Weighted average entry price
                new_entry_price = (
                    (ENTRY_PRICE * previous_size) + (current_price * DCA_size)
                ) / POSITION_SIZE
                logging.info(
                    f"DCA LONG: Old Entry Price: {ENTRY_PRICE}, New Entry Price: {new_entry_price}, DCA Size: {DCA_size}, Total Position Size: {POSITION_SIZE}"
                )
                ENTRY_PRICE = new_entry_price
                log_trade("DCA LONG", current_price, 0, "LONG", trade_id=trade_id)
            else:
                logging.info(f"DCA LONG skipped due to RSI={CURRENT_RSI}")
        elif POSITION_SIDE == "SHORT":
            # Close SHORT and Open LONG
            pnl = (ENTRY_PRICE - current_price) * POSITION_SIZE  # Profit from short
            TOTAL_PNL += pnl
            CUMULATIVE_PNL.append(TOTAL_PNL)
            RETURN_HISTORY.append(pnl)
            if pnl > 0:
                WIN_TRADES += 1
                logging.info(f"Short position closed profitably: PnL={pnl}")
            else:
                logging.info(f"Short position closed with loss: PnL={pnl}")
            TOTAL_TRADES += 1
            log_trade("Close SHORT", current_price, pnl, None, trade_id=trade_id)

            # Open LONG
            POSITION_SIDE = "LONG"
            ENTRY_PRICE = current_price
            POSITION_SIZE = BASE_POSITION_SIZE
            logging.info(
                f"Opened LONG position at {current_price} with size {POSITION_SIZE}"
            )
            log_trade("Open LONG", current_price, 0, "LONG", trade_id=trade_id)
        else:
            # Open LONG
            POSITION_SIDE = "LONG"
            ENTRY_PRICE = current_price
            POSITION_SIZE = BASE_POSITION_SIZE
            logging.info(
                f"Opened LONG position at {current_price} with size {POSITION_SIZE}"
            )
            log_trade("Open LONG", current_price, 0, "LONG", trade_id=trade_id)

    elif signal == "SELL":
        if POSITION_SIDE == "SHORT":
            # Only DCA if RSI is within favorable range
            if 40 < CURRENT_RSI < 70:
                # Adjust POSITION_SIZE
                previous_size = POSITION_SIZE
                POSITION_SIZE += DCA_size

                # Weighted average entry price
                new_entry_price = (
                    (ENTRY_PRICE * previous_size) + (current_price * DCA_size)
                ) / POSITION_SIZE
                logging.info(
                    f"DCA SHORT: Old Entry Price: {ENTRY_PRICE}, New Entry Price: {new_entry_price}, DCA Size: {DCA_size}, Total Position Size: {POSITION_SIZE}"
                )
                ENTRY_PRICE = new_entry_price
                log_trade("DCA SHORT", current_price, 0, "SHORT", trade_id=trade_id)
            else:
                logging.info(f"DCA SHORT skipped due to RSI={CURRENT_RSI}")
        elif POSITION_SIDE == "LONG":
            # Close LONG and Open SHORT
            pnl = (current_price - ENTRY_PRICE) * POSITION_SIZE  # Profit from long
            TOTAL_PNL += pnl
            CUMULATIVE_PNL.append(TOTAL_PNL)
            RETURN_HISTORY.append(pnl)
            if pnl > 0:
                WIN_TRADES += 1
                logging.info(f"Long position closed profitably: PnL={pnl}")
            else:
                logging.info(f"Long position closed with loss: PnL={pnl}")
            TOTAL_TRADES += 1
            log_trade("Close LONG", current_price, pnl, None, trade_id=trade_id)

            # Open SHORT
            POSITION_SIDE = "SHORT"
            ENTRY_PRICE = current_price
            POSITION_SIZE = BASE_POSITION_SIZE
            logging.info(
                f"Opened SHORT position at {current_price} with size {POSITION_SIZE}"
            )
            log_trade("Open SHORT", current_price, 0, "SHORT", trade_id=trade_id)
        else:
            # Open SHORT
            POSITION_SIDE = "SHORT"
            ENTRY_PRICE = current_price
            POSITION_SIZE = BASE_POSITION_SIZE
            logging.info(
                f"Opened SHORT position at {current_price} with size {POSITION_SIZE}"
            )
            log_trade("Open SHORT", current_price, 0, "SHORT", trade_id=trade_id)

    logging.debug(
        f"Post-trade Status: Position={POSITION_SIDE}, Entry Price={ENTRY_PRICE}, Position Size={POSITION_SIZE}, Total PnL={TOTAL_PNL}, "
        f"Win Trades={WIN_TRADES}, Total Trades={TOTAL_TRADES}"
    )


def check_risk_management(current_price: float):
    """
    Check and enforce risk management rules like stop-loss and take-profit.
    """
    global \
        POSITION_SIDE, \
        ENTRY_PRICE, \
        POSITION_SIZE, \
        TOTAL_PNL, \
        WIN_TRADES, \
        TOTAL_TRADES, \
        CUMULATIVE_PNL, \
        RETURN_HISTORY

    if POSITION_SIDE == "LONG":
        if (current_price - ENTRY_PRICE) / ENTRY_PRICE >= TAKE_PROFIT_PERCENT:
            # Close position at take profit
            pnl = (current_price - ENTRY_PRICE) * POSITION_SIZE
            TOTAL_PNL += pnl
            CUMULATIVE_PNL.append(TOTAL_PNL)
            RETURN_HISTORY.append(pnl)
            WIN_TRADES += 1
            TOTAL_TRADES += 1
            logging.info(f"Long position closed at take profit: PnL={pnl}")
            log_trade("Close LONG", current_price, pnl, None)
            POSITION_SIDE = None
            ENTRY_PRICE = 0.0
            POSITION_SIZE = 0.0
        elif (current_price - ENTRY_PRICE) / ENTRY_PRICE <= -STOP_LOSS_PERCENT:
            # Close position at stop loss
            pnl = (current_price - ENTRY_PRICE) * POSITION_SIZE
            TOTAL_PNL += pnl
            CUMULATIVE_PNL.append(TOTAL_PNL)
            RETURN_HISTORY.append(pnl)
            TOTAL_TRADES += 1
            logging.info(f"Long position closed at stop loss: PnL={pnl}")
            log_trade("Close LONG", current_price, pnl, None)
            POSITION_SIDE = None
            ENTRY_PRICE = 0.0
            POSITION_SIZE = 0.0

    elif POSITION_SIDE == "SHORT":
        if (ENTRY_PRICE - current_price) / ENTRY_PRICE >= TAKE_PROFIT_PERCENT:
            # Close position at take profit
            pnl = (ENTRY_PRICE - current_price) * POSITION_SIZE
            TOTAL_PNL += pnl
            CUMULATIVE_PNL.append(TOTAL_PNL)
            RETURN_HISTORY.append(pnl)
            WIN_TRADES += 1
            TOTAL_TRADES += 1
            logging.info(f"Short position closed at take profit: PnL={pnl}")
            log_trade("Close SHORT", current_price, pnl, None)
            POSITION_SIDE = None
            ENTRY_PRICE = 0.0
            POSITION_SIZE = 0.0
        elif (ENTRY_PRICE - current_price) / ENTRY_PRICE <= -STOP_LOSS_PERCENT:
            # Close position at stop loss
            pnl = (ENTRY_PRICE - current_price) * POSITION_SIZE
            TOTAL_PNL += pnl
            CUMULATIVE_PNL.append(TOTAL_PNL)
            RETURN_HISTORY.append(pnl)
            TOTAL_TRADES += 1
            logging.info(f"Short position closed at stop loss: PnL={pnl}")
            log_trade("Close SHORT", current_price, pnl, None)
            POSITION_SIDE = None
            ENTRY_PRICE = 0.0
            POSITION_SIZE = 0.0


def main():
    global \
        POSITION_SIDE, \
        ENTRY_PRICE, \
        POSITION_SIZE, \
        TOTAL_PNL, \
        WIN_TRADES, \
        TOTAL_TRADES, \
        CUMULATIVE_PNL, \
        RETURN_HISTORY, \
        data_df

    try:
        # Fetch initial data
        data_df = live_data()
        if data_df.empty:
            logging.error("Exiting due to no data fetched initially.")
            return

        close_prices = data_df["Close"].values
        data_index = data_df.index

        # Calculate EMA and add to data_df
        ema_period = 20
        data_df["EMA"] = calculate_ema(data_df["Close"], ema_period)

        # Calculate ATR and add to data_df
        atr_period = 14
        data_df["ATR"] = calculate_atr(data_df, atr_period)

        # Calculate RSI and add to data_df
        rsi_period = 14
        data_df["RSI"] = calculate_rsi(data_df["Close"], rsi_period)

        # Calculate Nadaraya-Watson estimator and MAE
        y, mae = calculate_nadaraya_watson(close_prices)

        # Generate signals
        nw_df = generate_signals(y, close_prices, mae, data_df)

        # Log the initial data to the data log CSV
        for idx in range(len(nw_df)):
            row = nw_df.iloc[idx]
            timestamp = data_index[idx]
            open_ = data_df["Open"].iloc[idx]
            high = data_df["High"].iloc[idx]
            low = data_df["Low"].iloc[idx]
            close = data_df["Close"].iloc[idx]
            volume = data_df["Volume"].iloc[idx]
            append_data_log(
                timestamp,
                open_,
                high,
                low,
                close,
                volume,
                row["y"],
                row["Upper_Band"],
                row["Lower_Band"],
                row["Buy_Signal"],
                row["Sell_Signal"],
            )
            logging.debug(f"Logged initial data point at {timestamp}.")

        # Initialize last processed candle time
        last_time = data_df.index[-1]
        last_performance_log_time = datetime.now(timezone.utc)

        logging.info("Starting live trading loop.")

        # Enter main loop for live data
        while True:
            try:
                # Fetch the latest closed candle (second last candle to ensure it's closed)
                latest_candles = client.get_klines(
                    symbol=SYMBOL, interval=INTERVAL, limit=2
                )
                if latest_candles and len(latest_candles) >= 2:
                    candle = latest_candles[-2]
                    candle_time = datetime.fromtimestamp(
                        candle[0] / 1000, tz=timezone.utc
                    )
                    logging.debug(
                        f"Fetched new candle at {candle_time}: Close Price={float(candle[4])}"
                    )
                else:
                    logging.warning("Not enough candle data fetched.")
                    time.sleep(10)
                    continue

                if candle_time <= last_time:
                    # No new candle
                    logging.debug("No new candle to process.")
                    time.sleep(10)
                    continue

                # Append new candle to DataFrame
                new_data = {
                    "Open": float(candle[1]),
                    "High": float(candle[2]),
                    "Low": float(candle[3]),
                    "Close": float(candle[4]),
                    "Volume": float(candle[5]),
                }
                # Use pandas.concat instead of append (append is deprecated)
                new_series = pd.Series(new_data, name=candle_time)
                data_df = pd.concat([data_df, new_series.to_frame().T])
                close_prices = np.append(close_prices, new_data["Close"])
                data_index = data_df.index

                # Update last_time
                last_time = candle_time
                logging.debug(f"Updated last_time to {last_time}.")

                # Calculate EMA and add to data_df
                data_df["EMA"] = calculate_ema(data_df["Close"], ema_period)

                # Calculate ATR and add to data_df
                data_df["ATR"] = calculate_atr(data_df, atr_period)

                # Calculate RSI and add to data_df
                data_df["RSI"] = calculate_rsi(data_df["Close"], rsi_period)

                # Set CURRENT_ATR and CURRENT_RSI
                CURRENT_ATR = data_df["ATR"].iloc[-1]
                CURRENT_RSI = data_df["RSI"].iloc[-1]

                # Recalculate Nadaraya-Watson and MAE
                y, mae = calculate_nadaraya_watson(close_prices)

                # Generate signals
                nw_df = generate_signals(y, close_prices, mae, data_df)

                # Append the latest data point to the data log
                latest_row = nw_df.iloc[-1]
                append_data_log(
                    candle_time,
                    new_data["Open"],
                    new_data["High"],
                    new_data["Low"],
                    new_data["Close"],
                    new_data["Volume"],
                    latest_row["y"],
                    latest_row["Upper_Band"],
                    latest_row["Lower_Band"],
                    latest_row["Buy_Signal"],
                    latest_row["Sell_Signal"],
                )
                logging.debug(f"Appended new data point at {candle_time}.")

                # Get the latest signal
                latest_signal = None
                if not np.isnan(latest_row["Buy_Signal"]):
                    latest_signal = "BUY"
                elif not np.isnan(latest_row["Sell_Signal"]):
                    latest_signal = "SELL"

                # Execute trade based on signal
                if latest_signal:
                    current_price = new_data["Close"]
                    logging.info(f"Signal detected: {latest_signal} at {current_price}")
                    execute_trade(
                        latest_signal, current_price, CURRENT_ATR, CURRENT_RSI
                    )

                # Check for risk management conditions
                current_price = new_data["Close"]
                check_risk_management(current_price)

                # Log performance every VISUALIZATION_UPDATE_INTERVAL minutes
                current_time = datetime.now(timezone.utc)
                if (
                    current_time - last_performance_log_time
                ).total_seconds() >= VISUALIZATION_UPDATE_INTERVAL * 60:
                    log_performance()
                    last_performance_log_time = current_time

                # Sleep until the next candle
                time_to_sleep = (
                    candle_time + timedelta(minutes=1) - datetime.now(timezone.utc)
                ).total_seconds()
                if time_to_sleep > 0:
                    time.sleep(time_to_sleep)
                else:
                    time.sleep(10)

            except BinanceAPIException as e:
                logging.error(
                    f"Binance API Exception in main loop: {e.status_code} - {e.message}"
                )
                time.sleep(60)
            except Exception as e:
                logging.error(f"Error in main loop: {e}")
                time.sleep(60)
    except Exception as e:
        logging.error(f"Unhandled exception in main(): {e}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.critical(f"Critical error in main execution: {e}")
