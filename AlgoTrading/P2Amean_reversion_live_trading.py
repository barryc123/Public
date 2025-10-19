"""Script to implement live trading of strategy using EMA Crossovers and Relative Strength Index with Alpaca's API"""

import time
from datetime import timedelta, datetime
from logging.handlers import RotatingFileHandler

import pandas as pd
from alpaca.data import CryptoHistoricalDataClient, CryptoBarsRequest, TimeFrame, TimeFrameUnit
from alpaca.trading import TradingClient, OrderSide, MarketOrderRequest, TimeInForce
from decouple import config
import sys
import logging

# Configure the logger
LOG_FILENAME = "mean_reversion_logs.log"
logger = logging.getLogger("MeanReversionLogger")
logger.setLevel(logging.DEBUG)
# File handler
file_handler = RotatingFileHandler(LOG_FILENAME, maxBytes=5 * 1024 * 1024, backupCount=2)  # 5MB per log file
file_handler.setLevel(logging.DEBUG)
# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
# Formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Get API keys
alpaca_api_key = config("ALPACA_MEAN_KEY")
alpaca_secret = config("ALPACA_MEAN_SECRET")

# Instantiate Trading Client
api = TradingClient(api_key=alpaca_api_key, secret_key=alpaca_secret, raw_data=True)

# set some global variables
SYMBOL: str = "symbol"
OPEN: str = "Open"
HIGH: str = "High"
LOW: str = "Low"
CLOSE: str = "Close"
VOLUME: str = "Volume"
TRADE_COUNT: str = "Trade_count"
VWAP: str = "Vwap"
EMA: str= "Ema"
RSI: str = "Rsi"
EQUITY: str = "equity"
LONG: str= "long"

class EmaRsiMeanReversionLiveTradingApp:
    def __init__(self):
        self.symbol = "BCH/USD"
        self.minute_interval = 30
        self.rsi_window = 10
        self.ema_window = 70
        self.upper_rsi_band = 80
        self.lower_rsi_band = 30
        self.client = CryptoHistoricalDataClient(api_key=alpaca_api_key, secret_key=alpaca_secret)
        self.data = None
        self.stop_loss = None
        self.take_profit = None
        self.current_position = None
        self.position_open = None

    def fetch_historical_data(self):
        logger.info("Fetching historical data.")
        data_request = CryptoBarsRequest(
            symbol_or_symbols=self.symbol,
            start=datetime.now() - timedelta(days=5),
            end=datetime.now(),
            timeframe=TimeFrame(self.minute_interval, TimeFrameUnit.Minute),
        )
        self.data = self.client.get_crypto_bars(data_request).df
        # Reset index to isolate 'timestamp' level as it is a multiindex
        self.data = self.data.reset_index(level=SYMBOL, drop=True)
        self.data.index = pd.to_datetime(self.data.index)
        logger.info("Historical data fetched.")

    def fetch_latest_data(self):
        try:
            # Ensure there is existing data to get the last timestamp
            if self.data is not None and not self.data.empty:
                last_timestamp = self.data.index[-1]
                self.data.columns = self.data.columns.str.capitalize()
            else:
                raise ValueError("No existing data found. Fetch historical data first.")

            logger.info("Fetching new data from the last timestamp onward.")
            data_request = CryptoBarsRequest(
                symbol_or_symbols=self.symbol,
                start=last_timestamp + timedelta(seconds=1),  # Avoid duplicate data
                end=datetime.now(),
                timeframe=TimeFrame(self.minute_interval, TimeFrameUnit.Minute),
            )
            new_data = self.client.get_crypto_bars(data_request).df

            if not new_data.empty:
                logger.info("New data fetched, adding to existing dataset.")
                new_data = new_data.reset_index(level=SYMBOL, drop=True)
                new_data.index = pd.to_datetime(new_data.index)
                new_data.columns = new_data.columns.str.capitalize()

                # Ensure new data has only original columns
                original_columns = [OPEN, HIGH, LOW, CLOSE, VOLUME, TRADE_COUNT, VWAP]
                new_data = new_data[original_columns]

                # Add missing columns to new_data
                for col in self.data.columns:
                    if col not in new_data.columns:
                        new_data[col] = None

                # Combine the historical and new data
                combined_data = pd.concat([self.data, new_data]).drop_duplicates()
                self.data = combined_data.sort_index()
                logger.info("Latest data added.")
            else:
                logger.info("No new data available.")
        except Exception as e:
            logger.error(f"Error fetching latest data: {e}")

    def calculate_indicators(self):
        logger.info("Calculating EMA.")
        self.data[EMA] = self.data[CLOSE].ewm(span=self.ema_window, adjust=False).mean()

        logger.info("Calculating RSI.")
        delta = self.data[CLOSE].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_window).mean()
        rs = gain / loss
        self.data[RSI] = 100 - (100 / (1 + rs))
        logger.info("Indicators Calculated.")

    def calculate_position_size(self):
        logger.info("Calculating position size to trade.")
        account = api.get_account()
        equity = float(account[EQUITY])  # Get the current account equity
        risk_amount = equity * 0.02  # 2% of equity
        current_price = self.data[CLOSE].iloc[-1]

        # Ensure current price is valid and greater than zero
        if current_price <= 0:
            raise ValueError("Invalid current price for position size calculation.")

        position_size = risk_amount / current_price  # Number of units to trade
        return position_size

    def execute_trade(self, side: OrderSide, size: float):
        try:
            logger.info(f"Executing {side} order of size {size}")
            order = MarketOrderRequest(
                symbol=self.symbol.replace("/", ""),
                qty=size,
                side=side,
                time_in_force=TimeInForce.GTC
            )
            api.submit_order(order)
            logger.info(f"Submitted {side} order with size {size}.")
            logger.info("Verifying if order has been executed.")
            if api.get_all_positions():
                logger.info("Open position found.")
                self.position_open = True
                # can only go long
                self.current_position = LONG
                logger.info("Setting stop loss and take profit levels.")
                self.stop_loss = self.data[CLOSE].iloc[-1] * 0.95
                self.take_profit = self.data[CLOSE].iloc[-1] * 1.10
            else:
                logger.info("No open position found, resetting position tracker, stop loss and take profit levels.")
                self.position_open = False
                self.current_position = None
                self.stop_loss = None
                self.take_profit = None

        except Exception as e:
            logger.info("Error trying to execute trade, handling...")
            logger.error(f"Error executing trade: {e}")
            logger.info("Verifying if there is an open position.")
            if api.get_all_positions():
                self.position_open = True
            else:
                self.position_open = False

    def manage_position(self):
        logger.info("Checking if sigal detected or stop loss / take profit hit.")
        if self.current_position == LONG:
            if (self.data[CLOSE].iloc[-1] <= self.stop_loss or self.data[CLOSE].iloc[-1] >= self.take_profit or
                    self.data[RSI].iloc[-1] > self.upper_rsi_band):
                logger.info("Closing long position.")
                self.close_position()


    def close_position(self):
        logger.info("Closing the current position.")
        self.execute_trade(OrderSide.SELL if self.current_position == LONG else OrderSide.BUY,
                           size=self.calculate_position_size())

    def run(self):
        start_time = datetime.now()
        # script will only run for the length of the max_duration
        # it should be scheduled via cron job or Windows Task Scheduler to run multiple times per day to avoid
        # cache building up
        max_duration = timedelta(minutes=240)
        self.fetch_historical_data()

        while True:
            if datetime.now() - start_time > max_duration:
                logger.info(f"Reached maximum runtime of {max_duration}. Exiting.")
                sys.exit()
            logger.info("Fetching latest data...")
            self.fetch_latest_data()
            self.calculate_indicators()

            # Check for open positions
            if self.position_open:
                logger.info("Managing existing position.")
                self.manage_position()
            else:
                logger.info("Checking for buy signals.")
                position_size = self.calculate_position_size()

                if (
                        self.data[CLOSE].iloc[-2] > self.data[EMA].iloc[-2]  # Previous close above EMA
                        and self.data[CLOSE].iloc[-1] < self.data[EMA].iloc[
                    -1]  # Current close below EMA (crossover)
                        and self.data[RSI].iloc[-1] < self.lower_rsi_band  # RSI below lower band
                ):
                    logger.info("Buy signal detected.")
                    logger.info("Attempting to execute trade.")
                    self.execute_trade(OrderSide.BUY, size=position_size)

            # set script to sleep to avoid making too many calls to API
            # 200 calls per minute is the maximum allowed
            # this script makes much fewer calls than the permitted maximum
            sleep_amount = 60
            logger.info(f"Sleeping for {sleep_amount} seconds")
            time.sleep(sleep_amount)

if __name__ == "__main__":
    try:
        app = EmaRsiMeanReversionLiveTradingApp()
        app.run()
    except KeyboardInterrupt:
        print("Program interrupted by user. Exiting.")
        logger.info("Program interrupted by user. Exiting.")