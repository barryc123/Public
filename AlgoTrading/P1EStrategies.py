"""Module to store the mean reversion and trend following strategies used."""
import pandas as pd
from backtesting import Strategy
from backtesting.lib import crossover
from P1Ftechnical_indicators import calculate_ema, calculate_rsi_smoothed, calculate_bollinger_bands, calculate_macd, \
    calculate_adx

CLOSE: str = "Close"
EQUITY_FRACTION_TO_TRADE: float = 0.02

def return_to_drawdown_optimiser(series):
    return abs(series["Return [%]"] / series["Max. Drawdown [%]"])


class EmaRsiMeanReversion(Strategy):
    upper_rsi_band = 80
    lower_rsi_band = 30
    rsi_window = 10
    ema_window = 70
    name = "EmaRsiMeanReversion"

    def init(self):
        self.ema = self.I(calculate_ema, self.data[CLOSE], self.ema_window)
        self.rsi = self.I(calculate_rsi_smoothed, self.data[CLOSE], self.rsi_window)

    def next(self):
        # Buy Signal: Price crosses below EMA & RSI < 40
        if (crossover(self.ema, self.data[CLOSE]) and self.data[CLOSE][-1] < self.ema[-1] and
                self.rsi[-1] < self.lower_rsi_band and not self.position.is_long):
            self.buy(size=EQUITY_FRACTION_TO_TRADE)

            # Set stop loss and take profit level for long
            self.stop_loss = self.data[CLOSE][-1] * 0.95  # Stop loss at 5% below entry price
            self.take_profit = self.data[CLOSE][-1] * 1.10 # Take profit at 10% above entry price

        # Close long position
        # if stop loss/ take profit triggered or based on trend
        if self.position.is_long:
            if (self.data[CLOSE][-1] <= self.stop_loss or self.data[CLOSE][-1] >= self.take_profit
                    or self.rsi[-1] > self.upper_rsi_band):
                self.position.close()

        elif self.position.is_short:
            # Should only have long position so exit if short
            self.position.close()


class BbRsiMeanReversion(Strategy):
    bb_window = 20
    rsi_window = 14
    lower_rsi_band = 30
    upper_rsi_band = 70
    name = "BbRsiMeanReversion"

    def init(self):
        # Calc BBs
        self.bb_sma, self.upper_bb, self.lower_bb = self.I(lambda prices:
                                                           calculate_bollinger_bands(prices, self.bb_window),
                                                           self.data[CLOSE])
        # Calculate RSI
        self.rsi = self.I(calculate_rsi_smoothed, self.data[CLOSE], self.rsi_window)

    def next(self):
        # Buy Signal: Price < Lower BB and RSI < 30
        if (crossover(self.data[CLOSE], self.lower_bb) and self.data[CLOSE][-1] < self.lower_bb[-1] and
            self.rsi[-1] < self.lower_rsi_band and not self.position.is_long):
            self.buy(size=EQUITY_FRACTION_TO_TRADE)

            # Set stop loss and take profit level for long
            self.stop_loss = self.data[CLOSE][-1] * 0.95  # Stop loss at 5% below entry price
            self.take_profit = self.data[CLOSE][-1] * 1.10  # Take profit at 10% above entry price


        # Close long position
        # if stop loss/ take profit triggered or based on trend
        if self.position.is_long:
            if (self.rsi[-1] > self.upper_rsi_band or self.data[CLOSE][-1] <= self.stop_loss or
                    self.data[CLOSE][-1] >= self.take_profit):
                self.position.close()

        elif self.position.is_short:
            # Should only have long position so exit if short
            self.position.close()


class MacdAdxTrendFollowing(Strategy):
    adx_window = 13
    adx_threshold = 25
    adx_exit_threshold = 20
    macd_short_window = 14
    macd_long_window = 28
    macd_signal_window = 9
    name = "MacdAdxTrendFollowing"

    def init(self):
        # Calculate MACD components
        self.macd, self.signal, _ = self.I(calculate_macd,
                                           self.data[CLOSE],
                                           self.macd_short_window,
                                           self.macd_long_window,
                                           self.macd_signal_window)

        # Calculate ADX and directional indices
        adx_data = calculate_adx(self.data.df.copy(), self.adx_window)
        self.adx = self.I(lambda x: adx_data["ADX"].values, self.data[CLOSE])
        self.plus_di = self.I(lambda x: adx_data["+DI"].values, self.data[CLOSE])
        self.minus_di = self.I(lambda x: adx_data["-DI"].values, self.data[CLOSE])

    def next(self):
        # Buy Signal: MACD line crosses above Signal line, +DI > -DI, ADX > threshold
        if (crossover(self.macd, self.signal) and
                self.plus_di[-1] > self.minus_di[-1] and
                self.adx[-1] > self.adx_threshold and not self.position.is_long):
            self.buy(size=EQUITY_FRACTION_TO_TRADE)

            # Set stop loss and take profit level for long
            self.stop_loss = self.data[CLOSE][-1] * 0.95  # Stop loss at 5% below entry price
            self.take_profit = self.data[CLOSE][-1] * 1.10  # Take profit at 10% above entry price


        # Exit Condition: ADX falls below exit threshold or stop loss/ take profit triggered
        if self.position.is_long:
            if (self.adx[-1] < self.adx_exit_threshold or self.data[CLOSE][-1] <= self.stop_loss
                    or self.data[CLOSE][-1] >= self.take_profit):
                self.position.close()

        elif self.position.is_short:
            # Should only have long position so exit if short
            self.position.close()



class EmaAdxTrendFollowing(Strategy):
    adx_window = 14
    adx_threshold = 25
    adx_exit_threshold = 20
    ema_short_window = 12
    ema_long_window = 26
    name = "EmaAdxTrendFollowing"

    def init(self):
        # Calculate EMAs
        self.short_ema = self.I(lambda prices: pd.Series(prices).ewm(span=self.ema_short_window, adjust=False).mean(),
                                self.data[CLOSE])
        self.long_ema = self.I(lambda prices: pd.Series(prices).ewm(span=self.ema_long_window, adjust=False).mean(),
                               self.data[CLOSE])

        # Calculate ADX and directional indices
        adx_data = calculate_adx(self.data.df.copy(), self.adx_window)
        self.adx = self.I(lambda x: adx_data["ADX"].values, self.data[CLOSE])
        self.plus_di = self.I(lambda x: adx_data["+DI"].values, self.data[CLOSE])
        self.minus_di = self.I(lambda x: adx_data["-DI"].values, self.data[CLOSE])


    def next(self):
        if not self.position:
        # Buy Signal: Short EMA crosses above Long EMA, +DI > -DI, ADX > threshold
            if (crossover(self.short_ema, self.long_ema) and
                    self.plus_di[-1] > self.minus_di[-1] and
                    self.adx[-1] > self.adx_threshold):
                self.buy(size=EQUITY_FRACTION_TO_TRADE)

                # Set stop loss and take profit level for long
                self.stop_loss = self.data[CLOSE][-1] * 0.95  # Stop loss at 5% below entry price
                self.take_profit = self.data[CLOSE][-1] * 1.10  # Take profit at 10% above entry price


            # Exit Condition: ADX falls below exit threshold or stop loss/ take profit triggered
            if self.position.is_long:
                if (self.adx[-1] < self.adx_exit_threshold or self.data[CLOSE][-1] <= self.stop_loss
                        or self.data[CLOSE][-1] >= self.take_profit):
                    self.position.close()
            elif self.position.is_short:
                # Should only have long position so exit if short
                self.position.close()
