"""Module to store functionality for getting data."""
import yfinance as yf

def save_ohlc_data_from_yahoo_finance(ticker: str, interval: str, start_time: str, end_time: str, file_name: str) \
        -> None:
    """Function to get OHLC data from Yahoo Finance and save it to a CSV.
    :param ticker: Ticker to get data for, e.g. SPY
    :param interval: Interval to use, e.g. 1d
    :param start_time: Start time string to use, e.g. 2015-11-01
    :param end_time: End time string to use.
    :param file_name: Name to save CSV file as.
    :return: None.
    """
    df = yf.download(ticker, start=start_time, end=end_time, interval=interval)
    df.columns = df.columns.get_level_values(0)
    df.to_csv(f"{file_name}.csv")


def main():
    """Get OHLC data for the SPY Index."""
    save_ohlc_data_from_yahoo_finance(
        ticker="SPY",
        start_time="2015-11-01",
        end_time="2025-11-01",
        interval="1d",
        file_name="SPYOHLC"
    )


if __name__ == "__main__":
    main()
