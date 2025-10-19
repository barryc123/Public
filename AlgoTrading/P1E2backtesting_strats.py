import os
import time

import pandas as pd
from backtesting import Backtest
from P1EStrategies import EmaAdxTrendFollowing, return_to_drawdown_optimiser, MacdAdxTrendFollowing, BbRsiMeanReversion, \
    EmaRsiMeanReversion

CLOSE: str = "Close"
DATE: str = "Date"
cash: int = 1000000
commission: float = 0.002
strategy_list = [EmaRsiMeanReversion, BbRsiMeanReversion, MacdAdxTrendFollowing, EmaAdxTrendFollowing]
sym_list = ["BCHUSD", "ETHUSD", "USDTUSD"]
# different minute windows to resample data
freq_window = [5, 15, 30, 60]

start_date: str = '2023-7-1'
end_date:str  = '2023-12-31'


# create a df to store results
results_df = pd.DataFrame(columns=["Return %", "Ann. Ret %", "Ann. Vol %", "Max Drawdown %", "R2D Ratio", "Win Rate %", "Num Trades", "Optimal Params"])
for strategy in strategy_list:
    for sym in sym_list:
        # read in the data
        data = pd.read_csv(f"{sym}_2023.csv")
        data.columns = data.columns.str.capitalize()
        data[DATE] = pd.to_datetime(data[DATE])
        data = data.set_index(DATE)
        data = data.sort_index()
        data = data.loc[start_date: end_date]

        for freq in freq_window:
            if freq == 5:
                freq_df = data.copy()
            # resample data if freq not equal to 5
            else:
                freq_df = data.copy().resample(f"{freq}min").agg(
                    {"Open": "first",
                     "High": "max",
                     "Low": "min",
                     "Close": "last",
                     "Volume": "sum",
                     "Trade_count": "sum"}
                )

                freq_df = freq_df.dropna()

            bt = Backtest(freq_df, strategy, cash=cash, commission=commission)

            # change the optimisation parameters based on the strategy inputs
            if strategy == EmaAdxTrendFollowing:
                stats = bt.optimize(
                    method="grid",
                    adx_window=range(7,16,2),
                    adx_threshold=25,
                    adx_exit_threshold=20,
                    ema_short_window=range(5,12,1),
                    ema_long_window=range(50, 110, 10),
                    maximize=return_to_drawdown_optimiser,
                )

            elif strategy == MacdAdxTrendFollowing:
                stats =  bt.optimize(
                    method="grid",
                    adx_window=range(7,16,2),
                    adx_threshold=25,
                    adx_exit_threshold=20,
                    macd_short_window=range(8,16,2),
                    macd_long_window=range(24, 30, 2),
                    macd_signal_window=9,
                    maximize=return_to_drawdown_optimiser,
                )

            elif strategy == BbRsiMeanReversion:
                stats =  bt.optimize(
                    method="grid",
                    bb_window=range(16, 24, 2),
                    rsi_window=range(8, 18, 2),
                    lower_rsi_band=range(10,50,10),
                    upper_rsi_band=range(50,100,10),
                    constraint=lambda param: param.upper_rsi_band > param.lower_rsi_band,
                    maximize=return_to_drawdown_optimiser,
                )

            elif strategy == EmaRsiMeanReversion:
                stats = bt.optimize(
                    method="grid",
                    rsi_window=range(8, 18, 2),
                    lower_rsi_band=range(10, 50, 10),
                    upper_rsi_band=range(50, 100, 10),
                    constraint=lambda param: param.upper_rsi_band > param.lower_rsi_band,
                    ema_window=range(30,80, 10),
                    maximize=return_to_drawdown_optimiser,
                )


            return_perc =  stats["Return [%]"]
            annualised_return = stats["Return (Ann.) [%]"]
            annual_vol = stats["Volatility (Ann.) [%]"]
            max_drawdown_perc = stats["Max. Drawdown [%]"]
            return_to_drawdown_ratio = abs(return_perc / max_drawdown_perc) if (return_perc != 0 and max_drawdown_perc!=0) else float("nan")
            win_rate = stats["Win Rate [%]"]
            num_trades = stats["# Trades"]
            # add the optimal params for the strategy as a list
            optimal_params = []
            for param, value in stats._strategy._params.items():
                optimal_params.append(f"{param}: {value}")

            results_df.loc[f"{strategy.name}_{sym}_{freq}min"] = {
                "Return %": return_perc,
                "Ann. Ret %": annualised_return,
                "Ann. Vol %": annual_vol,
                "Max Drawdown %": max_drawdown_perc,
                "R2D Ratio": return_to_drawdown_ratio,
                "Win Rate %": win_rate,
                "Num Trades": num_trades,
                "Optimal Params": optimal_params
            }
            print(f"frequency of {freq} done for {sym} for {strategy.name} strategy")
            pass
        print("Sleeping for 12 seconds")
        time.sleep(5)
        print("Sleeping done. ")

    # plot the result once optimisations done
    #bt.plot(filename="plots/plot.html", resample=False)
    output_dir = "historical_data"
    os.makedirs(output_dir, exist_ok=True)
    csv_filename = os.path.join(output_dir, f"opt_results_{start_date}_{end_date}.csv")
    results_df.to_csv(csv_filename)
    pass