from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go

TRADING_DAYS = 252


@dataclass(frozen=True)
class ConvexityExample:
    """
    Store the inputs and outputs for the convex payoff example.

    :param strike: Strike price used by the call option payoff.
    :type strike: float
    :param stock_scenarios: Possible stock prices at expiry.
    :type stock_scenarios: np.ndarray
    :param probabilities: Probability attached to each stock price scenario.
    :type probabilities: np.ndarray
    :param expected_stock: Probability-weighted expected stock price.
    :type expected_stock: float
    :param expected_payoff: Probability-weighted expected call payoff.
    :type expected_payoff: float
    :param payoff_at_expected_stock: Call payoff evaluated at the expected stock price.
    :type payoff_at_expected_stock: float
    :param figure: Plotly figure comparing expected payoff with payoff at expected stock.
    :type figure: go.Figure
    """

    strike: float
    stock_scenarios: np.ndarray
    probabilities: np.ndarray
    expected_stock: float
    expected_payoff: float
    payoff_at_expected_stock: float
    figure: go.Figure


def create_rng(seed: int) -> np.random.Generator:
    """
    Create a deterministic NumPy random number generator.

    :param seed: Seed used to initialise the generator.
    :type seed: int
    :return: Initialised NumPy random number generator.
    :rtype: np.random.Generator
    """
    return np.random.default_rng(seed=seed)


def call_payoff(stock_price: np.ndarray, strike: float) -> np.ndarray:
    """
    Compute a European call option payoff.

    :param stock_price: Stock prices at expiry.
    :type stock_price: np.ndarray
    :param strike: Strike price of the call option.
    :type strike: float
    :return: Payoff for each stock price.
    :rtype: np.ndarray
    """
    return np.maximum(stock_price - strike, 0.0)


def build_convexity_example(
    strike: float = 100.0,
    stock_scenarios: np.ndarray | None = None,
    probabilities: np.ndarray | None = None,
) -> ConvexityExample:
    """
    Build the Jensen inequality call payoff example.

    :param strike: Strike price used by the call option payoff.
    :type strike: float
    :param stock_scenarios: Optional stock prices at expiry.
    :type stock_scenarios: np.ndarray | None
    :param probabilities: Optional scenario probabilities.
    :type probabilities: np.ndarray | None
    :return: Inputs, calculated expectations, and the Plotly figure.
    :rtype: ConvexityExample
    """
    if stock_scenarios is None:
        stock_scenarios = np.array([70.0, 90.0, 100.0, 110.0, 130.0])
    if probabilities is None:
        probabilities = np.array([0.15, 0.20, 0.30, 0.20, 0.15])

    expected_stock = float(np.dot(probabilities, stock_scenarios))
    expected_payoff = float(np.dot(probabilities, call_payoff(stock_scenarios, strike)))
    payoff_at_expected_stock = float(call_payoff(np.array([expected_stock]), strike)[0])

    payoff_grid = np.linspace(60.0, 140.0, 200)
    payoff_curve = call_payoff(payoff_grid, strike)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=payoff_grid,
            y=payoff_curve,
            mode="lines",
            name="Call payoff",
            line={"width": 3},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=stock_scenarios,
            y=call_payoff(stock_scenarios, strike),
            mode="markers",
            name="Scenarios",
            marker={"size": probabilities * 70 + 8, "opacity": 0.75},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[expected_stock],
            y=[payoff_at_expected_stock],
            mode="markers+text",
            name="Payoff at expected stock",
            text=[f"f(E[S]) = {payoff_at_expected_stock:.2f}"],
            textposition="bottom right",
            marker={"size": 12, "symbol": "diamond"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[expected_stock],
            y=[expected_payoff],
            mode="markers+text",
            name="Expected payoff",
            text=[f"E[f(S)] = {expected_payoff:.2f}"],
            textposition="top left",
            marker={"size": 12, "symbol": "x"},
        )
    )
    fig.update_layout(
        title="Convexity: expected payoff versus payoff at expected stock price",
        xaxis_title="Stock price at expiry",
        yaxis_title="Payoff",
        template="plotly_white",
        hovermode="x unified",
    )

    return ConvexityExample(
        strike=strike,
        stock_scenarios=stock_scenarios,
        probabilities=probabilities,
        expected_stock=expected_stock,
        expected_payoff=expected_payoff,
        payoff_at_expected_stock=payoff_at_expected_stock,
        figure=fig,
    )


def simulate_price_path(
    initial_price: float,
    drift: float,
    volatility: float,
    years: float,
    steps_per_year: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Simulate one discrete random-walk asset price path.

    :param initial_price: Starting asset price.
    :type initial_price: float
    :param drift: Annual drift rate.
    :type drift: float
    :param volatility: Annual volatility.
    :type volatility: float
    :param years: Number of years to simulate.
    :type years: float
    :param steps_per_year: Number of simulation steps in one year.
    :type steps_per_year: int
    :param rng: Random number generator used for shocks.
    :type rng: np.random.Generator
    :return: DataFrame containing time, price, and simple return columns.
    :rtype: pd.DataFrame
    """
    steps = int(years * steps_per_year)
    dt = 1.0 / steps_per_year
    shocks = rng.normal(loc=0.0, scale=1.0, size=steps)
    returns = drift * dt + volatility * math.sqrt(dt) * shocks
    prices = initial_price * np.cumprod(np.insert(1.0 + returns, 0, 1.0))
    time = np.arange(steps + 1) * dt
    return pd.DataFrame({"time": time, "price": prices, "return": np.insert(returns, 0, np.nan)})


def plot_price_path(price_path: pd.DataFrame) -> go.Figure:
    """
    Plot a single simulated asset price path.

    :param price_path: DataFrame containing time and price columns.
    :type price_path: pd.DataFrame
    :return: Plotly figure showing the asset price over time.
    :rtype: go.Figure
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=price_path["time"],
            y=price_path["price"],
            mode="lines",
            name="Synthetic asset price",
            line={"width": 2.5},
        )
    )
    fig.update_layout(
        title="Synthetic asset price path",
        xaxis_title="Time in years",
        yaxis_title="Price",
        template="plotly_white",
    )
    return fig


def summarise_returns(returns: pd.Series) -> pd.DataFrame:
    """
    Summarise daily returns with their sample mean and standard deviation.

    :param returns: Series of simple returns.
    :type returns: pd.Series
    :return: DataFrame containing return summary statistics.
    :rtype: pd.DataFrame
    """
    return pd.DataFrame(
        {
            "Statistic": ["Mean daily return", "Sample daily standard deviation"],
            "Value": [float(returns.mean()), float(returns.std(ddof=1))],
        }
    )


def plot_scaled_returns(returns: pd.Series) -> go.Figure:
    """
    Plot scaled returns against the standard Normal density.

    :param returns: Series of simple returns.
    :type returns: pd.Series
    :return: Plotly figure with a histogram and standard Normal density curve.
    :rtype: go.Figure
    """
    scaled_returns = (returns - returns.mean()) / returns.std(ddof=1)
    x_grid = np.linspace(-4.0, 4.0, 400)
    normal_density = np.exp(-0.5 * x_grid**2) / math.sqrt(2.0 * math.pi)

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=scaled_returns,
            histnorm="probability density",
            nbinsx=35,
            name="Scaled synthetic returns",
            opacity=0.70,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_grid,
            y=normal_density,
            mode="lines",
            name="Standard Normal density",
            line={"width": 3},
        )
    )
    fig.update_layout(
        title="Scaled returns versus standard Normal density",
        xaxis_title="Scaled return",
        yaxis_title="Density",
        template="plotly_white",
        bargap=0.02,
    )
    return fig


def annualised_estimates(returns: pd.Series, steps_per_year: int = TRADING_DAYS) -> pd.DataFrame:
    """
    Estimate annualised drift and volatility from simple returns.

    :param returns: Series of simple returns.
    :type returns: pd.Series
    :param steps_per_year: Number of return observations in one year.
    :type steps_per_year: int
    :return: DataFrame containing annualised drift and volatility estimates.
    :rtype: pd.DataFrame
    """
    dt = 1.0 / steps_per_year
    annualised_drift = float(returns.sum() / (len(returns) * dt))
    annualised_volatility = float(
        math.sqrt(((returns - returns.mean()) ** 2).sum() / ((len(returns) - 1) * dt))
    )
    return pd.DataFrame(
        {
            "Estimate": ["Annualised drift", "Annualised volatility"],
            "Value": [annualised_drift, annualised_volatility],
        }
    )


def simulate_many_paths(
    initial_price: float,
    drift: float,
    volatility: float,
    years: float,
    steps_per_year: int,
    path_count: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Simulate multiple discrete random-walk asset price paths.

    :param initial_price: Starting asset price.
    :type initial_price: float
    :param drift: Annual drift rate.
    :type drift: float
    :param volatility: Annual volatility.
    :type volatility: float
    :param years: Number of years to simulate.
    :type years: float
    :param steps_per_year: Number of simulation steps in one year.
    :type steps_per_year: int
    :param path_count: Number of paths to simulate.
    :type path_count: int
    :param rng: Random number generator used for shocks.
    :type rng: np.random.Generator
    :return: DataFrame indexed by time with one column per path.
    :rtype: pd.DataFrame
    """
    steps = int(years * steps_per_year)
    dt = 1.0 / steps_per_year
    shocks = rng.normal(loc=0.0, scale=1.0, size=(steps, path_count))
    returns = drift * dt + volatility * math.sqrt(dt) * shocks
    prices = initial_price * np.cumprod(np.vstack([np.ones(path_count), 1.0 + returns]), axis=0)
    time = np.arange(steps + 1) * dt
    columns = [f"Path {index + 1}" for index in range(path_count)]
    return pd.DataFrame(prices, index=time, columns=columns)


def plot_paths(
    paths: pd.DataFrame,
    title: str,
    yaxis_title: str = "Price",
    line_width: float = 1.4,
    opacity: float = 0.75,
    showlegend: bool = False,
) -> go.Figure:
    """
    Plot a collection of paths stored in a DataFrame.

    :param paths: DataFrame indexed by time with one column per path.
    :type paths: pd.DataFrame
    :param title: Figure title.
    :type title: str
    :param yaxis_title: Label used for the y-axis.
    :type yaxis_title: str
    :param line_width: Width used for each path line.
    :type line_width: float
    :param opacity: Opacity used for each path line.
    :type opacity: float
    :param showlegend: Whether to show the Plotly legend.
    :type showlegend: bool
    :return: Plotly figure containing all paths.
    :rtype: go.Figure
    """
    fig = go.Figure()
    for column in paths.columns:
        fig.add_trace(
            go.Scatter(
                x=paths.index,
                y=paths[column],
                mode="lines",
                name=column,
                line={"width": line_width},
                opacity=opacity,
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="Time in years",
        yaxis_title=yaxis_title,
        template="plotly_white",
        showlegend=showlegend,
    )
    return fig


def time_scaling_table(mu: float, sigma: float) -> pd.DataFrame:
    """
    Build a table showing drift and volatility scaling over common horizons.

    :param mu: Annual drift rate.
    :type mu: float
    :param sigma: Annual volatility.
    :type sigma: float
    :return: DataFrame of horizons with expected returns and standard deviations.
    :rtype: pd.DataFrame
    """
    horizons = pd.DataFrame(
        {
            "Horizon": ["1 day", "1 week", "1 month", "1 quarter", "1 year"],
            "Years": [1 / 252, 5 / 252, 21 / 252, 63 / 252, 1.0],
        }
    )
    horizons["Expected return"] = mu * horizons["Years"]
    horizons["Return standard deviation"] = sigma * np.sqrt(horizons["Years"])
    return horizons


def plot_time_scaling(horizons: pd.DataFrame) -> go.Figure:
    """
    Plot expected return and return standard deviation by horizon.

    :param horizons: DataFrame returned by :func:`time_scaling_table`.
    :type horizons: pd.DataFrame
    :return: Plotly grouped bar chart for the time scaling table.
    :rtype: go.Figure
    """
    fig = go.Figure()
    fig.add_trace(go.Bar(x=horizons["Horizon"], y=horizons["Expected return"], name="Expected return"))
    fig.add_trace(
        go.Bar(
            x=horizons["Horizon"],
            y=horizons["Return standard deviation"],
            name="Return standard deviation",
        )
    )
    fig.update_layout(
        title="Time scaling: drift versus volatility",
        xaxis_title="Sampling horizon",
        yaxis_title="Scale",
        template="plotly_white",
        barmode="group",
    )
    return fig


def shock_return_series(seed: int = 7, length: int = 160, shock_day: int = 65) -> pd.Series:
    """
    Create a deterministic return series with one large negative shock.

    :param seed: Seed used to initialise the random number generator.
    :type seed: int
    :param length: Number of returns to simulate.
    :type length: int
    :param shock_day: Zero-based index where the shock return is inserted.
    :type shock_day: int
    :return: Series of simulated daily returns.
    :rtype: pd.Series
    """
    rng = create_rng(seed)
    shock_returns = pd.Series(rng.normal(loc=0.0, scale=0.012, size=length), name="return")
    shock_returns.iloc[shock_day] = -0.10
    return shock_returns


def rolling_annualised_volatility(
    returns: pd.Series,
    window: int = 30,
    steps_per_year: int = TRADING_DAYS,
) -> pd.Series:
    """
    Compute rolling annualised volatility from simple returns.

    :param returns: Series of simple returns.
    :type returns: pd.Series
    :param window: Rolling window length in observations.
    :type window: int
    :param steps_per_year: Number of return observations in one year.
    :type steps_per_year: int
    :return: Rolling annualised volatility series.
    :rtype: pd.Series
    """
    return returns.rolling(window=window).std(ddof=1) * math.sqrt(steps_per_year)


def plot_volatility_plateau(
    shock_returns: pd.Series,
    rolling_volatility: pd.Series,
) -> go.Figure:
    """
    Plot daily returns and their rolling annualised volatility.

    :param shock_returns: Series of daily returns containing a shock.
    :type shock_returns: pd.Series
    :param rolling_volatility: Rolling annualised volatility series.
    :type rolling_volatility: pd.Series
    :return: Plotly figure with daily returns and volatility on separate axes.
    :rtype: go.Figure
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=shock_returns.index,
            y=shock_returns,
            mode="lines",
            name="Daily return",
            yaxis="y1",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=rolling_volatility.index,
            y=rolling_volatility,
            mode="lines",
            name="30-day annualised volatility",
            yaxis="y2",
            line={"width": 3},
        )
    )
    fig.update_layout(
        title="Moving-window volatility plateau after a one-day shock",
        xaxis_title="Day",
        yaxis={"title": "Daily return", "tickformat": ".1%"},
        yaxis2={"title": "Annualised volatility", "overlaying": "y", "side": "right", "tickformat": ".0%"},
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


def simulate_wiener_paths(
    years: float,
    steps_per_year: int,
    path_count: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Simulate discrete approximations to Wiener process paths.

    :param years: Number of years to simulate.
    :type years: float
    :param steps_per_year: Number of simulation steps in one year.
    :type steps_per_year: int
    :param path_count: Number of Wiener paths to simulate.
    :type path_count: int
    :param rng: Random number generator used for Normal increments.
    :type rng: np.random.Generator
    :return: DataFrame indexed by time with one column per Wiener path.
    :rtype: pd.DataFrame
    """
    steps = int(years * steps_per_year)
    dt = 1.0 / steps_per_year
    increments = rng.normal(loc=0.0, scale=math.sqrt(dt), size=(steps, path_count))
    paths = np.vstack([np.zeros(path_count), increments.cumsum(axis=0)])
    time = np.arange(steps + 1) * dt
    columns = [f"Wiener path {index + 1}" for index in range(path_count)]
    return pd.DataFrame(paths, index=time, columns=columns)


def plot_wiener_paths(wiener_paths: pd.DataFrame) -> go.Figure:
    """
    Plot simulated Wiener process paths.

    :param wiener_paths: DataFrame indexed by time with one column per path.
    :type wiener_paths: pd.DataFrame
    :return: Plotly figure showing the Wiener paths.
    :rtype: go.Figure
    """
    return plot_paths(
        wiener_paths,
        title="Discrete approximation to Wiener paths",
        yaxis_title="X(t)",
        line_width=1.7,
        opacity=1.0,
        showlegend=False,
    )


def simulate_gbm_paths(
    initial_price: float,
    drift: float,
    volatility: float,
    years: float,
    steps_per_year: int,
    path_count: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Simulate geometric Brownian motion asset price paths.

    :param initial_price: Starting asset price.
    :type initial_price: float
    :param drift: Annual drift rate.
    :type drift: float
    :param volatility: Annual volatility.
    :type volatility: float
    :param years: Number of years to simulate.
    :type years: float
    :param steps_per_year: Number of simulation steps in one year.
    :type steps_per_year: int
    :param path_count: Number of paths to simulate.
    :type path_count: int
    :param rng: Random number generator used for shocks.
    :type rng: np.random.Generator
    :return: DataFrame indexed by time with one column per GBM path.
    :rtype: pd.DataFrame
    """
    steps = int(years * steps_per_year)
    dt = 1.0 / steps_per_year
    shocks = rng.normal(loc=0.0, scale=1.0, size=(steps, path_count))
    log_returns = (drift - 0.5 * volatility**2) * dt + volatility * math.sqrt(dt) * shocks
    prices = initial_price * np.exp(np.vstack([np.zeros(path_count), log_returns]).cumsum(axis=0))
    time = np.arange(steps + 1) * dt
    columns = [f"Path {index + 1}" for index in range(path_count)]
    return pd.DataFrame(prices, index=time, columns=columns)


def plot_gbm_paths(gbm_paths: pd.DataFrame, initial_price: float, drift: float) -> go.Figure:
    """
    Plot geometric Brownian motion paths and their expected exponential path.

    :param gbm_paths: DataFrame indexed by time with one column per GBM path.
    :type gbm_paths: pd.DataFrame
    :param initial_price: Starting asset price used for the expected path.
    :type initial_price: float
    :param drift: Annual drift rate used for the expected path.
    :type drift: float
    :return: Plotly figure containing simulated paths and expected path.
    :rtype: go.Figure
    """
    fig = plot_paths(
        gbm_paths,
        title="GBM-style simulation from dS = mu S dt + sigma S dX",
        line_width=1.2,
        opacity=0.55,
        showlegend=False,
    )
    expected_path = initial_price * np.exp(drift * gbm_paths.index)
    fig.add_trace(
        go.Scatter(
            x=gbm_paths.index,
            y=expected_path,
            mode="lines",
            name="Expected exponential path",
            line={"width": 4, "dash": "dash"},
        )
    )
    return fig
