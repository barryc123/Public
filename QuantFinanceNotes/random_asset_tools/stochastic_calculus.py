from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

TRADING_DAYS = 252


def create_rng(seed: int) -> np.random.Generator:
    """
    Create a deterministic NumPy random number generator.

    :param seed: Seed used to initialise the generator.
    :type seed: int
    :return: Initialised NumPy random number generator.
    :rtype: np.random.Generator
    """
    return np.random.default_rng(seed=seed)


def path_columns(path_count: int, prefix: str = "Path") -> list[str]:
    """
    Build standard column names for simulated paths.

    :param path_count: Number of path columns required.
    :type path_count: int
    :param prefix: Prefix used before the one-based path number.
    :type prefix: str
    :return: List of path column names.
    :rtype: list[str]
    """
    return [f"{prefix} {index + 1}" for index in range(path_count)]


def simulate_random_walk(
    years: float,
    steps_per_year: int,
    path_count: int,
    rng: np.random.Generator,
    initial_value: float = 0.0,
) -> pd.DataFrame:
    """
    Simulate scaled binomial random-walk paths.

    :param years: Number of years to simulate.
    :type years: float
    :param steps_per_year: Number of simulation steps in one year.
    :type steps_per_year: int
    :param path_count: Number of paths to simulate.
    :type path_count: int
    :param rng: Random number generator used for coin flips.
    :type rng: np.random.Generator
    :param initial_value: Starting value for every path.
    :type initial_value: float
    :return: DataFrame indexed by time with one column per random-walk path.
    :rtype: pd.DataFrame
    """
    steps = int(years * steps_per_year)
    dt = 1.0 / steps_per_year
    flips = rng.choice([-1.0, 1.0], size=(steps, path_count))
    increments = math.sqrt(dt) * flips
    values = initial_value + np.vstack([np.zeros(path_count), increments]).cumsum(axis=0)
    time = np.arange(steps + 1) * dt
    return pd.DataFrame(values, index=time, columns=path_columns(path_count))


def simulate_brownian_paths(
    years: float,
    steps_per_year: int,
    path_count: int,
    rng: np.random.Generator,
    initial_value: float = 0.0,
) -> pd.DataFrame:
    """
    Simulate Brownian motion paths with Normal increments.

    :param years: Number of years to simulate.
    :type years: float
    :param steps_per_year: Number of simulation steps in one year.
    :type steps_per_year: int
    :param path_count: Number of paths to simulate.
    :type path_count: int
    :param rng: Random number generator used for Normal increments.
    :type rng: np.random.Generator
    :param initial_value: Starting value for every path.
    :type initial_value: float
    :return: DataFrame indexed by time with one column per Brownian path.
    :rtype: pd.DataFrame
    """
    steps = int(years * steps_per_year)
    dt = 1.0 / steps_per_year
    increments = rng.normal(loc=0.0, scale=math.sqrt(dt), size=(steps, path_count))
    values = initial_value + np.vstack([np.zeros(path_count), increments]).cumsum(axis=0)
    time = np.arange(steps + 1) * dt
    return pd.DataFrame(values, index=time, columns=path_columns(path_count))


def simulate_brownian_motion_with_drift(
    initial_value: float,
    drift: float,
    volatility: float,
    years: float,
    steps_per_year: int,
    path_count: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Simulate arithmetic Brownian motion paths.

    :param initial_value: Starting value for every path.
    :type initial_value: float
    :param drift: Deterministic drift per year.
    :type drift: float
    :param volatility: Volatility multiplying the Brownian increment.
    :type volatility: float
    :param years: Number of years to simulate.
    :type years: float
    :param steps_per_year: Number of simulation steps in one year.
    :type steps_per_year: int
    :param path_count: Number of paths to simulate.
    :type path_count: int
    :param rng: Random number generator used for shocks.
    :type rng: np.random.Generator
    :return: DataFrame indexed by time with one column per simulated path.
    :rtype: pd.DataFrame
    """
    steps = int(years * steps_per_year)
    dt = 1.0 / steps_per_year
    shocks = rng.normal(loc=0.0, scale=1.0, size=(steps, path_count))
    increments = drift * dt + volatility * math.sqrt(dt) * shocks
    values = initial_value + np.vstack([np.zeros(path_count), increments]).cumsum(axis=0)
    time = np.arange(steps + 1) * dt
    return pd.DataFrame(values, index=time, columns=path_columns(path_count))


def simulate_geometric_brownian_motion(
    initial_price: float,
    drift: float,
    volatility: float,
    years: float,
    steps_per_year: int,
    path_count: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Simulate geometric Brownian motion paths with the exact log scheme.

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
    :return: DataFrame indexed by time with one column per price path.
    :rtype: pd.DataFrame
    """
    steps = int(years * steps_per_year)
    dt = 1.0 / steps_per_year
    shocks = rng.normal(loc=0.0, scale=1.0, size=(steps, path_count))
    log_increments = (drift - 0.5 * volatility**2) * dt
    log_increments = log_increments + volatility * math.sqrt(dt) * shocks
    log_paths = np.vstack([np.zeros(path_count), log_increments]).cumsum(axis=0)
    prices = initial_price * np.exp(log_paths)
    time = np.arange(steps + 1) * dt
    return pd.DataFrame(prices, index=time, columns=path_columns(path_count))


def simulate_ornstein_uhlenbeck(
    initial_value: float,
    long_run_mean: float,
    speed: float,
    volatility: float,
    years: float,
    steps_per_year: int,
    path_count: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Simulate an Ornstein-Uhlenbeck mean-reverting process.

    :param initial_value: Starting value for every path.
    :type initial_value: float
    :param long_run_mean: Level that the process reverts toward.
    :type long_run_mean: float
    :param speed: Mean-reversion speed.
    :type speed: float
    :param volatility: Volatility multiplying the Brownian increment.
    :type volatility: float
    :param years: Number of years to simulate.
    :type years: float
    :param steps_per_year: Number of simulation steps in one year.
    :type steps_per_year: int
    :param path_count: Number of paths to simulate.
    :type path_count: int
    :param rng: Random number generator used for shocks.
    :type rng: np.random.Generator
    :return: DataFrame indexed by time with one column per simulated path.
    :rtype: pd.DataFrame
    """
    steps = int(years * steps_per_year)
    dt = 1.0 / steps_per_year
    values = np.empty((steps + 1, path_count))
    values[0, :] = initial_value
    shocks = rng.normal(loc=0.0, scale=1.0, size=(steps, path_count))
    for index in range(steps):
        current = values[index, :]
        increments = speed * (long_run_mean - current) * dt
        increments = increments + volatility * math.sqrt(dt) * shocks[index, :]
        values[index + 1, :] = current + increments
    time = np.arange(steps + 1) * dt
    return pd.DataFrame(values, index=time, columns=path_columns(path_count))


def simulate_square_root_diffusion(
    initial_value: float,
    long_run_mean: float,
    speed: float,
    volatility: float,
    years: float,
    steps_per_year: int,
    path_count: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Simulate a non-negative square-root mean-reverting process.

    :param initial_value: Starting value for every path.
    :type initial_value: float
    :param long_run_mean: Level that the process reverts toward.
    :type long_run_mean: float
    :param speed: Mean-reversion speed.
    :type speed: float
    :param volatility: Volatility multiplying the square-root diffusion.
    :type volatility: float
    :param years: Number of years to simulate.
    :type years: float
    :param steps_per_year: Number of simulation steps in one year.
    :type steps_per_year: int
    :param path_count: Number of paths to simulate.
    :type path_count: int
    :param rng: Random number generator used for shocks.
    :type rng: np.random.Generator
    :return: DataFrame indexed by time with one column per simulated path.
    :rtype: pd.DataFrame
    """
    steps = int(years * steps_per_year)
    dt = 1.0 / steps_per_year
    values = np.empty((steps + 1, path_count))
    values[0, :] = initial_value
    shocks = rng.normal(loc=0.0, scale=1.0, size=(steps, path_count))
    for index in range(steps):
        current = np.maximum(values[index, :], 0.0)
        increments = speed * (long_run_mean - current) * dt
        increments = increments + volatility * np.sqrt(current) * math.sqrt(dt) * shocks[index, :]
        values[index + 1, :] = np.maximum(current + increments, 0.0)
    time = np.arange(steps + 1) * dt
    return pd.DataFrame(values, index=time, columns=path_columns(path_count))


def quadratic_variation(paths: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cumulative quadratic variation for each path.

    :param paths: DataFrame indexed by time with one column per path.
    :type paths: pd.DataFrame
    :return: DataFrame of cumulative squared increments for each path.
    :rtype: pd.DataFrame
    """
    increments = paths.diff().fillna(0.0)
    return (increments * increments).cumsum()


def quadratic_variation_convergence(
    years: float,
    step_counts: Sequence[int],
    path_count: int,
    seed: int,
) -> pd.DataFrame:
    """
    Summarise Brownian quadratic variation across partition sizes.

    :param years: Number of years to simulate.
    :type years: float
    :param step_counts: Partition sizes used for the Brownian simulations.
    :type step_counts: Sequence[int]
    :param path_count: Number of paths simulated for each partition size.
    :type path_count: int
    :param seed: Seed used to initialise the random number generator.
    :type seed: int
    :return: DataFrame summarising terminal quadratic variation by partition size.
    :rtype: pd.DataFrame
    """
    rng = create_rng(seed)
    rows = []
    for steps_per_year in step_counts:
        paths = simulate_brownian_paths(
            years=years,
            steps_per_year=steps_per_year,
            path_count=path_count,
            rng=rng,
        )
        terminal_qv = quadratic_variation(paths).iloc[-1]
        rows.append(
            {
                "Steps per year": steps_per_year,
                "Mean terminal quadratic variation": float(terminal_qv.mean()),
                "Standard deviation": float(terminal_qv.std(ddof=1)),
                "Expected value": years,
            }
        )
    return pd.DataFrame(rows)


def stochastic_integral_x_dX(paths: pd.DataFrame) -> pd.DataFrame:
    """
    Approximate the Ito integral of X against dX with left endpoints.

    :param paths: DataFrame indexed by time with one column per path.
    :type paths: pd.DataFrame
    :return: DataFrame containing cumulative stochastic integral estimates.
    :rtype: pd.DataFrame
    """
    increments = paths.diff().fillna(0.0)
    left_values = paths.shift(1).fillna(paths.iloc[0])
    return (left_values * increments).cumsum()


def ito_square_identity(paths: pd.DataFrame) -> pd.DataFrame:
    """
    Compare X squared with the Ito identity two integral X dX plus t.

    :param paths: DataFrame indexed by time with one column per Brownian path.
    :type paths: pd.DataFrame
    :return: DataFrame with path squares, Ito approximations, and errors.
    :rtype: pd.DataFrame
    """
    integrals = stochastic_integral_x_dX(paths)
    time = pd.Series(paths.index.to_numpy(dtype=float), index=paths.index)
    identity = pd.DataFrame(index=paths.index)
    for column in paths.columns:
        square_name = f"{column} X squared"
        ito_name = f"{column} two integral plus time"
        error_name = f"{column} error"
        identity[square_name] = paths[column] * paths[column]
        identity[ito_name] = 2.0 * integrals[column] + time
        identity[error_name] = identity[square_name] - identity[ito_name]
    return identity


def log_gbm_ito_check(paths: pd.DataFrame, drift: float, volatility: float) -> pd.DataFrame:
    """
    Compare simulated GBM log returns with the Ito drift correction.

    :param paths: DataFrame indexed by time with one column per GBM price path.
    :type paths: pd.DataFrame
    :param drift: Annual drift rate used in the simulation.
    :type drift: float
    :param volatility: Annual volatility used in the simulation.
    :type volatility: float
    :return: DataFrame comparing average log returns with Ito expectations.
    :rtype: pd.DataFrame
    """
    time = pd.Series(paths.index.to_numpy(dtype=float), index=paths.index)
    log_paths = np.log(paths / paths.iloc[0])
    expected_log_return = (drift - 0.5 * volatility**2) * time
    expected_price = float(paths.iloc[0, 0]) * np.exp(drift * time)
    return pd.DataFrame(
        {
            "Mean log return": log_paths.mean(axis=1),
            "Ito expected log return": expected_log_return,
            "Mean price": paths.mean(axis=1),
            "Expected price": expected_price,
        },
        index=paths.index,
    )


def build_sde_examples(seed: int = 606) -> dict[str, pd.DataFrame]:
    """
    Build a deterministic collection of common SDE examples.

    :param seed: Seed used to initialise the random number generator.
    :type seed: int
    :return: Mapping of example names to simulated path DataFrames.
    :rtype: dict[str, pd.DataFrame]
    """
    rng = create_rng(seed)
    return {
        "Brownian drift": simulate_brownian_motion_with_drift(
            initial_value=0.0,
            drift=0.20,
            volatility=0.55,
            years=2.0,
            steps_per_year=TRADING_DAYS,
            path_count=1,
            rng=rng,
        ),
        "Lognormal": simulate_geometric_brownian_motion(
            initial_price=1.0,
            drift=0.12,
            volatility=0.30,
            years=2.0,
            steps_per_year=TRADING_DAYS,
            path_count=1,
            rng=rng,
        ),
        "OU mean reverting": simulate_ornstein_uhlenbeck(
            initial_value=1.2,
            long_run_mean=0.0,
            speed=2.0,
            volatility=0.35,
            years=2.0,
            steps_per_year=TRADING_DAYS,
            path_count=1,
            rng=rng,
        ),
        "Square-root": simulate_square_root_diffusion(
            initial_value=0.08,
            long_run_mean=0.05,
            speed=1.4,
            volatility=0.22,
            years=2.0,
            steps_per_year=TRADING_DAYS,
            path_count=1,
            rng=rng,
        ),
    }


def plot_paths(
    paths: pd.DataFrame,
    title: str,
    yaxis_title: str = "Value",
    line_width: float = 1.5,
    opacity: float = 0.75,
    showlegend: bool = False,
) -> go.Figure:
    """
    Plot paths stored as columns in a DataFrame.

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
    :return: Plotly figure containing the paths.
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
        xaxis_title="Time",
        yaxis_title=yaxis_title,
        template="plotly_white",
        showlegend=showlegend,
    )
    return fig


def plot_quadratic_variation(qv_paths: pd.DataFrame) -> go.Figure:
    """
    Plot cumulative quadratic variation against elapsed time.

    :param qv_paths: DataFrame of cumulative quadratic variation paths.
    :type qv_paths: pd.DataFrame
    :return: Plotly figure showing quadratic variation and the time line.
    :rtype: go.Figure
    """
    fig = plot_paths(
        qv_paths,
        title="Quadratic variation of Brownian paths",
        yaxis_title="Cumulative squared increment",
        line_width=1.2,
        opacity=0.55,
        showlegend=False,
    )
    fig.add_trace(
        go.Scatter(
            x=qv_paths.index,
            y=qv_paths.index,
            mode="lines",
            name="Elapsed time",
            line={"width": 3, "dash": "dash"},
        )
    )
    fig.update_layout(showlegend=True)
    return fig


def plot_quadratic_variation_convergence(summary: pd.DataFrame) -> go.Figure:
    """
    Plot terminal quadratic variation against partition size.

    :param summary: DataFrame returned by :func:`quadratic_variation_convergence`.
    :type summary: pd.DataFrame
    :return: Plotly figure showing convergence of terminal quadratic variation.
    :rtype: go.Figure
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=summary["Steps per year"],
            y=summary["Mean terminal quadratic variation"],
            mode="lines+markers",
            name="Mean terminal quadratic variation",
            line={"width": 3},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=summary["Steps per year"],
            y=summary["Expected value"],
            mode="lines",
            name="Elapsed time",
            line={"width": 3, "dash": "dash"},
        )
    )
    fig.update_layout(
        title="Quadratic variation stabilises as the partition gets finer",
        xaxis_title="Steps per year",
        yaxis_title="Terminal quadratic variation",
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


def plot_stochastic_integral_identity(
    identity: pd.DataFrame,
    path_name: str = "Path 1",
) -> go.Figure:
    """
    Plot the Ito identity for the square of one Brownian path.

    :param identity: DataFrame returned by :func:`ito_square_identity`.
    :type identity: pd.DataFrame
    :param path_name: Path name to plot from the identity DataFrame.
    :type path_name: str
    :return: Plotly figure comparing X squared with the Ito expression.
    :rtype: go.Figure
    """
    square_name = f"{path_name} X squared"
    ito_name = f"{path_name} two integral plus time"
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=identity.index,
            y=identity[square_name],
            mode="lines",
            name="X squared",
            line={"width": 2.5},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=identity.index,
            y=identity[ito_name],
            mode="lines",
            name="2 integral X dX plus t",
            line={"width": 2.5, "dash": "dash"},
        )
    )
    fig.update_layout(
        title="Ito identity for F(X) = X squared",
        xaxis_title="Time",
        yaxis_title="Value",
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


def plot_sde_examples(examples: Mapping[str, pd.DataFrame]) -> go.Figure:
    """
    Plot one representative path from each SDE example.

    :param examples: Mapping of labels to path DataFrames.
    :type examples: Mapping[str, pd.DataFrame]
    :return: Plotly figure comparing the SDE examples.
    :rtype: go.Figure
    """
    fig = go.Figure()
    for label, paths in examples.items():
        first_column = paths.columns[0]
        fig.add_trace(
            go.Scatter(
                x=paths.index,
                y=paths[first_column],
                mode="lines",
                name=label,
                line={"width": 2.2},
            )
        )
    fig.update_layout(
        title="Common stochastic differential equation examples",
        xaxis_title="Time",
        yaxis_title="State variable",
        template="plotly_white",
        hovermode="x unified",
    )
    return fig
