import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression  # type: ignore
from sklearn.metrics import mean_squared_error  # type: ignore

from stock_valuation.exceptions import InvalidOptionError
from stock_valuation.utils import Config


def modelling(
    config: Config,
    past_fundamentals: pd.DataFrame,
    prices: pd.DataFrame,
    benchmark_prices: pd.DataFrame,
    future_years: int,
    freq: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    current_date, current_price = (
        prices["date"].iloc[0],
        prices["close_adj_origin_currency"].iloc[0],
    )

    predicted_fundamentals = _predict_future_funtamentals(
        past_fundamentals, freq, future_years, config.modelling
    )

    past_fundamentals = (
        pd.concat(
            [
                past_fundamentals.iloc[[0]].assign(
                    date=current_date, close_adj_origin_currency=current_price, period="present"
                ),
                past_fundamentals,
            ],
            ignore_index=True,
        )
        .assign(
            close_adj_origin_currency_pe_ct=past_fundamentals["close_adj_origin_currency"],
            close_adj_origin_currency_pe_exp=past_fundamentals["close_adj_origin_currency"],
            pe_ct=past_fundamentals["pe"],
            pe_exp=past_fundamentals["pe"],
        )
        .drop(columns=["pe", "close_adj_origin_currency"])
    )

    all_fundamentals = pd.concat(
        [
            predicted_fundamentals,
            past_fundamentals,
        ],
        ignore_index=True,
    )

    yearly_return = _model_benchmark_returns(benchmark_prices)
    end_of_simulation_date = all_fundamentals["date"].iloc[0]

    returns = pd.DataFrame(
        [
            {
                "date": end_of_simulation_date,
                "return_pe_ct": (
                    all_fundamentals["close_adj_origin_currency_pe_ct"].iloc[0] / current_price - 1
                )
                * 100,
                "return_pe_exp": (
                    all_fundamentals["close_adj_origin_currency_pe_exp"].iloc[0] / current_price - 1
                )
                * 100,
                "return_becnhmark": (
                    (yearly_return ** ((end_of_simulation_date - current_date).days / 365)) - 1
                )
                * 100,
            }
        ]
    )

    return all_fundamentals, returns


def _predict_future_funtamentals(
    past_fundamentals: pd.DataFrame,
    freq: str,
    future_years: int,
    modelling: dict[str, str],
) -> pd.DataFrame:
    past_periods = len(past_fundamentals)
    past_fundamentals["period"] = "past"
    pe_ct = past_fundamentals["pe"].median()

    X, y_eps, y_pe = (  # noqa: N806
        [[x] for x in range(past_periods)],
        list(reversed(past_fundamentals["eps"])),
        list(reversed(past_fundamentals["pe"])),
    )
    model_eps = _model_selection(modelling, X, y_eps, past_periods, "eps")
    model_pe = _model_selection(modelling, X, y_pe, past_periods, "pe")

    last_period_date = past_fundamentals["date"].iloc[0]
    pred = []
    for i, X_pred in enumerate(  # noqa: N806
        range(
            past_periods,
            past_periods + future_years if freq == "yearly" else past_periods + future_years * 4,
        )
    ):
        last_period_date = (
            last_period_date + pd.DateOffset(years=1)
            if freq == "yearly"
            else last_period_date + pd.DateOffset(months=3)
        )

        match modelling["pe"]:
            case "linear":
                pe_exp_pred = model_pe.predict([[X_pred]])[0]  # type: ignore
            case "exp":
                pe_exp_pred = model_pe.predict(i + 1)[-1]  # type: ignore

        match modelling["eps"]:
            case "linear":
                eps_pred = model_eps.predict([[X_pred]])[0]  # type: ignore
            case "exp":
                eps_pred = model_eps.predict(i + 1)[-1]  # type: ignore

        pred.append(
            {
                "date": last_period_date,
                "eps": eps_pred,
                "close_adj_origin_currency_pe_ct": eps_pred * pe_ct,
                "close_adj_origin_currency_pe_exp": eps_pred * pe_exp_pred,
                "pe_ct": pe_ct,
                "pe_exp": pe_exp_pred,
                "period": "future",
            }
        )

    return pd.DataFrame(reversed(pred))


def _lin_reg(
    X_train: list[list[int]],  # noqa: N803
    y_train: list[float],
) -> LinearRegression:
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)

    return lin_reg


class LinReg:
    def __init__(self) -> None:
        self.lin_reg = LinearRegression()

    def train(self, X_train: list[list[int]], y_train: list[float]) -> None:  # noqa: N803
        self.lin_reg.fit(X_train, y_train)

    def predict(self, value: list[list[int]]) -> float:
        return self.lin_reg.predict(value)  # type: ignore


class ExponentialModel:
    def __init__(self) -> None:
        self.latest_point = 0.0
        self.cqgr = 0.0

    def train(self, y_train: list[float]) -> None:
        self.latest_point = y_train[-1]
        self.cqgr = (y_train[-1] / y_train[0]) ** (1 / len(y_train))

    def predict(self, periods: int) -> list[float]:
        pred = [self.latest_point]

        for _ in range(periods):
            pred.append(pred[-1] * self.cqgr)

        return pred[1:]


def _model_selection(
    modelling: dict[str, str],
    X: list[list[int]],  # noqa: N803
    y: list[float],
    past_periods: int,
    modelling_type: str,
) -> LinReg | ExponentialModel:
    match modelling[modelling_type]:
        case "linear":
            lin_reg = LinReg()
            lin_reg.train(X, y)
            return lin_reg
        case "exp":
            exp = ExponentialModel()
            exp.train(y)
            return exp
        case "auto":
            train_size = int(past_periods * 0.8)
            X_train, X_test, y_train, y_test = (  # noqa: N806
                X[:train_size],
                X[train_size:],
                y[:train_size],
                y[train_size:],
            )

            lin_reg = LinReg()
            lin_reg.train(X_train, y_train)
            rmse_lin_reg = np.sqrt(mean_squared_error(lin_reg.predict(X_test), y_test))

            exp = ExponentialModel()
            exp.train(y_train)
            rmse_exp = np.sqrt(mean_squared_error(exp.predict(len(y_test)), y_test))

            if rmse_lin_reg < rmse_exp:
                modelling[modelling_type] = "linear"
                lin_reg = LinReg()
                lin_reg.train(X, y)
                return lin_reg

            modelling[modelling_type] = "exp"
            exp = ExponentialModel()
            exp.train(y)
            return exp
        case _:
            raise InvalidOptionError


def _model_benchmark_returns(benchmark_prices: pd.DataFrame) -> float:
    """Calculate CAGR of the benchmark.

    Args:
        benchmark_prices: DataFrame containing "date" and "close_adj_origin_currency".

    Returns:
        CAGR.
    """
    return float(
        (
            benchmark_prices["close_adj_origin_currency"].iloc[0]
            / benchmark_prices["close_adj_origin_currency"].iloc[-1]
        )
        ** (1 / (len(benchmark_prices) / 365))
    )
