import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import mean_squared_error  # type: ignore

from stock_valuation_tool.exceptions import InvalidOptionError
from stock_valuation_tool.utils import Config

from ._models import CustomPeModel, ExponentialModel, LinReg, MedianPeModel


def modelling(
    config: Config,
    past_fundamentals: pd.DataFrame,
    prices: pd.DataFrame,
    benchmark_prices: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    current_date, current_price = (
        prices["date"].iloc[0],
        prices["close_adj_origin_currency"].iloc[0],
    )

    predicted_fundamentals = _predict_future_funtamentals(config, past_fundamentals)

    past_fundamentals = pd.concat(
        [
            past_fundamentals.iloc[[0]].assign(
                date=current_date, close_adj_origin_currency=current_price, period="present"
            ),
            past_fundamentals,
        ],
        ignore_index=True,
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
                "return": round(
                    (all_fundamentals["close_adj_origin_currency"].iloc[0] / current_price - 1)
                    * 100,
                    2,
                ),
                "return_becnhmark": round(
                    ((yearly_return ** ((end_of_simulation_date - current_date).days / 365)) - 1)
                    * 100,
                    2,
                ),
            }
        ]
    )

    return all_fundamentals, returns


def _predict_future_funtamentals(
    config: Config,
    past_fundamentals: pd.DataFrame,
) -> pd.DataFrame:
    past_periods = len(past_fundamentals)
    past_fundamentals["period"] = "past"

    X, y_eps, y_pe = (  # noqa: N806
        [[x] for x in range(past_periods)],
        list(reversed(past_fundamentals["eps"])),
        list(reversed(past_fundamentals["pe"])),
    )
    model_eps = _model_selection(config, X, y_eps, past_periods, "eps")
    model_pe = _model_selection(config, X, y_pe, past_periods, "pe")

    last_period_date = past_fundamentals["date"].iloc[0]
    pred = []
    for i, X_pred in enumerate(  # noqa: N806
        range(
            past_periods,
            past_periods + config.future_years
            if config.freq == "yearly"
            else past_periods + config.future_years * 4,
        )
    ):
        last_period_date = (
            last_period_date + pd.DateOffset(years=1)
            if config.freq == "yearly"
            else last_period_date + pd.DateOffset(months=3)
        )

        match config.modelling["pe"]["model"]:
            case "linear":
                pe_pred = model_pe.predict([[X_pred]])[0]  # type: ignore
            case "exp" | "custom_cagr":
                pe_pred = model_pe.predict(i + 1)[-1]  # type: ignore
            case "median" | "custom_pe":
                pe_pred = model_pe.predict()  # type: ignore

        match config.modelling["eps"]["model"]:
            case "linear":
                eps_pred = model_eps.predict([[X_pred]])[0]  # type: ignore
            case "exp" | "custom_cagr":
                eps_pred = model_eps.predict(i + 1)[-1]  # type: ignore

        pred.append(
            {
                "date": last_period_date,
                "eps": eps_pred,
                "close_adj_origin_currency": eps_pred * pe_pred,
                "pe": pe_pred,
                "period": "future",
            }
        )

    return pd.DataFrame(reversed(pred))


def _model_selection(  # noqa: PLR0911
    config: Config,
    X: list[list[int]],  # noqa: N803
    y: list[float],
    past_periods: int,
    modelling_type: str,
) -> LinReg | ExponentialModel | MedianPeModel | CustomPeModel:
    match config.modelling[modelling_type]["model"]:
        case "median":
            median_pe = MedianPeModel()
            median_pe.train(y)
            return median_pe
        case "custom_pe":
            custom_pe = CustomPeModel()
            custom_pe.train(config.modelling[modelling_type]["value"])
            return custom_pe
        case "linear":
            lin_reg = LinReg()
            lin_reg.train(X, y)
            return lin_reg
        case "exp":
            exp = ExponentialModel()
            exp.train(y)
            return exp
        case "custom_cagr":
            return ExponentialModel(
                cqgr=round((config.modelling[modelling_type]["value"] / 100 + 1) ** 0.25, 4),
                latest_point=y[-1],
            )
        case "auto":
            rmse_lin_reg, rmse_exp = [], []

            # cross-validation
            for train_perc in [0.5, 0.8]:
                train_size = int(past_periods * train_perc)
                X_train, X_test, y_train, y_test = (  # noqa: N806
                    X[:train_size],
                    X[train_size:],
                    y[:train_size],
                    y[train_size:],
                )

                lin_reg = LinReg()
                lin_reg.train(X_train, y_train)
                rmse_lin_reg.append(np.sqrt(mean_squared_error(lin_reg.predict(X_test), y_test)))

                exp = ExponentialModel()
                exp.train(y_train)
                rmse_exp.append(np.sqrt(mean_squared_error(exp.predict(len(y_test)), y_test)))

            rmse_lin_reg, rmse_exp = np.mean(rmse_lin_reg), np.mean(rmse_exp)
            logger.info(f"RMSE lin_reg: {rmse_lin_reg}. RMSE exp: {rmse_exp}")

            if rmse_lin_reg < rmse_exp:
                config.modelling[modelling_type]["model"] = "linear"
                lin_reg = LinReg()
                lin_reg.train(X, y)
                return lin_reg

            config.modelling[modelling_type]["model"] = "exp"
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
