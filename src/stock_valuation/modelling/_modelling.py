import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def modelling(
    data: pd.DataFrame,
    prices: pd.DataFrame,
    benchmark_prices: pd.DataFrame,
    future_years: int,
    freq: str,
) -> pd.DataFrame:
    current_date, current_price = (
        prices["date"].iloc[0],
        prices["close_adj_origin_currency"].iloc[0],
    )

    pred = _calculate_future_eps_and_pe(data, freq, future_years)

    data = (
        pd.concat(
            [
                data.iloc[[0]].assign(
                    date=current_date, close_adj_origin_currency=current_price, period="present"
                ),
                data,
            ],
            ignore_index=True,
        )
        .assign(
            close_adj_origin_currency_pe_ct=data["close_adj_origin_currency"],
            close_adj_origin_currency_pe_exp=data["close_adj_origin_currency"],
            pe_ct=data["pe"],
            pe_exp=data["pe"],
        )
        .drop(columns=["pe", "close_adj_origin_currency"])
    )

    data_and_pred = pd.concat(
        [
            pred,
            data,
        ],
        ignore_index=True,
    )

    yearly_return = _model_benchmark_returns(benchmark_prices)
    end_of_simulation_date = data_and_pred["date"].iloc[0]

    returns = pd.DataFrame(
        [
            {
                "date": end_of_simulation_date,
                "return_pe_ct": (
                    data_and_pred["close_adj_origin_currency_pe_ct"].iloc[0] / current_price - 1
                )
                * 100,
                "return_pe_exp": (
                    data_and_pred["close_adj_origin_currency_pe_exp"].iloc[0] / current_price - 1
                )
                * 100,
                "return_becnhmark": (
                    (yearly_return ** ((end_of_simulation_date - current_date).days / 365)) - 1
                )
                * 100,
            }
        ]
    )

    return data_and_pred, returns


def _calculate_future_eps_and_pe(
    data: pd.DataFrame,
    freq: str,
    future_years: int,
    modelling: str = {"eps": "exp", "pe": "linear"},
) -> float:
    n_data_points = len(data)
    data["period"] = "past"
    pe_ct = data["pe"].mean()

    X, y = [[x] for x in range(n_data_points)], list(reversed(data["eps"]))  # noqa: N806
    model_eps = _model_selection(modelling, X, y, n_data_points, "eps")

    X, y = [[x] for x in range(n_data_points)], list(reversed(data["pe"]))  # noqa: N806
    model_pe = _model_selection(modelling, X, y, n_data_points, "pe")

    latest_date = data["date"].iloc[0]
    pred = []
    for i, X_pred in enumerate(  # noqa: N806
        range(
            n_data_points,
            n_data_points + future_years if freq == "yearly" else n_data_points + future_years * 4,
        )
    ):
        latest_date = (
            latest_date + pd.DateOffset(years=1)
            if freq == "yearly"
            else latest_date + pd.DateOffset(months=3)
        )

        match modelling["pe"]:
            case "linear":
                pe_exp_pred = model_pe.predict([[X_pred]])[0]
            case "exp":
                pe_exp_pred = model_pe.predict(i + 1)[-1]

        match modelling["eps"]:
            case "linear":
                eps_pred = model_eps.predict([[X_pred]])[0]
            case "exp":
                eps_pred = model_eps.predict(i + 1)[-1]

        pred.append(
            {
                "date": latest_date,
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


class ExponentialModel:
    def __init__(self) -> None:
        self.latest_point = None
        self.cqgr = None

    def train(self, y_train: list[float]) -> None:
        self.latest_point = y_train[-1]
        self.cqgr = (y_train[-1] / y_train[0]) ** (1 / len(y_train))

    def predict(self, periods: int) -> list[float | None]:
        pred = [self.latest_point]

        for _ in range(periods):
            pred.append(pred[-1] * self.cqgr)

        return pred[1:]


def _model_selection(
    modelling: dict[str:str],
    X: list[list[float]],  # noqa: N803
    y: list[float],
    n_data_points: int,
    modelling_type: str,
) -> LinearRegression | ExponentialModel | None:
    match modelling[modelling_type]:
        case "linear":
            return _lin_reg(X, y)
        case "exp":
            model = ExponentialModel()
            model.train(y)
            return model
        case "auto":
            train_size = int(n_data_points * 0.8)
            X_train, X_test, y_train, y_test = (  # noqa: N806
                X[:train_size],
                X[train_size:],
                y[:train_size],
                y[train_size:],
            )

            lin_reg = _lin_reg(X_train, y_train)
            rmse_lin_reg = np.sqrt(mean_squared_error(lin_reg.predict(X_test), y_test))

            exp = ExponentialModel()
            exp.train(y)
            rmse_exp = np.sqrt(mean_squared_error(exp.predict(len(y_test)), y_test))

            if rmse_lin_reg < rmse_exp:
                modelling[modelling_type] = "linear"
                return lin_reg
            modelling[modelling_type] = "exp"
            return exp


def _model_benchmark_returns(benchmark_prices: pd.DataFrame) -> float:
    """Calculate CAGR of the benchmark.

    Args:
        benchmark_prices: DataFrame containing "date" and "close_adj_origin_currency".

    Returns:
        _description_
    """
    return (
        benchmark_prices["close_adj_origin_currency"].iloc[0]
        / benchmark_prices["close_adj_origin_currency"].iloc[-1]
    ) ** (1 / (len(benchmark_prices) / 365))
