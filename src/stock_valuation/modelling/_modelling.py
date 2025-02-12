import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

def modelling(
    data: pd.DataFrame, prices: pd.DataFrame, benchmark_prices: pd.DataFrame, future_years: int, freq: str
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
                "return_becnhmark": ((yearly_return ** ((end_of_simulation_date-current_date).days/365)) - 1)*100
            }
        ]
    )
    
    return data_and_pred, returns


def _calculate_future_eps_and_pe(data: pd.DataFrame, freq: str, future_years: int, modelling: str = "linear") -> float:
    n_data_points = len(data)
    data["period"] = "past"
    pe_ct = data["pe"].mean()

    X_train, y_train = [[x] for x in range(n_data_points)], list(reversed(data["eps"]))  # noqa: N806
    if modelling == "linear":
        model_eps = _lin_reg(X_train, y_train)
    else:
        model_eps = ExponentialModel()
        model_eps.train(y_train)

    X_train, y_train = [[x] for x in range(n_data_points)], list(reversed(data["pe"]))  # noqa: N806
    if modelling == "linear":
        model_pe = _lin_reg(X_train, y_train)
    else:
        model_pe = ExponentialModel()
        model_pe.train(y_train)

    latest_date = data["date"].iloc[0]
    pred = []
    for X_pred in range(  # noqa: N806
        n_data_points,
        n_data_points + future_years if freq == "yearly" else n_data_points + future_years * 4,
    ):
        latest_date = (
            latest_date + pd.DateOffset(years=1)
            if freq == "yearly"
            else latest_date + pd.DateOffset(months=3)
        )

        if modelling == "linear":
            eps_pred = model_eps.predict([[X_pred]])[0]
            pe_exp_pred = model_pe.predict([[X_pred]])[0]
        else:
            eps_pred = model_eps.predict()
            pe_exp_pred = model_pe.predict()

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
    def __init__(self):
        self.growth_rate = None
        self.latest_point = None

    def train(self, y_train):
        growth_rates = []

        for i in range(1, len(y_train)):
            growth_rates.append(y_train[i] / y_train[i - 1])

        self.growth_rate = np.mean(growth_rates)
        self.latest_point = y_train[-1]
    
    def predict(self):
        self.latest_point *= self.growth_rate
        return self.latest_point


def _model_benchmark_returns(benchmark_prices) -> float:
    """Calculate CAGR of the benchmark.

    Args:
        benchmark_prices: DataFrame containing "date" and "close_adj_origin_currency".

    Returns:
        _description_
    """
    return (benchmark_prices["close_adj_origin_currency"].iloc[0] / benchmark_prices["close_adj_origin_currency"].iloc[-1]) ** (1/(len(benchmark_prices)/365))