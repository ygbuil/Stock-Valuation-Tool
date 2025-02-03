import pandas as pd
from sklearn.linear_model import LinearRegression


def modelling(
    data: pd.DataFrame, prices: pd.DataFrame, future_years: int, freq: str
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

    returns = pd.DataFrame(
        [
            {
                "date": data_and_pred["date"].iloc[0],
                "return_pe_ct": (
                    data_and_pred["close_adj_origin_currency_pe_ct"].iloc[0] / current_price - 1
                )
                * 100,
                "return_pe_exp": (
                    data_and_pred["close_adj_origin_currency_pe_exp"].iloc[0] / current_price - 1
                )
                * 100,
            }
        ]
    )
    return data_and_pred, returns


def _calculate_future_eps_and_pe(data: pd.DataFrame, freq: str, future_years: int) -> float:
    n_data_points = len(data)
    data["period"] = "past"
    pe_ct = data["pe"].mean()

    X_train, y_train = [[x] for x in range(n_data_points)], list(reversed(data["eps"]))  # noqa: N806
    lin_reg_eps = _lin_reg(X_train, y_train)

    X_train, y_train = [[x] for x in range(n_data_points)], list(reversed(data["pe"]))  # noqa: N806
    lin_reg_pe = _lin_reg(X_train, y_train)

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

        eps_pred = lin_reg_eps.predict([[X_pred]])[0]
        pe_exp_pred = lin_reg_pe.predict([[X_pred]])[0]

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
