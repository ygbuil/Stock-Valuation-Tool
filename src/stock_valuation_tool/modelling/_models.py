import numpy as np
from sklearn.linear_model import LinearRegression  # type: ignore

from stock_valuation_tool.exceptions import InvalidInputDataError


class LinReg:
    def __init__(self) -> None:
        self.lin_reg = LinearRegression()

    def train(self, X_train: list[list[int]], y_train: list[float]) -> None:  # noqa: N803
        self.lin_reg.fit(X_train, y_train)

    def predict(self, value: list[list[int]]) -> float:
        return self.lin_reg.predict(value)  # type: ignore


class MedianPeModel:
    def __ini__(self) -> None:
        self.median = 0.0

    def train(self, y_train: list[float]) -> None:
        self.median = float(np.median(y_train))

    def predict(self) -> float:
        return self.median


class CustomPeModel:
    def __ini__(self) -> None:
        self.custom_pe = 0.0

    def train(self, custom_pe: float) -> None:
        self.custom_pe = custom_pe

    def predict(self) -> float:
        return self.custom_pe


class ExponentialModel:
    def __init__(self, cqgr: float = 0.0, latest_point: float = 0.0) -> None:
        self.cqgr = cqgr
        self.latest_point = latest_point

    def train(self, y_train: list[float]) -> None:
        self.latest_point = y_train[-1]

        perc_growts = y_train[-1] / y_train[0]
        if perc_growts < 0:
            raise InvalidInputDataError

        self.cqgr = round(perc_growts ** (1 / len(y_train)), 4)

    def predict(self, periods: int) -> list[float]:
        pred = [self.latest_point]

        for _ in range(periods):
            pred.append(pred[-1] * self.cqgr)

        return [round(x, 4) for x in pred[1:]]
