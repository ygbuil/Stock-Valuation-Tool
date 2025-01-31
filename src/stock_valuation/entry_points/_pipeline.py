"""Main module to execute the project."""

import click
from loguru import logger

from stock_valuation import modelling, preprocessing, reporting
from stock_valuation.utils import timer


@click.command()
@click.option("--ticker")
@click.option("--past-years")
@click.option("--future-years")
@click.option("--freq")
def pipeline(ticker: str, past_years: str, future_years: str, freq: str) -> None:
    """Entry point for pipeline.

    Args:
        ticker: Stock ticker.
        past_years: Number of past years to run the analisis on.
        future_years: Number of future years to run the analisis on.
        freq: Frequency of the data. Options: yearly, quarterly, ttm.
    """
    _pipeline(ticker, int(past_years), int(future_years), freq)


@timer
def _pipeline(ticker: str, past_years: int, future_years: int, freq: str) -> None:
    """Execute the project end to end.

    Args:
        ticker: Stock ticker.
        past_years: Number of past years to run the analisis on.
        future_years: Number of future years to run the analisis on.
        freq: Frequency of the data. Options: yearly, quarterly, ttm.
    """
    logger.info("Start of execution.")

    logger.info("Start of preprocess.")
    data, prices = preprocessing.preprocess(ticker, past_years, freq)

    logger.info("Start of modelling.")
    data_and_pred, _ = modelling.modelling(data, prices, future_years, freq)

    logger.info("Start of reporting.")
    reporting.reporting(data_and_pred)

    logger.info("End of execution.")
