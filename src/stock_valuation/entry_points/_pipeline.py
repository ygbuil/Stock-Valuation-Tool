"""Main module to execute the project."""

import click
from loguru import logger

from stock_valuation import modelling, preprocessing
# from stock_portfolio_tracker.utils import timer


@click.command()
@click.option("--ticker")
def pipeline(ticker: str) -> None:
    """Entry point for pipeline.

    Args:
        config_file_name: File name for config.
        transactions_file_name: File name for transactions.
    """
    _pipeline(ticker)


# @timer
def _pipeline(ticker: str) -> None:
    """Execute the project end to end.

    Args:
        config_file_name: File name for config.
        transactions_file_name: File name for transactions.
    """
    logger.info("Start of execution.")

    logger.info("Start of preprocess.")
    data, prices = preprocessing.preprocess(ticker)

    logger.info("Start of modelling.")
    returns = modelling.modelling(data, prices)

    logger.info("End of execution.")
