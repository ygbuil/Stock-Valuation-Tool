"""Main module to execute the project."""

import click
from loguru import logger

from stock_valuation_tool import modelling, preprocessing, reporting
from stock_valuation_tool.utils import timer


@click.command()
@click.option("--ticker")
@click.option("--benchmark")
@click.option("--data-source", type=click.Choice(["api", "csv"]))
def pipeline(ticker: str, benchmark: str, data_source: str) -> None:
    """Entry point for pipeline.

    Args:
        ticker: Stock ticker.
        benchmark: Benchmark ticker.
        data_source: Source for the fundamentals data. Options: api, csv.
    """
    _pipeline(ticker, benchmark, data_source)


@timer
def _pipeline(ticker: str, benchmark: str, data_source: str) -> None:
    """Execute the project end to end.

    Args:
        ticker: Stock ticker.
        benchmark: Benchmark ticker.
        data_source: Source for the fundamentals data. Options: api, csv.
    """
    logger.info("Start of execution.")

    logger.info("Start of preprocess.")
    config, past_fundamentals, prices, benchmark_prices = preprocessing.preprocess(
        ticker, benchmark, data_source
    )

    logger.info("Start of modelling.")
    all_fundamentals, returns = modelling.modelling(
        config, past_fundamentals, prices, benchmark_prices
    )

    logger.info("Start of reporting.")
    reporting.reporting(config, all_fundamentals, returns)

    logger.info("End of execution.")
