"""Entry points for stock_portfolio_tracker."""

import click

from stock_valuation_tool import entry_points


def _main() -> None:
    """Gathers all entry points of the program."""

    @click.group(chain=True)
    def entry_point() -> None:
        """Entry point."""

    for command in (entry_points.pipeline,):
        entry_point.add_command(command)

    entry_point()
