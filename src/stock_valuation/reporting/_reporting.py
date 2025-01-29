import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def reporting(data, returns):
    pass


def _plot_dividends_year(data, returns) -> None:
    """Plot dividends year.

    Args:
        portfolio_currency: Currency of the portfolio.
        dividends_year: Total dividends paied by year.
    """
    years, dividends = (
        data["date"] + [pd.DateOffset(months=i * 3) for i in range(10)] + [returns["date"][0]],
        data["eps"] + list(range(10)) + [returns["eps"][0]],
    )
    n = len(dividends)
    bar_width = 0.4
    index = np.arange(n)

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.bar(index, dividends, bar_width, label="Dividend amount", color="blue")

    top_y_lim = max(list(dividends))
    bottom_y_lim = min(list(dividends))
    margin = (abs(top_y_lim) + abs(bottom_y_lim)) * 0.02

    top_y_lim += margin
    bottom_y_lim -= margin
    y_len = abs(top_y_lim) + abs(bottom_y_lim)

    # Set fixed axis limits (frame position)
    ax.set_xlim((-0.5, n))
    ax.set_ylim((bottom_y_lim, top_y_lim))

    # Add labels and title
    ax.set_ylabel("Dividend amount (EUR)")
    ax.set_title("Dividends per year")
    ax.set_xticks(index)
    ax.set_xticklabels(years)

    # Add legend
    ax.legend()

    y_offset = bottom_y_lim - y_len * 0.11

    for i in index:
        plt.text(
            i,
            y_offset,
            f"{dividends_year['total_dividend_asset'][i]:.2f}\n{portfolio_currency}",
            ha="center",
            color="blue",
            fontweight="bold",
            rotation=0,
        )

    plt.savefig(
        DIR_OUT
        / Path(
            "dividends_year.png",
        ),
    )
    plt.close()
