import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

DIR_OUT = Path("data/out")


def reporting(data_and_pred, returns):
    _plot_dividends_year(data_and_pred, returns)


def _plot_dividends_year(data_and_pred, returns) -> None:
    """Plot dividends year.

    Args:
        portfolio_currency: Currency of the portfolio.
        dividends_year: Total dividends paied by year.
    """
    dates, eps, periods = (
        list(reversed([date.strftime("%Y-%m") for date in data_and_pred["date"]])),
        list(reversed(data_and_pred["eps"])),
        list(reversed(data_and_pred["period"])),  # "past" or "future"
    )

    n = len(eps)
    bar_width = 0.4
    index = np.arange(n)
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot bars with respective colors
    for idx, height, period in zip(index, eps, periods):
        ax.bar(idx, height, bar_width, color="blue" if period == "past" else "orange")

    top_y_lim = max(list(eps))
    bottom_y_lim = min(list(eps))
    margin = (abs(top_y_lim) + abs(bottom_y_lim)) * 0.02

    top_y_lim += margin
    bottom_y_lim -= margin
    y_len = abs(top_y_lim) + abs(bottom_y_lim)

    # Set fixed axis limits (frame position)
    ax.set_xlim((-0.5, n))
    ax.set_ylim((bottom_y_lim, top_y_lim))

    # Add labels and title
    ax.set_ylabel("Past and projected EPS")
    ax.set_title("EPS")
    ax.set_xticks(index)
    ax.set_xticklabels(dates)

    # Add legend
    ax.legend()

    y_offset = bottom_y_lim - y_len * 0.11

    for i in index:
        plt.text(
            i,
            y_offset,
            f"{eps[i]:.2f}",
            ha="center",
            color="black",
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
