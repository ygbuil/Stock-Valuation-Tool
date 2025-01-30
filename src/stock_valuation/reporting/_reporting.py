import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.patches as mpatches


DIR_OUT = Path("data/out")


def reporting(data_and_pred, returns):
    _plot_dividends_year(data_and_pred, returns)


def _plot_dividends_year(data_and_pred, returns) -> None:
    """Plot past and projected EPS along with close-adjusted PE.

    Args:
        data_and_pred: DataFrame containing "date", "eps", "period", and "close_adj_origin_currency_pe_ct".
        returns: Not used in this function.
    """
    dates, eps, periods, close_adj_pe = (
        list(reversed([date.strftime("%Y-%m") for date in data_and_pred["date"]])),
        list(reversed(data_and_pred["eps"])),
        list(reversed(data_and_pred["period"])),
        list(reversed(data_and_pred["close_adj_origin_currency_pe_ct"])),
    )

    time_series_dim = len(eps)
    bar_width = 0.4
    index = np.arange(time_series_dim)

    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot bars with respective colors
    for idx, height, period in zip(index, eps, periods):
        ax.bar(idx, height, bar_width, color="blue" if period == "past" else "orange")

    # Set y-axis limits for EPS
    offset = max(eps)*0.02
    ax.set_xlim((-0.5, time_series_dim))
    ax.set_ylim((-offset, max(eps) + offset))

    # EPS legend
    ax.legend(handles=[mpatches.Patch(color="blue", label="Past EPS"), mpatches.Patch(color="orange", label="Future EPS")], loc="upper left")

    # Labels and title
    ax.set_ylabel("Past and projected EPS")
    ax.set_title("Plot")
    ax.set_xticks(index)
    ax.set_xticklabels(dates)

    ax2 = ax.twinx()
    offset_price = max(close_adj_pe)*0.02
    ax2.plot(index, close_adj_pe, color="red", marker="o", linestyle="-", label="Share price ct pe")
    ax2.set_ylabel("Share price")
    ax2.set_ylim((- offset_price, max(close_adj_pe) + offset_price))
    # Legends
    ax2.legend(loc="upper right")

    # Save plot
    plt.savefig(DIR_OUT / "dividends_year.png", bbox_inches="tight")
    plt.close()