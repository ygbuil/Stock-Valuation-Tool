import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

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

    n = len(eps)
    bar_width = 0.4
    index = np.arange(n)

    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot bars with respective colors
    for idx, height, period in zip(index, eps, periods):
        ax.bar(idx, height, bar_width, color="blue" if period == "past" else "orange")

    # Set y-axis limits for EPS
    offset = max(eps)*0.02
    top_y_lim = max(eps) + offset
    bottom_y_lim =  - offset
    margin = (abs(top_y_lim) + abs(bottom_y_lim)) * 0.02
    ax.set_xlim((-0.5, n))
    ax.set_ylim((bottom_y_lim, top_y_lim))

    # Labels and title
    ax.set_ylabel("Past and projected EPS")
    ax.set_title("Share price ct pe")
    ax.set_xticks(index)
    ax.set_xticklabels(dates)

    # Add secondary y-axis for Close-Adjusted PE
    ax2 = ax.twinx()
    ax2.plot(index, close_adj_pe, color="red", marker="o", linestyle="-", label="Close-Adjusted PE (CT)")
    ax2.set_ylabel("Close-Adjusted PE")
    
    # Legends
    ax.legend(["Past EPS", "Future EPS"], loc="upper left")
    ax2.legend(loc="upper right")

    # Annotate EPS bars
    y_offset = bottom_y_lim - (abs(top_y_lim) + abs(bottom_y_lim)) * 0.11
    for i, height in zip(index, eps):
        ax.text(i, y_offset, f"{height:.2f}", ha="center", color="black", fontweight="bold")

    # Save plot
    plt.savefig(DIR_OUT / "dividends_year.png", bbox_inches="tight")
    plt.close()