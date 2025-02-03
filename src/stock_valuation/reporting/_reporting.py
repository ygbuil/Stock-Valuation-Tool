from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DIR_OUT = Path("data/out")


def reporting(data_and_pred: pd.DataFrame) -> None:
    _plot_dividends_year(data_and_pred)


def _plot_dividends_year(data_and_pred: pd.DataFrame) -> None:
    """Plot past and projected EPS along with close-adjusted PE.

    Args:
        data_and_pred: DataFrame containing "date", "eps", "period", and
        "close_adj_origin_currency_pe_ct".
    """
    dates, eps, periods, close_pe_ct, close_pe_exp = (
        list(reversed([date.strftime("%Y-%m") for date in data_and_pred["date"]])),
        list(reversed(data_and_pred["eps"])),
        list(reversed(data_and_pred["period"])),
        list(reversed(data_and_pred["close_adj_origin_currency_pe_ct"])),
        list(reversed(data_and_pred["close_adj_origin_currency_pe_exp"])),
    )

    time_series_dim = len(eps)
    bar_width = 0.4
    index = np.arange(time_series_dim)

    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot bars with respective colors
    for idx, height, period in zip(index, eps, periods, strict=False):
        ax.bar(idx, height, bar_width, color="blue" if period == "past" else "orange")

    # Set y-axis limits for EPS
    max_y_ax = max(eps)
    ax.set_xlim((-0.5, time_series_dim))
    ax.set_ylim((-max_y_ax * 0.02, max(eps) + max_y_ax * 1.2))
    ax.legend(
        handles=[
            mpatches.Patch(color="blue", label="Past EPS"),
            mpatches.Patch(color="orange", label="Future EPS"),
        ],
        loc="upper left",
    )
    ax.set_ylabel("Past and future EPS")

    # Labels and title
    ax.set_title("Plot")
    ax.set_xticks(index)
    ax.set_xticklabels(dates)

    ax2 = ax.twinx()
    max_y_ax2 = max(close_pe_ct + close_pe_exp)
    ax2.plot(index, close_pe_ct, color="red", marker="o", linestyle="-", label="Share price ct pe")
    ax2.plot(
        index, close_pe_exp, color="green", marker="o", linestyle="-", label="Share price ct pe"
    )
    ax2.set_ylabel("Share price")
    ax2.set_ylim((-max_y_ax2 * 0.02, max(close_pe_ct + close_pe_exp) * 1.2))
    ax2.legend(loc="upper right")

    # Save plot
    plt.savefig(DIR_OUT / "dividends_year.png", bbox_inches="tight")
    plt.close()
