from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DIR_OUT = Path("data/out")


def reporting(data_and_pred: pd.DataFrame) -> None:
    _plot_dividends_year(data_and_pred)

def _plot_dividends_year(data_and_pred: pd.DataFrame) -> None:
    """Plot past and projected EPS along with close-adjusted PE in separate graphs.

    Args:
        data_and_pred: DataFrame containing "date", "eps", "period", "pe_ct", "pe_exp",
        "close_adj_origin_currency_pe_ct", and "close_adj_origin_currency_pe_exp".
    """
    dates, eps, periods, pe_ct, pe_exp, close_pe_ct, close_pe_exp = (
        list(reversed([date.strftime("%Y-%m") for date in data_and_pred["date"]])),
        list(reversed(data_and_pred["eps"])),
        list(reversed(data_and_pred["period"])),
        list(reversed(data_and_pred["pe_ct"])),
        list(reversed(data_and_pred["pe_exp"])),
        list(reversed(data_and_pred["close_adj_origin_currency_pe_ct"])),
        list(reversed(data_and_pred["close_adj_origin_currency_pe_exp"])),
    )

    time_series_dim = len(eps)
    bar_width = 0.4
    index = np.arange(time_series_dim)

    # Create figure with 2 subplots (one on top of the other)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10), sharex=True, gridspec_kw={'height_ratios': [1, 1]})

    # Upper plot: EPS (bars) + PE (lines)
    ax1.set_axisbelow(True)
    ax1.grid(True, axis="y", linestyle="--", alpha=0.6, zorder=0)

    for idx, height, period in zip(index, eps, periods, strict=False):
        ax1.bar(idx, height, bar_width, color="blue" if period == "past" else "orange", zorder=3)

    ax1.set_ylabel("EPS")
    ax1.set_title("EPS, PE Ratio, and Share Price Trends")

    # Secondary Y-axis for PE
    ax1b = ax1.twinx()
    ax1b.plot(index, pe_exp, color="green", marker="s", linestyle="-", label="PE EXP", zorder=2)
    ax1b.plot(index, pe_ct, color="red", marker="s", linestyle="-", label="PE CT", zorder=2)
    ax1b.set_ylabel("PE Ratio")
    
    # Legends
    ax1.legend(
        handles=[
            mpatches.Patch(color="blue", label="Past EPS"),
            mpatches.Patch(color="orange", label="Future EPS"),
        ],
        loc="upper left",
    )
    ax1b.legend(loc="upper right")

    # Lower plot: Adjusted PE (lines)
    ax2.set_axisbelow(True)
    ax2.grid(True, linestyle="--", alpha=0.6, zorder=0)

    ax2.plot(index, close_pe_exp, color="green", marker="o", linestyle="-", label="Share price exp pe", zorder=2)
    ax2.plot(index, close_pe_ct, color="red", marker="o", linestyle="-", label="Share price ct pe", zorder=2)

    ax2.set_ylabel("Share Price PE")
    ax2.legend(loc="upper left")

    # X-axis labels
    ax2.set_xticks(index)
    ax2.set_xticklabels(dates, rotation=80)

    # Save image
    plt.savefig(DIR_OUT / "dividends_year.png", bbox_inches="tight")
    plt.close()
