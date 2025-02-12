from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DIR_OUT = Path("data/out")


def reporting(data_and_pred: pd.DataFrame, returns: pd.DataFrame) -> None:
    _plot_funadamentals_projections(data_and_pred)

    _plot_performance(returns)


def _plot_funadamentals_projections(data_and_pred: pd.DataFrame) -> None:
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
    fig, (eps_and_pe, price) = plt.subplots(
        2, 1, figsize=(18, 10), gridspec_kw={"height_ratios": [1, 1]}
    )

    # Upper plot: EPS (bars) + PE (lines)
    eps_and_pe.set_axisbelow(True)
    eps_and_pe.grid(visible=True, axis="y", linestyle="--", alpha=0.6, zorder=0)

    for idx, height, period in zip(index, eps, periods, strict=False):
        match period:
            case "past":
                eps_and_pe.bar(idx, height, bar_width, color="blue", zorder=3)
            case "present":
                eps_and_pe.bar(idx, height, bar_width, color="black", zorder=3)
            case "future":
                eps_and_pe.bar(idx, height, bar_width, color="orange", zorder=3)

    eps_and_pe.set_ylabel("EPS")
    eps_and_pe.set_title("EPS, PE Ratio, and Share Price Trends")

    eps_and_pe.set_xticks(index)
    eps_and_pe.set_xticklabels(dates, rotation=80)

    # Secondary Y-axis for PE
    eps_and_peb = eps_and_pe.twinx()
    eps_and_peb.plot(
        index, pe_exp, color="green", marker="s", linestyle="-", label="PE EXP", zorder=2
    )
    eps_and_peb.plot(index, pe_ct, color="red", marker="s", linestyle="-", label="PE CT", zorder=2)
    eps_and_peb.set_ylabel("PE Ratio")

    # Legends
    eps_and_pe.legend(
        handles=[
            mpatches.Patch(color="blue", label="Past EPS"),
            mpatches.Patch(color="black", label="Current EPS"),
            mpatches.Patch(color="orange", label="Future EPS"),
        ],
        loc="upper left",
    )
    eps_and_peb.legend(loc="upper right")

    # Lower plot: Adjusted PE (lines)
    price.set_axisbelow(True)
    price.grid(visible=True, linestyle="--", alpha=0.6, zorder=0)

    price.plot(
        index,
        close_pe_exp,
        color="green",
        marker="o",
        linestyle="-",
        label="Share price exp pe",
        zorder=2,
    )
    price.plot(
        index,
        close_pe_ct,
        color="red",
        marker="o",
        linestyle="-",
        label="Share price ct pe",
        zorder=2,
    )

    price.set_ylabel("Share Price PE")
    price.legend(loc="upper left")

    # X-axis labels
    price.set_xticks(index)
    price.set_xticklabels(dates, rotation=80)

    # Save image
    plt.savefig(DIR_OUT / "funadamentals_projections.png", bbox_inches="tight")
    plt.close()


def _plot_performance(returns: pd.DataFrame) -> None:
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_axis_off()  # Hide axes

    ax.table(
        cellText=returns.values,  # type: ignore
        colLabels=returns.columns,  # type: ignore
        cellLoc="center",
        loc="center",
    )

    # Adjust layout
    plt.tight_layout()
    plt.savefig(
        DIR_OUT
        / Path(
            "returns.png",
        ),
    )
    plt.close()
