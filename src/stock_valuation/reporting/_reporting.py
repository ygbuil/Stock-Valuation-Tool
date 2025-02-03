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
        data_and_pred: DataFrame containing "date", "eps", "period", "close_adj_origin_currency_pe_ct",
        and "close_adj_origin_currency_pe_exp".
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

    # Crear figura con 2 subgráficos (uno encima del otro)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [1, 1]})

    # Gráfico superior: EPS (barras)
    for idx, height, period in zip(index, eps, periods, strict=False):
        ax1.bar(idx, height, bar_width, color="blue" if period == "past" else "orange")

    max_y_ax1 = max(eps)
    ax1.grid(True, axis="y")
    ax1.set_axisbelow(True)
    ax1.set_xlim((-0.5, time_series_dim))
    ax1.set_ylim((-max_y_ax1 * 0.02, max_y_ax1 * 1.05))
    ax1.legend(
        handles=[
            mpatches.Patch(color="blue", label="Past EPS"),
            mpatches.Patch(color="orange", label="Future EPS"),
        ],
        loc="upper left",
    )
    ax1.set_ylabel("Past and future EPS")
    ax1.set_title("EPS and Share Price Trends")

    # Gráfico inferior: PE ajustado (líneas)
    max_y_ax2 = max(close_pe_ct + close_pe_exp)
    ax2.grid(True)
    ax2.set_axisbelow(True)
    ax2.plot(index, close_pe_exp, color="green", marker="o", linestyle="-", label="Share price exp pe")
    ax2.plot(index, close_pe_ct, color="red", marker="o", linestyle="-", label="Share price ct pe")
    ax2.set_ylabel("Share price")
    ax2.set_ylim((-max_y_ax2 * 0.02, max_y_ax2 * 1.05))
    ax2.legend(loc="upper left")

    # Etiquetas de tiempo en el eje X
    ax2.set_xticks(index)
    ax2.set_xticklabels(dates, rotation=80)

    # Guardar la imagen
    plt.savefig(DIR_OUT / "dividends_year.png", bbox_inches="tight")
    plt.close()