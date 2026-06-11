"""Matplotlib plotting helpers for the PC2023 publication figures."""

import textwrap
from datetime import timedelta
from pathlib import Path
from typing import Literal, TypedDict, Unpack

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# import matplotlib as mpl
# mpl.use('macOsX')

plt.rcParams.update(
    {
        "text.usetex": False,
        "font.family": "cmr10",
        "font.serif": "cmr10",
        "axes.labelsize": 8,
        "axes.grid": True,
        "font.size": 8,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.figsize": mpl.rcParamsDefault["figure.figsize"],
        "figure.subplot.left": 0.1,
        "figure.subplot.bottom": 0.2,
        "figure.subplot.right": 0.95,
        "figure.subplot.top": 0.85,
        "axes.formatter.use_mathtext": True,
        # "backend": "macOsX"
    },
)

PLOT_WIDTH = 0.75 * 398.3386


class LimitsKwargs(TypedDict, total=False):
    """Optional keyword arguments shared by the limit plotting helpers."""

    ylim: tuple[float, float]
    xticks_on: pd.Series


class GridKwargs(TypedDict, total=False):
    """Optional keyword arguments for the grid plotting helper."""

    resample: str
    grace_period: int | timedelta


locator = mdates.AutoDateLocator()
formatter = mdates.ConciseDateFormatter(
    locator,
    formats=["%Y", "%d %b", "%d %b", "%H:%M", "%H:%M", "%S.%f"],
    offset_formats=["", "%Y", "", "", "", "%Y-%b-%d %H:%M"],
)


def set_size(
    width: float | Literal["thesis", "beamer"] = 307.28987,
    fraction: float = 1,
    subplots: tuple[float, float] = (1, 1),
) -> tuple[float, float]:
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches

    """
    if width == "thesis":
        width_pt = 426.79135
    elif width == "beamer":
        width_pt = 307.28987
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)


def set_axis_style(
    ax: plt.Axes,
    ser: pd.Series,
    xlabel: str = "",
    ylabel: str = "",
) -> None:
    """Apply shared axis labels, limits, and date formatting to an axis."""
    ylabel = "\n".join(textwrap.wrap(ylabel, 11))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(
        f"{ylabel}",
        rotation=0,
        horizontalalignment="left",
    )
    ax.yaxis.set_label_position("right")
    ax.set_xlim(left=ser.index.min(), right=ser.index.max())
    ax.set_ylim(ser.min(), ser.max())
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.tick_params(axis="x", labelrotation=50, labelsize=8)


def plot_anomalies(ax: plt.Axes, a: pd.Series) -> None:
    """Shade red vertical spans between anomaly start and end indices."""
    for x0, x1 in zip(a[a == 1].index, a[a == -1].index, strict=False):
        ax.axvspan(
            x0,
            x1,
            facecolor="red",
            alpha=0.5,
            linewidth=1.1,
            edgecolor="red",
        )


def make_name(
    name: str,
    window: timedelta | None,
    file_name: str | None,
) -> str:
    """Build a default output file name from the series name and window."""
    if file_name is None:
        if window:
            file_name = (
                f"{name.replace(' ', '_')}_"
                f"{int(window.total_seconds() / 60 / 60)}_hours_sliding"
            )
        else:
            file_name = f"{name.replace(' ', '_')}_sliding"
    return file_name


def plot_limits_(
    ser: pd.Series,
    anomalies: pd.Series | None = None,
    ser_high: pd.Series | None = None,
    ser_low: pd.Series | None = None,
    window: timedelta | None = None,
    file_name: str | None = None,
    save: bool = True,
    **kwargs: Unpack[LimitsKwargs],
) -> None:
    """Plot a signal with anomalies and dynamic limits, optionally saving.

    Args:
        ser: Signal series to plot.
        anomalies: Binary anomaly flags aligned with ``ser``.
        ser_high: Upper dynamic limit series.
        ser_low: Lower dynamic limit series.
        window: Sliding window length used to derive the file name.
        file_name: Output file name stem; derived from ``ser`` if None.
        save: Whether to save the figure as a PDF.
        **kwargs: Optional ``ylim`` tuple and ``xticks_on`` selector.

    """
    file_name = make_name(str(ser.name), window, file_name)

    fig, ax = plt.subplots(
        figsize=set_size(
            PLOT_WIDTH,
        ),
    )

    set_axis_style(ax, ser, "Date", f"{ser.name} [-]")
    if "ylim" not in kwargs:
        kwargs["ylim"] = (ser.min(), ser.max())
    ax.set_ylim(*kwargs["ylim"])

    if kwargs.get("xticks_on") == "anomalies" and anomalies is not None:
        a = anomalies.astype(int).diff()
        b = a[a == 1].resample("1d").sum()
        ax.set_xticks(b[b > 0].index.map(str))
    elif kwargs.get("xticks_on"):
        ax.set_xticks(kwargs["xticks_on"].index.map(str))

    ax.plot(ser.resample("1t").asfreq(), linewidth=0.7, label="Signal")

    if anomalies is not None:
        an_ser = ser.copy()
        an_ser[anomalies == 0] = None
        ax.plot(
            an_ser,
            linewidth=1.2,
            color="r",
            label="Anomalies",
            marker=".",
            markersize=0.8,
        )

    if ser_high is not None and ser_low is not None:
        ax = plot_limits(ax, ser_high, ser_low, kwargs["ylim"])

    ax.legend(
        bbox_to_anchor=(0.0, 1.05, 1.0, 0.102),
        loc="lower left",
        ncols=3,
        mode="expand",
        borderaxespad=0.0,
    )

    if save:
        fig.savefig(f"{file_name}_thresh.pdf", backend="pdf")


def plot_limits(
    ax: plt.Axes,
    ser_high: pd.Series,
    ser_low: pd.Series,
    ylim: tuple[float, float],
) -> plt.Axes:
    """Fill the regions outside the high/low limits on the given axis.

    Args:
        ax: Axis to draw on.
        ser_high: Upper limit series.
        ser_low: Lower limit series.
        ylim: Y-axis bounds used as the outer edge of the fills.

    Returns:
        The axis with the limit regions drawn.

    """
    if (ser_high is not None) and (ser_low is not None):
        ax.fill_between(
            ser_high.index,
            ser_high,
            ylim[1],
            label=r"Limits",
            color=(1, 0, 0, 0.1),
            edgecolor=(1, 0, 0, 0.25),
            linestyle="-",
            linewidth=0.7,
        )
        ax.fill_between(
            ser_low.index,
            ser_low,
            ylim[0],
            color=(1, 0, 0, 0.1),
            edgecolor=(1, 0, 0, 0.25),
            linestyle="-",
            linewidth=0.7,
        )
    return ax


def plot_compare_anomalies_(
    ser: pd.Series,
    anomalies: pd.DataFrame,
    window: timedelta | None = None,
    file_name: str | None = None,
    save: bool = True,
    **kwargs: Unpack[LimitsKwargs],
) -> None:
    """Plot the signal in stacked subplots, one per anomaly detector.

    Args:
        ser: Signal series to plot in each subplot.
        anomalies: DataFrame with one binary anomaly column per detector.
        window: Sliding window length used to derive the file name.
        file_name: Output file name stem; derived from ``ser`` if None.
        save: Whether to save each subplot as a PDF.
        **kwargs: Optional ``ylim`` tuple and ``xticks_on`` selector.

    """
    file_name = make_name(str(ser.name), window, file_name)

    n_rows = len(anomalies.columns)
    _, axs = plt.subplots(
        nrows=n_rows,
        ncols=1,
        figsize=set_size(subplots=(1, 1)),
        sharex=True,
    )

    if "ylim" not in kwargs:
        kwargs["ylim"] = (ser.min(), ser.max())

    [
        (set_axis_style(ax, ser, "", ""), ax.set_ylim(kwargs["ylim"]))
        for ax in axs
    ]

    if kwargs.get("xticks_on") == "anomalies":
        a = anomalies.iloc[:, -1].astype(int).diff()
        b = a[a == 1].resample("1d").sum()
        set_axis_style(axs[-1], ser, "Data", "")
        axs[-1].set_xticks(b[b > 0].index.map(str))
    elif kwargs.get("xticks_on"):
        axs[-1].set_xticks(kwargs["xticks_on"].index.map(str))

    for row, anomaly in enumerate(anomalies, start=0):
        axs[row].plot(
            ser.resample("1t").asfreq(),
            linewidth=0.7,
            label="Signal",
        )

        axs[row].set_ylim(kwargs["ylim"])

        a = anomalies[anomaly].astype(int).diff()
        plot_anomalies(axs[row], a)

        axs[0].legend(
            ["Signal", "Anomalies"],
            bbox_to_anchor=(0.0, 1.05, 1.0, 0.102),
            loc="lower left",
            ncols=2,
            mode="expand",
            borderaxespad=0.0,
        )

        if save:
            plt.savefig(f"{file_name}_compare_anomalies_{chr(97 + row)}.pdf")

    plt.show()


def plot_anomaly_bars(
    args: tuple[pd.Series | None, ...],
    colors: list[str],
    axs: np.ndarray,
) -> None:
    """Draw labelled anomaly-event bars on the trailing subplot axes.

    Args:
        args: Binary anomaly series; non-Series entries are skipped.
        colors: Color cycle indexed by the position of each series.
        axs: Subplot axes; bars fill the axes from the end backwards.

    """
    for i, a in enumerate(args, start=1):
        if isinstance(a, pd.Series):
            ax: plt.Axes = axs[-i]
            a_diff = a.astype(int).diff().fillna(0)
            if (a_diff != 0).any():
                if a_diff[a_diff != 0].iloc[0] == -1:
                    a_diff.iloc[1] = 1
                elif a_diff[a_diff != 0].iloc[-1] == 1:
                    a_diff.iloc[-1] = -1
            for s_idx, (x0, x1) in enumerate(
                zip(
                    a_diff[a_diff == 1].index,
                    a_diff[a_diff == -1].index,
                    strict=False,
                ),
            ):
                ax.axvspan(
                    x0,
                    x1,
                    color=colors[i],
                    alpha=1,
                    label="_" * s_idx + str(a_diff.name),
                    linewidth=2,
                )
            ylabel = "\n".join(textwrap.wrap(str(a_diff.name), 11))
            ax.set_ylabel(
                f"{ylabel}",
                rotation=0,
                horizontalalignment="left",
            )
            ax.yaxis.set_label_position("right")
            ax.set_yticks([])
            if a_diff.name == "Ground Truth":
                a_diff2 = a_diff.astype(int).diff()
                b = a_diff2[a_diff2 == 1].resample("1d").sum()
                axs[-1].set_xticks(b[b > 0].index.map(str))


def plot_limits_grid_(
    df: pd.DataFrame,
    *args: pd.Series | None,
    ser_high: pd.Series | None = None,
    ser_low: pd.Series | None = None,
    signal_anomaly: pd.Series | None = None,
    file_name: str | None = None,
    save: bool = True,
    # anomalies: pd.Series,
    # changepoints: Union[pd.Series, None] = None,
    # samplings: Union[pd.Series, None] = None,
    # ground_truth: Union[pd.Series, None] = None,
    **kwargs: Unpack[GridKwargs],
) -> None:
    """Plot a grid of signals with limits, anomalies, and event bars.

    Args:
        df: DataFrame with one signal column per subplot row.
        *args: Binary anomaly series rendered as bar rows at the bottom.
        ser_high: Per-row upper limit series keyed by column name.
        ser_low: Per-row lower limit series keyed by column name.
        signal_anomaly: Per-row anomaly flags keyed by column name.
        file_name: Output file name stem for the saved PDF.
        save: Whether to save the figure under ``plots/``.
        **kwargs: Optional ``resample`` rule and ``grace_period`` marker.

    """
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    # file_name = make_name(ser.name, window, file_name)

    n_rows = len(df.columns)
    # Count number of non nan args for subplots
    n_bar_plots = 0
    for i in args:
        if i is not None:
            n_bar_plots += 1

    fig, axs = plt.subplots(
        nrows=int(n_rows + n_bar_plots),
        ncols=1,
        figsize=set_size(
            "thesis",
            subplots=((n_rows + n_bar_plots * 0.2) / 2.3, 1),
        ),  # Kokam divide by 3.5
        sharex="col",
        sharey="row",
        gridspec_kw={
            "height_ratios": [*(n_rows * [1]), *(n_bar_plots * [0.2])],
        },
    )
    axs = np.array([axs]) if isinstance(axs, plt.Axes) else axs.T.flatten()

    fig.subplots_adjust(
        left=0.05,
        bottom=0.1,
        right=0.85,
        top=0.95,
        hspace=0.15,
    )

    for i, col_name in enumerate(df.columns):
        ser: pd.Series[float] = df[col_name]
        if kwargs.get("resample"):
            ser_ = (
                ser.resample(rule=kwargs["resample"])
                .asfreq()
                .interpolate(method="time")
            )
        else:
            ser_ = ser

        ax: plt.Axes = axs[i]
        ax.plot(ser_, linewidth=0.7, label="Signal")
        set_axis_style(ax, ser, ylabel=col_name)
        ser_high_ = (
            ser_high.apply(lambda x, c=col_name: x[c])
            if ser_high is not None
            else None
        )
        ser_low_ = (
            ser_low.apply(lambda x, c=col_name: x[c])
            if ser_low is not None
            else None
        )

        if ser_high_ is not None and ser_low_ is not None:
            ax = plot_limits(
                ax,
                ser_high_,
                ser_low_,
                (min(ser.min(), 0), max(ser.max(), 1)),
            )

        if signal_anomaly is not None:
            a = signal_anomaly.apply(lambda x, c=col_name: x[c])
            ax.scatter(
                ser[a].index,
                ser[a],
                color=colors[3],
                s=3,
                label="Signal Anomalies",
            )
        if kwargs.get("grace_period"):
            if isinstance(kwargs["grace_period"], int):
                xmax = ser.index[int(kwargs["grace_period"])]
            elif isinstance(kwargs["grace_period"], timedelta):
                xmax = ser.index[0] + kwargs["grace_period"]
            elif isinstance(kwargs["grace_period"], pd.Timestamp):
                xmax = kwargs["grace_period"]
            else:
                xmax = ser.index[0]
            ax.axvspan(
                ser.index[0],
                xmax,
                color="0.8",
                alpha=0.75,
                label="Grace Period",
            )

        ax.tick_params(axis="both", which="major", labelsize=8)

        # Kokam module - 1st case study - second
        # if a['2023-08-23 16:00':'2023-08-23 18:00'].any():
        #     axins1 = ax.inset_axes(
        #         [0.45, 0.1, 0.20, 0.40], xticklabels=[], yticklabels=[])
        #     axins1.plot(ser_['2023-08-23 16:00':'2023-08-23 18:00'])
        #     axins1.scatter(
        #         ser[a]['2023-08-23 16:00':'2023-08-23 18:00'].index,
        #         ser[a]['2023-08-23 16:00':'2023-08-23 18:00'],
        #         color=colors[3], s=3, label='Signal Anomalies',
        #         zorder=2)
        #     axins1.grid(False)
        #     ax.indicate_inset_zoom(axins1, edgecolor="black")
        # if a['2023-08-24 17:00':'2023-08-24 20:00'].any():
        #     axins1 = ax.inset_axes(
        #         [0.8, 0.1, 0.20, 0.40], xticklabels=[], yticklabels=[])
        #     axins1.plot(ser_['2023-08-24 17:00':'2023-08-24 20:00'])
        #     axins1.scatter(
        #             ser[a]['2023-08-24 17:00':'2023-08-24 20:00'].index,
        #             ser[a]['2023-08-24 17:00':'2023-08-24 20:00'],
        #                 color=colors[3], s=3, label='Signal Anomalies',
        #                 zorder=2)
        #     axins1.grid(False)
        #     ax.indicate_inset_zoom(axins1, edgecolor="black")

        # # Kokam module - 2nd case study - second
        # if a['2023-08-27 01:00':'2023-08-27 06:00'].any():
        #     axins1 = ax.inset_axes(
        #         [0.2, 0.6, 0.20, 0.40], xticklabels=[], yticklabels=[])
        #     axins1.plot(ser_['2023-08-27 01:00':'2023-08-27 06:00'])
        #     axins1.scatter(
        #         ser[a]['2023-08-27 01:00':'2023-08-27 06:00'].index,
        #         ser[a]['2023-08-27 01:00':'2023-08-27 06:00'],
        #         color=colors[3], s=3, label='Signal Anomalies',
        #         zorder=2)
        #     axins1.grid(False)
        #     ax.indicate_inset_zoom(axins1, edgecolor="black")
        # if a['2023-08-28 03:00':'2023-08-28 05:00'].any():
        #     axins1 = ax.inset_axes(
        #         [0.7, 0.6, 0.20, 0.40], xticklabels=[], yticklabels=[])
        #     axins1.plot(ser_['2023-08-28 03:00':'2023-08-28 05:00'])
        #     axins1.scatter(
        #             ser[a]['2023-08-28 03:00':'2023-08-28 05:00'].index,
        #             ser[a]['2023-08-28 03:00':'2023-08-28 05:00'],
        #                 color=colors[3], s=3, label='Signal Anomalies',
        #                 zorder=2)
        #     axins1.grid(False)
        #     ax.indicate_inset_zoom(axins1, edgecolor="black")

    plot_anomaly_bars(args, colors, axs)

    # # TERRA - 1st case study
    # axins1 = axs[0].inset_axes(
    #     [0.05, 0.1, 0.20, 0.40], xticklabels=[], yticklabels=[])
    # axins1.plot(ser_['2022-03-06 14:00':'2022-03-06 15:00'])
    # axins1.grid(False)
    # axs[0].indicate_inset_zoom(axins1, edgecolor="black")
    # axins1 = axs[0].inset_axes(
    #     [0.6, 0.1, 0.20, 0.40], xticklabels=[], yticklabels=[])
    # axins1.plot(ser_['2022-03-12 21:00':'2022-03-12 22:00'])
    # axins1.grid(False)
    # axs[0].indicate_inset_zoom(axins1, edgecolor="black")

    axs[0].legend(
        bbox_to_anchor=(0.0, 1.05, 1.0, 0.102),
        loc="lower left",
        ncols=4,
        mode="expand",
        borderaxespad=0.0,
    )

    axs[-1].tick_params(axis="x", labelrotation=50, labelsize=8)

    fig.tight_layout()

    if save:
        Path("plots").mkdir(parents=True, exist_ok=True)
        plt.savefig(f"plots/{file_name}_thresh.pdf")

    plt.show()
