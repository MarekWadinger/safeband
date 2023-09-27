import textwrap

from datetime import timedelta
from typing import Union

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
# import matplotlib as mpl
# mpl.use('macOsX')

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "Times New Roman",
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
    # "backend": "macOsX"
})

PLOT_WIDTH = 0.75*398.3386

locator = mdates.AutoDateLocator()
formatter = mdates.ConciseDateFormatter(
    locator,
    formats=['%Y', '%d %b', '%d %b', '%H:%M', '%H:%M', '%S.%f'],
    offset_formats=['', '%Y', '', '', '', '%Y-%b-%d %H:%M'])


def set_size(width=307.28987, fraction=1, subplots=(1, 1)):
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
    if width == 'thesis':
        width_pt = 426.79135
    elif width == 'beamer':
        width_pt = 307.28987
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)


def set_axis_style(ax: plt.Axes, ser, xlabel='', ylabel=''):
    ylabel = '\n'.join(textwrap.wrap(ylabel, 11))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"{ylabel}", rotation=0, horizontalalignment='left',)
    ax.yaxis.set_label_position("right")
    ax.set_xlim(left=ser.index.min(), right=ser.index.max())
    ax.set_ylim(ser.min(), ser.max())
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.tick_params(axis='x', labelrotation=50, labelsize=8)


def plot_anomalies(ax, a):
    for x0, x1 in zip(a[a == 1].index, a[a == -1].index):
        ax.axvspan(x0, x1, facecolor='red', alpha=0.5,
                   linewidth=1.1, edgecolor='red')


def make_name(name, window, file_name):
    if file_name is None:
        if window:
            file_name = (
                f"{name.replace(' ', '_')}_"
                f"{int(window.total_seconds()/60/60)}_hours_sliding"
            )
        else:
            file_name = f"{name.replace(' ', '_')}_sliding"
    return file_name


def plot_limits_(
    ser: pd.Series,
    anomalies: Union[pd.Series, None] = None,
    ser_high: Union[pd.Series, None] = None,
    ser_low: Union[pd.Series, None] = None,
    window: Union[timedelta, None] = None,
    file_name: Union[str, None] = None,
    save: bool = True,
    **kwargs
):
    file_name = make_name(ser.name, window, file_name)

    fig, ax = plt.subplots(figsize=set_size(PLOT_WIDTH, ))

    set_axis_style(ax, ser, "Date", f"{ser.name} [-]")
    if "ylim" not in kwargs:
        kwargs["ylim"] = (ser.min(), ser.max())
    ax.set_ylim(kwargs["ylim"])

    if kwargs.get("xticks_on") == "anomalies":
        a = anomalies.astype(int).diff()
        b = a[a == 1].resample('1d').sum()
        ax.set_xticks(b[b > 0].index.map(str))
    elif kwargs.get("xticks_on"):
        ax.set_xticks(kwargs["xticks_on"].index.map(str))

    ax.plot(ser.resample('1t').fillna(None), linewidth=0.7, label="Signal")

    if anomalies is not None:
        an_ser = ser.copy()
        an_ser[anomalies == 0] = None
        ax.plot(an_ser, linewidth=1.2, color="r", label="Anomalies",
                marker='.', markersize=0.8)

    if (ser_high is not None) and (ser_low is not None):
        ax.fill_between(ser_high.index, ser_high, kwargs["ylim"][1],
                        label=r'Limits',
                        color=(1, 0, 0, 0.1), edgecolor=(1, 0, 0, 0.5),
                        linestyle="-", linewidth=0.7,)
        ax.fill_between(ser_low.index, ser_low, kwargs["ylim"][0],
                        color=(1, 0, 0, 0.1), edgecolor=(1, 0, 0, 0.5),
                        linestyle="-", linewidth=0.7,)

    ax.legend(bbox_to_anchor=(0., 1.05, 1., .102),
              loc='lower left', ncols=3, mode="expand", borderaxespad=0.)

    if save:
        fig.savefig(f"{file_name}_thresh.pdf", backend='pdf')


def plot_compare_anomalies_(
        ser: pd.Series,
        anomalies: pd.DataFrame,
        window: Union[timedelta, None] = None,
        file_name: Union[str, None] = None,
        save: bool = True,
        **kwargs):

    file_name = make_name(ser.name, window, file_name)

    n_rows = len(anomalies.columns)
    _, axs = plt.subplots(nrows=n_rows, ncols=1,
                          figsize=set_size(subplots=(1, 1)),
                          sharex=True)

    if "ylim" not in kwargs:
        kwargs["ylim"] = (ser.min(), ser.max())

    [(set_axis_style(ax, ser, "", ''), ax.set_ylim(kwargs["ylim"]))
     for ax in axs]

    if kwargs.get("xticks_on") == "anomalies":
        a = anomalies.iloc[:, -1].astype(int).diff()
        b = a[a == 1].resample('1d').sum()
        set_axis_style(axs[-1], ser, "Data", '')
        axs[-1].set_xticks(b[b > 0].index.map(str))
    elif kwargs.get("xticks_on"):
        axs[-1].set_xticks(kwargs["xticks_on"].index.map(str))

    for row, anomaly in enumerate(anomalies, start=0):
        axs[row].plot(ser.resample('1t').fillna(None), linewidth=0.7,
                      label="Signal")

        axs[row].set_ylim(kwargs["ylim"])

        a = anomalies[anomaly].astype(int).diff()
        plot_anomalies(axs[row], a)

        axs[0].legend(
            ['Signal', "Anomalies"], bbox_to_anchor=(0., 1.05, 1., .102),
            loc='lower left', ncols=2, mode="expand", borderaxespad=0.)

        if save:
            plt.savefig(f"{file_name}_compare_anomalies_{chr(97+row)}.pdf")

    plt.show()


def plot_anomaly_bars(args, colors, axs):
    for i, a in enumerate(args, start=1):
        if isinstance(a, pd.Series):
            ax: plt.Axes = axs[-i]
            a = a.astype(int).diff()
            if a[a != 0].iloc[-1] == 1:
                a[-1] = -1
            for s_idx, (x0, x1) in enumerate(
                    zip(a[a == 1].index, a[a == -1].index)):
                ax.axvspan(x0, x1, color=colors[i], alpha=1,
                           label="_"*s_idx + a.name, linewidth=2)
            ylabel = '\n'.join(textwrap.wrap(a.name, 11))
            ax.set_ylabel(f"{ylabel}", rotation=0, horizontalalignment='left',)
            ax.yaxis.set_label_position("right")
            ax.set_yticks([])
            if a.name == "Ground Truth":
                a = a.astype(int).diff()
                b = a[a == 1].resample('1d').sum()
                axs[-1].set_xticks(b[b > 0].index.map(str))


def plot_limits_grid_(
        df: pd.DataFrame,
        *args,
        ser_high: Union[pd.Series, None] = None,
        ser_low: Union[pd.Series, None] = None,
        signal_anomaly: Union[pd.Series, None] = None,
        file_name: Union[str, None] = None,
        save: bool = True,
        # anomalies: pd.Series,
        # changepoints: Union[pd.Series, None] = None,
        # samplings: Union[pd.Series, None] = None,
        # ground_truth: Union[pd.Series, None] = None,
        **kwargs):
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    # file_name = make_name(ser.name, window, file_name)

    n_rows = len(df.columns)
    # Count number of non nan args for subplots
    n_bar_plots = 0
    for i in args:
        if i is not None:
            n_bar_plots += 1

    fig, axs = plt.subplots(
        nrows=n_rows+n_bar_plots, ncols=1,
        figsize=set_size('thesis', subplots=((n_rows+n_bar_plots)/6, 1)),
        sharex='col', sharey='row',
        gridspec_kw={
            # 'width_ratios': [3, 1],
            'height_ratios': [*(n_rows*[1]), *(n_bar_plots*[0.2])],
            }
        )

    axs = axs.T.flatten()
    fig.subplots_adjust(left=0.05, bottom=0.08, right=0.85, top=0.95)
    fig.subplots_adjust(hspace=0.15)

    for i, col_name in enumerate(df.columns):
        ser: pd.Series[float] = df[col_name]
        if kwargs.get('resample'):
            ser_ = ser.resample(
                rule=kwargs['resample']).fillna(None)
        else:
            ser_ = ser

        ax: plt.Axes = axs[i]
        ax.plot(ser_,
                linewidth=0.7, label='Signal')
        if kwargs.get('grace_period'):
            ax.axvspan(
                ser.index[0],  # type: ignore
                ser.index[int(kwargs['grace_period'])],  # type: ignore
                color='0.8', alpha=0.9, label='Grace Period')
        set_axis_style(ax, ser, ylabel=col_name)
        if signal_anomaly is not None:
            a = signal_anomaly.apply(lambda x: x[col_name])
            ax.scatter(ser[a].index, ser[a],
                       color=colors[3], s=3, label='Signal Anomalies')
        ser_high_ = ser_high.apply(
            lambda x: x[col_name]) if ser_high is not None else None
        ser_low_ = ser_low.apply(
            lambda x: x[col_name]) if ser_low is not None else None

        if ser_high_ is not None and ser_low_ is not None:
            ax.fill_between(ser_high_.index, ser_high_, ser.max(),
                            label=r'Limits',
                            color=(1, 0, 0, 0.1), edgecolor=(1, 0, 0, 0.25),
                            linestyle="-", linewidth=0.7,)
            ax.fill_between(ser_low_.index, ser_low_, ser.min(),
                            color=(1, 0, 0, 0.1), edgecolor=(1, 0, 0, 0.25),
                            linestyle="-", linewidth=0.7,)

        ax.tick_params(axis='both', which='major', labelsize=8)

    plot_anomaly_bars(args, colors, axs)

    axs[0].legend(bbox_to_anchor=(0., 1.05, 1., .102),
                  loc='lower left', ncols=4, mode="expand", borderaxespad=0.)

    axs[-1].tick_params(axis='x', labelrotation=50, labelsize=8)

    if save:
        plt.savefig(f"plots/{file_name}_thresh.pdf")

    plt.show()
