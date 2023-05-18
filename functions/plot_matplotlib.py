from datetime import timedelta

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib as mpl
mpl.use('macOsX')

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Times New Roman",
    "axes.labelsize": 10,
    "axes.grid": True,
    "font.size": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})
plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]
plt.rcParams["backend"] = "macOsX"


locator = mdates.AutoDateLocator()
formatter = mdates.ConciseDateFormatter(
    locator,
    formats=['%Y', '%d %b', '%d %b', '%H:%M', '%H:%M', '%S.%f'],
    offset_formats=['', '%Y', '', '%Y-%b-%d', '%Y-%b-%d', '%Y-%b-%d %H:%M'])


def set_size(width=307.28987, fraction=1.4, subplots=(1, 1)):
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


def set_axis_style(ax, ser, name):
    ax.set_xlabel('Date')
    ax.set_ylabel(f"{name}")
    ax.set_xlim([ser.index.min(), ser.index.max()])
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
    anomalies: pd.Series,
    ser_high: pd.Series | None = None,
    ser_low: pd.Series | None = None,
    window: timedelta | None = None,
    file_name: str | None = None,
    save: bool = True,
    **kwargs
):
    file_name = make_name(ser.name, window, file_name)

    a = anomalies.astype(int).diff()
    b = a[a == 1].resample('1d').sum()

    fig, ax = plt.subplots(figsize=set_size())

    ax.plot(ser.index, ser, linewidth=0.7)
    set_axis_style(ax, ser, f"{ser.name} [-]")
    ax.set_xticks(b[b > 0].index.map(str))

    plot_anomalies(ax, a)

    if (ser_high is not None) and (ser_low is not None):
        ax.plot(ser_high.index, ser_high, color=(
            1, 0, 0, 0.25), linewidth=0.7, label=r'Threshold')
        ax.plot(ser_low.index, ser_low, color=(1, 0, 0, 0.25),
                linewidth=0.7, label=r'Threshold')

        ax.fill_between(ser_high.index, ser_high, ser.max(),
                        color=(1, 0, 0, 0.1), alpha=0.1)
        ax.fill_between(ser_low.index, ser_low, ser.min(),
                        color=(1, 0, 0, 0.1), alpha=0.1)

    ax.legend(
        ['Signal', "Anomalies"], bbox_to_anchor=(0., 1.05, 1., .102),
        loc='lower left', ncols=2, mode="expand", borderaxespad=0.)
    # fig.tight_layout()
    plt.subplots_adjust(bottom=0.3)
    if save:
        fig.savefig(f"{file_name}_thresh.pdf", backend='pdf')


def plot_compare_anomalies_(
        ser: pd.Series,
        anomalies: pd.DataFrame,
        window: timedelta | None = None,
        file_name: str | None = None,
        save: bool = True,
        **kwargs):

    file_name = make_name(ser.name, window, file_name)

    n_rows = len(anomalies.columns)
    _, axs = plt.subplots(nrows=n_rows, ncols=1,
                          figsize=set_size(subplots=(1, 1)),
                          sharex=True)

    for row, anomaly in enumerate(anomalies, start=0):
        a = anomalies[anomaly].astype(int).diff()
        b = a[a == 1].resample('1d').sum()

        axs[row].plot(ser.index, ser, linewidth=0.7)
        set_axis_style(axs[row], ser, f'{chr(97+row)}) {anomaly}')
        axs[row].set_xticks(b[b > 0].index.map(str))

        plot_anomalies(axs[row], a)

    axs[0].legend(
        ['Signal', "Anomalies"], bbox_to_anchor=(0., 1.05, 1., .102),
        loc='lower left', ncols=2, mode="expand", borderaxespad=0.)
    plt.subplots_adjust(bottom=0.3)

    if save:
        plt.savefig(f"{file_name}_compare_anomalies.pdf")

    plt.show()
