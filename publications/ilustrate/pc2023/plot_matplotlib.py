from datetime import timedelta

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
# import matplotlib as mpl
# mpl.use('macOsX')

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Times New Roman",
    "axes.labelsize": 10,
    "axes.grid": True,
    "font.size": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.figsize": plt.rcParamsDefault["figure.figsize"],
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


def set_axis_style(ax, ser, xlabel='', ylabel=''):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"{ylabel}")
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
    anomalies: pd.Series | None = None,
    ser_high: pd.Series | None = None,
    ser_low: pd.Series | None = None,
    window: timedelta | None = None,
    file_name: str | None = None,
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
        window: timedelta | None = None,
        file_name: str | None = None,
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


def plot_limits_grid_(
        df: pd.DataFrame,
        anomalies: pd.Series,
        ser_high: pd.Series | None = None,
        ser_low: pd.Series | None = None,
        file_name: str | None = None,
        save: bool = True,
        changepoints: pd.Series | None = None,
        samplings: pd.Series | None = None,
        **kwargs):

    # file_name = make_name(ser.name, window, file_name)

    a = anomalies.astype(int).diff()
    b = a[a == 1].resample('1d').sum()

    n_rows = len(df.columns)
    fig, axs = plt.subplots(nrows=n_rows, ncols=1,
                            figsize=set_size(subplots=(1, 1)),
                            sharex=True)
    fig.subplots_adjust(hspace=0.05)
    # y_labels = [r"$T_{\downarrow}$", r"$T_{-}$", r"$T_{\uparrow}$",
    #  r"$e_P$"]
    for i, col_name in enumerate(df.columns):
        ser = df[col_name]
        ser_high_ = ser_high.apply(
            lambda x: x[i][i]) if ser_high is not None else None
        ser_low_ = ser_low.apply(
            lambda x: x[i][i]) if ser_low is not None else None

        ax = axs[i]
        ax.plot(ser.index, ser, color=(0, 0.5, 0.5),
                linewidth=0.7, label='Signal')
        set_axis_style(ax, ser, ylabel=f"{ser.name} [-]")
        ax.scatter(ser[anomalies == 1].index, ser[anomalies == 1],
                   color=(0.5, 0, 0), s=3, label='System Anomalies')

        if changepoints is not None:
            ax.scatter(ser[samplings == 1].index, ser[samplings == 1],
                       color=(0, 0, 0.5), s=2, label='Sampling Anomalies')

            a = samplings.astype(int).diff()
            for x0, x1 in zip(a[a == 1].index, a[a == -1].index):
                ax.axvspan(x0, x1, color='blue', alpha=1)

        if changepoints is not None:
            c = changepoints.astype(int).diff()
            ax.scatter(ser[c == 1].index, ser[c == 1],
                       color=(1, 1, 0), s=2, label='Changepoint')

            for x0 in c[c == 1].index:
                x1 = c.index[c.index.get_loc(x0) + 1]
                ax.axvspan(x0, x1, color='yellow', alpha=1)

        if ser_high_ is not None and ser_low_ is not None:
            ax.plot(ser_high_.index, ser_high_, color=(
                1, 0, 0, 0.25), linewidth=0.7, label=r'Threshold')
            ax.plot(ser_low_.index, ser_low_, color=(1, 0, 0, 0.25),
                    linewidth=0.7, label=r'Threshold')

            ax.fill_between(ser_high_.index, ser_high_, ser.max(),
                            color=(1, 0, 0, 0.1), alpha=0.1)
            ax.fill_between(ser_low_.index, ser_low_, ser.min(),
                            color=(1, 0, 0, 0.1), alpha=0.1)

        ax.tick_params(axis='both', which='major', labelsize=9)

    ax.set_xticks(b[b > 0].index.map(str))

    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if save:
        plt.savefig(f"plots/{file_name}_thresh.pdf")

    plt.show()
