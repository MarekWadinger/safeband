# IMPORTS
import textwrap
from datetime import timedelta
from typing import Union

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# CONSTANTS
FIG_LAYOUT = dict(
    height=90*3,
    width=120*3,

    yaxis_title_standoff=0,
    xaxis_tickfont_size=9,

    font_family="Times New Roman",
    font_size=9,

    autosize=True,

    margin=dict(l=40, r=15, t=5, b=0),
    bargap=0,

    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)


def plot_limits(
        ser: pd.Series,
        anomalies: pd.Series,
        ser_high: Union[pd.Series, None] = None,
        ser_low: Union[pd.Series, None] = None,
        window: Union[timedelta, None] = None,
        file_name: Union[str, None] = None,
        save: bool = True,
        **kwargs):

    if not file_name:
        file_name = get_file_name(ser, window)

    a = anomalies.astype(int).diff()
    # Show dates of anomalous events
    b = a[a == 1].resample('1d').sum()

    fig = go.Figure()

    fig.update_layout(
        FIG_LAYOUT.update(dict(
            yaxis_title=ser.name,
            yaxis_range=[ser.min(), ser.max()],
            xaxis_tickangle=60,
            xaxis_tickfont_size=9,
            xaxis_tickvals=b[b > 0].index)))

    plot_add_signal(fig, ser)

    if save:
        fig.write_image(f"{file_name}_signal.pdf")

    if kwargs.keys() == {"ser_mean", "ser_pos", "ser_neg"}:
        plot_add_mean_std(fig, ser, kwargs)

        if save:
            fig.write_image(f"{file_name}_mean.pdf")
        set_invisible(fig, [-1, -2])

    for x0, x1 in zip(a[a == 1].index, a[a == -1].index):
        fig.add_vrect(x0=x0, x1=x1, line_color="red", fillcolor="red",
                      opacity=0.25, layer="below")

    if save:
        fig.write_image(f"{file_name}_anomalies.pdf")

    if (ser_high is not None) and (ser_low is not None):
        yaxis_range = [ser.min(), ser.max()]
        add_thresholds(fig, ser_high, ser_low,
                       row=None, col=None,
                       yaxis_range=yaxis_range)

        if save:
            fig.write_image(f"{file_name}_thresh.pdf")

    fig.update_layout(
        height=90*10,
        width=120*10,
    )
    if save:
        fig.write_html(f"{file_name}_all.html")

    fig.show()


def get_file_name(
        ser,
        window=None
):
    if window is not None:
        file_name = (f"{ser.name.replace(' ', '_')}_"
                     f"{int(window.total_seconds()/60/60)}_hours_sliding")
    else:
        file_name = (f"{ser.name.replace(' ', '_')}_sliding")
    return file_name


def plot_add_signal(
        fig,
        ser
):
    fig.add_trace(go.Scatter(
        x=ser.index, y=ser,
        connectgaps=False,
        line_color='rgb(0,140,120)',
        name=ser.name, showlegend=True,
        line_width=0.7
    ))


def plot_add_mean_std(
        fig,
        ser,
        kwargs
):
    fig.add_trace(go.Scatter(
        x=kwargs["ser_pos"].index.append(kwargs["ser_pos"].index[::-1]),
        y=pd.concat([kwargs["ser_pos"], kwargs["ser_neg"][::-1]]),
        fill='toself',
        fillcolor='rgba(100,0,80,0.2)',
        line_color='rgba(255,255,255,0)',
        showlegend=False,
        name=f'Mov Mean {ser.name}',
        line_width=0.7
    ))

    fig.add_trace(go.Scatter(
        x=kwargs["ser_mean"].index, y=kwargs["ser_mean"],
        line_color='rgb(100,0,80)',
        name=f'Mov Mean {ser.name}',
        line_width=0.7
    ))


def set_invisible(fig, inv_lines: list):
    for trace in inv_lines:
        fig.data[trace].visible = False


def plot_limits_3d(
        df: pd.DataFrame,
        anomalies: pd.Series,
        ser_high: pd.Series,
        ser_low: pd.Series,
        signal_anomalies: Union[pd.Series, None] = None,
        y: Union[str, None] = None,
        z: Union[str, None] = None,
        file_name: Union[str, None] = None,
        save: bool = False,
        **kwargs):
    from plotly.subplots import make_subplots

    col1 = df.columns.get_loc(y)
    col2 = df.columns.get_loc(z)
    fig = make_subplots(
        rows=4, cols=2,
        specs=[[{"colspan": 2, "rowspan": 3, "type": "scatter3d"}, None],
               [None, None],
               [None, None],
               [{}, {}],
               ],
        subplot_titles=("3D view of anomalous operation", y, z))

    fig.add_trace(go.Scatter3d(
        x=df.index, y=df.iloc[:, col1], z=df.iloc[:, col2],
        connectgaps=False,
        line_color='rgb(0,140,120)',
        name='Trace',
        legendgroup='trace', showlegend=True,
        line_width=0.7, mode='lines'
    ), row=1, col=1)

    df_anomalies = df[anomalies == 1]
    fig.add_trace(go.Scatter3d(
        x=df_anomalies.index,
        y=df_anomalies.iloc[:, col1],
        z=df_anomalies.iloc[:, col2],
        connectgaps=False,
        name='Anomalies',
        line_color='rgb(150,0,0)', showlegend=True,
        line_width=0.7, mode='markers', marker_size=2
    ), row=1, col=1)

    # SECOND
    fig.add_trace(go.Scatter(
        x=df.index, y=df.iloc[:, col1],
        connectgaps=False,
        line_color='rgb(0,140,120)',
        legendgroup='trace', showlegend=False,
        line_width=0.7
    ), row=4, col=1)

    yaxis_range = [df.iloc[:, col2].min(), df.iloc[:, col2].max()]

    # THRESHOLD
    thresh_high = ser_high.apply(lambda x: x[col1][col1])
    thresh_low = ser_low.apply(lambda x: x[col1][col1])
    add_thresholds(fig, thresh_high, thresh_low,
                   row=4, col=2,
                   yaxis_range=yaxis_range)
    # THIRD
    fig.add_trace(go.Scatter(
        x=df.index, y=df.iloc[:, col2],
        connectgaps=False,
        line_color='rgb(0,140,120)',
        legendgroup='trace', showlegend=False,
        line_width=0.7
    ), row=4, col=2)

    yaxis_range = [df.iloc[:, col2].min(), df.iloc[:, col2].max()]

    # THRESHOLD
    thresh_high = ser_high.apply(lambda x: x[col2][col2])
    thresh_low = ser_low.apply(lambda x: x[col2][col2])
    add_thresholds(fig, thresh_high, thresh_low,
                   row=4, col=2,
                   yaxis_range=yaxis_range)

    fig.update_layout(
        height=90*10,
        width=120*10,
    )

    if save:
        fig.write_html("multi_all.html")

    return fig


def add_thresholds(
        fig,
        thresh_high,
        thresh_low,
        row: Union[int, None],
        col: Union[int, None],
        yaxis_range: list[int]):
    fig.add_trace(go.Scatter(
        x=thresh_high.index, y=(
            [max(yaxis_range)])*len(thresh_high),
        line_color='rgba(100,100,100, 0)',
        name='Threshold', legendgroup='thresh', showlegend=False,
        line_width=0.7
    ), row=row, col=col)

    fig.add_trace(go.Scatter(
        x=thresh_high.index, y=thresh_high,
        line_color='rgba(100,0,0,0.25)',
        fillcolor='rgba(100,0,0, 0.1)', fill="tonexty",
        name='Threshold', legendgroup='thresh', showlegend=False,
        line_width=0.7
    ), row=row, col=col)

    fig.add_trace(go.Scatter(
        x=thresh_low.index, y=(
            [min(yaxis_range)])*len(thresh_low),
        line_color='rgba(100,100,100, 0)',
        name='Threshold', legendgroup='thresh', showlegend=False,
        line_width=0.7
    ), row=row, col=col)

    fig.add_trace(go.Scatter(
        x=thresh_low.index, y=thresh_low,
        line_color='rgba(100,0,0,0.25)',
        fillcolor='rgba(100,0,0, 0.1)', fill="tonexty",
        name='Threshold', legendgroup='thresh', showlegend=False,
        line_width=0.7
    ), row=row, col=col)


def plot_compare_anomalies(
        ser: pd.Series,
        anomalies: pd.DataFrame,
        window: Union[timedelta, None] = None,
        file_name: Union[str, None] = None,
        save: bool = True,
        **kwargs):

    if not file_name:
        file_name = get_file_name(ser, window)

    n_rows = len(anomalies.columns)
    fig = make_subplots(rows=n_rows, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.05)

    fig.update_layout(FIG_LAYOUT.update(dict(yaxis_title=ser.name,
                                             yaxis_range=[ser.min(),
                                                          ser.max()],
                                             showlegend=True)))

    for row, anomaly in enumerate(anomalies, start=1):
        a = anomalies[anomaly].astype(int).diff()

        fig.add_trace(go.Scatter(
            x=ser.index, y=ser,
            connectgaps=False,
            line_color='rgb(0,140,120)',
            name=ser.name,
            line_width=0.7,
            text=anomaly,
            textposition="bottom center",
            legendgroup='thresh', showlegend=False,

        ), row=row, col=1)

        for x0, x1 in zip(a[a == 1].index, a[a == -1].index):
            fig.add_vrect(x0=x0, x1=x1, line_color="red", fillcolor="red",
                          opacity=0.25, layer="below", row=row, col=1)

        fig.update_yaxes(
            title_text=f"{ser.name}<br>{anomaly}", title_standoff=0,
            row=row, col=1)

    if save:
        fig.write_image(f"plot/{file_name}_compare_anomalies.pdf")

    fig.update_layout(
        height=90*10,
        width=120*10,
    )

    if save:
        fig.write_html(f"plot/{file_name}_compare_anomalies.html")

    fig.show()


def plot_limits_grid(
        df: pd.DataFrame,
        anomalies: pd.Series,
        ser_high: pd.Series,
        ser_low: pd.Series,
        file_name: Union[str, None] = None,
        save: bool = True,
        changepoints: Union[pd.Series, None] = None,
        samplings: Union[pd.Series, None] = None,
        **kwargs):

    a = anomalies.astype(int).diff()
    # Show dates of anomalous events
    b = a[a == 1].resample('1d').sum()

    n_rows = len(df.columns)
    fig = make_subplots(rows=n_rows, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.05)
    fig.update_layout(FIG_LAYOUT.update(
            dict(height=180*3,
                 width=60*3,)))

    for row, col_name in enumerate(df.columns, start=1):

        ser = df[col_name]
        ser_high_ = ser_high.apply(lambda x: x[col_name])
        ser_low_ = ser_low.apply(lambda x: x[col_name])

        text = "<br>".join(textwrap.wrap(f"{ser.name} [-]", 15))
        fig.update_yaxes(
            title_text=text, title_font_size=9, title_standoff=15,
            row=row, col=1)

        fig.add_trace(go.Scatter(
            x=ser.index, y=ser,
            connectgaps=False,
            line_color='rgb(0,140,120)',
            name="Signal", showlegend=True if row == 1 else False,
            line_width=0.7
        ), row=row, col=1)

        fig.add_trace(go.Scatter(
            x=ser[anomalies == 1].index,
            y=ser[anomalies == 1],
            connectgaps=False,
            name='System Anomalies',
            line_color='rgb(150,0,0)', showlegend=True if row == 1 else False,
            line_width=0.7, mode='markers', marker_size=3
        ), row=row, col=1)

        if isinstance(changepoints, pd.Series):
            fig.add_trace(go.Scatter(
                x=ser[samplings == 1].index,
                y=ser[samplings == 1],
                connectgaps=False,
                name='Sampling Anomalies',
                line_color='rgb(0,0,150)',
                showlegend=True if row == 1 else False,
                line_width=0.7, mode='markers', marker_size=2
            ), row=row, col=1)

            if samplings is not None:
                a = samplings.astype(int).diff()
                for x0, x1 in zip(a[a == 1].index, a[a == -1].index):
                    fig.add_vrect(
                        x0=x0, x1=x1,
                        line_color="blue", fillcolor="blue", opacity=0.25,
                        layer="below", row=row, col=1)  # type: ignore

        if isinstance(changepoints, pd.Series):
            c = changepoints.astype(int).diff()
            fig.add_trace(go.Scatter(
                x=ser[c == 1].index,
                y=ser[c == 1],
                connectgaps=False,
                name='Changepoint',
                line_color='rgb(255,255,0)',
                showlegend=True if row == 1 else False,
                line_width=0.7, mode='markers', marker_size=2
            ), row=row, col=1)

            for x0 in c[c == 1].index:
                x1 = c.index[c.index.get_loc(x0)+1]
                fig.add_vrect(x0=x0, x1=x1,
                              line_color="yellow", fillcolor="yellow",
                              opacity=0.75, layer="below", row=row, col=1)

        if (ser_high_ is not None) and (ser_low_ is not None):
            yaxis_range = [ser.min(), ser.max()]
            add_thresholds(fig, ser_high_, ser_low_,
                           row=row, col=1,
                           yaxis_range=yaxis_range)

        fig.update_yaxes(tickfont_size=9, row=row, col=1)

    fig.update_xaxes(tickangle=60, tickfont_size=9, tickvals=b[b > 0].index,
                     row=row, col=1)

    fig.update_layout(FIG_LAYOUT.update(
            dict(height=180*3,
                 width=60*3,
                 yaxis_title=ser.name,
                 yaxis_range=[ser.min(),
                              ser.max()],
                 xaxis_tickangle=60,
                 xaxis_tickfont_size=9,
                 xaxis_tickvals=b[b > 0].index,
                 )))

    if save:
        FIG_LAYOUT_SAVE = dict(
            height=160*3,
            width=90*3,

            yaxis_title_standoff=0,
            xaxis_tickfont_size=9,

            font_family="Times New Roman",
            font_size=9,

            autosize=False,

            margin=dict(l=0, r=0, t=0, b=0),
            bargap=0,

            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font_size=9
            )
        )
        fig.update_layout(FIG_LAYOUT_SAVE)
        fig.write_image(f"plots/{file_name}_thresh.pdf")

    fig.update_layout(
        height=90*10,
        width=120*10,
    )
    if save:
        fig.write_html(f"plots/{file_name}_all.html")

    fig.show()


if __name__ == '__main__':
    df = pd.read_csv('data/input/average_temperature.csv', index_col=0)
    df.index = pd.to_datetime(df.index)
    col = 'Average Cell Temperature'

    df_out = pd.read_json('data/output/dynamic_limits.json',
                          lines=True).set_index('time').dropna()
    df_out.index = pd.to_datetime(df_out.index)
    df_out = df_out.iloc[0:len(df)]

    df = df.tz_localize(None).loc[df_out.index]
    plot_limits(df[col], df_out.level_high, df_out.level_low, df_out.anomaly,
                save=False,)
