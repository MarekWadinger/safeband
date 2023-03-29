# IMPORTS
from datetime import timedelta

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# CONSTANTS
FIG_LAYOUT = dict(
    height=90*3,
    width=120*3,
    
    yaxis_title_standoff = 0,
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
        ser_high: pd.Series | None = None, 
        ser_low: pd.Series | None = None, 
        window: timedelta | None = None,
        file_name: str | None = None,
        save: bool = True,
        **kwargs):
    
    if not file_name:
        if window:
            file_name = (f"{ser.name.replace(' ', '_')}_"
             f"{int(window.total_seconds()/60/60)}_hours_sliding")
        else:
            file_name = (f"{ser.name.replace(' ', '_')}_sliding")

    a = anomalies.astype(int).diff()
    # Show dates of anomalous events
    b = a[a == 1].resample('1d').sum()

    fig = go.Figure()

    
    fig.update_layout(FIG_LAYOUT.update(dict(yaxis_title=ser.name,
                                             yaxis_range=[ser.min(), 
                                                          ser.max()],
                                             xaxis_tickangle=60,
                                             xaxis_tickfont_size=9,
                                             xaxis_tickvals=b[b > 0].index)))

    # To show discontinuation of the signal stream
    #d = ser.copy()
    #d.index = d.index.round("1T")
    #d = d.resample('2T').min()

    fig.add_trace(go.Scatter(
        x=ser.index, y=ser,
        connectgaps=False,
        line_color='rgb(0,140,120)',
        name=ser.name, showlegend=True,
        line_width=0.7
    ))

    if save:
        fig.write_image(f"{file_name}_signal.pdf")
    
    if kwargs.keys() == {"ser_mean", "ser_pos", "ser_neg"}:
        fig.add_trace(go.Scatter(
            x=kwargs["ser_pos"].index.append(kwargs["ser_pos"].index[::-1]),
            y= pd.concat([kwargs["ser_pos"], kwargs["ser_neg"][::-1]]),
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

        if save:
            fig.write_image(f"{file_name}_mean.pdf")
        for trace, visibility in zip([-1, -2],
                                    [False, False]):
            fig.data[trace].visible = visibility

    for x0, x1 in zip(a[a == 1].index, a[a == -1].index):
        fig.add_vrect(x0=x0, x1=x1, line_color="red", fillcolor="red", 
                    opacity=0.25, layer="below")
        
    if save:
        fig.write_image(f"{file_name}_anomalies.pdf")

    if (ser_high is not None) and (ser_low is not None):
        fig.add_trace(go.Scatter(
            x=ser.index, y=([1] if ser_high.max(skipna=True) < 1 
                        else [ser_high.max(skipna=True)])*len(ser),
            line_color='rgba(100,100,100, 0)', 
            name='Threshold', legendgroup='thresh', showlegend=False,
            line_width=0.7
        ))

        fig.add_trace(go.Scatter(
            x=ser_high.index, y=ser_high,
            line_color='rgba(100,0,0,0.25)',
            fillcolor='rgba(100,0,0, 0.1)', fill="tonexty",
            name='Threshold', legendgroup='thresh', showlegend=True,
            line_width=0.7
        ))

        fig.add_trace(go.Scatter(
            x=ser_low.index, y=ser_low,
            line_color='rgba(100,0,0,0.25)', 
            fillcolor='rgba(100,0,0, 0.1)', fill="tozeroy",
            name='Threshold', legendgroup='thresh', showlegend=False,
            line_width=0.7
        ))

        if save:
            fig.write_image(f"{file_name}_thresh.pdf")

    if window:
        text = (f"Sliding window: {window}\n"
        f"Proportion of anomalous samples: "
        f"{sum(anomalies)/len(anomalies)*100:.02f}%\n"
        f"Total number of anomalous events: "
        f"{sum(pd.Series(anomalies).diff().dropna() == 1)}")
        text = text.replace('\n', '<br>')
        fig.add_annotation(text=text, align='left',
                        xref="paper", yref="paper",
                        x=0, y=1.2, showarrow=False)

    fig.update_layout(
        height=90*10,
        width=120*10,
    )
    if save:
        fig.write_html(f"{file_name}_all.html")

    fig.show()
    

def plot_limits_3d(
        df: pd.DataFrame,
        anomalies: pd.Series,
        ser_high: pd.Series | None = None, 
        ser_low: pd.Series | None = None,
        y: str| None = None,
        z: str| None = None,
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
            x=df.index, y=df.iloc[:,col1], z=df.iloc[:,col2],
            connectgaps=False,
            line_color='rgb(0,140,120)', 
            name='Trace', 
            legendgroup='trace', showlegend=True,
            line_width=0.7, mode='lines'
        ),row=1,col=1)

    df_anomalies = df[anomalies == 1]
    fig.add_trace(go.Scatter3d(
            x=df_anomalies.index, y=df_anomalies.iloc[:,col1], z=df_anomalies.iloc[:,col2],
            connectgaps=False,
            name='Anomalies', 
            line_color='rgb(150,0,0)',showlegend=True,
            line_width=0.7, mode='markers', marker_size=2
        ),row=1,col=1)

    # SECOND
    fig.add_trace(go.Scatter(
            x=df.index, y=df.iloc[:,col1],
            connectgaps=False,
            line_color='rgb(0,140,120)',
            legendgroup='trace', showlegend=False,
            line_width=0.7
        ),row=4,col=1)

    # THRESHOLD
    thresh_high = ser_high.apply(lambda x: x[col1][col1])
    thresh_low = ser_low.apply(lambda x: x[col1][col1])
    fig.add_trace(go.Scatter(
        x=thresh_high.index, y=([thresh_high.max(skipna=True)])*len(thresh_high),
        line_color='rgba(100,100,100, 0)', 
        name='Threshold', legendgroup='thresh', showlegend=False,
        line_width=0.7
    ),row=4,col=1)

    fig.add_trace(go.Scatter(
        x=thresh_high.index, y=thresh_high,
        line_color='rgba(100,0,0,0.25)',
        fillcolor='rgba(100,0,0, 0.1)', fill="tonexty",
        name='Threshold', legendgroup='thresh', showlegend=False,
        line_width=0.7
    ),row=4,col=1)

    fig.add_trace(go.Scatter(
        x=thresh_low.index, y=thresh_low,
        line_color='rgba(100,0,0,0.25)', 
        fillcolor='rgba(100,0,0, 0.1)', fill="tozeroy",
        name='Threshold', legendgroup='thresh', showlegend=False,
        line_width=0.7
    ),row=4,col=1)
    fig.layout['yaxis'].update(dict(range=[thresh_low.min(), thresh_high.max()]))
                                                
    # THIRD
    fig.add_trace(go.Scatter(
            x=df.index, y=df.iloc[:,col2],
            connectgaps=False,
            line_color='rgb(0,140,120)',
            legendgroup='trace', showlegend=False,
            line_width=0.7
        ),row=4,col=2)

    # THRESHOLD
    thresh_high = ser_high.apply(lambda x: x[col2][col2])
    thresh_low = ser_low.apply(lambda x: x[col2][col2])
    fig.add_trace(go.Scatter(
        x=thresh_high.index, y=([thresh_high.max(skipna=True)])*len(thresh_high),
        line_color='rgba(100,100,100, 0)', 
        name='Threshold', legendgroup='thresh', showlegend=False,
        line_width=0.7
    ),row=4,col=2)

    fig.add_trace(go.Scatter(
        x=thresh_high.index, y=thresh_high,
        line_color='rgba(100,0,0,0.25)',
        fillcolor='rgba(100,0,0, 0.1)', fill="tonexty",
        name='Threshold', legendgroup='thresh', showlegend=False,
        line_width=0.7
    ),row=4,col=2)

    fig.add_trace(go.Scatter(
        x=thresh_low.index, y=thresh_low,
        line_color='rgba(100,0,0,0.25)', 
        fillcolor='rgba(100,0,0, 0.1)', fill="tozeroy",
        name='Threshold', legendgroup='thresh', showlegend=False,
        line_width=0.7
    ),row=4,col=2)
    fig.layout['yaxis2'].update(dict(range=[thresh_low.min(), thresh_high.max()]))


    fig.update_layout(
        height=90*10,
        width=120*10,
    )
    
    if save:
        fig.write_html(f"multi_all.html")
        
    return fig
    

def plot_compare_anomalies(
        ser: pd.Series,
        anomalies: pd.DataFrame,
        window: timedelta | None = None,
        file_name: str | None = None,
        save: bool = True,
        **kwargs):
    
    if not file_name:
        if window:
            file_name = (f"{ser.name.replace(' ', '_')}_"
             f"{int(window.total_seconds()/60/60)}_hours_sliding")
        else:
            file_name = (f"{ser.name.replace(' ', '_')}_sliding")

    n_rows  = len(anomalies.columns)
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
        
        fig.update_yaxes(title_text=f"{ser.name}<br>{anomaly}", title_standoff=0, row=row, col=1)
    
    if save:
        fig.write_image(f"{file_name}_compare_anomalies.pdf")
        
    fig.update_layout(
        height=90*10,
        width=120*10,
    )
    
    if save:
        fig.write_html(f"{file_name}_compare_anomalies.html")
    
    fig.show()
    

if __name__ == '__main__':
    df = pd.read_csv('data/input/average_temperature.csv', index_col=0)
    df.index = pd.to_datetime(df.index)
    col = 'Average Cell Temperature'
    
    df_out = pd.read_json('data/output/dynamic_limits.json', lines=True).set_index('time')
    df_out.index = pd.to_datetime(df_out.index)
    df_out = df_out.iloc[0:len(df)]

    plot_limits(df[col], df_out.level_high, df_out.level_low, df_out.anomaly,
                save=False,)
