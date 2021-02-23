import dash 
import dash_core_components as dcc 
import dash_html_components as html 
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.express as px 
import pandas as pd 
import heartpy as hp
import numpy as np
import plotly.graph_objects as go
import time
import pipeline

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

colors = {
    'background': '#fff',
    'text': '#7FDBFF'
}
 
df_g1 = pd.read_csv("file:///Users/ashwyn/sleepy_gamer_01_all_features_unscrambled_g1_no_outliers.csv")
df_g2 = pd.read_csv("file:///Users/ashwyn/sleepy_gamer_01_all_features_unscrambled_g2_no_outliers.csv")
df_g3 = pd.read_csv("file:///Users/ashwyn/sleepy_gamer_01_all_features_unscrambled_g3.csv")
df_g4 = pd.read_csv("file:///Users/ashwyn/sleepy_gamer_01_all_features_unscrambled_g4_no_outliers.csv")
df_g5 = pd.read_csv("file:///Users/ashwyn/sleepy_gamer_01_all_features_unscrambled_g5.csv")
df_g1 = df_g1.drop(columns=['Unnamed: 0'])
df_g2 = df_g2.drop(columns=['Unnamed: 0'])
df_g3 = df_g3.drop(columns=['Unnamed: 0'])
df_g4 = df_g4.drop(columns=['Unnamed: 0'])
df_g5 = df_g5.drop(columns=['Unnamed: 0'])
dfg = [df_g1, df_g2, df_g3, df_g4, df_g5]

df_ga1 = pd.read_csv("file:///Users/ashwyn/Desktop/PPG/gamer1-annotations.csv")
df_ga2 = pd.read_csv("file:///Users/ashwyn/Desktop/PPG/gamer2-annotations.csv")
df_ga3 = pd.read_csv("file:///Users/ashwyn/Desktop/PPG/gamer3-annotations.csv")
df_ga4 = pd.read_csv("file:///Users/ashwyn/Desktop/PPG/gamer4-annotations.csv")
df_ga5 = pd.read_csv("file:///Users/ashwyn/Desktop/PPG/gamer5-annotations.csv")
dfga = [df_ga1, df_ga2, df_ga3, df_ga4, df_ga5]

def generate_card(className, header, title):
    return  html.Div(
                className=className,
                children=[
                    html.Div(
                        className="card-header",
                        children=[header],
                    ),
                    html.Div(
                        className="card-body",
                        children=[
                            html.H4(
                                className="card-title",
                                children=[title]
                            )
                        ]
                    )
                ]
            )
      
def generate_table(dataframe, max_rows=10):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min (len(dataframe), max_rows))
        ])
    ])

def bpm_card_content():
    return [
    dbc.CardHeader("Heart Rate"),
    dbc.CardBody(
        [
            html.H3("78", className="card-title"),
        ]
    )
]

def generate_dropdown(id, options):
    return dcc.Dropdown(
        id=id,
        options=[options]
    )
    
def generate_fatigue_input():
    return html.Div(
            [
                html.P("Sleepiness Level Range 0-7"),
                dbc.Input(type="number", min=0, max=7, step=1),
            ],
            id="styled-numeric-input",
            )

def generate_activity_input():
    return dbc.FormGroup(
    [
        dbc.Label("Current Activity"),
        dbc.Input(placeholder="Playing Video games...", type="Text"),
    ]
)

def generate_fatigue_bar():
    return dbc.Progress(
    [
        dbc.Progress(value=60, color="success", bar=True),
        dbc.Progress(value=40, color="danger", bar=True),
    ],
    multi=True,
) 

def bpm_card():
    return dbc.Col(dbc.Card(bpm_card_content(), color="danger", inverse=True), style={'padding': '5%'}, width=6)

def tab_1_content():
    return html.Div([
        dbc.Row([
            dbc.Col([
                generate_card("card text-white bg-danger mb-3", "BPM", "78"), 
                generate_card("card bg-warning mb-3", "HRV", pipeline.acc),
                generate_card("card text-white bg-info mb-3", "Breathing Rate", "120"),
            ], 
            style={'padding': '5%'}, width=6),
            dbc.Col(dcc.Graph(id='5'), width=6)
        ]),
        dbc.Row([
            dbc.Col(generate_dropdown('testd1', {'label': 'test', 'value': 'test'}), style={'padding': '5%'}, width=6),
            dbc.Col(generate_dropdown('testd2', {'label': 'test', 'value': 'test'}), style={'padding': '5%'}, width=6)
        ]),
        dbc.Row([
        ])
    ])

def tab_2_content():
    return html.Div(style={'backgroundColor': colors['background']},children=[
                html.Div([
                    html.Div(style={'backgroundColor': colors['background']},
                        id="app-container",
                        children=[
                            html.Div([
                                html.Div([
                                    dbc.Row([
                                        dbc.Col(generate_card("card border-primary mb-3", "text", "text")),
                                        dbc.Col(generate_card("card border-primary mb-3", "text", "text")),
                                        dbc.Col(generate_card("card border-primary mb-3", "text", "text")),
                                    ])
                                ], style={'width': '98%', 'padding': '15px', 'padding-left': '3%', 'padding-right': '3%'}),
                                html.Div([
                                    dcc.Dropdown(
                                        id='yaxis-column',
                                        options=[
                                            {'label': 'Beats Per Minute', 'value': 'bpm'},
                                            {'label': 'Heart Rate Variability', 'value': 'hrv'},
                                            {'label': 'Breathing Rate', 'value': 'br'},
                                            {'label': 'Interbeat Interval', 'value': 'ibi'},
                                            {'label': 'Standard Deviation of RR Intervals', 'value': 'sdnn'},
                                            {'label': 'Standard Deviation of Successive Differences', 'value': 'sdsd'},
                                            {'label': 'Proportion of Succesive Differences above 20ms', 'value': 'pnn20'},
                                            {'label': 'Proportion of Successive Differences above 50ms', 'value': 'pnn50'},
                                            {'label': 'Poincare Analysis', 'value': 'sd1 sd2 s sd1/sd2'},
                                            {'label': 'Low Frequency Component', 'value': 'lf'},
                                            {'label': 'High Frequency Component', 'value': 'hf'},
                                            {'label': 'Low/High Frequency Ratio', 'value': 'lf/hf'},
                                            {'label': 'Level of Sleepiness', 'value': 'level'},
                                            {'label': 'Is Fatigued', 'value': 'fatigued'}
                                        ],
                                        value='bpm'
                                    ),
                                    
                                ],  style={'width': '49%', 'display': 'inline-block', 'flaot': 'right', 'padding': '5px', 'padding-right': '3%'}),
                            html.Div([
                                    dcc.Dropdown(
                                        id='gamer',
                                        options=[
                                            {'label': 'Gamer 1', 'value': 0},
                                            {'label': 'Gamer 2', 'value': 1},
                                            {'label': 'Gamer 3', 'value': 2},
                                            {'label': 'Gamer 4', 'value': 3},
                                            {'label': 'Gamer 5', 'value': 4}
                                        ],
                                        value=0
                                    )
                            ],
                            style={'width': '49%','display': 'inline-block', 'float': 'left', 'padding': '5px', 'padding-left': '3%'}),
                            ], style={
                                'width': '100%',
                                "display":"inline-block",
                                "position":"relative",
                                'padding': '10px 5px'
                            }),
                            html.Div([
                                dcc.Graph(
                                    id='5'
                                ),
                                dcc.Graph(
                                    id='4'
                                ),
                                ],
                                style = {"width": "49%", "display":"inline-block","position":"relative"}),
                            html.Div([
                                dcc.Graph(
                                    id='6'
                                ),
                                html.Div([
                                dcc.Slider(
                                    id='4s',
                                    updatemode='drag',
                                    vertical=False,
                                    min = 1,
                                    max = 7,
                                    value = 7,
                                    marks = {str(level) : str(level) for level in df_g3['level'].unique()},
                                    step = None
                                )
                                ], style={'width': '98%', 'display': 'inline-block',  'float': 'right', 'padding': '5px', 'padding-left': '3%'}),
                                ], 
                                style = {"width": "49%", "float": "left" , "display":"inline-block","position":"relative"}),
                    ]),
                ], 
                ),
            ])

app.layout = html.Div([
    dcc.Tabs(id="tabs", value="tab-1", children=[
        dcc.Tab(label='Tab one', value='tab-1'),
        dcc.Tab(label='Tab two', value='tab-2'),
    ]),
    html.Div(id='tabs-content')
])

@app.callback(Output('tabs-content', 'children'),
            Input('tabs', 'value'))

def render_content(tab):
    if tab == 'tab-1':
        return tab_1_content()
  
    elif tab == 'tab-2':
        return html.Div([
            tab_2_content()
        ])

@app.callback(
    Output('4', 'figure'),
    Output('5', 'figure'),
    Output('6', 'figure'),
    Input('yaxis-column', 'value'),
    Input('gamer', 'value'),
    Input('4s', 'value')
)

def update_figure2(yaxis_column_name, df, selected_level):
    
    if selected_level is None:
        raise dash.exceptions.PreventUpdate

    filtered_df2 = dfg[df][dfg[df].level <= selected_level]

    fig7 = px.line(filtered_df2, y=yaxis_column_name, color_discrete_map = {"level": "red"})
    fig7.update_layout(transition_duration=500)
    fig7.update_yaxes(title=yaxis_column_name)

    fig8 = px.scatter(filtered_df2, y=yaxis_column_name, color=yaxis_column_name)
    fig8.update_yaxes(title=yaxis_column_name)

    a = dfga[df].values
    count_sleep = 0
    count_reaction_time = 0
    count_diary_entry = 0

    for i in range(0, len(dfga[df])):
        if a[:,1][i] == 'Stanford Sleepiness Self-Assessment (1-7)':
            count_sleep = count_sleep + 1
    
        if a[:,1][i] == 'Sleep-2-Peak Reaction Time (ms)':
            count_reaction_time = count_reaction_time + 1
    
        if a[:,1][i] == 'Diary Entry (text)':
            count_diary_entry = count_diary_entry + 1
    
    stanford_sleep_levels = a[:][0:count_sleep] # 1 to 7 classes
    sleep_2_peak_reaction_time = a[:][count_sleep:count_sleep + count_reaction_time] 
    diary_entry = a[:][count_sleep + count_reaction_time: count_diary_entry + count_sleep + count_reaction_time]

    x = []
    y = []
    for k in range(0, len(diary_entry)):
        for i in range(0, len(stanford_sleep_levels)):
            if(diary_entry[k][0].split(':')[0] == stanford_sleep_levels[i][0].split(':')[0]):
                x.append(diary_entry[k][2])
                y.append(stanford_sleep_levels[i][2])

    fig9 = px.scatter(y=x, x=y, color=y)
            
    return fig7, fig8, fig9

if __name__ == '__main__':
    app.run_server(debug=True)
