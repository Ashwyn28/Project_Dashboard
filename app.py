import dash 
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import plotly.express as px 
import pandas as pd 
import heartpy as hp
import numpy as np
import plotly.graph_objects as go
import time
import pipeline
import base64
from ml_models import dt
from ml_models import adaboost
from ml_models import ann
from ml_models import svm
from data_processing.preprocessing import scale
import drawConfusionMatrix
#import drawDecisionTree
#import drawAdaboosts
#from live_data import processing
import predict
from decimal import Decimal

# Launches dash using boostrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Global background and color setting
colors = {
    'background': '#fff',
    'text': '#7FDBFF'
}


################################################################################################################

# STATIC DATA

################################################################################################################


# Static data for dataset plots

df_g1 = pd.read_csv("./data/data_g1_live.csv")
df_g2 = pd.read_csv("./data/data_g2_live.csv")
df_g3 = pd.read_csv("./data/data_g3_live.csv")
df_g4 = pd.read_csv("./data/data_g4_live.csv")
df_g5 = pd.read_csv("./data/data_g5_live.csv")
df_g1 = df_g1.drop(columns=['Unnamed: 0'])
df_g2 = df_g2.drop(columns=['Unnamed: 0'])
df_g3 = df_g3.drop(columns=['Unnamed: 0'])
df_g4 = df_g4.drop(columns=['Unnamed: 0'])
df_g5 = df_g5.drop(columns=['Unnamed: 0'])
dfg = [df_g1, df_g2, df_g3, df_g4, df_g5]

df_ga1 = pd.read_csv("./data/gamer1-annotations.csv")
df_ga2 = pd.read_csv("./data/gamer2-annotations.csv")
df_ga3 = pd.read_csv("./data/gamer3-annotations.csv")
df_ga4 = pd.read_csv("./data/gamer4-annotations.csv")
df_ga5 = pd.read_csv("./data/gamer5-annotations.csv")
dfga = [df_ga1, df_ga2, df_ga3, df_ga4, df_ga5]

df = pd.read_csv("pipeline_data_not_downsampled.csv")
df = df.drop(columns=['Unnamed: 0', 'level'])


# Static data to simulate live plots

df_live = pd.read_csv("./data/data_g5_live.csv")
df_live = df_live.drop(columns=['Unnamed: 0'])


# Global array to store updated labels

true_fatigued = []
true_fatigued_time = []


#image_filename = 'plot.png'
#encoded_image = base64.b64encode(open(image_filename, 'rb').read())

################################################################################################################

# HELPER FUNCTIONS
# used to create html components

################################################################################################################


# Get breathing rate
def get_br(index):
    br = pipeline.df['br'][index]
    return br

# Get heart rate variability
def get_hrv(index):
    hrv = pipeline.df['hrv'][index]
    return hrv

# Get heart rate per minute
def get_bpm(index):
    bpm = df_live['bpm'][index]
    return bpm

# Alert banner
def generate_alert(text, color):
    return html.P(dbc.Alert(text, color=color, dismissable=True, is_open=True), className="mt-10")

# Sleep image
def sleepy(src):
    return html.Img(src=src, width="50%", height="50%", id="image")

# Half awake image
def ok():
    return html.Img(src='https://cdn4.iconfinder.com/data/icons/emo-face/100/smile-512.png', width="50%", height="50%", id="image")

# Awake image
def awake():
    return html.Img(src='https://cdn1.iconfinder.com/data/icons/unigrid-bluetone-emoji/60/040_002_smile_face_happy_emoji-512.png', width="50%", height="50%")

# Card component
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
                        )]
        )])

# Card content for heart rate per minute
def bpm_card_content():
    return [
    dbc.CardHeader("Heart Rate"),
        dbc.CardBody([html.H3("78", className="card-title")])
    ]

# Dropdown component
def generate_dropdown(id, options):
    return dcc.Dropdown(
        id=id,
        options=[options]
    )
    
# Input for sleepiness
def generate_fatigue_input():
    return html.Div(
        [
            html.P("Sleepiness Level Range 0-7"),
            dbc.Input(type="number", min=0, max=7, step=1),
        ],
        id="styled-numeric-input",
    )

# Input for activity
def generate_activity_input():
    return dbc.FormGroup(
    [
        dbc.Label("Current Activity"),
        dbc.Input(placeholder="Playing Video games...", type="Text"),
    ]
)

# Card component for heart rate per minute
def bpm_card():
    return dbc.Col(dbc.Card(bpm_card_content(), color="danger", inverse=True), style={'padding': '5%'}, width=6)

# Jumbotron component
color = ''
def generate_jumbotron(state, img):
    return dbc.Jumbotron(
        [
            dbc.Row([
                dbc.Col(html.H1(children=[state], className="display-3", id="state"), width=7),
                dbc.Col(html.Div(img), width=5),
                dbc.Col(width=6)
            ]),
            html.Hr(className = "my-2"),
            dbc.Row([
                dbc.Col([
                    html.H4("Was I Right?"),
                    html.Hr(className = "my-2"),
                    dbc.Row([
                        dbc.Col([
                            dcc.Dropdown(
                                id='isFatigued',
                                options=[
                                    {'label': 'fatigued', 'value': 1},
                                    {'label': 'not fatigued', 'value': 0}
                                ],
                                value=0
                            ),
                            dcc.Interval(
                            id='interval-component-2',
                            interval=1*1000*60*30,
                            n_intervals=0
                            )
                        ]),
                    ], align="start", style={"padding": "5%"}),
                    html.H4("Update your labels"),
                    html.Hr(className = "my-2"),
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                dbc.Input(id="input_activity", placeholder="activity", type="text", bs_size="md"),
                                html.Br(),
                                html.P(id="output_activity")]
                            )
                        ]),
                        dbc.Col([
                            html.Div([
                                dbc.Input(id="input_s2p", placeholder="Sleep-2-Peak", type="text", bs_size="md"),
                                html.Br(),
                                html.P(id="output_s2p")]
                            )
                        ]),
                        dbc.Col([
                            html.Div([
                                dbc.Input(id="input_sss", placeholder="Sleepiness level", type="text", bs_size="md"),
                                html.Br(),
                                html.P(id="output")]
                            )
                        ])
                    ], align="start", style={"padding": "5%"}),
    ], width=6, style={'padding': '2%', 'paddingLeft': '2%'}),
                dbc.Col([
                    html.H4("Suggestions..."),
                    html.Hr(className = "my-2"),
                    generate_alert(html.Div(children=[], id="alert_1"), "primary"),
                    generate_alert(html.Div(children=[], id="alert_2"), "danger"),
                    generate_alert(html.Div(children=[], id="alert_3"), "success"),
                ], width=6, style={'padding': '2%', 'paddingRight': '2%'})
            ])
        ]
    )


################################################################################################################

# BUILDING HTML COMPONENETS

################################################################################################################


src = ''
state = ''
index = 0
update_live_graph_list = []

# Simple view

def tab_1_content():
    return html.Div([
        dbc.Row([
            dbc.Col(generate_jumbotron(state, sleepy(src)), width=12, style={'padding': '5%'})
        ]),
        dbc.Row([
            dbc.Col(
                dcc.Graph(id = 'timeline'), 
                width = 6,
            ),
            dbc.Col(dcc.Graph(id='7'), width=6),
            dcc.Interval(
                id='interval-component',
                interval=1*1000*60,
                n_intervals=0
            )
        ]),
        html.Div(children=[], id="fp")
    ])

# Detailed view

def tab_2_content():
    return html.Div(style={'backgroundColor': colors['background']},children=[
                html.Div([
                    html.Div(style={'backgroundColor': colors['background']}, id="app-container", children=[
                        html.Div([
                            html.Div([
                                dbc.Col([ html.H4("Dataset"), html.Hr(className = "my-2"),]),
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
                                        {'label': 'Low Frequency Component', 'value': 'lf'},
                                        {'label': 'High Frequency Component', 'value': 'hf'},
                                        {'label': 'Low/High Frequency Ratio', 'value': 'lf/hf'},
                                    ],
                                    value='bpm'
                                )],  
                                style={'width': '49%', 'display': 'inline-block', 'flaot': 'right', 'padding': '5px', 'padding-left': "2%" }),
                        html.Div([
                            dcc.Dropdown(
                                id='gamer',
                                options=[
                                    {'label': 'Gamer 1', 'value': 0},
                                    {'label': 'Gamer 2', 'value': 1},
                                    {'label': 'Gamer 3', 'value': 2},
                                    {'label': 'Gamer 4', 'value': 3},
                                    {'label': 'Gamer 5', 'value': 4}
                                ],value=4)
                            ], style={'width': '49%','display': 'inline-block', 'float': 'left', 'padding': '5px', 'padding-left': '2%'}),
                        ], style={'width': '98%', 'padding': '15px', 'padding-left': '3%', 'padding-right': '3%'}),
                        html.Div([
                            html.Div([ 
                                dbc.Col([ html.H4("Dataset plots"), html.Hr(className = "my-2")]),
                            ], style={'width': '97%', 'padding': '15px', 'padding-left': '3%', 'padding-right': '3%'})],
                        style={'width': '100%',"display":"inline-block","position":"relative",'padding': '10px 5px'}),
                        dbc.Row([
                            dbc.Col(dcc.Graph(
                                id = '6'
                            ), width=6),
                            dbc.Col(dcc.Graph(
                                id = '4',
                            ), width=6),],
                        style = {"width": "98%"}),
                            html.Div([
                                dbc.Col([ html.H4("Model Details"), html.Hr(className = "my-2"),]),
                                dbc.Row([
                                    dbc.Col(generate_card("card border-primary mb-3", "Accuracy", html.Div(children=[], id='accuracy2')), style={'width': '49%','display': 'inline-block', 'float': 'left', 'padding': '5px', 'padding-left': '2%'}),
                                    dbc.Col(generate_card("card border-primary mb-3", "Model", html.Div(children=[], id='model_name')), style={'width': '49%', 'display': 'inline-block', 'flaot': 'right', 'padding': '5px', 'padding-right': "2%" })
                                ])
                            ], style={'width': '98%', 'padding': '15px', 'padding-left': '3%', 'padding-right': '3%'}),
                            html.Div([
                                dbc.Col([ html.H4("Model Parameters"), html.Hr(className = "my-2"),]),
                                html.Div([
                                dbc.Row([
                                    dbc.Col(
                                    dcc.Dropdown(
                                        id='model_type',
                                        options=[
                                            {'label': 'Decision Tree', 'value': 'dt'},
                                            {'label': 'Ada Boost Tree', 'value': 'adaboost'},
                                            {'label': 'Artificial Neural Network', 'value': 'ann'},
                                            {'label': 'Support Vector Machine', 'value': 'svm'}
                                        ],
                                        value='dt'
                                    ),width = 5),
                                    dbc.Col(
                                        html.Div([
                                        dcc.Dropdown(
                                            id='gamer2',
                                            options=[
                                                {'label': 'Gamer 1', 'value': 0},
                                                {'label': 'Gamer 2', 'value': 1},
                                                {'label': 'Gamer 3', 'value': 2},
                                                {'label': 'Gamer 4', 'value': 3},
                                                {'label': 'Gamer 5', 'value': 4}
                                            ],
                                        value=4)]),width=5),
                                    dbc.Col(dbc.Button('Process', id='process'))    
                                ]),                            
                                ],  style={'width': '59%', 'display': 'inline-block', 'flaot': 'right', 'padding': '5px', 'padding-left': "2%" }),
                            ], style={'width': '98%', 'padding': '15px', 'padding-left': '3%', 'padding-right': '3%'}),
                            html.Div([
                                html.Div([
                                    dbc.Col([ html.H4("Model plots"), html.Hr(className = "my-2"),]),
                                ], style={'width': '97%', 'padding': '15px', 'padding-left': '3%', 'padding-right': '3%'}),],
                            style={ 'width': '100%', "display":"inline-block", "position":"relative", 'padding': '10px 5px' }),
                            dbc.Row([
                                dbc.Col(dcc.Graph(
                                    id = 'cm',
                                ), width = 12)
                            ], style = {"width": "98%" ,"position":"relative"}),
                        html.Div(children=[], id='hidden_test', style={'width': '98%', 'display': 'inline-block',  'float': 'right', 'padding': '5px', 'padding-right': '3%', 'display': 'none'}),
                        ], style={'width': '100%', "display":"inline-block", "position":"relative", 'padding': '10px 5px'}),
                ])
            ] 
        )
    ])

################################################################################################################

#   LOADING VIEW

################################################################################################################

app.layout = html.Div([
    tab_1_content(),
    tab_2_content()
])


################################################################################################################

#   CALLBACKS
#   takes inputs from html components using an id
#   outputs are returned to chosen id\s

################################################################################################################


# Updating dataset plots

@app.callback(
    Output('4', 'figure'),
    Output('6', 'figure'),
    Input('yaxis-column', 'value'),
    Input('gamer', 'value'),
)

def update_figure2(yaxis_column_name, df):

    chosen_df = dfg[df]

    fig8 = px.scatter(chosen_df, y=yaxis_column_name, color=yaxis_column_name)
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
    z = []
    for k in range(0, len(diary_entry)):
        for i in range(0, len(stanford_sleep_levels)):
            if(diary_entry[k][0].split(':')[0] == stanford_sleep_levels[i][0].split(':')[0]):
                x.append(diary_entry[k][2])
                z.append(diary_entry[k][0])
                y.append(stanford_sleep_levels[i][2])

    fig9 = px.scatter(y=x, x=z, color=y, labels={'x': "Timeline", 'y': "Activities"})
            
    return fig8, fig9

# Switching models

@app.callback(
    Output('isFatigued', 'value'),
    Input('interval-component-2', 'n_intervals'),
    Input('model_type', 'value')
)

def update_prediction(n, value):    

    if value == 'dt':
        X_test_scaled, X_train_scaled, model, y_pred, acc, model_name, y_test, y_train, X_test, X_train = pipeline.model(dt, df)
    elif value == 'adaboost':
        X_test_scaled, X_train_scaled, model, y_pred, acc, model_name, y_test, y_train, X_test, X_train = pipeline.model(adaboost, df)
    elif value == 'ann':
        X_test_scaled, X_train_scaled, model, y_pred, acc, model_name, y_test, y_train, X_test, X_train = pipeline.model(ann, df)
    elif value == 'svm':
        X_test_scaled, X_train_scaled, model, y_pred, acc, model_name, y_test, y_train, X_test, X_train = pipeline.model(svm, df)

    pred = predict.make_prediction(n, model)

    return pred

# Updating main jumbotron image

@app.callback(
    Output('image', 'src'),
    Output('state', 'children'),
    Output('timeline', 'figure'),
    Input('isFatigued', 'value')
)

def update_image(isFatigued_value):
    
  
    if(isFatigued_value == 1):
        src = 'https://cdn1.iconfinder.com/data/icons/emoticon-solid/48/emoticon_emoji._emotion_sleep-512.png'
        text = "Fatigued"
    else:
        src = 'https://cdn1.iconfinder.com/data/icons/unigrid-bluetone-emoji/60/040_002_smile_face_happy_emoji-512.png'
        text = "Not Fatigued"
    
    count = len(true_fatigued)

    true_fatigued.append(isFatigued_value)
    true_fatigued_time.append(time.localtime().tm_hour)

    updated_labels = [true_fatigued, true_fatigued_time]

    d = {'tf': true_fatigued, 'time': true_fatigued_time}
    df = pd.DataFrame(data=d)

    fig1 = px.strip(df, x="time", y="tf", color="tf", orientation='h', labels={"tf":"Fatigued/Not Fatigued"})

    return src, text, fig1

# Updating heart rate graph

@app.callback(
    Output('7', 'figure'),
    Input('interval-component', 'n_intervals')
)

def update_live_graph(n):

    bpm = get_bpm(n)

    update_live_graph_list.append(bpm)
    fig10 = px.line(update_live_graph_list)

    return fig10

# Updating suggestions

@app.callback(
    Output('alert_1', 'children'),
    Input('isFatigued', 'value')
)

def update_suggestions_1(n):
    if n==1:
        return "Take a break !"
    else:
        return "Check email ?"


@app.callback(
    Output('alert_2', 'children'),
    Input('isFatigued', 'value')
)

def update_suggestions_2(n):
    if n==1:
        return "Grab a coffee !"
    else:
        return "Check to do list ?"

@app.callback(
    Output('alert_3', 'children'),
    Input('isFatigued', 'value')
)

def update_suggestions_3(n):
    if n==1:
        return "Walk the dog !"
    else:
        return " Work on personal project ?"

# updating confusion matrix plot

@app.callback(
    Output('cm', 'figure'),
    Output('model_name', 'children'),
    Output('accuracy2', 'children'),
    Input('model_type', 'value'),
)

def update_model(value):


    if value == 'dt':
        X_test_scaled, X_train_scaled, model, y_pred, acc, model_name, y_test, y_train, X_test, X_train = pipeline.model(dt, df)
    elif value == 'adaboost':
        X_test_scaled, X_train_scaled, model, y_pred, acc, model_name, y_test, y_train, X_test, X_train = pipeline.model(adaboost, df)
    elif value == 'ann':
        X_test_scaled, X_train_scaled, model, y_pred, acc, model_name, y_test, y_train, X_test, X_train = pipeline.model(ann, df)
    elif value == 'svm':
        X_test_scaled, X_train_scaled, model, y_pred, acc, model_name, y_test, y_train, X_test, X_train = pipeline.model(svm, df)




    return drawConfusionMatrix.cm(y_test, y_pred), model_name, format(acc, ".3g")

# processing dataset

@app.callback(
    Output('fp', 'children'),
    Input('process', 'n_clicks'),
    Input('gamer2', 'value'),
)

def process_data(n_clicks, value):

    datasets = [
        ["gamer1-ppg-2000-01-01.csv", "gamer1-ppg-2000-01-02.csv", "gamer1-annotations.csv"],
        ["gamer2-ppg-2000-01-01.csv", "gamer2-ppg-2000-01-02.csv", "gamer2-annotations.csv"],
        ["gamer3-ppg-2000-01-01.csv", "gamer3-ppg-2000-01-02.csv", "gamer3-annotations.csv"],
        ["gamer4-ppg-2000-01-01.csv", "gamer4-ppg-2000-01-02.csv", "gamer4-annotations.csv"],
        ["gamer5-ppg-2000-01-01.csv", "gamer5-ppg-2000-01-02.csv", "gamer5-annotations.csv"],
    ]

    d1 = datasets[value][0]
    d2 = datasets[value][1]
    annotations = datasets[value][2]

    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'process' in changed_id:
        print("PROCESSING \n")
        print("DATASET= ",d1)
        print("ANNOTATIONS ", annotations)
        path = "../project/data"
        inc = 6000
        isFatigued = [0, 0, 1, 1, 1, 1, 1]
        csv_name = 'pipeline_data_testing.csv'
        pipeline.process(path, d1, d2, annotations, inc, isFatigued, csv_name, rand=0, ds=1)
        
 
    return ""

################################################################################################################

#   STARTING SERVER

################################################################################################################


if __name__ == '__main__':
    app.run_server(debug=True)
